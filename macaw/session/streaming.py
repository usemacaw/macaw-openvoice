"""StreamingSession — central orchestrator for streaming STT.

Coordinates the flow: preprocessing -> VAD -> gRPC worker -> post-processing.
Each WebSocket session has a StreamingSession instance that manages the
full streaming lifecycle.

State machine (M6): INIT -> ACTIVE -> SILENCE -> HOLD -> CLOSING -> CLOSED.
SessionStateMachine manages transitions and timeouts; StreamingSession
coordinates the flow based on the current state.

Ring Buffer (M6-05): preprocessed frames are written to the ring buffer for
recovery and LocalAgreement. The read fence protects uncommitted data.
transcript.final advances the fence; ring buffer force commit (>90%) triggers
automatic commit() for the segment.

Rules:
- ITN (post-processing) ONLY on transcript.final, NEVER on partial.
- Hot words are sent only on the FIRST frame of each speech segment.
- Frames in HOLD are not sent to the worker (GPU savings).
- INIT: wait for VAD speech_start before sending frames to the worker.
- CLOSING: do not accept new frames; flush pending.
- CLOSED: reject everything.

Extracted collaborators:
- StreamMetricsRecorder: TTFB, final_delay, force_commit counter, session lifecycle gauges.
- StreamRecoveryHandler: worker crash recovery (re-open stream, resend uncommitted).
"""

from __future__ import annotations

import asyncio
import contextlib
import time
from typing import TYPE_CHECKING

import numpy as np

from macaw._types import SessionState, STTArchitecture, TranscriptSegment
from macaw.exceptions import InvalidTransitionError, WorkerCrashError
from macaw.logging import get_logger
from macaw.server.models.events import (
    SessionHoldEvent,
    StreamingErrorEvent,
    TranscriptFinalEvent,
    TranscriptPartialEvent,
    VADSpeechEndEvent,
    VADSpeechStartEvent,
    WordEvent,
)
from macaw.session.metrics import (
    HAS_METRICS,
    stt_active_sessions,
    stt_confidence_avg,
    stt_final_delay_seconds,
    stt_muted_frames_total,
    stt_segments_force_committed_total,
    stt_session_duration_seconds,
    stt_ttfb_seconds,
    stt_vad_events_total,
    stt_worker_recoveries_total,
)
from macaw.session.mute import MuteController
from macaw.session.state_machine import SessionStateMachine, SessionTimeouts
from macaw.session.wal import SessionWAL
from macaw.vad.detector import VADEventType

if TYPE_CHECKING:
    from collections.abc import Awaitable, Callable

    from macaw.postprocessing.pipeline import PostProcessingPipeline
    from macaw.preprocessing.streaming import StreamingPreprocessor
    from macaw.scheduler.streaming import StreamHandle, StreamingGRPCClient
    from macaw.server.models.events import ServerEvent
    from macaw.session.cross_segment import CrossSegmentContext
    from macaw.session.ring_buffer import RingBuffer
    from macaw.vad.detector import VADDetector

logger = get_logger("session.streaming")


class StreamMetricsRecorder:
    """Records streaming STT metrics (TTFB, final_delay, force_commit).

    Owns the timing state for per-segment metric calculation and the
    synchronous ring-buffer force-commit callback. StreamingSession
    delegates metric recording here at the appropriate lifecycle moments.

    Args:
        session_id: Session identifier for structured logging.
    """

    __slots__ = (
        "_force_commit_pending",
        "_session_id",
        "_speech_end_monotonic",
        "_speech_start_monotonic",
        "_ttfb_recorded_for_segment",
    )

    def __init__(self, session_id: str) -> None:
        self._session_id = session_id
        self._speech_start_monotonic: float | None = None
        self._speech_end_monotonic: float | None = None
        self._ttfb_recorded_for_segment = False
        self._force_commit_pending = False

    def on_speech_start(self) -> None:
        """Record speech_start timestamp and reset TTFB flag."""
        self._speech_start_monotonic = time.monotonic()
        self._ttfb_recorded_for_segment = False

    def on_speech_end(self) -> None:
        """Record speech_end timestamp for final_delay calculation."""
        self._speech_end_monotonic = time.monotonic()

    def reset_segment(self) -> None:
        """Clear timing state after segment boundary (speech_end done)."""
        self._speech_start_monotonic = None
        self._speech_end_monotonic = None

    def record_ttfb(self) -> None:
        """Record TTFB on the first transcript event of a segment."""
        if (
            not HAS_METRICS
            or stt_ttfb_seconds is None
            or self._ttfb_recorded_for_segment
            or self._speech_start_monotonic is None
        ):
            return

        ttfb = time.monotonic() - self._speech_start_monotonic
        stt_ttfb_seconds.observe(ttfb)
        self._ttfb_recorded_for_segment = True

    def record_final_delay(self) -> None:
        """Record final_delay when transcript.final arrives after speech_end."""
        if (
            not HAS_METRICS
            or stt_final_delay_seconds is None
            or self._speech_end_monotonic is None
        ):
            return

        delay = time.monotonic() - self._speech_end_monotonic
        stt_final_delay_seconds.observe(delay)

    def consume_force_commit(self) -> bool:
        """Check and consume the force-commit flag (test-and-clear).

        Returns True if a force commit was pending, and clears the flag.
        Called by process_frame() to decide whether to execute commit().
        """
        if not self._force_commit_pending:
            return False
        self._force_commit_pending = False
        return True

    def on_ring_buffer_force_commit(self, _total_written: int) -> None:
        """Synchronous callback invoked by RingBuffer when >90% full.

        Sets flag for process_frame() (async) to execute the real commit.
        The total_written parameter is ignored — commit advances the fence
        to total_written at execution time.
        """
        self._force_commit_pending = True

        if HAS_METRICS and stt_segments_force_committed_total is not None:
            stt_segments_force_committed_total.inc()

        logger.debug(
            "ring_buffer_force_commit_pending",
            session_id=self._session_id,
        )


class StreamRecoveryHandler:
    """Handles worker crash recovery for a StreamingSession.

    Owns the recovery flag and timeout. Operates on the session's stream
    handles, ring buffer, WAL, and state machine to restore streaming
    after a worker crash.

    Args:
        session: The StreamingSession to recover.
        recovery_timeout_s: Max seconds to wait for new stream open.
    """

    __slots__ = ("_recovering", "_recovery_timeout_s", "_session")

    def __init__(
        self,
        session: StreamingSession,
        recovery_timeout_s: float = 10.0,
    ) -> None:
        self._session = session
        self._recovery_timeout_s = recovery_timeout_s
        self._recovering = False

    @property
    def is_recovering(self) -> bool:
        """True if recovery is in progress."""
        return self._recovering

    async def recover(self) -> bool:
        """Attempt to recover the session after a worker crash.

        Re-opens the gRPC stream, resends uncommitted ring buffer data,
        and restores segment_id from the WAL.

        Returns:
            True if recovery succeeded, False if failed or timed out.
        """
        if self._recovering:
            logger.warning(
                "recovery_already_in_progress",
                session_id=self._session.session_id,
            )
            return False

        self._recovering = True

        try:
            result = await self._do_recover()
            if HAS_METRICS and stt_worker_recoveries_total is not None:
                stt_worker_recoveries_total.labels(
                    result="success" if result else "failure",
                ).inc()
            return result
        finally:
            self._recovering = False

    async def _do_recover(self) -> bool:
        """Internal recovery logic (separated to guarantee flag reset).

        Returns:
            True if recovery succeeded, False if failed.
        """
        s = self._session
        logger.info(
            "recovery_starting",
            session_id=s.session_id,
            last_segment_id=s.wal.last_committed_segment_id,
            last_buffer_offset=s.wal.last_committed_buffer_offset,
        )

        # 1. Clean up previous stream and receiver task
        await s._cleanup_current_stream()

        # 2. Open new gRPC stream with timeout
        try:
            s._stream_handle = await asyncio.wait_for(
                s.grpc_client.open_stream(s.session_id),
                timeout=self._recovery_timeout_s,
            )
        except (asyncio.TimeoutError, WorkerCrashError) as exc:  # noqa: UP041
            logger.error(
                "recovery_open_stream_failed",
                session_id=s.session_id,
                error=str(exc),
            )
            s._stream_handle = None

            # Transition to CLOSED — recovery failed
            with contextlib.suppress(InvalidTransitionError):
                if s.state_machine.state != SessionState.CLOSING:
                    s.state_machine.transition(SessionState.CLOSING)
            with contextlib.suppress(InvalidTransitionError):
                s.state_machine.transition(SessionState.CLOSED)

            return False

        # 3. Resend uncommitted ring buffer data (if any)
        rb = s.ring_buffer
        if rb is not None and rb.uncommitted_bytes > 0:
            uncommitted_data = rb.read_from_offset(rb.read_fence)
            if uncommitted_data:
                try:
                    await s._stream_handle.send_frame(
                        pcm_data=uncommitted_data,
                    )
                    logger.info(
                        "recovery_resent_uncommitted",
                        session_id=s.session_id,
                        bytes_resent=len(uncommitted_data),
                    )
                except WorkerCrashError:
                    logger.error(
                        "recovery_resend_failed",
                        session_id=s.session_id,
                    )
                    s._stream_handle = None
                    return False

        # 4. Restore segment_id from WAL
        s._segment_id = s.wal.last_committed_segment_id + 1

        # 5. Start new receiver task
        s._receiver_task = asyncio.create_task(
            s._receive_worker_events(),
        )

        # 6. Reset hot words flag for next frame
        s._hot_words_sent_for_segment = False

        logger.info(
            "recovery_complete",
            session_id=s.session_id,
            segment_id=s._segment_id,
        )

        return True


class StreamingSession:
    """Streaming STT orchestrator for a WebSocket session.

    Coordinates preprocessing, VAD, gRPC streaming with the worker, and
    post-processing. Emits events to the WebSocket handler via callback.

    The state machine (SessionStateMachine) manages transitions between
    6 states: INIT -> ACTIVE -> SILENCE -> HOLD -> CLOSING -> CLOSED.
    VAD events trigger transitions and timeouts are checked periodically.

    Ring Buffer (optional): stores preprocessed frames for recovery and
    LocalAgreement. Ring buffer force commit (>90% full) triggers automatic
    segment commit(). transcript.final advances the read fence.

    Typical lifecycle:
        1. Create StreamingSession with injected dependencies
        2. Call process_frame() for each incoming audio frame
        3. Events are emitted via on_event callback
        4. Call close() to end the session

    Args:
        session_id: Unique session identifier.
        preprocessor: StreamingPreprocessor to normalize audio.
        vad: VADDetector to detect speech/silence.
        grpc_client: StreamingGRPCClient to open worker streams.
        postprocessor: PostProcessingPipeline for ITN on finals.
        on_event: Async callback to emit events to the WebSocket handler.
        hot_words: List of hot words for keyword boosting.
        enable_itn: If True, apply ITN to transcript.final.
        state_machine: SessionStateMachine to manage states (optional,
            creates default if None).
        ring_buffer: RingBuffer for preprocessed audio storage
            (optional, for backward compatibility with existing tests).
        wal: SessionWAL for recovery checkpoint recording (optional,
            creates default if None — every session always has WAL).
        cross_segment_context: CrossSegmentContext to condition the
            next segment with the previous transcript.final text
            (optional, for backward compatibility).
        architecture: STT backend architecture (CTC, ENCODER_DECODER).
            CTC produces native partials (no LocalAgreement) and does not support
            initial_prompt. Default: ENCODER_DECODER (backward compat).
    """

    def __init__(
        self,
        session_id: str,
        preprocessor: StreamingPreprocessor,
        vad: VADDetector,
        grpc_client: StreamingGRPCClient,
        postprocessor: PostProcessingPipeline | None,
        on_event: Callable[[ServerEvent], Awaitable[None]],
        hot_words: list[str] | None = None,
        enable_itn: bool = True,
        state_machine: SessionStateMachine | None = None,
        ring_buffer: RingBuffer | None = None,
        wal: SessionWAL | None = None,
        recovery_timeout_s: float = 10.0,
        cross_segment_context: CrossSegmentContext | None = None,
        engine_supports_hot_words: bool = False,
        architecture: STTArchitecture = STTArchitecture.ENCODER_DECODER,
    ) -> None:
        self._session_id = session_id
        self._preprocessor = preprocessor
        self._vad = vad
        self._grpc_client = grpc_client
        self._postprocessor = postprocessor
        self._on_event = on_event
        self._hot_words = hot_words
        self._enable_itn = enable_itn
        self._engine_supports_hot_words = engine_supports_hot_words
        self._architecture = architecture

        # State machine (M6)
        self._state_machine = state_machine or SessionStateMachine()
        self._segment_id = 0
        self._last_audio_time = time.monotonic()

        # Ring buffer (M6-05): stores preprocessed frames.
        # on_force_commit callback passed via constructor (not private mutation).
        self._ring_buffer = ring_buffer

        # WAL (M6-06): records checkpoints after transcript.final.
        # Always present — if not provided, create default.
        self._wal = wal or SessionWAL()

        # Cross-segment context (M6-09): stores last N tokens from the
        # previous transcript.final as initial_prompt for the next
        # segment. Improves continuity at segment boundaries.
        self._cross_segment_context = cross_segment_context

        # Metrics recorder (extracted collaborator)
        self._metrics = StreamMetricsRecorder(session_id)

        # Recovery handler (extracted collaborator)
        self._recovery = StreamRecoveryHandler(
            session=self,
            recovery_timeout_s=recovery_timeout_s,
        )

        # Wire ring buffer force-commit callback via constructor parameter
        if ring_buffer is not None:
            ring_buffer._on_force_commit = self._metrics.on_ring_buffer_force_commit

        # gRPC stream handle (open during speech)
        self._stream_handle: StreamHandle | None = None
        self._receiver_task: asyncio.Task[None] | None = None

        # Flag: hot words already sent for the current speech segment?
        self._hot_words_sent_for_segment = False

        # Timestamp of current speech segment start (ms)
        self._speech_start_ms: int | None = None

        # Mute-on-speak (M9): delegates to MuteController
        self._mute_controller = MuteController(session_id=session_id)

        # Session start timestamp for duration metric
        self._session_start_monotonic = time.monotonic()

        # Pre-allocated int16 buffer for float32->int16 conversion in
        # _send_frame_to_worker(). Avoids allocating a new int16 array
        # per frame. Grows on demand if a larger frame arrives.
        self._int16_buffer = np.empty(1024, dtype=np.int16)

        if HAS_METRICS and stt_active_sessions is not None:
            stt_active_sessions.inc()

    @property
    def session_id(self) -> str:
        """Session ID."""
        return self._session_id

    @property
    def is_closed(self) -> bool:
        """True if the session is closed."""
        return self._state_machine.state == SessionState.CLOSED

    @property
    def segment_id(self) -> int:
        """ID do segmento atual."""
        return self._segment_id

    @property
    def session_state(self) -> SessionState:
        """Estado atual da maquina de estados."""
        return self._state_machine.state

    @property
    def is_muted(self) -> bool:
        """True se STT esta silenciado (mute-on-speak ativo)."""
        return self._mute_controller.is_muted

    def mute(self) -> None:
        """Silencia o STT (mute-on-speak). Idempotente."""
        self._mute_controller.mute()

    def unmute(self) -> None:
        """Retoma o STT apos mute-on-speak. Idempotente."""
        self._mute_controller.unmute()

    @property
    def wal(self) -> SessionWAL:
        """WAL da sessao para consulta de checkpoints (usado em recovery)."""
        return self._wal

    @property
    def ring_buffer(self) -> RingBuffer | None:
        """Ring buffer (read-only access for recovery handler)."""
        return self._ring_buffer

    @property
    def grpc_client(self) -> StreamingGRPCClient:
        """gRPC streaming client (read-only access for recovery handler)."""
        return self._grpc_client

    @property
    def state_machine(self) -> SessionStateMachine:
        """State machine (read-only access for recovery handler)."""
        return self._state_machine

    def update_hot_words(self, hot_words: list[str] | None) -> None:
        """Atualiza hot words para a sessao (chamado via session.configure).

        Os novos hot words serao usados a partir do proximo segmento de fala.
        Se um segmento ja esta em andamento, os hot words atuais permanecem
        ate o proximo speech_start (quando _hot_words_sent_for_segment reseta).

        Args:
            hot_words: Nova lista de hot words, ou None para limpar.
        """
        self._hot_words = hot_words

    def update_itn(self, enabled: bool) -> None:
        """Update ITN (Inverse Text Normalization) setting.

        Takes effect on the next transcript.final — partials are never
        post-processed regardless of this setting.

        Args:
            enabled: True to enable ITN, False to disable.
        """
        self._enable_itn = enabled

    def update_session_timeouts(self, timeouts: SessionTimeouts) -> None:
        """Atualiza timeouts da state machine via session.configure.

        Args:
            timeouts: Novos timeouts.
        """
        self._state_machine.update_timeouts(timeouts)

    async def process_frame(self, raw_bytes: bytes) -> None:
        """Processa um frame de audio cru do WebSocket.

        Fluxo:
            1. Verifica se sessao aceita frames (nao CLOSING/CLOSED)
            2. Aplica preprocessing (PCM int16 -> float32 16kHz normalizado)
            3. Passa para VAD
            4. Se SPEECH_START: transita estado, abre gRPC stream, emite evento
            5. Durante speech: envia frames ao worker (exceto em HOLD)
            6. Se SPEECH_END: transita estado, fecha gRPC stream, emite evento

        Args:
            raw_bytes: Bytes PCM 16-bit little-endian mono.
        """
        state = self._state_machine.state
        if state in (SessionState.CLOSED, SessionState.CLOSING):
            return

        # Mute-on-speak: descartar frame sem processar (TTS ativo)
        if self._mute_controller.is_muted:
            if HAS_METRICS and stt_muted_frames_total is not None:
                stt_muted_frames_total.inc()
            return

        self._last_audio_time = time.monotonic()

        # 1. Preprocessing: PCM int16 bytes -> float32 16kHz
        frame = self._preprocessor.process_frame(raw_bytes)
        if len(frame) == 0:
            return

        # 2. VAD
        vad_event = self._vad.process_frame(frame)

        # 3. Atuar conforme evento VAD
        if vad_event is not None:
            if vad_event.type == VADEventType.SPEECH_START:
                await self._handle_speech_start(vad_event.timestamp_ms)
            elif vad_event.type == VADEventType.SPEECH_END:
                await self._handle_speech_end(vad_event.timestamp_ms)

        # 4. Se estamos em fala e temos stream aberto, enviar frame ao worker.
        #    Frames em HOLD nao sao enviados ao worker (economia de GPU).
        #    Frames em INIT nao sao enviados (esperando speech_start).
        current_state = self._state_machine.state
        if (
            self._vad.is_speaking
            and self._stream_handle is not None
            and current_state == SessionState.ACTIVE
        ):
            await self._send_frame_to_worker(frame)

        # 5. Verificar force commit pendente do ring buffer.
        #    O callback on_force_commit do ring buffer e sincrono (chamado de
        #    dentro de write()), entao ele seta a flag. Aqui, no contexto
        #    async, consumimos a flag e fazemos o commit real.
        if self._metrics.consume_force_commit():
            await self.commit()

    async def _drain_stream(self, timeout: float) -> bool:
        """Close the gRPC send stream and await the receiver task.

        Shared pattern used by commit(), _handle_speech_end(), and
        _flush_and_close(). Closes the send direction (is_last=True),
        waits for the receiver task to finish within ``timeout``, cancels
        it if the timeout is exceeded, and clears both handles.

        NOT used by close() — close() performs an abrupt cancel, not a
        graceful drain.

        Args:
            timeout: Max seconds to wait for the receiver task.

        Returns:
            True if the receiver completed within the timeout, False if
            it was cancelled or had already been cancelled.
        """
        # 1. Close gRPC send stream (sends is_last=True for flush)
        if self._stream_handle is not None and not self._stream_handle.is_closed:
            try:
                await self._stream_handle.close()
            except WorkerCrashError:
                logger.warning(
                    "drain_stream_close_worker_crash",
                    session_id=self._session_id,
                )

        # 2. Await receiver task (worker returns transcript.final)
        receiver_ok = True
        if self._receiver_task is not None and not self._receiver_task.done():
            try:
                await asyncio.wait_for(self._receiver_task, timeout=timeout)
            except asyncio.TimeoutError:  # noqa: UP041
                logger.warning(
                    "drain_receiver_task_timeout",
                    session_id=self._session_id,
                    timeout=timeout,
                )
                self._receiver_task.cancel()
                with contextlib.suppress(asyncio.CancelledError):
                    await self._receiver_task
                receiver_ok = False
            except asyncio.CancelledError:
                receiver_ok = False

        # 3. Clear handles
        self._receiver_task = None
        self._stream_handle = None

        return receiver_ok

    async def commit(self) -> None:
        """Force commit do segmento atual (manual commit).

        Fecha o stream gRPC atual, fazendo o worker emitir transcript.final
        para o audio acumulado. Incrementa segment_id e reseta estado para
        que o proximo audio abra novo stream.

        No-op se nao ha stream ativo (silencio) ou sessao fechada.
        """
        if self._state_machine.state == SessionState.CLOSED:
            return

        if self._stream_handle is None:
            return

        await self._drain_stream(timeout=5.0)

        # Incrementar segment_id para proximo segmento
        self._segment_id += 1

        # Resetar flag de hot words para que sejam enviados no proximo stream
        self._hot_words_sent_for_segment = False

        logger.debug(
            "manual_commit",
            session_id=self._session_id,
            segment_id=self._segment_id,
        )

    async def close(self) -> None:
        """Fecha a sessao e libera recursos.

        Transita para CLOSING -> CLOSED via state machine.
        Idempotente: chamadas em sessao ja fechada sao no-op.
        """
        if self._state_machine.state == SessionState.CLOSED:
            return

        # Transitar para CLOSING (se nao ja estiver em CLOSING)
        if self._state_machine.state != SessionState.CLOSING:
            with contextlib.suppress(InvalidTransitionError):
                self._state_machine.transition(SessionState.CLOSING)

        # Transitar para CLOSED
        with contextlib.suppress(InvalidTransitionError):
            self._state_machine.transition(SessionState.CLOSED)

        if HAS_METRICS and stt_active_sessions is not None:
            stt_active_sessions.dec()

        if HAS_METRICS and stt_session_duration_seconds is not None:
            duration = time.monotonic() - self._session_start_monotonic
            stt_session_duration_seconds.observe(duration)

        # Cancelar receiver task se ativa
        if self._receiver_task is not None and not self._receiver_task.done():
            self._receiver_task.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await self._receiver_task
            self._receiver_task = None

        # Fechar gRPC stream se aberto
        if self._stream_handle is not None:
            try:
                await self._stream_handle.cancel()
            except Exception:
                logger.debug(
                    "stream_cancel_error_on_close",
                    session_id=self._session_id,
                )
            self._stream_handle = None

        logger.info(
            "session_closed",
            session_id=self._session_id,
            segment_id=self._segment_id,
        )

    def check_inactivity(self) -> bool:
        """Check whether the session expired by state machine timeout.

        Thin wrapper over check_timeout() that returns a boolean instead
        of the new state. Kept for backward compatibility with callers
        that only need "should I close this session?".

        Returns:
            True if the session transitioned to CLOSED, False otherwise.
        """
        if self._state_machine.state == SessionState.CLOSED:
            return False

        target = self._state_machine.check_timeout()
        if target is None:
            return False

        try:
            self._state_machine.transition(target)
        except InvalidTransitionError:
            return False

        return self.is_closed

    async def check_timeout(self) -> SessionState | None:
        """Verifica timeout e executa transicao se necessario.

        Executa a logica de timeout da state machine e emite eventos
        apropriados (ex: SessionHoldEvent ao transitar para HOLD).

        Returns:
            O novo estado apos a transicao, ou None se nao houve timeout.
        """
        if self._state_machine.state == SessionState.CLOSED:
            return None

        target = self._state_machine.check_timeout()
        if target is None:
            return None

        previous = self._state_machine.state

        try:
            self._state_machine.transition(target)
        except InvalidTransitionError:
            return None

        new_state = self._state_machine.state

        # Emitir eventos conforme transicao
        if new_state == SessionState.HOLD:
            hold_timeout_ms = int(self._state_machine.timeouts.hold_timeout_s * 1000)
            await self._on_event(
                SessionHoldEvent(
                    timestamp_ms=self._state_machine.elapsed_in_state_ms,
                    hold_timeout_ms=hold_timeout_ms,
                ),
            )

        if new_state == SessionState.CLOSING:
            # Iniciar flush de pendentes
            await self._flush_and_close()

        logger.debug(
            "timeout_transition",
            session_id=self._session_id,
            from_state=previous.value,
            to_state=new_state.value,
        )

        return new_state

    async def recover(self) -> bool:
        """Delegate to StreamRecoveryHandler."""
        return await self._recovery.recover()

    async def _cleanup_current_stream(self) -> None:
        """Limpa stream gRPC e receiver task anteriores.

        Defesa contra duplo SPEECH_START: se o VAD emitir SPEECH_START sem
        SPEECH_END anterior (edge case de debounce), o stream/task anteriores
        seriam vazados. Este metodo garante cleanup antes de abrir novo stream.
        """
        if self._receiver_task is not None and not self._receiver_task.done():
            self._receiver_task.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await self._receiver_task
            self._receiver_task = None

        if self._stream_handle is not None and not self._stream_handle.is_closed:
            try:
                await self._stream_handle.cancel()
            except Exception:
                logger.debug(
                    "cleanup_stream_cancel_error",
                    session_id=self._session_id,
                )
            self._stream_handle = None

    async def _handle_speech_start(self, timestamp_ms: int) -> None:
        """Transita estado e abre gRPC stream ao detectar fala."""
        current_state = self._state_machine.state

        # Ignorar em estados onde speech_start nao faz sentido
        if current_state in (SessionState.CLOSING, SessionState.CLOSED):
            return

        # Transitar para ACTIVE
        if current_state in (SessionState.INIT, SessionState.SILENCE, SessionState.HOLD):
            try:
                self._state_machine.transition(SessionState.ACTIVE)
            except InvalidTransitionError:
                logger.warning(
                    "speech_start_invalid_transition",
                    session_id=self._session_id,
                    from_state=current_state.value,
                )
                return

        # Limpar stream anterior se existir (defesa contra duplo SPEECH_START)
        if self._stream_handle is not None or self._receiver_task is not None:
            await self._cleanup_current_stream()

        self._speech_start_ms = timestamp_ms
        self._hot_words_sent_for_segment = False
        self._metrics.on_speech_start()

        if HAS_METRICS and stt_vad_events_total is not None:
            stt_vad_events_total.labels(event_type="speech_start").inc()

        # Abrir stream gRPC com o worker
        try:
            self._stream_handle = await self._grpc_client.open_stream(
                self._session_id,
            )
        except WorkerCrashError:
            await self._emit_error(
                code="worker_crash",
                message="Worker unavailable, cannot open stream",
                recoverable=True,
            )
            return

        # Iniciar task de recepcao de eventos do worker
        self._receiver_task = asyncio.create_task(
            self._receive_worker_events(),
        )

        # Emitir evento VAD
        await self._on_event(
            VADSpeechStartEvent(timestamp_ms=timestamp_ms),
        )

        logger.debug(
            "speech_start",
            session_id=self._session_id,
            timestamp_ms=timestamp_ms,
            segment_id=self._segment_id,
        )

    async def _handle_speech_end(self, timestamp_ms: int) -> None:
        """Transita estado, fecha gRPC stream, emite vad.speech_end.

        Garante que TODOS os transcript.final do worker sejam emitidos ANTES
        do vad.speech_end, respeitando a semantica do protocolo WebSocket
        (PRD secao 9: transcript.final vem antes de vad.speech_end).
        """
        current_state = self._state_machine.state

        # Ignorar em estados onde speech_end nao faz sentido
        if current_state != SessionState.ACTIVE:
            return

        # Transitar para SILENCE
        try:
            self._state_machine.transition(SessionState.SILENCE)
        except InvalidTransitionError:
            logger.warning(
                "speech_end_invalid_transition",
                session_id=self._session_id,
                from_state=current_state.value,
            )

        self._metrics.on_speech_end()

        if HAS_METRICS and stt_vad_events_total is not None:
            stt_vad_events_total.labels(event_type="speech_end").inc()

        # Drain stream: close gRPC send + await receiver task.
        # Guarantees transcript.final is emitted BEFORE vad.speech_end.
        receiver_ok = await self._drain_stream(timeout=5.0)

        # Emitir vad.speech_end SOMENTE apos receiver task completar.
        # Se receiver falhou (timeout/cancel), ainda emitimos speech_end
        # para manter o contrato, mas logamos o problema.
        if not receiver_ok:
            logger.warning(
                "speech_end_after_receiver_failure",
                session_id=self._session_id,
                timestamp_ms=timestamp_ms,
            )

        await self._on_event(
            VADSpeechEndEvent(timestamp_ms=timestamp_ms),
        )

        # Incrementar segment_id para o proximo segmento de fala
        self._segment_id += 1
        self._speech_start_ms = None
        self._metrics.reset_segment()

        logger.debug(
            "speech_end",
            session_id=self._session_id,
            timestamp_ms=timestamp_ms,
            segment_id=self._segment_id,
        )

    async def _send_frame_to_worker(self, frame: np.ndarray) -> None:
        """Converte float32 para PCM int16 bytes e envia ao worker.

        Tambem escreve os bytes PCM no ring buffer (se configurado)
        para recovery e LocalAgreement.
        """
        if self._stream_handle is None or self._stream_handle.is_closed:
            return

        # Converter float32 [-1.0, 1.0] -> int16 bytes
        # In-place multiply+clip avoids 2 intermediate allocations per frame.
        # Safe: frame is not used after this method (VAD already processed it,
        # preprocessor creates a new array each call).
        np.multiply(frame, 32767.0, out=frame)
        np.clip(frame, -32768, 32767, out=frame)
        # Reuse pre-allocated int16 buffer to avoid per-frame allocation.
        # Grow if needed (rare: only on first oversized frame).
        frame_len = len(frame)
        if frame_len > len(self._int16_buffer):
            self._int16_buffer = np.empty(frame_len, dtype=np.int16)
        buf = self._int16_buffer[:frame_len]
        np.copyto(buf, frame, casting="unsafe")
        pcm_bytes = buf.tobytes()

        # Escrever no ring buffer (antes de enviar ao worker, para garantir
        # que os dados estao no buffer mesmo se o worker crashar).
        if self._ring_buffer is not None:
            self._ring_buffer.write(pcm_bytes)

        # Hot words e initial_prompt apenas no primeiro frame do segmento
        hot_words: list[str] | None = None
        initial_prompt: str | None = None
        if not self._hot_words_sent_for_segment:
            if self._hot_words:
                hot_words = self._hot_words
            initial_prompt = self._build_initial_prompt()
            self._hot_words_sent_for_segment = True

        try:
            await self._stream_handle.send_frame(
                pcm_data=pcm_bytes,
                initial_prompt=initial_prompt,
                hot_words=hot_words,
            )
        except WorkerCrashError:
            await self._emit_error(
                code="worker_crash",
                message="Worker crashed during streaming",
                recoverable=True,
            )

    def _build_initial_prompt(self) -> str | None:
        """Constroi initial_prompt combinando hot words e cross-segment context.

        Quando a engine suporta hot words nativamente
        (``_engine_supports_hot_words=True``), hot words NAO sao injetadas no
        initial_prompt — sao enviadas via campo ``hot_words`` do AudioFrame
        para que a engine use keyword boosting nativo. Apenas cross-segment
        context e incluido no prompt.

        Quando a engine NAO suporta hot words nativamente (Whisper), hot words
        sao injetadas via initial_prompt como workaround semantico.

        Formato (sem suporte nativo):
            - Hot words + contexto: "Termos: PIX, TED, Selic. {context}"
            - Apenas hot words: "Termos: PIX, TED, Selic."
            - Apenas contexto: "{context}"
            - Nenhum: None

        Returns:
            String de prompt ou None se nao ha conteudo.
        """
        hot_words_prompt: str | None = None
        if self._hot_words and not self._engine_supports_hot_words:
            hot_words_prompt = f"Termos: {', '.join(self._hot_words)}."

        # Cross-segment context: apenas para engines que suportam initial_prompt
        # (encoder-decoder como Whisper). CTC nao suporta conditioning via prompt.
        context_prompt: str | None = None
        if self._cross_segment_context is not None and self._architecture != STTArchitecture.CTC:
            context_prompt = self._cross_segment_context.get_prompt()

        if hot_words_prompt and context_prompt:
            return f"{hot_words_prompt} {context_prompt}"
        if hot_words_prompt:
            return hot_words_prompt
        if context_prompt:
            return context_prompt
        return None

    async def _receive_worker_events(self) -> None:
        """Background task: consome eventos do worker via gRPC.

        Converte TranscriptSegment em eventos do protocolo WebSocket
        e emite via callback. Post-processing (ITN) e aplicado APENAS
        em transcript.final.
        """
        if self._stream_handle is None:
            return

        try:
            async for segment in self._stream_handle.receive_events():
                if self._state_machine.state == SessionState.CLOSED:
                    break

                self._metrics.record_ttfb()

                if segment.is_final:
                    await self._handle_final_event(segment)
                else:
                    await self._handle_partial_event(segment)
        except WorkerCrashError:
            if not self._recovery.is_recovering:
                resume_segment = self._wal.last_committed_segment_id + 1
                await self._emit_error(
                    code="worker_crash",
                    message=(f"Worker crashed, attempting recovery from segment {resume_segment}"),
                    recoverable=True,
                )
                recovered = await self.recover()
                if not recovered:
                    await self._emit_error(
                        code="worker_crash",
                        message="Recovery failed, session closing",
                        recoverable=False,
                    )
        except asyncio.CancelledError:
            raise
        except Exception as exc:
            logger.error(
                "receiver_unexpected_error",
                session_id=self._session_id,
                error=str(exc),
            )
            await self._emit_error(
                code="internal_error",
                message=f"Unexpected error: {exc}",
                recoverable=False,
            )

    async def _handle_final_event(self, segment: TranscriptSegment) -> None:
        """Handle a transcript.final event from the worker.

        Applies ITN post-processing, emits the event, advances ring buffer
        read fence, records WAL checkpoint, and updates cross-segment context.
        """
        self._metrics.record_final_delay()

        # Apply post-processing (ITN) on finals
        text = segment.text
        if self._enable_itn and self._postprocessor is not None:
            text = self._postprocessor.process(text)

        # Convert word timestamps
        words: list[WordEvent] | None = None
        if segment.words:
            words = [WordEvent(word=w.word, start=w.start, end=w.end) for w in segment.words]

        await self._on_event(
            TranscriptFinalEvent(
                text=text,
                segment_id=self._segment_id,
                start_ms=segment.start_ms or 0,
                end_ms=segment.end_ms or 0,
                language=segment.language,
                confidence=segment.confidence,
                words=words,
            ),
        )

        # Record confidence metric
        if HAS_METRICS and stt_confidence_avg is not None and segment.confidence is not None:
            stt_confidence_avg.observe(segment.confidence)

        # Advance ring buffer read fence
        if self._ring_buffer is not None:
            self._ring_buffer.commit(self._ring_buffer.total_written)

        # WAL checkpoint
        self._wal.record_checkpoint(
            segment_id=self._segment_id,
            buffer_offset=(
                self._ring_buffer.total_written if self._ring_buffer is not None else 0
            ),
            timestamp_ms=int(time.monotonic() * 1000),
        )

        # Cross-segment context (encoder-decoder only, not CTC)
        if self._cross_segment_context is not None and self._architecture != STTArchitecture.CTC:
            self._cross_segment_context.update(text)

    async def _handle_partial_event(self, segment: TranscriptSegment) -> None:
        """Handle a transcript.partial event from the worker.

        Emits the partial without post-processing (ITN is never applied
        to partials — they are unstable and would produce confusing output).
        """
        await self._on_event(
            TranscriptPartialEvent(
                text=segment.text,
                segment_id=self._segment_id,
                timestamp_ms=segment.start_ms or 0,
            ),
        )

    async def _flush_and_close(self) -> None:
        """Flush de pendentes durante CLOSING e transita para CLOSED."""
        await self._drain_stream(timeout=2.0)

    async def _emit_error(
        self,
        code: str,
        message: str,
        recoverable: bool,
    ) -> None:
        """Emite evento de erro via callback."""
        await self._on_event(
            StreamingErrorEvent(
                code=code,
                message=message,
                recoverable=recoverable,
                resume_segment_id=self._segment_id if recoverable else None,
            ),
        )
