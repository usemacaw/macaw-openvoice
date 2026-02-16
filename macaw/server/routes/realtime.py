"""WS /v1/realtime -- endpoint WebSocket para streaming STT + TTS full-duplex."""

from __future__ import annotations

import asyncio
import contextlib
import time
import uuid
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

import grpc.aio
from fastapi import APIRouter, Query, WebSocket, WebSocketDisconnect
from fastapi.responses import JSONResponse

from macaw._types import STTArchitecture
from macaw.exceptions import ModelNotFoundError, WorkerUnavailableError
from macaw.logging import get_logger
from macaw.proto.tts_worker_pb2_grpc import TTSWorkerStub
from macaw.scheduler.tts_converters import build_tts_proto_request
from macaw.scheduler.tts_metrics import (
    HAS_TTS_METRICS,
    tts_active_sessions,
    tts_requests_total,
    tts_synthesis_duration_seconds,
    tts_ttfb_seconds,
)
from macaw.server.constants import (
    TTS_DEFAULT_SAMPLE_RATE,
    TTS_GRPC_TIMEOUT,
)
from macaw.server.grpc_channels import get_or_create_tts_channel
from macaw.server.models.events import (
    InputAudioBufferCommitCommand,
    SessionCancelCommand,
    SessionCloseCommand,
    SessionClosedEvent,
    SessionConfig,
    SessionConfigureCommand,
    SessionCreatedEvent,
    SessionFramesDroppedEvent,
    SessionRateLimitEvent,
    StreamingErrorEvent,
    TTSCancelCommand,
    TTSSpeakCommand,
    TTSSpeakingEndEvent,
    TTSSpeakingStartEvent,
)
from macaw.server.tts_service import find_default_tts_model, resolve_tts_resources
from macaw.server.ws_protocol import (
    AudioFrameResult,
    CommandResult,
    ErrorResult,
    dispatch_message,
)
from macaw.session.backpressure import FramesDroppedAction, RateLimitAction
from macaw.session.state_machine import timeouts_from_configure_command

if TYPE_CHECKING:
    from collections.abc import Awaitable, Callable

    from macaw.server.models.events import ServerEvent
    from macaw.session.streaming import StreamingSession

logger = get_logger("server.realtime")

router = APIRouter(tags=["Realtime"])


@router.get(
    "/v1/realtime",
    status_code=426,
    summary="WebSocket Realtime Streaming (STT + TTS full-duplex)",
    response_description="This endpoint requires a WebSocket connection. "
    "HTTP requests return 426 Upgrade Required.",
    responses={
        426: {
            "description": "Upgrade Required — connect via WebSocket",
            "content": {
                "application/json": {
                    "example": {
                        "error": "This endpoint requires a WebSocket connection",
                        "hint": "Connect using: ws://<host>/v1/realtime?model=<model_name>",
                        "protocol": {
                            "connection": {
                                "url": "ws://<host>/v1/realtime",
                                "query_params": {
                                    "model": "(required) STT model name",
                                    "language": "(optional) ISO 639-1 code, default: auto-detect",
                                },
                            },
                            "client_to_server": {
                                "binary_frames": "Raw PCM audio (16kHz, 16-bit, mono)",
                                "commands": [
                                    "session.configure",
                                    "session.cancel",
                                    "session.close",
                                    "input_audio_buffer.commit",
                                    "tts.speak",
                                    "tts.cancel",
                                ],
                            },
                            "server_to_client": {
                                "binary_frames": "TTS audio (PCM 24kHz, 16-bit, mono)",
                                "events": [
                                    "session.created",
                                    "vad.speech_start",
                                    "vad.speech_end",
                                    "transcript.partial",
                                    "transcript.final",
                                    "session.hold",
                                    "session.rate_limit",
                                    "session.frames_dropped",
                                    "tts.speaking_start",
                                    "tts.speaking_end",
                                    "error",
                                    "session.closed",
                                ],
                            },
                        },
                    },
                },
            },
        },
    },
)
async def realtime_docs(
    model: str = Query(
        ...,
        description="Name of the STT model to use (e.g. `faster-whisper-tiny`). "
        "Must be installed via `macaw pull`.",
    ),
    language: str | None = Query(
        None,
        description="ISO 639-1 language code (e.g. `en`, `pt`, `es`). "
        "Omit for automatic language detection.",
    ),
) -> JSONResponse:
    """WebSocket endpoint for real-time streaming STT + TTS full-duplex.

    **This endpoint requires a WebSocket connection.** HTTP requests return `426 Upgrade Required`.

    ---

    ## Connection

    ```
    ws://<host>/v1/realtime?model=faster-whisper-tiny&language=pt
    ```

    ## Binary Frames

    | Direction | Content |
    |-----------|---------|
    | Client → Server | Raw PCM audio: **16kHz, 16-bit signed integer, mono** |
    | Server → Client | TTS audio: **24kHz, 16-bit signed integer, mono** (only during `tts.speak`) |

    ---

    ## Client → Server Commands (JSON text frames)

    ### `session.configure` — Update session parameters
    ```json
    {
      "type": "session.configure",
      "language": "pt",
      "vad_sensitivity": "high",
      "silence_timeout_ms": 500,
      "hold_timeout_ms": 300000,
      "hot_words": ["Macaw", "OpenVoice"],
      "enable_itn": true,
      "enable_partial_transcripts": true,
      "model_tts": "kokoro-v1"
    }
    ```

    ### `input_audio_buffer.commit` — Force commit current audio segment
    ```json
    {"type": "input_audio_buffer.commit"}
    ```

    ### `tts.speak` — Synthesize speech (full-duplex: mutes STT during playback)
    ```json
    {
      "type": "tts.speak",
      "text": "Hello, how can I help you?",
      "voice": "af_heart",
      "request_id": "tts_abc123",
      "language": "English",
      "ref_audio": "<base64>",
      "ref_text": "reference transcript",
      "instruction": "Speak in a warm, friendly tone"
    }
    ```
    *Note: `language`, `ref_audio`, `ref_text`, `instruction` are for LLM-based TTS engines (e.g. Qwen3-TTS).*

    ### `tts.cancel` — Cancel active TTS synthesis
    ```json
    {"type": "tts.cancel"}
    ```

    ### `session.cancel` — Cancel and close the session
    ```json
    {"type": "session.cancel"}
    ```

    ### `session.close` — Gracefully close the session
    ```json
    {"type": "session.close"}
    ```

    ---

    ## Server → Client Events (JSON text frames)

    | Event | Description |
    |-------|-------------|
    | `session.created` | Session established — contains `session_id`, `model`, and `config` |
    | `vad.speech_start` | Voice Activity Detection: speech started |
    | `vad.speech_end` | Voice Activity Detection: speech ended |
    | `transcript.partial` | Intermediate hypothesis (may change) |
    | `transcript.final` | Confirmed segment (immutable, with word timestamps) |
    | `session.hold` | Session entered HOLD state (no speech for a while) |
    | `session.rate_limit` | Client sending audio faster than real-time |
    | `session.frames_dropped` | Audio frames dropped due to backlog |
    | `tts.speaking_start` | TTS audio streaming began (STT is muted) |
    | `tts.speaking_end` | TTS audio streaming ended (STT is unmuted) |
    | `error` | Error with `code`, `message`, and `recoverable` flag |
    | `session.closed` | Session ended — contains `reason` and `total_duration_ms` |

    ---

    ## Full-Duplex Behavior

    - When `tts.speak` is active, **STT is automatically muted** (audio frames are discarded)
    - When TTS finishes or is cancelled, **STT is automatically unmuted**
    - A new `tts.speak` cancels any previous in-progress synthesis
    - TTS audio arrives as **binary frames** (server → client)
    - STT audio is sent as **binary frames** (client → server)
    """
    return JSONResponse(
        status_code=426,
        content={
            "error": "This endpoint requires a WebSocket connection",
            "hint": f"Connect using: ws://<host>/v1/realtime?model={model}",
        },
        headers={"Upgrade": "websocket"},
    )


# Defaults (overrideable via app.state para testes com timeouts curtos)
_DEFAULT_HEARTBEAT_INTERVAL_S = 10.0
_DEFAULT_INACTIVITY_TIMEOUT_S = 60.0
_DEFAULT_CHECK_INTERVAL_S = 5.0


def _get_ws_timeouts(websocket: WebSocket) -> tuple[float, float, float]:
    """Retorna (inactivity_timeout, heartbeat_interval, check_interval).

    Valores sao lidos de ``app.state`` se presentes, senao usa defaults.
    Isso permite que testes sobrescrevam com timeouts curtos.
    """
    state = websocket.app.state
    inactivity = getattr(state, "ws_inactivity_timeout_s", _DEFAULT_INACTIVITY_TIMEOUT_S)
    heartbeat = getattr(state, "ws_heartbeat_interval_s", _DEFAULT_HEARTBEAT_INTERVAL_S)
    check = getattr(state, "ws_check_interval_s", _DEFAULT_CHECK_INTERVAL_S)
    return float(inactivity), float(heartbeat), float(check)


async def _send_event(
    websocket: WebSocket,
    event: ServerEvent,
    session_id: str | None = None,
) -> None:
    """Envia evento JSON para o cliente via WebSocket.

    Verifica se a conexao ainda esta ativa antes de enviar.

    Args:
        websocket: Conexao WebSocket.
        event: Evento server->client a enviar.
        session_id: ID da sessao para log correlation (opcional).
    """
    from starlette.websockets import WebSocketState as _WSState

    if websocket.client_state == _WSState.CONNECTED:
        await websocket.send_json(event.model_dump(mode="json"))
    else:
        logger.debug(
            "send_event_skipped_not_connected",
            session_id=session_id,
            event_type=event.type,
        )


async def _inactivity_monitor(
    ctx: SessionContext,
) -> str:
    """Background task que monitora inatividade e envia pings WebSocket.

    Executa periodicamente (a cada check_interval segundos) e verifica:
    1. Se nenhum audio frame foi recebido dentro do inactivity_timeout.
    2. Envia WebSocket ping a cada heartbeat_interval (best effort).

    Se inatividade for detectada, emite session.closed e fecha o WebSocket.

    Args:
        ctx: Session context with mutable per-session state.

    Returns:
        Razao de fechamento ("inactivity_timeout" ou "client_disconnect").
    """
    from starlette.websockets import WebSocketState as _WSState

    websocket = ctx.websocket
    session_id = ctx.session_id

    inactivity_timeout, heartbeat_interval, check_interval = _get_ws_timeouts(websocket)
    last_ping_sent = time.monotonic()

    while True:
        await asyncio.sleep(check_interval)

        if websocket.client_state != _WSState.CONNECTED:
            return "client_disconnect"

        now = time.monotonic()

        # Verificar inatividade (sem audio frames recebidos)
        if now - ctx.last_audio_time > inactivity_timeout:
            logger.info(
                "inactivity_timeout",
                session_id=session_id,
                timeout_s=inactivity_timeout,
            )
            # Fechar StreamingSession se existir
            session = ctx.session
            segments = session.segment_id if session is not None else 0
            if session is not None and not session.is_closed:
                await session.close()

            total_duration_ms = int((now - ctx.session_start) * 1000)
            closed_event = SessionClosedEvent(
                reason="inactivity_timeout",
                total_duration_ms=total_duration_ms,
                segments_transcribed=segments,
            )
            await _send_event(websocket, closed_event, session_id=session_id)

            with contextlib.suppress(WebSocketDisconnect, RuntimeError, OSError):
                await websocket.close(code=1000, reason="inactivity_timeout")
            return "inactivity_timeout"

        # Enviar ping periodicamente (best effort)
        if now - last_ping_sent >= heartbeat_interval:
            try:
                if websocket.client_state == _WSState.CONNECTED:
                    await websocket.send({"type": "websocket.ping", "bytes": b""})
                    last_ping_sent = now
            except Exception:
                logger.debug(
                    "heartbeat_ping_failed",
                    session_id=session_id,
                )


def _create_streaming_session(
    websocket: WebSocket,
    session_id: str,
    on_event: Callable[[ServerEvent], Awaitable[None]],
    language: str | None = None,
    architecture: STTArchitecture = STTArchitecture.ENCODER_DECODER,
    engine_supports_hot_words: bool = False,
) -> StreamingSession | None:
    """Cria StreamingSession se streaming_grpc_client esta disponivel.

    Instancia per-session: StreamingPreprocessor, VADDetector
    (EnergyPreFilter + SileroVADClassifier), BackpressureController,
    e StreamingSession.

    Returns None se streaming_grpc_client nao esta configurado
    (ex: testes sem infra de worker).
    """
    from macaw.session.streaming import StreamingSession as _StreamingSession

    state = websocket.app.state
    grpc_client = getattr(state, "streaming_grpc_client", None)
    if grpc_client is None:
        return None

    # Obter stages do preprocessing pipeline (batch) para reusar no streaming
    preprocessing_pipeline = getattr(state, "preprocessing_pipeline", None)
    stages = preprocessing_pipeline.stages if preprocessing_pipeline is not None else []

    # Obter postprocessor
    postprocessor = getattr(state, "postprocessing_pipeline", None)

    # Criar preprocessor de streaming
    from macaw.preprocessing.streaming import StreamingPreprocessor

    preprocessor = StreamingPreprocessor(stages=stages)

    # Criar VAD (energy pre-filter + silero classifier + detector)
    from macaw.vad.detector import VADDetector
    from macaw.vad.energy import EnergyPreFilter
    from macaw.vad.silero import SileroVADClassifier

    energy_pre_filter = EnergyPreFilter()
    silero_classifier = SileroVADClassifier()
    vad = VADDetector(
        energy_pre_filter=energy_pre_filter,
        silero_classifier=silero_classifier,
    )

    return _StreamingSession(
        session_id=session_id,
        preprocessor=preprocessor,
        vad=vad,
        grpc_client=grpc_client,
        postprocessor=postprocessor,
        on_event=on_event,
        architecture=architecture,
        engine_supports_hot_words=engine_supports_hot_words,
    )


# ---------------------------------------------------------------------------
# TTS Full-Duplex Support
# ---------------------------------------------------------------------------


async def _prepare_tts_request(
    *,
    websocket: WebSocket,
    session_id: str,
    request_id: str,
    text: str,
    voice: str,
    model_tts: str | None,
    send_event: Callable[[ServerEvent], Awaitable[None]],
    language: str | None = None,
    ref_audio: str | None = None,
    ref_text: str | None = None,
    instruction: str | None = None,
) -> tuple[str, Any] | None:
    """Resolve TTS model/worker and build gRPC proto request.

    Returns (worker_address, proto_request) on success, or None after
    sending an error event to the client.
    """
    import base64 as _b64

    state = websocket.app.state
    registry = getattr(state, "registry", None)
    worker_manager = getattr(state, "worker_manager", None)

    if registry is None or worker_manager is None:
        await send_event(
            StreamingErrorEvent(
                code="service_unavailable",
                message="TTS service not available",
                recoverable=True,
            )
        )
        return None

    if model_tts is None:
        model_tts = find_default_tts_model(registry)
        if model_tts is None:
            await send_event(
                StreamingErrorEvent(
                    code="model_not_found",
                    message="No TTS model available",
                    recoverable=True,
                )
            )
            return None

    try:
        _manifest, _worker, worker_address = resolve_tts_resources(
            registry, worker_manager, model_tts
        )
    except ModelNotFoundError:
        await send_event(
            StreamingErrorEvent(
                code="model_not_found",
                message=f"TTS model '{model_tts}' not found",
                recoverable=True,
            )
        )
        return None
    except WorkerUnavailableError:
        await send_event(
            StreamingErrorEvent(
                code="worker_unavailable",
                message=f"No ready TTS worker for model '{model_tts}'",
                recoverable=True,
            )
        )
        return None

    ref_audio_bytes: bytes | None = None
    if ref_audio:
        try:
            ref_audio_bytes = _b64.b64decode(ref_audio)
        except Exception:
            logger.warning(
                "tts_invalid_ref_audio_base64",
                session_id=session_id,
                request_id=request_id,
            )
            await send_event(
                StreamingErrorEvent(
                    code="invalid_request",
                    message="Invalid base64 in 'ref_audio'",
                    recoverable=True,
                )
            )
            return None

    proto_request = build_tts_proto_request(
        request_id=request_id,
        text=text,
        voice=voice,
        sample_rate=TTS_DEFAULT_SAMPLE_RATE,
        speed=1.0,
        language=language,
        ref_audio=ref_audio_bytes,
        ref_text=ref_text,
        instruction=instruction,
    )
    return worker_address, proto_request


async def _tts_speak_task(
    *,
    websocket: WebSocket,
    session_id: str,
    session: StreamingSession | None,
    request_id: str,
    text: str,
    voice: str,
    model_tts: str | None,
    send_event: Callable[[ServerEvent], Awaitable[None]],
    cancel_event: asyncio.Event,
    language: str | None = None,
    ref_audio: str | None = None,
    ref_text: str | None = None,
    instruction: str | None = None,
) -> None:
    """Background task that runs TTS synthesis and streams audio to the client.

    Flow:
        1. Resolve model + build proto request (_prepare_tts_request)
        2. Open gRPC Synthesize stream (pooled channel from app.state)
        3. Stream chunks as binary frames to WebSocket
        4. Mute STT on first chunk, unmute in finally
    """
    from starlette.websockets import WebSocketState as _WSState

    tts_start = time.monotonic()
    cancelled = False
    first_chunk_sent = False

    try:
        # 1. Resolve model/worker and build proto request
        result = await _prepare_tts_request(
            websocket=websocket,
            session_id=session_id,
            request_id=request_id,
            text=text,
            voice=voice,
            model_tts=model_tts,
            send_event=send_event,
            language=language,
            ref_audio=ref_audio,
            ref_text=ref_text,
            instruction=instruction,
        )
        if result is None:
            return
        worker_address, proto_request = result

        # 2. Open gRPC stream (pooled channel — shared with REST)
        tts_channels: dict[str, grpc.aio.Channel] = getattr(
            websocket.app.state, "tts_channels", {}
        )
        channel = get_or_create_tts_channel(tts_channels, worker_address)
        stub = TTSWorkerStub(channel)  # type: ignore[no-untyped-call]
        response_stream = stub.Synthesize(proto_request, timeout=TTS_GRPC_TIMEOUT)

        # 3. Stream chunks to client
        async for chunk in response_stream:
            if cancel_event.is_set():
                cancelled = True
                break

            if not chunk.audio_data:
                if chunk.is_last:
                    break
                continue

            # On first chunk: mute STT, emit speaking_start, record TTFB
            if not first_chunk_sent:
                if session is not None:
                    session.mute()

                ttfb = time.monotonic() - tts_start
                if HAS_TTS_METRICS and tts_ttfb_seconds is not None:
                    tts_ttfb_seconds.observe(ttfb)
                if HAS_TTS_METRICS and tts_active_sessions is not None:
                    tts_active_sessions.inc()

                await send_event(
                    TTSSpeakingStartEvent(
                        request_id=request_id,
                        timestamp_ms=int(ttfb * 1000),
                    )
                )
                first_chunk_sent = True

            # Send audio as binary frame
            if websocket.client_state == _WSState.CONNECTED:
                await websocket.send_bytes(chunk.audio_data)

            if chunk.is_last:
                break

    except asyncio.CancelledError:
        cancelled = True
    except grpc.aio.AioRpcError as exc:
        logger.error(
            "tts_grpc_error",
            session_id=session_id,
            request_id=request_id,
            code=str(exc.code()),
            details=exc.details(),
        )
        await send_event(
            StreamingErrorEvent(
                code="tts_worker_error",
                message=f"TTS worker error: {exc.details()}",
                recoverable=True,
            )
        )
    except Exception:
        logger.exception(
            "tts_speak_error",
            session_id=session_id,
            request_id=request_id,
        )
        await send_event(
            StreamingErrorEvent(
                code="tts_error",
                message="TTS synthesis failed",
                recoverable=True,
            )
        )
    finally:
        # Always unmute STT and emit speaking_end
        if session is not None:
            session.unmute()

        if first_chunk_sent:
            duration_s = time.monotonic() - tts_start
            duration_ms = int(duration_s * 1000)
            await send_event(
                TTSSpeakingEndEvent(
                    request_id=request_id,
                    timestamp_ms=duration_ms,
                    duration_ms=duration_ms,
                    cancelled=cancelled,
                )
            )

            # TTS metrics: synthesis duration and active gauge
            if HAS_TTS_METRICS and tts_synthesis_duration_seconds is not None:
                tts_synthesis_duration_seconds.observe(duration_s)
            if HAS_TTS_METRICS and tts_active_sessions is not None:
                tts_active_sessions.dec()

        # TTS metrics: requests counter
        if HAS_TTS_METRICS and tts_requests_total is not None:
            if cancelled:
                tts_requests_total.labels(status="cancelled").inc()
            elif first_chunk_sent:
                tts_requests_total.labels(status="ok").inc()
            else:
                tts_requests_total.labels(status="error").inc()

        logger.debug(
            "tts_task_done",
            session_id=session_id,
            request_id=request_id,
            cancelled=cancelled,
        )


async def _cancel_active_tts(ctx: SessionContext) -> None:
    """Cancel active TTS task if any. Awaits task completion."""
    if ctx.tts_cancel_event is not None and ctx.tts_task is not None and not ctx.tts_task.done():
        ctx.tts_cancel_event.set()
        with contextlib.suppress(asyncio.CancelledError, Exception):
            await ctx.tts_task
    ctx.tts_task = None
    ctx.tts_cancel_event = None


# ---------------------------------------------------------------------------
# SessionContext — replaces mutable list refs (M-26)
# ---------------------------------------------------------------------------

MAX_WS_FRAME_SIZE = 1_048_576  # 1 MB (M-09)


@dataclass
class SessionContext:
    """Mutable per-session state for the realtime endpoint."""

    session_id: str
    session_start: float
    websocket: WebSocket
    session: StreamingSession | None = None
    last_audio_time: float = 0.0
    tts_task: asyncio.Task[None] | None = None
    tts_cancel_event: asyncio.Event | None = None
    model_tts: str | None = None
    backpressure: Any = field(default=None)
    closed_reason: str = "client_disconnect"

    async def send_event(self, event: ServerEvent) -> None:
        """Send an event to the client WebSocket."""
        await _send_event(self.websocket, event, session_id=self.session_id)


# ---------------------------------------------------------------------------
# Command Handlers (H-05)
# ---------------------------------------------------------------------------


async def _handle_configure_command(
    ctx: SessionContext,
    cmd: SessionConfigureCommand,
) -> bool:
    """Handle session.configure. Returns False (never breaks loop)."""
    logger.info(
        "session_configure",
        session_id=ctx.session_id,
        language=cmd.language,
        vad_sensitivity=(cmd.vad_sensitivity.value if cmd.vad_sensitivity else None),
    )
    if ctx.session is not None:
        if cmd.hot_words is not None:
            ctx.session.update_hot_words(cmd.hot_words)
        if cmd.enable_itn is not None:
            ctx.session.update_itn(cmd.enable_itn)
        if cmd.silence_timeout_ms is not None or cmd.hold_timeout_ms is not None:
            new_timeouts = timeouts_from_configure_command(
                ctx.session.current_timeouts,
                silence_timeout_ms=cmd.silence_timeout_ms,
                hold_timeout_ms=cmd.hold_timeout_ms,
            )
            ctx.session.update_session_timeouts(new_timeouts)
    if cmd.model_tts is not None:
        ctx.model_tts = cmd.model_tts
    return False


async def _handle_tts_speak_command(
    ctx: SessionContext,
    cmd: TTSSpeakCommand,
) -> bool:
    """Handle tts.speak. Returns False (never breaks loop)."""
    tts_req_id = cmd.request_id or f"tts_{uuid.uuid4().hex[:8]}"
    logger.info(
        "tts_speak",
        session_id=ctx.session_id,
        request_id=tts_req_id,
        voice=cmd.voice,
        text_len=len(cmd.text),
    )
    await _cancel_active_tts(ctx)

    cancel_ev = asyncio.Event()
    ctx.tts_cancel_event = cancel_ev
    ctx.tts_task = asyncio.create_task(
        _tts_speak_task(
            websocket=ctx.websocket,
            session_id=ctx.session_id,
            session=ctx.session,
            request_id=tts_req_id,
            text=cmd.text,
            voice=cmd.voice,
            model_tts=ctx.model_tts,
            send_event=ctx.send_event,
            cancel_event=cancel_ev,
            language=cmd.language,
            ref_audio=cmd.ref_audio,
            ref_text=cmd.ref_text,
            instruction=cmd.instruction,
        ),
    )
    return False


async def _handle_tts_cancel_command(
    ctx: SessionContext,
    cmd: TTSCancelCommand,
) -> bool:
    """Handle tts.cancel. Returns False (never breaks loop)."""
    logger.info(
        "tts_cancel",
        session_id=ctx.session_id,
        request_id=cmd.request_id,
    )
    await _cancel_active_tts(ctx)
    return False


async def _close_session(ctx: SessionContext, reason: str) -> bool:
    """Close session, emit SessionClosedEvent, return True (breaks loop).

    Shared logic for session.cancel and session.close commands.
    """
    segments = ctx.session.segment_id if ctx.session is not None else 0
    if ctx.session is not None and not ctx.session.is_closed:
        await ctx.session.close()
    closed = SessionClosedEvent(
        reason=reason,
        total_duration_ms=int((time.monotonic() - ctx.session_start) * 1000),
        segments_transcribed=segments,
    )
    await ctx.send_event(closed)
    return True


async def _handle_session_cancel_command(
    ctx: SessionContext,
    cmd: SessionCancelCommand,
) -> bool:
    """Handle session.cancel. Returns True (breaks loop)."""
    logger.info("session_cancel", session_id=ctx.session_id)
    ctx.closed_reason = "cancelled"
    return await _close_session(ctx, reason="cancelled")


async def _handle_session_close_command(
    ctx: SessionContext,
    cmd: SessionCloseCommand,
) -> bool:
    """Handle session.close. Returns True (breaks loop)."""
    logger.info("session_close", session_id=ctx.session_id)
    ctx.closed_reason = "client_request"
    return await _close_session(ctx, reason="client_request")


async def _handle_commit_command(
    ctx: SessionContext,
    cmd: InputAudioBufferCommitCommand,
) -> bool:
    """Handle input_audio_buffer.commit. Returns False (never breaks loop)."""
    logger.info("input_audio_buffer_commit", session_id=ctx.session_id)
    if ctx.session is not None and not ctx.session.is_closed:
        await ctx.session.commit()
    return False


# Dispatch table: command type -> handler function
_COMMAND_HANDLERS: dict[
    type[Any],
    Callable[[SessionContext, Any], Awaitable[bool]],
] = {
    SessionConfigureCommand: _handle_configure_command,
    TTSSpeakCommand: _handle_tts_speak_command,
    TTSCancelCommand: _handle_tts_cancel_command,
    SessionCancelCommand: _handle_session_cancel_command,
    SessionCloseCommand: _handle_session_close_command,
    InputAudioBufferCommitCommand: _handle_commit_command,
}


# ---------------------------------------------------------------------------
# Main Endpoint
# ---------------------------------------------------------------------------


@router.websocket("/v1/realtime")
async def realtime_endpoint(
    websocket: WebSocket,
    model: str | None = None,
    language: str | None = None,
) -> None:
    """WebSocket endpoint for streaming STT + TTS full-duplex.

    Query params:
        model: STT model name (required).
        language: ISO 639-1 code (optional, default: auto-detect).
    """
    session_id = f"sess_{uuid.uuid4().hex[:12]}"

    # --- Pre-accept validation ---
    if model is None:
        await websocket.accept()
        await _send_event(
            websocket,
            StreamingErrorEvent(
                code="invalid_request",
                message="Query parameter 'model' is required",
                recoverable=False,
            ),
        )
        await websocket.close(code=1008, reason="Missing required query parameter: model")
        return

    registry = websocket.app.state.registry
    if registry is None:
        await websocket.accept()
        await _send_event(
            websocket,
            StreamingErrorEvent(
                code="service_unavailable", message="No models available", recoverable=False
            ),
        )
        await websocket.close(code=1008, reason="No models available")
        return

    try:
        manifest = registry.get_manifest(model)
    except ModelNotFoundError:
        await websocket.accept()
        await _send_event(
            websocket,
            StreamingErrorEvent(
                code="model_not_found",
                message=f"Model '{model}' not found in registry",
                recoverable=False,
            ),
        )
        await websocket.close(code=1008, reason=f"Model not found: {model}")
        return

    model_architecture = manifest.capabilities.architecture or STTArchitecture.ENCODER_DECODER
    model_supports_hot_words = manifest.capabilities.hot_words or False

    # --- Accept connection ---
    await websocket.accept()
    session_start = time.monotonic()

    logger.info("session_created", session_id=session_id, model=model, language=language)

    config = SessionConfig(language=language)
    await _send_event(
        websocket,
        SessionCreatedEvent(session_id=session_id, model=model, config=config),
        session_id=session_id,
    )

    # --- Build session context ---
    async def _on_session_event(event: ServerEvent) -> None:
        await _send_event(websocket, event, session_id=session_id)

    session: StreamingSession | None = _create_streaming_session(
        websocket=websocket,
        session_id=session_id,
        on_event=_on_session_event,
        language=language,
        architecture=model_architecture,
        engine_supports_hot_words=model_supports_hot_words,
    )

    from macaw.session.backpressure import BackpressureController

    ctx = SessionContext(
        session_id=session_id,
        session_start=session_start,
        websocket=websocket,
        session=session,
        last_audio_time=time.monotonic(),
        backpressure=BackpressureController(),
    )

    monitor_task = asyncio.create_task(
        _inactivity_monitor(ctx),
    )

    # --- Main loop ---
    try:
        while True:
            message = await websocket.receive()

            if message.get("type") == "websocket.disconnect":
                break

            result = dispatch_message(message)
            if result is None:
                continue

            if isinstance(result, ErrorResult):
                await ctx.send_event(result.event)
                continue

            if isinstance(result, AudioFrameResult):
                # M-09: frame size limit
                if len(result.data) > MAX_WS_FRAME_SIZE:
                    await ctx.send_event(
                        StreamingErrorEvent(
                            code="frame_too_large",
                            message=f"Frame too large: {len(result.data)} bytes (max {MAX_WS_FRAME_SIZE})",
                            recoverable=True,
                        )
                    )
                    continue

                ctx.last_audio_time = time.monotonic()
                logger.debug(
                    "audio_frame_received", session_id=session_id, size_bytes=len(result.data)
                )

                bp_action = ctx.backpressure.record_frame(len(result.data))
                if isinstance(bp_action, RateLimitAction):
                    await ctx.send_event(
                        SessionRateLimitEvent(
                            delay_ms=bp_action.delay_ms,
                            message="Client sending faster than real-time, please throttle",
                        )
                    )
                elif isinstance(bp_action, FramesDroppedAction):
                    await ctx.send_event(
                        SessionFramesDroppedEvent(
                            dropped_ms=bp_action.dropped_ms,
                            message=f"Backlog exceeded, {bp_action.dropped_ms}ms of audio dropped",
                        )
                    )
                    continue

                if session is not None and not session.is_closed:
                    await session.process_frame(result.data)
                continue

            if isinstance(result, CommandResult):
                handler = _COMMAND_HANDLERS.get(type(result.command))
                if handler is not None:
                    should_break = await handler(ctx, result.command)
                    if should_break:
                        break

    except WebSocketDisconnect:
        logger.info("client_disconnected", session_id=session_id)
    except Exception:
        logger.exception("session_error", session_id=session_id)
        await ctx.send_event(
            StreamingErrorEvent(
                code="internal_error", message="Internal server error", recoverable=False
            )
        )
    finally:
        await _cancel_active_tts(ctx)

        if session is not None and not session.is_closed:
            await session.close()

        monitor_task.cancel()
        try:
            monitor_result = await monitor_task
            if monitor_result == "inactivity_timeout":
                ctx.closed_reason = "inactivity_timeout"
        except asyncio.CancelledError:
            pass

        if ctx.closed_reason not in ("inactivity_timeout", "cancelled", "client_request"):
            segments = session.segment_id if session is not None else 0
            total_duration_ms = int((time.monotonic() - session_start) * 1000)
            await ctx.send_event(
                SessionClosedEvent(
                    reason=ctx.closed_reason,
                    total_duration_ms=total_duration_ms,
                    segments_transcribed=segments,
                )
            )

        logger.info("session_closed", session_id=session_id, reason=ctx.closed_reason)
