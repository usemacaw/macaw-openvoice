"""WS /v1/realtime -- endpoint WebSocket para streaming STT + TTS full-duplex."""

from __future__ import annotations

import asyncio
import contextlib
import time
import uuid
from typing import TYPE_CHECKING

import grpc.aio
from fastapi import APIRouter, WebSocket, WebSocketDisconnect

from macaw._types import ModelType, STTArchitecture
from macaw.exceptions import ModelNotFoundError
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
from macaw.server.ws_protocol import (
    AudioFrameResult,
    CommandResult,
    ErrorResult,
    dispatch_message,
)
from macaw.session.backpressure import FramesDroppedAction, RateLimitAction

if TYPE_CHECKING:
    from collections.abc import Awaitable, Callable

    from macaw.server.models.events import ServerEvent
    from macaw.session.streaming import StreamingSession

logger = get_logger("server.realtime")

router = APIRouter()

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
    websocket: WebSocket,
    session_id: str,
    session_start: float,
    last_audio_time_ref: list[float],
    session_ref: list[StreamingSession | None],
) -> str:
    """Background task que monitora inatividade e envia pings WebSocket.

    Executa periodicamente (a cada check_interval segundos) e verifica:
    1. Se nenhum audio frame foi recebido dentro do inactivity_timeout.
    2. Envia WebSocket ping a cada heartbeat_interval (best effort).

    Se inatividade for detectada, emite session.closed e fecha o WebSocket.

    Args:
        websocket: Conexao WebSocket ativa.
        session_id: ID da sessao para logging.
        session_start: Timestamp monotonic de inicio da sessao.
        last_audio_time_ref: Lista com um elemento float, usada como referencia
            mutavel para o ultimo timestamp de audio recebido.
        session_ref: Lista com referencia a StreamingSession (pode ser None).

    Returns:
        Razao de fechamento ("inactivity_timeout" ou "client_disconnect").
    """
    from starlette.websockets import WebSocketState as _WSState

    inactivity_timeout, heartbeat_interval, check_interval = _get_ws_timeouts(websocket)
    last_ping_sent = time.monotonic()

    while True:
        await asyncio.sleep(check_interval)

        if websocket.client_state != _WSState.CONNECTED:
            return "client_disconnect"

        now = time.monotonic()

        # Verificar inatividade (sem audio frames recebidos)
        if now - last_audio_time_ref[0] > inactivity_timeout:
            logger.info(
                "inactivity_timeout",
                session_id=session_id,
                timeout_s=inactivity_timeout,
            )
            # Fechar StreamingSession se existir
            session = session_ref[0]
            segments = session.segment_id if session is not None else 0
            if session is not None and not session.is_closed:
                await session.close()

            total_duration_ms = int((now - session_start) * 1000)
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

_TTS_SAMPLE_RATE = 24000
_TTS_GRPC_TIMEOUT = 60.0
_TTS_GRPC_CHANNEL_OPTIONS = [
    ("grpc.max_send_message_length", 30 * 1024 * 1024),
    ("grpc.max_receive_message_length", 30 * 1024 * 1024),
]


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
) -> None:
    """Background task que executa sintese TTS e envia chunks ao cliente.

    Fluxo:
        1. Resolve modelo TTS via registry
        2. Resolve worker TTS via WorkerManager
        3. Mute STT (antes de enviar primeiro byte)
        4. Abre gRPC Synthesize stream
        5. Envia chunks como binary frames ao WebSocket
        6. Emite tts.speaking_start / tts.speaking_end
        7. Unmute STT em finally (garante unmute mesmo em erro/cancel)

    Args:
        websocket: Conexao WebSocket ativa.
        session_id: ID da sessao para logging.
        session: StreamingSession (pode ser None se worker STT indisponivel).
        request_id: ID unico do request TTS.
        text: Texto para sintetizar.
        voice: Voz a usar.
        model_tts: Nome do modelo TTS (None usa default do registry).
        send_event: Callback para enviar eventos ao cliente.
        cancel_event: Event para sinalizar cancelamento externo.
    """
    from starlette.websockets import WebSocketState as _WSState

    tts_start = time.monotonic()
    cancelled = False
    first_chunk_sent = False
    channel = None

    try:
        # 1. Resolve modelo TTS
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
            return

        if model_tts is None:
            # Procurar qualquer modelo TTS registrado
            for m in registry.list_models():
                if m.model_type == ModelType.TTS:
                    model_tts = m.name
                    break
            if model_tts is None:
                await send_event(
                    StreamingErrorEvent(
                        code="model_not_found",
                        message="No TTS model available",
                        recoverable=True,
                    )
                )
                return

        try:
            manifest = registry.get_manifest(model_tts)
            if manifest.model_type != ModelType.TTS:
                raise ModelNotFoundError(model_tts)
        except ModelNotFoundError:
            await send_event(
                StreamingErrorEvent(
                    code="model_not_found",
                    message=f"TTS model '{model_tts}' not found",
                    recoverable=True,
                )
            )
            return

        # 2. Resolve worker TTS
        worker = worker_manager.get_ready_worker(model_tts)
        if worker is None:
            await send_event(
                StreamingErrorEvent(
                    code="worker_unavailable",
                    message=f"No ready TTS worker for model '{model_tts}'",
                    recoverable=True,
                )
            )
            return

        # 3. Build proto request
        proto_request = build_tts_proto_request(
            request_id=request_id,
            text=text,
            voice=voice,
            sample_rate=_TTS_SAMPLE_RATE,
            speed=1.0,
        )

        # 4. Open gRPC stream
        worker_address = f"localhost:{worker.port}"
        channel = grpc.aio.insecure_channel(
            worker_address,
            options=_TTS_GRPC_CHANNEL_OPTIONS,
        )
        stub = TTSWorkerStub(channel)  # type: ignore[no-untyped-call]
        response_stream = stub.Synthesize(proto_request, timeout=_TTS_GRPC_TIMEOUT)

        # 5. Stream chunks to client
        async for chunk in response_stream:
            # Check cancel
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

        if channel is not None:
            with contextlib.suppress(Exception):
                await channel.close()

        logger.debug(
            "tts_task_done",
            session_id=session_id,
            request_id=request_id,
            cancelled=cancelled,
        )


async def _cancel_active_tts(
    tts_task_ref: list[asyncio.Task[None] | None],
    tts_cancel_ref: list[asyncio.Event | None],
) -> None:
    """Cancela TTS ativa se existir. Aguarda finalizacao da task."""
    cancel_event = tts_cancel_ref[0]
    task = tts_task_ref[0]
    if cancel_event is not None and task is not None and not task.done():
        cancel_event.set()
        with contextlib.suppress(asyncio.CancelledError, Exception):
            await task
    tts_task_ref[0] = None
    tts_cancel_ref[0] = None


@router.websocket("/v1/realtime")
async def realtime_endpoint(
    websocket: WebSocket,
    model: str | None = None,
    language: str | None = None,
) -> None:
    """Endpoint WebSocket para streaming STT + TTS full-duplex.

    Query params:
        model: Nome do modelo STT (obrigatorio).
        language: Codigo ISO 639-1 do idioma (opcional, default: auto-detect).

    Protocolo:
        1. Handshake: valida modelo STT via registry.
        2. Emite session.created com session_id unico.
        3. Cria StreamingSession (se worker disponivel).
        4. Main loop: recebe binary (audio frames) e text (JSON commands).
        5. Audio frames -> backpressure check -> session.process_frame().
        6. TTS: tts.speak cria background task, mute STT, stream audio ao cliente.
        7. TTS: tts.cancel interrompe sintese ativa, unmute STT.
        8. Background task monitora inatividade e envia heartbeat pings.
        9. On disconnect: fecha session e emite session.closed.
    """
    session_id = f"sess_{uuid.uuid4().hex[:12]}"

    # --- Validacao pre-accept ---
    if model is None:
        await websocket.accept()
        error_event = StreamingErrorEvent(
            code="invalid_request",
            message="Query parameter 'model' is required",
            recoverable=False,
        )
        await _send_event(websocket, error_event)
        await websocket.close(code=1008, reason="Missing required query parameter: model")
        return

    registry = websocket.app.state.registry
    if registry is None:
        await websocket.accept()
        error_event = StreamingErrorEvent(
            code="service_unavailable",
            message="No models available",
            recoverable=False,
        )
        await _send_event(websocket, error_event)
        await websocket.close(code=1008, reason="No models available")
        return

    try:
        manifest = registry.get_manifest(model)
    except ModelNotFoundError:
        await websocket.accept()
        error_event = StreamingErrorEvent(
            code="model_not_found",
            message=f"Model '{model}' not found in registry",
            recoverable=False,
        )
        await _send_event(websocket, error_event)
        await websocket.close(code=1008, reason=f"Model not found: {model}")
        return

    # Extrair architecture e hot_words capability do manifesto
    model_architecture = manifest.capabilities.architecture or STTArchitecture.ENCODER_DECODER
    model_supports_hot_words = manifest.capabilities.hot_words or False

    # --- Accept conexao ---
    await websocket.accept()

    session_start = time.monotonic()

    logger.info(
        "session_created",
        session_id=session_id,
        model=model,
        language=language,
    )

    config = SessionConfig(language=language)
    created_event = SessionCreatedEvent(
        session_id=session_id,
        model=model,
        config=config,
    )
    await _send_event(websocket, created_event, session_id=session_id)

    # Referencia mutavel para timestamp do ultimo audio recebido.
    # Usa lista de um elemento para permitir mutacao pela background task e main loop.
    last_audio_time: list[float] = [time.monotonic()]

    # --- Criar StreamingSession (se worker disponivel) ---
    async def _on_session_event(event: ServerEvent) -> None:
        """Callback: emite eventos da StreamingSession para o WebSocket."""
        await _send_event(websocket, event, session_id=session_id)

    session: StreamingSession | None = _create_streaming_session(
        websocket=websocket,
        session_id=session_id,
        on_event=_on_session_event,
        language=language,
        architecture=model_architecture,
        engine_supports_hot_words=model_supports_hot_words,
    )

    # Referencia mutavel para a session (usada pelo monitor de inatividade)
    session_ref: list[StreamingSession | None] = [session]

    # Backpressure controller (per-session)
    from macaw.session.backpressure import BackpressureController

    backpressure = BackpressureController()

    # TTS full-duplex: referencia mutavel para task e cancel event
    model_tts: str | None = None
    tts_task_ref: list[asyncio.Task[None] | None] = [None]
    tts_cancel_ref: list[asyncio.Event | None] = [None]

    monitor_task = asyncio.create_task(
        _inactivity_monitor(
            websocket=websocket,
            session_id=session_id,
            session_start=session_start,
            last_audio_time_ref=last_audio_time,
            session_ref=session_ref,
        ),
    )

    # --- Main loop ---
    closed_reason = "client_disconnect"
    try:
        while True:
            message = await websocket.receive()

            if message.get("type") == "websocket.disconnect":
                break

            result = dispatch_message(message)

            if result is None:
                continue

            if isinstance(result, ErrorResult):
                await _send_event(websocket, result.event, session_id=session_id)
                continue

            if isinstance(result, AudioFrameResult):
                last_audio_time[0] = time.monotonic()
                logger.debug(
                    "audio_frame_received",
                    session_id=session_id,
                    size_bytes=len(result.data),
                )

                # Backpressure check
                bp_action = backpressure.record_frame(len(result.data))
                if isinstance(bp_action, RateLimitAction):
                    await _send_event(
                        websocket,
                        SessionRateLimitEvent(
                            delay_ms=bp_action.delay_ms,
                            message="Client sending faster than real-time, please throttle",
                        ),
                        session_id=session_id,
                    )
                elif isinstance(bp_action, FramesDroppedAction):
                    await _send_event(
                        websocket,
                        SessionFramesDroppedEvent(
                            dropped_ms=bp_action.dropped_ms,
                            message=f"Backlog exceeded, {bp_action.dropped_ms}ms of audio dropped",
                        ),
                        session_id=session_id,
                    )
                    continue  # Frame descartado, nao enviar ao session

                # Enviar ao StreamingSession (se disponivel)
                if session is not None and not session.is_closed:
                    await session.process_frame(result.data)
                continue

            if isinstance(result, CommandResult):
                cmd = result.command

                if isinstance(cmd, SessionConfigureCommand):
                    logger.info(
                        "session_configure",
                        session_id=session_id,
                        language=cmd.language,
                        vad_sensitivity=(
                            cmd.vad_sensitivity.value if cmd.vad_sensitivity else None
                        ),
                    )
                    # Aplicar hot words e ITN settings na session
                    if session is not None:
                        if cmd.hot_words is not None:
                            session._hot_words = cmd.hot_words
                        if cmd.enable_itn is not None:
                            session._enable_itn = cmd.enable_itn
                    # Track modelo TTS para full-duplex
                    if cmd.model_tts is not None:
                        model_tts = cmd.model_tts

                elif isinstance(cmd, TTSSpeakCommand):
                    tts_req_id = cmd.request_id or f"tts_{uuid.uuid4().hex[:8]}"
                    logger.info(
                        "tts_speak",
                        session_id=session_id,
                        request_id=tts_req_id,
                        voice=cmd.voice,
                        text_len=len(cmd.text),
                    )
                    # Cancelar TTS anterior se existir (sequential speaks)
                    await _cancel_active_tts(tts_task_ref, tts_cancel_ref)

                    # Criar nova TTS task
                    cancel_ev = asyncio.Event()
                    tts_cancel_ref[0] = cancel_ev
                    tts_task_ref[0] = asyncio.create_task(
                        _tts_speak_task(
                            websocket=websocket,
                            session_id=session_id,
                            session=session,
                            request_id=tts_req_id,
                            text=cmd.text,
                            voice=cmd.voice,
                            model_tts=model_tts,
                            send_event=_on_session_event,
                            cancel_event=cancel_ev,
                        ),
                    )

                elif isinstance(cmd, TTSCancelCommand):
                    logger.info(
                        "tts_cancel",
                        session_id=session_id,
                        request_id=cmd.request_id,
                    )
                    await _cancel_active_tts(tts_task_ref, tts_cancel_ref)

                elif isinstance(cmd, SessionCancelCommand):
                    logger.info("session_cancel", session_id=session_id)
                    closed_reason = "cancelled"
                    segments = session.segment_id if session is not None else 0
                    if session is not None and not session.is_closed:
                        await session.close()
                    closed = SessionClosedEvent(
                        reason="cancelled",
                        total_duration_ms=int(
                            (time.monotonic() - session_start) * 1000,
                        ),
                        segments_transcribed=segments,
                    )
                    await _send_event(websocket, closed, session_id=session_id)
                    break

                elif isinstance(cmd, SessionCloseCommand):
                    logger.info("session_close", session_id=session_id)
                    closed_reason = "client_request"
                    segments = session.segment_id if session is not None else 0
                    if session is not None and not session.is_closed:
                        await session.close()
                    closed = SessionClosedEvent(
                        reason="client_request",
                        total_duration_ms=int(
                            (time.monotonic() - session_start) * 1000,
                        ),
                        segments_transcribed=segments,
                    )
                    await _send_event(websocket, closed, session_id=session_id)
                    break

                elif isinstance(cmd, InputAudioBufferCommitCommand):
                    logger.info(
                        "input_audio_buffer_commit",
                        session_id=session_id,
                    )
                    if session is not None and not session.is_closed:
                        await session.commit()

    except WebSocketDisconnect:
        logger.info(
            "client_disconnected",
            session_id=session_id,
        )
    except Exception:
        logger.exception(
            "session_error",
            session_id=session_id,
        )
        error_event = StreamingErrorEvent(
            code="internal_error",
            message="Internal server error",
            recoverable=False,
        )
        await _send_event(websocket, error_event, session_id=session_id)
    finally:
        # Cancelar TTS ativa antes de fechar session
        await _cancel_active_tts(tts_task_ref, tts_cancel_ref)

        # Fechar StreamingSession se ativa
        if session is not None and not session.is_closed:
            await session.close()

        # Cancelar background task de inatividade
        monitor_task.cancel()
        try:
            monitor_result = await monitor_task
            if monitor_result == "inactivity_timeout":
                closed_reason = "inactivity_timeout"
        except asyncio.CancelledError:
            pass

        # Emitir session.closed apenas se nao foi emitido pelo monitor
        # ou por um comando (session.cancel / session.close)
        if closed_reason not in (
            "inactivity_timeout",
            "cancelled",
            "client_request",
        ):
            segments = session.segment_id if session is not None else 0
            total_duration_ms = int((time.monotonic() - session_start) * 1000)
            closed_event = SessionClosedEvent(
                reason=closed_reason,
                total_duration_ms=total_duration_ms,
                segments_transcribed=segments,
            )
            await _send_event(websocket, closed_event, session_id=session_id)

        logger.info(
            "session_closed",
            session_id=session_id,
            reason=closed_reason,
        )
