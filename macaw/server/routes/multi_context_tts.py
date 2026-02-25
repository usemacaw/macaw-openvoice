"""WS /v1/text-to-speech/{voice_id}/multi-stream-input — Multi-context TTS endpoint.

Supports multiple independent TTS contexts over a single WebSocket,
matching the ElevenLabs multi-stream-input API.
See ADR-008 for design rationale.
"""

from __future__ import annotations

import asyncio
import base64
import json
import time
import uuid
from typing import TYPE_CHECKING

import pydantic
from fastapi import APIRouter, WebSocket, WebSocketDisconnect
from starlette.websockets import WebSocketState

from macaw._audio_constants import TTS_DEFAULT_SAMPLE_RATE
from macaw.logging import get_logger
from macaw.proto.tts_worker_pb2_grpc import TTSWorkerStub
from macaw.scheduler.tts_converters import build_tts_proto_request
from macaw.server.constants import TTS_GRPC_TIMEOUT
from macaw.server.grpc_channels import get_or_create_tts_channel
from macaw.server.models.multi_context_events import (
    MULTI_CONTEXT_MESSAGE_TYPES,
    AudioEvent,
    CloseContext,
    CloseSocket,
    ConnectionInitializedEvent,
    ContextClosedEvent,
    ContextFlushedEvent,
    FlushContext,
    InitializeConnection,
    IsFinalEvent,
    KeepAlive,
    MultiContextErrorEvent,
    SendText,
)
from macaw.server.tts_service import find_default_tts_model, resolve_tts_resources
from macaw.server.ws_context_manager import ContextState, WSContextManager

if TYPE_CHECKING:
    import grpc.aio

    from macaw.registry.registry import ModelRegistry
    from macaw.server.models.multi_context_events import MultiContextClientMessage
    from macaw.workers.manager import WorkerManager

logger = get_logger("server.multi_context_tts")

router = APIRouter(tags=["Multi-Context TTS"])

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

_INACTIVITY_CHECK_INTERVAL_S = 5.0


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


async def _send_event(ws: WebSocket, event: pydantic.BaseModel) -> None:
    """Send JSON event if WebSocket is still connected."""
    if ws.client_state == WebSocketState.CONNECTED:
        await ws.send_json(event.model_dump(mode="json"))


async def _send_error(
    ws: WebSocket,
    message: str,
    *,
    context_id: str | None = None,
    code: str = "internal_error",
    recoverable: bool = True,
) -> None:
    """Send an error event."""
    await _send_event(
        ws,
        MultiContextErrorEvent(
            message=message,
            context_id=context_id,
            code=code,
            recoverable=recoverable,
        ),
    )


def _parse_message(raw_text: str) -> MultiContextClientMessage | MultiContextErrorEvent:
    """Parse a JSON text frame into a typed multi-context message."""
    try:
        data = json.loads(raw_text)
    except (json.JSONDecodeError, ValueError) as exc:
        return MultiContextErrorEvent(
            message=f"Invalid JSON: {exc}",
            code="malformed_json",
        )

    if not isinstance(data, dict):
        return MultiContextErrorEvent(
            message="Expected JSON object",
            code="malformed_json",
        )

    msg_type = data.get("type")
    if msg_type is None:
        return MultiContextErrorEvent(
            message="Missing required field: 'type'",
            code="unknown_command",
        )

    msg_class = MULTI_CONTEXT_MESSAGE_TYPES.get(msg_type)
    if msg_class is None:
        return MultiContextErrorEvent(
            message=f"Unknown message type: '{msg_type}'",
            code="unknown_command",
        )

    try:
        return msg_class.model_validate(data)
    except pydantic.ValidationError as exc:
        return MultiContextErrorEvent(
            message=f"Validation error for '{msg_type}': {exc}",
            code="validation_error",
        )


# ---------------------------------------------------------------------------
# TTS synthesis per context
# ---------------------------------------------------------------------------


async def _synthesize_text(
    ws: WebSocket,
    context_id: str,
    text: str,
    *,
    model_tts: str,
    voice_id: str,
    voice_settings: dict[str, object] | None,
    tts_channels: dict[str, grpc.aio.Channel],
    registry: ModelRegistry,
    worker_manager: WorkerManager,
    cancel_event: asyncio.Event,
) -> None:
    """Run TTS synthesis for a text segment and stream audio chunks."""
    from macaw.exceptions import ModelNotFoundError, WorkerUnavailableError

    try:
        _manifest, _worker, address = resolve_tts_resources(
            registry,
            worker_manager,
            model_tts,
        )
    except (ModelNotFoundError, WorkerUnavailableError) as exc:
        await _send_error(ws, str(exc), context_id=context_id, code="model_error")
        return

    channel = get_or_create_tts_channel(
        tts_channels,
        address,
    )
    stub = TTSWorkerStub(channel)  # type: ignore[no-untyped-call]

    proto_request = build_tts_proto_request(
        request_id=str(uuid.uuid4()),
        text=text,
        voice=voice_id,
        sample_rate=TTS_DEFAULT_SAMPLE_RATE,
        speed=1.0,
        voice_settings=voice_settings,
    )

    try:
        response_stream = stub.Synthesize(
            proto_request,
            timeout=TTS_GRPC_TIMEOUT,
        )
        async for chunk in response_stream:
            if cancel_event.is_set():
                break
            audio_b64 = base64.b64encode(chunk.audio_data).decode("ascii")
            await _send_event(
                ws,
                AudioEvent(context_id=context_id, audio=audio_b64),
            )
    except asyncio.CancelledError:
        logger.debug("tts_synthesis_cancelled", context_id=context_id)
    except Exception as exc:
        logger.error("tts_synthesis_error", context_id=context_id, error=str(exc))
        await _send_error(
            ws,
            f"Synthesis error: {exc}",
            context_id=context_id,
            code="synthesis_error",
        )


async def _dispatch_tts_for_context(
    ws: WebSocket,
    ctx_mgr: WSContextManager,
    context_id: str,
    text: str,
    *,
    model_tts: str,
    voice_id: str,
    voice_settings: dict[str, object] | None,
    tts_channels: dict[str, grpc.aio.Channel],
    registry: ModelRegistry,
    worker_manager: WorkerManager,
    tts_semaphore: asyncio.Semaphore,
) -> None:
    """Synthesize text for a context, bounded by concurrency semaphore."""
    ctx = ctx_mgr.get_active_context(context_id)
    if ctx is None:
        return

    async with tts_semaphore:
        await _synthesize_text(
            ws,
            context_id,
            text,
            model_tts=model_tts,
            voice_id=voice_id,
            voice_settings=voice_settings,
            tts_channels=tts_channels,
            registry=registry,
            worker_manager=worker_manager,
            cancel_event=ctx.cancel_event,
        )

    # Check if context transitioned to flushing -> send is_final
    ctx = ctx_mgr.get_context(context_id)
    if ctx is not None and ctx.state == ContextState.FLUSHING:
        ctx.state = ContextState.CLOSED
        await _send_event(ws, IsFinalEvent(context_id=context_id))


# ---------------------------------------------------------------------------
# Inactivity monitor
# ---------------------------------------------------------------------------


async def _inactivity_monitor(
    ws: WebSocket,
    ctx_mgr: WSContextManager,
    connection_last_activity: list[float],
) -> None:
    """Background task that closes inactive contexts."""
    while ws.client_state == WebSocketState.CONNECTED:
        await asyncio.sleep(_INACTIVITY_CHECK_INTERVAL_S)

        # Close individual inactive contexts
        for context_id in ctx_mgr.get_inactive_context_ids():
            if ctx_mgr.close_context(context_id):
                await _send_event(ws, ContextClosedEvent(context_id=context_id))
                logger.info("context_inactivity_timeout", context_id=context_id)

        # Check connection-level inactivity
        now = time.monotonic()
        if now - connection_last_activity[0] > ctx_mgr.inactivity_timeout_s * 3:
            logger.info("connection_inactivity_timeout")
            await _send_error(
                ws,
                "Connection inactivity timeout",
                code="inactivity_timeout",
                recoverable=False,
            )
            return


# ---------------------------------------------------------------------------
# WebSocket endpoint
# ---------------------------------------------------------------------------


@router.websocket("/v1/text-to-speech/{voice_id}/multi-stream-input")
async def multi_context_tts_ws(websocket: WebSocket, voice_id: str) -> None:
    """Multi-context TTS WebSocket endpoint.

    Supports multiple independent TTS contexts on a single connection.
    """
    await websocket.accept()

    state = websocket.app.state
    registry: ModelRegistry | None = getattr(state, "registry", None)
    worker_manager: WorkerManager | None = getattr(state, "worker_manager", None)
    tts_channels: dict[str, grpc.aio.Channel] = getattr(state, "tts_channels", {})

    if registry is None or worker_manager is None:
        await _send_error(
            websocket,
            "TTS service not available",
            code="service_unavailable",
            recoverable=False,
        )
        await websocket.close()
        return

    from macaw.config.settings import get_settings

    mc_settings = get_settings().multi_context
    ctx_mgr = WSContextManager(
        max_contexts=mc_settings.max_contexts,
        inactivity_timeout_s=mc_settings.inactivity_timeout_s,
    )
    tts_semaphore = asyncio.Semaphore(mc_settings.max_concurrent_tts)
    tts_tasks: set[asyncio.Task[None]] = set()

    # Connection state
    initialized = False
    model_tts: str | None = None
    voice_settings: dict[str, object] | None = None
    connection_last_activity: list[float] = [time.monotonic()]

    # Start inactivity monitor
    monitor_task = asyncio.create_task(
        _inactivity_monitor(websocket, ctx_mgr, connection_last_activity),
    )

    try:
        while True:
            message = await websocket.receive()
            connection_last_activity[0] = time.monotonic()

            raw_text = message.get("text")
            if raw_text is None:
                # Binary frames not used in multi-context protocol
                await _send_error(
                    websocket,
                    "Multi-context endpoint only accepts JSON text frames",
                    code="invalid_frame",
                )
                continue

            parsed = _parse_message(raw_text)

            # Parse error -> send error event
            if isinstance(parsed, MultiContextErrorEvent):
                await _send_event(websocket, parsed)
                continue

            # --- InitializeConnection ---
            if isinstance(parsed, InitializeConnection):
                if initialized:
                    await _send_error(
                        websocket,
                        "Connection already initialized",
                        code="already_initialized",
                    )
                    continue

                # Resolve TTS model
                model_tts = parsed.model_tts
                if model_tts is None and registry is not None:
                    model_tts = find_default_tts_model(registry)
                if model_tts is None:
                    await _send_error(
                        websocket,
                        "No TTS model available",
                        code="model_error",
                        recoverable=False,
                    )
                    break

                voice_settings = (
                    parsed.voice_settings if parsed.voice_settings is not None else None
                )
                initialized = True

                await _send_event(
                    websocket,
                    ConnectionInitializedEvent(
                        model_tts=model_tts,
                        max_contexts=ctx_mgr.max_contexts,
                    ),
                )
                logger.info(
                    "multi_context_connection_initialized",
                    model_tts=model_tts,
                    voice_id=voice_id,
                )
                continue

            # All other messages require initialization
            if not initialized:
                await _send_error(
                    websocket,
                    "Connection not initialized. Send 'initialize_connection' first.",
                    code="not_initialized",
                )
                continue

            assert model_tts is not None  # Guaranteed by initialized check

            # --- SendText ---
            if isinstance(parsed, SendText):
                context_id = parsed.context_id

                # Auto-create context if it doesn't exist
                ctx = ctx_mgr.get_active_context(context_id)
                if ctx is None:
                    try:
                        ctx = ctx_mgr.create_context(context_id)
                    except ValueError as exc:
                        await _send_error(
                            websocket,
                            str(exc),
                            context_id=context_id,
                            code="context_limit_reached",
                        )
                        continue

                ctx.touch()
                segments = ctx.text_buffer.append(parsed.text, request_id=context_id)

                # Dispatch synthesis for each ready segment
                for segment_text in segments:
                    task = asyncio.create_task(
                        _dispatch_tts_for_context(
                            websocket,
                            ctx_mgr,
                            context_id,
                            segment_text,
                            model_tts=model_tts,
                            voice_id=voice_id,
                            voice_settings=voice_settings,
                            tts_channels=tts_channels,
                            registry=registry,
                            worker_manager=worker_manager,
                            tts_semaphore=tts_semaphore,
                        ),
                    )
                    tts_tasks.add(task)
                    task.add_done_callback(tts_tasks.discard)
                continue

            # --- FlushContext ---
            if isinstance(parsed, FlushContext):
                text = ctx_mgr.flush_context(parsed.context_id)
                if text is not None:
                    task = asyncio.create_task(
                        _dispatch_tts_for_context(
                            websocket,
                            ctx_mgr,
                            parsed.context_id,
                            text,
                            model_tts=model_tts,
                            voice_id=voice_id,
                            voice_settings=voice_settings,
                            tts_channels=tts_channels,
                            registry=registry,
                            worker_manager=worker_manager,
                            tts_semaphore=tts_semaphore,
                        ),
                    )
                    tts_tasks.add(task)
                    task.add_done_callback(tts_tasks.discard)
                else:
                    # Buffer was empty — immediately send flushed + is_final
                    ctx = ctx_mgr.get_context(parsed.context_id)
                    if ctx is not None and ctx.is_active:
                        ctx.state = ContextState.CLOSED
                        await _send_event(
                            websocket,
                            ContextFlushedEvent(context_id=parsed.context_id),
                        )
                        await _send_event(
                            websocket,
                            IsFinalEvent(context_id=parsed.context_id),
                        )
                    else:
                        await _send_error(
                            websocket,
                            f"Context '{parsed.context_id}' not found or closed",
                            context_id=parsed.context_id,
                            code="context_not_found",
                        )
                continue

            # --- CloseContext ---
            if isinstance(parsed, CloseContext):
                closed = ctx_mgr.close_context(parsed.context_id)
                if closed:
                    await _send_event(
                        websocket,
                        ContextClosedEvent(context_id=parsed.context_id),
                    )
                else:
                    await _send_error(
                        websocket,
                        f"Context '{parsed.context_id}' not found or already closed",
                        context_id=parsed.context_id,
                        code="context_not_found",
                    )
                continue

            # --- CloseSocket ---
            if isinstance(parsed, CloseSocket):
                ctx_mgr.close_all()
                break

            # --- KeepAlive ---
            if isinstance(parsed, KeepAlive):
                connection_last_activity[0] = time.monotonic()
                continue

    except WebSocketDisconnect:
        logger.info("multi_context_client_disconnect", voice_id=voice_id)
    except Exception as exc:
        logger.error("multi_context_unexpected_error", error=str(exc))
    finally:
        # Cleanup: close all contexts, cancel all tasks
        ctx_mgr.close_all()
        monitor_task.cancel()
        for task in list(tts_tasks):
            task.cancel()
        # Wait for task cleanup (best effort)
        if tts_tasks:
            await asyncio.gather(*tts_tasks, return_exceptions=True)
