"""POST /v1/audio/speech — speech synthesis (TTS)."""

from __future__ import annotations

import base64
import io
import struct
import uuid
from typing import TYPE_CHECKING, Any

import grpc.aio
from fastapi import APIRouter, Depends, Request
from fastapi.responses import Response, StreamingResponse

from macaw.exceptions import (
    InvalidRequestError,
    VoiceNotFoundError,
    WorkerCrashError,
    WorkerTimeoutError,
    WorkerUnavailableError,
)
from macaw.logging import get_logger
from macaw.proto.tts_worker_pb2_grpc import TTSWorkerStub
from macaw.registry.registry import ModelRegistry  # noqa: TC001
from macaw.scheduler.tts_converters import build_tts_proto_request
from macaw.server.constants import TTS_DEFAULT_SAMPLE_RATE, TTS_GRPC_TIMEOUT
from macaw.server.dependencies import get_registry, get_worker_manager
from macaw.server.grpc_channels import get_or_create_tts_channel
from macaw.server.models.speech import SpeechRequest  # noqa: TC001
from macaw.server.tts_service import resolve_tts_resources
from macaw.workers.manager import WorkerManager  # noqa: TC001

if TYPE_CHECKING:
    from collections.abc import AsyncIterator

    from macaw.proto.tts_worker_pb2 import SynthesizeRequest

router = APIRouter(tags=["Audio"])

logger = get_logger("server.routes.speech")

# Supported response formats (enforced by Literal type in SpeechRequest)
_SUPPORTED_FORMATS = frozenset({"wav", "pcm"})


@router.post("/v1/audio/speech")
async def create_speech(
    body: SpeechRequest,
    request: Request,
    registry: ModelRegistry = Depends(get_registry),  # noqa: B008
    worker_manager: WorkerManager = Depends(get_worker_manager),  # noqa: B008
) -> Response:
    """Synthesize audio from text.

    Compatible with OpenAI Audio API POST /v1/audio/speech.
    Returns binary audio in the body (not JSON).

    Uses StreamingResponse to reduce TTFB: audio chunks are sent
    to the client as they arrive from the TTS worker, instead of
    accumulating all chunks before responding.
    """
    request_id = str(uuid.uuid4())

    logger.info(
        "speech_request",
        request_id=request_id,
        model=body.model,
        voice=body.voice,
        text_length=len(body.input),
        response_format=body.response_format,
    )

    # Validate non-empty text (whitespace-only not caught by min_length)
    if not body.input.strip():
        raise InvalidRequestError("The 'input' field cannot be empty.")

    # response_format validated by Pydantic Literal["wav", "pcm"]

    # Resolve TTS model + worker
    _manifest, worker, worker_address = resolve_tts_resources(registry, worker_manager, body.model)

    # Decode base64 ref_audio if present
    ref_audio_bytes: bytes | None = None
    if body.ref_audio:
        try:
            ref_audio_bytes = base64.b64decode(body.ref_audio)
        except Exception as exc:
            raise InvalidRequestError(f"Invalid base64 in 'ref_audio': {exc}") from exc

    # Resolve saved voice if voice starts with "voice_"
    voice = body.voice
    ref_text = body.ref_text
    instruction = body.instruction
    language = body.language

    if voice.startswith("voice_"):
        saved_voice_id = voice[len("voice_") :]
        voice_store = request.app.state.voice_store
        if voice_store is None:
            raise InvalidRequestError("VoiceStore not configured. Cannot resolve saved voice.")

        saved = await voice_store.get(saved_voice_id)
        if saved is None:
            raise VoiceNotFoundError(saved_voice_id)

        # Conflict: inline ref_audio + saved voice with ref_audio
        if ref_audio_bytes is not None and saved.ref_audio_path is not None:
            raise InvalidRequestError(
                "Cannot provide both inline ref_audio and a saved cloned voice."
            )

        # Inject saved voice params (saved values fill gaps, don't override inline)
        if saved.ref_audio_path is not None and ref_audio_bytes is None:
            with open(saved.ref_audio_path, "rb") as f:
                ref_audio_bytes = f.read()
        if saved.ref_text is not None and ref_text is None:
            ref_text = saved.ref_text
        if saved.instruction is not None and instruction is None:
            instruction = saved.instruction
        if saved.language is not None and language is None:
            language = saved.language

        # Use "default" as voice for the engine (saved voice provides params)
        voice = "default"

    # Build proto request
    proto_request = build_tts_proto_request(
        request_id=request_id,
        text=body.input,
        voice=voice,
        sample_rate=TTS_DEFAULT_SAMPLE_RATE,
        speed=body.speed,
        language=language,
        ref_audio=ref_audio_bytes,
        ref_text=ref_text,
        instruction=instruction,
    )

    # Get pooled TTS channel (reused across requests)
    tts_channels: dict[str, grpc.aio.Channel] = request.app.state.tts_channels
    channel = get_or_create_tts_channel(tts_channels, worker_address)

    # Pre-flight: open gRPC stream and fetch first audio chunk.
    # This validates the connection and request params BEFORE starting
    # the StreamingResponse (so we can still return proper HTTP errors).
    response_stream, first_audio_chunk = await _open_tts_stream(
        channel=channel,
        proto_request=proto_request,
        worker_id=worker.worker_id,
    )

    # Stream response — TTFB is now time-to-first-gRPC-chunk instead of
    # total synthesis time.
    is_wav = body.response_format == "wav"
    media_type = "audio/wav" if is_wav else "audio/pcm"
    return StreamingResponse(
        _stream_tts_audio(
            response_stream=response_stream,
            first_audio_chunk=first_audio_chunk,
            is_wav=is_wav,
            sample_rate=TTS_DEFAULT_SAMPLE_RATE,
            request_id=request_id,
        ),
        media_type=media_type,
    )


async def _open_tts_stream(
    *,
    channel: grpc.aio.Channel,
    proto_request: SynthesizeRequest,
    worker_id: str,
) -> tuple[grpc.aio.UnaryStreamCall[Any, Any], bytes]:
    """Open gRPC stream and fetch first audio chunk (pre-flight validation).

    Uses a pooled channel (not closed on error — gRPC channels handle
    reconnection automatically).

    Returns the response stream iterator and the first audio bytes.

    Raises domain exceptions on gRPC errors so the HTTP layer can return
    proper status codes before the StreamingResponse starts.
    """
    try:
        stub = TTSWorkerStub(channel)  # type: ignore[no-untyped-call]
        response_stream = stub.Synthesize(proto_request, timeout=TTS_GRPC_TIMEOUT)

        # Read chunks until we get one with audio_data (pre-flight)
        first_audio_chunk = b""
        async for chunk in response_stream:
            if chunk.audio_data:
                first_audio_chunk = chunk.audio_data
                break
            if chunk.is_last:
                break

        return response_stream, first_audio_chunk

    except grpc.aio.AioRpcError as exc:
        code = exc.code()
        if code == grpc.StatusCode.DEADLINE_EXCEEDED:
            raise WorkerTimeoutError(worker_id, TTS_GRPC_TIMEOUT) from exc
        if code == grpc.StatusCode.UNAVAILABLE:
            raise WorkerUnavailableError("tts") from exc
        if code == grpc.StatusCode.INVALID_ARGUMENT:
            detail = exc.details() or "Synthesis failed"
            raise InvalidRequestError(detail) from exc
        raise WorkerCrashError(worker_id) from exc


async def _stream_tts_audio(
    *,
    response_stream: grpc.aio.UnaryStreamCall[Any, Any],
    first_audio_chunk: bytes,
    is_wav: bool,
    sample_rate: int,
    request_id: str,
) -> AsyncIterator[bytes]:
    """Async generator that yields TTS audio chunks for StreamingResponse.

    For WAV format: yields a WAV header (with max data size placeholder)
    followed by raw PCM chunks. For PCM format: yields raw PCM directly.

    The gRPC channel is pooled and NOT closed here — it is reused across
    requests and closed on server shutdown via close_tts_channels().
    """
    total_audio_bytes = 0
    try:
        # WAV header with streaming-compatible max size placeholder
        if is_wav:
            yield _wav_streaming_header(sample_rate)

        # Yield pre-fetched first chunk
        if first_audio_chunk:
            total_audio_bytes += len(first_audio_chunk)
            yield first_audio_chunk

        # Stream remaining chunks
        async for chunk in response_stream:
            if chunk.audio_data:
                total_audio_bytes += len(chunk.audio_data)
                yield chunk.audio_data
            if chunk.is_last:
                break

        logger.info(
            "speech_done",
            request_id=request_id,
            audio_bytes=total_audio_bytes,
        )

    except grpc.aio.AioRpcError as exc:
        # Error after streaming started — can't change HTTP status.
        # Log and let the client detect truncation.
        logger.error(
            "tts_stream_error_mid_response",
            request_id=request_id,
            grpc_code=str(exc.code()),
            audio_bytes_sent=total_audio_bytes,
        )
    except Exception:
        logger.exception(
            "tts_stream_unexpected_error",
            request_id=request_id,
            audio_bytes_sent=total_audio_bytes,
        )


def _wav_streaming_header(sample_rate: int) -> bytes:
    """Build a WAV header for streaming (unknown total size).

    Uses 0x7FFFFFFF as data_size — a well-known convention for
    streaming WAV that most audio players handle correctly.
    """
    num_channels = 1
    bits_per_sample = 16
    byte_rate = sample_rate * num_channels * bits_per_sample // 8
    block_align = num_channels * bits_per_sample // 8
    # Max size placeholder for streaming
    data_size = 0x7FFFFFFF

    buf = io.BytesIO()
    # RIFF header
    buf.write(b"RIFF")
    buf.write(struct.pack("<I", 36 + data_size))
    buf.write(b"WAVE")
    # fmt subchunk
    buf.write(b"fmt ")
    buf.write(struct.pack("<I", 16))
    buf.write(struct.pack("<H", 1))  # PCM format
    buf.write(struct.pack("<H", num_channels))
    buf.write(struct.pack("<I", sample_rate))
    buf.write(struct.pack("<I", byte_rate))
    buf.write(struct.pack("<H", block_align))
    buf.write(struct.pack("<H", bits_per_sample))
    # data subchunk
    buf.write(b"data")
    buf.write(struct.pack("<I", data_size))

    return buf.getvalue()
