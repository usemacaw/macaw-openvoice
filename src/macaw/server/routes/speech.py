"""POST /v1/audio/speech — speech synthesis (TTS)."""

from __future__ import annotations

import io
import struct
import uuid
from typing import TYPE_CHECKING

import grpc.aio
from fastapi import APIRouter, Depends
from fastapi.responses import Response

from macaw._types import ModelType
from macaw.exceptions import (
    InvalidRequestError,
    ModelNotFoundError,
    WorkerCrashError,
    WorkerTimeoutError,
    WorkerUnavailableError,
)
from macaw.logging import get_logger
from macaw.proto.tts_worker_pb2_grpc import TTSWorkerStub
from macaw.registry.registry import ModelRegistry  # noqa: TC001
from macaw.scheduler.tts_converters import build_tts_proto_request, tts_proto_chunks_to_result
from macaw.server.dependencies import get_registry, get_worker_manager
from macaw.server.models.speech import SpeechRequest  # noqa: TC001
from macaw.workers.manager import WorkerManager  # noqa: TC001

if TYPE_CHECKING:
    from macaw._types import TTSSpeechResult
    from macaw.proto.tts_worker_pb2 import SynthesizeRequest

router = APIRouter()

logger = get_logger("server.routes.speech")

# Default sample rate for TTS (24kHz is the default for most TTS engines)
_DEFAULT_SAMPLE_RATE = 24000

# Timeout for the gRPC Synthesize call (seconds)
_TTS_GRPC_TIMEOUT = 60.0

# Supported response formats
_SUPPORTED_FORMATS = frozenset({"wav", "pcm"})

# gRPC channel options
_GRPC_CHANNEL_OPTIONS = [
    ("grpc.max_send_message_length", 30 * 1024 * 1024),
    ("grpc.max_receive_message_length", 30 * 1024 * 1024),
]


@router.post("/v1/audio/speech")
async def create_speech(
    body: SpeechRequest,
    registry: ModelRegistry = Depends(get_registry),  # noqa: B008
    worker_manager: WorkerManager = Depends(get_worker_manager),  # noqa: B008
) -> Response:
    """Synthesize audio from text.

    Compatible with OpenAI Audio API POST /v1/audio/speech.
    Returns binary audio in the body (not JSON).
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

    # Validate non-empty text
    if not body.input.strip():
        raise InvalidRequestError("O campo 'input' nao pode estar vazio.")

    # Validate response format
    if body.response_format not in _SUPPORTED_FORMATS:
        valid = ", ".join(sorted(_SUPPORTED_FORMATS))
        raise InvalidRequestError(
            f"Invalid response_format '{body.response_format}'. Accepted values: {valid}"
        )

    # Validate that the model exists and is of type TTS
    manifest = registry.get_manifest(body.model)
    if manifest.model_type != ModelType.TTS:
        raise ModelNotFoundError(body.model)

    # Find a ready TTS worker
    worker = worker_manager.get_ready_worker(body.model)
    if worker is None:
        raise WorkerUnavailableError(body.model)

    # Build proto request
    proto_request = build_tts_proto_request(
        request_id=request_id,
        text=body.input,
        voice=body.voice,
        sample_rate=_DEFAULT_SAMPLE_RATE,
        speed=body.speed,
    )

    # Send to TTS worker via gRPC (server-streaming)
    result = await _synthesize_via_grpc(
        worker_address=f"localhost:{worker.port}",
        proto_request=proto_request,
        voice=body.voice,
        worker_id=worker.worker_id,
    )

    logger.info(
        "speech_done",
        request_id=request_id,
        audio_bytes=len(result.audio_data),
        duration=result.duration,
    )

    # Format response
    if body.response_format == "wav":
        audio_bytes = _pcm_to_wav(result.audio_data, result.sample_rate)
        return Response(content=audio_bytes, media_type="audio/wav")

    # PCM raw
    return Response(content=result.audio_data, media_type="audio/pcm")


async def _synthesize_via_grpc(
    *,
    worker_address: str,
    proto_request: SynthesizeRequest,
    voice: str,
    worker_id: str,
) -> TTSSpeechResult:
    """Send TTS request to the worker via gRPC and collect audio chunks."""
    channel = grpc.aio.insecure_channel(worker_address, options=_GRPC_CHANNEL_OPTIONS)
    try:
        stub = TTSWorkerStub(channel)  # type: ignore[no-untyped-call]

        chunks: list[bytes] = []
        total_duration = 0.0

        response_stream = stub.Synthesize(proto_request, timeout=_TTS_GRPC_TIMEOUT)

        async for chunk in response_stream:
            if chunk.audio_data:
                chunks.append(chunk.audio_data)
            total_duration = chunk.duration
            if chunk.is_last:
                break

        return tts_proto_chunks_to_result(
            chunks,
            sample_rate=_DEFAULT_SAMPLE_RATE,
            voice=voice,
            total_duration=total_duration,
        )

    except grpc.aio.AioRpcError as exc:
        code = exc.code()
        if code == grpc.StatusCode.DEADLINE_EXCEEDED:
            raise WorkerTimeoutError(worker_id, _TTS_GRPC_TIMEOUT) from exc
        if code == grpc.StatusCode.UNAVAILABLE:
            raise WorkerUnavailableError("tts") from exc
        raise WorkerCrashError(worker_id) from exc
    finally:
        try:
            await channel.close()
        except Exception:
            logger.warning("tts_channel_close_error", worker_address=worker_address)


def _pcm_to_wav(pcm_data: bytes, sample_rate: int) -> bytes:
    """Convert 16-bit mono PCM audio to WAV format."""
    num_channels = 1
    bits_per_sample = 16
    byte_rate = sample_rate * num_channels * bits_per_sample // 8
    block_align = num_channels * bits_per_sample // 8
    data_size = len(pcm_data)

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
    buf.write(pcm_data)

    return buf.getvalue()
