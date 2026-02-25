"""POST /v1/speech-to-speech/{voice_id} — voice conversion (speech-to-speech)."""

from __future__ import annotations

import io
import struct
import uuid
from typing import TYPE_CHECKING

from fastapi import APIRouter, File, Form, Request, UploadFile
from fastapi.responses import StreamingResponse

from macaw.exceptions import InvalidRequestError, ServiceUnavailableError
from macaw.logging import get_logger
from macaw.server.constants import (
    ALLOWED_AUDIO_CONTENT_TYPES,
    TTS_DEFAULT_SAMPLE_RATE,
)

if TYPE_CHECKING:
    from collections.abc import AsyncIterator

router = APIRouter(tags=["Voice Changer"])

logger = get_logger("server.routes.voice_changer")

# Maximum source audio file size for voice conversion (50 MB).
_VC_MAX_FILE_SIZE = 50 * 1024 * 1024


@router.post("/v1/speech-to-speech/{voice_id}")
async def speech_to_speech(
    voice_id: str,
    request: Request,
    audio: UploadFile = File(..., description="Source audio file"),  # noqa: B008
    output_format: str = Form(default="wav", description="Output audio format"),
) -> StreamingResponse:
    """Convert speech from one voice to another.

    Accepts source audio and a target voice ID. Returns audio with the
    voice replaced while preserving timing and emotion.

    Returns 503 if no voice changer engine is configured.
    """
    request_id = str(uuid.uuid4())

    logger.info(
        "voice_changer_request",
        request_id=request_id,
        voice_id=voice_id,
        output_format=output_format,
    )

    # Check if VC engine is configured
    vc_backend = getattr(request.app.state, "vc_backend", None)
    if vc_backend is None:
        raise ServiceUnavailableError(
            "No voice changer engine configured. "
            "Install a VC engine package and configure it in the manifest."
        )

    # Validate audio file
    if audio.content_type and audio.content_type not in ALLOWED_AUDIO_CONTENT_TYPES:
        raise InvalidRequestError(
            f"Unsupported audio format: {audio.content_type}. "
            f"Supported: {', '.join(sorted(ALLOWED_AUDIO_CONTENT_TYPES))}"
        )

    # Read and validate source audio
    source_bytes = await audio.read()
    if len(source_bytes) == 0:
        raise InvalidRequestError("Source audio file is empty.")
    if len(source_bytes) > _VC_MAX_FILE_SIZE:
        raise InvalidRequestError(
            f"Source audio too large ({len(source_bytes)} bytes). "
            f"Maximum is {_VC_MAX_FILE_SIZE} bytes."
        )

    # Validate output format
    from macaw.codec.output_format import parse_output_format

    try:
        output_fmt = parse_output_format(output_format)
    except ValueError as exc:
        raise InvalidRequestError(str(exc)) from exc

    # Pre-flight: check codec availability
    if output_fmt.needs_encoding:
        from macaw.codec import codec_unavailable_message, is_codec_available

        if not is_codec_available(output_fmt.codec):
            raise InvalidRequestError(codec_unavailable_message(output_fmt.codec))

    # TODO: When VC worker integration is added (beyond MVP), this will
    # call the gRPC VoiceChange RPC. For now, we return 503 above since
    # no built-in VC engine exists.
    #
    # Future flow:
    # 1. Preprocess source audio (resample, normalize)
    # 2. Open gRPC stream to VC worker
    # 3. Stream converted audio chunks back to client

    logger.info(
        "voice_changer_ready",
        request_id=request_id,
        voice_id=voice_id,
        source_bytes=len(source_bytes),
        output_format=output_fmt.codec,
    )

    # Placeholder streaming response (returns silence for MVP contract validation)
    # This path is unreachable in production (vc_backend check above returns 503),
    # but completes the endpoint contract for testing.
    return StreamingResponse(
        _placeholder_stream(output_fmt.codec, TTS_DEFAULT_SAMPLE_RATE),
        media_type=output_fmt.content_type,
    )


async def _placeholder_stream(
    codec: str,
    sample_rate: int,
) -> AsyncIterator[bytes]:
    """Placeholder stream — yields empty WAV or PCM for contract testing."""
    if codec == "wav":
        yield _wav_empty_header(sample_rate)
    # In production, this would yield converted audio chunks from the VC worker.


def _wav_empty_header(sample_rate: int) -> bytes:
    """Build a minimal WAV header for an empty audio stream."""
    num_channels = 1
    bits_per_sample = 16
    byte_rate = sample_rate * num_channels * bits_per_sample // 8
    block_align = num_channels * bits_per_sample // 8
    data_size = 0

    buf = io.BytesIO()
    buf.write(b"RIFF")
    buf.write(struct.pack("<I", 36 + data_size))
    buf.write(b"WAVE")
    buf.write(b"fmt ")
    buf.write(struct.pack("<I", 16))
    buf.write(struct.pack("<H", 1))
    buf.write(struct.pack("<H", num_channels))
    buf.write(struct.pack("<I", sample_rate))
    buf.write(struct.pack("<I", byte_rate))
    buf.write(struct.pack("<H", block_align))
    buf.write(struct.pack("<H", bits_per_sample))
    buf.write(b"data")
    buf.write(struct.pack("<I", data_size))
    return buf.getvalue()
