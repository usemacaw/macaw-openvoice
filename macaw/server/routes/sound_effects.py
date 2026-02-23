"""POST /v1/sound-generation — text-to-sound-effect generation."""

from __future__ import annotations

import io
import struct
import uuid
from typing import TYPE_CHECKING

from fastapi import APIRouter, Request
from fastapi.responses import StreamingResponse

from macaw.exceptions import InvalidRequestError, ServiceUnavailableError
from macaw.logging import get_logger
from macaw.server.constants import TTS_DEFAULT_SAMPLE_RATE
from macaw.server.models.sound_effects import SoundGenerationRequest  # noqa: TC001

if TYPE_CHECKING:
    from collections.abc import AsyncIterator

router = APIRouter(tags=["Sound Effects"])

logger = get_logger("server.routes.sound_effects")


@router.post("/v1/sound-generation")
async def generate_sound(
    body: SoundGenerationRequest,
    request: Request,
) -> StreamingResponse:
    """Generate a sound effect from a text description.

    Accepts a text prompt describing the desired sound and optional
    parameters (duration, prompt influence, loop). Returns audio in
    the requested format.

    Returns 503 if no sound effect engine is configured.
    """
    request_id = str(uuid.uuid4())

    logger.info(
        "sound_generation_request",
        request_id=request_id,
        text_length=len(body.text),
        duration_seconds=body.duration_seconds,
        prompt_influence=body.prompt_influence,
        loop=body.loop,
        output_format=body.output_format,
    )

    # Check if SFX engine is configured
    sfx_backend = getattr(request.app.state, "sfx_backend", None)
    if sfx_backend is None:
        raise ServiceUnavailableError(
            "No sound effect engine configured. "
            "Install a sound effect engine package and configure it in the manifest."
        )

    # Validate output format
    from macaw.codec.output_format import parse_output_format

    try:
        output_fmt = parse_output_format(body.output_format)
    except ValueError as exc:
        raise InvalidRequestError(str(exc)) from exc

    # Pre-flight: check codec availability
    if output_fmt.needs_encoding:
        from macaw.codec import is_codec_available

        if not is_codec_available(output_fmt.codec):
            raise InvalidRequestError(
                f"Codec '{output_fmt.codec}' is not available. Install the required library."
            )

    logger.info(
        "sound_generation_ready",
        request_id=request_id,
        output_format=output_fmt.codec,
    )

    # Placeholder streaming response (returns silence for MVP contract validation)
    # This path is unreachable in production (sfx_backend check above returns 503),
    # but completes the endpoint contract for testing.
    return StreamingResponse(
        _placeholder_stream(output_fmt.codec, TTS_DEFAULT_SAMPLE_RATE, body.duration_seconds),
        media_type=output_fmt.content_type,
    )


async def _placeholder_stream(
    codec: str,
    sample_rate: int,
    duration_s: float,
) -> AsyncIterator[bytes]:
    """Placeholder stream — yields empty WAV or PCM for contract testing."""
    if codec == "wav":
        yield _wav_silence_header(sample_rate, duration_s)


def _wav_silence_header(sample_rate: int, duration_s: float) -> bytes:
    """Build a minimal WAV header for a silence segment."""
    num_channels = 1
    bits_per_sample = 16
    byte_rate = sample_rate * num_channels * bits_per_sample // 8
    block_align = num_channels * bits_per_sample // 8
    data_size = int(duration_s * byte_rate)

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
    # Append silence data
    buf.write(b"\x00" * data_size)
    return buf.getvalue()
