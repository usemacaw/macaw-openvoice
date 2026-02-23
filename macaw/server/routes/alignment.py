"""POST /v1/audio/align -- forced alignment endpoint."""

from __future__ import annotations

import io
import wave
from typing import Literal

from fastapi import APIRouter, Form, Request, UploadFile

from macaw._audio_constants import BYTES_PER_SAMPLE_INT16, STT_SAMPLE_RATE
from macaw.exceptions import InvalidRequestError, ServiceUnavailableError
from macaw.server.models.alignment import AlignmentItemResponse, ForceAlignmentResponse

router = APIRouter(tags=["Audio"])

_MS_PER_SECOND = 1000


def _read_wav(data: bytes) -> tuple[bytes, int]:
    """Extract raw PCM bytes and sample rate from a WAV file.

    Uses the Python stdlib ``wave`` module (no extra dependencies).

    Raises:
        InvalidRequestError: If the data is not a valid WAV file.
    """
    try:
        with wave.open(io.BytesIO(data)) as f:
            sample_rate = f.getframerate()
            pcm = f.readframes(f.getnframes())
    except wave.Error as exc:
        raise InvalidRequestError(f"Invalid WAV file: {exc}") from exc
    return pcm, sample_rate


def _is_wav(data: bytes) -> bool:
    """Return True if *data* starts with a RIFF/WAVE header."""
    return data[:4] == b"RIFF" and data[8:12] == b"WAVE"


@router.post("/v1/audio/align", response_model=ForceAlignmentResponse)
async def create_alignment(
    request: Request,
    file: UploadFile,
    text: str = Form(default=""),
    language: str = Form(default="en"),
    granularity: Literal["word", "character"] = Form(default="word"),
) -> ForceAlignmentResponse:
    """Align text to audio using CTC forced alignment.

    Accepts a WAV or raw PCM (16kHz, 16-bit mono) audio file and the
    corresponding text.  Returns per-word or per-character timing data.
    """
    # --- Validate text ---
    if not text.strip():
        raise InvalidRequestError("Text must not be empty.")

    # --- Read and validate audio ---
    raw_data = await file.read()
    if not raw_data:
        raise InvalidRequestError("Audio file must not be empty.")

    if _is_wav(raw_data):
        pcm_bytes, sample_rate = _read_wav(raw_data)
    else:
        # Treat as raw PCM 16kHz 16-bit mono.
        pcm_bytes = raw_data
        sample_rate = STT_SAMPLE_RATE

    if not pcm_bytes:
        raise InvalidRequestError("Audio file contains no audio data.")

    # --- Get or lazy-init aligner ---
    aligner = getattr(request.app.state, "aligner", None)
    if aligner is None:
        from macaw.alignment import create_aligner

        aligner = create_aligner()
        request.app.state.aligner = aligner

    if aligner is None:
        raise ServiceUnavailableError("Forced alignment requires torchaudio >= 2.1")

    # --- Run alignment ---
    items = await aligner.align(
        audio=pcm_bytes,
        text=text,
        sample_rate=sample_rate,
        language=language,
        granularity=granularity,
    )

    # --- Calculate audio duration ---
    audio_duration_ms = int(
        len(pcm_bytes) / (sample_rate * BYTES_PER_SAMPLE_INT16) * _MS_PER_SECOND
    )

    return ForceAlignmentResponse(
        items=[
            AlignmentItemResponse(
                text=item.text,
                start_ms=item.start_ms,
                duration_ms=item.duration_ms,
            )
            for item in items
        ],
        language=language,
        granularity=granularity,
        audio_duration_ms=audio_duration_ms,
    )
