"""Pydantic models for alignment responses.

Includes models for:
- TTS alignment NDJSON streaming (POST /v1/audio/speech with include_alignment)
- Forced alignment endpoint (POST /v1/audio/align)
"""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel


class AlignmentItemResponse(BaseModel):
    """Single word/character timing in an alignment response."""

    text: str
    start_ms: int
    duration_ms: int


class ChunkAlignmentResponse(BaseModel):
    """Alignment data for an audio chunk."""

    items: list[AlignmentItemResponse]
    granularity: Literal["word", "character"] = "word"


class AudioChunkWithAlignment(BaseModel):
    """NDJSON line: audio chunk with optional alignment data.

    audio is base64-encoded PCM/WAV/Opus bytes.
    ``alignment`` maps timing to the original text.
    ``normalized_alignment`` maps timing to the normalized/phonemized text
    (only available for engines that expose phoneme data, e.g. Kokoro).
    """

    type: Literal["audio"] = "audio"
    audio: str
    alignment: ChunkAlignmentResponse | None = None
    normalized_alignment: ChunkAlignmentResponse | None = None


class AlignmentStreamDone(BaseModel):
    """NDJSON line: signals end of audio stream."""

    type: Literal["done"] = "done"
    duration: float
    alignment_available: bool = True


# --- Forced alignment endpoint (POST /v1/audio/align) ---


class ForceAlignmentResponse(BaseModel):
    """Response for POST /v1/audio/align."""

    items: list[AlignmentItemResponse]
    language: str
    granularity: Literal["word", "character"]
    audio_duration_ms: int
