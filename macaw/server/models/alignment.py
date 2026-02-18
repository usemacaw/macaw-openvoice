"""Pydantic models for TTS alignment NDJSON response.

When include_alignment=true on POST /v1/audio/speech, the response
switches from binary audio to NDJSON (application/x-ndjson). Each line
is a JSON object: either an audio chunk with alignment or a done marker.
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
