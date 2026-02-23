"""Pydantic models for the dialogue (multi-speaker TTS) endpoint."""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field

from macaw.server.constants import TTS_MAX_TEXT_LENGTH

# Max unique voices allowed per dialogue request.
MAX_DIALOGUE_VOICES = 10

# Max segments per dialogue request.
MAX_DIALOGUE_SEGMENTS = 100


class DialogueInput(BaseModel):
    """A single dialogue segment: text + voice identifier."""

    text: str = Field(
        min_length=1,
        max_length=TTS_MAX_TEXT_LENGTH,
        description="Text to synthesize for this segment.",
    )
    voice_id: str = Field(
        default="default",
        description="Voice identifier for this segment. Use 'voice_<id>' prefix for saved voices.",
    )


class DialogueRequest(BaseModel):
    """Request body for POST /v1/text-to-dialogue."""

    model: str = Field(description="TTS model name in the registry.")
    inputs: list[DialogueInput] = Field(
        min_length=1,
        max_length=MAX_DIALOGUE_SEGMENTS,
        description="Ordered list of dialogue segments to synthesize.",
    )
    response_format: Literal["wav", "pcm"] = Field(
        default="wav",
        description="Output audio format.",
    )
    output_format: str | None = Field(
        default=None,
        description="Output format string (e.g., 'pcm_16000'). "
        "Takes precedence over response_format.",
    )
    speed: float = Field(
        default=1.0,
        ge=0.25,
        le=4.0,
        description="Synthesis speed applied to all segments.",
    )
    language: str | None = Field(
        default=None,
        description="Target language for all segments.",
    )


class DialogueSegmentInfo(BaseModel):
    """Metadata about a synthesized dialogue segment (for the summary)."""

    index: int
    voice_id: str
    text_length: int
    audio_bytes: int
