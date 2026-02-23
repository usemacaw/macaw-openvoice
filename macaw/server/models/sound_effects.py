"""Pydantic models for the sound effects generation endpoint."""

from __future__ import annotations

from pydantic import BaseModel, Field


class SoundGenerationRequest(BaseModel):
    """Request body for POST /v1/sound-generation."""

    text: str = Field(
        ...,
        min_length=1,
        max_length=1000,
        description="Text description of the desired sound effect.",
    )
    duration_seconds: float = Field(
        default=5.0,
        ge=0.5,
        le=30.0,
        description="Desired duration of the sound effect in seconds.",
    )
    prompt_influence: float = Field(
        default=0.3,
        ge=0.0,
        le=1.0,
        description="How closely to follow the text prompt (0=loose, 1=strict).",
    )
    loop: bool = Field(
        default=False,
        description="Generate seamless looping audio.",
    )
    output_format: str = Field(
        default="wav",
        description="Output audio format (wav, pcm, mp3, opus).",
    )


class SoundGenerationResponse(BaseModel):
    """Metadata response for sound generation requests."""

    request_id: str
    duration_seconds: float
    output_format: str
