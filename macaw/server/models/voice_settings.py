"""Pydantic model for voice_settings (ElevenLabs-compatible schema).

Provides abstract voice characteristics that are mapped to engine-specific
parameters by each TTSBackend via ``map_voice_settings()``.
"""

from __future__ import annotations

from pydantic import BaseModel, Field


class VoiceSettings(BaseModel):
    """Abstract voice settings matching the ElevenLabs voice_settings schema.

    These settings describe desired voice characteristics in an
    engine-agnostic way. Each TTSBackend maps them to its own
    parameters (e.g., stability -> temperature inversion for
    LLM-based engines).
    """

    stability: float = Field(
        default=0.5,
        ge=0.0,
        le=1.0,
        description="Voice stability (0.0 = variable/expressive, 1.0 = stable/consistent).",
    )
    similarity_boost: float = Field(
        default=0.75,
        ge=0.0,
        le=1.0,
        description="How closely the output matches the original voice.",
    )
    style: float = Field(
        default=0.0,
        ge=0.0,
        le=1.0,
        description="Speaking style exaggeration (0.0 = neutral, 1.0 = maximum style).",
    )
    use_speaker_boost: bool = Field(
        default=True,
        description="Whether to apply speaker similarity boost.",
    )
    speed: float = Field(
        default=1.0,
        ge=0.25,
        le=4.0,
        description="Synthesis speed (0.25-4.0).",
    )
