"""Pydantic model for audio effects parameters."""

from __future__ import annotations

from pydantic import BaseModel, Field


class AudioEffectsParams(BaseModel):
    """Audio effects applied after TTS synthesis, before transport.

    All fields are optional — omitting or using defaults means no effect.
    """

    pitch_shift_semitones: float = Field(
        default=0.0,
        ge=-12.0,
        le=12.0,
        description="Pitch shift in semitones (-12 to +12). 0 = no shift.",
    )
    reverb_room_size: float | None = Field(
        default=None,
        ge=0.0,
        le=1.0,
        description="Reverb room size (0.0 = small, 1.0 = large).",
    )
    reverb_damping: float | None = Field(
        default=None,
        ge=0.0,
        le=1.0,
        description="Reverb high-frequency damping (0.0 = bright, 1.0 = dark).",
    )
    reverb_wet_dry_mix: float | None = Field(
        default=None,
        ge=0.0,
        le=1.0,
        description="Reverb wet/dry balance (0.0 = dry, 1.0 = fully wet).",
    )
