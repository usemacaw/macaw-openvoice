"""Post-Processing Pipeline configuration."""

from __future__ import annotations

from pydantic import BaseModel

from macaw._audio_constants import DEFAULT_ITN_LANGUAGE


class ITNConfig(BaseModel):
    """Inverse Text Normalization configuration.

    Default language imported from ``_audio_constants`` (single source of truth
    shared with ``PostProcessingSettings``).
    """

    enabled: bool = True
    language: str = DEFAULT_ITN_LANGUAGE


class PostProcessingConfig(BaseModel):
    """Text post-processing pipeline configuration."""

    itn: ITNConfig = ITNConfig()
