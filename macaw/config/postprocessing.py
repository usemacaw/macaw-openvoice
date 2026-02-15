"""Post-Processing Pipeline configuration."""

from __future__ import annotations

from pydantic import BaseModel


class ITNConfig(BaseModel):
    """Inverse Text Normalization configuration."""

    enabled: bool = True
    language: str = "pt"


class PostProcessingConfig(BaseModel):
    """Text post-processing pipeline configuration."""

    itn: ITNConfig = ITNConfig()
