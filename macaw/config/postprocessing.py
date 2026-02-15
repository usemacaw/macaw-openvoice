"""Post-Processing Pipeline configuration."""

from __future__ import annotations

from pydantic import BaseModel


class ITNConfig(BaseModel):
    """Inverse Text Normalization configuration."""

    enabled: bool = True
    language: str = "pt"


class EntityFormattingConfig(BaseModel):
    """Entity formatting configuration."""

    enabled: bool = False
    domain: str = "generic"


class HotWordCorrectionConfig(BaseModel):
    """Hot word correction configuration."""

    enabled: bool = False
    max_edit_distance: int = 2


class PostProcessingConfig(BaseModel):
    """Text post-processing pipeline configuration."""

    itn: ITNConfig = ITNConfig()
    entity_formatting: EntityFormattingConfig = EntityFormattingConfig()
    hot_word_correction: HotWordCorrectionConfig = HotWordCorrectionConfig()
