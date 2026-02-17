"""Audio Preprocessing Pipeline configuration."""

from __future__ import annotations

from pydantic import BaseModel

from macaw._audio_constants import DEFAULT_DC_CUTOFF_HZ, DEFAULT_TARGET_DBFS, STT_SAMPLE_RATE


class PreprocessingConfig(BaseModel):
    """Audio preprocessing pipeline configuration.

    Each stage can be toggled independently.
    Defaults are imported from ``_audio_constants`` (single source of truth
    shared with ``PreprocessingSettings``).
    """

    resample: bool = True
    target_sample_rate: int = STT_SAMPLE_RATE
    dc_remove: bool = True
    dc_remove_cutoff_hz: int = DEFAULT_DC_CUTOFF_HZ
    gain_normalize: bool = True
    target_dbfs: float = DEFAULT_TARGET_DBFS
