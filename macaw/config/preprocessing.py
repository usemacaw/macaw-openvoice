"""Audio Preprocessing Pipeline configuration."""

from __future__ import annotations

from pydantic import BaseModel


class PreprocessingConfig(BaseModel):
    """Audio preprocessing pipeline configuration.

    Each stage can be toggled independently.
    """

    resample: bool = True
    target_sample_rate: int = 16000
    dc_remove: bool = True
    dc_remove_cutoff_hz: int = 20
    gain_normalize: bool = True
    target_dbfs: float = -3.0
