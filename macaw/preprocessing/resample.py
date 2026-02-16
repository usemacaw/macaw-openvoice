"""ResampleStage — converts audio to target sample rate.

Uses scipy.signal.resample_poly for high-quality resampling.
Pipeline contract guarantees mono input; no multi-channel handling here.
"""

from __future__ import annotations

from math import gcd

import numpy as np
from scipy.signal import resample_poly

from macaw._audio_constants import STT_SAMPLE_RATE
from macaw.preprocessing.stages import AudioStage


class ResampleStage(AudioStage):
    """Resampling stage for the preprocessing pipeline.

    Converts audio from any sample rate to the target sample rate (default 16kHz).
    Expects mono input (pipeline contract).

    Args:
        target_sample_rate: Target sample rate in Hz (default: 16000).
    """

    def __init__(self, target_sample_rate: int = STT_SAMPLE_RATE) -> None:
        self._target_sample_rate = target_sample_rate

    @property
    def name(self) -> str:
        """Identifier name for the stage."""
        return "resample"

    def process(self, audio: np.ndarray, sample_rate: int) -> tuple[np.ndarray, int]:
        """Convert audio to target sample rate.

        If the audio is already at the target sample rate, returns unchanged.
        If the audio is empty, returns unchanged.

        Args:
            audio: Numpy array with audio samples.
            sample_rate: Current audio sample rate in Hz.

        Returns:
            Tuple (resampled float32 audio, target sample rate).
        """
        if audio.size == 0:
            return audio, sample_rate

        # Skip if already at target sample rate
        if sample_rate == self._target_sample_rate:
            return audio, sample_rate

        # Calculate up/down factors simplified by GCD
        divisor = gcd(self._target_sample_rate, sample_rate)
        up = self._target_sample_rate // divisor
        down = sample_rate // divisor

        resampled = resample_poly(audio, up, down).astype(np.float32)

        return resampled, self._target_sample_rate
