"""Gain Normalize stage for the Audio Preprocessing Pipeline.

Normalizes audio amplitude to a target peak level in dBFS.
Ensures consistent input for VAD and inference engines.
"""

from __future__ import annotations

import numpy as np

from macaw.preprocessing.stages import AudioStage

# Threshold below which audio is considered silence.
# Prevents division by zero and background noise amplification.
_SILENCE_THRESHOLD = 1e-10


class GainNormalizeStage(AudioStage):
    """Normalize audio amplitude to target peak level.

    Calculates the signal peak and applies gain to reach the
    target_dbfs level. Includes clipping protection.

    Args:
        target_dbfs: Target peak level in dBFS. Default: -3.0.
    """

    def __init__(self, target_dbfs: float = -3.0) -> None:
        self._target_dbfs = target_dbfs
        self._target_linear = 10 ** (target_dbfs / 20)

    @property
    def name(self) -> str:
        """Identifier name for the stage."""
        return "gain_normalize"

    def process(self, audio: np.ndarray, sample_rate: int) -> tuple[np.ndarray, int]:
        """Normalize audio amplitude to target peak level.

        Args:
            audio: Numpy float32 array with audio samples (mono).
            sample_rate: Current audio sample rate in Hz.

        Returns:
            Tuple (normalized float32 audio, unchanged sample rate).
        """
        if len(audio) == 0:
            return audio, sample_rate

        peak = np.max(np.abs(audio))

        if peak < _SILENCE_THRESHOLD:
            return audio, sample_rate

        gain = self._target_linear / peak
        normalized = np.clip(audio * gain, -1.0, 1.0)

        return normalized, sample_rate
