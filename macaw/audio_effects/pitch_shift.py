"""Pitch shift effect via time-axis interpolation."""

from __future__ import annotations

import numpy as np

from macaw.audio_effects.interface import AudioEffect


class PitchShiftEffect(AudioEffect):
    """Shift audio pitch by a given number of semitones.

    Uses direct time-axis manipulation: reads the input at a different
    speed via linear interpolation. For pitch UP (ratio > 1), samples are
    read faster; for pitch DOWN (ratio < 1), samples are read slower.
    Positions beyond the input range are clamped to the last sample.

    This approach is simple, uses no new dependencies, and produces
    acceptable quality for +/-12 semitones on short TTS audio.
    """

    def __init__(self, semitones: float) -> None:
        self._semitones = semitones
        self._ratio = 2.0 ** (semitones / 12.0)

    @property
    def name(self) -> str:
        return "pitch_shift"

    def process(self, audio: np.ndarray, sample_rate: int) -> np.ndarray:
        """Apply pitch shift while preserving output length."""
        if self._semitones == 0.0:
            return audio

        n = len(audio)
        if n == 0:
            return audio

        # Map output positions to input positions at shifted speed.
        # ratio > 1 reads faster (higher pitch), ratio < 1 reads slower (lower pitch).
        input_positions = np.arange(n, dtype=np.float64) * self._ratio
        np.clip(input_positions, 0, n - 1, out=input_positions)

        original_indices = np.arange(n, dtype=np.float64)
        result: np.ndarray = np.interp(input_positions, original_indices, audio).astype(np.float32)
        return result
