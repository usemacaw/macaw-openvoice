"""Freeverb reverb effect — algorithmic room simulation.

Implements the classic Freeverb algorithm:
- 8 parallel comb filters with lowpass damping
- 4 series allpass filters
- Configurable room size, damping, and wet/dry mix

Reference: https://ccrma.stanford.edu/~jos/pasp/Freeverb.html
"""

from __future__ import annotations

import numpy as np

from macaw.audio_effects.interface import AudioEffect

# Standard Freeverb delay lengths (tuned for 44100 Hz).
# Scaled to actual sample rate at construction time.
_COMB_DELAYS_44100 = (1116, 1188, 1277, 1356, 1422, 1491, 1557, 1617)
_ALLPASS_DELAYS_44100 = (556, 441, 341, 225)

_REFERENCE_RATE = 44100
_ALLPASS_FEEDBACK = 0.5


class _CombFilter:
    """Feedback comb filter with lowpass damping."""

    __slots__ = (
        "_buffer",
        "_buffer_size",
        "_damp1",
        "_damp2",
        "_feedback",
        "_filterstore",
        "_index",
    )

    def __init__(self, delay_samples: int, feedback: float, damping: float) -> None:
        self._buffer = np.zeros(delay_samples, dtype=np.float64)
        self._buffer_size = delay_samples
        self._index = 0
        self._filterstore = 0.0
        self._feedback = feedback
        self._damp1 = damping
        self._damp2 = 1.0 - damping

    def process_sample(self, inp: float) -> float:
        output: float = float(self._buffer[self._index])
        self._filterstore = output * self._damp2 + self._filterstore * self._damp1
        self._buffer[self._index] = inp + self._filterstore * self._feedback
        self._index += 1
        if self._index >= self._buffer_size:
            self._index = 0
        return output

    def reset(self) -> None:
        self._buffer[:] = 0.0
        self._index = 0
        self._filterstore = 0.0


class _AllPassFilter:
    """Schroeder allpass filter."""

    __slots__ = ("_buffer", "_buffer_size", "_index")

    def __init__(self, delay_samples: int) -> None:
        self._buffer = np.zeros(delay_samples, dtype=np.float64)
        self._buffer_size = delay_samples
        self._index = 0

    def process_sample(self, inp: float) -> float:
        buffered: float = float(self._buffer[self._index])
        output = buffered - inp
        self._buffer[self._index] = inp + buffered * _ALLPASS_FEEDBACK
        self._index += 1
        if self._index >= self._buffer_size:
            self._index = 0
        return output

    def reset(self) -> None:
        self._buffer[:] = 0.0
        self._index = 0


def _scale_delay(delay_44100: int, sample_rate: int) -> int:
    """Scale a delay length from 44100 Hz reference to target sample rate."""
    return max(1, int(delay_44100 * sample_rate / _REFERENCE_RATE))


class ReverbEffect(AudioEffect):
    """Freeverb algorithmic reverb.

    Args:
        room_size: Room size (0.0 = small, 1.0 = large). Controls feedback amount.
        damping: High-frequency damping (0.0 = bright, 1.0 = dark).
        wet_dry_mix: Wet/dry balance (0.0 = fully dry, 1.0 = fully wet).
        sample_rate: Target sample rate for delay line scaling.
    """

    def __init__(
        self,
        room_size: float = 0.5,
        damping: float = 0.5,
        wet_dry_mix: float = 0.3,
        sample_rate: int = 24000,
    ) -> None:
        self._room_size = room_size
        self._damping = damping
        self._wet_dry_mix = wet_dry_mix

        feedback = 0.28 + room_size * 0.7

        self._combs = [
            _CombFilter(_scale_delay(d, sample_rate), feedback, damping)
            for d in _COMB_DELAYS_44100
        ]
        self._allpasses = [
            _AllPassFilter(_scale_delay(d, sample_rate)) for d in _ALLPASS_DELAYS_44100
        ]

    @property
    def name(self) -> str:
        return "reverb"

    def process(self, audio: np.ndarray, sample_rate: int) -> np.ndarray:
        """Apply reverb to audio. Returns array of same length."""
        n = len(audio)
        if n == 0:
            return audio

        wet = self._wet_dry_mix
        dry = 1.0 - wet

        output = np.empty(n, dtype=np.float64)
        combs = self._combs
        allpasses = self._allpasses

        for i in range(n):
            inp = float(audio[i])

            # Parallel comb filters (summed)
            comb_sum = 0.0
            for comb in combs:
                comb_sum += comb.process_sample(inp)

            # Series allpass filters
            out = comb_sum
            for ap in allpasses:
                out = ap.process_sample(out)

            # Mix wet + dry
            output[i] = out * wet + inp * dry

        # Clip to [-1, 1] to prevent overflow
        np.clip(output, -1.0, 1.0, out=output)
        return output.astype(np.float32)

    def reset(self) -> None:
        """Clear all delay line buffers."""
        for comb in self._combs:
            comb.reset()
        for ap in self._allpasses:
            ap.reset()
