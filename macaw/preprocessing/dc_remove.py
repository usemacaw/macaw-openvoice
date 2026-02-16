"""DCRemoveStage — High-pass filter to remove DC offset from audio.

Uses a 2nd-order Butterworth filter as HPF. Removes hardware DC offset
common in telephony without affecting the speech band (80Hz-8kHz).
"""

from __future__ import annotations

import numpy as np
from scipy.signal import butter, sosfilt, sosfilt_zi

from macaw.preprocessing.stages import AudioStage


class DCRemoveStage(AudioStage):
    """Remove DC offset via 2nd-order Butterworth high-pass filter.

    Filter coefficients are lazy-computed and cached per sample rate.
    Recomputed only when the sample rate changes.

    Args:
        cutoff_hz: HPF cutoff frequency in Hz (default: 20).
    """

    def __init__(self, cutoff_hz: int = 20) -> None:
        self._cutoff_hz = cutoff_hz
        self._cached_sample_rate: int | None = None
        self._cached_sos: np.ndarray | None = None
        self._zi: np.ndarray | None = None

    @property
    def name(self) -> str:
        """Identifier name for the stage."""
        return "dc_remove"

    def _get_sos(self, sample_rate: int) -> np.ndarray:
        """Return filter SOS coefficients, recomputing if necessary.

        When the sample rate changes, filter state is also reset since
        the SOS coefficients change.

        Args:
            sample_rate: Current audio sample rate in Hz.

        Returns:
            Array of Butterworth filter SOS coefficients.
        """
        if self._cached_sample_rate != sample_rate or self._cached_sos is None:
            self._cached_sos = butter(
                2,
                self._cutoff_hz,
                btype="highpass",
                fs=sample_rate,
                output="sos",
            )
            self._cached_sample_rate = sample_rate
            self._zi = None
        return self._cached_sos

    def reset(self) -> None:
        """Reset filter state for new streaming session.

        Clears the internal filter state (zi) so the next call to
        process() starts with fresh initial conditions. Does not
        clear cached SOS coefficients.
        """
        self._zi = None

    def process(self, audio: np.ndarray, sample_rate: int) -> tuple[np.ndarray, int]:
        """Apply HPF to remove DC offset from audio.

        Maintains filter state across calls for seamless streaming.
        On the first call (or after reset), initial conditions are
        computed via sosfilt_zi() to minimize transient artifacts.

        Args:
            audio: Numpy float32 array with audio samples (mono).
            sample_rate: Current audio sample rate in Hz.

        Returns:
            Tuple (filtered float32 audio, unchanged sample rate).
        """
        if len(audio) == 0:
            return audio, sample_rate

        sos = self._get_sos(sample_rate)

        if self._zi is None:
            self._zi = sosfilt_zi(sos) * audio[0]

        filtered, self._zi = sosfilt(sos, audio, zi=self._zi)

        if filtered.dtype != np.float32:
            filtered = filtered.astype(np.float32)

        return filtered, sample_rate
