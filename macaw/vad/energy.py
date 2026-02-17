"""Energy pre-filter for Voice Activity Detection.

Energy (RMS) and spectral flatness pre-filter to reduce calls to Silero VAD.
Frames with low energy and flat spectrum (indicating white noise or silence)
are classified as silence without invoking the neural model.

Estimated false positive reduction: 60-70% in noisy environments.
Cost: ~0.1ms/frame.
"""

from __future__ import annotations

import numpy as np

from macaw._audio_constants import DEFAULT_SPECTRAL_FLATNESS_THRESHOLD
from macaw._types import VADSensitivity

# Energy thresholds (dBFS) by sensitivity level.
# More negative values = more sensitive (detects weaker sounds).
_ENERGY_THRESHOLDS: dict[VADSensitivity, float] = {
    VADSensitivity.HIGH: -50.0,  # Very sensitive (whisper, banking)
    VADSensitivity.NORMAL: -40.0,  # Normal conversation (default)
    VADSensitivity.LOW: -30.0,  # Noisy environment, call center
}

_SPECTRAL_FLATNESS_THRESHOLD = DEFAULT_SPECTRAL_FLATNESS_THRESHOLD

# Minimum samples for FFT to produce meaningful result.
_MIN_SAMPLES_FOR_FFT = 2

# Epsilon to avoid log(0) and division by zero.
_EPSILON = 1e-10


def _spectral_flatness(magnitude: np.ndarray) -> float:
    """Compute spectral flatness (Wiener entropy) from magnitude spectrum.

    Spectral flatness = geometric_mean / arithmetic_mean.
    Values close to 1.0 indicate flat spectrum (noise/silence).
    Values close to 0.0 indicate tonal content (speech).

    Args:
        magnitude: Magnitude spectrum (must be clamped above epsilon).

    Returns:
        Spectral flatness in [0.0, 1.0].
    """
    clamped = np.maximum(magnitude, _EPSILON)
    arithmetic_mean = np.mean(clamped)
    geometric_mean = np.exp(np.mean(np.log(clamped)))
    return float(geometric_mean / arithmetic_mean)


class EnergyPreFilter:
    """Energy-based pre-filter to reduce calls to Silero VAD.

    Computes RMS and spectral flatness for the frame. If RMS in dBFS < threshold
    AND spectral flatness > 0.8 (indicates white noise/silence), classify
    as silence without calling Silero VAD.

    Cost: ~0.1ms/frame.
    """

    def __init__(
        self,
        sensitivity: VADSensitivity = VADSensitivity.NORMAL,
        *,
        energy_threshold_dbfs_override: float | None = None,
    ) -> None:
        self._sensitivity = sensitivity
        self._energy_threshold_dbfs = (
            energy_threshold_dbfs_override
            if energy_threshold_dbfs_override is not None
            else _ENERGY_THRESHOLDS[sensitivity]
        )

    @property
    def energy_threshold_dbfs(self) -> float:
        """Current threshold in dBFS."""
        return self._energy_threshold_dbfs

    def is_silence(self, frame: np.ndarray) -> bool:
        """Check whether the frame is silence based on energy and spectral flatness.

        Args:
            frame: Float32 mono numpy array, any size
                   (typically 64ms = 1024 samples at 16kHz).

        Returns:
            True if the frame is classified as silence
            (low RMS AND high spectral flatness).
        """
        if len(frame) < _MIN_SAMPLES_FOR_FFT:
            return True

        # np.dot returns scalar (sum of squares) without allocating frame**2.
        rms = np.sqrt(np.dot(frame, frame) / len(frame))
        rms_dbfs = 20.0 * np.log10(rms + _EPSILON)

        if rms_dbfs >= self._energy_threshold_dbfs:
            return False

        magnitude = np.abs(np.fft.rfft(frame))
        flatness = _spectral_flatness(magnitude)

        return bool(flatness > _SPECTRAL_FLATNESS_THRESHOLD)
