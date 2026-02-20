"""Tests for EnergyPreFilter.

Validates that the energy pre-filter correctly classifies frames as
silence or non-silence based on RMS (dBFS) and spectral flatness.
Tests sensitivity mapping and edge cases.
"""

from __future__ import annotations

import numpy as np

from macaw._types import VADSensitivity
from macaw.vad.energy import EnergyPreFilter


def _make_sine(
    frequency: float = 440.0,
    sample_rate: int = 16000,
    duration: float = 0.064,
    amplitude: float = 0.5,
) -> np.ndarray:
    """Generate float32 sine signal (simulates tonal speech)."""
    t = np.arange(int(sample_rate * duration)) / sample_rate
    signal = amplitude * np.sin(2 * np.pi * frequency * t)
    return signal.astype(np.float32)


def _make_white_noise(
    sample_rate: int = 16000,
    duration: float = 0.064,
    amplitude: float = 0.001,
) -> np.ndarray:
    """Generate float32 white noise with controlled amplitude."""
    rng = np.random.default_rng(seed=42)
    n_samples = int(sample_rate * duration)
    noise = amplitude * rng.standard_normal(n_samples)
    return noise.astype(np.float32)


class TestEnergyPreFilter:
    def test_silence_frame_classified_as_silence(self) -> None:
        """Frame of zeros is classified as silence."""
        # Arrange
        pre_filter = EnergyPreFilter(sensitivity=VADSensitivity.NORMAL)
        frame = np.zeros(1024, dtype=np.float32)

        # Act
        result = pre_filter.is_silence(frame)

        # Assert
        assert result is True

    def test_speech_frame_classified_as_non_silence(self) -> None:
        """440Hz sine wave with high amplitude is not silence."""
        # Arrange
        pre_filter = EnergyPreFilter(sensitivity=VADSensitivity.NORMAL)
        frame = _make_sine(frequency=440.0, amplitude=0.5)

        # Act
        result = pre_filter.is_silence(frame)

        # Assert
        assert result is False

    def test_low_energy_noise_classified_as_silence(self) -> None:
        """White noise with very low amplitude is silence."""
        # Arrange
        pre_filter = EnergyPreFilter(sensitivity=VADSensitivity.NORMAL)
        frame = _make_white_noise(amplitude=0.0001)

        # Act
        result = pre_filter.is_silence(frame)

        # Assert
        assert result is True

    def test_high_energy_noise_classified_as_non_silence(self) -> None:
        """White noise with high amplitude is not silence (high RMS)."""
        # Arrange
        pre_filter = EnergyPreFilter(sensitivity=VADSensitivity.NORMAL)
        frame = _make_white_noise(amplitude=0.5)

        # Act
        result = pre_filter.is_silence(frame)

        # Assert
        assert result is False

    def test_sensitivity_high_detects_quiet_speech(self) -> None:
        """Frame with low amplitude (whisper) is not silence in HIGH."""
        # Arrange -- low amplitude between -50dBFS and -40dBFS
        # amplitude=0.005 -> RMS ~0.0035 -> ~-49dBFS (acima de -50)
        pre_filter = EnergyPreFilter(sensitivity=VADSensitivity.HIGH)
        frame = _make_sine(frequency=440.0, amplitude=0.005)

        # Act
        result = pre_filter.is_silence(frame)

        # Assert -- HIGH threshold (-50dBFS) does not classify as silence
        assert result is False

    def test_tonal_signal_not_classified_as_silence_despite_low_energy(self) -> None:
        """Sine wave with low energy is not silence -- spectral flatness is low (tonal).

        The pre-filter requires BOTH criteria: low energy AND high flatness.
        A tonal signal (sine wave) has flatness ~0, so even with RMS below
        the threshold, it is not classified as silence. This avoids false positives
        in whispered speech.
        """
        # Arrange -- sine wave with low energy (RMS ~-49dBFS, below -30dBFS)
        pre_filter = EnergyPreFilter(sensitivity=VADSensitivity.LOW)
        frame = _make_sine(frequency=440.0, amplitude=0.005)

        # Act
        result = pre_filter.is_silence(frame)

        # Assert -- not silence because sine wave flatness is ~0 (tonal)
        assert result is False

    def test_sensitivity_low_classifies_weak_noise_as_silence(self) -> None:
        """White noise with amplitude between -40 and -30 dBFS: LOW says silence, NORMAL does not."""
        # Arrange -- amplitude that generates RMS ~-35dBFS
        # amplitude=0.01 -> RMS ~0.01 -> 20*log10(0.01) = -40dBFS
        # amplitude=0.02 -> RMS ~0.02 -> 20*log10(0.02) = -34dBFS
        frame = _make_white_noise(amplitude=0.015)

        pre_filter_low = EnergyPreFilter(sensitivity=VADSensitivity.LOW)
        pre_filter_normal = EnergyPreFilter(sensitivity=VADSensitivity.NORMAL)

        # Act
        result_low = pre_filter_low.is_silence(frame)
        result_normal = pre_filter_normal.is_silence(frame)

        # Assert -- LOW (-30dBFS) classifies as silence, NORMAL (-40dBFS) does not
        assert result_low is True
        assert result_normal is False

    def test_empty_frame_is_silence(self) -> None:
        """Empty array is classified as silence."""
        # Arrange
        pre_filter = EnergyPreFilter(sensitivity=VADSensitivity.NORMAL)
        frame = np.array([], dtype=np.float32)

        # Act
        result = pre_filter.is_silence(frame)

        # Assert
        assert result is True

    def test_single_sample_frame_is_silence(self) -> None:
        """Frame with 1 sample is classified as silence."""
        # Arrange
        pre_filter = EnergyPreFilter(sensitivity=VADSensitivity.NORMAL)
        frame = np.array([0.5], dtype=np.float32)

        # Act
        result = pre_filter.is_silence(frame)

        # Assert
        assert result is True

    def test_energy_threshold_property(self) -> None:
        """energy_threshold_dbfs property returns correct value per sensitivity."""
        # Arrange & Act & Assert
        assert EnergyPreFilter(VADSensitivity.HIGH).energy_threshold_dbfs == -50.0
        assert EnergyPreFilter(VADSensitivity.NORMAL).energy_threshold_dbfs == -40.0
        assert EnergyPreFilter(VADSensitivity.LOW).energy_threshold_dbfs == -30.0

    def test_energy_threshold_override(self) -> None:
        """Threshold override ignores the sensitivity preset."""
        pf = EnergyPreFilter(VADSensitivity.NORMAL, energy_threshold_dbfs_override=-45.0)
        assert pf.energy_threshold_dbfs == -45.0

    def test_energy_threshold_override_none_uses_preset(self) -> None:
        """None override uses the normal preset."""
        pf = EnergyPreFilter(VADSensitivity.HIGH, energy_threshold_dbfs_override=None)
        assert pf.energy_threshold_dbfs == -50.0
