"""Tests for GainNormalizeStage.

Validates amplitude normalization, clipping protection,
silence handling, and empty audio.
"""

from __future__ import annotations

import numpy as np
import pytest

from macaw.preprocessing.gain_normalize import GainNormalizeStage


def peak_dbfs(audio: np.ndarray) -> float:
    """Calculates peak level in dBFS."""
    peak = np.max(np.abs(audio))
    if peak < 1e-10:
        return -float("inf")
    return 20 * np.log10(peak)


def make_sine(
    frequency: float = 440.0,
    sample_rate: int = 16000,
    duration: float = 0.1,
    amplitude: float = 0.5,
) -> np.ndarray:
    """Creates a float32 sine wave with specified amplitude."""
    t = np.arange(int(sample_rate * duration), dtype=np.float32) / sample_rate
    return (amplitude * np.sin(2 * np.pi * frequency * t)).astype(np.float32)


class TestGainNormalizeStage:
    def test_gain_normalize_low_amplitude(self) -> None:
        """Audio with peak at -20dBFS normalized to -3dBFS."""
        # Arrange
        amplitude_linear = 10 ** (-20.0 / 20)  # ~0.1
        audio = make_sine(amplitude=amplitude_linear)
        stage = GainNormalizeStage(target_dbfs=-3.0)

        # Act
        result, _sr = stage.process(audio, 16000)

        # Assert
        result_dbfs = peak_dbfs(result)
        assert result_dbfs == pytest.approx(-3.0, abs=0.5)

    def test_gain_normalize_high_amplitude(self) -> None:
        """Audio with peak at -1dBFS normalized to -3dBFS."""
        # Arrange
        amplitude_linear = 10 ** (-1.0 / 20)  # ~0.891
        audio = make_sine(amplitude=amplitude_linear)
        stage = GainNormalizeStage(target_dbfs=-3.0)

        # Act
        result, _sr = stage.process(audio, 16000)

        # Assert
        result_dbfs = peak_dbfs(result)
        assert result_dbfs == pytest.approx(-3.0, abs=0.5)

    def test_gain_normalize_silence(self) -> None:
        """All-zeros audio returned unchanged, no division by zero."""
        # Arrange
        audio = np.zeros(1600, dtype=np.float32)
        stage = GainNormalizeStage(target_dbfs=-3.0)

        # Act
        result, sr = stage.process(audio, 16000)

        # Assert
        np.testing.assert_array_equal(result, audio)
        assert sr == 16000

    def test_gain_normalize_near_zero(self) -> None:
        """Audio with peak below 1e-10 returned unchanged."""
        # Arrange
        audio = np.full(1600, 1e-12, dtype=np.float32)
        stage = GainNormalizeStage(target_dbfs=-3.0)

        # Act
        result, sr = stage.process(audio, 16000)

        # Assert
        np.testing.assert_array_equal(result, audio)
        assert sr == 16000

    def test_gain_normalize_clipping_protection(self) -> None:
        """Audio that would exceed 0dBFS after gain is clipped to [-1.0, 1.0]."""
        # Arrange: audio with peak at -40dBFS, target at -0.1dBFS
        # Required gain would be enormous, resulting in clipping
        amplitude_linear = 10 ** (-40.0 / 20)  # ~0.01
        audio = make_sine(amplitude=amplitude_linear)
        # Very high target forces large gain
        stage = GainNormalizeStage(target_dbfs=-0.1)

        # Act
        result, _sr = stage.process(audio, 16000)

        # Assert: no sample exceeds [-1.0, 1.0]
        assert np.all(result >= -1.0)
        assert np.all(result <= 1.0)

    def test_gain_normalize_preserves_float32(self) -> None:
        """Output is always float32."""
        # Arrange
        audio = make_sine(amplitude=0.5)
        stage = GainNormalizeStage(target_dbfs=-3.0)

        # Act
        result, _sr = stage.process(audio, 16000)

        # Assert
        assert result.dtype == np.float32

    def test_gain_normalize_preserves_sample_rate(self) -> None:
        """Sample rate is not modified by the stage."""
        # Arrange
        audio = make_sine(amplitude=0.5)
        stage = GainNormalizeStage(target_dbfs=-3.0)

        # Act
        for sr in [8000, 16000, 44100, 48000]:
            _result, result_sr = stage.process(audio, sr)

            # Assert
            assert result_sr == sr

    def test_gain_normalize_empty_audio(self) -> None:
        """Empty array returned unchanged."""
        # Arrange
        audio = np.array([], dtype=np.float32)
        stage = GainNormalizeStage(target_dbfs=-3.0)

        # Act
        result, sr = stage.process(audio, 16000)

        # Assert
        assert len(result) == 0
        assert sr == 16000

    def test_gain_normalize_name_property(self) -> None:
        """Name property returns 'gain_normalize'."""
        # Arrange & Act
        stage = GainNormalizeStage()

        # Assert
        assert stage.name == "gain_normalize"

    def test_gain_normalize_custom_target(self) -> None:
        """Custom target (-6.0 dBFS) works correctly."""
        # Arrange
        amplitude_linear = 10 ** (-20.0 / 20)  # ~0.1
        audio = make_sine(amplitude=amplitude_linear)
        stage = GainNormalizeStage(target_dbfs=-6.0)

        # Act
        result, _sr = stage.process(audio, 16000)

        # Assert
        result_dbfs = peak_dbfs(result)
        assert result_dbfs == pytest.approx(-6.0, abs=0.5)
