"""Tests for ResampleStage.

Validates audio resampling between sample rates, stereo->mono conversion,
float32 type preservation, and edge cases (empty audio, unchanged sample rate).
"""

from __future__ import annotations

import numpy as np

from macaw.preprocessing.resample import ResampleStage


def _make_sine(
    sample_rate: int,
    duration: float = 0.1,
    frequency: float = 440.0,
    amplitude: float = 0.5,
) -> np.ndarray:
    """Generate float32 sine wave signal for tests."""
    n_samples = int(sample_rate * duration)
    t = np.arange(n_samples, dtype=np.float32) / sample_rate
    return (amplitude * np.sin(2 * np.pi * frequency * t)).astype(np.float32)


class TestResampleStage:
    def test_resample_44khz_to_16khz(self) -> None:
        """Audio at 44.1kHz is resampled to 16kHz with proportional length."""
        # Arrange
        audio = _make_sine(sample_rate=44100, duration=1.0)
        stage = ResampleStage(target_sample_rate=16000)

        # Act
        result, result_sr = stage.process(audio, 44100)

        # Assert
        assert result_sr == 16000
        expected_length = 16000  # 1s * 16000
        assert len(result) == expected_length

    def test_resample_48khz_to_16khz(self) -> None:
        """Audio at 48kHz is resampled to 16kHz with proportional length."""
        # Arrange
        audio = _make_sine(sample_rate=48000, duration=1.0)
        stage = ResampleStage(target_sample_rate=16000)

        # Act
        result, result_sr = stage.process(audio, 48000)

        # Assert
        assert result_sr == 16000
        expected_length = 16000  # 1s * 16000
        assert len(result) == expected_length

    def test_resample_8khz_to_16khz(self) -> None:
        """Audio at 8kHz is upsampled to 16kHz with proportional length."""
        # Arrange
        audio = _make_sine(sample_rate=8000, duration=1.0)
        stage = ResampleStage(target_sample_rate=16000)

        # Act
        result, result_sr = stage.process(audio, 8000)

        # Assert
        assert result_sr == 16000
        expected_length = 16000  # 1s * 16000
        assert len(result) == expected_length

    def test_resample_16khz_skip(self) -> None:
        """Audio already at 16kHz is returned unchanged (same object)."""
        # Arrange
        audio = _make_sine(sample_rate=16000, duration=0.1)
        stage = ResampleStage(target_sample_rate=16000)

        # Act
        result, result_sr = stage.process(audio, 16000)

        # Assert
        assert result_sr == 16000
        assert result is audio  # Same object, no copy

    def test_resample_mono_from_high_sample_rate(self) -> None:
        """Mono audio at 44.1kHz is correctly resampled to 16kHz."""
        # Arrange — pipeline contract guarantees mono input
        audio = _make_sine(sample_rate=44100, duration=0.5, frequency=440.0, amplitude=0.8)
        stage = ResampleStage(target_sample_rate=16000)

        # Act
        result, result_sr = stage.process(audio, 44100)

        # Assert
        assert result_sr == 16000
        assert result.ndim == 1
        expected_length = int(0.5 * 16000)
        assert len(result) == expected_length

    def test_resample_preserves_float32(self) -> None:
        """Output is always float32, regardless of input sample rate."""
        # Arrange
        stage = ResampleStage(target_sample_rate=16000)

        for input_sr in [8000, 16000, 44100, 48000]:
            audio = _make_sine(sample_rate=input_sr, duration=0.1)

            # Act
            result, _ = stage.process(audio, input_sr)

            # Assert
            assert result.dtype == np.float32, (
                f"Expected float32 for input_sr={input_sr}, got {result.dtype}"
            )

    def test_resample_empty_audio(self) -> None:
        """Empty array is returned unchanged without error."""
        # Arrange
        audio = np.array([], dtype=np.float32)
        stage = ResampleStage(target_sample_rate=16000)

        # Act
        result, result_sr = stage.process(audio, 44100)

        # Assert
        assert len(result) == 0
        assert result_sr == 44100  # Original sample rate preserved

    def test_resample_name_property(self) -> None:
        """name property returns 'resample'."""
        # Arrange & Act
        stage = ResampleStage()

        # Assert
        assert stage.name == "resample"

    def test_resample_custom_target_sample_rate(self) -> None:
        """ResampleStage accepts custom target sample rate."""
        # Arrange
        audio = _make_sine(sample_rate=44100, duration=1.0)
        stage = ResampleStage(target_sample_rate=8000)

        # Act
        result, result_sr = stage.process(audio, 44100)

        # Assert
        assert result_sr == 8000
        assert len(result) == 8000

    def test_resample_signal_energy_preserved(self) -> None:
        """Signal energy is approximately preserved after resample."""
        # Arrange
        audio = _make_sine(sample_rate=48000, duration=1.0, frequency=200.0)
        stage = ResampleStage(target_sample_rate=16000)

        # Act
        result, _ = stage.process(audio, 48000)

        # Assert
        # RMS of original and resampled signal should be close
        rms_original = np.sqrt(np.mean(audio**2))
        rms_resampled = np.sqrt(np.mean(result**2))
        # Tolerancia de 5% na energia
        assert abs(rms_original - rms_resampled) / rms_original < 0.05

    def test_resample_preserves_mono_shape(self) -> None:
        """Mono audio keeps ndim=1 after resampling."""
        # Arrange — pipeline contract guarantees mono input
        audio = np.ones(4410, dtype=np.float32) * 0.5
        stage = ResampleStage(target_sample_rate=16000)

        # Act
        result, result_sr = stage.process(audio, 44100)

        # Assert
        assert result.ndim == 1
        assert result_sr == 16000
