"""Unit tests for StreamingPreprocessor.

Validates the frame-by-frame preprocessing adapter for streaming.
Uses real stages (ResampleStage, DCRemoveStage, GainNormalizeStage).
"""

from __future__ import annotations

import numpy as np
import pytest

from macaw.exceptions import AudioFormatError
from macaw.preprocessing.dc_remove import DCRemoveStage
from macaw.preprocessing.gain_normalize import GainNormalizeStage
from macaw.preprocessing.resample import ResampleStage
from macaw.preprocessing.streaming import StreamingPreprocessor


def _make_pcm16_frame(freq_hz: float, duration_ms: float, sample_rate: int) -> bytes:
    """Generate PCM 16-bit frame with sine wave.

    Args:
        freq_hz: Sine wave frequency in Hz.
        duration_ms: Frame duration in milliseconds.
        sample_rate: Sample rate in Hz.

    Returns:
        PCM 16-bit little-endian bytes (mono).
    """
    num_samples = int(sample_rate * duration_ms / 1000)
    t = np.arange(num_samples) / sample_rate
    audio = np.sin(2 * np.pi * freq_hz * t) * 0.5  # amplitude 0.5
    int16 = (audio * 32767).astype(np.int16)
    return int16.tobytes()


class TestStreamingPreprocessor:
    """Tests for StreamingPreprocessor."""

    def test_process_frame_pcm16_44khz_to_float32_16khz(self) -> None:
        """PCM 44.1kHz frame is converted to float32 16kHz mono."""
        stages = [
            ResampleStage(target_sample_rate=16000),
            DCRemoveStage(cutoff_hz=20),
            GainNormalizeStage(target_dbfs=-3.0),
        ]
        preprocessor = StreamingPreprocessor(stages=stages, input_sample_rate=44100)

        frame = _make_pcm16_frame(freq_hz=440.0, duration_ms=40.0, sample_rate=44100)

        result = preprocessor.process_frame(frame)

        assert result.dtype == np.float32
        # 40ms a 44100Hz = 1764 samples, resampleado para 16kHz ~= 640 samples
        expected_samples = int(16000 * 40 / 1000)
        # Tolerance of +-2 samples for resample rounding
        assert abs(len(result) - expected_samples) <= 2
        # Should have content (not silence)
        assert np.max(np.abs(result)) > 0.01

    def test_process_frame_pcm16_16khz_no_resample(self) -> None:
        """PCM 16kHz frame returns without resample, keeping ~same number of samples."""
        stages = [
            ResampleStage(target_sample_rate=16000),
            DCRemoveStage(cutoff_hz=20),
            GainNormalizeStage(target_dbfs=-3.0),
        ]
        preprocessor = StreamingPreprocessor(stages=stages, input_sample_rate=16000)

        duration_ms = 30.0
        frame = _make_pcm16_frame(freq_hz=440.0, duration_ms=duration_ms, sample_rate=16000)

        result = preprocessor.process_frame(frame)

        assert result.dtype == np.float32
        expected_samples = int(16000 * duration_ms / 1000)
        assert len(result) == expected_samples

    def test_process_frame_removes_dc_offset(self) -> None:
        """DC offset is removed by DCRemoveStage."""
        stages = [DCRemoveStage(cutoff_hz=20)]
        preprocessor = StreamingPreprocessor(stages=stages, input_sample_rate=16000)

        # Generate signal with DC offset: sine 440Hz + offset of 0.3
        num_samples = 16000  # 1 segundo para o filtro ser efetivo
        t = np.arange(num_samples) / 16000
        audio_with_dc = np.sin(2 * np.pi * 440 * t) * 0.3 + 0.3
        int16 = (audio_with_dc * 32767).astype(np.int16)
        frame = int16.tobytes()

        result = preprocessor.process_frame(frame)

        # DC offset (mean) should be significantly smaller after filtering
        assert abs(np.mean(result)) < 0.05

    def test_process_frame_normalizes_gain(self) -> None:
        """Gain is normalized to -3dBFS by GainNormalizeStage."""
        stages = [GainNormalizeStage(target_dbfs=-3.0)]
        preprocessor = StreamingPreprocessor(stages=stages, input_sample_rate=16000)

        # Signal with low amplitude (0.1 = ~-20dBFS)
        frame = _make_pcm16_frame(freq_hz=440.0, duration_ms=30.0, sample_rate=16000)
        # Reduce amplitude: recreate with lower amplitude
        num_samples = int(16000 * 30 / 1000)
        t = np.arange(num_samples) / 16000
        audio_quiet = np.sin(2 * np.pi * 440 * t) * 0.1
        int16 = (audio_quiet * 32767).astype(np.int16)
        frame = int16.tobytes()

        result = preprocessor.process_frame(frame)

        # Peak should be close to -3dBFS (~0.708)
        peak = np.max(np.abs(result))
        target_linear = 10 ** (-3.0 / 20)  # ~0.708
        assert abs(peak - target_linear) < 0.05

    def test_process_frame_empty_bytes_returns_empty(self) -> None:
        """Empty bytes return empty array."""
        stages = [
            ResampleStage(target_sample_rate=16000),
            DCRemoveStage(cutoff_hz=20),
            GainNormalizeStage(target_dbfs=-3.0),
        ]
        preprocessor = StreamingPreprocessor(stages=stages, input_sample_rate=16000)

        result = preprocessor.process_frame(b"")

        assert result.dtype == np.float32
        assert len(result) == 0

    def test_process_frame_odd_bytes_raises_error(self) -> None:
        """Odd bytes raise AudioFormatError."""
        stages = [ResampleStage(target_sample_rate=16000)]
        preprocessor = StreamingPreprocessor(stages=stages, input_sample_rate=16000)

        with pytest.raises(AudioFormatError, match="even number of bytes"):
            preprocessor.process_frame(b"\x00\x01\x02")

    def test_set_input_sample_rate(self) -> None:
        """set_input_sample_rate changes the input sample rate."""
        stages = [ResampleStage(target_sample_rate=16000)]
        preprocessor = StreamingPreprocessor(stages=stages, input_sample_rate=16000)

        assert preprocessor.input_sample_rate == 16000

        preprocessor.set_input_sample_rate(8000)

        assert preprocessor.input_sample_rate == 8000

        # 8kHz frame should be resampled to 16kHz
        frame = _make_pcm16_frame(freq_hz=440.0, duration_ms=40.0, sample_rate=8000)
        result = preprocessor.process_frame(frame)

        # 40ms a 8kHz = 320 samples, resampleado para 16kHz = 640 samples
        expected_samples = int(16000 * 40 / 1000)
        assert abs(len(result) - expected_samples) <= 2
