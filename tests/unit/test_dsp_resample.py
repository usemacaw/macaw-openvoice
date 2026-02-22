"""Tests for resample_output DSP primitive.

Validates:
- Same rate returns input unchanged
- 24kHz to 8kHz produces ~1/3 samples
- Output is float32
- 16kHz to 48kHz upsampling
- Empty input
"""

from __future__ import annotations

import numpy as np

from macaw._dsp import resample_output


class TestResampleOutput:
    def test_same_rate_returns_input(self) -> None:
        audio = np.random.randn(160).astype(np.float32)
        result = resample_output(audio, 16000, 16000)
        np.testing.assert_array_equal(result, audio)

    def test_same_rate_returns_same_object(self) -> None:
        """When rates match, should return the exact same array (no copy)."""
        audio = np.random.randn(160).astype(np.float32)
        result = resample_output(audio, 16000, 16000)
        assert result is audio

    def test_downsample_24k_to_8k(self) -> None:
        n_samples = 480  # 20ms at 24kHz
        audio = np.random.randn(n_samples).astype(np.float32)
        result = resample_output(audio, 24000, 8000)
        expected = n_samples // 3  # 24000/8000 = 3
        assert abs(len(result) - expected) <= 1

    def test_upsample_16k_to_48k(self) -> None:
        n_samples = 160  # 10ms at 16kHz
        audio = np.random.randn(n_samples).astype(np.float32)
        result = resample_output(audio, 16000, 48000)
        expected = n_samples * 3  # 48000/16000 = 3
        assert abs(len(result) - expected) <= 1

    def test_output_dtype_is_float32(self) -> None:
        audio = np.random.randn(160).astype(np.float32)
        result = resample_output(audio, 24000, 8000)
        assert result.dtype == np.float32

    def test_empty_input(self) -> None:
        audio = np.array([], dtype=np.float32)
        result = resample_output(audio, 24000, 8000)
        assert len(result) == 0
        assert result.dtype == np.float32
