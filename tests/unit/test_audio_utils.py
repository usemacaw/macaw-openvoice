"""Tests for macaw.workers.audio_utils.

Validates the centralized PCM int16 -> float32 conversion function.
"""

from __future__ import annotations

import numpy as np
import pytest

from macaw.exceptions import AudioFormatError
from macaw.workers.audio_utils import pcm_bytes_to_float32


class TestPcmBytesToFloat32:
    def test_valid_pcm_16bit_converts_correctly(self) -> None:
        """PCM int16 bytes are converted to normalized float32 [-1.0, 1.0]."""
        # Arrange: 0 and 32767 (max positive)
        audio = b"\x00\x00\xff\x7f"

        # Act
        result = pcm_bytes_to_float32(audio)

        # Assert
        assert result.dtype == np.float32
        assert len(result) == 2
        assert abs(result[0]) < 0.01
        assert abs(result[1] - 1.0) < 0.01

    def test_negative_sample_converts_correctly(self) -> None:
        """Negative int16 value maps to negative float32."""
        # Arrange: -32768 (min negative) = 0x0080 little-endian
        audio = b"\x00\x80"

        # Act
        result = pcm_bytes_to_float32(audio)

        # Assert
        assert result.dtype == np.float32
        assert len(result) == 1
        assert result[0] == pytest.approx(-1.0, abs=0.001)

    def test_roundtrip_preserves_signal(self) -> None:
        """float32 -> int16 -> bytes -> pcm_bytes_to_float32 roundtrip."""
        # Arrange
        original = np.array([0.0, 0.5, -0.5, 1.0, -1.0], dtype=np.float32)
        int16_data = (original * 32768.0).clip(-32768, 32767).astype(np.int16)
        pcm_bytes = int16_data.tobytes()

        # Act
        result = pcm_bytes_to_float32(pcm_bytes)

        # Assert
        np.testing.assert_allclose(result, original, atol=1e-4)

    def test_odd_bytes_raises_audio_format_error(self) -> None:
        """Odd byte count raises AudioFormatError."""
        with pytest.raises(AudioFormatError, match="numero par"):
            pcm_bytes_to_float32(b"\x00\x01\x02")

    def test_empty_bytes_returns_empty_array(self) -> None:
        """Empty input returns empty float32 array."""
        result = pcm_bytes_to_float32(b"")
        assert result.dtype == np.float32
        assert len(result) == 0

    def test_output_range_bounded(self) -> None:
        """Output values are in [-1.0, 1.0] for any valid int16 input."""
        # Arrange: full range of int16
        all_values = np.array([-32768, -16384, 0, 16384, 32767], dtype=np.int16)
        pcm_bytes = all_values.tobytes()

        # Act
        result = pcm_bytes_to_float32(pcm_bytes)

        # Assert
        assert np.all(result >= -1.0)
        assert np.all(result <= 1.0)

    def test_in_place_division_produces_correct_result(self) -> None:
        """Verify in-place division matches explicit division."""
        # Arrange
        samples = np.array([100, -200, 32767, -32768, 0], dtype=np.int16)
        pcm_bytes = samples.tobytes()
        expected = samples.astype(np.float32) / 32768.0

        # Act
        result = pcm_bytes_to_float32(pcm_bytes)

        # Assert
        np.testing.assert_array_equal(result, expected)
