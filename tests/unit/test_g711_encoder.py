"""Tests for G.711 mu-law and A-law codec encoders.

Validates:
- Codec name property
- Silence encoding (zero input produces ~128 output center)
- 8-bit output (uint8)
- Resample from 24kHz to 8kHz (output length is 1/3 of input)
- Empty input returns empty bytes
- Flush is always empty (no internal buffer)
- Companding range (output values stay in 0-255)
- Both MuLawEncoder and ALawEncoder
"""

from __future__ import annotations

import numpy as np

from macaw.codec.g711 import (
    G711_SAMPLE_RATE,
    ALawEncoder,
    MuLawEncoder,
    _a_law_compress,
    _mu_law_compress,
)

# ---------------------------------------------------------------------------
# MuLawEncoder
# ---------------------------------------------------------------------------


class TestMuLawEncoderProperties:
    def test_codec_name(self) -> None:
        encoder = MuLawEncoder(sample_rate=24000)
        assert encoder.codec_name == "mulaw"

    def test_flush_returns_empty(self) -> None:
        encoder = MuLawEncoder(sample_rate=24000)
        assert encoder.flush() == b""

    def test_flush_after_encode_returns_empty(self) -> None:
        encoder = MuLawEncoder(sample_rate=24000)
        pcm = _silence_pcm(160)
        encoder.encode(pcm)
        assert encoder.flush() == b""

    def test_encode_empty_returns_empty(self) -> None:
        encoder = MuLawEncoder(sample_rate=24000)
        assert encoder.encode(b"") == b""


class TestMuLawEncoderOutput:
    def test_output_is_8bit(self) -> None:
        encoder = MuLawEncoder(sample_rate=8000)
        pcm = _sine_pcm(160, sample_rate=8000)
        result = encoder.encode(pcm)
        arr = np.frombuffer(result, dtype=np.uint8)
        assert arr.dtype == np.uint8

    def test_silence_encodes_near_center(self) -> None:
        """Silence (zeros) should encode near the center value ~127-128."""
        encoder = MuLawEncoder(sample_rate=8000)
        pcm = _silence_pcm(160)
        result = encoder.encode(pcm)
        arr = np.frombuffer(result, dtype=np.uint8)
        # All values should be close to 128 (center of uint8 range)
        assert np.all(np.abs(arr.astype(np.int16) - 128) < 2)

    def test_output_length_no_resample(self) -> None:
        """At 8kHz input, output length equals input samples (1 byte per sample)."""
        n_samples = 160
        encoder = MuLawEncoder(sample_rate=8000)
        pcm = _sine_pcm(n_samples, sample_rate=8000)
        result = encoder.encode(pcm)
        assert len(result) == n_samples

    def test_resample_24k_to_8k(self) -> None:
        """At 24kHz input, output should be ~1/3 the input samples."""
        n_samples = 480  # 20ms at 24kHz
        encoder = MuLawEncoder(sample_rate=24000)
        pcm = _sine_pcm(n_samples, sample_rate=24000)
        result = encoder.encode(pcm)
        expected_samples = n_samples // 3  # 24000/8000 = 3
        assert abs(len(result) - expected_samples) <= 1

    def test_values_in_uint8_range(self) -> None:
        encoder = MuLawEncoder(sample_rate=8000)
        pcm = _sine_pcm(320, sample_rate=8000, amplitude=0.9)
        result = encoder.encode(pcm)
        arr = np.frombuffer(result, dtype=np.uint8)
        assert arr.min() >= 0
        assert arr.max() <= 255


# ---------------------------------------------------------------------------
# ALawEncoder
# ---------------------------------------------------------------------------


class TestALawEncoderProperties:
    def test_codec_name(self) -> None:
        encoder = ALawEncoder(sample_rate=24000)
        assert encoder.codec_name == "alaw"

    def test_flush_returns_empty(self) -> None:
        encoder = ALawEncoder(sample_rate=24000)
        assert encoder.flush() == b""

    def test_encode_empty_returns_empty(self) -> None:
        encoder = ALawEncoder(sample_rate=24000)
        assert encoder.encode(b"") == b""


class TestALawEncoderOutput:
    def test_output_is_8bit(self) -> None:
        encoder = ALawEncoder(sample_rate=8000)
        pcm = _sine_pcm(160, sample_rate=8000)
        result = encoder.encode(pcm)
        arr = np.frombuffer(result, dtype=np.uint8)
        assert arr.dtype == np.uint8

    def test_silence_encodes_near_center(self) -> None:
        encoder = ALawEncoder(sample_rate=8000)
        pcm = _silence_pcm(160)
        result = encoder.encode(pcm)
        arr = np.frombuffer(result, dtype=np.uint8)
        assert np.all(np.abs(arr.astype(np.int16) - 128) < 2)

    def test_output_length_no_resample(self) -> None:
        n_samples = 160
        encoder = ALawEncoder(sample_rate=8000)
        pcm = _sine_pcm(n_samples, sample_rate=8000)
        result = encoder.encode(pcm)
        assert len(result) == n_samples

    def test_resample_24k_to_8k(self) -> None:
        n_samples = 480
        encoder = ALawEncoder(sample_rate=24000)
        pcm = _sine_pcm(n_samples, sample_rate=24000)
        result = encoder.encode(pcm)
        expected_samples = n_samples // 3
        assert abs(len(result) - expected_samples) <= 1

    def test_values_in_uint8_range(self) -> None:
        encoder = ALawEncoder(sample_rate=8000)
        pcm = _sine_pcm(320, sample_rate=8000, amplitude=0.9)
        result = encoder.encode(pcm)
        arr = np.frombuffer(result, dtype=np.uint8)
        assert arr.min() >= 0
        assert arr.max() <= 255


# ---------------------------------------------------------------------------
# Compression functions (pure math)
# ---------------------------------------------------------------------------


class TestMuLawCompression:
    def test_zero_input(self) -> None:
        audio = np.zeros(10, dtype=np.float32)
        result = _mu_law_compress(audio)
        assert result.dtype == np.uint8
        # Center value for zero input
        assert np.all(np.abs(result.astype(np.int16) - 128) < 2)

    def test_full_scale_positive(self) -> None:
        audio = np.ones(10, dtype=np.float32)
        result = _mu_law_compress(audio)
        assert np.all(result == 255)

    def test_full_scale_negative(self) -> None:
        audio = -np.ones(10, dtype=np.float32)
        result = _mu_law_compress(audio)
        assert np.all(result == 0)

    def test_monotonic_positive(self) -> None:
        """Increasing input should produce increasing compressed values."""
        audio = np.linspace(0, 1, 50, dtype=np.float32)
        result = _mu_law_compress(audio)
        # Compressed values should be non-decreasing
        assert np.all(np.diff(result.astype(np.int16)) >= 0)


class TestALawCompression:
    def test_zero_input(self) -> None:
        audio = np.zeros(10, dtype=np.float32)
        result = _a_law_compress(audio)
        assert result.dtype == np.uint8
        assert np.all(np.abs(result.astype(np.int16) - 128) < 2)

    def test_full_scale_positive(self) -> None:
        audio = np.ones(10, dtype=np.float32)
        result = _a_law_compress(audio)
        assert np.all(result == 255)

    def test_full_scale_negative(self) -> None:
        audio = -np.ones(10, dtype=np.float32)
        result = _a_law_compress(audio)
        assert np.all(result == 0)


# ---------------------------------------------------------------------------
# G711 constant
# ---------------------------------------------------------------------------


class TestG711Constants:
    def test_sample_rate(self) -> None:
        assert G711_SAMPLE_RATE == 8000


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _silence_pcm(n_samples: int) -> bytes:
    """Generate silent PCM (all zeros) as 16-bit bytes."""
    return b"\x00\x00" * n_samples


def _sine_pcm(
    n_samples: int,
    sample_rate: int = 8000,
    freq: float = 440.0,
    amplitude: float = 0.5,
) -> bytes:
    """Generate a sine wave as 16-bit PCM bytes."""
    t = np.arange(n_samples, dtype=np.float32) / sample_rate
    signal = (amplitude * np.sin(2 * np.pi * freq * t) * 32767).astype(np.int16)
    return signal.tobytes()
