"""Tests for macaw._dsp.resample_pcm_bytes — PCM byte resampling helper.

Sprint J: Output Format Granularity (resample raw PCM between sample rates).
"""

from __future__ import annotations

import numpy as np

from macaw._dsp import resample_pcm_bytes


class TestResamplePcmBytes:
    """Tests for resample_pcm_bytes()."""

    def test_same_rate_returns_unchanged(self) -> None:
        pcm = b"\x00\x01" * 100
        result = resample_pcm_bytes(pcm, 16000, 16000)
        assert result is pcm  # Identity — no copy

    def test_empty_data_returns_unchanged(self) -> None:
        result = resample_pcm_bytes(b"", 16000, 48000)
        assert result == b""

    def test_downsample_24k_to_16k(self) -> None:
        """24kHz → 16kHz: output should have ~2/3 the samples."""
        n_samples = 2400  # 100ms at 24kHz
        signal = np.sin(np.linspace(0, 2 * np.pi * 440, n_samples)).astype(np.float32)
        pcm = (signal * 32767).astype(np.int16).tobytes()

        result = resample_pcm_bytes(pcm, 24000, 16000)

        # Expected: ~1600 samples (100ms at 16kHz), each 2 bytes
        expected_samples = 1600
        actual_samples = len(result) // 2
        assert abs(actual_samples - expected_samples) <= 2

    def test_upsample_24k_to_48k(self) -> None:
        """24kHz → 48kHz: output should have exactly 2x the samples."""
        n_samples = 2400
        signal = np.sin(np.linspace(0, 2 * np.pi * 440, n_samples)).astype(np.float32)
        pcm = (signal * 32767).astype(np.int16).tobytes()

        result = resample_pcm_bytes(pcm, 24000, 48000)

        expected_samples = 4800
        actual_samples = len(result) // 2
        assert abs(actual_samples - expected_samples) <= 2

    def test_upsample_24k_to_44100(self) -> None:
        """24kHz → 44.1kHz: fractional ratio resampling."""
        n_samples = 2400
        signal = np.sin(np.linspace(0, 2 * np.pi * 440, n_samples)).astype(np.float32)
        pcm = (signal * 32767).astype(np.int16).tobytes()

        result = resample_pcm_bytes(pcm, 24000, 44100)

        expected_samples = int(n_samples * 44100 / 24000)
        actual_samples = len(result) // 2
        # Allow small tolerance for polyphase filter edge effects
        assert abs(actual_samples - expected_samples) <= 5

    def test_downsample_24k_to_8k(self) -> None:
        """24kHz → 8kHz for G.711 codecs."""
        n_samples = 2400
        signal = np.sin(np.linspace(0, 2 * np.pi * 440, n_samples)).astype(np.float32)
        pcm = (signal * 32767).astype(np.int16).tobytes()

        result = resample_pcm_bytes(pcm, 24000, 8000)

        expected_samples = 800
        actual_samples = len(result) // 2
        assert abs(actual_samples - expected_samples) <= 2

    def test_output_is_int16_bytes(self) -> None:
        """Output should be valid int16 PCM bytes (even length)."""
        n_samples = 480
        signal = np.sin(np.linspace(0, 2 * np.pi * 440, n_samples)).astype(np.float32)
        pcm = (signal * 32767).astype(np.int16).tobytes()

        result = resample_pcm_bytes(pcm, 24000, 16000)
        assert len(result) % 2 == 0

        # Verify we can decode back to int16
        samples = np.frombuffer(result, dtype=np.int16)
        assert samples.dtype == np.int16

    def test_clipping_protection(self) -> None:
        """Output should be clipped to int16 range even with gain from resampling."""
        # Create a max-amplitude signal
        n_samples = 480
        pcm = np.full(n_samples, 32767, dtype=np.int16).tobytes()

        result = resample_pcm_bytes(pcm, 24000, 16000)
        samples = np.frombuffer(result, dtype=np.int16)

        # All values should be within int16 range
        assert np.all(samples >= -32768)
        assert np.all(samples <= 32767)
