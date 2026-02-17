"""Tests for macaw._dsp — window functions, STFT/iSTFT, mel filterbank, ISTFTCache."""

from __future__ import annotations

import numpy as np
from numpy.testing import assert_allclose

from macaw._audio_constants import DEFAULT_FFT_SIZE, DEFAULT_HOP_LENGTH, DEFAULT_N_MELS
from macaw._dsp import (
    ISTFTCache,
    hamming_window,
    hanning_window,
    istft,
    mel_filterbank,
    stft,
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _sine_wave(freq: float = 440.0, duration: float = 0.5, sr: int = 16000) -> np.ndarray:
    """Generate a float32 sine wave."""
    t = np.arange(int(sr * duration), dtype=np.float32) / sr
    return np.sin(2 * np.pi * freq * t).astype(np.float32)


# ---------------------------------------------------------------------------
# Window functions
# ---------------------------------------------------------------------------


class TestHanningWindow:
    def test_shape_and_dtype(self) -> None:
        w = hanning_window(512)
        assert w.shape == (512,)
        assert w.dtype == np.float32

    def test_cache_returns_same_values(self) -> None:
        w1 = hanning_window(256)
        w2 = hanning_window(256)
        assert_allclose(w1, w2)

    def test_symmetry(self) -> None:
        w = hanning_window(128)
        assert_allclose(w, w[::-1], atol=1e-7)

    def test_endpoints_near_zero(self) -> None:
        w = hanning_window(64)
        assert w[0] < 0.01
        assert w[-1] < 0.01


class TestHammingWindow:
    def test_shape_and_dtype(self) -> None:
        w = hamming_window(512)
        assert w.shape == (512,)
        assert w.dtype == np.float32

    def test_cache_returns_same_values(self) -> None:
        w1 = hamming_window(256)
        w2 = hamming_window(256)
        assert_allclose(w1, w2)

    def test_symmetry(self) -> None:
        w = hamming_window(128)
        assert_allclose(w, w[::-1], atol=1e-7)

    def test_endpoints_nonzero(self) -> None:
        """Hamming window has nonzero endpoints (~0.08), unlike Hann."""
        w = hamming_window(64)
        assert w[0] > 0.05


# ---------------------------------------------------------------------------
# STFT
# ---------------------------------------------------------------------------


class TestSTFT:
    def test_output_shape(self) -> None:
        sig = _sine_wave(duration=0.5, sr=16000)
        spec = stft(sig)
        n_freqs = DEFAULT_FFT_SIZE // 2 + 1
        assert spec.shape[0] == n_freqs
        assert spec.ndim == 2

    def test_output_dtype_complex64(self) -> None:
        sig = _sine_wave()
        spec = stft(sig)
        assert spec.dtype == np.complex64

    def test_silence_near_zero_magnitude(self) -> None:
        silence = np.zeros(4096, dtype=np.float32)
        spec = stft(silence)
        magnitudes = np.abs(spec)
        assert magnitudes.max() < 1e-7

    def test_custom_window(self) -> None:
        sig = _sine_wave()
        w = hamming_window(DEFAULT_FFT_SIZE)
        spec = stft(sig, window=w)
        assert spec.dtype == np.complex64
        assert spec.shape[0] == DEFAULT_FFT_SIZE // 2 + 1

    def test_custom_fft_size(self) -> None:
        sig = _sine_wave(duration=0.5)
        n_fft = 512
        hop = 128
        spec = stft(sig, n_fft=n_fft, hop_length=hop)
        assert spec.shape[0] == n_fft // 2 + 1


# ---------------------------------------------------------------------------
# iSTFT
# ---------------------------------------------------------------------------


class TestISTFT:
    def test_output_shape(self) -> None:
        sig = _sine_wave(duration=0.5)
        spec = stft(sig)
        rec = istft(spec)
        assert rec.ndim == 1
        assert rec.dtype == np.float32

    def test_roundtrip_error(self) -> None:
        """STFT -> iSTFT roundtrip should have max error < 1e-5."""
        sig = _sine_wave(freq=440.0, duration=0.5, sr=16000)
        spec = stft(sig)
        rec = istft(spec)
        # Lengths may differ slightly; compare the overlap
        min_len = min(len(sig), len(rec))
        assert_allclose(rec[:min_len], sig[:min_len], atol=1e-5)

    def test_roundtrip_preserves_energy(self) -> None:
        sig = _sine_wave(freq=1000.0, duration=0.3)
        spec = stft(sig)
        rec = istft(spec)
        min_len = min(len(sig), len(rec))
        orig_energy = np.sum(sig[:min_len] ** 2)
        rec_energy = np.sum(rec[:min_len] ** 2)
        assert_allclose(rec_energy, orig_energy, rtol=1e-4)


# ---------------------------------------------------------------------------
# Mel filterbank
# ---------------------------------------------------------------------------


class TestMelFilterbank:
    def test_shape(self) -> None:
        fb = mel_filterbank()
        expected_rows = DEFAULT_N_MELS
        expected_cols = DEFAULT_FFT_SIZE // 2 + 1
        assert fb.shape == (expected_rows, expected_cols)

    def test_dtype(self) -> None:
        fb = mel_filterbank()
        assert fb.dtype == np.float32

    def test_values_non_negative(self) -> None:
        fb = mel_filterbank()
        assert np.all(fb >= 0)

    def test_rows_not_all_zero(self) -> None:
        """Each mel filter should have nonzero energy."""
        fb = mel_filterbank(n_mels=40, n_fft=1024, sample_rate=16000)
        row_sums = fb.sum(axis=1)
        assert np.all(row_sums > 0)

    def test_custom_params(self) -> None:
        fb = mel_filterbank(n_mels=40, n_fft=2048, sample_rate=22050, fmin=80.0, fmax=8000.0)
        assert fb.shape == (40, 2048 // 2 + 1)
        assert fb.dtype == np.float32

    def test_cache_returns_same_values(self) -> None:
        fb1 = mel_filterbank(n_mels=64, n_fft=512, sample_rate=16000)
        fb2 = mel_filterbank(n_mels=64, n_fft=512, sample_rate=16000)
        assert_allclose(fb1, fb2)

    def test_triangular_filters_have_reasonable_peak(self) -> None:
        """Each triangular filter should peak between 0.5 and 1.0.

        Peaks can be below 1.0 when FFT frequency bins don't align
        exactly with mel center frequencies (discretization effect).
        """
        fb = mel_filterbank(n_mels=40, n_fft=1024, sample_rate=16000)
        for i in range(fb.shape[0]):
            peak = fb[i].max()
            assert 0.5 <= peak <= 1.0, f"Filter {i} peak is {peak}"


# ---------------------------------------------------------------------------
# ISTFTCache
# ---------------------------------------------------------------------------


class TestISTFTCacheCreation:
    def test_default_params(self) -> None:
        cache = ISTFTCache()
        assert cache.n_fft == DEFAULT_FFT_SIZE
        assert cache.hop_length == DEFAULT_HOP_LENGTH

    def test_custom_params(self) -> None:
        w = hamming_window(512)
        cache = ISTFTCache(n_fft=512, hop_length=128, window=w)
        assert cache.n_fft == 512
        assert cache.hop_length == 128

    def test_properties_return_correct_values(self) -> None:
        cache = ISTFTCache(n_fft=2048, hop_length=512)
        assert cache.n_fft == 2048
        assert cache.hop_length == 512


class TestISTFTCacheOLA:
    def test_ola_matches_istft(self) -> None:
        """apply_ola on a spectrogram should match scipy.istft within tolerance.

        scipy.signal.stft with boundary=True (default) pads the signal with
        nperseg//2 zeros on each side.  scipy.istft strips that padding back,
        but our OLA does not, so the output is offset by n_fft//2 samples.
        After aligning, the two should match within 1e-5.
        """
        sig = _sine_wave(freq=440.0, duration=0.5, sr=16000)
        n_fft = 1024
        hop = 256

        spec = stft(sig, n_fft=n_fft, hop_length=hop)
        cache = ISTFTCache(n_fft=n_fft, hop_length=hop)

        ola_result = cache.apply_ola(spec)
        scipy_result = istft(spec, hop_length=hop)

        # Align: our OLA output is offset by n_fft//2 due to stft boundary padding.
        offset = n_fft // 2
        ola_aligned = ola_result[offset : offset + len(scipy_result)]
        assert_allclose(ola_aligned, scipy_result, atol=1e-5)

    def test_ola_roundtrip_preserves_signal(self) -> None:
        """STFT -> apply_ola roundtrip should reconstruct the original signal.

        scipy.signal.stft pads the signal with n_fft//2 zeros on each side
        (boundary=True), so the reconstructed signal is offset by n_fft//2.
        After aligning, max error should be < 1e-5.
        """
        sig = _sine_wave(freq=1000.0, duration=0.3, sr=16000)
        n_fft = 1024
        hop = 256

        spec = stft(sig, n_fft=n_fft, hop_length=hop)
        cache = ISTFTCache(n_fft=n_fft, hop_length=hop)
        rec = cache.apply_ola(spec)

        # Align: boundary padding shifts output by n_fft//2.
        offset = n_fft // 2
        rec_aligned = rec[offset : offset + len(sig)]
        assert_allclose(rec_aligned, sig, atol=1e-5)

    def test_ola_single_frame(self) -> None:
        """Edge case: spectrogram with a single frame."""
        n_fft = 256
        hop = 64
        # Single frame: shape (n_fft//2+1, 1)
        single_frame = np.zeros((n_fft // 2 + 1, 1), dtype=np.complex64)
        # Set DC component so output is nonzero
        single_frame[0, 0] = 1.0 + 0j

        cache = ISTFTCache(n_fft=n_fft, hop_length=hop)
        result = cache.apply_ola(single_frame)

        assert result.dtype == np.float32
        assert result.ndim == 1
        assert len(result) == n_fft

    def test_ola_empty_spectrogram(self) -> None:
        """Edge case: spectrogram with 0 frames returns empty array."""
        n_fft = 1024
        hop = 256
        empty = np.zeros((n_fft // 2 + 1, 0), dtype=np.complex64)

        cache = ISTFTCache(n_fft=n_fft, hop_length=hop)
        result = cache.apply_ola(empty)

        assert result.dtype == np.float32
        assert len(result) == 0

    def test_ola_real_frames(self) -> None:
        """apply_ola accepts pre-IFFT'd real frames (n_fft, n_frames)."""
        n_fft = 512
        hop = 128
        n_frames = 10
        # Simulate time-domain frames (real, not complex).
        time_frames = (
            np.random.default_rng(42).standard_normal((n_fft, n_frames)).astype(np.float32)
        )

        cache = ISTFTCache(n_fft=n_fft, hop_length=hop)
        result = cache.apply_ola(time_frames)

        expected_length = n_fft + (n_frames - 1) * hop
        assert result.dtype == np.float32
        assert len(result) == expected_length

    def test_ola_output_dtype_float32(self) -> None:
        """Output must always be float32."""
        sig = _sine_wave(duration=0.2)
        spec = stft(sig)
        cache = ISTFTCache()
        result = cache.apply_ola(spec)
        assert result.dtype == np.float32
