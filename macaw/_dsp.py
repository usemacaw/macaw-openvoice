"""Centralized DSP primitives for the Macaw runtime.

Pure signal processing functions: window generation, STFT/iSTFT, mel filterbanks,
and streaming iSTFT via pre-computed overlap-add buffers.
Zero imports from macaw.workers, macaw.server, macaw.scheduler — this module
sits at the bottom of the dependency graph alongside _audio_constants.
"""

from __future__ import annotations

from functools import lru_cache

import numpy as np
from scipy.signal import istft as _scipy_istft
from scipy.signal import stft as _scipy_stft

from macaw._audio_constants import DEFAULT_FFT_SIZE, DEFAULT_HOP_LENGTH, DEFAULT_N_MELS

__all__ = [
    "ISTFTCache",
    "hamming_window",
    "hanning_window",
    "istft",
    "mel_filterbank",
    "resample_output",
    "stft",
]

# ---------------------------------------------------------------------------
# Window functions
# ---------------------------------------------------------------------------


@lru_cache(maxsize=8)
def _hanning_window_cached(size: int) -> tuple[float, ...]:
    return tuple(np.hanning(size).astype(np.float32).tolist())


def hanning_window(size: int) -> np.ndarray:
    """Hann window of given size (float32, cached)."""
    return np.array(_hanning_window_cached(size), dtype=np.float32)


@lru_cache(maxsize=8)
def _hamming_window_cached(size: int) -> tuple[float, ...]:
    return tuple(np.hamming(size).astype(np.float32).tolist())


def hamming_window(size: int) -> np.ndarray:
    """Hamming window of given size (float32, cached)."""
    return np.array(_hamming_window_cached(size), dtype=np.float32)


# ---------------------------------------------------------------------------
# STFT / iSTFT
# ---------------------------------------------------------------------------


def stft(
    signal: np.ndarray,
    n_fft: int = DEFAULT_FFT_SIZE,
    hop_length: int = DEFAULT_HOP_LENGTH,
    window: np.ndarray | None = None,
) -> np.ndarray:
    """Short-Time Fourier Transform.

    Args:
        signal: 1-D float32 time-domain signal.
        n_fft: FFT window size.
        hop_length: Hop between successive frames.
        window: Analysis window. Defaults to Hann window of size *n_fft*.

    Returns:
        Complex64 spectrogram of shape ``(n_fft // 2 + 1, n_frames)``.
    """
    if window is None:
        window = hanning_window(n_fft)

    _freqs, _times, zxx = _scipy_stft(  # type: ignore[call-overload]
        signal,
        nperseg=n_fft,
        noverlap=n_fft - hop_length,
        window=window,
    )
    return np.asarray(zxx, dtype=np.complex64)


def istft(
    spectrogram: np.ndarray,
    hop_length: int = DEFAULT_HOP_LENGTH,
    window: np.ndarray | None = None,
) -> np.ndarray:
    """Inverse Short-Time Fourier Transform.

    Args:
        spectrogram: Complex spectrogram from :func:`stft`.
        hop_length: Hop length used during the forward transform.
        window: Synthesis window. Must match the analysis window.

    Returns:
        Float32 time-domain signal.
    """
    n_fft = (spectrogram.shape[0] - 1) * 2
    if window is None:
        window = hanning_window(n_fft)

    _times, reconstructed = _scipy_istft(  # type: ignore[call-overload]
        spectrogram,
        nperseg=n_fft,
        noverlap=n_fft - hop_length,
        window=window,
    )
    return np.asarray(reconstructed, dtype=np.float32)


# ---------------------------------------------------------------------------
# Streaming iSTFT (pre-computed OLA)
# ---------------------------------------------------------------------------


class ISTFTCache:
    """Pre-computed overlap-add buffers for streaming iSTFT.

    Caches the synthesis window normalization buffer and OLA positions
    computed once at construction time. Useful for vocoder engines that
    process spectrogram frames incrementally rather than calling full iSTFT.

    Args:
        n_fft: FFT size.
        hop_length: Hop between successive frames.
        window: Synthesis window. Defaults to Hann window.
    """

    def __init__(
        self,
        n_fft: int = DEFAULT_FFT_SIZE,
        hop_length: int = DEFAULT_HOP_LENGTH,
        window: np.ndarray | None = None,
    ) -> None:
        self._n_fft = n_fft
        self._hop_length = hop_length
        self._window = window if window is not None else hanning_window(n_fft)
        # Pre-compute the squared window for NOLA normalization.
        self._window_sq = self._window**2

    @property
    def n_fft(self) -> int:
        """FFT size used by this cache."""
        return self._n_fft

    @property
    def hop_length(self) -> int:
        """Hop length used by this cache."""
        return self._hop_length

    def apply_ola(self, frames: np.ndarray) -> np.ndarray:
        """Reconstruct time-domain signal from STFT frames via overlap-add.

        This is equivalent to ``scipy.signal.istft`` but uses pre-computed
        window buffers, avoiding redundant allocation on each call.

        When *frames* is a complex spectrogram produced by :func:`stft` (which
        uses ``scipy.signal.stft`` internally with ``scaling='spectrum'``),
        the IRFFT result is rescaled by ``sum(window)`` to undo the forward
        scaling — exactly as ``scipy.signal.istft`` does.

        Args:
            frames: Complex spectrogram of shape ``(n_fft//2+1, n_frames)``
                    OR real IFFT'd frames of shape ``(n_fft, n_frames)``.
                    An empty 2-D array (zero frames) returns an empty signal.

        Returns:
            Float32 time-domain signal.
        """
        if frames.ndim == 2 and frames.shape[1] == 0:
            return np.array([], dtype=np.float32)

        if np.iscomplexobj(frames):
            # IRFFT undoes the FFT but not the spectrum scaling applied by
            # scipy.signal.stft (which divides by sum(window)).  Multiply
            # by sum(window) to restore the original amplitude — this is
            # identical to what scipy.signal.istft does internally.
            time_frames = np.fft.irfft(frames, n=self._n_fft, axis=0)
            time_frames *= float(np.sum(self._window))
        else:
            time_frames = frames

        if time_frames.ndim == 1:
            time_frames = time_frames[:, np.newaxis]

        n_fft = self._n_fft
        hop = self._hop_length
        n_frames = time_frames.shape[1]

        output_length = n_fft + (n_frames - 1) * hop
        output = np.zeros(output_length, dtype=np.float64)
        norm = np.zeros(output_length, dtype=np.float64)

        for i in range(n_frames):
            start = i * hop
            end = start + n_fft
            output[start:end] += time_frames[:, i] * self._window
            norm[start:end] += self._window_sq

        # Normalize, avoiding division by zero.
        nonzero = norm > 1e-10
        output[nonzero] /= norm[nonzero]

        return output.astype(np.float32)


# ---------------------------------------------------------------------------
# Mel filterbank
# ---------------------------------------------------------------------------


def _hz_to_mel(hz: float) -> float:
    """Convert frequency in Hz to mel scale (HTK formula)."""
    return float(2595.0 * np.log10(1.0 + hz / 700.0))


def _mel_to_hz(mel: float) -> float:
    """Convert mel scale value back to Hz."""
    return float(700.0 * (10.0 ** (mel / 2595.0) - 1.0))


@lru_cache(maxsize=8)
def _mel_filterbank_cached(
    n_mels: int,
    n_fft: int,
    sample_rate: int,
    fmin: float,
    fmax: float,
) -> tuple[tuple[float, ...], ...]:
    n_freqs = n_fft // 2 + 1
    fft_freqs = np.linspace(0, sample_rate / 2, n_freqs)

    mel_min = _hz_to_mel(fmin)
    mel_max = _hz_to_mel(fmax)
    mel_points = np.linspace(mel_min, mel_max, n_mels + 2)
    hz_points = np.array([_mel_to_hz(float(m)) for m in mel_points])

    filterbank = np.zeros((n_mels, n_freqs), dtype=np.float32)
    for i in range(n_mels):
        lower = hz_points[i]
        center = hz_points[i + 1]
        upper = hz_points[i + 2]

        # Rising slope
        up_mask = (fft_freqs >= lower) & (fft_freqs <= center)
        if center > lower:
            filterbank[i, up_mask] = (fft_freqs[up_mask] - lower) / (center - lower)

        # Falling slope
        down_mask = (fft_freqs > center) & (fft_freqs <= upper)
        if upper > center:
            filterbank[i, down_mask] = (upper - fft_freqs[down_mask]) / (upper - center)

    return tuple(tuple(float(v) for v in row) for row in filterbank)


def mel_filterbank(
    n_mels: int = DEFAULT_N_MELS,
    n_fft: int = DEFAULT_FFT_SIZE,
    sample_rate: int = 16000,
    fmin: float = 0.0,
    fmax: float | None = None,
) -> np.ndarray:
    """Mel-spaced triangular filterbank matrix.

    Args:
        n_mels: Number of mel bands.
        n_fft: FFT size (determines frequency resolution).
        sample_rate: Audio sample rate in Hz.
        fmin: Lowest frequency (Hz) for the filterbank.
        fmax: Highest frequency (Hz). Defaults to ``sample_rate / 2``.

    Returns:
        Float32 array of shape ``(n_mels, n_fft // 2 + 1)``.
    """
    if fmax is None:
        fmax = sample_rate / 2.0

    cached = _mel_filterbank_cached(n_mels, n_fft, sample_rate, fmin, fmax)
    return np.array(cached, dtype=np.float32)


# ---------------------------------------------------------------------------
# Resample
# ---------------------------------------------------------------------------


def resample_output(audio: np.ndarray, from_rate: int, to_rate: int) -> np.ndarray:
    """Resample audio from one sample rate to another.

    Uses ``scipy.signal.resample_poly`` for high-quality polyphase resampling.
    Returns the input unchanged when ``from_rate == to_rate``.

    Args:
        audio: 1-D float32 audio signal.
        from_rate: Source sample rate in Hz.
        to_rate: Target sample rate in Hz.

    Returns:
        Float32 resampled audio.
    """
    if from_rate == to_rate:
        return audio

    from math import gcd

    from scipy.signal import resample_poly as _resample_poly

    divisor = gcd(from_rate, to_rate)
    up = to_rate // divisor
    down = from_rate // divisor

    resampled = _resample_poly(audio, up, down)
    return np.asarray(resampled, dtype=np.float32)
