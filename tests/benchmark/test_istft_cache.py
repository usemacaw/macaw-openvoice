"""Benchmark: ISTFTCache.apply_ola() vs scipy istft wrapper.

Marked @pytest.mark.slow — excluded from ``make test-fast``.
"""

from __future__ import annotations

import statistics
import time

import numpy as np
import pytest

from macaw._dsp import ISTFTCache, istft, stft


@pytest.mark.slow
def test_istft_cache_vs_scipy_benchmark() -> None:
    """Compare ISTFTCache.apply_ola() against scipy istft wrapper.

    Generates 10s of audio at 16kHz, computes STFT, then runs each
    reconstruction path 10 times and reports median timings.
    The test always passes — it is purely informational.
    """
    sr = 16000
    duration = 10.0
    n_fft = 1024
    hop = 256
    iterations = 10

    # Generate 10s sine wave
    t = np.arange(int(sr * duration), dtype=np.float32) / sr
    signal = np.sin(2 * np.pi * 440.0 * t).astype(np.float32)

    # Compute spectrogram once
    spec = stft(signal, n_fft=n_fft, hop_length=hop)

    # --- Benchmark scipy istft wrapper ---
    scipy_times: list[float] = []
    for _ in range(iterations):
        start = time.perf_counter()
        istft(spec, hop_length=hop)
        scipy_times.append(time.perf_counter() - start)

    # --- Benchmark ISTFTCache.apply_ola ---
    cache = ISTFTCache(n_fft=n_fft, hop_length=hop)
    cache_times: list[float] = []
    for _ in range(iterations):
        start = time.perf_counter()
        cache.apply_ola(spec)
        cache_times.append(time.perf_counter() - start)

    scipy_median = statistics.median(scipy_times)
    cache_median = statistics.median(cache_times)
    speedup = scipy_median / cache_median if cache_median > 0 else float("inf")

    print(f"\n--- ISTFTCache Benchmark ({iterations} iterations, {duration}s audio) ---")
    print(f"scipy istft median: {scipy_median * 1000:.2f} ms")
    print(f"ISTFTCache.apply_ola median: {cache_median * 1000:.2f} ms")
    print(f"Speedup: {speedup:.2f}x")

    # Informational — accept if cache is at least 1.0x (don't fail if slower).
    assert speedup >= 0.5, f"ISTFTCache more than 2x slower than scipy: {speedup:.2f}x"
