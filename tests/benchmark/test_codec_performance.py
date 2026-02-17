"""Benchmark tests for codec performance.

Measures Opus encoding bandwidth savings and TTFB impact.
Results are printed for manual review — the test itself always passes.

Run with: .venv/bin/python -m pytest tests/benchmark/test_codec_performance.py -v -s
"""

from __future__ import annotations

import struct
import time

import pytest


@pytest.mark.slow
def test_opus_encoding_performance() -> None:
    """Measure Opus encoding: bandwidth ratio and encoding speed.

    Generates 10s of 24kHz mono PCM (sine wave) and encodes to Opus.
    Prints bandwidth ratio and real-time factor.

    Acceptance criteria (manual):
    - Opus size <= 50% of PCM size
    - Encoding TTFB degradation < 20% of audio duration
    """
    try:
        import opuslib  # noqa: F401
    except ImportError:
        pytest.skip("opuslib not installed")

    from macaw.codec.opus import OpusEncoder

    # Generate 10 seconds of 24kHz 16-bit mono sine wave (440Hz)
    sample_rate = 24000
    duration_s = 10.0
    num_samples = int(sample_rate * duration_s)
    frequency = 440.0

    import math

    pcm_samples = []
    for i in range(num_samples):
        t = i / sample_rate
        sample = int(16000 * math.sin(2 * math.pi * frequency * t))
        pcm_samples.append(struct.pack("<h", max(-32768, min(32767, sample))))
    pcm_data = b"".join(pcm_samples)

    pcm_size = len(pcm_data)
    assert pcm_size == num_samples * 2  # 16-bit

    # Encode in chunks (simulating streaming TTS output)
    encoder = OpusEncoder(sample_rate=sample_rate, bitrate=64000)
    chunk_size = 4096  # Same as TTS default chunk size

    encoded_parts: list[bytes] = []
    start_time = time.monotonic()

    for offset in range(0, len(pcm_data), chunk_size):
        chunk = pcm_data[offset : offset + chunk_size]
        encoded = encoder.encode(chunk)
        if encoded:
            encoded_parts.append(encoded)

    # Flush remaining
    flushed = encoder.flush()
    if flushed:
        encoded_parts.append(flushed)

    encoding_time = time.monotonic() - start_time
    opus_size = sum(len(p) for p in encoded_parts)

    # Print results
    bandwidth_ratio = opus_size / pcm_size
    realtime_factor = encoding_time / duration_s

    print(f"\n{'=' * 60}")
    print("Opus Encoding Benchmark Results")
    print(f"{'=' * 60}")
    print(f"PCM size:          {pcm_size:>10,} bytes ({duration_s:.1f}s @ {sample_rate}Hz)")
    print(f"Opus size:         {opus_size:>10,} bytes")
    print(f"Bandwidth ratio:   {bandwidth_ratio:>10.2%}")
    print(f"Encoding time:     {encoding_time:>10.4f}s")
    print(f"Real-time factor:  {realtime_factor:>10.4f}x")
    print(f"{'=' * 60}")
    print(f"Bandwidth target:  <= 50% -> {'PASS' if bandwidth_ratio <= 0.50 else 'FAIL'}")
    print(f"TTFB target:       < 20%  -> {'PASS' if realtime_factor < 0.20 else 'FAIL'}")
    print(f"{'=' * 60}")
