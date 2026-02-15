"""Shared audio utilities for TTS backends.

Pure functions for audio format conversion, shared across all TTS backends.
"""

from __future__ import annotations

import numpy as np

from macaw.workers.audio_utils import PCM_INT16_MAX

# Size of audio chunks returned by synthesize (bytes).
# 4096 bytes = 2048 PCM 16-bit samples = ~85ms at 24kHz.
CHUNK_SIZE_BYTES = 4096


def float32_to_pcm16_bytes(audio_array: np.ndarray) -> bytes:
    """Convert normalized float32 array [-1, 1] to 16-bit PCM bytes."""
    int16_data = (audio_array * PCM_INT16_MAX).clip(-32768, 32767).astype(np.int16)
    return int16_data.tobytes()
