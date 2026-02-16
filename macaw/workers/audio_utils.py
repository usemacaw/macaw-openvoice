"""Shared audio conversion utilities for STT/TTS workers.

Centralizes PCM int16 <-> float32 conversion to avoid duplicating
the same logic across backends and the streaming preprocessor.
"""

from __future__ import annotations

import numpy as np

from macaw._audio_constants import PCM_INT16_SCALE, PCM_INT32_SCALE
from macaw.exceptions import AudioFormatError

# Re-export for backward compatibility (used by tests and other modules).
__all__ = ["PCM_INT16_SCALE", "PCM_INT32_SCALE", "pcm_bytes_to_float32"]


def pcm_bytes_to_float32(audio_data: bytes) -> np.ndarray:
    """Convert 16-bit PCM bytes to normalized float32 numpy array.

    Uses in-place division to avoid allocating a temporary float32 array.

    Args:
        audio_data: 16-bit PCM audio (little-endian).

    Returns:
        Normalized float32 array in [-1.0, 1.0].

    Raises:
        AudioFormatError: If the byte length is not even (16-bit PCM = 2 bytes/sample).
    """
    if len(audio_data) % 2 != 0:
        msg = "PCM 16-bit audio must have an even number of bytes"
        raise AudioFormatError(msg)

    int16_array = np.frombuffer(audio_data, dtype=np.int16)
    audio = int16_array.astype(np.float32)
    audio /= PCM_INT16_SCALE
    return audio
