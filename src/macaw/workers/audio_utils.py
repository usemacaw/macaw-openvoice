"""Shared audio conversion utilities for STT/TTS workers.

Centralizes PCM int16 <-> float32 conversion to avoid duplicating
the same logic across backends and the streaming preprocessor.
"""

from __future__ import annotations

import numpy as np

from macaw.exceptions import AudioFormatError


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
        msg = "Audio PCM 16-bit deve ter numero par de bytes"
        raise AudioFormatError(msg)

    int16_array = np.frombuffer(audio_data, dtype=np.int16)
    audio = int16_array.astype(np.float32)
    audio /= 32768.0
    return audio
