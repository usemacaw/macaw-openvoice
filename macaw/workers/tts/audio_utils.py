"""Shared audio utilities for TTS backends.

Pure functions for audio format conversion, shared across all TTS backends.

``CHUNK_SIZE_BYTES`` is resolved lazily from ``TTSSettings`` via PEP 562.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

from macaw._audio_constants import PCM_INT16_MAX, PCM_INT16_MIN, PCM_INT16_SCALE


def get_chunk_size_bytes() -> int:
    """Return configured TTS chunk size from settings."""
    from macaw.config.settings import get_settings

    return get_settings().tts.chunk_size_bytes


def float32_to_pcm16_bytes(audio_array: np.ndarray) -> bytes:
    """Convert normalized float32 array [-1, 1] to 16-bit PCM bytes."""
    int16_data = (
        (audio_array * PCM_INT16_SCALE).clip(PCM_INT16_MIN, PCM_INT16_MAX).astype(np.int16)
    )
    return int16_data.tobytes()


# --- PEP 562 lazy resolution ---
def __getattr__(name: str) -> int:
    if name == "CHUNK_SIZE_BYTES":
        return get_chunk_size_bytes()
    msg = f"module {__name__!r} has no attribute {name!r}"
    raise AttributeError(msg)


if TYPE_CHECKING:
    CHUNK_SIZE_BYTES: int
