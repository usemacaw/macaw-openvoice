"""Constantes compartilhadas do server."""

from __future__ import annotations

MAX_FILE_SIZE_BYTES = 25 * 1024 * 1024  # 25MB

ALLOWED_AUDIO_CONTENT_TYPES = frozenset(
    {
        "audio/wav",
        "audio/x-wav",
        "audio/wave",
        "audio/mpeg",
        "audio/mp3",
        "audio/flac",
        "audio/ogg",
        "audio/webm",
        "audio/x-flac",
        "application/octet-stream",  # fallback generico
    }
)
