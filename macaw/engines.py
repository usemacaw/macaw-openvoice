"""Engine availability checks and dependency mapping."""

from __future__ import annotations

import importlib.util

# Engine name → Python package name (for importlib.util.find_spec).
ENGINE_PACKAGE: dict[str, str] = {
    "faster-whisper": "faster_whisper",
    "kokoro": "kokoro",
    "qwen3-tts": "qwen_tts",
}


def is_engine_available(engine: str) -> bool:
    """Check whether the Python package for *engine* is importable.

    Returns ``True`` for unknown engines (let the subprocess handle it).
    """
    package = ENGINE_PACKAGE.get(engine)
    if package is None:
        return True
    return importlib.util.find_spec(package) is not None
