"""Engine availability checks and dependency mapping."""

from __future__ import annotations

import importlib.util

# Engine name → Python package name (for importlib.util.find_spec).
ENGINE_PACKAGE: dict[str, str] = {
    "faster-whisper": "faster_whisper",
    "kokoro": "kokoro",
    "qwen3-tts": "qwen_tts",
    "qwen3-asr": "qwen_asr",
    "chatterbox": "chatterbox",
}


def is_engine_available(engine: str, *, python_package: str | None = None) -> bool:
    """Check whether the Python package for *engine* is importable.

    When *python_package* is set (external engine), checks importability of
    that package instead of looking up ``ENGINE_PACKAGE``.

    Returns ``True`` for unknown built-in engines (let the subprocess handle it).
    """
    if python_package:
        return importlib.util.find_spec(python_package) is not None

    package = ENGINE_PACKAGE.get(engine)
    if package is None:
        return True
    return importlib.util.find_spec(package) is not None
