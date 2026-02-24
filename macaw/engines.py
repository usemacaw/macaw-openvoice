"""Engine availability checks and dependency mapping."""

from __future__ import annotations

import importlib.util
import re
import subprocess

# Python module name pattern — guards against command injection in
# subprocess "-c import {package}" calls.
_SAFE_MODULE_RE = re.compile(r"^[a-zA-Z_][a-zA-Z0-9_]*(\.[a-zA-Z_][a-zA-Z0-9_]*)*$")

# Engine name → Python package name (for importlib.util.find_spec).
ENGINE_PACKAGE: dict[str, str] = {
    "faster-whisper": "faster_whisper",
    "kokoro": "kokoro",
    "qwen3-tts": "qwen_tts",
    "qwen3-asr": "qwen_asr",
    "chatterbox": "chatterbox",
}

# Engine name → pyproject.toml extra name (for venv provisioning).
ENGINE_EXTRAS: dict[str, str] = {
    "faster-whisper": "faster-whisper",
    "kokoro": "kokoro",
    "qwen3-tts": "qwen3-tts",
    "qwen3-asr": "qwen3-asr",
    "chatterbox": "chatterbox",
}


def is_engine_available(
    engine: str,
    *,
    python_package: str | None = None,
    venv_python: str | None = None,
) -> bool:
    """Check whether the Python package for *engine* is importable.

    When *venv_python* is set, checks importability inside the venv via
    subprocess instead of the parent process.

    When *python_package* is set (external engine), checks importability of
    that package instead of looking up ``ENGINE_PACKAGE``.

    Returns ``True`` for unknown built-in engines (let the subprocess handle it).
    """
    if venv_python is not None:
        package = python_package or ENGINE_PACKAGE.get(engine)
        if package is None:
            return True
        # Validate package name before interpolating into subprocess command
        # to prevent command injection.
        if not _SAFE_MODULE_RE.match(package):
            return False
        try:
            result = subprocess.run(
                [venv_python, "-c", f"import {package}"],
                capture_output=True,
                timeout=10,
            )
            return result.returncode == 0
        except (subprocess.TimeoutExpired, FileNotFoundError, OSError):
            return False

    if python_package:
        return importlib.util.find_spec(python_package) is not None

    package = ENGINE_PACKAGE.get(engine)
    if package is None:
        return True
    return importlib.util.find_spec(package) is not None
