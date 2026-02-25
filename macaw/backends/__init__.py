"""Backend venv isolation for engine dependency management.

Provides per-engine Python virtual environments to prevent dependency
conflicts between ML engines (e.g., torchaudio vs pyannote.audio).
"""

from macaw.backends.venv_manager import (
    VenvManager,
    resolve_python_for_engine,
)

__all__ = ["VenvManager", "resolve_python_for_engine"]
