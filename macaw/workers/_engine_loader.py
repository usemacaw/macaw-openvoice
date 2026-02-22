"""Dynamic loader for external engine backends.

Loads a backend class from an external Python package specified via the
``python_package`` field in macaw.yaml.  Scans the module for a concrete
subclass of the given ABC (STTBackend or TTSBackend) and returns an instance.
"""

from __future__ import annotations

import importlib
import inspect
from typing import TypeVar

from macaw.exceptions import ModelLoadError
from macaw.logging import get_logger

logger = get_logger("worker.engine_loader")

_T = TypeVar("_T")


def load_external_backend(python_package: str, backend_type: type[_T]) -> _T:
    """Import a module and find a concrete subclass of *backend_type*.

    Args:
        python_package: Dotted Python module path (e.g. ``my_company.engines.whisper``).
        backend_type: The ABC to search for (``STTBackend`` or ``TTSBackend``).

    Returns:
        A fresh (unloaded) instance of the discovered backend class.

    Raises:
        ModelLoadError: If the module cannot be imported, or exactly one
            concrete subclass of *backend_type* is not found.
    """
    try:
        module = importlib.import_module(python_package)
    except ImportError as exc:
        raise ModelLoadError(
            python_package,
            f"Cannot import external engine package '{python_package}': {exc}",
        ) from exc

    candidates: list[type[_T]] = []
    for _name, obj in inspect.getmembers(module, inspect.isclass):
        if (
            issubclass(obj, backend_type)
            and obj is not backend_type
            and not inspect.isabstract(obj)
        ):
            candidates.append(obj)

    if len(candidates) == 0:
        raise ModelLoadError(
            python_package,
            f"Module '{python_package}' does not contain a concrete subclass "
            f"of {backend_type.__name__}.",
        )

    if len(candidates) > 1:
        names = ", ".join(c.__name__ for c in candidates)
        raise ModelLoadError(
            python_package,
            f"Module '{python_package}' contains multiple concrete subclasses "
            f"of {backend_type.__name__}: {names}. "
            f"Expose exactly one per module.",
        )

    backend_cls = candidates[0]
    logger.info(
        "external_backend_loaded",
        package=python_package,
        backend_class=backend_cls.__name__,
    )
    return backend_cls()
