"""TTS service facade — shared model/worker resolution for speech + realtime routes."""

from __future__ import annotations

from typing import TYPE_CHECKING

from macaw._types import ModelType
from macaw.exceptions import ModelNotFoundError, WorkerUnavailableError

if TYPE_CHECKING:
    from macaw.config.manifest import ModelManifest
    from macaw.registry.registry import ModelRegistry
    from macaw.workers.manager import WorkerHandle, WorkerManager


def resolve_tts_resources(
    registry: ModelRegistry,
    worker_manager: WorkerManager,
    model: str,
) -> tuple[ModelManifest, WorkerHandle, str]:
    """Resolve TTS model, validate manifest, find a ready worker.

    Returns (manifest, worker_handle, worker_address).

    Raises:
        ModelNotFoundError: If model is not found or is not a TTS model.
        WorkerUnavailableError: If no ready worker is available for the model.
    """
    manifest = registry.get_manifest(model)
    if manifest.model_type != ModelType.TTS:
        raise ModelNotFoundError(model)

    worker = worker_manager.get_ready_worker(model)
    if worker is None:
        raise WorkerUnavailableError(model)

    from macaw.config.settings import get_settings

    worker_host = get_settings().worker.worker_host
    worker_address = f"{worker_host}:{worker.port}"
    return manifest, worker, worker_address


def find_default_tts_model(registry: ModelRegistry) -> str | None:
    """Find the first registered TTS model name, or None if none available."""
    for m in registry.list_models():
        if m.model_type == ModelType.TTS:
            return m.name
    return None
