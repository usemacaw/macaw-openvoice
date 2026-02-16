"""FastAPI dependencies for injection of Registry, Scheduler, and WorkerManager."""

from __future__ import annotations

from typing import TYPE_CHECKING

from fastapi import Request  # noqa: TC002

from macaw.exceptions import ServiceNotConfiguredError

if TYPE_CHECKING:
    from macaw.postprocessing.pipeline import PostProcessingPipeline
    from macaw.preprocessing.pipeline import AudioPreprocessingPipeline
    from macaw.registry.registry import ModelRegistry
    from macaw.scheduler.scheduler import Scheduler
    from macaw.server.voice_store import VoiceStore
    from macaw.workers.manager import WorkerManager


def get_preprocessing_pipeline(request: Request) -> AudioPreprocessingPipeline | None:
    """Retorna o AudioPreprocessingPipeline do app state, ou None se nao configurado."""
    return request.app.state.preprocessing_pipeline  # type: ignore[no-any-return]


def get_postprocessing_pipeline(request: Request) -> PostProcessingPipeline | None:
    """Retorna o PostProcessingPipeline do app state, ou None se nao configurado."""
    return request.app.state.postprocessing_pipeline  # type: ignore[no-any-return]


def get_registry(request: Request) -> ModelRegistry:
    """Return the ModelRegistry from app state.

    Raises:
        ServiceNotConfiguredError: If registry was not configured in create_app().
    """
    registry = request.app.state.registry
    if registry is None:
        raise ServiceNotConfiguredError("Registry")
    return registry  # type: ignore[no-any-return]


def get_scheduler(request: Request) -> Scheduler:
    """Return the Scheduler from app state.

    Raises:
        ServiceNotConfiguredError: If scheduler was not configured in create_app().
    """
    scheduler = request.app.state.scheduler
    if scheduler is None:
        raise ServiceNotConfiguredError("Scheduler")
    return scheduler  # type: ignore[no-any-return]


def get_voice_store(request: Request) -> VoiceStore | None:
    """Retorna o VoiceStore do app state, ou None se nao configurado."""
    return request.app.state.voice_store  # type: ignore[no-any-return]


def require_voice_store(request: Request) -> VoiceStore:
    """Return the VoiceStore, raising InvalidRequestError if not configured."""
    from macaw.exceptions import InvalidRequestError

    store = request.app.state.voice_store
    if store is None:
        raise InvalidRequestError("VoiceStore not configured on this server.")
    return store  # type: ignore[no-any-return]


def get_worker_manager(request: Request) -> WorkerManager:
    """Return the WorkerManager from app state.

    Raises:
        ServiceNotConfiguredError: If worker_manager was not configured in create_app().
    """
    worker_manager = request.app.state.worker_manager
    if worker_manager is None:
        raise ServiceNotConfiguredError("WorkerManager")
    return worker_manager  # type: ignore[no-any-return]
