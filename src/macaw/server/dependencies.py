"""FastAPI dependencies para injecao do Registry e Scheduler."""

from __future__ import annotations

from typing import TYPE_CHECKING

from fastapi import Request  # noqa: TC002

if TYPE_CHECKING:
    from macaw.postprocessing.pipeline import PostProcessingPipeline
    from macaw.preprocessing.pipeline import AudioPreprocessingPipeline
    from macaw.registry.registry import ModelRegistry
    from macaw.scheduler.scheduler import Scheduler
    from macaw.workers.manager import WorkerManager


def get_preprocessing_pipeline(request: Request) -> AudioPreprocessingPipeline | None:
    """Retorna o AudioPreprocessingPipeline do app state, ou None se nao configurado."""
    return request.app.state.preprocessing_pipeline  # type: ignore[no-any-return]


def get_postprocessing_pipeline(request: Request) -> PostProcessingPipeline | None:
    """Retorna o PostProcessingPipeline do app state, ou None se nao configurado."""
    return request.app.state.postprocessing_pipeline  # type: ignore[no-any-return]


def get_registry(request: Request) -> ModelRegistry:
    """Retorna o ModelRegistry do app state.

    Raises:
        RuntimeError: Se registry nao foi configurado em create_app().
    """
    registry = request.app.state.registry
    if registry is None:
        raise RuntimeError("Registry nao configurado. Passe registry= em create_app().")
    return registry  # type: ignore[no-any-return]


def get_scheduler(request: Request) -> Scheduler:
    """Retorna o Scheduler do app state.

    Raises:
        RuntimeError: Se scheduler nao foi configurado em create_app().
    """
    scheduler = request.app.state.scheduler
    if scheduler is None:
        raise RuntimeError("Scheduler nao configurado. Passe scheduler= em create_app().")
    return scheduler  # type: ignore[no-any-return]


def get_worker_manager(request: Request) -> WorkerManager:
    """Retorna o WorkerManager do app state.

    Raises:
        RuntimeError: Se worker_manager nao foi configurado em create_app().
    """
    worker_manager = request.app.state.worker_manager
    if worker_manager is None:
        raise RuntimeError("WorkerManager nao configurado. Passe worker_manager= em create_app().")
    return worker_manager  # type: ignore[no-any-return]
