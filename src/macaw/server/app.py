"""FastAPI application factory para o Macaw OpenVoice."""

from __future__ import annotations

from typing import TYPE_CHECKING

from fastapi import FastAPI

import macaw
from macaw.server.error_handlers import register_error_handlers
from macaw.server.routes import health, realtime, speech, transcriptions, translations

if TYPE_CHECKING:
    from macaw.postprocessing.pipeline import PostProcessingPipeline
    from macaw.preprocessing.pipeline import AudioPreprocessingPipeline
    from macaw.registry.registry import ModelRegistry
    from macaw.scheduler.scheduler import Scheduler
    from macaw.workers.manager import WorkerManager


def create_app(
    registry: ModelRegistry | None = None,
    scheduler: Scheduler | None = None,
    preprocessing_pipeline: AudioPreprocessingPipeline | None = None,
    postprocessing_pipeline: PostProcessingPipeline | None = None,
    worker_manager: WorkerManager | None = None,
    cors_origins: list[str] | None = None,
) -> FastAPI:
    """Cria a aplicacao FastAPI.

    Args:
        registry: Model Registry (opcional, None apenas para testes do health endpoint).
        scheduler: Scheduler (opcional, None apenas para testes do health endpoint).
        preprocessing_pipeline: Pipeline de preprocessamento de audio (opcional).
        postprocessing_pipeline: Pipeline de pos-processamento de texto (opcional).
        worker_manager: Worker Manager para TTS (opcional).
        cors_origins: Lista de CORS origins permitidos (opcional).

    Returns:
        FastAPI application configurada.
    """
    app = FastAPI(
        title="Macaw OpenVoice",
        version=macaw.__version__,
        description="Runtime unificado de voz (STT + TTS) com API OpenAI-compatible",
    )

    app.state.registry = registry
    app.state.scheduler = scheduler
    app.state.preprocessing_pipeline = preprocessing_pipeline
    app.state.postprocessing_pipeline = postprocessing_pipeline
    app.state.worker_manager = worker_manager

    if cors_origins:
        from fastapi.middleware.cors import CORSMiddleware

        app.add_middleware(
            CORSMiddleware,
            allow_origins=cors_origins,
            allow_credentials=True,
            allow_methods=["*"],
            allow_headers=["*"],
        )

    register_error_handlers(app)

    app.include_router(health.router)
    app.include_router(transcriptions.router)
    app.include_router(translations.router)
    app.include_router(speech.router)
    app.include_router(realtime.router)

    return app
