"""FastAPI application factory for Macaw OpenVoice."""

from __future__ import annotations

import uuid
from contextlib import asynccontextmanager
from typing import TYPE_CHECKING

from fastapi import FastAPI, Request, Response
from starlette.middleware.base import BaseHTTPMiddleware, RequestResponseEndpoint

import macaw
from macaw.server.error_handlers import register_error_handlers
from macaw.server.routes import health, realtime, speech, transcriptions, translations, voices


class RequestIDMiddleware(BaseHTTPMiddleware):
    """Assign a unique request_id to each HTTP request."""

    async def dispatch(self, request: Request, call_next: RequestResponseEndpoint) -> Response:
        request.state.request_id = str(uuid.uuid4())
        return await call_next(request)


if TYPE_CHECKING:
    from collections.abc import AsyncIterator

    from macaw.postprocessing.pipeline import PostProcessingPipeline
    from macaw.preprocessing.pipeline import AudioPreprocessingPipeline
    from macaw.registry.registry import ModelRegistry
    from macaw.scheduler.scheduler import Scheduler
    from macaw.server.voice_store import VoiceStore
    from macaw.workers.manager import WorkerManager


def create_app(
    registry: ModelRegistry | None = None,
    scheduler: Scheduler | None = None,
    preprocessing_pipeline: AudioPreprocessingPipeline | None = None,
    postprocessing_pipeline: PostProcessingPipeline | None = None,
    worker_manager: WorkerManager | None = None,
    voice_store: VoiceStore | None = None,
    cors_origins: list[str] | None = None,
) -> FastAPI:
    """Create the FastAPI application.

    Args:
        registry: Model Registry (optional, None only for health endpoint tests).
        scheduler: Scheduler (optional, None only for health endpoint tests).
        preprocessing_pipeline: Audio preprocessing pipeline (optional).
        postprocessing_pipeline: Text post-processing pipeline (optional).
        worker_manager: Worker Manager for TTS (optional).
        cors_origins: List of allowed CORS origins (optional).

    Returns:
        Configured FastAPI application.
    """

    @asynccontextmanager
    async def lifespan(app: FastAPI) -> AsyncIterator[None]:
        yield
        # Shutdown: close pooled TTS gRPC channels
        tts_channels = getattr(app.state, "tts_channels", None)
        if tts_channels:
            from macaw.server.grpc_channels import close_tts_channels

            await close_tts_channels(tts_channels)

    app = FastAPI(
        title="Macaw OpenVoice",
        version=macaw.__version__,
        description="Unified voice runtime (STT + TTS) with OpenAI-compatible API",
        lifespan=lifespan,
    )

    app.state.registry = registry
    app.state.scheduler = scheduler
    app.state.preprocessing_pipeline = preprocessing_pipeline
    app.state.postprocessing_pipeline = postprocessing_pipeline
    app.state.worker_manager = worker_manager
    app.state.voice_store = voice_store
    app.state.tts_channels = {}

    if cors_origins:
        from fastapi.middleware.cors import CORSMiddleware

        app.add_middleware(
            CORSMiddleware,
            allow_origins=cors_origins,
            allow_credentials=True,
            allow_methods=["*"],
            allow_headers=["*"],
        )

    app.add_middleware(RequestIDMiddleware)

    register_error_handlers(app)

    app.include_router(health.router)
    app.include_router(transcriptions.router)
    app.include_router(translations.router)
    app.include_router(speech.router)
    app.include_router(realtime.router)
    app.include_router(voices.router)

    return app
