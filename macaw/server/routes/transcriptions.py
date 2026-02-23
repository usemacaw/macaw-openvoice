"""POST /v1/audio/transcriptions — file transcription + cancel."""

from __future__ import annotations

import asyncio
import json
import uuid
from typing import Any

from fastapi import APIRouter, Depends, Form, Request, UploadFile
from fastapi.responses import JSONResponse

from macaw.postprocessing.pipeline import PostProcessingPipeline  # noqa: TC001
from macaw.preprocessing.pipeline import AudioPreprocessingPipeline  # noqa: TC001
from macaw.scheduler.scheduler import Scheduler  # noqa: TC001
from macaw.server.constants import DEFAULT_TEMPERATURE, MAX_TEMPERATURE, MODEL_NAME_MAX_LENGTH
from macaw.server.dependencies import (
    get_postprocessing_pipeline,
    get_preprocessing_pipeline,
    get_scheduler,
)
from macaw.server.models.responses import CancelResponse, WebhookAcceptedResponse
from macaw.server.routes._common import handle_audio_request

router = APIRouter(tags=["Audio"])


@router.post("/v1/audio/transcriptions/{request_id}/cancel", response_model=CancelResponse)
async def cancel_transcription(
    request_id: str,
    scheduler: Scheduler = Depends(get_scheduler),  # noqa: B008
) -> CancelResponse:
    """Cancel a transcription request in the queue or in flight.

    Idempotent: cancelling a missing or already completed request returns
    ``cancelled: false`` without error.

    Returns:
        JSON with ``request_id`` and ``cancelled`` (bool).
    """
    cancelled = scheduler.cancel(request_id)
    return CancelResponse(request_id=request_id, cancelled=cancelled)


@router.post("/v1/audio/transcriptions")
async def create_transcription(
    request: Request,
    file: UploadFile,
    model: str = Form(max_length=MODEL_NAME_MAX_LENGTH),
    language: str | None = Form(default=None),
    prompt: str | None = Form(default=None),
    response_format: str = Form(default="json"),
    temperature: float = Form(default=DEFAULT_TEMPERATURE, ge=0.0, le=MAX_TEMPERATURE),
    timestamp_granularities: list[str] = Form(  # noqa: B008
        default=["segment"], alias="timestamp_granularities[]"
    ),
    hot_words: str | None = Form(default=None),
    diarize: bool = Form(default=False),
    max_speakers: int | None = Form(default=None),
    webhook_url: str | None = Form(default=None),
    webhook_secret: str | None = Form(default=None),
    webhook_metadata: str | None = Form(default=None),
    itn: bool = Form(default=True),
    scheduler: Scheduler = Depends(get_scheduler),  # noqa: B008
    preprocessing_pipeline: AudioPreprocessingPipeline | None = Depends(  # noqa: B008
        get_preprocessing_pipeline
    ),
    postprocessing_pipeline: PostProcessingPipeline | None = Depends(get_postprocessing_pipeline),  # noqa: B008
) -> Any:
    """Transcribe an audio file.

    Compatible with OpenAI Audio API POST /v1/audio/transcriptions.

    When ``webhook_url`` is provided, returns HTTP 202 immediately and delivers
    the transcription result to the webhook URL asynchronously.
    """
    if webhook_url:
        from macaw.scheduler.async_jobs import AsyncJobManager
        from macaw.server.webhooks import WebhookDelivery

        job_manager: AsyncJobManager = getattr(request.app.state, "async_job_manager", None)  # type: ignore[assignment]
        if job_manager is None:
            job_manager = AsyncJobManager()
            request.app.state.async_job_manager = job_manager

        parsed_metadata: dict[str, object] | None = (
            json.loads(webhook_metadata) if webhook_metadata else None
        )

        request_id = str(uuid.uuid4())
        job_id = job_manager.submit(
            request_id=request_id,
            webhook_url=webhook_url,
            webhook_secret=webhook_secret,
            metadata=parsed_metadata,
        )

        delivery = WebhookDelivery()

        coro = handle_audio_request(
            file=file,
            model=model,
            language=language,
            prompt=prompt,
            response_format=response_format,
            temperature=temperature,
            timestamp_granularities=tuple(timestamp_granularities),
            task="transcribe",
            scheduler=scheduler,
            preprocessing_pipeline=preprocessing_pipeline,
            postprocessing_pipeline=postprocessing_pipeline,
            itn=itn,
            hot_words=hot_words,
            diarize=diarize,
            max_speakers=max_speakers,
        )

        task = asyncio.create_task(job_manager.run_job(job_id, coro, delivery))
        job_manager.track_task(job_id, task)

        return JSONResponse(
            status_code=202,
            content=WebhookAcceptedResponse(
                request_id=request_id,
                job_id=job_id,
            ).model_dump(),
        )

    return await handle_audio_request(
        file=file,
        model=model,
        language=language,
        prompt=prompt,
        response_format=response_format,
        temperature=temperature,
        timestamp_granularities=tuple(timestamp_granularities),
        task="transcribe",
        scheduler=scheduler,
        preprocessing_pipeline=preprocessing_pipeline,
        postprocessing_pipeline=postprocessing_pipeline,
        itn=itn,
        hot_words=hot_words,
        diarize=diarize,
        max_speakers=max_speakers,
    )
