"""POST /v1/audio/transcriptions — file transcription + cancel."""

from __future__ import annotations

from typing import Any

from fastapi import APIRouter, Depends, Form, UploadFile

from macaw.postprocessing.pipeline import PostProcessingPipeline  # noqa: TC001
from macaw.preprocessing.pipeline import AudioPreprocessingPipeline  # noqa: TC001
from macaw.scheduler.scheduler import Scheduler  # noqa: TC001
from macaw.server.dependencies import (
    get_postprocessing_pipeline,
    get_preprocessing_pipeline,
    get_scheduler,
)
from macaw.server.models.responses import CancelResponse
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
    file: UploadFile,
    model: str = Form(max_length=256),
    language: str | None = Form(default=None),
    prompt: str | None = Form(default=None),
    response_format: str = Form(default="json"),
    temperature: float = Form(default=0.0, ge=0.0, le=2.0),
    timestamp_granularities: list[str] = Form(  # noqa: B008
        default=["segment"], alias="timestamp_granularities[]"
    ),
    itn: bool = Form(default=True),
    scheduler: Scheduler = Depends(get_scheduler),  # noqa: B008
    preprocessing_pipeline: AudioPreprocessingPipeline | None = Depends(  # noqa: B008
        get_preprocessing_pipeline
    ),
    postprocessing_pipeline: PostProcessingPipeline | None = Depends(get_postprocessing_pipeline),  # noqa: B008
) -> Any:
    """Transcribe an audio file.

    Compatible with OpenAI Audio API POST /v1/audio/transcriptions.
    """
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
    )
