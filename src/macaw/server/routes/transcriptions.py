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
from macaw.server.routes._common import handle_audio_request

router = APIRouter()


@router.post("/v1/audio/transcriptions/{request_id}/cancel")
async def cancel_transcription(
    request_id: str,
    scheduler: Scheduler = Depends(get_scheduler),  # noqa: B008
) -> dict[str, Any]:
    """Cancel a transcription request in the queue or in flight.

    Idempotent: cancelling a missing or already completed request returns
    ``cancelled: false`` without error.

    Returns:
        JSON with ``request_id`` and ``cancelled`` (bool).
    """
    cancelled = scheduler.cancel(request_id)
    return {"request_id": request_id, "cancelled": cancelled}


@router.post("/v1/audio/transcriptions")
async def create_transcription(
    file: UploadFile,
    model: str = Form(),
    language: str | None = Form(default=None),
    prompt: str | None = Form(default=None),
    response_format: str = Form(default="json"),
    temperature: float = Form(default=0.0),
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
        task="transcribe",
        scheduler=scheduler,
        preprocessing_pipeline=preprocessing_pipeline,
        postprocessing_pipeline=postprocessing_pipeline,
        itn=itn,
    )
