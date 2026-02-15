"""POST /v1/audio/translations — translate audio to English."""

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


@router.post("/v1/audio/translations")
async def create_translation(
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
    """Translate audio to English.

    Compatible with OpenAI Audio API POST /v1/audio/translations.
    """
    return await handle_audio_request(
        file=file,
        model=model,
        language=language,
        prompt=prompt,
        response_format=response_format,
        temperature=temperature,
        task="translate",
        scheduler=scheduler,
        preprocessing_pipeline=preprocessing_pipeline,
        postprocessing_pipeline=postprocessing_pipeline,
        itn=itn,
    )
