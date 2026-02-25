"""POST /v1/audio/translations — translate audio to English."""

from __future__ import annotations

from typing import Any

from fastapi import APIRouter, Depends, Form, UploadFile

from macaw.postprocessing.pipeline import PostProcessingPipeline  # noqa: TC001
from macaw.preprocessing.pipeline import AudioPreprocessingPipeline  # noqa: TC001
from macaw.scheduler.scheduler import Scheduler  # noqa: TC001
from macaw.server.constants import DEFAULT_TEMPERATURE, MAX_TEMPERATURE, MODEL_NAME_MAX_LENGTH
from macaw.server.dependencies import (
    get_postprocessing_pipeline,
    get_preprocessing_pipeline,
    get_scheduler,
)
from macaw.server.routes._common import handle_audio_request

router = APIRouter(tags=["Audio"])


@router.post("/v1/audio/translations")
async def create_translation(
    file: UploadFile,
    model: str = Form(max_length=MODEL_NAME_MAX_LENGTH),
    language: str | None = Form(default=None),
    prompt: str | None = Form(default=None),
    response_format: str = Form(default="json"),
    temperature: float = Form(default=DEFAULT_TEMPERATURE, ge=0.0, le=MAX_TEMPERATURE),
    hot_words: str | None = Form(default=None),
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
        timestamp_granularities=tuple(timestamp_granularities),
        task="translate",
        scheduler=scheduler,
        preprocessing_pipeline=preprocessing_pipeline,
        postprocessing_pipeline=postprocessing_pipeline,
        itn=itn,
        hot_words=hot_words,
    )
