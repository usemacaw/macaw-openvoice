"""Shared logic between audio routes (transcriptions and translations)."""

from __future__ import annotations

import uuid
from typing import TYPE_CHECKING, Any

from fastapi import UploadFile  # noqa: TC002

from macaw._types import ResponseFormat
from macaw.exceptions import AudioFormatError, AudioTooLargeError, InvalidRequestError
from macaw.logging import get_logger
from macaw.server.constants import ALLOWED_AUDIO_CONTENT_TYPES, MAX_FILE_SIZE_BYTES
from macaw.server.formatters import format_response
from macaw.server.models.requests import TranscribeRequest

if TYPE_CHECKING:
    from macaw.postprocessing.pipeline import PostProcessingPipeline
    from macaw.preprocessing.pipeline import AudioPreprocessingPipeline
    from macaw.scheduler.scheduler import Scheduler

logger = get_logger("server.routes")


async def handle_audio_request(
    *,
    file: UploadFile,
    model: str,
    language: str | None,
    prompt: str | None,
    response_format: str,
    temperature: float,
    task: str,
    scheduler: Scheduler,
    preprocessing_pipeline: AudioPreprocessingPipeline | None = None,
    postprocessing_pipeline: PostProcessingPipeline | None = None,
    itn: bool = True,
    timestamp_granularities: tuple[str, ...] = ("segment",),
) -> Any:
    """Process audio request (transcription or translation).

    Validates input, reads audio, applies preprocessing, sends to scheduler,
    applies post-processing, formats response.

    Args:
        file: Audio file uploaded by the client.
        model: Model name in the registry.
        language: ISO 639-1 code or None (auto-detect).
        prompt: Context to guide transcription.
        response_format: Desired response format.
        temperature: Sampling temperature.
        task: "transcribe" or "translate".
        scheduler: Scheduler to route to the worker.
        preprocessing_pipeline: Audio preprocessing pipeline (optional).
        postprocessing_pipeline: Text post-processing pipeline (optional).
        itn: If True (default), apply post-processing to the result.

    Returns:
        Response formatted according to response_format.
    """
    request_id = str(uuid.uuid4())

    logger.info(
        "audio_request",
        task=task,
        request_id=request_id,
        model=model,
        language=language,
        response_format=response_format,
    )

    # Validate response_format
    try:
        fmt = ResponseFormat(response_format)
    except ValueError:
        valid = ", ".join(e.value for e in ResponseFormat)
        raise InvalidRequestError(
            f"Invalid response_format '{response_format}'. Accepted values: {valid}"
        ) from None

    # Validate file content-type
    if file.content_type and file.content_type not in ALLOWED_AUDIO_CONTENT_TYPES:
        raise AudioFormatError(
            f"Unsupported content-type '{file.content_type}'. "
            "Accepted formats: WAV, MP3, FLAC, OGG, WebM"
        )

    # Validate size (pre-read if available)
    if file.size is not None and file.size > MAX_FILE_SIZE_BYTES:
        raise AudioTooLargeError(file.size, MAX_FILE_SIZE_BYTES)

    # Read audio with limit to prevent OOM on uploads without Content-Length
    audio_data = await file.read(MAX_FILE_SIZE_BYTES + 1)

    if len(audio_data) > MAX_FILE_SIZE_BYTES:
        raise AudioTooLargeError(len(audio_data), MAX_FILE_SIZE_BYTES)

    # Apply preprocessing if pipeline is configured
    if preprocessing_pipeline is not None:
        audio_data = preprocessing_pipeline.process(audio_data)

    request = TranscribeRequest(
        request_id=request_id,
        model_name=model,
        audio_data=audio_data,
        language=language,
        response_format=fmt,
        temperature=temperature,
        timestamp_granularities=timestamp_granularities,
        initial_prompt=prompt,
        task=task,
    )

    result = await scheduler.transcribe(request)

    # Apply post-processing if pipeline is configured and ITN is enabled
    if postprocessing_pipeline is not None and itn:
        result = postprocessing_pipeline.process_result(result)

    return format_response(result, fmt, task=task)
