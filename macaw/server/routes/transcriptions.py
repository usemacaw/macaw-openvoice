"""POST /v1/audio/transcriptions — file transcription + cancel."""

from __future__ import annotations

import asyncio
import json
import uuid
from datetime import UTC, datetime
from io import BytesIO
from typing import Any

from fastapi import APIRouter, Depends, File, Form, Request, UploadFile
from fastapi.responses import JSONResponse

from macaw.exceptions import InvalidRequestError
from macaw.logging import get_logger
from macaw.postprocessing.pipeline import PostProcessingPipeline  # noqa: TC001
from macaw.preprocessing.pipeline import AudioPreprocessingPipeline  # noqa: TC001
from macaw.scheduler.scheduler import Scheduler  # noqa: TC001
from macaw.server.constants import DEFAULT_TEMPERATURE, MAX_TEMPERATURE, MODEL_NAME_MAX_LENGTH
from macaw.server.dependencies import (
    get_postprocessing_pipeline,
    get_preprocessing_pipeline,
    get_scheduler,
    get_transcript_store,
)
from macaw.server.models.responses import CancelResponse, WebhookAcceptedResponse
from macaw.server.routes._common import handle_audio_request

logger = get_logger("server.routes.transcriptions")

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
    file: UploadFile | None = File(default=None),  # noqa: B008
    cloud_storage_url: str | None = Form(default=None),
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
    entity_detection: str | None = Form(default=None),
    additional_formats: str | None = Form(default=None),
    tag_audio_events: bool = Form(default=True),
    use_multi_channel: bool = Form(default=False),
    scheduler: Scheduler = Depends(get_scheduler),  # noqa: B008
    preprocessing_pipeline: AudioPreprocessingPipeline | None = Depends(  # noqa: B008
        get_preprocessing_pipeline
    ),
    postprocessing_pipeline: PostProcessingPipeline | None = Depends(get_postprocessing_pipeline),  # noqa: B008
) -> Any:
    """Transcribe an audio file or audio from a URL.

    Compatible with OpenAI Audio API POST /v1/audio/transcriptions.
    Accepts either a file upload or a ``cloud_storage_url`` (HTTPS) —
    exactly one must be provided.

    When ``webhook_url`` is provided, returns HTTP 202 immediately and delivers
    the transcription result to the webhook URL asynchronously.
    """
    # Validate exactly one of file/url is provided
    if file is not None and cloud_storage_url is not None:
        raise InvalidRequestError("Provide either 'file' or 'cloud_storage_url', not both.")
    if file is None and cloud_storage_url is None:
        raise InvalidRequestError("Either 'file' or 'cloud_storage_url' is required.")

    # Download audio from URL if provided
    if cloud_storage_url is not None:
        from macaw.config.settings import get_settings
        from macaw.server.audio_downloader import download_audio

        settings = get_settings()
        audio_bytes = await download_audio(
            cloud_storage_url,
            max_size_bytes=settings.stt_download.max_size_bytes,
            timeout_s=settings.stt_download.timeout_s,
        )
        file = UploadFile(file=BytesIO(audio_bytes), filename="downloaded_audio")

    assert file is not None  # Guaranteed by validation above

    parsed_entity_detection: list[str] | None = None
    if entity_detection is not None:
        parsed_entity_detection = [c.strip() for c in entity_detection.split(",") if c.strip()]

    parsed_additional_formats: list[dict[str, str]] | None = None
    if additional_formats is not None:
        try:
            parsed_additional_formats = json.loads(additional_formats)
        except (json.JSONDecodeError, TypeError) as exc:
            raise InvalidRequestError(f"Invalid additional_formats JSON: {exc}") from exc

        if not isinstance(parsed_additional_formats, list):
            raise InvalidRequestError("additional_formats must be a JSON array of objects.")

        # Validate format names up-front (fail-fast before transcription)
        from macaw.postprocessing.formatters import SUPPORTED_EXPORT_FORMATS

        for fmt_req in parsed_additional_formats:
            fmt_name = fmt_req.get("format", "")
            if fmt_name not in SUPPORTED_EXPORT_FORMATS:
                raise InvalidRequestError(
                    f"Unsupported export format '{fmt_name}'. "
                    f"Supported: {', '.join(sorted(SUPPORTED_EXPORT_FORMATS))}."
                )

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
            entity_detection=parsed_entity_detection,
            additional_formats=parsed_additional_formats,
            tag_audio_events=tag_audio_events,
            use_multi_channel=use_multi_channel,
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

    response = await handle_audio_request(
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
        entity_detection=parsed_entity_detection,
        additional_formats=parsed_additional_formats,
        tag_audio_events=tag_audio_events,
        use_multi_channel=use_multi_channel,
    )

    # Store transcript if transcript store is configured and response is a dict
    transcript_store = get_transcript_store(request)
    if transcript_store is not None and isinstance(response, dict):
        from macaw.server.transcript_store.interface import StoredTranscript

        transcription_id = str(uuid.uuid4())
        transcript = StoredTranscript(
            transcription_id=transcription_id,
            text=response.get("text", ""),
            language=response.get("language"),
            duration=response.get("duration"),
            model=model,
            created_at=datetime.now(UTC).isoformat(),
            metadata=response,
        )
        try:
            await transcript_store.save(transcript)
            response["transcription_id"] = transcription_id
        except Exception:
            logger.warning("transcript_store_save_failed", transcription_id=transcription_id)

    return response
