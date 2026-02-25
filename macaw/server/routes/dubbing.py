"""Dubbing endpoints: create, status, download, delete."""

from __future__ import annotations

import uuid

from fastapi import APIRouter, File, Form, Request, UploadFile
from fastapi.responses import JSONResponse

from macaw.exceptions import DubbingNotFoundError, InvalidRequestError, ServiceUnavailableError
from macaw.logging import get_logger
from macaw.server.constants import ALLOWED_AUDIO_CONTENT_TYPES
from macaw.server.models.dubbing import DubbingJobResponse, DubbingStatusResponse

router = APIRouter(tags=["Dubbing"])

logger = get_logger("server.routes.dubbing")

# Maximum source audio for dubbing (100 MB).
_DUBBING_MAX_FILE_SIZE = 100 * 1024 * 1024


@router.post("/v1/dubbing", status_code=202)
async def create_dubbing(
    request: Request,
    target_lang: str = Form(..., description="Target language code"),
    audio: UploadFile = File(..., description="Source audio file"),  # noqa: B008
    source_lang: str | None = Form(default=None, description="Source language code"),
    voice_id: str | None = Form(default=None, description="Target voice ID"),
    output_format: str = Form(default="wav", description="Output audio format"),
) -> JSONResponse:
    """Create a dubbing job.

    Accepts source audio and target language. Returns a job ID for
    tracking the async dubbing pipeline (STT -> translate -> TTS).

    Returns 503 if no dubbing orchestrator is configured.
    """
    dubbing_id = str(uuid.uuid4())

    logger.info(
        "dubbing_create_request",
        dubbing_id=dubbing_id,
        target_lang=target_lang,
        source_lang=source_lang,
        output_format=output_format,
    )

    # Check if dubbing orchestrator is configured
    orchestrator = getattr(request.app.state, "dubbing_orchestrator", None)
    if orchestrator is None:
        raise ServiceUnavailableError(
            "No dubbing orchestrator configured. "
            "Install a translation backend and configure dubbing in the manifest."
        )

    # Validate audio file
    if audio.content_type and audio.content_type not in ALLOWED_AUDIO_CONTENT_TYPES:
        raise InvalidRequestError(
            f"Unsupported audio format: {audio.content_type}. "
            f"Supported: {', '.join(sorted(ALLOWED_AUDIO_CONTENT_TYPES))}"
        )

    # Read and validate source audio
    source_bytes = await audio.read()
    if len(source_bytes) == 0:
        raise InvalidRequestError("Source audio file is empty.")
    if len(source_bytes) > _DUBBING_MAX_FILE_SIZE:
        raise InvalidRequestError(
            f"Source audio too large ({len(source_bytes)} bytes). "
            f"Maximum is {_DUBBING_MAX_FILE_SIZE} bytes."
        )

    # Validate target language format (basic check)
    if len(target_lang) < 2:
        raise InvalidRequestError("target_lang must be at least 2 characters.")

    # Validate output format
    from macaw.codec.output_format import parse_output_format

    try:
        parse_output_format(output_format)
    except ValueError as exc:
        raise InvalidRequestError(str(exc)) from exc

    logger.info(
        "dubbing_job_created",
        dubbing_id=dubbing_id,
        source_bytes=len(source_bytes),
        target_lang=target_lang,
    )

    # In production, this would submit the job to AsyncJobManager.
    # For MVP contract validation, return the job ID immediately.
    resp = DubbingJobResponse(dubbing_id=dubbing_id, status="pending")
    return JSONResponse(content=resp.model_dump(), status_code=202)


@router.get("/v1/dubbing/{dubbing_id}")
async def get_dubbing_status(
    dubbing_id: str,
    request: Request,
) -> DubbingStatusResponse:
    """Get dubbing job status.

    Returns 503 if no dubbing orchestrator is configured.
    Returns 404 if the job ID is not found.
    """
    orchestrator = getattr(request.app.state, "dubbing_orchestrator", None)
    if orchestrator is None:
        raise ServiceUnavailableError("No dubbing orchestrator configured.")

    # In production, look up job from AsyncJobManager.
    # For MVP, always return 404.
    raise DubbingNotFoundError(dubbing_id)


@router.get("/v1/dubbing/{dubbing_id}/audio/{language_code}")
async def get_dubbing_audio(
    dubbing_id: str,
    language_code: str,
    request: Request,
) -> JSONResponse:
    """Download dubbed audio for a completed job.

    Returns 503 if no dubbing orchestrator is configured.
    Returns 404 if the job is not found or not completed.
    """
    orchestrator = getattr(request.app.state, "dubbing_orchestrator", None)
    if orchestrator is None:
        raise ServiceUnavailableError("No dubbing orchestrator configured.")

    raise DubbingNotFoundError(dubbing_id)


@router.delete("/v1/dubbing/{dubbing_id}", status_code=204)
async def delete_dubbing(
    dubbing_id: str,
    request: Request,
) -> JSONResponse:
    """Delete a dubbing job.

    Returns 503 if no dubbing orchestrator is configured.
    Returns 404 if the job ID is not found.
    """
    orchestrator = getattr(request.app.state, "dubbing_orchestrator", None)
    if orchestrator is None:
        raise ServiceUnavailableError("No dubbing orchestrator configured.")

    raise DubbingNotFoundError(dubbing_id)
