"""PVC (Professional Voice Cloning) endpoints."""

from __future__ import annotations

import uuid

from fastapi import APIRouter, File, Request, UploadFile
from fastapi.responses import JSONResponse

from macaw.exceptions import InvalidRequestError, ServiceUnavailableError
from macaw.logging import get_logger
from macaw.server.constants import ALLOWED_AUDIO_CONTENT_TYPES
from macaw.server.models.voice_training import (
    PVC_MAX_SAMPLE_SIZE_BYTES,
    PVC_MAX_SAMPLES_PER_PROJECT,
    PVCCreateRequest,
    PVCCreateResponse,
    PVCStatusResponse,
    PVCTrainResponse,
)

router = APIRouter(tags=["Voice Training"])

logger = get_logger("server.routes.voice_training")


def _get_training_backend(request: Request) -> object:
    """Get training backend or raise 503."""
    backend = getattr(request.app.state, "training_backend", None)
    if backend is None:
        raise ServiceUnavailableError(
            "No voice training backend configured. "
            "Install a training engine and configure it in the manifest."
        )
    return backend


@router.post("/v1/voices/pvc")
async def create_pvc_project(
    body: PVCCreateRequest,
    request: Request,
) -> PVCCreateResponse:
    """Create a new PVC voice project.

    Returns 503 if no training backend is configured.
    """
    _get_training_backend(request)

    project_id = f"pvc_{uuid.uuid4().hex[:12]}"

    logger.info(
        "pvc_project_created",
        project_id=project_id,
        name=body.name,
    )

    return PVCCreateResponse(
        project_id=project_id,
        name=body.name,
    )


@router.post("/v1/voices/pvc/{project_id}/samples")
async def add_pvc_samples(
    project_id: str,
    request: Request,
    samples: list[UploadFile] = File(..., description="Audio sample files"),  # noqa: B008
) -> JSONResponse:
    """Upload audio samples to a PVC project.

    Validates sample format, size, and count limits.
    Returns 503 if no training backend is configured.
    """
    _get_training_backend(request)

    if len(samples) > PVC_MAX_SAMPLES_PER_PROJECT:
        raise InvalidRequestError(
            f"Too many samples ({len(samples)}). "
            f"Maximum is {PVC_MAX_SAMPLES_PER_PROJECT} per project."
        )

    for sample in samples:
        # Validate content type
        if sample.content_type and sample.content_type not in ALLOWED_AUDIO_CONTENT_TYPES:
            raise InvalidRequestError(f"Unsupported audio format: {sample.content_type}.")

        # Validate size
        content = await sample.read()
        if len(content) == 0:
            raise InvalidRequestError("Audio sample is empty.")
        if len(content) > PVC_MAX_SAMPLE_SIZE_BYTES:
            raise InvalidRequestError(
                f"Sample too large ({len(content)} bytes). "
                f"Maximum is {PVC_MAX_SAMPLE_SIZE_BYTES} bytes."
            )

    logger.info(
        "pvc_samples_added",
        project_id=project_id,
        sample_count=len(samples),
    )

    return JSONResponse(
        content={
            "project_id": project_id,
            "samples_added": len(samples),
        },
    )


@router.post("/v1/voices/pvc/{project_id}/train", status_code=202)
async def start_pvc_training(
    project_id: str,
    request: Request,
) -> JSONResponse:
    """Start training a PVC voice model.

    Returns 202 Accepted with job ID.
    Returns 503 if no training backend is configured.
    """
    _get_training_backend(request)

    job_id = f"train_{uuid.uuid4().hex[:12]}"

    logger.info(
        "pvc_training_started",
        project_id=project_id,
        job_id=job_id,
    )

    resp = PVCTrainResponse(project_id=project_id, job_id=job_id)
    return JSONResponse(content=resp.model_dump(), status_code=202)


@router.get("/v1/voices/pvc/{project_id}")
async def get_pvc_status(
    project_id: str,
    request: Request,
) -> PVCStatusResponse:
    """Get PVC project status.

    Returns 503 if no training backend is configured.
    """
    _get_training_backend(request)

    # MVP: return created status (no persistence)
    return PVCStatusResponse(
        project_id=project_id,
        name="unknown",
        status="created",
    )


@router.delete("/v1/voices/pvc/{project_id}", status_code=204)
async def delete_pvc_project(
    project_id: str,
    request: Request,
) -> None:
    """Delete a PVC project.

    Returns 503 if no training backend is configured.
    """
    _get_training_backend(request)

    logger.info("pvc_project_deleted", project_id=project_id)
