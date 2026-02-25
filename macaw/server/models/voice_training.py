"""Pydantic models for voice training (PVC) endpoints."""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field

# Security constants
PVC_MAX_SAMPLE_SIZE_BYTES = 20 * 1024 * 1024  # 20 MB per sample
PVC_MAX_SAMPLES_PER_PROJECT = 25


class PVCCreateRequest(BaseModel):
    """Request body for POST /v1/voices/pvc."""

    name: str = Field(
        ...,
        min_length=1,
        max_length=200,
        description="Human-readable name for the PVC voice project.",
    )
    description: str = Field(
        default="",
        max_length=1000,
        description="Optional description of the voice project.",
    )


class PVCCreateResponse(BaseModel):
    """Response for PVC project creation."""

    project_id: str
    name: str
    status: Literal["created"] = "created"


class PVCTrainResponse(BaseModel):
    """Response for starting a training job (202 Accepted)."""

    project_id: str
    job_id: str
    status: Literal["pending"] = "pending"


class PVCStatusResponse(BaseModel):
    """Response for GET /v1/voices/pvc/{id}."""

    project_id: str
    name: str
    status: Literal["created", "training", "completed", "error"]
    voice_id: str | None = None
    error: str | None = None
    sample_count: int = 0
