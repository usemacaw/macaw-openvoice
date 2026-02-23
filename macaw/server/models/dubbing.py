"""Pydantic models for the dubbing endpoint."""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field


class DubbingRequest(BaseModel):
    """Request body for POST /v1/dubbing (multipart form data documented here)."""

    target_lang: str = Field(
        ...,
        min_length=2,
        max_length=10,
        description="BCP-47 target language code (e.g., 'pt', 'es', 'fr').",
    )
    source_lang: str | None = Field(
        default=None,
        description="BCP-47 source language code. Auto-detected if omitted.",
    )
    voice_id: str | None = Field(
        default=None,
        description="Target voice ID for TTS synthesis. Uses default voice if omitted.",
    )
    output_format: str = Field(
        default="wav",
        description="Output audio format.",
    )


class DubbingJobResponse(BaseModel):
    """Response for dubbing job creation (202 Accepted)."""

    dubbing_id: str
    status: Literal["pending", "processing", "completed", "error"] = "pending"


class DubbingStatusResponse(BaseModel):
    """Response for GET /v1/dubbing/{id}."""

    dubbing_id: str
    status: Literal["pending", "processing", "completed", "error"]
    target_lang: str
    error: str | None = None
