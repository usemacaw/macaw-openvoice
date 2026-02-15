"""Pydantic models for the voices endpoints."""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field


class VoiceResponse(BaseModel):
    """A single voice available in the runtime."""

    voice_id: str = Field(description="Unique identifier for the voice.")
    name: str = Field(description="Human-readable name.")
    language: str = Field(description="Language code or 'multi' for multilingual.")
    gender: str | None = Field(default=None, description="Voice gender (male/female) or None.")
    model: str = Field(description="Model name this voice belongs to.")


class VoiceListResponse(BaseModel):
    """Response for GET /v1/voices — follows OpenAI list convention."""

    object: Literal["list"] = "list"
    data: list[VoiceResponse] = Field(default_factory=list)


class SavedVoiceResponse(BaseModel):
    """Response for a saved voice (CRUD operations)."""

    voice_id: str = Field(
        description="Unique voice identifier (use as 'voice_{id}' in speech requests)."
    )
    name: str = Field(description="Human-readable name.")
    voice_type: str = Field(description="Voice type: 'cloned' or 'designed'.")
    language: str | None = Field(default=None)
    ref_text: str | None = Field(default=None)
    instruction: str | None = Field(default=None)
    has_ref_audio: bool = Field(default=False, description="Whether reference audio is stored.")
    created_at: float = Field(description="Unix timestamp of creation.")
