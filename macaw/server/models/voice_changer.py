"""Pydantic models for the voice changer (speech-to-speech) endpoint."""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field

from macaw.server.models.voice_settings import VoiceSettings


class VoiceChangerResponse(BaseModel):
    """Metadata response for voice change requests."""

    request_id: str
    voice_id: str
    output_format: str


class VoiceChangerFormParams(BaseModel):
    """Form parameters for the voice changer endpoint.

    These are documented for OpenAPI but parsed manually from
    Form(...) parameters in the route function.
    """

    output_format: Literal["wav", "pcm", "mp3", "opus"] = Field(
        default="wav",
        description="Output audio format.",
    )
    voice_settings: VoiceSettings | None = Field(
        default=None,
        description="Abstract voice characteristics for the target voice.",
    )
