"""Pydantic response models for GET /v1/models."""

from __future__ import annotations

from pydantic import BaseModel


class ModelCapabilitiesResponse(BaseModel):
    """Serialized capabilities for a model.

    Includes only client-facing capability fields.  Internal-only fields
    (``architecture``, ``partial_transcripts``, ``initial_prompt``) are
    excluded from the response.
    """

    # STT
    streaming: bool = False
    languages: list[str] = []
    word_timestamps: bool = False
    translation: bool = False
    hot_words: bool = False
    hot_words_mode: str = "none"
    batch_inference: bool = False
    diarization: bool = False
    language_detection: bool = False
    # TTS
    voice_cloning: bool = False
    instruct_mode: bool = False
    alignment: bool = False
    character_alignment: bool = False
    voice_design: bool = False


class ModelInfo(BaseModel):
    """Single model entry in the ``/v1/models`` response."""

    id: str
    object: str = "model"
    owned_by: str = "macaw"
    created: int = 0
    type: str
    engine: str
    capabilities: ModelCapabilitiesResponse
