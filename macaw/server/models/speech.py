"""Pydantic models para o endpoint TTS POST /v1/audio/speech."""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field

from macaw.server.constants import TTS_MAX_TEXT_LENGTH
from macaw.server.models.effects import AudioEffectsParams


class SpeechRequest(BaseModel):
    """Request body para POST /v1/audio/speech.

    Compativel com o contrato da OpenAI Audio API.
    """

    model: str = Field(description="Nome do modelo TTS no registry.")
    input: str = Field(
        min_length=1, max_length=TTS_MAX_TEXT_LENGTH, description="Texto a ser sintetizado."
    )
    voice: str = Field(
        default="default",
        description="Identificador da voz.",
    )
    response_format: Literal["wav", "pcm", "opus"] = Field(
        default="wav",
        description="Formato de audio de saida (wav, pcm ou opus).",
    )
    speed: float = Field(
        default=1.0,
        ge=0.25,
        le=4.0,
        description="Velocidade da sintese (0.25-4.0).",
    )
    # Extended options for LLM-based TTS engines (e.g., Qwen3-TTS)
    language: str | None = Field(
        default=None,
        description="Idioma alvo (e.g., 'English', 'Chinese').",
    )
    ref_audio: str | None = Field(
        default=None,
        description="Audio de referencia em base64 para voice cloning.",
    )
    ref_text: str | None = Field(
        default=None,
        description="Transcricao do audio de referencia.",
    )
    instruction: str | None = Field(
        default=None,
        description="Instrucao de estilo/voz para voice design.",
    )
    # Post-synthesis audio effects (applied server-side before transport)
    effects: AudioEffectsParams | None = Field(
        default=None,
        description="Audio effects applied after synthesis (pitch shift, reverb).",
    )
    # Alignment options
    include_alignment: bool = Field(
        default=False,
        description="Include word-level timing alignment in response. "
        "When true, response switches to NDJSON (application/x-ndjson) "
        "regardless of response_format. Not compatible with response_format='opus'.",
    )
    alignment_granularity: Literal["word", "character"] = Field(
        default="word",
        description="Alignment granularity: 'word' or 'character'.",
    )
    # Seed and generation control
    seed: int | None = Field(
        default=None,
        ge=1,
        description="Random seed for reproducible generation (must be >= 1). "
        "0 is reserved as 'not set' (proto3 convention). "
        "No-op for deterministic engines (e.g. Kokoro).",
    )
    text_normalization: Literal["auto", "on", "off"] = Field(
        default="auto",
        description="Text normalization mode: 'auto' (engine decides), "
        "'on' (force normalization), 'off' (skip normalization).",
    )
    # Sampling parameters (LLM-based TTS engines like Qwen3-TTS)
    temperature: float | None = Field(
        default=None,
        ge=0.0,
        le=2.0,
        description="Sampling temperature for LLM-based TTS (0 = engine default).",
    )
    top_k: int | None = Field(
        default=None,
        ge=0,
        description="Top-k sampling for LLM-based TTS (0 = engine default).",
    )
    top_p: float | None = Field(
        default=None,
        ge=0.0,
        le=1.0,
        description="Nucleus sampling threshold for LLM-based TTS (0 = engine default).",
    )
