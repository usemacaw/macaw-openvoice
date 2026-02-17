"""Pydantic models para o endpoint TTS POST /v1/audio/speech."""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field

from macaw.server.constants import TTS_MAX_TEXT_LENGTH


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
