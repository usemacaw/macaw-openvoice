"""Pydantic models para o endpoint TTS POST /v1/audio/speech."""

from __future__ import annotations

from pydantic import BaseModel, Field


class SpeechRequest(BaseModel):
    """Request body para POST /v1/audio/speech.

    Compativel com o contrato da OpenAI Audio API.
    """

    model: str = Field(description="Nome do modelo TTS no registry.")
    input: str = Field(description="Texto a ser sintetizado.")
    voice: str = Field(
        default="default",
        description="Identificador da voz.",
    )
    response_format: str = Field(
        default="wav",
        description="Formato de audio de saida (wav ou pcm).",
    )
    speed: float = Field(
        default=1.0,
        ge=0.25,
        le=4.0,
        description="Velocidade da sintese (0.25-4.0).",
    )
