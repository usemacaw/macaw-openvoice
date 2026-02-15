"""Modelos internos de request para transporte entre API e Scheduler."""

from __future__ import annotations

from dataclasses import dataclass

from macaw._types import ResponseFormat


@dataclass(frozen=True, slots=True)
class TranscribeRequest:
    """Request interna de transcricao.

    Nao e um Pydantic model porque nao e usado para validacao HTTP.
    A validacao e feita pelo FastAPI na rota via Form() e UploadFile.
    Este dataclass transporta os dados validados da rota ao Scheduler.
    """

    request_id: str
    model_name: str
    audio_data: bytes
    language: str | None = None
    response_format: ResponseFormat = ResponseFormat.JSON
    temperature: float = 0.0
    timestamp_granularities: tuple[str, ...] = ("segment",)
    initial_prompt: str | None = None
    hot_words: tuple[str, ...] | None = None
    task: str = "transcribe"
