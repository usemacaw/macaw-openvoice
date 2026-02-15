"""Modelos de resposta da API — Pydantic models para serializacao JSON."""

from __future__ import annotations

from pydantic import BaseModel


class TranscriptionResponse(BaseModel):
    """Resposta formato json (default): {"text": "..."}."""

    text: str


class WordResponse(BaseModel):
    """Palavra com timestamps."""

    word: str
    start: float
    end: float


class SegmentResponse(BaseModel):
    """Segmento de transcricao detalhado."""

    id: int
    start: float
    end: float
    text: str
    tokens: list[int] = []
    temperature: float = 0.0
    avg_logprob: float = 0.0
    compression_ratio: float = 0.0
    no_speech_prob: float = 0.0


class VerboseTranscriptionResponse(BaseModel):
    """Resposta formato verbose_json."""

    task: str = "transcribe"
    language: str
    duration: float
    text: str
    segments: list[SegmentResponse] = []
    words: list[WordResponse] | None = None


class CancelResponse(BaseModel):
    """Response for POST /v1/audio/transcriptions/{request_id}/cancel."""

    request_id: str
    cancelled: bool
