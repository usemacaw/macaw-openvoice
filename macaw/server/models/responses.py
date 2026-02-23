"""API response models — Pydantic models for JSON serialization."""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field


class TranscriptionResponse(BaseModel):
    """JSON format response (default): {"text": "..."}."""

    text: str
    transcription_id: str | None = None


WordType = Literal["word", "audio_event"]


class WordResponse(BaseModel):
    """Word with timestamps."""

    word: str
    start: float
    end: float
    word_type: WordType = Field(default="word", description="Type: 'word' or 'audio_event'.")
    channel_index: int | None = Field(
        default=None, description="Channel index for multi-channel audio."
    )


class SegmentResponse(BaseModel):
    """Detailed transcription segment."""

    id: int
    start: float
    end: float
    text: str
    tokens: list[int] = []
    temperature: float = 0.0
    avg_logprob: float = 0.0
    compression_ratio: float = 0.0
    no_speech_prob: float = 0.0


class SpeakerSegmentResponse(BaseModel):
    """Speaker-attributed segment from diarization."""

    speaker_id: str
    start: float
    end: float
    text: str


class DetectedEntityResponse(BaseModel):
    """A detected entity (PII/PHI/PCI) in transcription text."""

    text: str
    entity_type: str
    category: str
    start_char: int
    end_char: int


class VerboseTranscriptionResponse(BaseModel):
    """Response in verbose_json format."""

    task: str = "transcribe"
    language: str
    duration: float
    text: str
    transcription_id: str | None = None
    segments: list[SegmentResponse] = []
    words: list[WordResponse] | None = None
    speaker_segments: list[SpeakerSegmentResponse] | None = None
    entities: list[DetectedEntityResponse] | None = None
    additional_formats: list[AdditionalFormatResponse] | None = None


class CancelResponse(BaseModel):
    """Response for POST /v1/audio/transcriptions/{request_id}/cancel."""

    request_id: str
    cancelled: bool


class AdditionalFormatResponse(BaseModel):
    """A single export format result (base64-encoded content)."""

    format: str
    content: str
    content_type: str
    file_extension: str


class WebhookAcceptedResponse(BaseModel):
    """Response when webhook-based async processing is accepted."""

    request_id: str
    job_id: str
    status: str = "accepted"
    message: str = "Transcription job submitted. Results will be delivered via webhook."
