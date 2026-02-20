"""Internal request models for transport between API and Scheduler."""

from __future__ import annotations

from dataclasses import dataclass

from macaw._types import ResponseFormat


@dataclass(frozen=True, slots=True)
class TranscribeRequest:
    """Internal transcription request.

    Not a Pydantic model because it is not used for HTTP validation.
    Validation is done by FastAPI in the route via Form() and UploadFile.
    This dataclass transports validated data from the route to the Scheduler.
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
