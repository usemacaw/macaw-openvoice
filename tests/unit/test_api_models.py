"""Tests for API request and response models."""

from __future__ import annotations

import uuid

import pytest

from macaw._types import ResponseFormat
from macaw.server.models.requests import TranscribeRequest
from macaw.server.models.responses import (
    SegmentResponse,
    TranscriptionResponse,
    VerboseTranscriptionResponse,
    WordResponse,
)


class TestTranscribeRequest:
    """Tests for TranscribeRequest dataclass."""

    def test_create_with_required_fields(self) -> None:
        req = TranscribeRequest(
            request_id=str(uuid.uuid4()),
            model_name="faster-whisper-tiny",
            audio_data=b"fake-audio",
        )
        assert req.model_name == "faster-whisper-tiny"
        assert req.language is None
        assert req.response_format == ResponseFormat.JSON
        assert req.temperature == 0.0
        assert req.task == "transcribe"

    def test_create_with_all_fields(self) -> None:
        req = TranscribeRequest(
            request_id="test-123",
            model_name="faster-whisper-large-v3",
            audio_data=b"fake-audio",
            language="pt",
            response_format=ResponseFormat.VERBOSE_JSON,
            temperature=0.5,
            timestamp_granularities=("word", "segment"),
            initial_prompt="Termos: PIX, TED",
            hot_words=("PIX", "TED"),
            task="translate",
        )
        assert req.language == "pt"
        assert req.response_format == ResponseFormat.VERBOSE_JSON
        assert req.temperature == 0.5
        assert req.task == "translate"
        assert req.hot_words == ("PIX", "TED")

    def test_frozen_immutable(self) -> None:
        req = TranscribeRequest(
            request_id="test",
            model_name="model",
            audio_data=b"data",
        )
        with pytest.raises(AttributeError):
            req.model_name = "other"  # type: ignore[misc]


class TestTranscriptionResponse:
    """Tests for JSON response model."""

    def test_json_format(self) -> None:
        resp = TranscriptionResponse(text="Ola mundo")
        assert resp.model_dump() == {"text": "Ola mundo"}

    def test_empty_text(self) -> None:
        resp = TranscriptionResponse(text="")
        assert resp.text == ""


class TestVerboseTranscriptionResponse:
    """Tests for verbose_json response model."""

    def test_verbose_format(self) -> None:
        resp = VerboseTranscriptionResponse(
            task="transcribe",
            language="pt",
            duration=2.5,
            text="Ola, como posso ajudar?",
            segments=[
                SegmentResponse(
                    id=0,
                    start=0.0,
                    end=2.5,
                    text="Ola, como posso ajudar?",
                    avg_logprob=-0.25,
                    no_speech_prob=0.01,
                ),
            ],
            words=[
                WordResponse(word="Ola", start=0.0, end=0.5),
                WordResponse(word="como", start=0.6, end=0.9),
            ],
        )
        data = resp.model_dump()
        assert data["language"] == "pt"
        assert data["duration"] == 2.5
        assert len(data["segments"]) == 1
        assert len(data["words"]) == 2

    def test_verbose_without_words(self) -> None:
        resp = VerboseTranscriptionResponse(
            language="en",
            duration=1.0,
            text="Hello",
        )
        assert resp.words is None
        assert resp.segments == []

    def test_task_default(self) -> None:
        resp = VerboseTranscriptionResponse(
            language="pt",
            duration=1.0,
            text="Teste",
        )
        assert resp.task == "transcribe"
