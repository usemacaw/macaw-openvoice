"""Tests for diarization wiring in the REST transcriptions endpoint.

Verifies that diarize/max_speakers params flow through to the scheduler,
and that speaker_segments appear in verbose_json responses.
"""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock

import httpx

from macaw._types import BatchResult, ResponseFormat, SegmentDetail, SpeakerSegment
from macaw.server.app import create_app
from macaw.server.formatters import format_response
from macaw.server.models.responses import (
    SpeakerSegmentResponse,
    VerboseTranscriptionResponse,
)


def _make_mock_registry() -> MagicMock:
    registry = MagicMock()
    registry.has_model.return_value = True
    registry.get_manifest.return_value = MagicMock()
    return registry


def _make_mock_scheduler(
    result: BatchResult | None = None,
) -> MagicMock:
    scheduler = MagicMock()
    scheduler.transcribe = AsyncMock(
        return_value=result
        or BatchResult(
            text="hello world",
            language="en",
            duration=2.0,
            segments=(SegmentDetail(id=0, start=0.0, end=2.0, text="hello world"),),
        )
    )
    return scheduler


def _make_result_with_speakers() -> BatchResult:
    return BatchResult(
        text="hello world",
        language="en",
        duration=2.0,
        segments=(SegmentDetail(id=0, start=0.0, end=2.0, text="hello world"),),
        speaker_segments=(
            SpeakerSegment(speaker_id="SPEAKER_00", start=0.0, end=1.0, text="hello"),
            SpeakerSegment(speaker_id="SPEAKER_01", start=1.0, end=2.0, text="world"),
        ),
    )


class TestDiarizeFormParam:
    """diarize and max_speakers form params are forwarded to scheduler."""

    async def test_diarize_true_forwarded(self) -> None:
        scheduler = _make_mock_scheduler()
        app = create_app(registry=_make_mock_registry(), scheduler=scheduler)

        async with httpx.AsyncClient(
            transport=httpx.ASGITransport(app=app),
            base_url="http://test",
        ) as client:
            await client.post(
                "/v1/audio/transcriptions",
                files={"file": ("audio.wav", b"fake-audio", "audio/wav")},
                data={"model": "test-model", "diarize": "true"},
            )

        call_args = scheduler.transcribe.call_args[0][0]
        assert call_args.diarize is True

    async def test_diarize_default_false(self) -> None:
        scheduler = _make_mock_scheduler()
        app = create_app(registry=_make_mock_registry(), scheduler=scheduler)

        async with httpx.AsyncClient(
            transport=httpx.ASGITransport(app=app),
            base_url="http://test",
        ) as client:
            await client.post(
                "/v1/audio/transcriptions",
                files={"file": ("audio.wav", b"fake-audio", "audio/wav")},
                data={"model": "test-model"},
            )

        call_args = scheduler.transcribe.call_args[0][0]
        assert call_args.diarize is False

    async def test_max_speakers_forwarded(self) -> None:
        scheduler = _make_mock_scheduler()
        app = create_app(registry=_make_mock_registry(), scheduler=scheduler)

        async with httpx.AsyncClient(
            transport=httpx.ASGITransport(app=app),
            base_url="http://test",
        ) as client:
            await client.post(
                "/v1/audio/transcriptions",
                files={"file": ("audio.wav", b"fake-audio", "audio/wav")},
                data={
                    "model": "test-model",
                    "diarize": "true",
                    "max_speakers": "4",
                },
            )

        call_args = scheduler.transcribe.call_args[0][0]
        assert call_args.max_speakers == 4


class TestVerboseJsonWithSpeakerSegments:
    """verbose_json format includes speaker_segments when present."""

    async def test_verbose_json_includes_speaker_segments(self) -> None:
        result = _make_result_with_speakers()
        scheduler = _make_mock_scheduler(result)
        app = create_app(registry=_make_mock_registry(), scheduler=scheduler)

        async with httpx.AsyncClient(
            transport=httpx.ASGITransport(app=app),
            base_url="http://test",
        ) as client:
            response = await client.post(
                "/v1/audio/transcriptions",
                files={"file": ("audio.wav", b"fake-audio", "audio/wav")},
                data={
                    "model": "test-model",
                    "response_format": "verbose_json",
                    "diarize": "true",
                },
            )

        data = response.json()
        assert "speaker_segments" in data
        assert len(data["speaker_segments"]) == 2
        assert data["speaker_segments"][0]["speaker_id"] == "SPEAKER_00"
        assert data["speaker_segments"][1]["speaker_id"] == "SPEAKER_01"

    async def test_verbose_json_without_diarize_no_speaker_segments(self) -> None:
        scheduler = _make_mock_scheduler()
        app = create_app(registry=_make_mock_registry(), scheduler=scheduler)

        async with httpx.AsyncClient(
            transport=httpx.ASGITransport(app=app),
            base_url="http://test",
        ) as client:
            response = await client.post(
                "/v1/audio/transcriptions",
                files={"file": ("audio.wav", b"fake-audio", "audio/wav")},
                data={
                    "model": "test-model",
                    "response_format": "verbose_json",
                },
            )

        data = response.json()
        # speaker_segments should not be in response when not requested
        assert "speaker_segments" not in data

    async def test_json_format_no_speaker_segments(self) -> None:
        """JSON format (default) does not include speaker_segments."""
        result = _make_result_with_speakers()
        scheduler = _make_mock_scheduler(result)
        app = create_app(registry=_make_mock_registry(), scheduler=scheduler)

        async with httpx.AsyncClient(
            transport=httpx.ASGITransport(app=app),
            base_url="http://test",
        ) as client:
            response = await client.post(
                "/v1/audio/transcriptions",
                files={"file": ("audio.wav", b"fake-audio", "audio/wav")},
                data={"model": "test-model", "diarize": "true"},
            )

        data = response.json()
        # JSON format only returns {"text": "..."}
        assert "speaker_segments" not in data
        assert "text" in data


class TestSpeakerSegmentResponseModel:
    """SpeakerSegmentResponse Pydantic model."""

    def test_has_correct_fields(self) -> None:
        seg = SpeakerSegmentResponse(
            speaker_id="s0",
            start=1.0,
            end=2.5,
            text="hello",
        )
        assert seg.speaker_id == "s0"
        assert seg.start == 1.0
        assert seg.end == 2.5
        assert seg.text == "hello"

    def test_schema_contains_expected_fields(self) -> None:
        schema = SpeakerSegmentResponse.model_json_schema()
        props = schema["properties"]
        assert "speaker_id" in props
        assert "start" in props
        assert "end" in props
        assert "text" in props


class TestVerboseTranscriptionResponseModel:
    """VerboseTranscriptionResponse includes speaker_segments field."""

    def test_speaker_segments_field_exists(self) -> None:
        schema = VerboseTranscriptionResponse.model_json_schema()
        props = schema["properties"]
        assert "speaker_segments" in props

    def test_speaker_segments_default_none(self) -> None:
        response = VerboseTranscriptionResponse(
            language="en",
            duration=1.0,
            text="hello",
        )
        assert response.speaker_segments is None


class TestFormatResponseSpeakerSegments:
    """format_response correctly handles speaker_segments in verbose_json."""

    def test_verbose_json_includes_speaker_segments(self) -> None:
        result = _make_result_with_speakers()
        response = format_response(result, ResponseFormat.VERBOSE_JSON)
        assert "speaker_segments" in response
        assert len(response["speaker_segments"]) == 2

    def test_verbose_json_omits_when_none(self) -> None:
        result = BatchResult(
            text="hello",
            language="en",
            duration=1.0,
            segments=(SegmentDetail(id=0, start=0.0, end=1.0, text="hello"),),
        )
        response = format_response(result, ResponseFormat.VERBOSE_JSON)
        assert "speaker_segments" not in response

    def test_verbose_json_empty_speaker_segments(self) -> None:
        result = BatchResult(
            text="hello",
            language="en",
            duration=1.0,
            segments=(SegmentDetail(id=0, start=0.0, end=1.0, text="hello"),),
            speaker_segments=(),
        )
        response = format_response(result, ResponseFormat.VERBOSE_JSON)
        assert "speaker_segments" in response
        assert response["speaker_segments"] == []
