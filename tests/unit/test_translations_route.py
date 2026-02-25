"""Tests for the POST /v1/audio/translations route."""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock

import httpx

from macaw._types import BatchResult, SegmentDetail
from macaw.server.app import create_app


def _make_mock_scheduler(text: str = "Hello world") -> MagicMock:
    scheduler = MagicMock()
    scheduler.transcribe = AsyncMock(
        return_value=BatchResult(
            text=text,
            language="en",
            duration=1.5,
            segments=(SegmentDetail(id=0, start=0.0, end=1.5, text=text),),
        )
    )
    return scheduler


def _make_mock_registry() -> MagicMock:
    registry = MagicMock()
    registry.has_model.return_value = True
    registry.get_manifest.return_value = MagicMock()
    return registry


async def test_translate_json_format() -> None:
    scheduler = _make_mock_scheduler()
    app = create_app(registry=_make_mock_registry(), scheduler=scheduler)

    async with httpx.AsyncClient(
        transport=httpx.ASGITransport(app=app),
        base_url="http://test",
    ) as client:
        response = await client.post(
            "/v1/audio/translations",
            files={"file": ("audio.wav", b"fake-audio-data", "audio/wav")},
            data={"model": "faster-whisper-tiny"},
        )

    assert response.status_code == 200
    assert response.json() == {"text": "Hello world"}


async def test_translate_task_is_translate() -> None:
    scheduler = _make_mock_scheduler()
    app = create_app(registry=_make_mock_registry(), scheduler=scheduler)

    async with httpx.AsyncClient(
        transport=httpx.ASGITransport(app=app),
        base_url="http://test",
    ) as client:
        await client.post(
            "/v1/audio/translations",
            files={"file": ("audio.wav", b"fake-audio-data", "audio/wav")},
            data={"model": "faster-whisper-tiny"},
        )

    call_args = scheduler.transcribe.call_args[0][0]
    assert call_args.task == "translate"


async def test_translate_passes_language() -> None:
    scheduler = _make_mock_scheduler()
    app = create_app(registry=_make_mock_registry(), scheduler=scheduler)

    async with httpx.AsyncClient(
        transport=httpx.ASGITransport(app=app),
        base_url="http://test",
    ) as client:
        await client.post(
            "/v1/audio/translations",
            files={"file": ("audio.wav", b"fake-audio-data", "audio/wav")},
            data={"model": "faster-whisper-tiny", "language": "pt"},
        )

    call_args = scheduler.transcribe.call_args[0][0]
    assert call_args.language == "pt"


async def test_translate_invalid_response_format_returns_400() -> None:
    app = create_app(registry=_make_mock_registry(), scheduler=_make_mock_scheduler())

    async with httpx.AsyncClient(
        transport=httpx.ASGITransport(app=app),
        base_url="http://test",
    ) as client:
        response = await client.post(
            "/v1/audio/translations",
            files={"file": ("audio.wav", b"fake-audio-data", "audio/wav")},
            data={"model": "faster-whisper-tiny", "response_format": "xml"},
        )

    assert response.status_code == 400
    body = response.json()
    assert body["error"]["code"] == "invalid_request"
    assert "xml" in body["error"]["message"]


async def test_translate_requires_model() -> None:
    app = create_app(registry=_make_mock_registry(), scheduler=_make_mock_scheduler())

    async with httpx.AsyncClient(
        transport=httpx.ASGITransport(app=app),
        base_url="http://test",
    ) as client:
        response = await client.post(
            "/v1/audio/translations",
            files={"file": ("audio.wav", b"fake-audio-data", "audio/wav")},
        )

    assert response.status_code == 422


# --- hot_words ---


async def test_translate_passes_hot_words_to_scheduler() -> None:
    """hot_words Form param is parsed and forwarded to TranscribeRequest."""
    scheduler = _make_mock_scheduler()
    app = create_app(registry=_make_mock_registry(), scheduler=scheduler)

    async with httpx.AsyncClient(
        transport=httpx.ASGITransport(app=app),
        base_url="http://test",
    ) as client:
        await client.post(
            "/v1/audio/translations",
            files={"file": ("audio.wav", b"fake-audio-data", "audio/wav")},
            data={"model": "faster-whisper-tiny", "hot_words": "PIX,TED"},
        )

    call_args = scheduler.transcribe.call_args[0][0]
    assert call_args.hot_words == ("PIX", "TED")


async def test_translate_without_hot_words_defaults_to_none() -> None:
    """Backward compat: omitting hot_words keeps None."""
    scheduler = _make_mock_scheduler()
    app = create_app(registry=_make_mock_registry(), scheduler=scheduler)

    async with httpx.AsyncClient(
        transport=httpx.ASGITransport(app=app),
        base_url="http://test",
    ) as client:
        await client.post(
            "/v1/audio/translations",
            files={"file": ("audio.wav", b"fake-audio-data", "audio/wav")},
            data={"model": "faster-whisper-tiny"},
        )

    call_args = scheduler.transcribe.call_args[0][0]
    assert call_args.hot_words is None


# --- timestamp_granularities ---


async def test_translate_passes_timestamp_granularities() -> None:
    """timestamp_granularities Form param is forwarded to TranscribeRequest."""
    scheduler = _make_mock_scheduler()
    app = create_app(registry=_make_mock_registry(), scheduler=scheduler)

    async with httpx.AsyncClient(
        transport=httpx.ASGITransport(app=app),
        base_url="http://test",
    ) as client:
        await client.post(
            "/v1/audio/translations",
            files={"file": ("audio.wav", b"fake-audio-data", "audio/wav")},
            data={
                "model": "faster-whisper-tiny",
                "timestamp_granularities[]": ["word", "segment"],
            },
        )

    call_args = scheduler.transcribe.call_args[0][0]
    assert call_args.timestamp_granularities == ("word", "segment")


async def test_translate_timestamp_granularities_defaults_to_segment() -> None:
    """Backward compat: omitting timestamp_granularities defaults to segment."""
    scheduler = _make_mock_scheduler()
    app = create_app(registry=_make_mock_registry(), scheduler=scheduler)

    async with httpx.AsyncClient(
        transport=httpx.ASGITransport(app=app),
        base_url="http://test",
    ) as client:
        await client.post(
            "/v1/audio/translations",
            files={"file": ("audio.wav", b"fake-audio-data", "audio/wav")},
            data={"model": "faster-whisper-tiny"},
        )

    call_args = scheduler.transcribe.call_args[0][0]
    assert call_args.timestamp_granularities == ("segment",)
