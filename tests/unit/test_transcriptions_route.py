"""Testes da rota POST /v1/audio/transcriptions."""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock

import httpx

from macaw._types import BatchResult, SegmentDetail
from macaw.server.app import create_app


def _make_mock_scheduler(text: str = "Ola mundo") -> MagicMock:
    scheduler = MagicMock()
    scheduler.transcribe = AsyncMock(
        return_value=BatchResult(
            text=text,
            language="pt",
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


async def test_transcribe_json_format() -> None:
    scheduler = _make_mock_scheduler()
    app = create_app(registry=_make_mock_registry(), scheduler=scheduler)

    async with httpx.AsyncClient(
        transport=httpx.ASGITransport(app=app),
        base_url="http://test",
    ) as client:
        response = await client.post(
            "/v1/audio/transcriptions",
            files={"file": ("audio.wav", b"fake-audio-data", "audio/wav")},
            data={"model": "faster-whisper-tiny"},
        )

    assert response.status_code == 200
    assert response.json() == {"text": "Ola mundo"}
    scheduler.transcribe.assert_awaited_once()


async def test_transcribe_passes_language() -> None:
    scheduler = _make_mock_scheduler()
    app = create_app(registry=_make_mock_registry(), scheduler=scheduler)

    async with httpx.AsyncClient(
        transport=httpx.ASGITransport(app=app),
        base_url="http://test",
    ) as client:
        await client.post(
            "/v1/audio/transcriptions",
            files={"file": ("audio.wav", b"fake-audio-data", "audio/wav")},
            data={"model": "faster-whisper-tiny", "language": "pt"},
        )

    call_args = scheduler.transcribe.call_args[0][0]
    assert call_args.language == "pt"


async def test_transcribe_passes_temperature() -> None:
    scheduler = _make_mock_scheduler()
    app = create_app(registry=_make_mock_registry(), scheduler=scheduler)

    async with httpx.AsyncClient(
        transport=httpx.ASGITransport(app=app),
        base_url="http://test",
    ) as client:
        await client.post(
            "/v1/audio/transcriptions",
            files={"file": ("audio.wav", b"fake-audio-data", "audio/wav")},
            data={"model": "faster-whisper-tiny", "temperature": "0.5"},
        )

    call_args = scheduler.transcribe.call_args[0][0]
    assert call_args.temperature == 0.5


async def test_transcribe_task_is_transcribe() -> None:
    scheduler = _make_mock_scheduler()
    app = create_app(registry=_make_mock_registry(), scheduler=scheduler)

    async with httpx.AsyncClient(
        transport=httpx.ASGITransport(app=app),
        base_url="http://test",
    ) as client:
        await client.post(
            "/v1/audio/transcriptions",
            files={"file": ("audio.wav", b"fake-audio-data", "audio/wav")},
            data={"model": "faster-whisper-tiny"},
        )

    call_args = scheduler.transcribe.call_args[0][0]
    assert call_args.task == "transcribe"


async def test_transcribe_requires_model_field() -> None:
    app = create_app(registry=_make_mock_registry(), scheduler=_make_mock_scheduler())

    async with httpx.AsyncClient(
        transport=httpx.ASGITransport(app=app),
        base_url="http://test",
    ) as client:
        response = await client.post(
            "/v1/audio/transcriptions",
            files={"file": ("audio.wav", b"fake-audio-data", "audio/wav")},
        )

    assert response.status_code == 422


async def test_transcribe_requires_file() -> None:
    app = create_app(registry=_make_mock_registry(), scheduler=_make_mock_scheduler())

    async with httpx.AsyncClient(
        transport=httpx.ASGITransport(app=app),
        base_url="http://test",
    ) as client:
        response = await client.post(
            "/v1/audio/transcriptions",
            data={"model": "faster-whisper-tiny"},
        )

    assert response.status_code == 422


async def test_transcribe_file_too_large() -> None:
    app = create_app(registry=_make_mock_registry(), scheduler=_make_mock_scheduler())

    large_data = b"x" * (26 * 1024 * 1024)  # 26MB

    async with httpx.AsyncClient(
        transport=httpx.ASGITransport(app=app),
        base_url="http://test",
    ) as client:
        response = await client.post(
            "/v1/audio/transcriptions",
            files={"file": ("audio.wav", large_data, "audio/wav")},
            data={"model": "faster-whisper-tiny"},
        )

    assert response.status_code == 413
    assert response.json()["error"]["code"] == "file_too_large"


async def test_transcribe_invalid_response_format_returns_400() -> None:
    app = create_app(registry=_make_mock_registry(), scheduler=_make_mock_scheduler())

    async with httpx.AsyncClient(
        transport=httpx.ASGITransport(app=app),
        base_url="http://test",
    ) as client:
        response = await client.post(
            "/v1/audio/transcriptions",
            files={"file": ("audio.wav", b"fake-audio-data", "audio/wav")},
            data={"model": "faster-whisper-tiny", "response_format": "xml"},
        )

    assert response.status_code == 400
    body = response.json()
    assert body["error"]["code"] == "invalid_request"
    assert "xml" in body["error"]["message"]
    assert "json" in body["error"]["message"]


async def test_transcribe_unsupported_content_type_returns_400() -> None:
    app = create_app(registry=_make_mock_registry(), scheduler=_make_mock_scheduler())

    async with httpx.AsyncClient(
        transport=httpx.ASGITransport(app=app),
        base_url="http://test",
    ) as client:
        response = await client.post(
            "/v1/audio/transcriptions",
            files={"file": ("doc.pdf", b"fake-data", "application/pdf")},
            data={"model": "faster-whisper-tiny"},
        )

    assert response.status_code == 400
    body = response.json()
    assert body["error"]["code"] == "invalid_audio"


async def test_transcribe_text_format() -> None:
    scheduler = _make_mock_scheduler(text="Hello world")
    app = create_app(registry=_make_mock_registry(), scheduler=scheduler)

    async with httpx.AsyncClient(
        transport=httpx.ASGITransport(app=app),
        base_url="http://test",
    ) as client:
        response = await client.post(
            "/v1/audio/transcriptions",
            files={"file": ("audio.wav", b"fake-audio-data", "audio/wav")},
            data={"model": "faster-whisper-tiny", "response_format": "text"},
        )

    assert response.status_code == 200
    assert response.text == "Hello world"


async def test_transcribe_passes_timestamp_granularities() -> None:
    scheduler = _make_mock_scheduler()
    app = create_app(registry=_make_mock_registry(), scheduler=scheduler)

    async with httpx.AsyncClient(
        transport=httpx.ASGITransport(app=app),
        base_url="http://test",
    ) as client:
        await client.post(
            "/v1/audio/transcriptions",
            files={"file": ("audio.wav", b"fake-audio-data", "audio/wav")},
            data={
                "model": "faster-whisper-tiny",
                "timestamp_granularities[]": ["word", "segment"],
            },
        )

    call_args = scheduler.transcribe.call_args[0][0]
    assert call_args.timestamp_granularities == ("word", "segment")


async def test_transcribe_timestamp_granularities_defaults_to_segment() -> None:
    scheduler = _make_mock_scheduler()
    app = create_app(registry=_make_mock_registry(), scheduler=scheduler)

    async with httpx.AsyncClient(
        transport=httpx.ASGITransport(app=app),
        base_url="http://test",
    ) as client:
        await client.post(
            "/v1/audio/transcriptions",
            files={"file": ("audio.wav", b"fake-audio-data", "audio/wav")},
            data={"model": "faster-whisper-tiny"},
        )

    call_args = scheduler.transcribe.call_args[0][0]
    assert call_args.timestamp_granularities == ("segment",)
