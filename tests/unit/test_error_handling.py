"""Testes dos exception handlers HTTP."""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock

import httpx

from macaw._types import BatchResult, SegmentDetail
from macaw.exceptions import (
    AudioFormatError,
    InvalidRequestError,
    ModelNotFoundError,
    WorkerCrashError,
    WorkerTimeoutError,
    WorkerUnavailableError,
)
from macaw.server.app import create_app


def _make_mock_registry() -> MagicMock:
    registry = MagicMock()
    registry.has_model.return_value = True
    registry.get_manifest.return_value = MagicMock()
    return registry


def _make_mock_scheduler_raising(exc: Exception) -> MagicMock:
    scheduler = MagicMock()
    scheduler.transcribe = AsyncMock(side_effect=exc)
    return scheduler


def _make_mock_scheduler_ok() -> MagicMock:
    scheduler = MagicMock()
    scheduler.transcribe = AsyncMock(
        return_value=BatchResult(
            text="ok",
            language="pt",
            duration=1.0,
            segments=(SegmentDetail(id=0, start=0.0, end=1.0, text="ok"),),
        )
    )
    return scheduler


async def test_invalid_request_error_returns_400() -> None:
    scheduler = _make_mock_scheduler_raising(InvalidRequestError("parametro invalido"))
    app = create_app(registry=_make_mock_registry(), scheduler=scheduler)

    async with httpx.AsyncClient(
        transport=httpx.ASGITransport(app=app),
        base_url="http://test",
    ) as client:
        response = await client.post(
            "/v1/audio/transcriptions",
            files={"file": ("audio.wav", b"fake-audio", "audio/wav")},
            data={"model": "faster-whisper-tiny"},
        )

    assert response.status_code == 400
    body = response.json()
    assert body["error"]["code"] == "invalid_request"
    assert body["error"]["type"] == "invalid_request_error"
    assert "parametro invalido" in body["error"]["message"]


async def test_model_not_found_returns_404() -> None:
    scheduler = _make_mock_scheduler_raising(ModelNotFoundError("whisper-inexistente"))
    app = create_app(registry=_make_mock_registry(), scheduler=scheduler)

    async with httpx.AsyncClient(
        transport=httpx.ASGITransport(app=app),
        base_url="http://test",
    ) as client:
        response = await client.post(
            "/v1/audio/transcriptions",
            files={"file": ("audio.wav", b"fake-audio", "audio/wav")},
            data={"model": "whisper-inexistente"},
        )

    assert response.status_code == 404
    body = response.json()
    assert body["error"]["code"] == "model_not_found"
    assert "whisper-inexistente" in body["error"]["message"]


async def test_audio_format_error_returns_400() -> None:
    scheduler = _make_mock_scheduler_raising(AudioFormatError("formato nao suportado"))
    app = create_app(registry=_make_mock_registry(), scheduler=scheduler)

    async with httpx.AsyncClient(
        transport=httpx.ASGITransport(app=app),
        base_url="http://test",
    ) as client:
        response = await client.post(
            "/v1/audio/transcriptions",
            files={"file": ("audio.wav", b"fake-audio", "audio/wav")},
            data={"model": "faster-whisper-tiny"},
        )

    assert response.status_code == 400
    assert response.json()["error"]["code"] == "invalid_audio"


async def test_audio_too_large_returns_413() -> None:
    app = create_app(registry=_make_mock_registry(), scheduler=_make_mock_scheduler_ok())

    large_data = b"x" * (26 * 1024 * 1024)

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


async def test_worker_unavailable_returns_503() -> None:
    scheduler = _make_mock_scheduler_raising(WorkerUnavailableError("faster-whisper-tiny"))
    app = create_app(registry=_make_mock_registry(), scheduler=scheduler)

    async with httpx.AsyncClient(
        transport=httpx.ASGITransport(app=app),
        base_url="http://test",
    ) as client:
        response = await client.post(
            "/v1/audio/transcriptions",
            files={"file": ("audio.wav", b"fake-audio", "audio/wav")},
            data={"model": "faster-whisper-tiny"},
        )

    assert response.status_code == 503
    assert response.json()["error"]["code"] == "service_unavailable"
    assert response.headers.get("retry-after") == "5"


async def test_unexpected_error_returns_500_without_stack_trace() -> None:
    scheduler = _make_mock_scheduler_raising(RuntimeError("algo inesperado"))
    app = create_app(registry=_make_mock_registry(), scheduler=scheduler)

    # raise_server_exceptions=False para que o ASGITransport nao re-levante
    # a excecao e permita que o exception handler do FastAPI responda 500
    async with httpx.AsyncClient(
        transport=httpx.ASGITransport(app=app, raise_app_exceptions=False),
        base_url="http://test",
    ) as client:
        response = await client.post(
            "/v1/audio/transcriptions",
            files={"file": ("audio.wav", b"fake-audio", "audio/wav")},
            data={"model": "faster-whisper-tiny"},
        )

    assert response.status_code == 500
    body = response.json()
    assert body["error"]["code"] == "internal_error"
    # Nao deve expor detalhes internos
    assert "algo inesperado" not in body["error"]["message"]
    assert body["error"]["message"] == "Erro interno do servidor"


async def test_worker_timeout_returns_504() -> None:
    scheduler = _make_mock_scheduler_raising(WorkerTimeoutError("fw-50051", 30.0))
    app = create_app(registry=_make_mock_registry(), scheduler=scheduler)

    async with httpx.AsyncClient(
        transport=httpx.ASGITransport(app=app),
        base_url="http://test",
    ) as client:
        response = await client.post(
            "/v1/audio/transcriptions",
            files={"file": ("audio.wav", b"fake-audio", "audio/wav")},
            data={"model": "faster-whisper-tiny"},
        )

    assert response.status_code == 504
    body = response.json()
    assert body["error"]["code"] == "gateway_timeout"
    assert body["error"]["type"] == "worker_timeout_error"


async def test_worker_crash_returns_502() -> None:
    scheduler = _make_mock_scheduler_raising(WorkerCrashError("fw-50051", exit_code=1))
    app = create_app(registry=_make_mock_registry(), scheduler=scheduler)

    async with httpx.AsyncClient(
        transport=httpx.ASGITransport(app=app),
        base_url="http://test",
    ) as client:
        response = await client.post(
            "/v1/audio/transcriptions",
            files={"file": ("audio.wav", b"fake-audio", "audio/wav")},
            data={"model": "faster-whisper-tiny"},
        )

    assert response.status_code == 502
    body = response.json()
    assert body["error"]["code"] == "bad_gateway"
    assert response.headers.get("retry-after") == "5"


async def test_error_response_format_matches_openai() -> None:
    scheduler = _make_mock_scheduler_raising(ModelNotFoundError("test-model"))
    app = create_app(registry=_make_mock_registry(), scheduler=scheduler)

    async with httpx.AsyncClient(
        transport=httpx.ASGITransport(app=app),
        base_url="http://test",
    ) as client:
        response = await client.post(
            "/v1/audio/transcriptions",
            files={"file": ("audio.wav", b"fake-audio", "audio/wav")},
            data={"model": "test-model"},
        )

    body = response.json()
    # Formato OpenAI: {"error": {"message": "...", "type": "...", "code": "..."}}
    assert "error" in body
    assert "message" in body["error"]
    assert "type" in body["error"]
    assert "code" in body["error"]
