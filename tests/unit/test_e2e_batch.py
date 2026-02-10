"""Testes end-to-end do fluxo batch: HTTP -> API Server -> Scheduler (mock) -> Response.

Valida que o pipeline completo funciona para todos os formatos de resposta
e cenarios de erro, sem necessidade de worker gRPC real.
"""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock

import httpx

from macaw._types import BatchResult, SegmentDetail, WordTimestamp
from macaw.server.app import create_app


def _make_batch_result() -> BatchResult:
    return BatchResult(
        text="Ola, como posso ajudar?",
        language="pt",
        duration=2.5,
        segments=(
            SegmentDetail(
                id=0,
                start=0.0,
                end=1.2,
                text="Ola,",
                avg_logprob=-0.25,
                no_speech_prob=0.01,
                compression_ratio=1.1,
            ),
            SegmentDetail(
                id=1,
                start=1.3,
                end=2.5,
                text="como posso ajudar?",
                avg_logprob=-0.30,
                no_speech_prob=0.02,
                compression_ratio=1.0,
            ),
        ),
        words=(
            WordTimestamp(word="Ola", start=0.0, end=0.5),
            WordTimestamp(word="como", start=0.6, end=0.9),
            WordTimestamp(word="posso", start=1.0, end=1.3),
            WordTimestamp(word="ajudar", start=1.4, end=2.5),
        ),
    )


def _make_registry() -> MagicMock:
    registry = MagicMock()
    registry.has_model.return_value = True
    registry.get_manifest.return_value = MagicMock()
    return registry


def _make_scheduler(result: BatchResult | None = None) -> MagicMock:
    scheduler = MagicMock()
    scheduler.transcribe = AsyncMock(return_value=result or _make_batch_result())
    return scheduler


def _make_app(
    registry: MagicMock | None = None,
    scheduler: MagicMock | None = None,
) -> object:
    return create_app(
        registry=registry or _make_registry(),
        scheduler=scheduler or _make_scheduler(),
    )


# --- Formatos de resposta ---


class TestTranscribeJsonFormat:
    async def test_returns_text_field(self) -> None:
        app = _make_app()
        async with httpx.AsyncClient(
            transport=httpx.ASGITransport(app=app), base_url="http://test"
        ) as client:
            response = await client.post(
                "/v1/audio/transcriptions",
                files={"file": ("audio.wav", b"fake-audio", "audio/wav")},
                data={"model": "faster-whisper-tiny"},
            )
        assert response.status_code == 200
        body = response.json()
        assert body == {"text": "Ola, como posso ajudar?"}

    async def test_content_type_is_json(self) -> None:
        app = _make_app()
        async with httpx.AsyncClient(
            transport=httpx.ASGITransport(app=app), base_url="http://test"
        ) as client:
            response = await client.post(
                "/v1/audio/transcriptions",
                files={"file": ("audio.wav", b"fake-audio", "audio/wav")},
                data={"model": "faster-whisper-tiny"},
            )
        assert "application/json" in response.headers["content-type"]


class TestTranscribeVerboseJsonFormat:
    async def test_contains_all_fields(self) -> None:
        app = _make_app()
        async with httpx.AsyncClient(
            transport=httpx.ASGITransport(app=app), base_url="http://test"
        ) as client:
            response = await client.post(
                "/v1/audio/transcriptions",
                files={"file": ("audio.wav", b"fake-audio", "audio/wav")},
                data={"model": "faster-whisper-tiny", "response_format": "verbose_json"},
            )
        assert response.status_code == 200
        body = response.json()
        assert body["task"] == "transcribe"
        assert body["language"] == "pt"
        assert body["duration"] == 2.5
        assert body["text"] == "Ola, como posso ajudar?"
        assert len(body["segments"]) == 2
        assert len(body["words"]) == 4

    async def test_segments_have_required_fields(self) -> None:
        app = _make_app()
        async with httpx.AsyncClient(
            transport=httpx.ASGITransport(app=app), base_url="http://test"
        ) as client:
            response = await client.post(
                "/v1/audio/transcriptions",
                files={"file": ("audio.wav", b"fake-audio", "audio/wav")},
                data={"model": "faster-whisper-tiny", "response_format": "verbose_json"},
            )
        seg = response.json()["segments"][0]
        assert "id" in seg
        assert "start" in seg
        assert "end" in seg
        assert "text" in seg


class TestTranscribeTextFormat:
    async def test_returns_plain_text(self) -> None:
        app = _make_app()
        async with httpx.AsyncClient(
            transport=httpx.ASGITransport(app=app), base_url="http://test"
        ) as client:
            response = await client.post(
                "/v1/audio/transcriptions",
                files={"file": ("audio.wav", b"fake-audio", "audio/wav")},
                data={"model": "faster-whisper-tiny", "response_format": "text"},
            )
        assert response.status_code == 200
        assert response.text == "Ola, como posso ajudar?"

    async def test_content_type_is_plain_text(self) -> None:
        app = _make_app()
        async with httpx.AsyncClient(
            transport=httpx.ASGITransport(app=app), base_url="http://test"
        ) as client:
            response = await client.post(
                "/v1/audio/transcriptions",
                files={"file": ("audio.wav", b"fake-audio", "audio/wav")},
                data={"model": "faster-whisper-tiny", "response_format": "text"},
            )
        assert "text/plain" in response.headers["content-type"]


class TestTranscribeSrtFormat:
    async def test_returns_srt_content(self) -> None:
        app = _make_app()
        async with httpx.AsyncClient(
            transport=httpx.ASGITransport(app=app), base_url="http://test"
        ) as client:
            response = await client.post(
                "/v1/audio/transcriptions",
                files={"file": ("audio.wav", b"fake-audio", "audio/wav")},
                data={"model": "faster-whisper-tiny", "response_format": "srt"},
            )
        assert response.status_code == 200
        body = response.text
        # SRT tem numeros de segmento e timestamps com virgula
        assert "1\n" in body
        assert "00:00:00,000 --> 00:00:01,200" in body
        assert "Ola," in body


class TestTranscribeVttFormat:
    async def test_returns_vtt_content(self) -> None:
        app = _make_app()
        async with httpx.AsyncClient(
            transport=httpx.ASGITransport(app=app), base_url="http://test"
        ) as client:
            response = await client.post(
                "/v1/audio/transcriptions",
                files={"file": ("audio.wav", b"fake-audio", "audio/wav")},
                data={"model": "faster-whisper-tiny", "response_format": "vtt"},
            )
        assert response.status_code == 200
        body = response.text
        assert body.startswith("WEBVTT\n")
        # VTT usa ponto no timestamp
        assert "00:00:00.000 --> 00:00:01.200" in body


# --- Translations ---


class TestTranslateE2E:
    async def test_translate_returns_json(self) -> None:
        app = _make_app()
        async with httpx.AsyncClient(
            transport=httpx.ASGITransport(app=app), base_url="http://test"
        ) as client:
            response = await client.post(
                "/v1/audio/translations",
                files={"file": ("audio.wav", b"fake-audio", "audio/wav")},
                data={"model": "faster-whisper-tiny"},
            )
        assert response.status_code == 200
        assert "text" in response.json()

    async def test_translate_verbose_json_has_translate_task(self) -> None:
        app = _make_app()
        async with httpx.AsyncClient(
            transport=httpx.ASGITransport(app=app), base_url="http://test"
        ) as client:
            response = await client.post(
                "/v1/audio/translations",
                files={"file": ("audio.wav", b"fake-audio", "audio/wav")},
                data={"model": "faster-whisper-tiny", "response_format": "verbose_json"},
            )
        assert response.json()["task"] == "translate"


# --- Cenarios de erro ---


class TestErrorScenarios:
    async def test_model_not_found_returns_404(self) -> None:
        from macaw.exceptions import ModelNotFoundError

        scheduler = MagicMock()
        scheduler.transcribe = AsyncMock(side_effect=ModelNotFoundError("inexistente"))
        app = _make_app(scheduler=scheduler)

        async with httpx.AsyncClient(
            transport=httpx.ASGITransport(app=app), base_url="http://test"
        ) as client:
            response = await client.post(
                "/v1/audio/transcriptions",
                files={"file": ("audio.wav", b"fake-audio", "audio/wav")},
                data={"model": "inexistente"},
            )
        assert response.status_code == 404
        assert response.json()["error"]["code"] == "model_not_found"

    async def test_audio_format_error_returns_400(self) -> None:
        from macaw.exceptions import AudioFormatError

        scheduler = MagicMock()
        scheduler.transcribe = AsyncMock(side_effect=AudioFormatError("formato invalido"))
        app = _make_app(scheduler=scheduler)

        async with httpx.AsyncClient(
            transport=httpx.ASGITransport(app=app), base_url="http://test"
        ) as client:
            response = await client.post(
                "/v1/audio/transcriptions",
                files={"file": ("audio.wav", b"fake-audio", "audio/wav")},
                data={"model": "faster-whisper-tiny"},
            )
        assert response.status_code == 400
        assert response.json()["error"]["code"] == "invalid_audio"

    async def test_file_too_large_returns_413(self) -> None:
        app = _make_app()
        large_data = b"x" * (26 * 1024 * 1024)

        async with httpx.AsyncClient(
            transport=httpx.ASGITransport(app=app), base_url="http://test"
        ) as client:
            response = await client.post(
                "/v1/audio/transcriptions",
                files={"file": ("audio.wav", large_data, "audio/wav")},
                data={"model": "faster-whisper-tiny"},
            )
        assert response.status_code == 413
        assert response.json()["error"]["code"] == "file_too_large"

    async def test_worker_unavailable_returns_503(self) -> None:
        from macaw.exceptions import WorkerUnavailableError

        scheduler = MagicMock()
        scheduler.transcribe = AsyncMock(side_effect=WorkerUnavailableError("faster-whisper-tiny"))
        app = _make_app(scheduler=scheduler)

        async with httpx.AsyncClient(
            transport=httpx.ASGITransport(app=app), base_url="http://test"
        ) as client:
            response = await client.post(
                "/v1/audio/transcriptions",
                files={"file": ("audio.wav", b"fake-audio", "audio/wav")},
                data={"model": "faster-whisper-tiny"},
            )
        assert response.status_code == 503
        assert response.json()["error"]["code"] == "service_unavailable"
        assert response.headers.get("retry-after") == "5"

    async def test_invalid_response_format_returns_400(self) -> None:
        app = _make_app()

        async with httpx.AsyncClient(
            transport=httpx.ASGITransport(app=app), base_url="http://test"
        ) as client:
            response = await client.post(
                "/v1/audio/transcriptions",
                files={"file": ("audio.wav", b"fake-audio", "audio/wav")},
                data={"model": "faster-whisper-tiny", "response_format": "xml"},
            )
        assert response.status_code == 400
        body = response.json()
        assert body["error"]["code"] == "invalid_request"
        assert "xml" in body["error"]["message"]
        assert "json" in body["error"]["message"]

    async def test_unexpected_error_returns_500(self) -> None:
        scheduler = MagicMock()
        scheduler.transcribe = AsyncMock(side_effect=RuntimeError("kaboom"))
        app = _make_app(scheduler=scheduler)

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
        assert "kaboom" not in body["error"]["message"]


# --- Health endpoint ---


class TestHealthE2E:
    async def test_health_returns_ok(self) -> None:
        app = _make_app()
        async with httpx.AsyncClient(
            transport=httpx.ASGITransport(app=app), base_url="http://test"
        ) as client:
            response = await client.get("/health")
        assert response.status_code == 200
        assert response.json()["status"] == "ok"


# --- Validacao de request ---


class TestRequestValidation:
    async def test_missing_model_returns_422(self) -> None:
        app = _make_app()
        async with httpx.AsyncClient(
            transport=httpx.ASGITransport(app=app), base_url="http://test"
        ) as client:
            response = await client.post(
                "/v1/audio/transcriptions",
                files={"file": ("audio.wav", b"fake-audio", "audio/wav")},
            )
        assert response.status_code == 422

    async def test_missing_file_returns_422(self) -> None:
        app = _make_app()
        async with httpx.AsyncClient(
            transport=httpx.ASGITransport(app=app), base_url="http://test"
        ) as client:
            response = await client.post(
                "/v1/audio/transcriptions",
                data={"model": "faster-whisper-tiny"},
            )
        assert response.status_code == 422
