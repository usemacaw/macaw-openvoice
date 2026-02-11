"""Testes de compatibilidade com o SDK OpenAI Python.

Valida que o SDK `openai` funciona como cliente do Macaw sem modificacoes.
Usa servidor Macaw real (via uvicorn in-process) para testar o contrato completo.

Marcado como @pytest.mark.integration — requer `openai` instalado.
"""

from __future__ import annotations

import io
from unittest.mock import AsyncMock, MagicMock

import anyio
import httpx
import pytest

from fastapi import FastAPI

from macaw._types import BatchResult, SegmentDetail, WordTimestamp
from macaw.server.app import create_app


def _make_batch_result() -> BatchResult:
    return BatchResult(
        text="Hello, how can I help you?",
        language="en",
        duration=2.0,
        segments=(SegmentDetail(id=0, start=0.0, end=2.0, text="Hello, how can I help you?"),),
        words=(
            WordTimestamp(word="Hello", start=0.0, end=0.3),
            WordTimestamp(word="how", start=0.4, end=0.6),
            WordTimestamp(word="can", start=0.7, end=0.8),
            WordTimestamp(word="I", start=0.9, end=0.95),
            WordTimestamp(word="help", start=1.0, end=1.3),
            WordTimestamp(word="you", start=1.4, end=2.0),
        ),
    )


def _make_app() -> FastAPI:
    registry = MagicMock()
    registry.has_model.return_value = True
    registry.get_manifest.return_value = MagicMock()

    scheduler = MagicMock()
    scheduler.transcribe = AsyncMock(return_value=_make_batch_result())

    return create_app(registry=registry, scheduler=scheduler)


class _SyncASGITransport(httpx.BaseTransport):
    """Bridge httpx sync client with FastAPI ASGI app."""

    def __init__(self, app: FastAPI) -> None:
        self._transport = httpx.ASGITransport(app=app)

    def handle_request(self, request: httpx.Request) -> httpx.Response:
        async def _send() -> tuple[int, httpx.Headers, bytes, dict[str, object]]:
            response = await self._transport.handle_async_request(request)
            body = await response.aread()
            await response.aclose()
            return response.status_code, response.headers, body, response.extensions

        status_code, headers, content, extensions = anyio.run(_send)
        return httpx.Response(
            status_code=status_code,
            headers=headers,
            content=content,
            extensions=extensions,
            request=request,
        )

    def close(self) -> None:
        async def _close() -> None:
            await self._transport.aclose()

        anyio.run(_close)


@pytest.mark.integration
class TestOpenAISDKCompat:
    """Testes usando o SDK `openai` como cliente real."""

    @pytest.fixture(autouse=True)
    def _client(self) -> object:  # type: ignore[misc]
        app = _make_app()
        transport = _SyncASGITransport(app)
        http_client = httpx.Client(transport=transport, base_url="http://macaw.test")
        self._http_client = http_client
        self._base_url = "http://macaw.test/v1"
        yield
        http_client.close()

    def test_transcribe_returns_text(self) -> None:
        from openai import OpenAI

        client = OpenAI(
            base_url=self._base_url,
            api_key="not-needed",
            http_client=self._http_client,
        )

        audio_file = io.BytesIO(b"fake-audio-data")
        audio_file.name = "audio.wav"

        result = client.audio.transcriptions.create(
            model="faster-whisper-tiny",
            file=audio_file,
        )

        assert result.text == "Hello, how can I help you?"

    def test_transcribe_verbose_json(self) -> None:
        from openai import OpenAI

        client = OpenAI(
            base_url=self._base_url,
            api_key="not-needed",
            http_client=self._http_client,
        )

        audio_file = io.BytesIO(b"fake-audio-data")
        audio_file.name = "audio.wav"

        result = client.audio.transcriptions.create(
            model="faster-whisper-tiny",
            file=audio_file,
            response_format="verbose_json",
        )

        assert result.text == "Hello, how can I help you?"
        assert result.language == "en"
        assert result.duration is not None
        assert result.segments is not None
        assert len(result.segments) > 0

    def test_translate_returns_text(self) -> None:
        from openai import OpenAI

        client = OpenAI(
            base_url=self._base_url,
            api_key="not-needed",
            http_client=self._http_client,
        )

        audio_file = io.BytesIO(b"fake-audio-data")
        audio_file.name = "audio.wav"

        result = client.audio.translations.create(
            model="faster-whisper-tiny",
            file=audio_file,
        )

        assert result.text == "Hello, how can I help you?"

    def test_transcribe_text_format(self) -> None:
        from openai import OpenAI

        client = OpenAI(
            base_url=self._base_url,
            api_key="not-needed",
            http_client=self._http_client,
        )

        audio_file = io.BytesIO(b"fake-audio-data")
        audio_file.name = "audio.wav"

        result = client.audio.transcriptions.create(
            model="faster-whisper-tiny",
            file=audio_file,
            response_format="text",
        )

        # text format retorna string diretamente
        assert "Hello" in str(result)
