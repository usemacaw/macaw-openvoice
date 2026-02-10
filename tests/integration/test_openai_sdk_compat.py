"""Testes de compatibilidade com o SDK OpenAI Python.

Valida que o SDK `openai` funciona como cliente do Macaw sem modificacoes.
Usa servidor Macaw real (via uvicorn in-process) para testar o contrato completo.

Marcado como @pytest.mark.integration — requer `openai` instalado.
"""

from __future__ import annotations

import asyncio
import io
import threading
from unittest.mock import AsyncMock, MagicMock

import pytest
import uvicorn

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


def _make_app() -> object:
    registry = MagicMock()
    registry.has_model.return_value = True
    registry.get_manifest.return_value = MagicMock()

    scheduler = MagicMock()
    scheduler.transcribe = AsyncMock(return_value=_make_batch_result())

    return create_app(registry=registry, scheduler=scheduler)


class _ServerThread:
    """Roda uvicorn em thread separada para testes de integracao."""

    def __init__(self, app: object, host: str = "127.0.0.1", port: int = 18765) -> None:
        self.host = host
        self.port = port
        self.config = uvicorn.Config(app, host=host, port=port, log_level="error", ws="none")
        self.server = uvicorn.Server(self.config)
        self._thread: threading.Thread | None = None
        self._loop: asyncio.AbstractEventLoop | None = None

    def start(self) -> None:
        self._loop = asyncio.new_event_loop()

        def _run() -> None:
            self._loop.run_until_complete(self.server.serve())
            self._loop.close()

        self._thread = threading.Thread(target=_run)
        self._thread.daemon = True
        self._thread.start()

        # Esperar server ficar pronto
        import time

        for _ in range(50):
            if self.server.started:
                break
            time.sleep(0.1)

    def stop(self) -> None:
        self.server.should_exit = True
        if self._thread:
            self._thread.join(timeout=5)

    @property
    def base_url(self) -> str:
        return f"http://{self.host}:{self.port}"


@pytest.mark.integration
class TestOpenAISDKCompat:
    """Testes usando o SDK `openai` como cliente real."""

    @pytest.fixture(autouse=True)
    def _server(self) -> object:  # type: ignore[misc]
        app = _make_app()
        srv = _ServerThread(app)
        srv.start()
        self.server = srv
        yield
        srv.stop()

    def test_transcribe_returns_text(self) -> None:
        from openai import OpenAI

        client = OpenAI(
            base_url=f"{self.server.base_url}/v1",
            api_key="not-needed",
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
            base_url=f"{self.server.base_url}/v1",
            api_key="not-needed",
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
            base_url=f"{self.server.base_url}/v1",
            api_key="not-needed",
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
            base_url=f"{self.server.base_url}/v1",
            api_key="not-needed",
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
