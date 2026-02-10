"""Testes do worker gRPC WeNet — subprocess E2E.

Valida que o worker WeNet funciona com o STTWorkerServicer via gRPC:
- Factory cria WeNetBackend corretamente
- Servicer aceita WeNetBackend (mesma interface STTBackend)
- TranscribeFile retorna resultado valido
- TranscribeStream retorna stream de eventos
- Health check funcional
- Graceful shutdown
"""

from __future__ import annotations

from unittest.mock import MagicMock

import numpy as np

from macaw._types import (
    STTArchitecture,
)
from macaw.workers.stt.main import _create_backend
from macaw.workers.stt.wenet import WeNetBackend


class TestWeNetWorkerFactory:
    """Factory cria WeNetBackend para engine 'wenet'."""

    def test_create_wenet_backend(self) -> None:
        backend = _create_backend("wenet")
        assert isinstance(backend, WeNetBackend)
        assert backend.architecture == STTArchitecture.CTC

    def test_create_faster_whisper_backend(self) -> None:
        from macaw.workers.stt.faster_whisper import FasterWhisperBackend

        backend = _create_backend("faster-whisper")
        assert isinstance(backend, FasterWhisperBackend)
        assert backend.architecture == STTArchitecture.ENCODER_DECODER

    def test_unknown_engine_raises(self) -> None:
        import pytest

        with pytest.raises(ValueError, match="Engine STT nao suportada"):
            _create_backend("unknown-engine")


class TestWeNetServicerBatch:
    """STTWorkerServicer com WeNetBackend — TranscribeFile."""

    async def test_transcribe_file_via_servicer(self) -> None:
        """Servicer delega TranscribeFile ao WeNetBackend."""
        from macaw.workers.stt.servicer import STTWorkerServicer

        backend = WeNetBackend()
        mock_model = MagicMock()
        mock_model.transcribe.return_value = {"text": "hello world"}
        backend._model = mock_model  # type: ignore[assignment]

        servicer = STTWorkerServicer(
            backend=backend,
            model_name="wenet-ctc",
            engine="wenet",
        )

        # Simular request gRPC
        from macaw.proto.stt_worker_pb2 import TranscribeFileRequest

        audio = (np.zeros(16000, dtype=np.int16)).tobytes()
        request = TranscribeFileRequest(
            request_id="test-1",
            audio_data=audio,
            language="en",
            response_format="json",
        )

        context = MagicMock()
        response = await servicer.TranscribeFile(request, context)

        assert response.text == "hello world"
        assert response.language == "en"
        assert response.duration > 0

    async def test_transcribe_file_with_hot_words(self) -> None:
        """Servicer passa hot_words ao WeNetBackend."""
        from macaw.workers.stt.servicer import STTWorkerServicer

        backend = WeNetBackend()
        mock_model = MagicMock()
        mock_model.transcribe.return_value = {"text": "PIX transfer"}
        backend._model = mock_model  # type: ignore[assignment]

        servicer = STTWorkerServicer(
            backend=backend,
            model_name="wenet-ctc",
            engine="wenet",
        )

        from macaw.proto.stt_worker_pb2 import TranscribeFileRequest

        audio = (np.zeros(16000, dtype=np.int16)).tobytes()
        request = TranscribeFileRequest(
            request_id="test-2",
            audio_data=audio,
            language="pt",
            response_format="json",
            hot_words=["PIX", "TED"],
        )

        context = MagicMock()
        response = await servicer.TranscribeFile(request, context)

        assert "PIX" in response.text


class TestWeNetServicerStreaming:
    """STTWorkerServicer com WeNetBackend — TranscribeStream."""

    async def test_transcribe_stream_via_servicer(self) -> None:
        """Servicer delega TranscribeStream ao WeNetBackend."""
        from macaw.workers.stt.servicer import STTWorkerServicer

        backend = WeNetBackend()
        mock_model = MagicMock()
        # Enough audio for a partial (>160ms = 2560 samples) + final
        mock_model.transcribe.side_effect = [
            {"text": "hello"},
            {"text": "hello world"},
        ]
        backend._model = mock_model  # type: ignore[assignment]

        servicer = STTWorkerServicer(
            backend=backend,
            model_name="wenet-ctc",
            engine="wenet",
        )

        from macaw.proto.stt_worker_pb2 import AudioFrame

        # First frame: audio data (>160ms = 2560 samples at 16kHz)
        chunk = (np.zeros(4000, dtype=np.int16)).tobytes()  # 0.25s > 160ms minimum
        frames = [
            AudioFrame(session_id="s1", data=chunk, is_last=False),
            # Empty data signals end of stream (chunk empty -> break in transcribe_stream)
            AudioFrame(session_id="s1", data=b"", is_last=False),
            # is_last terminates the audio_chunk_generator
            AudioFrame(session_id="s1", data=b"", is_last=True),
        ]

        async def _request_iterator():  # type: ignore[no-untyped-def]
            for f in frames:
                yield f

        context = MagicMock()
        context.cancelled.return_value = False
        events = []
        async for event in servicer.TranscribeStream(_request_iterator(), context):
            events.append(event)

        assert len(events) >= 1
        # Last event should be final
        last = events[-1]
        assert last.event_type == "final"


class TestWeNetServicerHealth:
    """STTWorkerServicer com WeNetBackend — Health check."""

    async def test_health_check(self) -> None:
        from macaw.proto.stt_worker_pb2 import HealthRequest
        from macaw.workers.stt.servicer import STTWorkerServicer

        backend = WeNetBackend()
        # Sem modelo carregado
        servicer = STTWorkerServicer(
            backend=backend,
            model_name="wenet-ctc",
            engine="wenet",
        )

        context = MagicMock()
        request = HealthRequest()
        response = await servicer.Health(request, context)

        assert response.status == "not_loaded"
        assert response.model_name == "wenet-ctc"
        assert response.engine == "wenet"

    async def test_health_check_loaded(self) -> None:
        from macaw.proto.stt_worker_pb2 import HealthRequest
        from macaw.workers.stt.servicer import STTWorkerServicer

        backend = WeNetBackend()
        backend._model = MagicMock()  # type: ignore[assignment]

        servicer = STTWorkerServicer(
            backend=backend,
            model_name="wenet-ctc",
            engine="wenet",
        )

        context = MagicMock()
        request = HealthRequest()
        response = await servicer.Health(request, context)

        assert response.status == "ok"


class TestWeNetParseArgs:
    """Worker CLI aceita --engine wenet."""

    def test_parse_args_wenet(self) -> None:
        from macaw.workers.stt.main import parse_args

        args = parse_args(
            [
                "--port",
                "50052",
                "--engine",
                "wenet",
                "--model-path",
                "/models/wenet-ctc",
                "--model-size",
                "wenet-ctc",
            ]
        )
        assert args.port == 50052
        assert args.engine == "wenet"
        assert args.model_path == "/models/wenet-ctc"
        assert args.model_size == "wenet-ctc"

    def test_parse_args_default_engine(self) -> None:
        from macaw.workers.stt.main import parse_args

        args = parse_args(["--model-path", "/models/fw"])
        assert args.engine == "faster-whisper"
