"""Testes para o STTWorkerServicer gRPC."""

from __future__ import annotations

from typing import TYPE_CHECKING
from unittest.mock import AsyncMock, MagicMock

import grpc
import pytest

from macaw._types import (
    BatchResult,
    EngineCapabilities,
    SegmentDetail,
    STTArchitecture,
    WordTimestamp,
)
from macaw.proto.stt_worker_pb2 import (
    HealthRequest,
    TranscribeFileRequest,
)
from macaw.workers.stt.interface import STTBackend
from macaw.workers.stt.servicer import STTWorkerServicer

if TYPE_CHECKING:
    from collections.abc import AsyncIterator

    from macaw._types import TranscriptSegment


class MockBackend(STTBackend):
    """Backend mock para testes do servicer."""

    def __init__(self, transcribe_result: BatchResult | None = None) -> None:
        self._result = transcribe_result or BatchResult(
            text="mock transcript",
            language="en",
            duration=1.0,
            segments=(SegmentDetail(id=0, start=0.0, end=1.0, text="mock transcript"),),
            words=None,
        )
        self._health_status = "ok"
        self._loaded = True

    @property
    def architecture(self) -> STTArchitecture:
        return STTArchitecture.ENCODER_DECODER

    async def load(self, model_path: str, config: dict[str, object]) -> None:
        self._loaded = True

    async def capabilities(self) -> EngineCapabilities:
        return EngineCapabilities(supports_initial_prompt=True)

    async def transcribe_file(
        self,
        audio_data: bytes,
        language: str | None = None,
        initial_prompt: str | None = None,
        hot_words: list[str] | None = None,
        temperature: float = 0.0,
        word_timestamps: bool = False,
    ) -> BatchResult:
        return self._result

    async def transcribe_stream(
        self,
        audio_chunks: AsyncIterator[bytes],
        language: str | None = None,
        initial_prompt: str | None = None,
        hot_words: list[str] | None = None,
    ) -> AsyncIterator[TranscriptSegment]:
        raise NotImplementedError
        yield  # pragma: no cover

    async def unload(self) -> None:
        self._loaded = False

    async def health(self) -> dict[str, str]:
        return {"status": self._health_status}


def _make_context() -> MagicMock:
    """Cria mock de grpc.aio.ServicerContext."""
    ctx = MagicMock()
    ctx.abort = AsyncMock()
    ctx.cancelled = MagicMock(return_value=False)
    return ctx


class TestTranscribeFile:
    @pytest.fixture
    def servicer(self) -> STTWorkerServicer:
        return STTWorkerServicer(
            backend=MockBackend(),
            model_name="large-v3",
            engine="faster-whisper",
        )

    async def test_returns_text(self, servicer: STTWorkerServicer) -> None:
        request = TranscribeFileRequest(
            request_id="req-1",
            audio_data=b"\x00\x01" * 100,
            language="en",
        )
        ctx = _make_context()
        response = await servicer.TranscribeFile(request, ctx)
        assert response.text == "mock transcript"

    async def test_returns_language(self, servicer: STTWorkerServicer) -> None:
        request = TranscribeFileRequest(
            request_id="req-2",
            audio_data=b"\x00\x01" * 100,
        )
        ctx = _make_context()
        response = await servicer.TranscribeFile(request, ctx)
        assert response.language == "en"

    async def test_returns_segments(self, servicer: STTWorkerServicer) -> None:
        request = TranscribeFileRequest(
            request_id="req-3",
            audio_data=b"\x00\x01" * 100,
        )
        ctx = _make_context()
        response = await servicer.TranscribeFile(request, ctx)
        assert len(response.segments) == 1
        assert response.segments[0].text == "mock transcript"

    async def test_with_words(self) -> None:
        result = BatchResult(
            text="hello world",
            language="en",
            duration=1.0,
            segments=(SegmentDetail(id=0, start=0.0, end=1.0, text="hello world"),),
            words=(
                WordTimestamp(word="hello", start=0.0, end=0.5, probability=0.9),
                WordTimestamp(word="world", start=0.5, end=1.0, probability=0.8),
            ),
        )
        servicer = STTWorkerServicer(
            backend=MockBackend(transcribe_result=result),
            model_name="large-v3",
            engine="faster-whisper",
        )
        request = TranscribeFileRequest(
            request_id="req-4",
            audio_data=b"\x00\x01" * 100,
            timestamp_granularities=["word"],
        )
        ctx = _make_context()
        response = await servicer.TranscribeFile(request, ctx)
        assert len(response.words) == 2

    async def test_backend_error_aborts(self) -> None:
        backend = MockBackend()
        backend.transcribe_file = AsyncMock(side_effect=RuntimeError("GPU OOM"))  # type: ignore[method-assign]
        servicer = STTWorkerServicer(
            backend=backend,
            model_name="large-v3",
            engine="faster-whisper",
        )
        request = TranscribeFileRequest(
            request_id="req-5",
            audio_data=b"\x00\x01" * 100,
        )
        ctx = _make_context()
        # context.abort is async and in real gRPC raises an exception.
        # We simulate that by making abort raise to stop execution.
        ctx.abort = AsyncMock(
            side_effect=grpc.aio.AbortError(  # type: ignore[attr-defined]
                grpc.StatusCode.INTERNAL, "GPU OOM"
            )
        )
        with pytest.raises(grpc.aio.AbortError):  # type: ignore[attr-defined]
            await servicer.TranscribeFile(request, ctx)
        ctx.abort.assert_called_once_with(grpc.StatusCode.INTERNAL, "GPU OOM")

    async def test_passes_params_to_backend(self) -> None:
        backend = MockBackend()
        backend.transcribe_file = AsyncMock(return_value=backend._result)  # type: ignore[method-assign]
        servicer = STTWorkerServicer(
            backend=backend,
            model_name="large-v3",
            engine="faster-whisper",
        )
        request = TranscribeFileRequest(
            request_id="req-6",
            audio_data=b"\x00\x01" * 100,
            language="pt",
            initial_prompt="Termos: PIX",
            hot_words=["PIX"],
            temperature=0.3,
            timestamp_granularities=["word"],
        )
        ctx = _make_context()
        await servicer.TranscribeFile(request, ctx)
        backend.transcribe_file.assert_called_once()
        call_kwargs = backend.transcribe_file.call_args
        assert call_kwargs.kwargs["language"] == "pt"
        assert call_kwargs.kwargs["initial_prompt"] == "Termos: PIX"
        assert call_kwargs.kwargs["hot_words"] == ["PIX"]
        assert call_kwargs.kwargs["word_timestamps"] is True


class TestTranscribeStream:
    async def test_empty_stream_produces_no_events(self) -> None:
        servicer = STTWorkerServicer(
            backend=MockBackend(),
            model_name="large-v3",
            engine="faster-whisper",
        )
        ctx = _make_context()

        async def empty_iterator() -> AsyncIterator[object]:
            return
            yield  # pragma: no cover — make it an async generator

        events = []
        async for event in servicer.TranscribeStream(empty_iterator(), ctx):
            events.append(event)
        assert len(events) == 0


class TestCancel:
    async def test_returns_acknowledged(self) -> None:
        servicer = STTWorkerServicer(
            backend=MockBackend(),
            model_name="large-v3",
            engine="faster-whisper",
        )
        ctx = _make_context()
        from macaw.proto.stt_worker_pb2 import CancelRequest

        response = await servicer.Cancel(CancelRequest(request_id="req-1"), ctx)
        assert response.acknowledged is True
        ctx.abort.assert_not_called()


class TestHealth:
    async def test_returns_ok(self) -> None:
        servicer = STTWorkerServicer(
            backend=MockBackend(),
            model_name="large-v3",
            engine="faster-whisper",
        )
        ctx = _make_context()
        response = await servicer.Health(HealthRequest(), ctx)
        assert response.status == "ok"

    async def test_returns_not_loaded(self) -> None:
        backend = MockBackend()
        backend._health_status = "not_loaded"
        servicer = STTWorkerServicer(
            backend=backend,
            model_name="large-v3",
            engine="faster-whisper",
        )
        ctx = _make_context()
        response = await servicer.Health(HealthRequest(), ctx)
        assert response.status == "not_loaded"

    async def test_returns_model_name_and_engine(self) -> None:
        servicer = STTWorkerServicer(
            backend=MockBackend(),
            model_name="large-v3",
            engine="faster-whisper",
        )
        ctx = _make_context()
        response = await servicer.Health(HealthRequest(), ctx)
        assert response.model_name == "large-v3"
        assert response.engine == "faster-whisper"
