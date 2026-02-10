"""Testes para STTWorkerServicer.TranscribeStream."""

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
    TranscriptSegment,
    WordTimestamp,
)
from macaw.proto.stt_worker_pb2 import AudioFrame
from macaw.workers.stt.interface import STTBackend
from macaw.workers.stt.servicer import STTWorkerServicer

if TYPE_CHECKING:
    from collections.abc import AsyncIterator


class StreamingMockBackend(STTBackend):
    """Backend mock com suporte a transcribe_stream."""

    def __init__(
        self,
        stream_segments: tuple[TranscriptSegment, ...] = (),
    ) -> None:
        self._stream_segments = stream_segments
        self._received_chunks: list[bytes] = []
        self._stream_kwargs: dict[str, object] = {}

    @property
    def architecture(self) -> STTArchitecture:
        return STTArchitecture.ENCODER_DECODER

    async def load(self, model_path: str, config: dict[str, object]) -> None:
        pass

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
        return BatchResult(
            text="",
            language="en",
            duration=0.0,
            segments=(SegmentDetail(id=0, start=0.0, end=0.0, text=""),),
        )

    async def transcribe_stream(
        self,
        audio_chunks: AsyncIterator[bytes],
        language: str | None = None,
        initial_prompt: str | None = None,
        hot_words: list[str] | None = None,
    ) -> AsyncIterator[TranscriptSegment]:
        self._stream_kwargs = {
            "language": language,
            "initial_prompt": initial_prompt,
            "hot_words": hot_words,
        }
        # Consume all chunks from the generator
        async for chunk in audio_chunks:
            self._received_chunks.append(chunk)
        # Yield pre-configured segments
        for segment in self._stream_segments:
            yield segment

    async def unload(self) -> None:
        pass

    async def health(self) -> dict[str, str]:
        return {"status": "ok"}


def _make_context(*, cancelled: bool = False) -> MagicMock:
    """Cria mock de grpc.aio.ServicerContext."""
    ctx = MagicMock()
    ctx.abort = AsyncMock()
    ctx.cancelled = MagicMock(return_value=cancelled)
    return ctx


async def _make_frame_iterator(
    frames: list[AudioFrame],
) -> AsyncIterator[AudioFrame]:
    """Cria async iterator a partir de lista de AudioFrames."""
    for frame in frames:
        yield frame


async def _collect_events(
    async_gen: AsyncIterator[object],
) -> list[object]:
    """Coleta todos os eventos de um async generator."""
    events = []
    async for event in async_gen:
        events.append(event)
    return events


class TestTranscribeStream:
    @pytest.fixture
    def segments(self) -> tuple[TranscriptSegment, ...]:
        return (
            TranscriptSegment(
                text="Ola como",
                is_final=False,
                segment_id=0,
                start_ms=100,
                end_ms=500,
                language="pt",
                confidence=0.8,
            ),
            TranscriptSegment(
                text="Ola, como posso ajudar?",
                is_final=True,
                segment_id=0,
                start_ms=100,
                end_ms=2000,
                language="pt",
                confidence=0.95,
                words=(
                    WordTimestamp(word="Ola", start=0.1, end=0.3, probability=0.9),
                    WordTimestamp(word="como", start=0.3, end=0.5, probability=0.85),
                ),
            ),
        )

    async def test_basic_stream_returns_events(
        self, segments: tuple[TranscriptSegment, ...]
    ) -> None:
        backend = StreamingMockBackend(stream_segments=segments)
        servicer = STTWorkerServicer(
            backend=backend, model_name="large-v3", engine="faster-whisper"
        )

        frames = [
            AudioFrame(
                session_id="sess_1",
                data=b"\x00\x01" * 100,
                is_last=False,
                initial_prompt="Termos: PIX",
                hot_words=["PIX", "TED"],
            ),
            AudioFrame(
                session_id="sess_1",
                data=b"\x00\x02" * 100,
                is_last=False,
            ),
            AudioFrame(
                session_id="sess_1",
                data=b"",
                is_last=True,
            ),
        ]

        ctx = _make_context()
        events = await _collect_events(
            servicer.TranscribeStream(_make_frame_iterator(frames), ctx)
        )

        assert len(events) == 2
        assert events[0].event_type == "partial"
        assert events[0].text == "Ola como"
        assert events[0].session_id == "sess_1"
        assert events[1].event_type == "final"
        assert events[1].text == "Ola, como posso ajudar?"
        assert len(events[1].words) == 2

    async def test_is_last_stops_audio_generation(
        self, segments: tuple[TranscriptSegment, ...]
    ) -> None:
        backend = StreamingMockBackend(stream_segments=segments)
        servicer = STTWorkerServicer(
            backend=backend, model_name="large-v3", engine="faster-whisper"
        )

        frames = [
            AudioFrame(
                session_id="sess_2",
                data=b"\x00\x01" * 50,
                is_last=False,
            ),
            AudioFrame(
                session_id="sess_2",
                data=b"",
                is_last=True,
            ),
            # This frame should never be consumed because is_last was sent
            AudioFrame(
                session_id="sess_2",
                data=b"\xff\xff" * 50,
                is_last=False,
            ),
        ]

        ctx = _make_context()
        await _collect_events(servicer.TranscribeStream(_make_frame_iterator(frames), ctx))

        # Backend should only receive the first frame's data
        assert len(backend._received_chunks) == 1
        assert backend._received_chunks[0] == b"\x00\x01" * 50

    async def test_first_frame_metadata_passed_to_backend(self) -> None:
        backend = StreamingMockBackend(stream_segments=())
        servicer = STTWorkerServicer(
            backend=backend, model_name="large-v3", engine="faster-whisper"
        )

        frames = [
            AudioFrame(
                session_id="sess_3",
                data=b"\x00\x01" * 50,
                is_last=False,
                initial_prompt="Contexto anterior",
                hot_words=["PIX", "Selic"],
            ),
            AudioFrame(session_id="sess_3", data=b"", is_last=True),
        ]

        ctx = _make_context()
        await _collect_events(servicer.TranscribeStream(_make_frame_iterator(frames), ctx))

        assert backend._stream_kwargs["initial_prompt"] == "Contexto anterior"
        assert backend._stream_kwargs["hot_words"] == ["PIX", "Selic"]
        assert backend._stream_kwargs["language"] is None

    async def test_empty_prompt_becomes_none(self) -> None:
        backend = StreamingMockBackend(stream_segments=())
        servicer = STTWorkerServicer(
            backend=backend, model_name="large-v3", engine="faster-whisper"
        )

        frames = [
            AudioFrame(
                session_id="sess_4",
                data=b"\x00\x01" * 50,
                is_last=False,
                initial_prompt="",  # empty string from proto default
            ),
            AudioFrame(session_id="sess_4", data=b"", is_last=True),
        ]

        ctx = _make_context()
        await _collect_events(servicer.TranscribeStream(_make_frame_iterator(frames), ctx))

        assert backend._stream_kwargs["initial_prompt"] is None
        assert backend._stream_kwargs["hot_words"] is None

    async def test_empty_stream_is_last_immediately(self) -> None:
        backend = StreamingMockBackend(stream_segments=())
        servicer = STTWorkerServicer(
            backend=backend, model_name="large-v3", engine="faster-whisper"
        )

        frames = [
            AudioFrame(session_id="sess_5", data=b"", is_last=True),
        ]

        ctx = _make_context()
        events = await _collect_events(
            servicer.TranscribeStream(_make_frame_iterator(frames), ctx)
        )

        assert len(events) == 0
        assert len(backend._received_chunks) == 0

    async def test_cancellation_stops_yielding_events(self) -> None:
        # Create segments that would be yielded
        many_segments = tuple(
            TranscriptSegment(
                text=f"segment {i}",
                is_final=i == 4,
                segment_id=i,
            )
            for i in range(5)
        )
        backend = StreamingMockBackend(stream_segments=many_segments)
        servicer = STTWorkerServicer(
            backend=backend, model_name="large-v3", engine="faster-whisper"
        )

        frames = [
            AudioFrame(session_id="sess_6", data=b"\x00" * 50, is_last=False),
            AudioFrame(session_id="sess_6", data=b"", is_last=True),
        ]

        # Context starts not cancelled, but after first event it returns cancelled
        ctx = _make_context()
        call_count = 0

        def side_effect_cancelled() -> bool:
            nonlocal call_count
            call_count += 1
            # Allow first event through, cancel on second check
            return call_count > 2

        ctx.cancelled = MagicMock(side_effect=side_effect_cancelled)

        events = await _collect_events(
            servicer.TranscribeStream(_make_frame_iterator(frames), ctx)
        )

        # Should have fewer events than total segments
        assert len(events) < len(many_segments)

    async def test_backend_error_aborts_with_internal(self) -> None:
        backend = StreamingMockBackend(stream_segments=())

        # Make transcribe_stream raise
        async def failing_stream(
            audio_chunks: AsyncIterator[bytes],
            language: str | None = None,
            initial_prompt: str | None = None,
            hot_words: list[str] | None = None,
        ) -> AsyncIterator[TranscriptSegment]:
            async for _ in audio_chunks:
                pass
            msg = "GPU OOM during streaming"
            raise RuntimeError(msg)
            yield  # pragma: no cover — make it an async generator

        backend.transcribe_stream = failing_stream  # type: ignore[assignment]

        servicer = STTWorkerServicer(
            backend=backend, model_name="large-v3", engine="faster-whisper"
        )

        frames = [
            AudioFrame(session_id="sess_7", data=b"\x00" * 50, is_last=False),
            AudioFrame(session_id="sess_7", data=b"", is_last=True),
        ]

        ctx = _make_context()
        ctx.abort = AsyncMock(
            side_effect=grpc.aio.AbortError(  # type: ignore[attr-defined]
                grpc.StatusCode.INTERNAL, "GPU OOM during streaming"
            )
        )

        with pytest.raises(grpc.aio.AbortError):  # type: ignore[attr-defined]
            await _collect_events(servicer.TranscribeStream(_make_frame_iterator(frames), ctx))

        ctx.abort.assert_called_once_with(grpc.StatusCode.INTERNAL, "GPU OOM during streaming")

    async def test_no_frames_produces_no_events(self) -> None:
        backend = StreamingMockBackend(stream_segments=())
        servicer = STTWorkerServicer(
            backend=backend, model_name="large-v3", engine="faster-whisper"
        )

        ctx = _make_context()
        events = await _collect_events(servicer.TranscribeStream(_make_frame_iterator([]), ctx))

        assert len(events) == 0

    async def test_session_id_propagated_to_events(self) -> None:
        segments = (TranscriptSegment(text="hello", is_final=True, segment_id=0),)
        backend = StreamingMockBackend(stream_segments=segments)
        servicer = STTWorkerServicer(
            backend=backend, model_name="large-v3", engine="faster-whisper"
        )

        frames = [
            AudioFrame(
                session_id="sess_unique_id_123",
                data=b"\x00" * 50,
                is_last=False,
            ),
            AudioFrame(session_id="sess_unique_id_123", data=b"", is_last=True),
        ]

        ctx = _make_context()
        events = await _collect_events(
            servicer.TranscribeStream(_make_frame_iterator(frames), ctx)
        )

        assert len(events) == 1
        assert events[0].session_id == "sess_unique_id_123"

    async def test_multiple_audio_frames_forwarded_to_backend(self) -> None:
        backend = StreamingMockBackend(stream_segments=())
        servicer = STTWorkerServicer(
            backend=backend, model_name="large-v3", engine="faster-whisper"
        )

        chunk_a = b"\x01\x02" * 50
        chunk_b = b"\x03\x04" * 60
        chunk_c = b"\x05\x06" * 70

        frames = [
            AudioFrame(session_id="sess_8", data=chunk_a, is_last=False),
            AudioFrame(session_id="sess_8", data=chunk_b, is_last=False),
            AudioFrame(session_id="sess_8", data=chunk_c, is_last=False),
            AudioFrame(session_id="sess_8", data=b"", is_last=True),
        ]

        ctx = _make_context()
        await _collect_events(servicer.TranscribeStream(_make_frame_iterator(frames), ctx))

        assert len(backend._received_chunks) == 3
        assert backend._received_chunks[0] == chunk_a
        assert backend._received_chunks[1] == chunk_b
        assert backend._received_chunks[2] == chunk_c
