"""Testes unitarios para StreamingGRPCClient e StreamHandle.

Todos os testes usam mocks para gRPC — nenhum servidor real e iniciado.
"""

from __future__ import annotations

from typing import TYPE_CHECKING
from unittest.mock import AsyncMock, MagicMock, patch

import grpc
import grpc.aio
import pytest

from macaw.exceptions import WorkerCrashError, WorkerTimeoutError
from macaw.proto.stt_worker_pb2 import AudioFrame, TranscriptEvent, Word
from macaw.scheduler.streaming import (
    StreamHandle,
    StreamingGRPCClient,
    _proto_event_to_transcript_segment,
)

if TYPE_CHECKING:
    from collections.abc import AsyncIterator, Sequence

    from macaw._types import TranscriptSegment


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


class AsyncIterableFromList:
    """Wrapper que torna uma lista iteravel com `async for`."""

    def __init__(self, items: Sequence[object]) -> None:
        self._items = items
        self._index = 0

    def __aiter__(self) -> AsyncIterableFromList:
        return self

    async def __anext__(self) -> object:
        if self._index >= len(self._items):
            raise StopAsyncIteration
        item = self._items[self._index]
        self._index += 1
        return item


def _make_async_iterable_call(
    mock_call: AsyncMock,
    events: Sequence[object],
) -> None:
    """Configura mock_call para ser async-iteravel sobre a lista de eventos."""
    ait = AsyncIterableFromList(events)
    mock_call.__aiter__ = MagicMock(return_value=ait)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def mock_call() -> AsyncMock:
    """Cria mock de grpc.aio.StreamStreamCall."""
    call = AsyncMock(spec_set=["write", "done_writing", "cancel", "__aiter__", "__anext__"])
    call.write = AsyncMock()
    call.done_writing = AsyncMock()
    call.cancel = MagicMock()
    # Default: iterador vazio
    call.__aiter__ = MagicMock(return_value=call)
    call.__anext__ = AsyncMock(side_effect=StopAsyncIteration)
    return call


@pytest.fixture
def stream_handle(mock_call: AsyncMock) -> StreamHandle:
    """Cria StreamHandle com mock call."""
    return StreamHandle(session_id="sess_test_001", call=mock_call)


# ---------------------------------------------------------------------------
# StreamHandle — send_frame
# ---------------------------------------------------------------------------


class TestStreamHandleSendFrame:
    """Testes de StreamHandle.send_frame()."""

    async def test_send_frame_creates_correct_audio_frame(
        self,
        stream_handle: StreamHandle,
        mock_call: AsyncMock,
    ) -> None:
        """send_frame deve criar AudioFrame com campos corretos e chamar write."""
        pcm_data = b"\x00\x01" * 160  # 20ms a 16kHz

        await stream_handle.send_frame(
            pcm_data=pcm_data,
            initial_prompt="Contexto anterior",
            hot_words=["PIX", "TED"],
        )

        mock_call.write.assert_called_once()
        frame: AudioFrame = mock_call.write.call_args[0][0]
        assert frame.session_id == "sess_test_001"
        assert frame.data == pcm_data
        assert frame.is_last is False
        assert frame.initial_prompt == "Contexto anterior"
        assert list(frame.hot_words) == ["PIX", "TED"]

    async def test_send_frame_defaults_empty_prompt_and_hot_words(
        self,
        stream_handle: StreamHandle,
        mock_call: AsyncMock,
    ) -> None:
        """send_frame sem prompt e hot_words deve usar defaults vazios."""
        await stream_handle.send_frame(pcm_data=b"\x00" * 320)

        frame: AudioFrame = mock_call.write.call_args[0][0]
        assert frame.initial_prompt == ""
        assert list(frame.hot_words) == []

    async def test_send_frame_on_closed_stream_raises_worker_crash(
        self,
        stream_handle: StreamHandle,
    ) -> None:
        """send_frame em stream ja fechado deve levantar WorkerCrashError."""
        await stream_handle.close()
        assert stream_handle.is_closed is True

        with pytest.raises(WorkerCrashError):
            await stream_handle.send_frame(pcm_data=b"\x00" * 320)

    async def test_send_frame_grpc_error_raises_worker_crash_and_marks_closed(
        self,
        stream_handle: StreamHandle,
        mock_call: AsyncMock,
    ) -> None:
        """Erro gRPC durante write deve levantar WorkerCrashError e marcar closed."""
        mock_call.write.side_effect = grpc.aio.AioRpcError(
            code=grpc.StatusCode.UNAVAILABLE,
            initial_metadata=grpc.aio.Metadata(),
            trailing_metadata=grpc.aio.Metadata(),
            details="Connection refused",
            debug_error_string=None,
        )

        with pytest.raises(WorkerCrashError):
            await stream_handle.send_frame(pcm_data=b"\x00" * 320)

        assert stream_handle.is_closed is True


# ---------------------------------------------------------------------------
# StreamHandle — receive_events
# ---------------------------------------------------------------------------


class TestStreamHandleReceiveEvents:
    """Testes de StreamHandle.receive_events()."""

    async def test_receive_events_converts_proto_to_transcript_segment(
        self,
        stream_handle: StreamHandle,
        mock_call: AsyncMock,
    ) -> None:
        """receive_events deve converter TranscriptEvent proto em TranscriptSegment."""
        event = TranscriptEvent(
            session_id="sess_test_001",
            event_type="partial",
            text="ola como",
            segment_id=0,
            start_ms=1500,
            end_ms=2000,
            language="pt",
            confidence=0.85,
        )
        _make_async_iterable_call(mock_call, [event])

        segments: list[TranscriptSegment] = []
        async for seg in stream_handle.receive_events():
            segments.append(seg)

        assert len(segments) == 1
        assert segments[0].text == "ola como"
        assert segments[0].is_final is False
        assert segments[0].segment_id == 0
        assert segments[0].start_ms == 1500
        assert segments[0].end_ms == 2000
        assert segments[0].language == "pt"
        assert segments[0].confidence == pytest.approx(0.85)

    async def test_receive_events_final_event(
        self,
        stream_handle: StreamHandle,
        mock_call: AsyncMock,
    ) -> None:
        """receive_events com event_type 'final' gera TranscriptSegment is_final=True."""
        event = TranscriptEvent(
            session_id="sess_test_001",
            event_type="final",
            text="ola como posso ajudar",
            segment_id=0,
            start_ms=1500,
            end_ms=4000,
            language="pt",
            confidence=0.95,
            words=[
                Word(word="ola", start=1.5, end=2.0, probability=0.99),
                Word(word="como", start=2.1, end=2.4, probability=0.97),
            ],
        )
        _make_async_iterable_call(mock_call, [event])

        segments: list[TranscriptSegment] = []
        async for seg in stream_handle.receive_events():
            segments.append(seg)

        assert len(segments) == 1
        seg = segments[0]
        assert seg.is_final is True
        assert seg.text == "ola como posso ajudar"
        assert seg.words is not None
        assert len(seg.words) == 2
        assert seg.words[0].word == "ola"
        assert seg.words[0].probability == pytest.approx(0.99)
        assert seg.words[1].word == "como"

    async def test_receive_events_multiple_events(
        self,
        stream_handle: StreamHandle,
        mock_call: AsyncMock,
    ) -> None:
        """receive_events deve iterar sobre multiplos eventos."""
        events = [
            TranscriptEvent(event_type="partial", text="ola", segment_id=0),
            TranscriptEvent(event_type="partial", text="ola como", segment_id=0),
            TranscriptEvent(event_type="final", text="ola como vai", segment_id=0),
        ]
        _make_async_iterable_call(mock_call, events)

        segments: list[TranscriptSegment] = []
        async for seg in stream_handle.receive_events():
            segments.append(seg)

        assert len(segments) == 3
        assert segments[0].is_final is False
        assert segments[1].is_final is False
        assert segments[2].is_final is True

    async def test_receive_events_grpc_error_raises_worker_crash(
        self,
        stream_handle: StreamHandle,
        mock_call: AsyncMock,
    ) -> None:
        """Erro gRPC durante iteracao deve levantar WorkerCrashError."""

        # Simula async generator que levanta erro antes de yield
        async def _error_iter() -> AsyncIterator[TranscriptEvent]:
            raise grpc.aio.AioRpcError(
                code=grpc.StatusCode.UNAVAILABLE,
                initial_metadata=grpc.aio.Metadata(),
                trailing_metadata=grpc.aio.Metadata(),
                details="Worker crashed",
                debug_error_string=None,
            )
            yield TranscriptEvent()  # pragma: no cover — unreachable, needed for generator type

        mock_call.__aiter__ = MagicMock(return_value=_error_iter())

        with pytest.raises(WorkerCrashError):
            async for _ in stream_handle.receive_events():
                pass  # pragma: no cover

        assert stream_handle.is_closed is True

    async def test_receive_events_deadline_exceeded_raises_timeout(
        self,
        stream_handle: StreamHandle,
        mock_call: AsyncMock,
    ) -> None:
        """DEADLINE_EXCEEDED durante iteracao deve levantar WorkerTimeoutError."""

        # Simula async generator que levanta DEADLINE_EXCEEDED antes de yield
        async def _timeout_iter() -> AsyncIterator[TranscriptEvent]:
            raise grpc.aio.AioRpcError(
                code=grpc.StatusCode.DEADLINE_EXCEEDED,
                initial_metadata=grpc.aio.Metadata(),
                trailing_metadata=grpc.aio.Metadata(),
                details="Deadline exceeded",
                debug_error_string=None,
            )
            yield TranscriptEvent()  # pragma: no cover — unreachable, needed for generator type

        mock_call.__aiter__ = MagicMock(return_value=_timeout_iter())

        with pytest.raises(WorkerTimeoutError):
            async for _ in stream_handle.receive_events():
                pass  # pragma: no cover

        assert stream_handle.is_closed is True

    async def test_receive_events_empty_stream(
        self,
        stream_handle: StreamHandle,
        mock_call: AsyncMock,
    ) -> None:
        """Stream vazio deve completar sem eventos."""
        _make_async_iterable_call(mock_call, [])

        segments: list[TranscriptSegment] = []
        async for seg in stream_handle.receive_events():
            segments.append(seg)  # pragma: no cover

        assert len(segments) == 0


# ---------------------------------------------------------------------------
# StreamHandle — close
# ---------------------------------------------------------------------------


class TestStreamHandleClose:
    """Testes de StreamHandle.close()."""

    async def test_close_sends_is_last_and_done_writing(
        self,
        stream_handle: StreamHandle,
        mock_call: AsyncMock,
    ) -> None:
        """close() deve enviar frame com is_last=True e chamar done_writing."""
        await stream_handle.close()

        assert stream_handle.is_closed is True

        # Verificar que write foi chamado com is_last=True
        mock_call.write.assert_called_once()
        frame: AudioFrame = mock_call.write.call_args[0][0]
        assert frame.is_last is True
        assert frame.data == b""
        assert frame.session_id == "sess_test_001"

        mock_call.done_writing.assert_called_once()

    async def test_close_idempotent_on_already_closed(
        self,
        stream_handle: StreamHandle,
        mock_call: AsyncMock,
    ) -> None:
        """close() em stream ja fechado deve ser no-op."""
        await stream_handle.close()
        mock_call.write.reset_mock()
        mock_call.done_writing.reset_mock()

        # Segunda chamada nao deve fazer nada
        await stream_handle.close()

        mock_call.write.assert_not_called()
        mock_call.done_writing.assert_not_called()

    async def test_close_handles_grpc_error_gracefully(
        self,
        stream_handle: StreamHandle,
        mock_call: AsyncMock,
    ) -> None:
        """close() deve tratar erro gRPC sem propagar excecao."""
        mock_call.write.side_effect = grpc.aio.AioRpcError(
            code=grpc.StatusCode.CANCELLED,
            initial_metadata=grpc.aio.Metadata(),
            trailing_metadata=grpc.aio.Metadata(),
            details="Already cancelled",
            debug_error_string=None,
        )

        # Nao deve levantar excecao
        await stream_handle.close()

        assert stream_handle.is_closed is True


# ---------------------------------------------------------------------------
# StreamHandle — cancel
# ---------------------------------------------------------------------------


class TestStreamHandleCancel:
    """Testes de StreamHandle.cancel()."""

    async def test_cancel_calls_cancel_and_marks_closed(
        self,
        stream_handle: StreamHandle,
        mock_call: AsyncMock,
    ) -> None:
        """cancel() deve chamar call.cancel() e marcar closed."""
        await stream_handle.cancel()

        mock_call.cancel.assert_called_once()
        assert stream_handle.is_closed is True


# ---------------------------------------------------------------------------
# StreamHandle — properties
# ---------------------------------------------------------------------------


class TestStreamHandleProperties:
    """Testes de propriedades do StreamHandle."""

    async def test_session_id_returns_correct_value(
        self,
        stream_handle: StreamHandle,
    ) -> None:
        """session_id deve retornar o ID da sessao."""
        assert stream_handle.session_id == "sess_test_001"

    async def test_is_closed_initially_false(
        self,
        stream_handle: StreamHandle,
    ) -> None:
        """is_closed deve ser False inicialmente."""
        assert stream_handle.is_closed is False


# ---------------------------------------------------------------------------
# StreamingGRPCClient
# ---------------------------------------------------------------------------


class TestStreamingGRPCClient:
    """Testes de StreamingGRPCClient."""

    async def test_open_stream_without_connect_raises_worker_crash(self) -> None:
        """open_stream sem connect() deve levantar WorkerCrashError."""
        client = StreamingGRPCClient("localhost:50051")

        with pytest.raises(WorkerCrashError):
            await client.open_stream("sess_test_001")

    async def test_connect_creates_channel_and_stub(self) -> None:
        """connect() deve criar canal gRPC e stub."""
        client = StreamingGRPCClient("localhost:50051")

        with patch("macaw.scheduler.streaming.grpc.aio.insecure_channel") as mock_channel_fn:
            mock_channel = MagicMock()
            mock_channel_fn.return_value = mock_channel

            await client.connect()

            mock_channel_fn.assert_called_once()
            assert client._channel is not None
            assert client._stub is not None

        # Cleanup
        client._channel = None
        client._stub = None

    async def test_open_stream_returns_stream_handle(self) -> None:
        """open_stream apos connect deve retornar StreamHandle."""
        client = StreamingGRPCClient("localhost:50051")

        mock_call = AsyncMock()
        mock_stub = MagicMock()
        mock_stub.TranscribeStream.return_value = mock_call

        # Injetar stub diretamente
        client._stub = mock_stub
        client._channel = MagicMock()

        handle = await client.open_stream("sess_test_002")

        assert isinstance(handle, StreamHandle)
        assert handle.session_id == "sess_test_002"
        assert handle.is_closed is False

        # Cleanup
        client._channel = None
        client._stub = None

    async def test_close_closes_channel(self) -> None:
        """close() deve fechar o canal gRPC e limpar referencias."""
        client = StreamingGRPCClient("localhost:50051")

        mock_channel = AsyncMock()
        client._channel = mock_channel
        client._stub = MagicMock()

        await client.close()

        mock_channel.close.assert_called_once()
        assert client._channel is None
        assert client._stub is None

    async def test_close_idempotent_when_not_connected(self) -> None:
        """close() sem conexao deve ser no-op sem erro."""
        client = StreamingGRPCClient("localhost:50051")
        await client.close()  # Nao deve levantar excecao


# ---------------------------------------------------------------------------
# _proto_event_to_transcript_segment (funcao pura)
# ---------------------------------------------------------------------------


class TestProtoEventToTranscriptSegment:
    """Testes da funcao de conversao proto -> dominio."""

    def test_converts_final_event(self) -> None:
        """Evento 'final' deve gerar TranscriptSegment com is_final=True."""
        event = TranscriptEvent(
            session_id="sess_001",
            event_type="final",
            text="ola mundo",
            segment_id=3,
            start_ms=1000,
            end_ms=2500,
            language="pt",
            confidence=0.92,
        )

        result = _proto_event_to_transcript_segment(event)

        assert result.text == "ola mundo"
        assert result.is_final is True
        assert result.segment_id == 3
        assert result.start_ms == 1000
        assert result.end_ms == 2500
        assert result.language == "pt"
        assert result.confidence == pytest.approx(0.92)
        assert result.words is None

    def test_converts_partial_event(self) -> None:
        """Evento 'partial' deve gerar TranscriptSegment com is_final=False."""
        event = TranscriptEvent(
            session_id="sess_001",
            event_type="partial",
            text="ola",
            segment_id=0,
        )

        result = _proto_event_to_transcript_segment(event)

        assert result.text == "ola"
        assert result.is_final is False
        assert result.segment_id == 0
        assert result.start_ms == 0  # Proto default 0 preserved (valid timestamp)
        assert result.end_ms == 0
        assert result.language is None  # Proto default "" -> None
        assert result.confidence is None  # Proto default 0.0 -> None

    def test_converts_event_with_words(self) -> None:
        """Evento com words deve converter para tuple de WordTimestamp."""
        event = TranscriptEvent(
            session_id="sess_001",
            event_type="final",
            text="PIX transferencia",
            segment_id=1,
            start_ms=500,
            end_ms=2000,
            words=[
                Word(word="PIX", start=0.5, end=0.8, probability=0.99),
                Word(word="transferencia", start=0.9, end=2.0, probability=0.85),
            ],
        )

        result = _proto_event_to_transcript_segment(event)

        assert result.words is not None
        assert len(result.words) == 2

        assert result.words[0].word == "PIX"
        assert result.words[0].start == pytest.approx(0.5)
        assert result.words[0].end == pytest.approx(0.8)
        assert result.words[0].probability == pytest.approx(0.99)

        assert result.words[1].word == "transferencia"
        assert result.words[1].start == pytest.approx(0.9)
        assert result.words[1].end == pytest.approx(2.0)
        assert result.words[1].probability == pytest.approx(0.85)

    def test_word_probability_zero_becomes_none(self) -> None:
        """Probabilidade 0.0 do proto (default) deve virar None no dominio."""
        event = TranscriptEvent(
            session_id="sess_001",
            event_type="final",
            text="teste",
            segment_id=0,
            words=[
                Word(word="teste", start=0.0, end=1.0, probability=0.0),
            ],
        )

        result = _proto_event_to_transcript_segment(event)

        assert result.words is not None
        assert result.words[0].probability is None

    def test_confidence_zero_becomes_none(self) -> None:
        """Confidence 0.0 do proto (default) deve virar None no dominio."""
        event = TranscriptEvent(
            session_id="sess_001",
            event_type="final",
            text="teste",
            segment_id=0,
            confidence=0.0,
        )

        result = _proto_event_to_transcript_segment(event)

        assert result.confidence is None

    def test_empty_language_becomes_none(self) -> None:
        """Language vazia do proto (default) deve virar None no dominio."""
        event = TranscriptEvent(
            session_id="sess_001",
            event_type="partial",
            text="teste",
            segment_id=0,
            language="",
        )

        result = _proto_event_to_transcript_segment(event)

        assert result.language is None

    def test_result_is_immutable(self) -> None:
        """TranscriptSegment retornado deve ser frozen (imutavel)."""
        event = TranscriptEvent(
            event_type="final",
            text="teste",
            segment_id=0,
        )

        result = _proto_event_to_transcript_segment(event)

        with pytest.raises(AttributeError):
            result.text = "modificado"  # type: ignore[misc]
