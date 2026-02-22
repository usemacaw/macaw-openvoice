"""Tests for TTS alignment over WebSocket (Sprint 3).

Validates:
- TTSAlignmentEvent and TTSAlignmentItemEvent models
- TTSSpeakCommand include_alignment / alignment_granularity fields
- _tts_speak_task emits tts.alignment events before binary audio frames
- No alignment events when include_alignment=false (backward compat)
- tts.alignment event in ServerEvent union
"""

from __future__ import annotations

import asyncio
import json
from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from pydantic import ValidationError

from macaw.server.models.events import (
    ServerEvent,
    TTSAlignmentEvent,
    TTSAlignmentItemEvent,
    TTSSpeakCommand,
)
from macaw.server.routes.realtime import _tts_speak_task

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_mock_websocket() -> MagicMock:
    from starlette.websockets import WebSocketState

    ws = AsyncMock()
    ws.client_state = WebSocketState.CONNECTED
    ws.send_json = AsyncMock()
    ws.send_bytes = AsyncMock()
    return ws


def _make_mock_session() -> MagicMock:
    session = MagicMock()
    session.is_closed = False
    session.is_muted = False
    session.segment_id = 0

    def _mute() -> None:
        session.is_muted = True

    def _unmute() -> None:
        session.is_muted = False

    session.mute = _mute
    session.unmute = _unmute
    session.close = AsyncMock()
    return session


def _make_mock_registry() -> MagicMock:
    from macaw._types import ModelType

    registry = MagicMock()
    manifest = MagicMock()
    manifest.model_type = ModelType.TTS
    manifest.name = "kokoro-v1"
    registry.get_manifest.return_value = manifest
    registry.list_models.return_value = [manifest]
    return registry


def _make_mock_worker_manager() -> MagicMock:
    worker = MagicMock()
    worker.port = 50052
    wm = MagicMock()
    wm.get_ready_worker.return_value = worker
    return wm


def _make_send_event() -> tuple[AsyncMock, list[Any]]:
    events: list[Any] = []

    async def _send(event: Any) -> None:
        events.append(event)

    return AsyncMock(side_effect=_send), events


def _make_alignment_mock(
    items: list[tuple[str, int, int]],
    granularity: str = "word",
) -> MagicMock:
    """Create a mock ChunkAlignment proto."""
    alignment = MagicMock()
    alignment.granularity = granularity
    mock_items = []
    for text, start_ms, duration_ms in items:
        item = MagicMock()
        item.text = text
        item.start_ms = start_ms
        item.duration_ms = duration_ms
        mock_items.append(item)
    alignment.items = mock_items
    return alignment


def _make_chunk(
    audio_data: bytes,
    *,
    is_last: bool = False,
    alignment: Any = None,
    normalized_alignment: Any = None,
) -> MagicMock:
    chunk = MagicMock()
    chunk.audio_data = audio_data
    chunk.is_last = is_last
    chunk.alignment = alignment
    # Default: empty normalized_alignment (mimics proto default message)
    if normalized_alignment is None:
        na = MagicMock()
        na.items = []
        na.granularity = ""
        chunk.normalized_alignment = na
    else:
        chunk.normalized_alignment = normalized_alignment
    return chunk


def _make_grpc_stream(chunks: list[MagicMock]) -> Any:
    class _FakeStream:
        def __init__(self, items: list[MagicMock]) -> None:
            self._items = items
            self._idx = 0

        def __aiter__(self) -> _FakeStream:
            return self

        async def __anext__(self) -> MagicMock:
            if self._idx >= len(self._items):
                raise StopAsyncIteration
            item = self._items[self._idx]
            self._idx += 1
            return item

    return _FakeStream(chunks)


async def _run_tts_speak_task(
    *,
    chunks: list[MagicMock],
    include_alignment: bool = False,
    alignment_granularity: str = "word",
) -> tuple[list[Any], MagicMock]:
    """Run _tts_speak_task with given chunks and return (events, websocket)."""
    ws = _make_mock_websocket()
    session = _make_mock_session()
    send_event, events = _make_send_event()
    cancel = asyncio.Event()

    ws.app = MagicMock()
    ws.app.state.registry = _make_mock_registry()
    ws.app.state.worker_manager = _make_mock_worker_manager()

    stream = _make_grpc_stream(chunks)

    cmd = TTSSpeakCommand(
        text="Hello world",
        voice="default",
        speed=1.0,
        include_alignment=include_alignment,
        alignment_granularity=alignment_granularity,
    )

    with patch("macaw.server.routes.realtime.get_or_create_tts_channel") as mock_get_ch:
        mock_channel = AsyncMock()
        mock_get_ch.return_value = mock_channel
        mock_stub = MagicMock()
        mock_stub.Synthesize.return_value = stream

        with patch("macaw.server.routes.realtime.TTSWorkerStub", return_value=mock_stub):
            await _tts_speak_task(
                websocket=ws,
                session_id="sess_test",
                session=session,
                request_id="req_align",
                cmd=cmd,
                model_tts="kokoro-v1",
                send_event=send_event,
                cancel_event=cancel,
            )

    return events, ws


# ---------------------------------------------------------------------------
# TTSAlignmentEvent Model
# ---------------------------------------------------------------------------


class TestTTSAlignmentEvent:
    def test_type_literal(self) -> None:
        event = TTSAlignmentEvent(
            request_id="req-1",
            items=[TTSAlignmentItemEvent(text="Hello", start_ms=0, duration_ms=300)],
        )
        assert event.type == "tts.alignment"

    def test_default_granularity(self) -> None:
        event = TTSAlignmentEvent(
            request_id="req-1",
            items=[],
        )
        assert event.granularity == "word"

    def test_character_granularity(self) -> None:
        event = TTSAlignmentEvent(
            request_id="req-1",
            items=[],
            granularity="character",
        )
        assert event.granularity == "character"

    def test_frozen(self) -> None:
        event = TTSAlignmentEvent(request_id="req-1", items=[])
        with pytest.raises(ValidationError):
            event.request_id = "changed"  # type: ignore[misc]

    def test_serialization(self) -> None:
        event = TTSAlignmentEvent(
            request_id="req-1",
            items=[
                TTSAlignmentItemEvent(text="Hello", start_ms=0, duration_ms=300),
                TTSAlignmentItemEvent(text="world", start_ms=300, duration_ms=400),
            ],
            granularity="word",
        )
        d = json.loads(event.model_dump_json())
        assert d["type"] == "tts.alignment"
        assert d["request_id"] == "req-1"
        assert len(d["items"]) == 2
        assert d["items"][0]["text"] == "Hello"
        assert d["items"][1]["start_ms"] == 300
        assert d["granularity"] == "word"

    def test_in_server_event_union(self) -> None:
        """TTSAlignmentEvent is part of the ServerEvent union."""
        assert TTSAlignmentEvent in ServerEvent.__args__  # type: ignore[attr-defined]


# ---------------------------------------------------------------------------
# TTSAlignmentItemEvent Model
# ---------------------------------------------------------------------------


class TestTTSAlignmentItemEvent:
    def test_fields(self) -> None:
        item = TTSAlignmentItemEvent(text="Hello", start_ms=0, duration_ms=300)
        assert item.text == "Hello"
        assert item.start_ms == 0
        assert item.duration_ms == 300

    def test_frozen(self) -> None:
        item = TTSAlignmentItemEvent(text="Hello", start_ms=0, duration_ms=300)
        with pytest.raises(ValidationError):
            item.text = "changed"  # type: ignore[misc]


# ---------------------------------------------------------------------------
# TTSSpeakCommand Alignment Fields
# ---------------------------------------------------------------------------


class TestTTSSpeakCommandAlignment:
    def test_default_alignment_off(self) -> None:
        cmd = TTSSpeakCommand(text="Hello")
        assert cmd.include_alignment is False
        assert cmd.alignment_granularity == "word"

    def test_include_alignment_true(self) -> None:
        cmd = TTSSpeakCommand(text="Hello", include_alignment=True)
        assert cmd.include_alignment is True

    def test_character_granularity(self) -> None:
        cmd = TTSSpeakCommand(
            text="Hello",
            include_alignment=True,
            alignment_granularity="character",
        )
        assert cmd.alignment_granularity == "character"

    def test_invalid_granularity_rejected(self) -> None:
        with pytest.raises(ValidationError):
            TTSSpeakCommand(text="Hello", alignment_granularity="phoneme")

    def test_serialization_roundtrip(self) -> None:
        cmd = TTSSpeakCommand(
            text="Hello",
            include_alignment=True,
            alignment_granularity="character",
        )
        data = cmd.model_dump()
        assert data["include_alignment"] is True
        assert data["alignment_granularity"] == "character"
        restored = TTSSpeakCommand.model_validate(data)
        assert restored == cmd


# ---------------------------------------------------------------------------
# _tts_speak_task alignment event emission
# ---------------------------------------------------------------------------


class TestTTSSpeakTaskAlignment:
    async def test_alignment_events_emitted_before_audio(self) -> None:
        """When include_alignment=true, tts.alignment events precede binary frames."""
        align = _make_alignment_mock([("Hello", 0, 300), ("world", 300, 400)])
        chunks = [
            _make_chunk(b"\x00\x01" * 100, alignment=align),
            _make_chunk(b"\x00\x02" * 100, alignment=None, is_last=True),
        ]

        events, _ws = await _run_tts_speak_task(
            chunks=chunks,
            include_alignment=True,
        )

        # Events: speaking_start, alignment, speaking_end
        event_types = [e.type for e in events]
        assert "tts.alignment" in event_types
        assert "tts.speaking_start" in event_types
        assert "tts.speaking_end" in event_types

        # Alignment should come after speaking_start
        start_idx = event_types.index("tts.speaking_start")
        align_idx = event_types.index("tts.alignment")
        assert align_idx > start_idx

    async def test_alignment_event_contains_correct_data(self) -> None:
        align = _make_alignment_mock([("Hello", 0, 300), ("world", 300, 400)])
        chunks = [
            _make_chunk(b"\x00\x01" * 100, alignment=align, is_last=True),
        ]

        events, _ws = await _run_tts_speak_task(
            chunks=chunks,
            include_alignment=True,
        )

        alignment_events = [e for e in events if e.type == "tts.alignment"]
        assert len(alignment_events) == 1
        ae = alignment_events[0]
        assert ae.request_id == "req_align"
        assert len(ae.items) == 2
        assert ae.items[0].text == "Hello"
        assert ae.items[0].start_ms == 0
        assert ae.items[0].duration_ms == 300
        assert ae.items[1].text == "world"
        assert ae.granularity == "word"

    async def test_alignment_event_respects_granularity(self) -> None:
        align = _make_alignment_mock(
            [("H", 0, 50), ("e", 50, 50)],
            granularity="character",
        )
        chunks = [
            _make_chunk(b"\x00\x01" * 100, alignment=align, is_last=True),
        ]

        events, _ws = await _run_tts_speak_task(
            chunks=chunks,
            include_alignment=True,
            alignment_granularity="character",
        )

        alignment_events = [e for e in events if e.type == "tts.alignment"]
        assert len(alignment_events) == 1
        assert alignment_events[0].granularity == "character"

    async def test_multiple_chunks_with_alignment(self) -> None:
        align_a = _make_alignment_mock([("Hello", 0, 300)])
        align_b = _make_alignment_mock([("world", 300, 400)])
        chunks = [
            _make_chunk(b"\x00\x01" * 100, alignment=align_a),
            _make_chunk(b"\x00\x02" * 100, alignment=align_b, is_last=True),
        ]

        events, mock_ws = await _run_tts_speak_task(
            chunks=chunks,
            include_alignment=True,
        )

        alignment_events = [e for e in events if e.type == "tts.alignment"]
        assert len(alignment_events) == 2
        assert alignment_events[0].items[0].text == "Hello"
        assert alignment_events[1].items[0].text == "world"
        # Binary frames still sent
        assert mock_ws.send_bytes.call_count == 2

    async def test_chunk_without_alignment_no_event(self) -> None:
        """Chunks with alignment=None don't produce tts.alignment events."""
        chunks = [
            _make_chunk(b"\x00\x01" * 100, alignment=None),
            _make_chunk(b"\x00\x02" * 100, alignment=None, is_last=True),
        ]

        events, ws = await _run_tts_speak_task(
            chunks=chunks,
            include_alignment=True,
        )

        alignment_events = [e for e in events if e.type == "tts.alignment"]
        assert len(alignment_events) == 0
        # Audio still sent
        assert ws.send_bytes.call_count == 2

    async def test_empty_alignment_items_no_event(self) -> None:
        """Chunk with alignment but empty items list doesn't produce event."""
        alignment = MagicMock()
        alignment.items = []
        alignment.granularity = "word"
        chunks = [
            _make_chunk(b"\x00\x01" * 100, alignment=alignment, is_last=True),
        ]

        events, _ws = await _run_tts_speak_task(
            chunks=chunks,
            include_alignment=True,
        )

        alignment_events = [e for e in events if e.type == "tts.alignment"]
        assert len(alignment_events) == 0


# ---------------------------------------------------------------------------
# Backward Compatibility — no alignment by default
# ---------------------------------------------------------------------------


class TestTTSSpeakTaskNoAlignment:
    async def test_no_alignment_events_by_default(self) -> None:
        """When include_alignment=false (default), no tts.alignment events."""
        align = _make_alignment_mock([("Hello", 0, 300)])
        chunks = [
            _make_chunk(b"\x00\x01" * 100, alignment=align, is_last=True),
        ]

        events, ws = await _run_tts_speak_task(
            chunks=chunks,
            include_alignment=False,
        )

        alignment_events = [e for e in events if e.type == "tts.alignment"]
        assert len(alignment_events) == 0
        # Normal events still emitted
        event_types = [e.type for e in events]
        assert "tts.speaking_start" in event_types
        assert "tts.speaking_end" in event_types
        # Audio still sent
        assert ws.send_bytes.call_count == 1

    async def test_speaking_events_unchanged_with_alignment(self) -> None:
        """tts.speaking_start and tts.speaking_end still emitted with alignment."""
        chunks = [
            _make_chunk(b"\x00\x01" * 100, alignment=None, is_last=True),
        ]

        events, _ws = await _run_tts_speak_task(
            chunks=chunks,
            include_alignment=True,
        )

        event_types = [e.type for e in events]
        assert event_types[0] == "tts.speaking_start"
        assert event_types[-1] == "tts.speaking_end"
