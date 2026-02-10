"""Teste de integracao end-to-end dos componentes M6.

Valida que todos os componentes M6 funcionam juntos em um fluxo real:
- SessionStateMachine (6 estados com transicoes e timeouts)
- RingBuffer (escrita, read fence, force commit)
- WAL (checkpoint apos transcript.final)
- CrossSegmentContext (initial_prompt entre segmentos)
- Hot Words (envio no primeiro frame de cada segmento)
- Recovery (crash de worker -> retomada sem duplicacao)

Todos os componentes sao mocked no nivel do gRPC e VAD, mas os
componentes M6 (state machine, ring buffer, WAL, cross-segment)
sao reais.

Executar com:
    python -m pytest tests/integration/test_m6_integration.py -v --tb=short
"""

from __future__ import annotations

import asyncio
from typing import TYPE_CHECKING, Any

import numpy as np
import pytest

from macaw._types import SessionState, TranscriptSegment
from macaw.server.models.events import (
    StreamingErrorEvent,
    TranscriptFinalEvent,
    VADSpeechEndEvent,
    VADSpeechStartEvent,
)
from macaw.session.cross_segment import CrossSegmentContext
from macaw.session.ring_buffer import RingBuffer
from macaw.session.state_machine import SessionStateMachine, SessionTimeouts
from macaw.session.streaming import StreamingSession
from macaw.session.wal import SessionWAL
from macaw.vad.detector import VADEvent, VADEventType

if TYPE_CHECKING:
    from collections.abc import Callable

pytestmark = [pytest.mark.integration]

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

_FRAME_SIZE = 1024
_SAMPLE_RATE = 16000


# ---------------------------------------------------------------------------
# Lightweight mocks (no call history accumulation)
# ---------------------------------------------------------------------------


class _LightPreprocessor:
    """Preprocessor mock."""

    def __init__(self) -> None:
        self._frame = np.zeros(_FRAME_SIZE, dtype=np.float32)

    def process_frame(self, raw_bytes: bytes) -> np.ndarray:
        return self._frame


class _LightVAD:
    """VAD mock with controllable events."""

    def __init__(self) -> None:
        self.is_speaking = False
        self._next_event: VADEvent | None = None

    def set_next_event(self, event: VADEvent | None) -> None:
        self._next_event = event

    def process_frame(self, frame: np.ndarray) -> VADEvent | None:
        event = self._next_event
        self._next_event = None
        return event

    def reset(self) -> None:
        pass


class _LightPostprocessor:
    """Postprocessor mock que simula ITN simples."""

    def process(self, text: str) -> str:
        # Simula ITN: "dois mil" -> "2000"
        return text.replace("dois mil", "2000")


class _AsyncIterFromList:
    """Async iterator from list."""

    def __init__(self, items: list[Any]) -> None:
        self._items = list(items)
        self._index = 0

    def __aiter__(self) -> _AsyncIterFromList:
        return self

    async def __anext__(self) -> Any:
        if self._index >= len(self._items):
            raise StopAsyncIteration
        item = self._items[self._index]
        self._index += 1
        if isinstance(item, Exception):
            raise item
        return item


class _LightStreamHandle:
    """Stream handle mock."""

    def __init__(self, events: list[Any]) -> None:
        self.is_closed = False
        self.session_id = "integration_test"
        self._events = events
        self.sent_frames: list[dict[str, Any]] = []

    def receive_events(self) -> _AsyncIterFromList:
        return _AsyncIterFromList(self._events)

    async def send_frame(
        self,
        *,
        pcm_data: bytes,
        initial_prompt: str | None = None,
        hot_words: list[str] | None = None,
    ) -> None:
        self.sent_frames.append(
            {
                "pcm_data_len": len(pcm_data),
                "initial_prompt": initial_prompt,
                "hot_words": hot_words,
            }
        )

    async def close(self) -> None:
        self.is_closed = True

    async def cancel(self) -> None:
        self.is_closed = True


class _LightGRPCClient:
    """gRPC client mock with configurable handles per call."""

    def __init__(self, handle_factory: Callable[[], _LightStreamHandle]) -> None:
        self._handle_factory = handle_factory
        self.open_stream_count = 0

    async def open_stream(self, session_id: str) -> _LightStreamHandle:
        self.open_stream_count += 1
        return self._handle_factory()

    async def close(self) -> None:
        pass


def _make_raw_bytes(n_samples: int = _FRAME_SIZE) -> bytes:
    """PCM int16 zeros."""
    return np.zeros(n_samples, dtype=np.int16).tobytes()


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


async def test_full_session_lifecycle_with_all_m6_components() -> None:
    """Sessao completa: INIT -> ACTIVE -> SILENCE -> ACTIVE -> SILENCE -> close.

    Verifica que todos os componentes M6 participam corretamente:
    - State machine transita entre estados
    - Ring buffer armazena frames
    - WAL registra checkpoints apos transcript.final
    - Cross-segment context fornece initial_prompt no segundo segmento
    - Hot words enviados no primeiro frame de cada segmento
    """
    preprocessor = _LightPreprocessor()
    postprocessor = _LightPostprocessor()
    vad = _LightVAD()
    raw_frame = _make_raw_bytes()

    # Collected events
    events: list[Any] = []

    async def on_event(event: Any) -> None:
        events.append(event)

    # Segment counter for handle factory
    segment_counter = 0

    def _make_handle() -> _LightStreamHandle:
        nonlocal segment_counter
        final_segment = TranscriptSegment(
            text=f"segmento {segment_counter} com dois mil",
            is_final=True,
            segment_id=segment_counter,
            start_ms=segment_counter * 4000,
            end_ms=segment_counter * 4000 + 3000,
            language="pt",
            confidence=0.92,
        )
        segment_counter += 1
        return _LightStreamHandle(events=[final_segment])

    grpc_client = _LightGRPCClient(handle_factory=_make_handle)

    # Real M6 components
    state_machine = SessionStateMachine(
        timeouts=SessionTimeouts(
            init_timeout_s=30.0,
            silence_timeout_s=30.0,
            hold_timeout_s=300.0,
            closing_timeout_s=2.0,
        ),
    )
    ring_buffer = RingBuffer(duration_s=10.0)
    wal = SessionWAL()
    cross_segment = CrossSegmentContext(max_tokens=224)

    session = StreamingSession(
        session_id="integration_test_full",
        preprocessor=preprocessor,  # type: ignore[arg-type]
        vad=vad,  # type: ignore[arg-type]
        grpc_client=grpc_client,  # type: ignore[arg-type]
        postprocessor=postprocessor,  # type: ignore[arg-type]
        on_event=on_event,
        hot_words=["PIX", "TED"],
        enable_itn=True,
        state_machine=state_machine,
        ring_buffer=ring_buffer,
        wal=wal,
        cross_segment_context=cross_segment,
    )

    # --- Segment 0: INIT -> ACTIVE ---

    # Verify initial state
    assert session.session_state == SessionState.INIT

    # Speech start -> ACTIVE
    vad.set_next_event(VADEvent(type=VADEventType.SPEECH_START, timestamp_ms=0))
    vad.is_speaking = True
    await session.process_frame(raw_frame)
    assert session.session_state == SessionState.ACTIVE

    # Send speech frames
    for _ in range(5):
        await session.process_frame(raw_frame)

    # Speech end -> SILENCE
    vad.set_next_event(VADEvent(type=VADEventType.SPEECH_END, timestamp_ms=3000))
    vad.is_speaking = False
    await session.process_frame(raw_frame)
    await asyncio.sleep(0.01)  # Let receiver task complete
    assert session.session_state == SessionState.SILENCE

    # Verify events for segment 0
    speech_starts = [e for e in events if isinstance(e, VADSpeechStartEvent)]
    speech_ends = [e for e in events if isinstance(e, VADSpeechEndEvent)]
    finals = [e for e in events if isinstance(e, TranscriptFinalEvent)]

    assert len(speech_starts) == 1
    assert len(speech_ends) == 1
    assert len(finals) == 1
    # ITN should transform "dois mil" -> "2000"
    assert "2000" in finals[0].text

    # Verify ring buffer has data
    assert ring_buffer.total_written > 0
    # After transcript.final, read fence should have advanced
    assert ring_buffer.read_fence > 0

    # Verify WAL checkpoint was recorded
    assert wal.last_committed_segment_id == 0

    # Verify cross-segment context was updated
    assert cross_segment.get_prompt() is not None
    assert "2000" in cross_segment.get_prompt()  # type: ignore[operator]

    # --- Segment 1: SILENCE -> ACTIVE (new speech) ---

    events.clear()

    # New speech -> ACTIVE again
    vad.set_next_event(VADEvent(type=VADEventType.SPEECH_START, timestamp_ms=5000))
    vad.is_speaking = True
    await session.process_frame(raw_frame)
    assert session.session_state == SessionState.ACTIVE

    # Send speech frames
    for _ in range(3):
        await session.process_frame(raw_frame)

    # Speech end -> SILENCE
    vad.set_next_event(VADEvent(type=VADEventType.SPEECH_END, timestamp_ms=7000))
    vad.is_speaking = False
    await session.process_frame(raw_frame)
    await asyncio.sleep(0.01)

    # Verify segment 1 events
    finals_1 = [e for e in events if isinstance(e, TranscriptFinalEvent)]
    assert len(finals_1) == 1

    # WAL should have segment_id=1 now
    assert wal.last_committed_segment_id == 1

    # Verify session segment counter advanced
    assert session.segment_id == 2  # 0 and 1 completed

    # gRPC client should have opened 2 streams (one per segment)
    assert grpc_client.open_stream_count == 2

    # --- Close session ---

    await session.close()
    assert session.is_closed
    assert session.session_state == SessionState.CLOSED


async def test_hot_words_and_cross_segment_in_initial_prompt() -> None:
    """Verifica que hot words + cross-segment context sao combinados no initial_prompt."""
    preprocessor = _LightPreprocessor()
    vad = _LightVAD()
    raw_frame = _make_raw_bytes()

    events: list[Any] = []

    async def on_event(event: Any) -> None:
        events.append(event)

    # Track the handles to inspect sent frames
    handles: list[_LightStreamHandle] = []
    seg_counter = 0

    def _make_handle() -> _LightStreamHandle:
        nonlocal seg_counter
        handle = _LightStreamHandle(
            events=[
                TranscriptSegment(
                    text=f"segmento {seg_counter}",
                    is_final=True,
                    segment_id=seg_counter,
                    start_ms=0,
                    end_ms=3000,
                    language="pt",
                    confidence=0.9,
                ),
            ]
        )
        handles.append(handle)
        seg_counter += 1
        return handle

    grpc_client = _LightGRPCClient(handle_factory=_make_handle)
    cross_segment = CrossSegmentContext(max_tokens=224)

    session = StreamingSession(
        session_id="prompt_test",
        preprocessor=preprocessor,  # type: ignore[arg-type]
        vad=vad,  # type: ignore[arg-type]
        grpc_client=grpc_client,  # type: ignore[arg-type]
        postprocessor=None,
        on_event=on_event,
        hot_words=["PIX", "TED"],
        enable_itn=False,
        cross_segment_context=cross_segment,
    )

    # --- Segment 0 ---

    # Speech start
    vad.set_next_event(VADEvent(type=VADEventType.SPEECH_START, timestamp_ms=0))
    vad.is_speaking = True
    await session.process_frame(raw_frame)

    # First frame of segment 0: should have hot words in initial_prompt
    # (no cross-segment context yet since no previous segment)
    await session.process_frame(raw_frame)

    # Check first handle's sent frames
    assert len(handles) == 1
    first_frame = handles[0].sent_frames[0]
    assert first_frame["hot_words"] == ["PIX", "TED"]
    assert first_frame["initial_prompt"] is not None
    assert "PIX" in first_frame["initial_prompt"]
    assert "TED" in first_frame["initial_prompt"]

    # Second frame: no hot words (already sent)
    if len(handles[0].sent_frames) > 1:
        second_frame = handles[0].sent_frames[1]
        assert second_frame["hot_words"] is None
        assert second_frame["initial_prompt"] is None

    # Speech end
    vad.set_next_event(VADEvent(type=VADEventType.SPEECH_END, timestamp_ms=3000))
    vad.is_speaking = False
    await session.process_frame(raw_frame)
    await asyncio.sleep(0.01)

    # Cross-segment context should now have "segmento 0"
    assert cross_segment.get_prompt() == "segmento 0"

    # --- Segment 1 ---

    # Speech start
    vad.set_next_event(VADEvent(type=VADEventType.SPEECH_START, timestamp_ms=5000))
    vad.is_speaking = True
    await session.process_frame(raw_frame)

    # First frame of segment 1: should have hot words + cross-segment context
    await session.process_frame(raw_frame)

    assert len(handles) == 2
    first_frame_seg1 = handles[1].sent_frames[0]
    assert first_frame_seg1["hot_words"] == ["PIX", "TED"]
    assert first_frame_seg1["initial_prompt"] is not None
    # Should combine: "Termos: PIX, TED. segmento 0"
    prompt = first_frame_seg1["initial_prompt"]
    assert "Termos: PIX, TED." in prompt
    assert "segmento 0" in prompt

    # Cleanup
    vad.set_next_event(VADEvent(type=VADEventType.SPEECH_END, timestamp_ms=7000))
    vad.is_speaking = False
    await session.process_frame(raw_frame)
    await asyncio.sleep(0.01)
    await session.close()


async def test_ring_buffer_and_wal_consistency() -> None:
    """Verifica consistencia entre ring buffer e WAL apos multiplos segmentos."""
    preprocessor = _LightPreprocessor()
    vad = _LightVAD()
    raw_frame = _make_raw_bytes()

    events: list[Any] = []

    async def on_event(event: Any) -> None:
        events.append(event)

    seg_counter = 0

    def _make_handle() -> _LightStreamHandle:
        nonlocal seg_counter
        handle = _LightStreamHandle(
            events=[
                TranscriptSegment(
                    text=f"seg {seg_counter}",
                    is_final=True,
                    segment_id=seg_counter,
                    start_ms=0,
                    end_ms=1000,
                    language="pt",
                    confidence=0.9,
                ),
            ]
        )
        seg_counter += 1
        return handle

    grpc_client = _LightGRPCClient(handle_factory=_make_handle)
    ring_buffer = RingBuffer(duration_s=10.0)
    wal = SessionWAL()

    session = StreamingSession(
        session_id="consistency_test",
        preprocessor=preprocessor,  # type: ignore[arg-type]
        vad=vad,  # type: ignore[arg-type]
        grpc_client=grpc_client,  # type: ignore[arg-type]
        postprocessor=None,
        on_event=on_event,
        enable_itn=False,
        ring_buffer=ring_buffer,
        wal=wal,
    )

    # Process 5 segments
    for seg in range(5):
        # Speech start
        vad.set_next_event(
            VADEvent(type=VADEventType.SPEECH_START, timestamp_ms=seg * 2000),
        )
        vad.is_speaking = True
        await session.process_frame(raw_frame)

        # 3 speech frames per segment
        for _ in range(3):
            await session.process_frame(raw_frame)

        # Speech end
        vad.set_next_event(
            VADEvent(type=VADEventType.SPEECH_END, timestamp_ms=seg * 2000 + 1000),
        )
        vad.is_speaking = False
        await session.process_frame(raw_frame)
        await asyncio.sleep(0.01)

    # Verify WAL has last segment
    assert wal.last_committed_segment_id == 4

    # Verify ring buffer: fence should equal total_written
    # (all segments committed via transcript.final)
    assert ring_buffer.read_fence == ring_buffer.total_written
    assert ring_buffer.uncommitted_bytes == 0

    # Verify 5 finals emitted
    finals = [e for e in events if isinstance(e, TranscriptFinalEvent)]
    assert len(finals) == 5

    # Verify segment IDs are sequential
    segment_ids = [f.segment_id for f in finals]
    assert segment_ids == [0, 1, 2, 3, 4]

    await session.close()


async def test_state_machine_timeouts_integration() -> None:
    """Verifica que timeouts da state machine disparam transicoes corretas.

    Usa clock mockado para controlar tempo sem esperar timeouts reais.
    """
    preprocessor = _LightPreprocessor()
    vad = _LightVAD()

    events: list[Any] = []

    async def on_event(event: Any) -> None:
        events.append(event)

    current_time = 0.0

    def mock_clock() -> float:
        return current_time

    state_machine = SessionStateMachine(
        timeouts=SessionTimeouts(
            init_timeout_s=5.0,
            silence_timeout_s=3.0,
            hold_timeout_s=10.0,
            closing_timeout_s=2.0,
        ),
        clock=mock_clock,
    )

    def _make_handle() -> _LightStreamHandle:
        return _LightStreamHandle(
            events=[
                TranscriptSegment(
                    text="test",
                    is_final=True,
                    segment_id=0,
                    start_ms=0,
                    end_ms=1000,
                    language="pt",
                    confidence=0.9,
                ),
            ]
        )

    grpc_client = _LightGRPCClient(handle_factory=_make_handle)

    session = StreamingSession(
        session_id="timeout_test",
        preprocessor=preprocessor,  # type: ignore[arg-type]
        vad=vad,  # type: ignore[arg-type]
        grpc_client=grpc_client,  # type: ignore[arg-type]
        postprocessor=None,
        on_event=on_event,
        enable_itn=False,
        state_machine=state_machine,
    )

    # State: INIT
    assert session.session_state == SessionState.INIT

    # Advance time past INIT timeout (5s)
    current_time = 6.0
    result = await session.check_timeout()
    assert result == SessionState.CLOSED
    assert session.is_closed


async def test_silence_to_hold_timeout_integration() -> None:
    """Verifica transicao SILENCE -> HOLD -> CLOSING via timeouts."""
    preprocessor = _LightPreprocessor()
    vad = _LightVAD()
    raw_frame = _make_raw_bytes()

    events: list[Any] = []

    async def on_event(event: Any) -> None:
        events.append(event)

    current_time = 0.0

    def mock_clock() -> float:
        return current_time

    state_machine = SessionStateMachine(
        timeouts=SessionTimeouts(
            init_timeout_s=30.0,
            silence_timeout_s=3.0,
            hold_timeout_s=10.0,
            closing_timeout_s=2.0,
        ),
        clock=mock_clock,
    )

    def _make_handle() -> _LightStreamHandle:
        return _LightStreamHandle(
            events=[
                TranscriptSegment(
                    text="test",
                    is_final=True,
                    segment_id=0,
                    start_ms=0,
                    end_ms=1000,
                    language="pt",
                    confidence=0.9,
                ),
            ]
        )

    grpc_client = _LightGRPCClient(handle_factory=_make_handle)

    session = StreamingSession(
        session_id="hold_test",
        preprocessor=preprocessor,  # type: ignore[arg-type]
        vad=vad,  # type: ignore[arg-type]
        grpc_client=grpc_client,  # type: ignore[arg-type]
        postprocessor=None,
        on_event=on_event,
        enable_itn=False,
        state_machine=state_machine,
    )

    # Speech start -> ACTIVE
    vad.set_next_event(VADEvent(type=VADEventType.SPEECH_START, timestamp_ms=0))
    vad.is_speaking = True
    await session.process_frame(raw_frame)
    assert session.session_state == SessionState.ACTIVE

    # Speech end -> SILENCE
    current_time = 1.0
    vad.set_next_event(VADEvent(type=VADEventType.SPEECH_END, timestamp_ms=1000))
    vad.is_speaking = False
    await session.process_frame(raw_frame)
    await asyncio.sleep(0.01)
    assert session.session_state == SessionState.SILENCE

    # Advance time past SILENCE timeout (3s) -> should transition to HOLD
    current_time = 5.0
    result = await session.check_timeout()
    assert result == SessionState.HOLD
    assert session.session_state == SessionState.HOLD

    # Verify SessionHoldEvent was emitted
    from macaw.server.models.events import SessionHoldEvent

    hold_events = [e for e in events if isinstance(e, SessionHoldEvent)]
    assert len(hold_events) == 1

    # Advance time past HOLD timeout (10s) -> should transition to CLOSING
    current_time = 16.0
    result = await session.check_timeout()
    assert result == SessionState.CLOSING
    # CLOSING should auto-close (via _flush_and_close)
    # Note: in check_timeout, CLOSING transitions to CLOSED via _flush_and_close

    await session.close()
    assert session.is_closed


async def test_force_commit_via_ring_buffer_threshold() -> None:
    """Verifica que force commit dispara quando ring buffer atinge 90% de uso."""
    preprocessor = _LightPreprocessor()
    vad = _LightVAD()

    events: list[Any] = []

    async def on_event(event: Any) -> None:
        events.append(event)

    def _make_handle() -> _LightStreamHandle:
        return _LightStreamHandle(
            events=[
                TranscriptSegment(
                    text="force committed",
                    is_final=True,
                    segment_id=0,
                    start_ms=0,
                    end_ms=5000,
                    language="pt",
                    confidence=0.9,
                ),
            ]
        )

    grpc_client = _LightGRPCClient(handle_factory=_make_handle)

    # Small ring buffer: 1 second = 32000 bytes
    ring_buffer = RingBuffer(duration_s=1.0, sample_rate=16000, bytes_per_sample=2)
    wal = SessionWAL()

    session = StreamingSession(
        session_id="force_commit_test",
        preprocessor=preprocessor,  # type: ignore[arg-type]
        vad=vad,  # type: ignore[arg-type]
        grpc_client=grpc_client,  # type: ignore[arg-type]
        postprocessor=None,
        on_event=on_event,
        enable_itn=False,
        ring_buffer=ring_buffer,
        wal=wal,
    )

    # Speech start
    vad.set_next_event(VADEvent(type=VADEventType.SPEECH_START, timestamp_ms=0))
    vad.is_speaking = True

    # Use large frames to fill ring buffer quickly
    # 1024 samples * 2 bytes = 2048 bytes per frame
    # Ring buffer = 32000 bytes, 90% threshold = 28800 bytes
    # Need ~15 frames to trigger force commit
    large_frame = _make_raw_bytes(1024)

    # Process frames until force commit triggers
    force_committed = False
    for _i in range(20):
        await session.process_frame(large_frame)
        await asyncio.sleep(0.001)

        # Check if force commit happened (segment_id would advance)
        if session.segment_id > 0:
            force_committed = True
            break

    # Force commit should have triggered
    assert force_committed, (
        f"Force commit did not trigger after 20 frames. "
        f"Ring buffer: {ring_buffer.used_bytes}/{ring_buffer.capacity_bytes} "
        f"({ring_buffer.usage_percent:.1f}%)"
    )

    await session.close()


async def test_update_hot_words_mid_session() -> None:
    """Verifica que hot words podem ser atualizados durante a sessao."""
    preprocessor = _LightPreprocessor()
    vad = _LightVAD()
    raw_frame = _make_raw_bytes()

    events: list[Any] = []

    async def on_event(event: Any) -> None:
        events.append(event)

    handles: list[_LightStreamHandle] = []
    seg_counter = 0

    def _make_handle() -> _LightStreamHandle:
        nonlocal seg_counter
        handle = _LightStreamHandle(
            events=[
                TranscriptSegment(
                    text=f"seg {seg_counter}",
                    is_final=True,
                    segment_id=seg_counter,
                    start_ms=0,
                    end_ms=1000,
                    language="pt",
                    confidence=0.9,
                ),
            ]
        )
        handles.append(handle)
        seg_counter += 1
        return handle

    grpc_client = _LightGRPCClient(handle_factory=_make_handle)

    session = StreamingSession(
        session_id="update_hot_words_test",
        preprocessor=preprocessor,  # type: ignore[arg-type]
        vad=vad,  # type: ignore[arg-type]
        grpc_client=grpc_client,  # type: ignore[arg-type]
        postprocessor=None,
        on_event=on_event,
        hot_words=["PIX"],
        enable_itn=False,
    )

    # --- Segment 0: hot words = ["PIX"] ---

    vad.set_next_event(VADEvent(type=VADEventType.SPEECH_START, timestamp_ms=0))
    vad.is_speaking = True
    await session.process_frame(raw_frame)
    await session.process_frame(raw_frame)

    assert handles[0].sent_frames[0]["hot_words"] == ["PIX"]

    vad.set_next_event(VADEvent(type=VADEventType.SPEECH_END, timestamp_ms=1000))
    vad.is_speaking = False
    await session.process_frame(raw_frame)
    await asyncio.sleep(0.01)

    # --- Update hot words ---

    session.update_hot_words(["TED", "Selic"])

    # --- Segment 1: hot words = ["TED", "Selic"] ---

    vad.set_next_event(VADEvent(type=VADEventType.SPEECH_START, timestamp_ms=3000))
    vad.is_speaking = True
    await session.process_frame(raw_frame)
    await session.process_frame(raw_frame)

    assert len(handles) == 2
    assert handles[1].sent_frames[0]["hot_words"] == ["TED", "Selic"]

    vad.set_next_event(VADEvent(type=VADEventType.SPEECH_END, timestamp_ms=4000))
    vad.is_speaking = False
    await session.process_frame(raw_frame)
    await asyncio.sleep(0.01)
    await session.close()


async def test_recovery_with_ring_buffer_and_wal() -> None:
    """Verifica recovery apos crash com ring buffer e WAL reais.

    Simula:
    1. Segmento 0 completo (transcript.final -> WAL checkpoint)
    2. Segmento 1 em andamento (frames no ring buffer, nao commitados)
    3. Worker crash
    4. Recovery: reabre stream, reenvia uncommitted do ring buffer
    5. Restaura segment_id do WAL
    """
    preprocessor = _LightPreprocessor()
    vad = _LightVAD()
    raw_frame = _make_raw_bytes()

    events: list[Any] = []

    async def on_event(event: Any) -> None:
        events.append(event)

    call_count = 0
    crash_on_call = 2  # crash on second open_stream (during segment 1)

    def _make_handle() -> _LightStreamHandle:
        nonlocal call_count
        call_count += 1
        if call_count == crash_on_call:
            # This handle will simulate a crash when receiving events
            from macaw.exceptions import WorkerCrashError

            return _LightStreamHandle(events=[WorkerCrashError("simulated crash")])
        return _LightStreamHandle(
            events=[
                TranscriptSegment(
                    text="recovered seg",
                    is_final=True,
                    segment_id=0,
                    start_ms=0,
                    end_ms=1000,
                    language="pt",
                    confidence=0.9,
                ),
            ]
        )

    grpc_client = _LightGRPCClient(handle_factory=_make_handle)
    ring_buffer = RingBuffer(duration_s=10.0)
    wal = SessionWAL()

    session = StreamingSession(
        session_id="recovery_test",
        preprocessor=preprocessor,  # type: ignore[arg-type]
        vad=vad,  # type: ignore[arg-type]
        grpc_client=grpc_client,  # type: ignore[arg-type]
        postprocessor=None,
        on_event=on_event,
        enable_itn=False,
        ring_buffer=ring_buffer,
        wal=wal,
        recovery_timeout_s=5.0,
    )

    # --- Segment 0: complete successfully ---

    vad.set_next_event(VADEvent(type=VADEventType.SPEECH_START, timestamp_ms=0))
    vad.is_speaking = True
    await session.process_frame(raw_frame)

    for _ in range(3):
        await session.process_frame(raw_frame)

    vad.set_next_event(VADEvent(type=VADEventType.SPEECH_END, timestamp_ms=2000))
    vad.is_speaking = False
    await session.process_frame(raw_frame)
    await asyncio.sleep(0.05)

    # Verify segment 0 committed
    assert wal.last_committed_segment_id == 0
    assert ring_buffer.uncommitted_bytes == 0

    # --- Segment 1: will crash ---

    vad.set_next_event(VADEvent(type=VADEventType.SPEECH_START, timestamp_ms=4000))
    vad.is_speaking = True
    await session.process_frame(raw_frame)

    # Send frames (these go to ring buffer but worker will crash)
    for _ in range(3):
        await session.process_frame(raw_frame)

    # Give time for crash detection and recovery
    await asyncio.sleep(0.1)

    # Verify recovery happened
    error_events = [e for e in events if isinstance(e, StreamingErrorEvent)]
    recoverable_errors = [e for e in error_events if e.recoverable]
    assert len(recoverable_errors) >= 1, f"Expected recoverable error, got: {error_events}"

    # gRPC client should have opened at least 3 streams
    # (seg 0, seg 1 crash, recovery)
    assert grpc_client.open_stream_count >= 3

    # Session should still be alive (not CLOSED)
    assert not session.is_closed

    # Cleanup
    vad.set_next_event(VADEvent(type=VADEventType.SPEECH_END, timestamp_ms=6000))
    vad.is_speaking = False
    await session.process_frame(raw_frame)
    await asyncio.sleep(0.05)
    await session.close()


async def test_zero_errors_in_normal_multi_segment_session() -> None:
    """Sessao com multiplos segmentos: zero erros emitidos."""
    preprocessor = _LightPreprocessor()
    vad = _LightVAD()
    raw_frame = _make_raw_bytes()

    errors: list[StreamingErrorEvent] = []

    async def on_event(event: Any) -> None:
        if isinstance(event, StreamingErrorEvent):
            errors.append(event)

    seg_counter = 0

    def _make_handle() -> _LightStreamHandle:
        nonlocal seg_counter
        handle = _LightStreamHandle(
            events=[
                TranscriptSegment(
                    text=f"seg {seg_counter}",
                    is_final=True,
                    segment_id=seg_counter,
                    start_ms=0,
                    end_ms=1000,
                    language="pt",
                    confidence=0.9,
                ),
            ]
        )
        seg_counter += 1
        return handle

    grpc_client = _LightGRPCClient(handle_factory=_make_handle)

    session = StreamingSession(
        session_id="no_errors_test",
        preprocessor=preprocessor,  # type: ignore[arg-type]
        vad=vad,  # type: ignore[arg-type]
        grpc_client=grpc_client,  # type: ignore[arg-type]
        postprocessor=None,
        on_event=on_event,
        enable_itn=False,
        ring_buffer=RingBuffer(duration_s=10.0),
        wal=SessionWAL(),
        cross_segment_context=CrossSegmentContext(),
    )

    # Process 10 segments
    for seg in range(10):
        vad.set_next_event(
            VADEvent(type=VADEventType.SPEECH_START, timestamp_ms=seg * 2000),
        )
        vad.is_speaking = True
        await session.process_frame(raw_frame)

        for _ in range(3):
            await session.process_frame(raw_frame)

        vad.set_next_event(
            VADEvent(type=VADEventType.SPEECH_END, timestamp_ms=seg * 2000 + 1000),
        )
        vad.is_speaking = False
        await session.process_frame(raw_frame)
        await asyncio.sleep(0.01)

    await session.close()

    # Zero errors in normal operation
    assert len(errors) == 0, f"Unexpected errors: {[e.message for e in errors]}"
    assert session.segment_id >= 10
