"""Testes de integracao end-to-end para recovery de crash (M6-16).

Cenarios exclusivos nao cobertos por testes unitarios ou test_m6_integration.py:
1. Continuidade de segment_id apos recovery (sem duplicacao)
2. Recovery com ring buffer vazio (crash imediatamente apos commit)
3. Multiplas recoveries na mesma sessao (dois crashes consecutivos)

Usa _Light* mocks (sem unittest.mock.Mock) para evitar memory bloat.
Componentes M6 (state machine, ring buffer, WAL) sao reais.

Executar com:
    python -m pytest tests/integration/test_m6_recovery.py -v --tb=short
"""

from __future__ import annotations

import asyncio
from typing import TYPE_CHECKING, Any

import numpy as np

if TYPE_CHECKING:
    from collections.abc import Callable
import pytest

from macaw._types import SessionState, TranscriptSegment
from macaw.exceptions import WorkerCrashError
from macaw.server.models.events import StreamingErrorEvent, TranscriptFinalEvent
from macaw.session.ring_buffer import RingBuffer
from macaw.session.streaming import StreamingSession
from macaw.session.wal import SessionWAL
from macaw.vad.detector import VADEvent, VADEventType

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
    def __init__(self) -> None:
        self._frame = np.zeros(_FRAME_SIZE, dtype=np.float32)

    def process_frame(self, raw_bytes: bytes) -> np.ndarray:
        return self._frame


class _LightVAD:
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
    def process(self, text: str, **kwargs: object) -> str:
        return text


class _AsyncIterFromList:
    """Async iterator from list. Raises items that are Exception subclasses."""

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
    def __init__(self, events: list[Any]) -> None:
        self.is_closed = False
        self.session_id = "recovery_test"
        self._events = events

    def receive_events(self) -> _AsyncIterFromList:
        return _AsyncIterFromList(self._events)

    async def send_frame(
        self,
        *,
        pcm_data: bytes,
        initial_prompt: str | None = None,
        hot_words: list[str] | None = None,
    ) -> None:
        pass

    async def close(self) -> None:
        self.is_closed = True

    async def cancel(self) -> None:
        self.is_closed = True


class _LightGRPCClient:
    """gRPC client mock with counter-based handle factory."""

    def __init__(self, handle_factory: Callable[[], _LightStreamHandle]) -> None:
        self._handle_factory = handle_factory
        self.open_stream_count = 0

    async def open_stream(self, session_id: str) -> _LightStreamHandle:
        self.open_stream_count += 1
        return self._handle_factory()

    async def close(self) -> None:
        pass


def _make_raw_bytes(n_samples: int = _FRAME_SIZE) -> bytes:
    return np.zeros(n_samples, dtype=np.int16).tobytes()


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


async def _do_segment(
    session: StreamingSession,
    vad: _LightVAD,
    raw_frame: bytes,
    start_ms: int,
    end_ms: int,
    *,
    n_speech_frames: int = 3,
) -> None:
    """Execute a complete speech segment: speech_start -> frames -> speech_end."""
    vad.set_next_event(VADEvent(type=VADEventType.SPEECH_START, timestamp_ms=start_ms))
    vad.is_speaking = True
    await session.process_frame(raw_frame)

    for _ in range(n_speech_frames):
        await session.process_frame(raw_frame)

    vad.set_next_event(VADEvent(type=VADEventType.SPEECH_END, timestamp_ms=end_ms))
    vad.is_speaking = False
    await session.process_frame(raw_frame)
    await asyncio.sleep(0.05)


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


async def test_recovery_preserves_segment_id_continuity() -> None:
    """Crash during segment 2 -> recovery -> segment IDs are continuous.

    Flow: seg 0 (ok) -> seg 1 (ok) -> seg 2 (crash) -> recovery -> seg 2 (ok).
    Final segment_ids in transcript.final events: [0, 1, 2].
    """
    preprocessor = _LightPreprocessor()
    postprocessor = _LightPostprocessor()
    vad = _LightVAD()
    raw_frame = _make_raw_bytes()

    events: list[Any] = []

    async def on_event(event: Any) -> None:
        events.append(event)

    # Counter: handles 1 and 2 succeed, handle 3 crashes, handle 4 succeeds
    call_count = 0

    def _make_handle() -> _LightStreamHandle:
        nonlocal call_count
        call_count += 1
        if call_count == 3:
            # Third open_stream: seg 2, crash
            return _LightStreamHandle(events=[WorkerCrashError("crash during seg 2")])
        # Normal handle: transcript.final with segment_id based on session state
        # segment_id in event is informational; the real one comes from session
        return _LightStreamHandle(
            events=[
                TranscriptSegment(
                    text="texto do segmento",
                    is_final=True,
                    segment_id=0,
                    start_ms=0,
                    end_ms=2000,
                    language="pt",
                    confidence=0.9,
                ),
            ]
        )

    grpc_client = _LightGRPCClient(handle_factory=_make_handle)
    ring_buffer = RingBuffer(duration_s=10.0)
    wal = SessionWAL()

    session = StreamingSession(
        session_id="seg_id_continuity_test",
        preprocessor=preprocessor,  # type: ignore[arg-type]
        vad=vad,  # type: ignore[arg-type]
        grpc_client=grpc_client,  # type: ignore[arg-type]
        postprocessor=postprocessor,  # type: ignore[arg-type]
        on_event=on_event,
        enable_itn=False,
        ring_buffer=ring_buffer,
        wal=wal,
        recovery_timeout_s=5.0,
    )

    # --- Segment 0: ok ---
    await _do_segment(session, vad, raw_frame, start_ms=0, end_ms=2000)
    assert wal.last_committed_segment_id == 0

    # --- Segment 1: ok ---
    await _do_segment(session, vad, raw_frame, start_ms=3000, end_ms=5000)
    assert wal.last_committed_segment_id == 1

    # --- Segment 2: crash -> recovery -> ok ---
    # Start speech (opens handle 3, which crashes)
    vad.set_next_event(VADEvent(type=VADEventType.SPEECH_START, timestamp_ms=6000))
    vad.is_speaking = True
    await session.process_frame(raw_frame)

    # Send frames — crash will be detected by receiver task
    for _ in range(3):
        await session.process_frame(raw_frame)

    # Wait for crash detection and automatic recovery
    await asyncio.sleep(0.15)

    # Session should still be alive (recovery succeeded via handle 4)
    assert not session.is_closed

    # Recovery restores segment_id from WAL: last_committed=1 -> segment_id=2
    assert session.segment_id == 2

    # End speech on recovered stream
    vad.set_next_event(VADEvent(type=VADEventType.SPEECH_END, timestamp_ms=8000))
    vad.is_speaking = False
    await session.process_frame(raw_frame)
    await asyncio.sleep(0.05)

    # Verify: transcript.final segment IDs are [0, 1, 2]
    finals = [e for e in events if isinstance(e, TranscriptFinalEvent)]
    segment_ids = [f.segment_id for f in finals]
    assert segment_ids == [0, 1, 2], f"Expected [0, 1, 2], got {segment_ids}"

    # Verify: no irrecoverable errors
    irrecoverable = [e for e in events if isinstance(e, StreamingErrorEvent) and not e.recoverable]
    assert len(irrecoverable) == 0, f"Unexpected irrecoverable errors: {irrecoverable}"

    # At least 4 open_stream calls: seg 0, seg 1, seg 2 (crash), recovery
    assert grpc_client.open_stream_count >= 4

    await session.close()
    assert session.is_closed


async def test_recovery_with_empty_ring_buffer() -> None:
    """Crash immediately after commit (ring buffer has no uncommitted data).

    Recovery should succeed without resending any data.
    """
    preprocessor = _LightPreprocessor()
    vad = _LightVAD()
    raw_frame = _make_raw_bytes()

    events: list[Any] = []

    async def on_event(event: Any) -> None:
        events.append(event)

    call_count = 0

    def _make_handle() -> _LightStreamHandle:
        nonlocal call_count
        call_count += 1
        if call_count == 2:
            # Second open_stream: crash immediately
            return _LightStreamHandle(events=[WorkerCrashError("crash after commit")])
        return _LightStreamHandle(
            events=[
                TranscriptSegment(
                    text="segment ok",
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
        session_id="empty_rb_recovery_test",
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

    # --- Segment 0: complete successfully (commits everything) ---
    await _do_segment(session, vad, raw_frame, start_ms=0, end_ms=2000)
    assert wal.last_committed_segment_id == 0
    assert ring_buffer.uncommitted_bytes == 0

    # --- Segment 1: crash immediately ---
    # Speech start opens handle 2 (crash)
    vad.set_next_event(VADEvent(type=VADEventType.SPEECH_START, timestamp_ms=3000))
    vad.is_speaking = True
    await session.process_frame(raw_frame)

    # Wait for crash detection + recovery
    await asyncio.sleep(0.15)

    # Session should still be alive
    assert not session.is_closed

    # Verify recoverable error was emitted
    recoverable_errors = [
        e for e in events if isinstance(e, StreamingErrorEvent) and e.recoverable
    ]
    assert len(recoverable_errors) >= 1

    # No irrecoverable errors
    irrecoverable = [e for e in events if isinstance(e, StreamingErrorEvent) and not e.recoverable]
    assert len(irrecoverable) == 0

    await session.close()
    assert session.is_closed


async def test_multiple_recoveries_in_session() -> None:
    """Two crashes in the same session. Both recoveries succeed.

    Flow: seg 0 (ok) -> seg 1 (crash 1) -> recovery -> seg 1 (ok) ->
          seg 2 (crash 2) -> recovery -> seg 2 (ok).
    """
    preprocessor = _LightPreprocessor()
    postprocessor = _LightPostprocessor()
    vad = _LightVAD()
    raw_frame = _make_raw_bytes()

    events: list[Any] = []

    async def on_event(event: Any) -> None:
        events.append(event)

    # Handle sequence:
    # 1: seg 0 ok
    # 2: seg 1 crash
    # 3: seg 1 recovery ok
    # 4: seg 2 crash
    # 5: seg 2 recovery ok
    call_count = 0

    def _make_handle() -> _LightStreamHandle:
        nonlocal call_count
        call_count += 1
        if call_count in (2, 4):
            return _LightStreamHandle(events=[WorkerCrashError(f"crash #{call_count}")])
        return _LightStreamHandle(
            events=[
                TranscriptSegment(
                    text=f"handle {call_count}",
                    is_final=True,
                    segment_id=0,
                    start_ms=0,
                    end_ms=2000,
                    language="pt",
                    confidence=0.9,
                ),
            ]
        )

    grpc_client = _LightGRPCClient(handle_factory=_make_handle)
    ring_buffer = RingBuffer(duration_s=10.0)
    wal = SessionWAL()

    session = StreamingSession(
        session_id="multi_recovery_test",
        preprocessor=preprocessor,  # type: ignore[arg-type]
        vad=vad,  # type: ignore[arg-type]
        grpc_client=grpc_client,  # type: ignore[arg-type]
        postprocessor=postprocessor,  # type: ignore[arg-type]
        on_event=on_event,
        enable_itn=False,
        ring_buffer=ring_buffer,
        wal=wal,
        recovery_timeout_s=5.0,
    )

    # --- Segment 0: ok (handle 1) ---
    await _do_segment(session, vad, raw_frame, start_ms=0, end_ms=2000)
    assert wal.last_committed_segment_id == 0

    # --- Segment 1: crash (handle 2) -> recovery (handle 3) ---
    vad.set_next_event(VADEvent(type=VADEventType.SPEECH_START, timestamp_ms=3000))
    vad.is_speaking = True
    await session.process_frame(raw_frame)

    for _ in range(2):
        await session.process_frame(raw_frame)

    # Wait for crash detection + recovery
    await asyncio.sleep(0.15)

    assert not session.is_closed
    assert session.segment_id == 1  # WAL last=0, so segment_id=1

    # Complete segment 1 on recovered stream (handle 3)
    vad.set_next_event(VADEvent(type=VADEventType.SPEECH_END, timestamp_ms=5000))
    vad.is_speaking = False
    await session.process_frame(raw_frame)
    await asyncio.sleep(0.05)

    assert wal.last_committed_segment_id == 1

    # --- Segment 2: crash (handle 4) -> recovery (handle 5) ---
    vad.set_next_event(VADEvent(type=VADEventType.SPEECH_START, timestamp_ms=6000))
    vad.is_speaking = True
    await session.process_frame(raw_frame)

    for _ in range(2):
        await session.process_frame(raw_frame)

    # Wait for crash detection + recovery
    await asyncio.sleep(0.15)

    assert not session.is_closed
    assert session.segment_id == 2  # WAL last=1, so segment_id=2

    # Complete segment 2 on recovered stream (handle 5)
    vad.set_next_event(VADEvent(type=VADEventType.SPEECH_END, timestamp_ms=8000))
    vad.is_speaking = False
    await session.process_frame(raw_frame)
    await asyncio.sleep(0.05)

    # Verify all finals
    finals = [e for e in events if isinstance(e, TranscriptFinalEvent)]
    segment_ids = [f.segment_id for f in finals]
    assert segment_ids == [0, 1, 2], f"Expected [0, 1, 2], got {segment_ids}"

    # Verify exactly 2 recoverable errors (one per crash)
    recoverable_errors = [
        e for e in events if isinstance(e, StreamingErrorEvent) and e.recoverable
    ]
    assert len(recoverable_errors) == 2, (
        f"Expected 2 recoverable errors, got {len(recoverable_errors)}"
    )

    # No irrecoverable errors
    irrecoverable = [e for e in events if isinstance(e, StreamingErrorEvent) and not e.recoverable]
    assert len(irrecoverable) == 0, f"Unexpected irrecoverable errors: {irrecoverable}"

    # 5 open_stream calls total
    assert grpc_client.open_stream_count == 5

    await session.close()
    assert session.is_closed
    assert session.session_state == SessionState.CLOSED
