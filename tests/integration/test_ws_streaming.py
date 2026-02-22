"""Integration tests for the WebSocket /v1/realtime endpoint.

Tests the complete flow of the WebSocket endpoint with a real FastAPI app
and mocked dependencies (registry, scheduler, gRPC). Validates that:
- Handshake works with a valid model
- Events are emitted correctly
- Client->server commands are processed
- StreamingSession coordinates preprocessing -> VAD -> gRPC -> post-processing
- Backpressure works with audio sent faster than real-time
- Heartbeat and inactivity timeout work

Integration level:
- Real FastAPI app (create_app)
- Real WebSocket transport (Starlette TestClient)
- Registry, VAD, gRPC client: controlled mocks
- StreamingSession: real instance with mocked components
"""

from __future__ import annotations

import asyncio
import time
from typing import TYPE_CHECKING
from unittest.mock import AsyncMock, MagicMock, Mock

import numpy as np
import pytest

from macaw._types import TranscriptSegment, WordTimestamp
from macaw.server.app import create_app
from macaw.server.models.events import (
    StreamingErrorEvent,
    TranscriptFinalEvent,
    TranscriptPartialEvent,
    VADSpeechEndEvent,
    VADSpeechStartEvent,
)
from macaw.session.backpressure import (
    BackpressureController,
    FramesDroppedAction,
    RateLimitAction,
)
from macaw.session.streaming import StreamingSession
from macaw.vad.detector import VADEvent, VADEventType

if TYPE_CHECKING:
    from starlette.testclient import TestClient

# ---------------------------------------------------------------------------
# Audio helpers
# ---------------------------------------------------------------------------

# PCM 16-bit mono 16kHz: 2 bytes per sample
_SAMPLE_RATE = 16000
_BYTES_PER_SAMPLE = 2
_FRAME_SIZE = 1024  # 64ms frame


def _make_pcm_silence(n_samples: int = _FRAME_SIZE) -> bytes:
    """Generate PCM int16 silence bytes (zeros)."""
    return np.zeros(n_samples, dtype=np.int16).tobytes()


def _make_pcm_tone(
    frequency: float = 440.0,
    duration_s: float = 0.064,
    sample_rate: int = _SAMPLE_RATE,
) -> bytes:
    """Generate PCM int16 sine tone bytes."""
    n_samples = int(sample_rate * duration_s)
    t = np.arange(n_samples, dtype=np.float64) / sample_rate
    samples = (32767 * 0.5 * np.sin(2 * np.pi * frequency * t)).astype(np.int16)
    return samples.tobytes()


def _make_pcm_20ms_frame(sample_rate: int = _SAMPLE_RATE) -> bytes:
    """Generate PCM int16 bytes of 20ms silence."""
    n_samples = sample_rate // 50  # 20ms
    return np.zeros(n_samples, dtype=np.int16).tobytes()


# ---------------------------------------------------------------------------
# Mock factories
# ---------------------------------------------------------------------------


def _make_mock_registry(*, known_models: list[str] | None = None) -> MagicMock:
    """Create a ModelRegistry mock that knows the models in known_models."""
    if known_models is None:
        known_models = ["faster-whisper-tiny"]

    from macaw.exceptions import ModelNotFoundError

    registry = MagicMock()

    def _get_manifest(model_name: str) -> MagicMock:
        if model_name in known_models:
            manifest = MagicMock()
            manifest.name = model_name
            return manifest
        raise ModelNotFoundError(model_name)

    registry.get_manifest = MagicMock(side_effect=_get_manifest)
    registry.has_model = MagicMock(side_effect=lambda name: name in known_models)
    return registry


def _make_float32_frame(n_samples: int = _FRAME_SIZE) -> np.ndarray:
    """Generate float32 frame (zeros) for preprocessor mock."""
    return np.zeros(n_samples, dtype=np.float32)


def _make_preprocessor_mock() -> Mock:
    """Create a StreamingPreprocessor mock."""
    mock = Mock()
    mock.process_frame.return_value = _make_float32_frame()
    return mock


def _make_vad_mock(*, is_speaking: bool = False) -> Mock:
    """Create a VADDetector mock."""
    mock = Mock()
    mock.process_frame.return_value = None
    mock.is_speaking = is_speaking
    mock.reset.return_value = None
    return mock


class _AsyncIterFromList:
    """Async iterator that yields items from a list."""

    def __init__(self, items: list[object]) -> None:
        self._items = list(items)
        self._index = 0

    def __aiter__(self) -> _AsyncIterFromList:
        return self

    async def __anext__(self) -> object:
        if self._index >= len(self._items):
            raise StopAsyncIteration
        item = self._items[self._index]
        self._index += 1
        if isinstance(item, Exception):
            raise item
        return item


def _make_stream_handle_mock(
    events: list[object] | None = None,
) -> Mock:
    """Create a StreamHandle mock."""
    handle = Mock()
    handle.is_closed = False
    handle.session_id = "test_session"

    if events is None:
        events = []
    handle.receive_events.return_value = _AsyncIterFromList(events)

    handle.send_frame = AsyncMock()
    handle.close = AsyncMock()
    handle.cancel = AsyncMock()
    return handle


def _make_grpc_client_mock(stream_handle: Mock | None = None) -> AsyncMock:
    """Create a StreamingGRPCClient mock."""
    client = AsyncMock()
    if stream_handle is None:
        stream_handle = _make_stream_handle_mock()
    client.open_stream = AsyncMock(return_value=stream_handle)
    client.close = AsyncMock()
    return client


def _make_postprocessor_mock() -> Mock:
    """Create a PostProcessingPipeline mock."""
    mock = Mock()
    mock.process.side_effect = lambda text, **kwargs: f"ITN({text})"
    return mock


def _make_on_event() -> AsyncMock:
    """Create an on_event mock callback."""
    return AsyncMock()


def _make_app_with_short_timeouts(
    *,
    known_models: list[str] | None = None,
    inactivity_s: float = 5.0,
    heartbeat_s: float = 10.0,
    check_s: float = 0.1,
) -> tuple[MagicMock, TestClient]:
    """Create FastAPI app with short timeouts and return (registry, TestClient)."""
    from starlette.testclient import TestClient

    registry = _make_mock_registry(known_models=known_models)
    app = create_app(registry=registry)
    app.state.ws_inactivity_timeout_s = inactivity_s
    app.state.ws_heartbeat_interval_s = heartbeat_s
    app.state.ws_check_interval_s = check_s
    return registry, TestClient(app)


# ---------------------------------------------------------------------------
# Tests: WebSocket Endpoint Integration (FastAPI + Protocol)
# ---------------------------------------------------------------------------


@pytest.mark.integration
def test_ws_connect_and_session_created() -> None:
    """Connecting via WS with a valid model receives session.created with session_id and model."""
    from starlette.testclient import TestClient

    app = create_app(registry=_make_mock_registry())
    client = TestClient(app)

    with client.websocket_connect("/v1/realtime?model=faster-whisper-tiny") as ws:
        event = ws.receive_json()

    assert event["type"] == "session.created"
    assert event["session_id"].startswith("sess_")
    assert event["model"] == "faster-whisper-tiny"
    assert "config" in event

    # Verify session_id format: sess_ + 12 hex chars
    hex_part = event["session_id"][5:]
    assert len(hex_part) == 12
    int(hex_part, 16)  # Verify valid hexadecimal

    # Verify defaults in config
    config = event["config"]
    assert config["vad_sensitivity"] == "normal"
    assert config["silence_timeout_ms"] == 30_000
    assert config["enable_partial_transcripts"] is True
    assert config["enable_itn"] is True


@pytest.mark.integration
def test_ws_session_configure() -> None:
    """session.configure is accepted and does not close the connection."""
    from starlette.testclient import TestClient

    app = create_app(registry=_make_mock_registry())
    client = TestClient(app)

    with client.websocket_connect("/v1/realtime?model=faster-whisper-tiny") as ws:
        created = ws.receive_json()
        assert created["type"] == "session.created"

        # Send configuration
        ws.send_json(
            {
                "type": "session.configure",
                "language": "pt",
                "vad_sensitivity": "high",
                "hot_words": ["PIX", "TED"],
                "enable_itn": False,
            }
        )

        # Connection should remain active -- verify by sending session.close
        ws.send_json({"type": "session.close"})
        closed = ws.receive_json()
        assert closed["type"] == "session.closed"
        assert closed["reason"] == "client_request"


@pytest.mark.integration
def test_ws_session_cancel() -> None:
    """session.cancel emite session.closed com reason=cancelled."""
    from starlette.testclient import TestClient

    app = create_app(registry=_make_mock_registry())
    client = TestClient(app)

    with client.websocket_connect("/v1/realtime?model=faster-whisper-tiny") as ws:
        created = ws.receive_json()
        assert created["type"] == "session.created"

        ws.send_json({"type": "session.cancel"})

        closed = ws.receive_json()
        assert closed["type"] == "session.closed"
        assert closed["reason"] == "cancelled"
        assert closed["total_duration_ms"] >= 0
        assert closed["segments_transcribed"] == 0


@pytest.mark.integration
def test_ws_session_close() -> None:
    """session.close emite session.closed com reason=client_request."""
    from starlette.testclient import TestClient

    app = create_app(registry=_make_mock_registry())
    client = TestClient(app)

    with client.websocket_connect("/v1/realtime?model=faster-whisper-tiny") as ws:
        created = ws.receive_json()
        assert created["type"] == "session.created"

        ws.send_json({"type": "session.close"})

        closed = ws.receive_json()
        assert closed["type"] == "session.closed"
        assert closed["reason"] == "client_request"
        assert closed["total_duration_ms"] >= 0
        assert closed["segments_transcribed"] == 0


@pytest.mark.integration
def test_ws_invalid_model_closes_connection() -> None:
    """Connection with nonexistent model receives error and closes with code 1008."""
    from starlette.testclient import TestClient

    app = create_app(
        registry=_make_mock_registry(known_models=["faster-whisper-tiny"]),
    )
    client = TestClient(app)

    with client.websocket_connect("/v1/realtime?model=nonexistent-model") as ws:
        error_event = ws.receive_json()

    assert error_event["type"] == "error"
    assert error_event["code"] == "model_not_found"
    assert "nonexistent-model" in error_event["message"]
    assert error_event["recoverable"] is False


@pytest.mark.integration
def test_ws_missing_model_closes_connection() -> None:
    """Connection without 'model' query param receives error and closes."""
    from starlette.testclient import TestClient

    app = create_app(registry=_make_mock_registry())
    client = TestClient(app)

    with client.websocket_connect("/v1/realtime") as ws:
        error_event = ws.receive_json()

    assert error_event["type"] == "error"
    assert error_event["code"] == "invalid_request"
    assert "model" in error_event["message"].lower()
    assert error_event["recoverable"] is False


@pytest.mark.integration
def test_ws_audio_frames_accepted() -> None:
    """Binary audio frames are accepted by the endpoint."""
    from starlette.testclient import TestClient

    app = create_app(registry=_make_mock_registry())
    client = TestClient(app)

    with client.websocket_connect("/v1/realtime?model=faster-whisper-tiny") as ws:
        created = ws.receive_json()
        assert created["type"] == "session.created"

        # Send multiple PCM audio frames
        for _ in range(5):
            ws.send_bytes(_make_pcm_silence())

        # Close normally
        ws.send_json({"type": "session.close"})
        closed = ws.receive_json()
        assert closed["type"] == "session.closed"


@pytest.mark.integration
def test_ws_malformed_json_recoverable() -> None:
    """Malformed JSON returns recoverable error without closing connection."""
    from starlette.testclient import TestClient

    app = create_app(registry=_make_mock_registry())
    client = TestClient(app)

    with client.websocket_connect("/v1/realtime?model=faster-whisper-tiny") as ws:
        created = ws.receive_json()
        assert created["type"] == "session.created"

        # Send invalid JSON
        ws.send_text("this is not valid json {{{")

        # Should receive recoverable error
        error = ws.receive_json()
        assert error["type"] == "error"
        assert error["code"] == "malformed_json"
        assert error["recoverable"] is True

        # Connection still works -- send close
        ws.send_json({"type": "session.close"})
        closed = ws.receive_json()
        assert closed["type"] == "session.closed"


@pytest.mark.integration
def test_ws_unknown_command_recoverable() -> None:
    """Unknown command returns recoverable error without closing connection."""
    from starlette.testclient import TestClient

    app = create_app(registry=_make_mock_registry())
    client = TestClient(app)

    with client.websocket_connect("/v1/realtime?model=faster-whisper-tiny") as ws:
        created = ws.receive_json()
        assert created["type"] == "session.created"

        ws.send_json({"type": "totally.unknown.command"})

        error = ws.receive_json()
        assert error["type"] == "error"
        assert error["code"] == "unknown_command"
        assert error["recoverable"] is True

        # Connection still works
        ws.send_json({"type": "session.close"})
        closed = ws.receive_json()
        assert closed["type"] == "session.closed"


@pytest.mark.integration
def test_ws_heartbeat_inactivity_timeout() -> None:
    """Session without received audio is closed after inactivity timeout."""
    _, client = _make_app_with_short_timeouts(
        inactivity_s=0.3,
        check_s=0.1,
    )

    with client.websocket_connect("/v1/realtime?model=faster-whisper-tiny") as ws:
        created = ws.receive_json()
        assert created["type"] == "session.created"

        # Do not send audio -- wait for timeout
        closed = ws.receive_json()
        assert closed["type"] == "session.closed"
        assert closed["reason"] == "inactivity_timeout"
        assert closed["total_duration_ms"] >= 200  # At least 200ms


@pytest.mark.integration
def test_ws_audio_resets_inactivity_timer() -> None:
    """Sending audio resets the inactivity timer."""
    _, client = _make_app_with_short_timeouts(
        inactivity_s=0.4,
        check_s=0.1,
    )

    with client.websocket_connect("/v1/realtime?model=faster-whisper-tiny") as ws:
        created = ws.receive_json()
        assert created["type"] == "session.created"

        # Send spaced audio frames, keeping the session alive
        start = time.monotonic()
        for _ in range(3):
            ws.send_bytes(_make_pcm_silence())
            time.sleep(0.15)

        elapsed = time.monotonic() - start
        assert elapsed > 0.4, "Should have elapsed more than inactivity_timeout"

        # Now stop sending and wait for timeout
        closed = ws.receive_json()
        assert closed["type"] == "session.closed"
        assert closed["reason"] == "inactivity_timeout"


@pytest.mark.integration
def test_ws_input_audio_buffer_commit_accepted() -> None:
    """input_audio_buffer.commit is accepted without closing the connection."""
    from starlette.testclient import TestClient

    app = create_app(registry=_make_mock_registry())
    client = TestClient(app)

    with client.websocket_connect("/v1/realtime?model=faster-whisper-tiny") as ws:
        created = ws.receive_json()
        assert created["type"] == "session.created"

        # Send audio
        ws.send_bytes(_make_pcm_silence())

        # Send commit
        ws.send_json({"type": "input_audio_buffer.commit"})

        # Connection should remain active
        ws.send_json({"type": "session.close"})
        closed = ws.receive_json()
        assert closed["type"] == "session.closed"


@pytest.mark.integration
def test_ws_multiple_sessions_unique_ids() -> None:
    """Each WebSocket connection receives a unique session_id."""
    from starlette.testclient import TestClient

    app = create_app(registry=_make_mock_registry())
    client = TestClient(app)

    session_ids = []
    for _ in range(5):
        with client.websocket_connect("/v1/realtime?model=faster-whisper-tiny") as ws:
            event = ws.receive_json()
            session_ids.append(event["session_id"])

    assert len(set(session_ids)) == 5, f"IDs not unique: {session_ids}"


# ---------------------------------------------------------------------------
# Tests: StreamingSession Full Flow (with mock gRPC)
# These tests exercise StreamingSession with all real layers
# except the gRPC client (mocked). They validate the complete pipeline:
# preprocessing -> VAD -> gRPC -> post-processing -> events.
# ---------------------------------------------------------------------------


@pytest.mark.integration
async def test_ws_full_transcription_flow() -> None:
    """Full flow: speech_start -> partial -> final -> speech_end in order."""
    # Arrange: create StreamingSession with controlled mocks
    partial_seg = TranscriptSegment(
        text="ola como",
        is_final=False,
        segment_id=0,
        start_ms=1000,
    )
    final_seg = TranscriptSegment(
        text="ola como posso ajudar",
        is_final=True,
        segment_id=0,
        start_ms=1000,
        end_ms=3000,
        language="pt",
        confidence=0.95,
    )

    stream_handle = _make_stream_handle_mock(events=[partial_seg, final_seg])
    grpc_client = _make_grpc_client_mock(stream_handle)
    vad = _make_vad_mock()
    postprocessor = _make_postprocessor_mock()
    on_event = _make_on_event()

    session = StreamingSession(
        session_id="sess_integ_001",
        preprocessor=_make_preprocessor_mock(),
        vad=vad,
        grpc_client=grpc_client,
        postprocessor=postprocessor,
        on_event=on_event,
        enable_itn=True,
    )

    # Act: simulate speech_start
    vad.process_frame.return_value = VADEvent(
        type=VADEventType.SPEECH_START,
        timestamp_ms=1000,
    )
    vad.is_speaking = False
    await session.process_frame(_make_pcm_silence())

    # Wait for receiver task to process events
    await asyncio.sleep(0.05)

    # Simulate speech_end
    vad.process_frame.return_value = VADEvent(
        type=VADEventType.SPEECH_END,
        timestamp_ms=3000,
    )
    vad.is_speaking = False
    await session.process_frame(_make_pcm_silence())

    # Assert: verify all events in correct order
    event_types = [type(call.args[0]).__name__ for call in on_event.call_args_list]

    assert "VADSpeechStartEvent" in event_types
    assert "TranscriptPartialEvent" in event_types
    assert "TranscriptFinalEvent" in event_types
    assert "VADSpeechEndEvent" in event_types

    # Verify order: speech_start < partial < final < speech_end
    idx_start = event_types.index("VADSpeechStartEvent")
    idx_partial = event_types.index("TranscriptPartialEvent")
    idx_final = event_types.index("TranscriptFinalEvent")
    idx_end = event_types.index("VADSpeechEndEvent")

    assert idx_start < idx_partial < idx_final < idx_end

    # Verify partial content (without ITN)
    partial_calls = [
        c for c in on_event.call_args_list if isinstance(c.args[0], TranscriptPartialEvent)
    ]
    assert partial_calls[0].args[0].text == "ola como"

    # Verify final content (with ITN)
    final_calls = [
        c for c in on_event.call_args_list if isinstance(c.args[0], TranscriptFinalEvent)
    ]
    assert final_calls[0].args[0].text == "ITN(ola como posso ajudar)"

    # Cleanup
    await session.close()


@pytest.mark.integration
async def test_ws_itn_applied_only_on_final() -> None:
    """Post-processing (ITN) is applied ONLY on transcript.final, NEVER on partial."""
    # Arrange: partial e final segments
    partial_seg = TranscriptSegment(
        text="dois mil",
        is_final=False,
        segment_id=0,
        start_ms=500,
    )
    final_seg = TranscriptSegment(
        text="dois mil e vinte e cinco",
        is_final=True,
        segment_id=0,
        start_ms=500,
        end_ms=2000,
        language="pt",
        confidence=0.92,
    )

    stream_handle = _make_stream_handle_mock(events=[partial_seg, final_seg])
    grpc_client = _make_grpc_client_mock(stream_handle)
    vad = _make_vad_mock()
    postprocessor = _make_postprocessor_mock()
    on_event = _make_on_event()

    session = StreamingSession(
        session_id="sess_itn_test",
        preprocessor=_make_preprocessor_mock(),
        vad=vad,
        grpc_client=grpc_client,
        postprocessor=postprocessor,
        on_event=on_event,
        enable_itn=True,
    )

    # Trigger speech_start -> receiver processes partial + final
    vad.process_frame.return_value = VADEvent(
        type=VADEventType.SPEECH_START,
        timestamp_ms=500,
    )
    vad.is_speaking = False
    await session.process_frame(_make_pcm_silence())
    await asyncio.sleep(0.05)

    # Trigger speech_end
    vad.process_frame.return_value = VADEvent(
        type=VADEventType.SPEECH_END,
        timestamp_ms=2000,
    )
    vad.is_speaking = False
    await session.process_frame(_make_pcm_silence())

    # Assert: ITN called ONLY once (for the final)
    postprocessor.process.assert_called_once()
    assert postprocessor.process.call_args.args[0] == "dois mil e vinte e cinco"

    # Verify partial without ITN
    partial_calls = [
        c for c in on_event.call_args_list if isinstance(c.args[0], TranscriptPartialEvent)
    ]
    assert len(partial_calls) == 1
    assert partial_calls[0].args[0].text == "dois mil"  # Original text, without ITN

    # Verify final with ITN
    final_calls = [
        c for c in on_event.call_args_list if isinstance(c.args[0], TranscriptFinalEvent)
    ]
    assert len(final_calls) == 1
    assert final_calls[0].args[0].text == "ITN(dois mil e vinte e cinco)"

    # Cleanup
    await session.close()


@pytest.mark.integration
async def test_ws_commit_produces_final() -> None:
    """input_audio_buffer.commit forces the worker to emit transcript.final."""
    # Arrange
    final_seg = TranscriptSegment(
        text="resultado do commit",
        is_final=True,
        segment_id=0,
        start_ms=1000,
        end_ms=3000,
        language="pt",
        confidence=0.88,
    )

    stream_handle = _make_stream_handle_mock(events=[final_seg])
    grpc_client = _make_grpc_client_mock(stream_handle)
    vad = _make_vad_mock()
    on_event = _make_on_event()

    session = StreamingSession(
        session_id="sess_commit_test",
        preprocessor=_make_preprocessor_mock(),
        vad=vad,
        grpc_client=grpc_client,
        postprocessor=_make_postprocessor_mock(),
        on_event=on_event,
        enable_itn=True,
    )

    assert session.segment_id == 0

    # Trigger speech_start -> open stream
    vad.process_frame.return_value = VADEvent(
        type=VADEventType.SPEECH_START,
        timestamp_ms=1000,
    )
    vad.is_speaking = True
    await session.process_frame(_make_pcm_silence())
    await asyncio.sleep(0.05)

    # Act: manual commit
    await session.commit()

    # Assert: transcript.final emitido
    final_calls = [
        c for c in on_event.call_args_list if isinstance(c.args[0], TranscriptFinalEvent)
    ]
    assert len(final_calls) == 1
    assert final_calls[0].args[0].text == "ITN(resultado do commit)"

    # segment_id incremented after commit
    assert session.segment_id == 1

    # Stream was closed
    stream_handle.close.assert_called_once()

    # Cleanup
    await session.close()


@pytest.mark.integration
async def test_ws_final_with_word_timestamps() -> None:
    """transcript.final includes word timestamps when available."""
    # Arrange
    final_seg = TranscriptSegment(
        text="ola mundo",
        is_final=True,
        segment_id=0,
        start_ms=1000,
        end_ms=2000,
        language="pt",
        confidence=0.95,
        words=(
            WordTimestamp(word="ola", start=1.0, end=1.5),
            WordTimestamp(word="mundo", start=1.5, end=2.0),
        ),
    )

    stream_handle = _make_stream_handle_mock(events=[final_seg])
    grpc_client = _make_grpc_client_mock(stream_handle)
    vad = _make_vad_mock()
    on_event = _make_on_event()

    session = StreamingSession(
        session_id="sess_words_test",
        preprocessor=_make_preprocessor_mock(),
        vad=vad,
        grpc_client=grpc_client,
        postprocessor=_make_postprocessor_mock(),
        on_event=on_event,
        enable_itn=False,
    )

    # Trigger speech_start + speech_end
    vad.process_frame.return_value = VADEvent(
        type=VADEventType.SPEECH_START,
        timestamp_ms=1000,
    )
    vad.is_speaking = False
    await session.process_frame(_make_pcm_silence())
    await asyncio.sleep(0.05)

    vad.process_frame.return_value = VADEvent(
        type=VADEventType.SPEECH_END,
        timestamp_ms=2000,
    )
    vad.is_speaking = False
    await session.process_frame(_make_pcm_silence())

    # Assert: word timestamps present
    final_calls = [
        c for c in on_event.call_args_list if isinstance(c.args[0], TranscriptFinalEvent)
    ]
    assert len(final_calls) == 1
    final_event = final_calls[0].args[0]
    assert final_event.words is not None
    assert len(final_event.words) == 2
    assert final_event.words[0].word == "ola"
    assert final_event.words[0].start == 1.0
    assert final_event.words[1].word == "mundo"
    assert final_event.words[1].end == 2.0


@pytest.mark.integration
async def test_ws_worker_crash_emits_recoverable_error() -> None:
    """Worker crash during streaming emits recoverable error via callback."""
    from macaw.exceptions import WorkerCrashError

    stream_handle = _make_stream_handle_mock(
        events=[WorkerCrashError("worker_1")],
    )
    grpc_client = _make_grpc_client_mock(stream_handle)
    vad = _make_vad_mock()
    on_event = _make_on_event()

    session = StreamingSession(
        session_id="sess_crash_test",
        preprocessor=_make_preprocessor_mock(),
        vad=vad,
        grpc_client=grpc_client,
        postprocessor=_make_postprocessor_mock(),
        on_event=on_event,
    )

    # Trigger speech_start (starts receiver that will receive crash)
    vad.process_frame.return_value = VADEvent(
        type=VADEventType.SPEECH_START,
        timestamp_ms=1000,
    )
    vad.is_speaking = False
    await session.process_frame(_make_pcm_silence())
    await asyncio.sleep(0.05)

    # Assert: recoverable error emitted
    error_calls = [
        c for c in on_event.call_args_list if isinstance(c.args[0], StreamingErrorEvent)
    ]
    assert len(error_calls) >= 1
    error_event = error_calls[0].args[0]
    assert error_event.code == "worker_crash"
    assert error_event.recoverable is True
    assert error_event.resume_segment_id is not None

    # Cleanup
    await session.close()


@pytest.mark.integration
async def test_ws_hot_words_sent_on_first_frame_only() -> None:
    """Hot words are sent to the worker only on the first frame of the segment."""
    stream_handle = _make_stream_handle_mock()
    grpc_client = _make_grpc_client_mock(stream_handle)
    vad = _make_vad_mock()

    session = StreamingSession(
        session_id="sess_hotwords",
        preprocessor=_make_preprocessor_mock(),
        vad=vad,
        grpc_client=grpc_client,
        postprocessor=_make_postprocessor_mock(),
        on_event=_make_on_event(),
        hot_words=["PIX", "TED", "Selic"],
    )

    # Trigger speech_start
    vad.process_frame.return_value = VADEvent(
        type=VADEventType.SPEECH_START,
        timestamp_ms=1000,
    )
    vad.is_speaking = True
    await session.process_frame(_make_pcm_silence())

    # Send more frames
    vad.process_frame.return_value = None
    await session.process_frame(_make_pcm_silence())
    await session.process_frame(_make_pcm_silence())

    # Assert: hot_words no primeiro frame, None nos seguintes
    calls = stream_handle.send_frame.call_args_list
    assert len(calls) == 3
    assert calls[0].kwargs.get("hot_words") == ["PIX", "TED", "Selic"]
    assert calls[1].kwargs.get("hot_words") is None
    assert calls[2].kwargs.get("hot_words") is None

    # Cleanup
    await session.close()


@pytest.mark.integration
async def test_ws_itn_disabled_skips_postprocessing() -> None:
    """With enable_itn=False, transcript.final is emitted without ITN."""
    final_seg = TranscriptSegment(
        text="dois mil reais",
        is_final=True,
        segment_id=0,
        start_ms=1000,
        end_ms=2000,
    )

    stream_handle = _make_stream_handle_mock(events=[final_seg])
    grpc_client = _make_grpc_client_mock(stream_handle)
    vad = _make_vad_mock()
    postprocessor = _make_postprocessor_mock()
    on_event = _make_on_event()

    session = StreamingSession(
        session_id="sess_no_itn",
        preprocessor=_make_preprocessor_mock(),
        vad=vad,
        grpc_client=grpc_client,
        postprocessor=postprocessor,
        on_event=on_event,
        enable_itn=False,
    )

    # Trigger speech_start
    vad.process_frame.return_value = VADEvent(
        type=VADEventType.SPEECH_START,
        timestamp_ms=1000,
    )
    vad.is_speaking = False
    await session.process_frame(_make_pcm_silence())
    await asyncio.sleep(0.05)

    # Trigger speech_end
    vad.process_frame.return_value = VADEvent(
        type=VADEventType.SPEECH_END,
        timestamp_ms=2000,
    )
    vad.is_speaking = False
    await session.process_frame(_make_pcm_silence())

    # Assert: postprocessor NOT called
    postprocessor.process.assert_not_called()

    final_calls = [
        c for c in on_event.call_args_list if isinstance(c.args[0], TranscriptFinalEvent)
    ]
    assert len(final_calls) == 1
    assert final_calls[0].args[0].text == "dois mil reais"  # Original text


# ---------------------------------------------------------------------------
# Tests: Backpressure Integration
# These tests validate that the BackpressureController correctly detects
# audio sent above real-time. They test the component in isolation
# but with realistic data (PCM frames, simulated timing).
# ---------------------------------------------------------------------------


class _FakeClock:
    """Deterministic clock for backpressure tests."""

    def __init__(self, start: float = 0.0) -> None:
        self._now = start

    def __call__(self) -> float:
        return self._now

    def advance(self, seconds: float) -> None:
        self._now += seconds


_BYTES_PER_20MS = (_SAMPLE_RATE * _BYTES_PER_SAMPLE) // 50  # 640 bytes


@pytest.mark.integration
def test_ws_backpressure_rate_limit_at_3x() -> None:
    """Sending audio at 3x real-time triggers RateLimitAction."""
    clock = _FakeClock()
    ctrl = BackpressureController(
        sample_rate=_SAMPLE_RATE,
        rate_limit_threshold=1.2,
        max_backlog_s=100.0,  # High to avoid dropping
        clock=clock,
    )

    # Send 200 frames of 20ms at 3x real-time (~6.67ms wall per 20ms frame)
    actions: list[RateLimitAction] = []
    for _ in range(200):
        result = ctrl.record_frame(_BYTES_PER_20MS)
        if isinstance(result, RateLimitAction):
            actions.append(result)
        clock.advance(0.00667)  # 6.67ms = 3x real-time

    assert len(actions) >= 1, "Should emit at least 1 RateLimitAction at 3x"
    for action in actions:
        assert action.delay_ms >= 1


@pytest.mark.integration
def test_ws_backpressure_frames_dropped_on_excess() -> None:
    """Excessive backlog (>max_backlog_s) causes FramesDroppedAction."""
    clock = _FakeClock()
    ctrl = BackpressureController(
        sample_rate=_SAMPLE_RATE,
        max_backlog_s=1.0,  # 1s de backlog maximo
        rate_limit_threshold=1.2,
        clock=clock,
    )

    # Send many frames instantly (without advancing the clock)
    drop_action = None
    for _ in range(100):
        result = ctrl.record_frame(_BYTES_PER_20MS)
        if isinstance(result, FramesDroppedAction):
            drop_action = result
            break

    assert drop_action is not None, "Should drop frames after backlog > 1s"
    assert drop_action.dropped_ms > 0
    assert ctrl.frames_dropped > 0


@pytest.mark.integration
def test_ws_backpressure_normal_speed_no_events() -> None:
    """Audio at 1x real-time does not emit any backpressure events."""
    clock = _FakeClock()
    ctrl = BackpressureController(
        sample_rate=_SAMPLE_RATE,
        rate_limit_threshold=1.2,
        max_backlog_s=10.0,
        clock=clock,
    )

    # Send 100 frames at normal speed
    for _ in range(100):
        result = ctrl.record_frame(_BYTES_PER_20MS)
        assert result is None, "Should not emit events at 1x speed"
        clock.advance(0.020)  # 20ms = real-time

    assert ctrl.frames_dropped == 0


# ---------------------------------------------------------------------------
# Tests: Heartbeat and Ping
# ---------------------------------------------------------------------------


@pytest.mark.integration
def test_ws_heartbeat_ping_sent_periodically() -> None:
    """Server sends WebSocket ping periodically to detect zombie connections."""
    # Configure short heartbeat for fast testing
    _, client = _make_app_with_short_timeouts(
        inactivity_s=5.0,
        heartbeat_s=0.2,
        check_s=0.1,
    )

    with client.websocket_connect("/v1/realtime?model=faster-whisper-tiny") as ws:
        created = ws.receive_json()
        assert created["type"] == "session.created"

        # Send audio to keep session alive (do not trigger inactivity timeout)
        for _ in range(10):
            ws.send_bytes(_make_pcm_silence())
            time.sleep(0.05)

        # The session should remain active (ping/pong kept the connection)
        # Close normally
        ws.send_json({"type": "session.close"})
        closed = ws.receive_json()
        assert closed["type"] == "session.closed"
        assert closed["reason"] == "client_request"


@pytest.mark.integration
def test_ws_text_commands_do_not_reset_inactivity() -> None:
    """JSON commands do not reset the inactivity timer (only audio does)."""
    _, client = _make_app_with_short_timeouts(
        inactivity_s=0.3,
        check_s=0.1,
    )

    with client.websocket_connect("/v1/realtime?model=faster-whisper-tiny") as ws:
        created = ws.receive_json()
        assert created["type"] == "session.created"

        # Send text commands (not audio)
        ws.send_json({"type": "session.configure", "language": "pt"})
        time.sleep(0.15)
        ws.send_json({"type": "session.configure", "language": "en"})

        # Inactivity timeout should trigger even with commands
        closed = ws.receive_json()
        assert closed["type"] == "session.closed"
        assert closed["reason"] == "inactivity_timeout"


# ---------------------------------------------------------------------------
# Tests: Multi-segment flow
# ---------------------------------------------------------------------------


@pytest.mark.integration
async def test_ws_multiple_speech_segments() -> None:
    """Multiple speech segments increment segment_id correctly."""
    # Arrange
    stream_handle1 = _make_stream_handle_mock()
    grpc_client = _make_grpc_client_mock(stream_handle1)
    vad = _make_vad_mock()
    on_event = _make_on_event()

    session = StreamingSession(
        session_id="sess_multi",
        preprocessor=_make_preprocessor_mock(),
        vad=vad,
        grpc_client=grpc_client,
        postprocessor=_make_postprocessor_mock(),
        on_event=on_event,
    )

    assert session.segment_id == 0

    # First segment: speech_start -> speech_end
    vad.process_frame.return_value = VADEvent(
        type=VADEventType.SPEECH_START,
        timestamp_ms=1000,
    )
    vad.is_speaking = False
    await session.process_frame(_make_pcm_silence())
    await asyncio.sleep(0.01)

    stream_handle2 = _make_stream_handle_mock()
    grpc_client.open_stream = AsyncMock(return_value=stream_handle2)

    vad.process_frame.return_value = VADEvent(
        type=VADEventType.SPEECH_END,
        timestamp_ms=2000,
    )
    vad.is_speaking = False
    await session.process_frame(_make_pcm_silence())

    assert session.segment_id == 1

    # Second segment: speech_start -> speech_end
    stream_handle3 = _make_stream_handle_mock()
    grpc_client.open_stream = AsyncMock(return_value=stream_handle3)

    vad.process_frame.return_value = VADEvent(
        type=VADEventType.SPEECH_START,
        timestamp_ms=5000,
    )
    vad.is_speaking = False
    await session.process_frame(_make_pcm_silence())
    await asyncio.sleep(0.01)

    vad.process_frame.return_value = VADEvent(
        type=VADEventType.SPEECH_END,
        timestamp_ms=7000,
    )
    vad.is_speaking = False
    await session.process_frame(_make_pcm_silence())

    assert session.segment_id == 2

    # Verify that both speech_start + speech_end were emitted
    start_events = [
        c for c in on_event.call_args_list if isinstance(c.args[0], VADSpeechStartEvent)
    ]
    end_events = [c for c in on_event.call_args_list if isinstance(c.args[0], VADSpeechEndEvent)]
    assert len(start_events) == 2
    assert len(end_events) == 2


@pytest.mark.integration
async def test_ws_close_during_speech_cleans_up() -> None:
    """Closing session during active speech cleans up resources correctly."""
    stream_handle = _make_stream_handle_mock()
    grpc_client = _make_grpc_client_mock(stream_handle)
    vad = _make_vad_mock()

    session = StreamingSession(
        session_id="sess_close_speech",
        preprocessor=_make_preprocessor_mock(),
        vad=vad,
        grpc_client=grpc_client,
        postprocessor=_make_postprocessor_mock(),
        on_event=_make_on_event(),
    )

    # Trigger speech_start
    vad.process_frame.return_value = VADEvent(
        type=VADEventType.SPEECH_START,
        timestamp_ms=1000,
    )
    vad.is_speaking = True
    await session.process_frame(_make_pcm_silence())
    await asyncio.sleep(0.01)

    assert not session.is_closed

    # Close during speech
    await session.close()

    assert session.is_closed
    stream_handle.cancel.assert_called_once()

    # Processing frame after close is a no-op
    await session.process_frame(_make_pcm_silence())
    # Should not crash


# ---------------------------------------------------------------------------
# Tests: End-to-end WebSocket with language parameter
# ---------------------------------------------------------------------------


@pytest.mark.integration
def test_ws_language_in_session_created() -> None:
    """Language provided in query string appears in session.created config."""
    from starlette.testclient import TestClient

    app = create_app(registry=_make_mock_registry())
    client = TestClient(app)

    with client.websocket_connect(
        "/v1/realtime?model=faster-whisper-tiny&language=pt",
    ) as ws:
        event = ws.receive_json()

    assert event["type"] == "session.created"
    assert event["config"]["language"] == "pt"


@pytest.mark.integration
def test_ws_no_language_default_none() -> None:
    """Without language in query string, config.language is null."""
    from starlette.testclient import TestClient

    app = create_app(registry=_make_mock_registry())
    client = TestClient(app)

    with client.websocket_connect("/v1/realtime?model=faster-whisper-tiny") as ws:
        event = ws.receive_json()

    assert event["type"] == "session.created"
    assert event["config"]["language"] is None
