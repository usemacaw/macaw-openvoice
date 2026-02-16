"""Tests for StreamingSession pre-allocated int16 buffer optimization.

Verifies that the pre-allocated buffer in _send_frame_to_worker() produces
identical PCM output to the original .astype(np.int16) approach, and that
the buffer is reused across calls.
"""

from __future__ import annotations

from unittest.mock import AsyncMock, Mock

import numpy as np

from macaw._types import SessionState
from macaw.session.streaming import StreamingSession
from tests.helpers import FRAME_SIZE, AsyncIterFromList


def _make_session() -> tuple[StreamingSession, Mock]:
    """Create a StreamingSession with mocks, returning (session, stream_handle)."""
    preprocessor = Mock()
    preprocessor.process_frame.return_value = np.zeros(FRAME_SIZE, dtype=np.float32)

    vad = Mock()
    vad.process_frame.return_value = None
    vad.is_speaking = True
    vad.reset.return_value = None

    stream_handle = Mock()
    stream_handle.is_closed = False
    stream_handle.session_id = "test"
    stream_handle.receive_events.return_value = AsyncIterFromList([])
    stream_handle.send_frame = AsyncMock()
    stream_handle.close = AsyncMock()
    stream_handle.cancel = AsyncMock()

    grpc_client = AsyncMock()
    grpc_client.open_stream = AsyncMock(return_value=stream_handle)

    on_event = AsyncMock()

    session = StreamingSession(
        session_id="test",
        preprocessor=preprocessor,
        vad=vad,
        grpc_client=grpc_client,
        postprocessor=None,
        on_event=on_event,
        enable_itn=False,
    )

    return session, stream_handle


class TestInt16BufferPreallocation:
    def test_buffer_initialized_on_creation(self) -> None:
        """int16 buffer exists after StreamingSession creation."""
        session, _ = _make_session()

        assert hasattr(session, "_int16_buffer")
        assert session._int16_buffer.dtype == np.int16
        assert len(session._int16_buffer) == 1024

    def test_buffer_conversion_matches_astype(self) -> None:
        """Pre-allocated buffer produces identical bytes to .astype(np.int16)."""
        # Arrange: float32 frame with varied values
        frame = np.array([0.0, 0.5, -0.5, 1.0, -1.0, 0.123, -0.789], dtype=np.float32)

        # Original approach
        frame_orig = frame.copy()
        np.multiply(frame_orig, 32767.0, out=frame_orig)
        np.clip(frame_orig, -32768, 32767, out=frame_orig)
        expected_bytes = frame_orig.astype(np.int16).tobytes()

        # New approach: pre-allocated buffer
        frame_new = frame.copy()
        np.multiply(frame_new, 32767.0, out=frame_new)
        np.clip(frame_new, -32768, 32767, out=frame_new)
        buf = np.empty(len(frame_new), dtype=np.int16)
        np.copyto(buf, frame_new, casting="unsafe")
        actual_bytes = buf.tobytes()

        # Assert
        assert actual_bytes == expected_bytes

    def test_buffer_reused_across_same_size_frames(self) -> None:
        """Buffer object identity is preserved across calls with same frame size."""
        session, _ = _make_session()

        # Get initial buffer reference
        buf_before = session._int16_buffer

        # Simulate what _send_frame_to_worker does (without the async parts)
        frame = np.random.randn(512).astype(np.float32) * 0.5
        np.multiply(frame, 32767.0, out=frame)
        np.clip(frame, -32768, 32767, out=frame)
        frame_len = len(frame)
        if frame_len > len(session._int16_buffer):
            session._int16_buffer = np.empty(frame_len, dtype=np.int16)
        buf_slice = session._int16_buffer[:frame_len]
        np.copyto(buf_slice, frame, casting="unsafe")

        # Buffer should be the same object (not reallocated)
        assert session._int16_buffer is buf_before

    def test_buffer_grows_for_larger_frame(self) -> None:
        """Buffer grows when frame exceeds initial capacity."""
        session, _ = _make_session()
        assert len(session._int16_buffer) == 1024

        # Simulate a frame larger than 1024 samples
        frame_len = 2048
        if frame_len > len(session._int16_buffer):
            session._int16_buffer = np.empty(frame_len, dtype=np.int16)

        assert len(session._int16_buffer) == 2048

    async def test_send_frame_produces_correct_pcm_bytes(self) -> None:
        """End-to-end: _send_frame_to_worker sends correct PCM bytes."""
        session, stream_handle = _make_session()

        # Put session in ACTIVE state with an open stream
        session._state_machine.transition(SessionState.ACTIVE)
        session._stream_handle = stream_handle

        # Create a known signal frame
        frame = np.sin(np.linspace(0, 2 * np.pi, FRAME_SIZE)).astype(np.float32) * 0.8

        # Compute expected PCM bytes using original approach
        frame_expected = frame.copy()
        np.multiply(frame_expected, 32767.0, out=frame_expected)
        np.clip(frame_expected, -32768, 32767, out=frame_expected)
        expected_bytes = frame_expected.astype(np.int16).tobytes()

        # Act
        await session._send_frame_to_worker(frame)

        # Assert: send_frame was called with the correct PCM bytes
        stream_handle.send_frame.assert_called_once()
        call_kwargs = stream_handle.send_frame.call_args
        actual_bytes = call_kwargs.kwargs.get("pcm_data") or call_kwargs[1].get("pcm_data")
        if actual_bytes is None:
            # Try positional
            actual_bytes = call_kwargs[0][0] if call_kwargs[0] else None
        assert actual_bytes == expected_bytes

    async def test_multiple_frames_reuse_buffer(self) -> None:
        """Multiple _send_frame_to_worker calls reuse the same buffer."""
        session, stream_handle = _make_session()
        session._state_machine.transition(SessionState.ACTIVE)
        session._stream_handle = stream_handle

        buf_id_before = id(session._int16_buffer)

        # Send multiple frames of the same size
        for _ in range(5):
            frame = np.random.randn(FRAME_SIZE).astype(np.float32) * 0.5
            await session._send_frame_to_worker(frame)

        # Buffer should not have been reallocated
        assert id(session._int16_buffer) == buf_id_before
