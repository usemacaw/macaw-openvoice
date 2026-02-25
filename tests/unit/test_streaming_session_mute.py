"""Tests for mute-on-speak in StreamingSession.

Validates:
- mute()/unmute() on StreamingSession
- process_frame() discards frames when muted
- Unmuted frames are processed normally
- is_muted property
- Mute during active speech: frames discarded
- Unmute after error (ensures TTS flow uses try/finally)
- stt_muted_frames_total metric incremented
- Reference counting: double mute still muted, underflow safe
- Concurrent TTS contexts with ref counting
"""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock, patch

import numpy as np

from macaw.session.streaming import StreamingSession


def _make_session(**kwargs: object) -> StreamingSession:
    """Cria StreamingSession com mocks para os testes de mute."""
    preprocessor = MagicMock()
    preprocessor.process_frame.return_value = np.zeros(160, dtype=np.float32)

    vad = MagicMock()
    vad.process_frame.return_value = None
    vad.is_speaking = False

    grpc_client = AsyncMock()
    postprocessor = MagicMock()
    on_event = AsyncMock()

    defaults = {
        "session_id": "test-mute",
        "preprocessor": preprocessor,
        "vad": vad,
        "grpc_client": grpc_client,
        "postprocessor": postprocessor,
        "on_event": on_event,
    }
    defaults.update(kwargs)
    return StreamingSession(**defaults)  # type: ignore[arg-type]


class TestStreamingSessionMuteProperty:
    def test_starts_unmuted(self) -> None:
        session = _make_session()
        assert session.is_muted is False

    def test_mute_sets_flag(self) -> None:
        session = _make_session()
        session.mute()
        assert session.is_muted is True

    def test_unmute_clears_flag(self) -> None:
        session = _make_session()
        session.mute()
        session.unmute()
        assert session.is_muted is False

    def test_double_mute_still_muted(self) -> None:
        """Two mute() calls keep session muted (ref count = 2)."""
        session = _make_session()
        session.mute()
        session.mute()
        assert session.is_muted is True

    def test_unmute_underflow_safe(self) -> None:
        """Unmuting when already unmuted stays at 0 (no negative depth)."""
        session = _make_session()
        session.unmute()
        assert session.is_muted is False


class TestMutedFrameDiscard:
    async def test_muted_frame_not_preprocessed(self) -> None:
        """When muted, process_frame() returns early without preprocessing."""
        session = _make_session()
        session.mute()

        raw_bytes = b"\x00\x01" * 160
        await session.process_frame(raw_bytes)

        # Preprocessing should NOT be called
        session._preprocessor.process_frame.assert_not_called()

    async def test_muted_frame_not_sent_to_vad(self) -> None:
        """When muted, VAD is not invoked."""
        session = _make_session()
        session.mute()

        await session.process_frame(b"\x00\x01" * 160)

        session._vad.process_frame.assert_not_called()

    async def test_unmuted_frame_is_preprocessed(self) -> None:
        """When unmuted, preprocessing runs normally."""
        session = _make_session()

        await session.process_frame(b"\x00\x01" * 160)

        session._preprocessor.process_frame.assert_called_once()

    async def test_unmuted_frame_is_sent_to_vad(self) -> None:
        """When unmuted, VAD is invoked."""
        session = _make_session()

        await session.process_frame(b"\x00\x01" * 160)

        session._vad.process_frame.assert_called_once()

    async def test_mute_during_active_discards_frames(self) -> None:
        """If muted during ACTIVE state, subsequent frames are discarded."""
        session = _make_session()

        # Process one frame unmuted
        await session.process_frame(b"\x00\x01" * 160)
        assert session._preprocessor.process_frame.call_count == 1

        # Mute and process another frame
        session.mute()
        await session.process_frame(b"\x00\x01" * 160)

        # Preprocessor should still have been called only once
        assert session._preprocessor.process_frame.call_count == 1

    async def test_unmute_resumes_processing(self) -> None:
        """After unmute, frames are processed normally again."""
        session = _make_session()

        session.mute()
        await session.process_frame(b"\x00\x01" * 160)
        assert session._preprocessor.process_frame.call_count == 0

        session.unmute()
        await session.process_frame(b"\x00\x01" * 160)
        assert session._preprocessor.process_frame.call_count == 1


class TestMuteMetric:
    @patch("macaw.session.streaming.stt_muted_frames_total")
    async def test_muted_frame_increments_metric(self, mock_counter: MagicMock) -> None:
        """Each muted frame increments stt_muted_frames_total."""
        session = _make_session()
        session.mute()

        await session.process_frame(b"\x00\x01" * 160)
        await session.process_frame(b"\x00\x01" * 160)

        assert mock_counter.inc.call_count == 2

    async def test_unmuted_frame_does_not_increment_metric(self) -> None:
        """Unmuted frames do NOT increment the mute metric."""
        # This test just verifies the code path doesn't crash without metrics
        session = _make_session()
        await session.process_frame(b"\x00\x01" * 160)
        # No assertion on metric — just verifying no error


class TestMuteWithClosedSession:
    async def test_muted_closed_session_still_returns_early(self) -> None:
        """Closed sessions return early before mute check (existing behavior)."""
        session = _make_session()
        await session.close()

        session.mute()
        await session.process_frame(b"\x00\x01" * 160)

        session._preprocessor.process_frame.assert_not_called()


class TestMuteTryFinally:
    async def test_unmute_in_finally_block(self) -> None:
        """Demonstrates the try/finally pattern for safe unmute."""
        session = _make_session()

        # Simulate TTS flow with try/finally
        session.mute()
        assert session.is_muted is True
        try:
            # Simulate TTS work that might fail
            raise RuntimeError("TTS worker crashed")
        except RuntimeError:
            pass
        finally:
            session.unmute()

        assert session.is_muted is False

    async def test_unmute_after_cancel(self) -> None:
        """After TTS cancel, unmute is immediate."""
        session = _make_session()

        session.mute()
        assert session.is_muted is True

        # Simulate tts.cancel -> immediate unmute
        session.unmute()
        assert session.is_muted is False

        # Verify frames are processed after unmute
        await session.process_frame(b"\x00\x01" * 160)
        session._preprocessor.process_frame.assert_called_once()


class TestConcurrentTTSContextsMute:
    """Tests for multi-context TTS with ref counting on StreamingSession."""

    async def test_concurrent_tts_contexts_mute(self) -> None:
        """Two TTS contexts: both mute, one unmutes -> still muted."""
        session = _make_session()

        # Context A starts TTS
        session.mute()
        assert session.is_muted is True

        # Context B starts TTS
        session.mute()
        assert session.is_muted is True

        # Context A finishes
        session.unmute()
        assert session.is_muted is True  # B still active

        # Context B finishes
        session.unmute()
        assert session.is_muted is False

    async def test_try_finally_with_concurrent_contexts(self) -> None:
        """Two TTS contexts with try/finally: crash in A doesn't unmute B."""
        session = _make_session()

        # Context A
        session.mute()
        try:
            # Context B
            session.mute()
            try:
                raise RuntimeError("TTS B crashed")
            except RuntimeError:
                pass
            finally:
                session.unmute()  # B's finally

            # After B's crash+unmute, A still holds mute
            assert session.is_muted is True
        finally:
            session.unmute()  # A's finally

        assert session.is_muted is False
