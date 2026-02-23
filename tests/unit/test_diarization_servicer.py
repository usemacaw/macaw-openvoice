"""Tests for diarization integration in the STT servicer (batch only).

Verifies that TranscribeFile correctly invokes the diarizer when
diarize=True, handles missing diarizer gracefully, and produces
proper proto responses with speaker_segments.
"""

from __future__ import annotations

from typing import TYPE_CHECKING
from unittest.mock import AsyncMock, MagicMock

import grpc
import pytest

from macaw._types import (
    BatchResult,
    EngineCapabilities,
    SegmentDetail,
    SpeakerSegment,
    STTArchitecture,
)
from macaw.proto.stt_worker_pb2 import TranscribeFileRequest
from macaw.workers.stt.interface import STTBackend
from macaw.workers.stt.servicer import STTWorkerServicer

if TYPE_CHECKING:
    from collections.abc import AsyncIterator

    from macaw._types import TranscriptSegment


class _MockBackend(STTBackend):
    """STT backend mock for diarization servicer tests."""

    def __init__(self, result: BatchResult | None = None) -> None:
        self._result = result or BatchResult(
            text="hello world",
            language="en",
            duration=2.0,
            segments=(SegmentDetail(id=0, start=0.0, end=2.0, text="hello world"),),
        )

    @property
    def architecture(self) -> STTArchitecture:
        return STTArchitecture.ENCODER_DECODER

    async def load(self, model_path: str, config: dict[str, object]) -> None:
        pass

    async def capabilities(self) -> EngineCapabilities:
        return EngineCapabilities()

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
        pass

    async def health(self) -> dict[str, str]:
        return {"status": "ok"}


def _make_context() -> MagicMock:
    """Create a mock gRPC ServicerContext."""
    ctx = MagicMock()
    ctx.abort = AsyncMock()
    ctx.cancelled = MagicMock(return_value=False)
    return ctx


def _make_diarizer(
    segments: tuple[SpeakerSegment, ...] = (),
) -> AsyncMock:
    """Create a mock DiarizationBackend."""
    diarizer = AsyncMock()
    diarizer.diarize = AsyncMock(return_value=segments)
    diarizer.health = AsyncMock(return_value={"status": "ok"})
    return diarizer


class TestTranscribeFileWithDiarizeFalse:
    """When diarize=False, transcription works unchanged."""

    async def test_no_diarizer_invoked(self) -> None:
        diarizer = _make_diarizer()
        servicer = STTWorkerServicer(
            backend=_MockBackend(),
            model_name="test",
            engine="mock",
            diarizer=diarizer,
        )
        request = TranscribeFileRequest(
            request_id="req-1",
            audio_data=b"\x00\x00" * 100,
            diarize=False,
        )
        ctx = _make_context()
        response = await servicer.TranscribeFile(request, ctx)

        assert response.text == "hello world"
        diarizer.diarize.assert_not_called()

    async def test_no_speaker_segments_in_response(self) -> None:
        servicer = STTWorkerServicer(
            backend=_MockBackend(),
            model_name="test",
            engine="mock",
        )
        request = TranscribeFileRequest(
            request_id="req-2",
            audio_data=b"\x00\x00" * 100,
            diarize=False,
        )
        ctx = _make_context()
        response = await servicer.TranscribeFile(request, ctx)

        assert len(response.speaker_segments) == 0


class TestTranscribeFileWithDiarizeTrue:
    """When diarize=True and diarizer is available."""

    async def test_returns_speaker_segments(self) -> None:
        segments = (
            SpeakerSegment(speaker_id="SPEAKER_00", start=0.0, end=1.0, text=""),
            SpeakerSegment(speaker_id="SPEAKER_01", start=1.0, end=2.0, text=""),
        )
        diarizer = _make_diarizer(segments)
        servicer = STTWorkerServicer(
            backend=_MockBackend(),
            model_name="test",
            engine="mock",
            diarizer=diarizer,
        )
        request = TranscribeFileRequest(
            request_id="req-3",
            audio_data=b"\x00\x00" * 100,
            diarize=True,
        )
        ctx = _make_context()
        response = await servicer.TranscribeFile(request, ctx)

        assert len(response.speaker_segments) == 2
        assert response.speaker_segments[0].speaker_id == "SPEAKER_00"
        assert response.speaker_segments[1].speaker_id == "SPEAKER_01"

    async def test_forwards_max_speakers(self) -> None:
        diarizer = _make_diarizer()
        servicer = STTWorkerServicer(
            backend=_MockBackend(),
            model_name="test",
            engine="mock",
            diarizer=diarizer,
        )
        request = TranscribeFileRequest(
            request_id="req-4",
            audio_data=b"\x00\x00" * 100,
            diarize=True,
            max_speakers=3,
        )
        ctx = _make_context()
        await servicer.TranscribeFile(request, ctx)

        diarizer.diarize.assert_called_once()
        call_kwargs = diarizer.diarize.call_args
        assert call_kwargs.kwargs["max_speakers"] == 3

    async def test_max_speakers_zero_means_auto(self) -> None:
        diarizer = _make_diarizer()
        servicer = STTWorkerServicer(
            backend=_MockBackend(),
            model_name="test",
            engine="mock",
            diarizer=diarizer,
        )
        request = TranscribeFileRequest(
            request_id="req-5",
            audio_data=b"\x00\x00" * 100,
            diarize=True,
            max_speakers=0,
        )
        ctx = _make_context()
        await servicer.TranscribeFile(request, ctx)

        call_kwargs = diarizer.diarize.call_args
        assert call_kwargs.kwargs["max_speakers"] is None

    async def test_proto_response_includes_speaker_segments_field(self) -> None:
        segments = (SpeakerSegment(speaker_id="s0", start=0.5, end=1.5, text=""),)
        diarizer = _make_diarizer(segments)
        servicer = STTWorkerServicer(
            backend=_MockBackend(),
            model_name="test",
            engine="mock",
            diarizer=diarizer,
        )
        request = TranscribeFileRequest(
            request_id="req-6",
            audio_data=b"\x00\x00" * 100,
            diarize=True,
        )
        ctx = _make_context()
        response = await servicer.TranscribeFile(request, ctx)

        # Verify proto fields are populated
        seg = response.speaker_segments[0]
        assert seg.speaker_id == "s0"
        assert seg.start == pytest.approx(0.5)
        assert seg.end == pytest.approx(1.5)

    async def test_empty_diarization_returns_empty_speaker_segments(self) -> None:
        diarizer = _make_diarizer(())
        servicer = STTWorkerServicer(
            backend=_MockBackend(),
            model_name="test",
            engine="mock",
            diarizer=diarizer,
        )
        request = TranscribeFileRequest(
            request_id="req-7",
            audio_data=b"\x00\x00" * 100,
            diarize=True,
        )
        ctx = _make_context()
        response = await servicer.TranscribeFile(request, ctx)

        assert len(response.speaker_segments) == 0

    async def test_transcription_text_preserved(self) -> None:
        segments = (SpeakerSegment(speaker_id="s0", start=0.0, end=2.0, text=""),)
        diarizer = _make_diarizer(segments)
        servicer = STTWorkerServicer(
            backend=_MockBackend(),
            model_name="test",
            engine="mock",
            diarizer=diarizer,
        )
        request = TranscribeFileRequest(
            request_id="req-8",
            audio_data=b"\x00\x00" * 100,
            diarize=True,
        )
        ctx = _make_context()
        response = await servicer.TranscribeFile(request, ctx)

        # Transcription text should be preserved alongside diarization
        assert response.text == "hello world"


class TestTranscribeFileDiarizerUnavailable:
    """When diarize=True but no diarizer backend is configured."""

    async def test_returns_unavailable_error(self) -> None:
        servicer = STTWorkerServicer(
            backend=_MockBackend(),
            model_name="test",
            engine="mock",
            diarizer=None,
        )
        request = TranscribeFileRequest(
            request_id="req-9",
            audio_data=b"\x00\x00" * 100,
            diarize=True,
        )
        ctx = _make_context()
        await servicer.TranscribeFile(request, ctx)

        ctx.abort.assert_called_once()
        abort_args = ctx.abort.call_args
        assert abort_args[0][0] == grpc.StatusCode.UNAVAILABLE
        assert "not available" in abort_args[0][1].lower()


class TestDiarizationErrorGracefulDegradation:
    """When diarizer raises an error, transcription result is still returned."""

    async def test_diarization_error_returns_result_without_segments(self) -> None:
        diarizer = _make_diarizer()
        diarizer.diarize.side_effect = RuntimeError("Diarization model crashed")
        servicer = STTWorkerServicer(
            backend=_MockBackend(),
            model_name="test",
            engine="mock",
            diarizer=diarizer,
        )
        request = TranscribeFileRequest(
            request_id="req-10",
            audio_data=b"\x00\x00" * 100,
            diarize=True,
        )
        ctx = _make_context()
        response = await servicer.TranscribeFile(request, ctx)

        # Transcription should still be returned
        assert response.text == "hello world"
        # No speaker_segments since diarization failed
        assert len(response.speaker_segments) == 0


class TestBackwardCompatibility:
    """Existing servicer behavior is unaffected by diarizer parameter."""

    async def test_servicer_without_diarizer_param(self) -> None:
        """Servicer works without passing diarizer (default None)."""
        servicer = STTWorkerServicer(
            backend=_MockBackend(),
            model_name="test",
            engine="mock",
        )
        request = TranscribeFileRequest(
            request_id="req-11",
            audio_data=b"\x00\x00" * 100,
        )
        ctx = _make_context()
        response = await servicer.TranscribeFile(request, ctx)

        assert response.text == "hello world"
        assert len(response.speaker_segments) == 0
