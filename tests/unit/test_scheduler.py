"""Testes do Scheduler."""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from macaw._types import ResponseFormat, SegmentDetail, WordTimestamp
from macaw.exceptions import (
    ModelNotFoundError,
    WorkerCrashError,
    WorkerTimeoutError,
    WorkerUnavailableError,
)
from macaw.proto import Segment, TranscribeFileResponse, Word
from macaw.scheduler.converters import build_proto_request, proto_response_to_batch_result
from macaw.scheduler.scheduler import Scheduler
from macaw.server.models.requests import TranscribeRequest
from macaw.workers.manager import WorkerHandle, WorkerState


def _make_request(
    model_name: str = "faster-whisper-tiny",
    task: str = "transcribe",
) -> TranscribeRequest:
    return TranscribeRequest(
        request_id="req-001",
        model_name=model_name,
        audio_data=b"fake-audio-data",
        language="pt",
        response_format=ResponseFormat.JSON,
        temperature=0.0,
        timestamp_granularities=("segment",),
        task=task,
    )


def _make_worker(
    model_name: str = "faster-whisper-tiny",
    port: int = 50051,
) -> WorkerHandle:
    return WorkerHandle(
        worker_id=f"faster-whisper-{port}",
        port=port,
        model_name=model_name,
        engine="faster-whisper",
        state=WorkerState.READY,
    )


def _make_proto_response() -> TranscribeFileResponse:
    return TranscribeFileResponse(
        text="Ola mundo",
        language="pt",
        duration=1.5,
        segments=[
            Segment(
                id=0,
                start=0.0,
                end=1.5,
                text="Ola mundo",
                avg_logprob=-0.3,
                no_speech_prob=0.02,
                compression_ratio=1.1,
            ),
        ],
        words=[
            Word(word="Ola", start=0.0, end=0.5, probability=0.95),
            Word(word="mundo", start=0.6, end=1.5, probability=0.90),
        ],
    )


class TestBuildProtoRequest:
    """Testes da conversao TranscribeRequest -> protobuf."""

    def test_maps_required_fields(self) -> None:
        request = _make_request()
        proto = build_proto_request(request)

        assert proto.request_id == "req-001"
        assert proto.audio_data == b"fake-audio-data"
        assert proto.language == "pt"
        assert proto.temperature == 0.0

    def test_none_language_becomes_empty_string(self) -> None:
        request = TranscribeRequest(
            request_id="req-002",
            model_name="model",
            audio_data=b"data",
            language=None,
        )
        proto = build_proto_request(request)
        assert proto.language == ""

    def test_none_prompt_becomes_empty_string(self) -> None:
        request = TranscribeRequest(
            request_id="req-003",
            model_name="model",
            audio_data=b"data",
            initial_prompt=None,
        )
        proto = build_proto_request(request)
        assert proto.initial_prompt == ""

    def test_hot_words_mapped(self) -> None:
        request = TranscribeRequest(
            request_id="req-004",
            model_name="model",
            audio_data=b"data",
            hot_words=("PIX", "TED"),
        )
        proto = build_proto_request(request)
        assert list(proto.hot_words) == ["PIX", "TED"]

    def test_none_hot_words_becomes_empty(self) -> None:
        request = TranscribeRequest(
            request_id="req-005",
            model_name="model",
            audio_data=b"data",
            hot_words=None,
        )
        proto = build_proto_request(request)
        assert list(proto.hot_words) == []

    def test_task_field_mapped(self) -> None:
        request = _make_request(task="translate")
        proto = build_proto_request(request)
        assert proto.task == "translate"

    def test_task_field_default_transcribe(self) -> None:
        request = _make_request()
        proto = build_proto_request(request)
        assert proto.task == "transcribe"

    def test_response_format_as_string(self) -> None:
        request = TranscribeRequest(
            request_id="req-006",
            model_name="model",
            audio_data=b"data",
            response_format=ResponseFormat.VERBOSE_JSON,
        )
        proto = build_proto_request(request)
        assert proto.response_format == "verbose_json"


class TestProtoResponseToBatchResult:
    """Testes da conversao protobuf -> BatchResult."""

    def test_maps_text_and_metadata(self) -> None:
        proto = _make_proto_response()
        result = proto_response_to_batch_result(proto)

        assert result.text == "Ola mundo"
        assert result.language == "pt"
        assert result.duration == pytest.approx(1.5)

    def test_maps_segments(self) -> None:
        proto = _make_proto_response()
        result = proto_response_to_batch_result(proto)

        assert len(result.segments) == 1
        seg = result.segments[0]
        assert isinstance(seg, SegmentDetail)
        assert seg.text == "Ola mundo"
        assert seg.start == pytest.approx(0.0)
        assert seg.end == pytest.approx(1.5)

    def test_maps_words(self) -> None:
        proto = _make_proto_response()
        result = proto_response_to_batch_result(proto)

        assert result.words is not None
        assert len(result.words) == 2
        assert isinstance(result.words[0], WordTimestamp)
        assert result.words[0].word == "Ola"

    def test_empty_words_returns_none(self) -> None:
        proto = TranscribeFileResponse(
            text="Test",
            language="en",
            duration=1.0,
            segments=[],
            words=[],
        )
        result = proto_response_to_batch_result(proto)
        assert result.words is None


class TestSchedulerTranscribe:
    """Testes do metodo Scheduler.transcribe."""

    async def test_model_not_found_raises_error(self) -> None:
        registry = MagicMock()
        registry.get_manifest.side_effect = ModelNotFoundError("inexistente")
        worker_manager = MagicMock()

        scheduler = Scheduler(worker_manager, registry)
        request = _make_request(model_name="inexistente")

        with pytest.raises(ModelNotFoundError):
            await scheduler.transcribe(request)

    async def test_no_ready_worker_raises_error(self) -> None:
        registry = MagicMock()
        registry.get_manifest.return_value = MagicMock()
        worker_manager = MagicMock()
        worker_manager.get_ready_worker.return_value = None

        scheduler = Scheduler(worker_manager, registry)
        request = _make_request()

        with pytest.raises(WorkerUnavailableError):
            await scheduler.transcribe(request)

    async def test_successful_transcribe(self) -> None:
        registry = MagicMock()
        registry.get_manifest.return_value = MagicMock()

        worker = _make_worker()
        worker_manager = MagicMock()
        worker_manager.get_ready_worker.return_value = worker

        proto_response = _make_proto_response()

        mock_stub = AsyncMock()
        mock_stub.TranscribeFile.return_value = proto_response

        mock_channel = AsyncMock()

        scheduler = Scheduler(worker_manager, registry)
        request = _make_request()

        with (
            patch(
                "macaw.scheduler.scheduler.grpc.aio.insecure_channel", return_value=mock_channel
            ),
            patch("macaw.scheduler.scheduler.STTWorkerStub", return_value=mock_stub),
        ):
            result = await scheduler.transcribe(request)

        assert result.text == "Ola mundo"
        assert result.language == "pt"
        mock_channel.close.assert_awaited_once()

    async def test_channel_closed_on_grpc_error(self) -> None:
        registry = MagicMock()
        registry.get_manifest.return_value = MagicMock()

        worker = _make_worker()
        worker_manager = MagicMock()
        worker_manager.get_ready_worker.return_value = worker

        mock_stub = AsyncMock()
        mock_stub.TranscribeFile.side_effect = Exception("gRPC error")

        mock_channel = AsyncMock()

        scheduler = Scheduler(worker_manager, registry)
        request = _make_request()

        with (
            patch(
                "macaw.scheduler.scheduler.grpc.aio.insecure_channel", return_value=mock_channel
            ),
            patch("macaw.scheduler.scheduler.STTWorkerStub", return_value=mock_stub),
            pytest.raises(Exception, match="gRPC error"),
        ):
            await scheduler.transcribe(request)

        mock_channel.close.assert_awaited_once()

    async def test_validates_model_before_finding_worker(self) -> None:
        """Registry e consultado antes do WorkerManager."""
        registry = MagicMock()
        registry.get_manifest.side_effect = ModelNotFoundError("model")
        worker_manager = MagicMock()

        scheduler = Scheduler(worker_manager, registry)
        request = _make_request()

        with pytest.raises(ModelNotFoundError):
            await scheduler.transcribe(request)

        worker_manager.get_ready_worker.assert_not_called()

    async def test_grpc_deadline_exceeded_raises_worker_timeout(self) -> None:
        import grpc
        import grpc.aio

        registry = MagicMock()
        registry.get_manifest.return_value = MagicMock()

        worker = _make_worker()
        worker_manager = MagicMock()
        worker_manager.get_ready_worker.return_value = worker

        mock_stub = AsyncMock()
        mock_stub.TranscribeFile.side_effect = grpc.aio.AioRpcError(
            code=grpc.StatusCode.DEADLINE_EXCEEDED,
            initial_metadata=grpc.aio.Metadata(),
            trailing_metadata=grpc.aio.Metadata(),
            details="Deadline Exceeded",
        )

        mock_channel = AsyncMock()

        scheduler = Scheduler(worker_manager, registry)
        request = _make_request()

        with (
            patch(
                "macaw.scheduler.scheduler.grpc.aio.insecure_channel", return_value=mock_channel
            ),
            patch("macaw.scheduler.scheduler.STTWorkerStub", return_value=mock_stub),
            pytest.raises(WorkerTimeoutError),
        ):
            await scheduler.transcribe(request)

        mock_channel.close.assert_awaited_once()

    async def test_grpc_unavailable_raises_worker_crash(self) -> None:
        import grpc
        import grpc.aio

        registry = MagicMock()
        registry.get_manifest.return_value = MagicMock()

        worker = _make_worker()
        worker_manager = MagicMock()
        worker_manager.get_ready_worker.return_value = worker

        mock_stub = AsyncMock()
        mock_stub.TranscribeFile.side_effect = grpc.aio.AioRpcError(
            code=grpc.StatusCode.UNAVAILABLE,
            initial_metadata=grpc.aio.Metadata(),
            trailing_metadata=grpc.aio.Metadata(),
            details="Connection refused",
        )

        mock_channel = AsyncMock()

        scheduler = Scheduler(worker_manager, registry)
        request = _make_request()

        with (
            patch(
                "macaw.scheduler.scheduler.grpc.aio.insecure_channel", return_value=mock_channel
            ),
            patch("macaw.scheduler.scheduler.STTWorkerStub", return_value=mock_stub),
            pytest.raises(WorkerCrashError),
        ):
            await scheduler.transcribe(request)
