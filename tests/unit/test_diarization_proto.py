"""Tests for diarization proto fields, SpeakerSegment dataclass, and related changes."""

from __future__ import annotations

from unittest.mock import MagicMock

import pytest

from macaw._types import BatchResult, EngineCapabilities, ModelType, SegmentDetail, SpeakerSegment
from macaw.config.manifest import ModelCapabilities
from macaw.proto.stt_worker_pb2 import (
    SpeakerSegment as SpeakerSegmentPb2,
)
from macaw.proto.stt_worker_pb2 import (
    TranscribeFileRequest,
    TranscribeFileResponse,
    TranscriptEvent,
)
from macaw.server.models.models import ModelCapabilitiesResponse
from macaw.workers.stt.converters import (
    TranscribeFileParams,
    batch_result_to_proto_response,
    proto_request_to_transcribe_params,
    speaker_segment_to_dataclass,
    speaker_segment_to_proto,
)


class TestSpeakerSegmentDataclass:
    """SpeakerSegment dataclass creation and properties."""

    def test_creation(self) -> None:
        seg = SpeakerSegment(
            speaker_id="speaker_0",
            start=0.5,
            end=2.3,
            text="Hello world",
        )
        assert seg.speaker_id == "speaker_0"
        assert seg.start == 0.5
        assert seg.end == 2.3
        assert seg.text == "Hello world"

    def test_frozen_immutable(self) -> None:
        seg = SpeakerSegment(speaker_id="s0", start=0.0, end=1.0, text="hi")
        with pytest.raises(AttributeError):
            seg.speaker_id = "s1"  # type: ignore[misc]

    def test_slots(self) -> None:
        seg = SpeakerSegment(speaker_id="s0", start=0.0, end=1.0, text="hi")
        assert not hasattr(seg, "__dict__")


class TestEngineCapabilitiesDiarization:
    """EngineCapabilities.supports_diarization field."""

    def test_default_false(self) -> None:
        caps = EngineCapabilities()
        assert caps.supports_diarization is False

    def test_enabled(self) -> None:
        caps = EngineCapabilities(supports_diarization=True)
        assert caps.supports_diarization is True


class TestBatchResultSpeakerSegments:
    """BatchResult with speaker_segments field."""

    def test_default_none(self) -> None:
        result = BatchResult(
            text="hello",
            language="en",
            duration=1.0,
            segments=(SegmentDetail(id=0, start=0.0, end=1.0, text="hello"),),
        )
        assert result.speaker_segments is None

    def test_with_speaker_segments(self) -> None:
        segs = (
            SpeakerSegment(speaker_id="s0", start=0.0, end=1.0, text="hello"),
            SpeakerSegment(speaker_id="s1", start=1.0, end=2.0, text="world"),
        )
        result = BatchResult(
            text="hello world",
            language="en",
            duration=2.0,
            segments=(SegmentDetail(id=0, start=0.0, end=2.0, text="hello world"),),
            speaker_segments=segs,
        )
        assert result.speaker_segments is not None
        assert len(result.speaker_segments) == 2
        assert result.speaker_segments[0].speaker_id == "s0"
        assert result.speaker_segments[1].speaker_id == "s1"


class TestProtoTranscribeFileRequest:
    """Proto TranscribeFileRequest diarize and max_speakers fields."""

    def test_diarize_field_default(self) -> None:
        req = TranscribeFileRequest()
        assert req.diarize is False

    def test_diarize_field_set(self) -> None:
        req = TranscribeFileRequest(diarize=True)
        assert req.diarize is True

    def test_max_speakers_field_default(self) -> None:
        req = TranscribeFileRequest()
        assert req.max_speakers == 0

    def test_max_speakers_field_set(self) -> None:
        req = TranscribeFileRequest(max_speakers=3)
        assert req.max_speakers == 3


class TestProtoTranscribeFileResponse:
    """Proto TranscribeFileResponse speaker_segments field."""

    def test_speaker_segments_empty_by_default(self) -> None:
        resp = TranscribeFileResponse()
        assert len(resp.speaker_segments) == 0

    def test_speaker_segments_roundtrip(self) -> None:
        seg = SpeakerSegmentPb2(
            speaker_id="speaker_0",
            start=0.5,
            end=2.0,
            text="Hello",
        )
        resp = TranscribeFileResponse(
            text="Hello",
            language="en",
            duration=2.0,
            speaker_segments=[seg],
        )
        assert len(resp.speaker_segments) == 1
        assert resp.speaker_segments[0].speaker_id == "speaker_0"
        assert resp.speaker_segments[0].start == pytest.approx(0.5)
        assert resp.speaker_segments[0].end == pytest.approx(2.0)
        assert resp.speaker_segments[0].text == "Hello"


class TestProtoTranscriptEvent:
    """Proto TranscriptEvent speaker_id field."""

    def test_speaker_id_default_empty(self) -> None:
        event = TranscriptEvent()
        assert event.speaker_id == ""

    def test_speaker_id_set(self) -> None:
        event = TranscriptEvent(speaker_id="speaker_1")
        assert event.speaker_id == "speaker_1"


class TestProtoSpeakerSegmentMessage:
    """Proto SpeakerSegment message."""

    def test_creation(self) -> None:
        seg = SpeakerSegmentPb2(
            speaker_id="speaker_2",
            start=1.5,
            end=3.0,
            text="Testing",
        )
        assert seg.speaker_id == "speaker_2"
        assert seg.start == pytest.approx(1.5)
        assert seg.end == pytest.approx(3.0)
        assert seg.text == "Testing"


class TestConverterSpeakerSegmentToProto:
    """speaker_segment_to_proto converter."""

    def test_roundtrip(self) -> None:
        dc = SpeakerSegment(speaker_id="s0", start=0.1, end=1.5, text="hello")
        proto = speaker_segment_to_proto(dc)
        assert proto.speaker_id == "s0"
        assert proto.start == pytest.approx(0.1)
        assert proto.end == pytest.approx(1.5)
        assert proto.text == "hello"


class TestConverterSpeakerSegmentToDataclass:
    """speaker_segment_to_dataclass converter."""

    def test_roundtrip(self) -> None:
        proto = SpeakerSegmentPb2(speaker_id="s1", start=2.0, end=4.0, text="world")
        dc = speaker_segment_to_dataclass(proto)
        assert dc.speaker_id == "s1"
        assert dc.start == pytest.approx(2.0)
        assert dc.end == pytest.approx(4.0)
        assert dc.text == "world"


class TestConverterBatchResultWithSpeakerSegments:
    """batch_result_to_proto_response with speaker_segments."""

    def test_without_speaker_segments(self) -> None:
        result = BatchResult(
            text="hello",
            language="en",
            duration=1.0,
            segments=(SegmentDetail(id=0, start=0.0, end=1.0, text="hello"),),
        )
        proto = batch_result_to_proto_response(result)
        assert len(proto.speaker_segments) == 0

    def test_with_speaker_segments(self) -> None:
        segs = (
            SpeakerSegment(speaker_id="s0", start=0.0, end=1.0, text="hello"),
            SpeakerSegment(speaker_id="s1", start=1.0, end=2.0, text="world"),
        )
        result = BatchResult(
            text="hello world",
            language="en",
            duration=2.0,
            segments=(SegmentDetail(id=0, start=0.0, end=2.0, text="hello world"),),
            speaker_segments=segs,
        )
        proto = batch_result_to_proto_response(result)
        assert len(proto.speaker_segments) == 2
        assert proto.speaker_segments[0].speaker_id == "s0"
        assert proto.speaker_segments[1].speaker_id == "s1"


class TestConverterProtoRequestDiarizeFields:
    """proto_request_to_transcribe_params extracts diarize and max_speakers."""

    def test_diarize_false_by_default(self) -> None:
        req = TranscribeFileRequest(
            audio_data=b"\x00\x01",
            language="en",
        )
        params = proto_request_to_transcribe_params(req)
        assert params.diarize is False
        assert params.max_speakers == 0

    def test_diarize_true(self) -> None:
        req = TranscribeFileRequest(
            audio_data=b"\x00\x01",
            language="en",
            diarize=True,
            max_speakers=5,
        )
        params = proto_request_to_transcribe_params(req)
        assert params.diarize is True
        assert params.max_speakers == 5


class TestTranscribeFileParamsDiarizeFields:
    """TranscribeFileParams includes diarize and max_speakers."""

    def test_defaults(self) -> None:
        params = TranscribeFileParams(
            audio_data=b"\x00",
            language="en",
            initial_prompt=None,
            hot_words=None,
            temperature=0.0,
            word_timestamps=False,
        )
        assert params.diarize is False
        assert params.max_speakers == 0

    def test_set(self) -> None:
        params = TranscribeFileParams(
            audio_data=b"\x00",
            language="en",
            initial_prompt=None,
            hot_words=None,
            temperature=0.0,
            word_timestamps=False,
            diarize=True,
            max_speakers=3,
        )
        assert params.diarize is True
        assert params.max_speakers == 3


class TestModelCapabilitiesDiarization:
    """ModelCapabilities (manifest) diarization field."""

    def test_default_false(self) -> None:
        caps = ModelCapabilities()
        assert caps.diarization is False

    def test_enabled(self) -> None:
        caps = ModelCapabilities(diarization=True)
        assert caps.diarization is True


class TestModelCapabilitiesResponseDiarization:
    """ModelCapabilitiesResponse (API) diarization field."""

    def test_default_false(self) -> None:
        resp = ModelCapabilitiesResponse()
        assert resp.diarization is False

    def test_enabled(self) -> None:
        resp = ModelCapabilitiesResponse(diarization=True)
        assert resp.diarization is True

    def test_serialization(self) -> None:
        resp = ModelCapabilitiesResponse(diarization=True)
        data = resp.model_dump()
        assert data["diarization"] is True


class TestListModelsDiarization:
    """GET /v1/models response includes diarization from manifest."""

    @pytest.mark.asyncio
    async def test_list_models_includes_diarization(self) -> None:
        from macaw.config.manifest import (
            EngineConfig,
            ModelCapabilities,
            ModelManifest,
            ModelResources,
        )
        from macaw.server.routes.health import list_models

        manifest = ModelManifest(
            name="test-model",
            version="1.0",
            engine="pyannote",
            model_type=ModelType.STT,
            capabilities=ModelCapabilities(diarization=True),
            resources=ModelResources(memory_mb=500),
            engine_config=EngineConfig(),
        )

        mock_registry = MagicMock()
        mock_registry.list_models.return_value = [manifest]

        mock_request = MagicMock()
        mock_request.app.state.registry = mock_registry
        mock_request.app.state.worker_manager = None

        result = await list_models(mock_request)

        assert result["data"][0]["capabilities"]["diarization"] is True


class TestSchedulerConverterDiarize:
    """Scheduler build_proto_request includes diarize/max_speakers."""

    def test_diarize_in_proto_request(self) -> None:
        from macaw.scheduler.converters import build_proto_request
        from macaw.server.models.requests import TranscribeRequest

        request = TranscribeRequest(
            request_id="r1",
            model_name="model",
            audio_data=b"\x00\x01",
            diarize=True,
            max_speakers=4,
        )
        proto = build_proto_request(request)
        assert proto.diarize is True
        assert proto.max_speakers == 4

    def test_diarize_default_false(self) -> None:
        from macaw.scheduler.converters import build_proto_request
        from macaw.server.models.requests import TranscribeRequest

        request = TranscribeRequest(
            request_id="r1",
            model_name="model",
            audio_data=b"\x00\x01",
        )
        proto = build_proto_request(request)
        assert proto.diarize is False
        assert proto.max_speakers == 0


class TestSchedulerConverterSpeakerSegmentsResponse:
    """Scheduler proto_response_to_batch_result includes speaker_segments."""

    def test_without_speaker_segments(self) -> None:
        from macaw.scheduler.converters import proto_response_to_batch_result

        resp = TranscribeFileResponse(
            text="hello",
            language="en",
            duration=1.0,
        )
        result = proto_response_to_batch_result(resp)
        assert result.speaker_segments is None

    def test_with_speaker_segments(self) -> None:
        from macaw.scheduler.converters import proto_response_to_batch_result

        seg = SpeakerSegmentPb2(
            speaker_id="speaker_0",
            start=0.0,
            end=1.5,
            text="hello",
        )
        resp = TranscribeFileResponse(
            text="hello",
            language="en",
            duration=1.5,
            speaker_segments=[seg],
        )
        result = proto_response_to_batch_result(resp)
        assert result.speaker_segments is not None
        assert len(result.speaker_segments) == 1
        assert result.speaker_segments[0].speaker_id == "speaker_0"
        assert result.speaker_segments[0].start == pytest.approx(0.0)
        assert result.speaker_segments[0].end == pytest.approx(1.5)
        assert result.speaker_segments[0].text == "hello"


class TestTranscribeRequestDiarizeFields:
    """TranscribeRequest (internal) diarize and max_speakers fields."""

    def test_defaults(self) -> None:
        from macaw.server.models.requests import TranscribeRequest

        req = TranscribeRequest(
            request_id="r1",
            model_name="model",
            audio_data=b"\x00",
        )
        assert req.diarize is False
        assert req.max_speakers is None

    def test_set(self) -> None:
        from macaw.server.models.requests import TranscribeRequest

        req = TranscribeRequest(
            request_id="r1",
            model_name="model",
            audio_data=b"\x00",
            diarize=True,
            max_speakers=3,
        )
        assert req.diarize is True
        assert req.max_speakers == 3
