"""Smoke tests verifying the diarization infrastructure is ready.

These tests validate that all proto fields, dataclasses, converters, manifest
fields, and API parameters needed for diarization are in place. They do NOT
test diarization accuracy -- only that the plumbing exists and round-trips
correctly.
"""

from __future__ import annotations

import pytest

from macaw._types import (
    BatchResult,
    EngineCapabilities,
    SegmentDetail,
    SpeakerSegment,
)
from macaw.config.manifest import ModelCapabilities
from macaw.proto.stt_worker_pb2 import (
    SpeakerSegment as SpeakerSegmentPb2,
)
from macaw.proto.stt_worker_pb2 import (
    TranscribeFileRequest,
    TranscribeFileResponse,
    TranscriptEvent,
)
from macaw.workers.stt.converters import (
    speaker_segment_to_dataclass,
    speaker_segment_to_proto,
)


class TestSpeakerSegmentDataclass:
    """SpeakerSegment dataclass creation, immutability, and slot optimization."""

    def test_creation_with_all_fields(self) -> None:
        seg = SpeakerSegment(
            speaker_id="speaker_0",
            start=1.5,
            end=3.7,
            text="Hello from speaker zero",
        )
        assert seg.speaker_id == "speaker_0"
        assert seg.start == 1.5
        assert seg.end == 3.7
        assert seg.text == "Hello from speaker zero"

    def test_frozen_rejects_mutation(self) -> None:
        seg = SpeakerSegment(speaker_id="s0", start=0.0, end=1.0, text="hi")
        with pytest.raises(AttributeError):
            seg.speaker_id = "s1"  # type: ignore[misc]

    def test_slots_no_dict(self) -> None:
        seg = SpeakerSegment(speaker_id="s0", start=0.0, end=1.0, text="hi")
        assert not hasattr(seg, "__dict__")


class TestEngineCapabilitiesDiarization:
    """EngineCapabilities.supports_diarization field defaults and override."""

    def test_default_is_false(self) -> None:
        caps = EngineCapabilities()
        assert caps.supports_diarization is False

    def test_can_enable(self) -> None:
        caps = EngineCapabilities(supports_diarization=True)
        assert caps.supports_diarization is True


class TestBatchResultWithSpeakerSegments:
    """BatchResult accepts and stores speaker_segments."""

    def test_default_is_none(self) -> None:
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


class TestProtoTranscribeFileRequestDiarizeField:
    """Proto TranscribeFileRequest has diarize and max_speakers fields."""

    def test_diarize_default_false(self) -> None:
        req = TranscribeFileRequest()
        assert req.diarize is False

    def test_diarize_set_true(self) -> None:
        req = TranscribeFileRequest(diarize=True)
        assert req.diarize is True

    def test_max_speakers_default_zero(self) -> None:
        req = TranscribeFileRequest()
        assert req.max_speakers == 0

    def test_max_speakers_set(self) -> None:
        req = TranscribeFileRequest(max_speakers=5)
        assert req.max_speakers == 5


class TestProtoTranscribeFileResponseSpeakerSegments:
    """Proto TranscribeFileResponse has speaker_segments repeated field."""

    def test_empty_by_default(self) -> None:
        resp = TranscribeFileResponse()
        assert len(resp.speaker_segments) == 0

    def test_with_segments(self) -> None:
        seg = SpeakerSegmentPb2(
            speaker_id="speaker_0",
            start=0.5,
            end=2.0,
            text="Hello",
        )
        resp = TranscribeFileResponse(speaker_segments=[seg])
        assert len(resp.speaker_segments) == 1
        assert resp.speaker_segments[0].speaker_id == "speaker_0"


class TestProtoSpeakerSegmentMessage:
    """Proto SpeakerSegment message has all required fields."""

    def test_all_fields_present(self) -> None:
        seg = SpeakerSegmentPb2(
            speaker_id="speaker_2",
            start=1.5,
            end=3.0,
            text="Testing diarization",
        )
        assert seg.speaker_id == "speaker_2"
        assert seg.start == pytest.approx(1.5)
        assert seg.end == pytest.approx(3.0)
        assert seg.text == "Testing diarization"

    def test_defaults_are_empty(self) -> None:
        seg = SpeakerSegmentPb2()
        assert seg.speaker_id == ""
        assert seg.start == pytest.approx(0.0)
        assert seg.end == pytest.approx(0.0)
        assert seg.text == ""


class TestProtoTranscriptEventSpeakerId:
    """Proto TranscriptEvent has speaker_id field for streaming diarization."""

    def test_default_empty_string(self) -> None:
        event = TranscriptEvent()
        assert event.speaker_id == ""

    def test_set_speaker_id(self) -> None:
        event = TranscriptEvent(speaker_id="speaker_1")
        assert event.speaker_id == "speaker_1"


class TestModelCapabilitiesDiarizationField:
    """ModelCapabilities (manifest) has diarization field."""

    def test_default_false(self) -> None:
        caps = ModelCapabilities()
        assert caps.diarization is False

    def test_can_enable(self) -> None:
        caps = ModelCapabilities(diarization=True)
        assert caps.diarization is True


class TestConverterSpeakerSegmentRoundtrip:
    """Converter functions round-trip between dataclass and proto."""

    def test_dataclass_to_proto_to_dataclass(self) -> None:
        original = SpeakerSegment(
            speaker_id="speaker_0",
            start=0.1,
            end=1.5,
            text="round trip test",
        )
        proto = speaker_segment_to_proto(original)
        restored = speaker_segment_to_dataclass(proto)

        assert restored.speaker_id == original.speaker_id
        assert restored.start == pytest.approx(original.start)
        assert restored.end == pytest.approx(original.end)
        assert restored.text == original.text

    def test_proto_to_dataclass_to_proto(self) -> None:
        original_proto = SpeakerSegmentPb2(
            speaker_id="speaker_1",
            start=2.0,
            end=4.5,
            text="reverse trip",
        )
        dc = speaker_segment_to_dataclass(original_proto)
        restored_proto = speaker_segment_to_proto(dc)

        assert restored_proto.speaker_id == original_proto.speaker_id
        assert restored_proto.start == pytest.approx(original_proto.start)
        assert restored_proto.end == pytest.approx(original_proto.end)
        assert restored_proto.text == original_proto.text


class TestTranscriptionEndpointAcceptsDiarizeParam:
    """REST transcription endpoint declares diarize as a Form parameter."""

    @staticmethod
    def _get_transcription_form_properties() -> dict[str, object]:
        """Extract form properties from the transcriptions endpoint schema.

        FastAPI generates a $ref to components/schemas for multipart
        request bodies, so we resolve the reference to get the actual
        properties dict.
        """
        from macaw.server.app import create_app

        app = create_app()
        schema = app.openapi()

        transcriptions_path = schema.get("paths", {}).get("/v1/audio/transcriptions", {})
        post_op = transcriptions_path.get("post", {})
        request_body = post_op.get("requestBody", {})
        content = request_body.get("content", {})
        form_data = content.get("multipart/form-data", {})
        form_schema = form_data.get("schema", {})

        # Resolve $ref if present (FastAPI places form body schemas
        # under components/schemas via a $ref pointer).
        ref = form_schema.get("$ref", "")
        if ref:
            # $ref format: "#/components/schemas/SomeName"
            ref_name = ref.rsplit("/", 1)[-1]
            form_schema = schema.get("components", {}).get("schemas", {}).get(ref_name, {})

        return form_schema.get("properties", {})  # type: ignore[return-value]

    def test_diarize_param_in_openapi_schema(self) -> None:
        properties = self._get_transcription_form_properties()
        assert "diarize" in properties, (
            "diarize field not found in transcriptions endpoint form schema"
        )

    def test_max_speakers_param_in_openapi_schema(self) -> None:
        properties = self._get_transcription_form_properties()
        assert "max_speakers" in properties, (
            "max_speakers field not found in transcriptions endpoint form schema"
        )
