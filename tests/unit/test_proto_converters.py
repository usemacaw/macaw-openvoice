"""Testes para conversores proto <-> tipos Macaw."""

from __future__ import annotations

from macaw._types import BatchResult, SegmentDetail, WordTimestamp
from macaw.proto.stt_worker_pb2 import TranscribeFileRequest
from macaw.workers.stt.converters import (
    batch_result_to_proto_response,
    health_dict_to_proto_response,
    proto_request_to_transcribe_params,
    segment_detail_to_proto,
    word_timestamp_to_proto,
)


class TestProtoRequestToTranscribeParams:
    def test_audio_data_extracted(self) -> None:
        request = TranscribeFileRequest(audio_data=b"\x00\x01\x02\x03")
        params = proto_request_to_transcribe_params(request)
        assert params["audio_data"] == b"\x00\x01\x02\x03"

    def test_empty_language_becomes_none(self) -> None:
        request = TranscribeFileRequest(language="")
        params = proto_request_to_transcribe_params(request)
        assert params["language"] is None

    def test_present_language_preserved(self) -> None:
        request = TranscribeFileRequest(language="pt")
        params = proto_request_to_transcribe_params(request)
        assert params["language"] == "pt"

    def test_empty_prompt_becomes_none(self) -> None:
        request = TranscribeFileRequest(initial_prompt="")
        params = proto_request_to_transcribe_params(request)
        assert params["initial_prompt"] is None

    def test_present_prompt_preserved(self) -> None:
        request = TranscribeFileRequest(initial_prompt="Termos: PIX, TED")
        params = proto_request_to_transcribe_params(request)
        assert params["initial_prompt"] == "Termos: PIX, TED"

    def test_empty_hot_words_becomes_none(self) -> None:
        request = TranscribeFileRequest()
        params = proto_request_to_transcribe_params(request)
        assert params["hot_words"] is None

    def test_hot_words_converted_to_list(self) -> None:
        request = TranscribeFileRequest(hot_words=["PIX", "TED"])
        params = proto_request_to_transcribe_params(request)
        assert params["hot_words"] == ["PIX", "TED"]

    def test_word_granularity_enables_word_timestamps(self) -> None:
        request = TranscribeFileRequest(timestamp_granularities=["word"])
        params = proto_request_to_transcribe_params(request)
        assert params["word_timestamps"] is True

    def test_segment_granularity_disables_word_timestamps(self) -> None:
        request = TranscribeFileRequest(timestamp_granularities=["segment"])
        params = proto_request_to_transcribe_params(request)
        assert params["word_timestamps"] is False

    def test_temperature_preserved(self) -> None:
        request = TranscribeFileRequest(temperature=0.5)
        params = proto_request_to_transcribe_params(request)
        assert abs(params["temperature"] - 0.5) < 0.01  # type: ignore[operator]


class TestBatchResultToProtoResponse:
    def test_text_and_metadata(self, sample_batch_result: BatchResult) -> None:
        response = batch_result_to_proto_response(sample_batch_result)
        assert response.text == "Ola, como posso ajudar?"
        assert response.language == "pt"
        assert abs(response.duration - 2.5) < 0.01

    def test_segments_mapped(self, sample_batch_result: BatchResult) -> None:
        response = batch_result_to_proto_response(sample_batch_result)
        assert len(response.segments) == 1
        seg = response.segments[0]
        assert seg.text == "Ola, como posso ajudar?"
        assert abs(seg.compression_ratio - 1.1) < 0.01

    def test_words_mapped(self, sample_batch_result: BatchResult) -> None:
        response = batch_result_to_proto_response(sample_batch_result)
        assert len(response.words) == 4
        assert response.words[0].word == "Ola"
        assert abs(response.words[0].probability - 0.95) < 0.01

    def test_words_none_produces_empty_list(self) -> None:
        result = BatchResult(
            text="test",
            language="en",
            duration=1.0,
            segments=(),
            words=None,
        )
        response = batch_result_to_proto_response(result)
        assert len(response.words) == 0


class TestSegmentDetailToProto:
    def test_all_fields_mapped(self) -> None:
        detail = SegmentDetail(
            id=0,
            start=0.0,
            end=2.5,
            text="hello",
            avg_logprob=-0.3,
            no_speech_prob=0.01,
            compression_ratio=1.2,
        )
        proto = segment_detail_to_proto(detail)
        assert proto.id == 0
        assert abs(proto.start - 0.0) < 0.01
        assert abs(proto.end - 2.5) < 0.01
        assert proto.text == "hello"
        assert abs(proto.compression_ratio - 1.2) < 0.01


class TestWordTimestampToProto:
    def test_with_probability(self) -> None:
        word = WordTimestamp(word="test", start=0.0, end=0.5, probability=0.9)
        proto = word_timestamp_to_proto(word)
        assert proto.word == "test"
        assert abs(proto.probability - 0.9) < 0.01

    def test_none_probability_becomes_zero(self) -> None:
        word = WordTimestamp(word="test", start=0.0, end=0.5, probability=None)
        proto = word_timestamp_to_proto(word)
        assert abs(proto.probability) < 0.01


class TestHealthDictToProtoResponse:
    def test_ok_status(self) -> None:
        response = health_dict_to_proto_response(
            {"status": "ok"}, model_name="large-v3", engine="faster-whisper"
        )
        assert response.status == "ok"
        assert response.model_name == "large-v3"
        assert response.engine == "faster-whisper"

    def test_missing_status_returns_unknown(self) -> None:
        response = health_dict_to_proto_response({}, model_name="test", engine="test")
        assert response.status == "unknown"
