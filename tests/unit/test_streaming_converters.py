"""Testes para transcript_segment_to_proto_event."""

from __future__ import annotations

from macaw._types import TranscriptSegment, WordTimestamp
from macaw.workers.stt.converters import transcript_segment_to_proto_event


class TestTranscriptSegmentToProtoEvent:
    def test_final_segment_all_fields(self) -> None:
        segment = TranscriptSegment(
            text="Ola, como posso ajudar?",
            is_final=True,
            segment_id=3,
            start_ms=1500,
            end_ms=4000,
            language="pt",
            confidence=0.95,
            words=(
                WordTimestamp(word="Ola", start=1.5, end=2.0, probability=0.9),
                WordTimestamp(word="como", start=2.1, end=2.4, probability=0.85),
            ),
        )
        event = transcript_segment_to_proto_event(segment, session_id="sess_abc")

        assert event.session_id == "sess_abc"
        assert event.event_type == "final"
        assert event.text == "Ola, como posso ajudar?"
        assert event.segment_id == 3
        assert event.start_ms == 1500
        assert event.end_ms == 4000
        assert event.language == "pt"
        assert abs(event.confidence - 0.95) < 0.01
        assert len(event.words) == 2

    def test_partial_segment_event_type(self) -> None:
        segment = TranscriptSegment(
            text="Ola como",
            is_final=False,
            segment_id=0,
        )
        event = transcript_segment_to_proto_event(segment, session_id="sess_1")

        assert event.event_type == "partial"
        assert event.text == "Ola como"
        assert event.segment_id == 0

    def test_none_optional_fields_default_to_zero_or_empty(self) -> None:
        segment = TranscriptSegment(
            text="test",
            is_final=True,
            segment_id=1,
            start_ms=None,
            end_ms=None,
            language=None,
            confidence=None,
            words=None,
        )
        event = transcript_segment_to_proto_event(segment, session_id="sess_2")

        assert event.start_ms == 0
        assert event.end_ms == 0
        assert event.language == ""
        assert abs(event.confidence) < 0.01
        assert len(event.words) == 0

    def test_words_converted_correctly(self) -> None:
        segment = TranscriptSegment(
            text="hello world",
            is_final=True,
            segment_id=0,
            words=(
                WordTimestamp(word="hello", start=0.0, end=0.5, probability=0.9),
                WordTimestamp(word="world", start=0.5, end=1.0, probability=None),
            ),
        )
        event = transcript_segment_to_proto_event(segment, session_id="sess_3")

        assert len(event.words) == 2
        assert event.words[0].word == "hello"
        assert abs(event.words[0].start - 0.0) < 0.01
        assert abs(event.words[0].end - 0.5) < 0.01
        assert abs(event.words[0].probability - 0.9) < 0.01
        assert event.words[1].word == "world"
        assert abs(event.words[1].probability) < 0.01  # None -> 0.0

    def test_session_id_propagated(self) -> None:
        segment = TranscriptSegment(
            text="test",
            is_final=False,
            segment_id=5,
        )
        event = transcript_segment_to_proto_event(segment, session_id="sess_custom_id")
        assert event.session_id == "sess_custom_id"

    def test_empty_text(self) -> None:
        segment = TranscriptSegment(
            text="",
            is_final=True,
            segment_id=0,
        )
        event = transcript_segment_to_proto_event(segment, session_id="sess_4")
        assert event.text == ""
        assert event.event_type == "final"
