"""Tests for TTS cross-generation continuity fields (Sprint L, Task L2).

Validates:
 - SpeechRequest accepts previous_text, next_text, previous_request_ids, next_request_ids
 - TTSSpeakCommand accepts the same 4 fields
 - build_tts_proto_request packs fields into proto
 - proto_request_to_synthesize_params extracts them into options dict
 - previous_request_ids max 3 validated on SpeechRequest
 - next_request_ids max 3 validated on SpeechRequest
"""

from __future__ import annotations

import pytest

from macaw.scheduler.tts_converters import build_tts_proto_request
from macaw.server.models.events import TTSSpeakCommand
from macaw.server.models.speech import SpeechRequest
from macaw.workers.tts.converters import proto_request_to_synthesize_params

# ---------------------------------------------------------------------------
# SpeechRequest model tests
# ---------------------------------------------------------------------------


class TestSpeechRequestContinuity:
    """SpeechRequest accepts cross-generation continuity fields."""

    def test_default_values_are_none(self) -> None:
        req = SpeechRequest(model="test", input="Hello")
        assert req.previous_text is None
        assert req.next_text is None
        assert req.previous_request_ids is None
        assert req.next_request_ids is None

    def test_set_previous_text(self) -> None:
        req = SpeechRequest(model="test", input="World", previous_text="Hello")
        assert req.previous_text == "Hello"

    def test_set_next_text(self) -> None:
        req = SpeechRequest(model="test", input="Hello", next_text="World")
        assert req.next_text == "World"

    def test_set_previous_request_ids(self) -> None:
        req = SpeechRequest(model="test", input="Hello", previous_request_ids=["id1", "id2"])
        assert req.previous_request_ids == ["id1", "id2"]

    def test_set_next_request_ids(self) -> None:
        req = SpeechRequest(model="test", input="Hello", next_request_ids=["id1", "id2", "id3"])
        assert req.next_request_ids == ["id1", "id2", "id3"]

    def test_previous_request_ids_max_3(self) -> None:
        import pydantic

        with pytest.raises(pydantic.ValidationError):
            SpeechRequest(
                model="test",
                input="Hello",
                previous_request_ids=["id1", "id2", "id3", "id4"],
            )

    def test_next_request_ids_max_3(self) -> None:
        import pydantic

        with pytest.raises(pydantic.ValidationError):
            SpeechRequest(
                model="test",
                input="Hello",
                next_request_ids=["id1", "id2", "id3", "id4"],
            )


# ---------------------------------------------------------------------------
# TTSSpeakCommand model tests
# ---------------------------------------------------------------------------


class TestTTSSpeakCommandContinuity:
    """TTSSpeakCommand accepts cross-generation continuity fields."""

    def test_default_values_are_none(self) -> None:
        cmd = TTSSpeakCommand(text="Hello")
        assert cmd.previous_text is None
        assert cmd.next_text is None
        assert cmd.previous_request_ids is None
        assert cmd.next_request_ids is None

    def test_set_all_fields(self) -> None:
        cmd = TTSSpeakCommand(
            text="Hello",
            previous_text="Before",
            next_text="After",
            previous_request_ids=["p1", "p2"],
            next_request_ids=["n1"],
        )
        assert cmd.previous_text == "Before"
        assert cmd.next_text == "After"
        assert cmd.previous_request_ids == ["p1", "p2"]
        assert cmd.next_request_ids == ["n1"]


# ---------------------------------------------------------------------------
# build_tts_proto_request tests
# ---------------------------------------------------------------------------


class TestBuildTTSProtoRequestContinuity:
    """build_tts_proto_request packs continuity fields into proto."""

    def test_previous_text_set_on_proto(self) -> None:
        req = build_tts_proto_request(
            request_id="r1",
            text="Hello",
            voice="default",
            sample_rate=24000,
            speed=1.0,
            previous_text="Before",
        )
        assert req.previous_text == "Before"

    def test_next_text_set_on_proto(self) -> None:
        req = build_tts_proto_request(
            request_id="r1",
            text="Hello",
            voice="default",
            sample_rate=24000,
            speed=1.0,
            next_text="After",
        )
        assert req.next_text == "After"

    def test_previous_request_ids_set_on_proto(self) -> None:
        req = build_tts_proto_request(
            request_id="r1",
            text="Hello",
            voice="default",
            sample_rate=24000,
            speed=1.0,
            previous_request_ids=["p1", "p2"],
        )
        assert list(req.previous_request_ids) == ["p1", "p2"]

    def test_next_request_ids_set_on_proto(self) -> None:
        req = build_tts_proto_request(
            request_id="r1",
            text="Hello",
            voice="default",
            sample_rate=24000,
            speed=1.0,
            next_request_ids=["n1", "n2", "n3"],
        )
        assert list(req.next_request_ids) == ["n1", "n2", "n3"]

    def test_request_ids_truncated_to_3(self) -> None:
        """Even if caller passes >3, proto only gets first 3."""
        req = build_tts_proto_request(
            request_id="r1",
            text="Hello",
            voice="default",
            sample_rate=24000,
            speed=1.0,
            previous_request_ids=["p1", "p2", "p3", "p4"],
        )
        assert list(req.previous_request_ids) == ["p1", "p2", "p3"]

    def test_no_continuity_fields_leaves_proto_defaults(self) -> None:
        req = build_tts_proto_request(
            request_id="r1",
            text="Hello",
            voice="default",
            sample_rate=24000,
            speed=1.0,
        )
        assert req.previous_text == ""
        assert req.next_text == ""
        assert list(req.previous_request_ids) == []
        assert list(req.next_request_ids) == []

    def test_all_continuity_fields_together(self) -> None:
        req = build_tts_proto_request(
            request_id="r1",
            text="Current",
            voice="default",
            sample_rate=24000,
            speed=1.0,
            previous_text="Before",
            next_text="After",
            previous_request_ids=["p1"],
            next_request_ids=["n1", "n2"],
        )
        assert req.previous_text == "Before"
        assert req.next_text == "After"
        assert list(req.previous_request_ids) == ["p1"]
        assert list(req.next_request_ids) == ["n1", "n2"]


# ---------------------------------------------------------------------------
# proto_request_to_synthesize_params tests
# ---------------------------------------------------------------------------


class TestProtoToSynthesizeParamsContinuity:
    """proto_request_to_synthesize_params extracts continuity into options dict."""

    def test_previous_text_in_options(self) -> None:
        proto = build_tts_proto_request(
            request_id="r1",
            text="Hello",
            voice="default",
            sample_rate=24000,
            speed=1.0,
            previous_text="Before",
        )
        params = proto_request_to_synthesize_params(proto)
        assert params.options is not None
        assert params.options["previous_text"] == "Before"

    def test_next_text_in_options(self) -> None:
        proto = build_tts_proto_request(
            request_id="r1",
            text="Hello",
            voice="default",
            sample_rate=24000,
            speed=1.0,
            next_text="After",
        )
        params = proto_request_to_synthesize_params(proto)
        assert params.options is not None
        assert params.options["next_text"] == "After"

    def test_previous_request_ids_in_options(self) -> None:
        proto = build_tts_proto_request(
            request_id="r1",
            text="Hello",
            voice="default",
            sample_rate=24000,
            speed=1.0,
            previous_request_ids=["p1", "p2"],
        )
        params = proto_request_to_synthesize_params(proto)
        assert params.options is not None
        assert params.options["previous_request_ids"] == ["p1", "p2"]

    def test_next_request_ids_in_options(self) -> None:
        proto = build_tts_proto_request(
            request_id="r1",
            text="Hello",
            voice="default",
            sample_rate=24000,
            speed=1.0,
            next_request_ids=["n1"],
        )
        params = proto_request_to_synthesize_params(proto)
        assert params.options is not None
        assert params.options["next_request_ids"] == ["n1"]

    def test_no_continuity_fields_gives_no_options(self) -> None:
        proto = build_tts_proto_request(
            request_id="r1",
            text="Hello",
            voice="default",
            sample_rate=24000,
            speed=1.0,
        )
        params = proto_request_to_synthesize_params(proto)
        assert params.options is None

    def test_all_continuity_fields_in_options(self) -> None:
        proto = build_tts_proto_request(
            request_id="r1",
            text="Current",
            voice="default",
            sample_rate=24000,
            speed=1.0,
            previous_text="Before",
            next_text="After",
            previous_request_ids=["p1"],
            next_request_ids=["n1", "n2"],
        )
        params = proto_request_to_synthesize_params(proto)
        assert params.options is not None
        assert params.options["previous_text"] == "Before"
        assert params.options["next_text"] == "After"
        assert params.options["previous_request_ids"] == ["p1"]
        assert params.options["next_request_ids"] == ["n1", "n2"]
