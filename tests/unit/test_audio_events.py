"""Tests for tag_audio_events STT parameter (Sprint U2)."""

from __future__ import annotations

from macaw.server.models.requests import TranscribeRequest
from macaw.server.models.responses import WordResponse

# ---------------------------------------------------------------------------
# TranscribeRequest — tag_audio_events field
# ---------------------------------------------------------------------------


class TestTagAudioEventsField:
    def test_default_is_true(self) -> None:
        req = TranscribeRequest(
            request_id="r1",
            model_name="test",
            audio_data=b"\x00",
        )
        assert req.tag_audio_events is True

    def test_can_set_false(self) -> None:
        req = TranscribeRequest(
            request_id="r1",
            model_name="test",
            audio_data=b"\x00",
            tag_audio_events=False,
        )
        assert req.tag_audio_events is False


# ---------------------------------------------------------------------------
# WordResponse — word_type field
# ---------------------------------------------------------------------------


class TestWordTypeField:
    def test_default_is_word(self) -> None:
        w = WordResponse(word="hello", start=0.0, end=0.5)
        assert w.word_type == "word"

    def test_audio_event_type(self) -> None:
        w = WordResponse(word="[laughter]", start=1.0, end=2.0, word_type="audio_event")
        assert w.word_type == "audio_event"

    def test_channel_index_default_none(self) -> None:
        w = WordResponse(word="hello", start=0.0, end=0.5)
        assert w.channel_index is None

    def test_channel_index_set(self) -> None:
        w = WordResponse(word="hello", start=0.0, end=0.5, channel_index=1)
        assert w.channel_index == 1

    def test_serialization_includes_word_type(self) -> None:
        w = WordResponse(word="hello", start=0.0, end=0.5)
        data = w.model_dump()
        assert "word_type" in data
        assert data["word_type"] == "word"
