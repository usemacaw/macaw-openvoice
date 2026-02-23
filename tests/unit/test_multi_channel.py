"""Tests for multi-channel audio STT parameter (Sprint U3)."""

from __future__ import annotations

from macaw.server.models.requests import TranscribeRequest
from macaw.server.models.responses import WordResponse

# ---------------------------------------------------------------------------
# TranscribeRequest — use_multi_channel field
# ---------------------------------------------------------------------------


class TestMultiChannelField:
    def test_default_is_false(self) -> None:
        req = TranscribeRequest(
            request_id="r1",
            model_name="test",
            audio_data=b"\x00",
        )
        assert req.use_multi_channel is False

    def test_can_set_true(self) -> None:
        req = TranscribeRequest(
            request_id="r1",
            model_name="test",
            audio_data=b"\x00",
            use_multi_channel=True,
        )
        assert req.use_multi_channel is True


# ---------------------------------------------------------------------------
# WordResponse — channel_index field
# ---------------------------------------------------------------------------


class TestChannelIndexField:
    def test_default_none(self) -> None:
        w = WordResponse(word="hello", start=0.0, end=0.5)
        assert w.channel_index is None

    def test_set_channel_index(self) -> None:
        w = WordResponse(word="hello", start=0.0, end=0.5, channel_index=0)
        assert w.channel_index == 0

    def test_multiple_channels(self) -> None:
        w0 = WordResponse(word="hello", start=0.0, end=0.5, channel_index=0)
        w1 = WordResponse(word="world", start=0.0, end=0.5, channel_index=1)
        assert w0.channel_index != w1.channel_index

    def test_channel_index_serialization(self) -> None:
        w = WordResponse(word="hello", start=0.0, end=0.5, channel_index=2)
        data = w.model_dump()
        assert data["channel_index"] == 2

    def test_channel_index_none_in_serialization(self) -> None:
        w = WordResponse(word="hello", start=0.0, end=0.5)
        data = w.model_dump()
        assert data["channel_index"] is None


# ---------------------------------------------------------------------------
# Combined fields
# ---------------------------------------------------------------------------


class TestMultiChannelWithAudioEvents:
    def test_all_new_fields_together(self) -> None:
        req = TranscribeRequest(
            request_id="r1",
            model_name="test",
            audio_data=b"\x00",
            tag_audio_events=True,
            use_multi_channel=True,
        )
        assert req.tag_audio_events is True
        assert req.use_multi_channel is True

    def test_word_with_all_new_fields(self) -> None:
        w = WordResponse(
            word="[applause]",
            start=0.0,
            end=1.0,
            word_type="audio_event",
            channel_index=0,
        )
        assert w.word_type == "audio_event"
        assert w.channel_index == 0
