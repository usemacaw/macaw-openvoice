"""Testes dos tipos fundamentais do Macaw."""

import pytest

from macaw._types import (
    BatchResult,
    EngineCapabilities,
    ModelType,
    ResponseFormat,
    SegmentDetail,
    SessionState,
    STTArchitecture,
    TranscriptSegment,
    VADSensitivity,
    WordTimestamp,
)


class TestSTTArchitecture:
    def test_values(self) -> None:
        assert STTArchitecture.ENCODER_DECODER.value == "encoder-decoder"
        assert STTArchitecture.CTC.value == "ctc"
        assert STTArchitecture.STREAMING_NATIVE.value == "streaming-native"

    def test_from_string(self) -> None:
        assert STTArchitecture("encoder-decoder") == STTArchitecture.ENCODER_DECODER


class TestModelType:
    def test_values(self) -> None:
        assert ModelType.STT.value == "stt"
        assert ModelType.TTS.value == "tts"


class TestSessionState:
    def test_all_states_exist(self) -> None:
        states = {s.value for s in SessionState}
        assert states == {"init", "active", "silence", "hold", "closing", "closed"}


class TestVADSensitivity:
    def test_values(self) -> None:
        assert VADSensitivity.HIGH.value == "high"
        assert VADSensitivity.NORMAL.value == "normal"
        assert VADSensitivity.LOW.value == "low"


class TestResponseFormat:
    def test_all_formats(self) -> None:
        formats = {f.value for f in ResponseFormat}
        assert formats == {"json", "verbose_json", "text", "srt", "vtt"}


class TestWordTimestamp:
    def test_creation(self) -> None:
        w = WordTimestamp(word="ola", start=0.0, end=0.5)
        assert w.word == "ola"
        assert w.start == 0.0
        assert w.end == 0.5

    def test_frozen(self) -> None:
        w = WordTimestamp(word="ola", start=0.0, end=0.5)
        with pytest.raises(AttributeError):
            w.word = "outro"  # type: ignore[misc]


class TestTranscriptSegment:
    def test_minimal(self) -> None:
        seg = TranscriptSegment(text="ola", is_final=True, segment_id=0)
        assert seg.text == "ola"
        assert seg.is_final is True
        assert seg.start_ms is None

    def test_with_words(self) -> None:
        words = (WordTimestamp(word="ola", start=0.0, end=0.5),)
        seg = TranscriptSegment(
            text="ola",
            is_final=True,
            segment_id=0,
            words=words,
        )
        assert seg.words is not None
        assert seg.words[0].word == "ola"

    def test_frozen(self) -> None:
        seg = TranscriptSegment(text="ola", is_final=True, segment_id=0)
        with pytest.raises(AttributeError):
            seg.text = "outro"  # type: ignore[misc]


class TestBatchResult:
    def test_creation(self) -> None:
        result = BatchResult(
            text="ola mundo",
            language="pt",
            duration=1.5,
            segments=(SegmentDetail(id=0, start=0.0, end=1.5, text="ola mundo"),),
        )
        assert result.text == "ola mundo"
        assert result.language == "pt"
        assert len(result.segments) == 1

    def test_with_words(self) -> None:
        result = BatchResult(
            text="ola",
            language="pt",
            duration=0.5,
            segments=(),
            words=(WordTimestamp(word="ola", start=0.0, end=0.5),),
        )
        assert result.words is not None
        assert len(result.words) == 1


class TestEngineCapabilities:
    def test_defaults(self) -> None:
        caps = EngineCapabilities()
        assert caps.supports_hot_words is False
        assert caps.supports_initial_prompt is False
        assert caps.supports_batch is False
        assert caps.supports_word_timestamps is False
        assert caps.max_concurrent_sessions == 1

    def test_custom_values(self) -> None:
        caps = EngineCapabilities(
            supports_hot_words=True,
            max_concurrent_sessions=4,
        )
        assert caps.supports_hot_words is True
        assert caps.max_concurrent_sessions == 4
