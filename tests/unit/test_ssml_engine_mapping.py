"""Tests for SSML-to-engine directive mapping."""

from __future__ import annotations

from unittest.mock import MagicMock

from macaw._types import TTSEngineCapabilities
from macaw.postprocessing.ssml.directives import (
    BreakDirective,
    EmphasisDirective,
    PhonemeDirective,
    ProsodyDirective,
    SayAsDirective,
)
from macaw.workers.tts.ssml_mapper import SSMLMappingResult, map_ssml_directives

# ---------------------------------------------------------------------------
# map_ssml_directives — Break
# ---------------------------------------------------------------------------


class TestBreakMapping:
    def test_break_generates_silence(self) -> None:
        result = map_ssml_directives(
            "Hello world",
            [BreakDirective(time_ms=100, position=6)],
            sample_rate=16000,
        )
        assert len(result.silence_insertions) == 1
        pos, pcm = result.silence_insertions[0]
        assert pos == 6
        # 100ms at 16kHz = 1600 samples * 2 bytes = 3200 bytes
        assert len(pcm) == 3200
        assert pcm == b"\x00\x00" * 1600

    def test_break_500ms(self) -> None:
        result = map_ssml_directives(
            "Hello",
            [BreakDirective(time_ms=500, position=0)],
            sample_rate=16000,
        )
        _, pcm = result.silence_insertions[0]
        # 500ms at 16kHz = 8000 samples * 2 bytes = 16000 bytes
        assert len(pcm) == 16000

    def test_multiple_breaks(self) -> None:
        result = map_ssml_directives(
            "Hello world",
            [
                BreakDirective(time_ms=100, position=5),
                BreakDirective(time_ms=200, position=6),
            ],
            sample_rate=16000,
        )
        assert len(result.silence_insertions) == 2


# ---------------------------------------------------------------------------
# map_ssml_directives — Prosody
# ---------------------------------------------------------------------------


class TestProsodyMapping:
    def test_prosody_slow(self) -> None:
        result = map_ssml_directives(
            "Hello",
            [ProsodyDirective(text="Hello", rate="slow", position=0)],
        )
        assert result.speed_override == 0.75

    def test_prosody_fast(self) -> None:
        result = map_ssml_directives(
            "Hello",
            [ProsodyDirective(text="Hello", rate="fast", position=0)],
        )
        assert result.speed_override == 1.25

    def test_prosody_x_slow(self) -> None:
        result = map_ssml_directives(
            "Hello",
            [ProsodyDirective(text="Hello", rate="x-slow", position=0)],
        )
        assert result.speed_override == 0.5

    def test_prosody_x_fast(self) -> None:
        result = map_ssml_directives(
            "Hello",
            [ProsodyDirective(text="Hello", rate="x-fast", position=0)],
        )
        assert result.speed_override == 1.5

    def test_prosody_medium(self) -> None:
        result = map_ssml_directives(
            "Hello",
            [ProsodyDirective(text="Hello", rate="medium", position=0)],
        )
        assert result.speed_override == 1.0

    def test_prosody_percentage(self) -> None:
        result = map_ssml_directives(
            "Hello",
            [ProsodyDirective(text="Hello", rate="120%", position=0)],
        )
        assert result.speed_override == 1.2

    def test_prosody_invalid_rate_logged(self) -> None:
        result = map_ssml_directives(
            "Hello",
            [ProsodyDirective(text="Hello", rate="bogus", position=0)],
            engine_name="test-engine",
        )
        assert result.speed_override is None

    def test_prosody_pitch_only_no_speed(self) -> None:
        result = map_ssml_directives(
            "Hello",
            [ProsodyDirective(text="Hello", pitch="high", position=0)],
        )
        assert result.speed_override is None


# ---------------------------------------------------------------------------
# map_ssml_directives — Emphasis
# ---------------------------------------------------------------------------


class TestEmphasisMapping:
    def test_emphasis_strong(self) -> None:
        result = map_ssml_directives(
            "important",
            [EmphasisDirective(text="important", level="strong", position=0)],
        )
        assert result.speed_override == 0.8

    def test_emphasis_moderate(self) -> None:
        result = map_ssml_directives(
            "word",
            [EmphasisDirective(text="word", level="moderate", position=0)],
        )
        assert result.speed_override == 0.9

    def test_emphasis_reduced(self) -> None:
        result = map_ssml_directives(
            "word",
            [EmphasisDirective(text="word", level="reduced", position=0)],
        )
        assert result.speed_override == 1.1

    def test_emphasis_none(self) -> None:
        result = map_ssml_directives(
            "word",
            [EmphasisDirective(text="word", level="none", position=0)],
        )
        # 1.0 means no override needed
        assert result.speed_override is None


# ---------------------------------------------------------------------------
# map_ssml_directives — SayAs and Phoneme (pass-through)
# ---------------------------------------------------------------------------


class TestSayAsAndPhonemeMapping:
    def test_say_as_passes_through_text(self) -> None:
        result = map_ssml_directives(
            "42",
            [SayAsDirective(text="42", interpret_as="cardinal", position=0)],
        )
        assert result.text == "42"
        assert result.speed_override is None
        assert result.silence_insertions == []

    def test_phoneme_passes_through_text(self) -> None:
        result = map_ssml_directives(
            "tomato",
            [PhonemeDirective(text="tomato", alphabet="ipa", ph="t@meItoU", position=0)],
        )
        assert result.text == "tomato"
        assert result.speed_override is None


# ---------------------------------------------------------------------------
# SSMLMappingResult
# ---------------------------------------------------------------------------


class TestSSMLMappingResult:
    def test_result_attributes(self) -> None:
        result = SSMLMappingResult(
            text="Hello",
            silence_insertions=[(0, b"\x00\x00")],
            speed_override=0.75,
        )
        assert result.text == "Hello"
        assert len(result.silence_insertions) == 1
        assert result.speed_override == 0.75

    def test_result_no_speed_override(self) -> None:
        result = SSMLMappingResult(text="Hello", silence_insertions=[])
        assert result.speed_override is None

    def test_no_directives_empty_result(self) -> None:
        result = map_ssml_directives("Hello world", [])
        assert result.text == "Hello world"
        assert result.silence_insertions == []
        assert result.speed_override is None


# ---------------------------------------------------------------------------
# TTSBackend.apply_ssml_directives (default implementation)
# ---------------------------------------------------------------------------


class TestTTSBackendDefaultSSML:
    def test_default_handles_break(self) -> None:
        from macaw.workers.tts.interface import TTSBackend

        # Create a concrete mock subclass
        backend = MagicMock(spec=TTSBackend)
        backend.apply_ssml_directives = TTSBackend.apply_ssml_directives.__get__(backend)

        text, silences = backend.apply_ssml_directives(
            "Hello world",
            [BreakDirective(time_ms=100, position=6)],
            sample_rate=16000,
        )
        assert text == "Hello world"
        assert len(silences) == 1
        pos, pcm = silences[0]
        assert pos == 6
        assert len(pcm) == 3200  # 100ms at 16kHz

    def test_default_ignores_non_break_directives(self) -> None:
        from macaw.workers.tts.interface import TTSBackend

        backend = MagicMock(spec=TTSBackend)
        backend.apply_ssml_directives = TTSBackend.apply_ssml_directives.__get__(backend)

        text, silences = backend.apply_ssml_directives(
            "Hello",
            [ProsodyDirective(text="Hello", rate="slow", position=0)],
            sample_rate=16000,
        )
        assert text == "Hello"
        assert silences == []


# ---------------------------------------------------------------------------
# TTSEngineCapabilities — supports_ssml field
# ---------------------------------------------------------------------------


class TestSSMLCapabilities:
    def test_default_false(self) -> None:
        caps = TTSEngineCapabilities()
        assert caps.supports_ssml is False

    def test_explicit_true(self) -> None:
        caps = TTSEngineCapabilities(supports_ssml=True)
        assert caps.supports_ssml is True


# ---------------------------------------------------------------------------
# Mixed directives
# ---------------------------------------------------------------------------


class TestMixedDirectives:
    def test_break_and_prosody_combined(self) -> None:
        result = map_ssml_directives(
            "Hello world",
            [
                BreakDirective(time_ms=200, position=6),
                ProsodyDirective(text="world", rate="slow", position=6),
            ],
            sample_rate=16000,
        )
        assert len(result.silence_insertions) == 1
        assert result.speed_override == 0.75

    def test_prosody_overrides_emphasis(self) -> None:
        """When both prosody and emphasis exist, last wins."""
        result = map_ssml_directives(
            "Hello world",
            [
                ProsodyDirective(text="Hello", rate="fast", position=0),
                EmphasisDirective(text="world", level="strong", position=6),
            ],
        )
        # Last directive's speed wins
        assert result.speed_override == 0.8
