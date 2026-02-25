"""Tests for SSML parser — directive extraction from SSML markup."""

from __future__ import annotations

import pytest

from macaw.postprocessing.ssml.directives import (
    BreakDirective,
    EmphasisDirective,
    PhonemeDirective,
    ProsodyDirective,
    SayAsDirective,
)
from macaw.postprocessing.ssml.parser import SSMLParseError, SSMLParser

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _parser() -> SSMLParser:
    return SSMLParser()


# ---------------------------------------------------------------------------
# Break directive
# ---------------------------------------------------------------------------


class TestBreakDirective:
    def test_break_with_milliseconds(self) -> None:
        result = _parser().parse('<speak>Hello <break time="500ms"/> world</speak>')
        assert result.text == "Hello  world"
        assert len(result.directives) == 1
        d = result.directives[0]
        assert isinstance(d, BreakDirective)
        assert d.time_ms == 500
        assert d.position == 6

    def test_break_with_seconds(self) -> None:
        result = _parser().parse('<speak>Hello <break time="1.5s"/> world</speak>')
        d = result.directives[0]
        assert isinstance(d, BreakDirective)
        assert d.time_ms == 1500

    def test_break_without_time_defaults_250ms(self) -> None:
        result = _parser().parse("<speak>Hello <break/> world</speak>")
        d = result.directives[0]
        assert isinstance(d, BreakDirective)
        assert d.time_ms == 250

    def test_break_invalid_time_ignored(self) -> None:
        result = _parser().parse('<speak>Hello <break time="invalid"/> world</speak>')
        assert len(result.directives) == 0
        assert result.text == "Hello  world"


# ---------------------------------------------------------------------------
# Prosody directive
# ---------------------------------------------------------------------------


class TestProsodyDirective:
    def test_prosody_rate(self) -> None:
        result = _parser().parse('<speak><prosody rate="slow">Hello world</prosody></speak>')
        assert result.text == "Hello world"
        assert len(result.directives) == 1
        d = result.directives[0]
        assert isinstance(d, ProsodyDirective)
        assert d.text == "Hello world"
        assert d.rate == "slow"
        assert d.pitch is None

    def test_prosody_pitch(self) -> None:
        result = _parser().parse('<speak><prosody pitch="high">Hello</prosody></speak>')
        d = result.directives[0]
        assert isinstance(d, ProsodyDirective)
        assert d.pitch == "high"
        assert d.rate is None

    def test_prosody_rate_and_pitch(self) -> None:
        result = _parser().parse(
            '<speak><prosody rate="fast" pitch="+2st">Hello</prosody></speak>'
        )
        d = result.directives[0]
        assert isinstance(d, ProsodyDirective)
        assert d.rate == "fast"
        assert d.pitch == "+2st"

    def test_prosody_percentage_rate(self) -> None:
        result = _parser().parse('<speak><prosody rate="120%">Hello</prosody></speak>')
        d = result.directives[0]
        assert isinstance(d, ProsodyDirective)
        assert d.rate == "120%"


# ---------------------------------------------------------------------------
# Emphasis directive
# ---------------------------------------------------------------------------


class TestEmphasisDirective:
    def test_emphasis_strong(self) -> None:
        result = _parser().parse(
            '<speak><emphasis level="strong">important</emphasis> text</speak>'
        )
        assert result.text == "important text"
        d = result.directives[0]
        assert isinstance(d, EmphasisDirective)
        assert d.text == "important"
        assert d.level == "strong"

    def test_emphasis_default_moderate(self) -> None:
        result = _parser().parse("<speak><emphasis>word</emphasis></speak>")
        d = result.directives[0]
        assert isinstance(d, EmphasisDirective)
        assert d.level == "moderate"

    def test_emphasis_invalid_level_defaults_moderate(self) -> None:
        result = _parser().parse('<speak><emphasis level="bogus">word</emphasis></speak>')
        d = result.directives[0]
        assert isinstance(d, EmphasisDirective)
        assert d.level == "moderate"


# ---------------------------------------------------------------------------
# SayAs directive
# ---------------------------------------------------------------------------


class TestSayAsDirective:
    def test_say_as_cardinal(self) -> None:
        result = _parser().parse('<speak><say-as interpret-as="cardinal">42</say-as></speak>')
        assert result.text == "42"
        d = result.directives[0]
        assert isinstance(d, SayAsDirective)
        assert d.text == "42"
        assert d.interpret_as == "cardinal"
        assert d.format_str is None

    def test_say_as_date_with_format(self) -> None:
        result = _parser().parse(
            '<speak><say-as interpret-as="date" format="mdy">12/25/2025</say-as></speak>'
        )
        d = result.directives[0]
        assert isinstance(d, SayAsDirective)
        assert d.interpret_as == "date"
        assert d.format_str == "mdy"

    def test_say_as_missing_interpret_as_preserves_text(self) -> None:
        result = _parser().parse("<speak><say-as>42</say-as></speak>")
        assert result.text == "42"
        assert len(result.directives) == 0


# ---------------------------------------------------------------------------
# Phoneme directive
# ---------------------------------------------------------------------------


class TestPhonemeDirective:
    def test_phoneme_ipa(self) -> None:
        result = _parser().parse(
            '<speak><phoneme alphabet="ipa" ph="t@m.eI.toU">tomato</phoneme></speak>'
        )
        assert result.text == "tomato"
        d = result.directives[0]
        assert isinstance(d, PhonemeDirective)
        assert d.text == "tomato"
        assert d.alphabet == "ipa"
        assert d.ph == "t@m.eI.toU"

    def test_phoneme_default_alphabet_ipa(self) -> None:
        result = _parser().parse('<speak><phoneme ph="hEloU">hello</phoneme></speak>')
        d = result.directives[0]
        assert isinstance(d, PhonemeDirective)
        assert d.alphabet == "ipa"

    def test_phoneme_missing_ph_preserves_text(self) -> None:
        result = _parser().parse('<speak><phoneme alphabet="ipa">hello</phoneme></speak>')
        assert result.text == "hello"
        assert len(result.directives) == 0


# ---------------------------------------------------------------------------
# General parsing behavior
# ---------------------------------------------------------------------------


class TestGeneralParsing:
    def test_plain_text_no_directives(self) -> None:
        result = _parser().parse("Hello world")
        assert result.text == "Hello world"
        assert result.directives == []

    def test_auto_wraps_in_speak(self) -> None:
        result = _parser().parse("Hello world")
        assert result.text == "Hello world"

    def test_unsupported_tag_stripped_text_preserved(self) -> None:
        result = _parser().parse("<speak><voice name='en'>Hello</voice></speak>")
        assert result.text == "Hello"
        assert len(result.directives) == 0

    def test_nested_unsupported_tags_all_text_preserved(self) -> None:
        result = _parser().parse(
            "<speak><voice name='en'><p>Hello <s>world</s></p></voice></speak>"
        )
        assert result.text == "Hello world"
        assert len(result.directives) == 0

    def test_invalid_xml_raises_ssml_parse_error(self) -> None:
        with pytest.raises(SSMLParseError, match="Invalid SSML"):
            _parser().parse("<speak><break</speak>")

    def test_empty_speak(self) -> None:
        result = _parser().parse("<speak></speak>")
        assert result.text == ""
        assert result.directives == []

    def test_multiple_directives_in_sequence(self) -> None:
        ssml = (
            "<speak>"
            'Hello <break time="200ms"/>'
            '<prosody rate="fast">quick text</prosody> '
            '<emphasis level="strong">important</emphasis>'
            "</speak>"
        )
        result = _parser().parse(ssml)
        assert result.text == "Hello quick text important"
        assert len(result.directives) == 3
        assert isinstance(result.directives[0], BreakDirective)
        assert isinstance(result.directives[1], ProsodyDirective)
        assert isinstance(result.directives[2], EmphasisDirective)

    def test_mixed_text_and_tags(self) -> None:
        ssml = '<speak>Start <break time="100ms"/> middle <break time="200ms"/> end</speak>'
        result = _parser().parse(ssml)
        assert result.text == "Start  middle  end"
        assert len(result.directives) == 2

    def test_directive_positions_track_correctly(self) -> None:
        ssml = '<speak>AB<break time="100ms"/>CD</speak>'
        result = _parser().parse(ssml)
        d = result.directives[0]
        assert isinstance(d, BreakDirective)
        assert d.position == 2  # After "AB"
