"""Tests for SSML API wiring — REST and WebSocket endpoints."""

from __future__ import annotations

from unittest.mock import MagicMock

import pytest

from macaw.server.models.events import TTSSpeakCommand
from macaw.server.models.speech import SpeechRequest

# ---------------------------------------------------------------------------
# SpeechRequest model — enable_ssml_parsing field
# ---------------------------------------------------------------------------


class TestSpeechRequestSSML:
    def test_default_false(self) -> None:
        req = SpeechRequest(model="kokoro-v1", input="Hello")
        assert req.enable_ssml_parsing is False

    def test_explicit_true(self) -> None:
        req = SpeechRequest(model="kokoro-v1", input="Hello", enable_ssml_parsing=True)
        assert req.enable_ssml_parsing is True

    def test_ssml_input_with_flag(self) -> None:
        req = SpeechRequest(
            model="kokoro-v1",
            input='<speak>Hello <break time="500ms"/> world</speak>',
            enable_ssml_parsing=True,
        )
        assert req.enable_ssml_parsing is True
        assert "<break" in req.input


# ---------------------------------------------------------------------------
# TTSSpeakCommand model — enable_ssml_parsing field
# ---------------------------------------------------------------------------


class TestTTSSpeakCommandSSML:
    def test_default_false(self) -> None:
        cmd = TTSSpeakCommand(text="Hello")
        assert cmd.enable_ssml_parsing is False

    def test_explicit_true(self) -> None:
        cmd = TTSSpeakCommand(text="Hello", enable_ssml_parsing=True)
        assert cmd.enable_ssml_parsing is True

    def test_from_json(self) -> None:
        data = {
            "type": "tts.speak",
            "text": '<speak>Hello <break time="200ms"/> world</speak>',
            "enable_ssml_parsing": True,
        }
        cmd = TTSSpeakCommand.model_validate(data)
        assert cmd.enable_ssml_parsing is True


# ---------------------------------------------------------------------------
# REST endpoint — SSML parsing wiring
# ---------------------------------------------------------------------------


class TestSpeechEndpointSSML:
    @pytest.fixture()
    def _mock_tts_deps(self) -> MagicMock:
        """Mock TTS dependencies for endpoint tests."""
        from macaw._types import ModelType

        registry = MagicMock()
        manifest = MagicMock()
        manifest.model_type = ModelType.TTS
        manifest.name = "kokoro-v1"
        registry.get_manifest.return_value = manifest
        registry.has_model.return_value = True
        return registry

    async def test_ssml_disabled_passes_text_unchanged(self) -> None:
        """When enable_ssml_parsing=false, SSML tags are NOT parsed."""
        body = SpeechRequest(
            model="kokoro-v1",
            input='<speak>Hello <break time="500ms"/> world</speak>',
            enable_ssml_parsing=False,
        )
        # The input text should remain unchanged
        assert "<speak>" in body.input
        assert "<break" in body.input

    async def test_ssml_enabled_strips_tags(self) -> None:
        """When enable_ssml_parsing=true, parser extracts directives."""
        from macaw.postprocessing.ssml.parser import SSMLParser

        body = SpeechRequest(
            model="kokoro-v1",
            input='<speak>Hello <break time="500ms"/> world</speak>',
            enable_ssml_parsing=True,
        )

        parser = SSMLParser()
        result = parser.parse(body.input)
        assert result.text == "Hello  world"
        assert len(result.directives) == 1

    async def test_ssml_invalid_raises_parse_error(self) -> None:
        """Invalid SSML input raises SSMLParseError."""
        from macaw.postprocessing.ssml.parser import SSMLParseError, SSMLParser

        parser = SSMLParser()
        with pytest.raises(SSMLParseError, match="Invalid SSML"):
            parser.parse("<speak><break</speak>")

    async def test_ssml_prosody_maps_to_speed(self) -> None:
        """SSML <prosody rate='slow'> maps to speed override."""
        from macaw.postprocessing.ssml.parser import SSMLParser
        from macaw.workers.tts.ssml_mapper import map_ssml_directives

        parser = SSMLParser()
        result = parser.parse('<speak><prosody rate="slow">Hello world</prosody></speak>')

        mapping = map_ssml_directives(
            result.text,
            result.directives,
            sample_rate=24000,
        )
        assert mapping.speed_override == 0.75
        assert mapping.text == "Hello world"


# ---------------------------------------------------------------------------
# WebSocket — SSML parsing wiring
# ---------------------------------------------------------------------------


class TestWSTTSSpeakSSML:
    async def test_ssml_command_parsed_correctly(self) -> None:
        """TTSSpeakCommand with SSML is parsed into directives."""
        from macaw.postprocessing.ssml.parser import SSMLParser
        from macaw.workers.tts.ssml_mapper import map_ssml_directives

        cmd = TTSSpeakCommand(
            text='<speak>Hello <break time="200ms"/> world</speak>',
            enable_ssml_parsing=True,
        )

        parser = SSMLParser()
        result = parser.parse(cmd.text)
        assert result.text == "Hello  world"

        mapping = map_ssml_directives(result.text, result.directives)
        assert len(mapping.silence_insertions) == 1

    async def test_ssml_speed_override_from_prosody(self) -> None:
        """SSML prosody rate overrides cmd.speed in WS context."""
        from macaw.postprocessing.ssml.parser import SSMLParser
        from macaw.workers.tts.ssml_mapper import map_ssml_directives

        cmd = TTSSpeakCommand(
            text='<speak><prosody rate="fast">Hello</prosody></speak>',
            speed=1.0,
            enable_ssml_parsing=True,
        )

        parser = SSMLParser()
        result = parser.parse(cmd.text)
        mapping = map_ssml_directives(result.text, result.directives)

        effective_speed = (
            mapping.speed_override if mapping.speed_override is not None else cmd.speed
        )
        assert effective_speed == 1.25

    async def test_ssml_disabled_no_parsing(self) -> None:
        """When enable_ssml_parsing=false, text is used as-is."""
        cmd = TTSSpeakCommand(
            text="<speak>Hello</speak>",
            enable_ssml_parsing=False,
        )
        # Without parsing, the full SSML string would be sent to TTS
        assert cmd.text == "<speak>Hello</speak>"
        assert cmd.enable_ssml_parsing is False


# ---------------------------------------------------------------------------
# Integration: SSML parse + map pipeline
# ---------------------------------------------------------------------------


class TestSSMLPipeline:
    def test_full_pipeline_break_and_prosody(self) -> None:
        """Full pipeline: SSML → parse → map → speed + silence."""
        from macaw.postprocessing.ssml.parser import SSMLParser
        from macaw.workers.tts.ssml_mapper import map_ssml_directives

        ssml = '<speak><prosody rate="slow">Hello <break time="300ms"/>world</prosody></speak>'

        parser = SSMLParser()
        result = parser.parse(ssml)
        assert result.text == "Hello world"

        mapping = map_ssml_directives(result.text, result.directives, sample_rate=16000)
        assert mapping.speed_override == 0.75
        assert len(mapping.silence_insertions) == 1
        _, silence = mapping.silence_insertions[0]
        # 300ms at 16kHz = 4800 samples * 2 bytes
        assert len(silence) == 9600

    def test_full_pipeline_emphasis_and_say_as(self) -> None:
        """Full pipeline: emphasis + say-as directives."""
        from macaw.postprocessing.ssml.parser import SSMLParser
        from macaw.workers.tts.ssml_mapper import map_ssml_directives

        ssml = (
            "<speak>"
            '<emphasis level="strong">Pay attention</emphasis> '
            '<say-as interpret-as="cardinal">42</say-as>'
            "</speak>"
        )

        parser = SSMLParser()
        result = parser.parse(ssml)
        assert result.text == "Pay attention 42"

        mapping = map_ssml_directives(result.text, result.directives)
        assert mapping.speed_override == 0.8  # strong emphasis
        assert mapping.silence_insertions == []

    def test_full_pipeline_plain_text_no_change(self) -> None:
        """Plain text with enable_ssml_parsing=true works fine."""
        from macaw.postprocessing.ssml.parser import SSMLParser
        from macaw.workers.tts.ssml_mapper import map_ssml_directives

        parser = SSMLParser()
        result = parser.parse("Hello world, this is plain text.")

        mapping = map_ssml_directives(result.text, result.directives)
        assert mapping.text == "Hello world, this is plain text."
        assert mapping.speed_override is None
        assert mapping.silence_insertions == []
