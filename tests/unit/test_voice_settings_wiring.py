"""Tests for voice_settings wiring through API, proto, and servicer."""

from __future__ import annotations

import json
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from macaw._types import TTSEngineCapabilities
from macaw.scheduler.tts_converters import build_tts_proto_request
from macaw.server.models.events import TTSSpeakCommand
from macaw.server.models.speech import SpeechRequest
from macaw.server.models.voice_settings import VoiceSettings
from macaw.workers.tts.converters import SynthesizeParams, proto_request_to_synthesize_params
from macaw.workers.tts.servicer import TTSWorkerServicer


class TestSpeechRequestVoiceSettings:
    """SpeechRequest accepts voice_settings."""

    def test_voice_settings_accepted(self) -> None:
        req = SpeechRequest(
            model="test-model",
            input="Hello world",
            voice_settings=VoiceSettings(stability=0.8, speed=1.5),
        )
        assert req.voice_settings is not None
        assert req.voice_settings.stability == 0.8
        assert req.voice_settings.speed == 1.5

    def test_voice_settings_defaults_to_none(self) -> None:
        req = SpeechRequest(model="test-model", input="Hello world")
        assert req.voice_settings is None

    def test_voice_settings_invalid_values_rejected(self) -> None:
        from pydantic import ValidationError

        with pytest.raises(ValidationError, match="stability"):
            SpeechRequest(
                model="test-model",
                input="Hello world",
                voice_settings=VoiceSettings(stability=2.0),  # out of range
            )


class TestTTSSpeakCommandVoiceSettings:
    """TTSSpeakCommand accepts voice_settings dict."""

    def test_voice_settings_dict_accepted(self) -> None:
        cmd = TTSSpeakCommand(
            text="Hello",
            voice_settings={"stability": 0.3, "speed": 2.0},
        )
        assert cmd.voice_settings is not None
        assert cmd.voice_settings["stability"] == 0.3

    def test_voice_settings_defaults_to_none(self) -> None:
        cmd = TTSSpeakCommand(text="Hello")
        assert cmd.voice_settings is None


class TestBuildTtsProtoRequestVoiceSettings:
    """build_tts_proto_request packs voice_settings into proto."""

    def test_voice_settings_packed_into_proto(self) -> None:
        vs = {"stability": 0.5, "speed": 1.5}
        req = build_tts_proto_request(
            request_id="test-1",
            text="Hello",
            voice="default",
            sample_rate=24000,
            speed=1.0,
            voice_settings=vs,
        )
        assert req.voice_settings_json != ""
        parsed = json.loads(req.voice_settings_json)
        assert parsed["stability"] == 0.5
        assert parsed["speed"] == 1.5

    def test_no_voice_settings_leaves_proto_empty(self) -> None:
        req = build_tts_proto_request(
            request_id="test-2",
            text="Hello",
            voice="default",
            sample_rate=24000,
            speed=1.0,
        )
        assert req.voice_settings_json == ""


class TestProtoToSynthesizeParamsVoiceSettings:
    """proto_request_to_synthesize_params extracts voice_settings."""

    def test_voice_settings_extracted_to_options(self) -> None:
        vs = {"stability": 0.5, "speed": 1.5}
        proto_req = build_tts_proto_request(
            request_id="test-3",
            text="Hello",
            voice="default",
            sample_rate=24000,
            speed=1.0,
            voice_settings=vs,
        )
        params = proto_request_to_synthesize_params(proto_req)
        assert params.options is not None
        assert "voice_settings" in params.options
        assert params.options["voice_settings"]["stability"] == 0.5  # type: ignore[index]


class TestServicerVoiceSettings:
    """Servicer extracts voice_settings and calls map_voice_settings."""

    def _make_servicer(
        self,
        supports_voice_settings: bool = True,
    ) -> tuple[TTSWorkerServicer, MagicMock]:
        backend = MagicMock()
        backend.capabilities = AsyncMock(
            return_value=TTSEngineCapabilities(
                supports_voice_settings=supports_voice_settings,
            )
        )
        backend.map_voice_settings = MagicMock(return_value={"speed": 2.0})
        servicer = TTSWorkerServicer(backend, "test-model", "test-engine")
        return servicer, backend

    def test_apply_voice_settings_calls_map(self) -> None:
        servicer, backend = self._make_servicer(supports_voice_settings=True)
        caps = TTSEngineCapabilities(supports_voice_settings=True)
        params = SynthesizeParams(
            text="Hello",
            voice="default",
            sample_rate=24000,
            speed=1.0,
            options={"voice_settings": {"stability": 0.3}},
        )
        result = servicer._apply_voice_settings(params, caps, "req-1")
        backend.map_voice_settings.assert_called_once_with({"stability": 0.3})
        assert result.speed == 2.0

    def test_apply_voice_settings_logs_warning_when_not_supported(self) -> None:
        servicer, backend = self._make_servicer(supports_voice_settings=False)
        caps = TTSEngineCapabilities(supports_voice_settings=False)
        params = SynthesizeParams(
            text="Hello",
            voice="default",
            sample_rate=24000,
            speed=1.0,
            options={"voice_settings": {"stability": 0.3}},
        )
        with patch("macaw.workers.tts.servicer.logger") as mock_logger:
            result = servicer._apply_voice_settings(params, caps, "req-1")
            mock_logger.warning.assert_called_once()
        # Params unchanged
        assert result.speed == 1.0
        backend.map_voice_settings.assert_not_called()

    def test_apply_voice_settings_noop_without_voice_settings(self) -> None:
        servicer, backend = self._make_servicer()
        caps = TTSEngineCapabilities(supports_voice_settings=True)
        params = SynthesizeParams(
            text="Hello",
            voice="default",
            sample_rate=24000,
            speed=1.0,
        )
        result = servicer._apply_voice_settings(params, caps, "req-2")
        assert result is params
        backend.map_voice_settings.assert_not_called()

    def test_apply_voice_settings_merges_temperature(self) -> None:
        servicer, backend = self._make_servicer()
        backend.map_voice_settings = MagicMock(return_value={"temperature": 0.7})
        caps = TTSEngineCapabilities(supports_voice_settings=True)
        params = SynthesizeParams(
            text="Hello",
            voice="default",
            sample_rate=24000,
            speed=1.0,
            options={"voice_settings": {"stability": 0.3}},
        )
        result = servicer._apply_voice_settings(params, caps, "req-3")
        assert result.options is not None
        assert result.options["temperature"] == 0.7
        # voice_settings key removed from options
        assert "voice_settings" not in result.options

    def test_backward_compat_no_voice_settings(self) -> None:
        """Request without voice_settings works unchanged."""
        req = build_tts_proto_request(
            request_id="compat-1",
            text="Hello",
            voice="default",
            sample_rate=24000,
            speed=1.0,
        )
        params = proto_request_to_synthesize_params(req)
        assert params.options is None

    def test_voice_settings_with_only_speed(self) -> None:
        vs = {"speed": 2.0}
        req = build_tts_proto_request(
            request_id="speed-only",
            text="Hello",
            voice="default",
            sample_rate=24000,
            speed=1.0,
            voice_settings=vs,
        )
        params = proto_request_to_synthesize_params(req)
        assert params.options is not None
        assert params.options["voice_settings"]["speed"] == 2.0  # type: ignore[index]

    def test_voice_settings_with_all_fields(self) -> None:
        vs = {
            "stability": 0.5,
            "similarity_boost": 0.75,
            "style": 0.0,
            "use_speaker_boost": True,
            "speed": 1.0,
        }
        req = build_tts_proto_request(
            request_id="all-fields",
            text="Hello",
            voice="default",
            sample_rate=24000,
            speed=1.0,
            voice_settings=vs,
        )
        params = proto_request_to_synthesize_params(req)
        assert params.options is not None
        parsed_vs = params.options["voice_settings"]
        assert isinstance(parsed_vs, dict)
        assert len(parsed_vs) == 5  # type: ignore[arg-type]
