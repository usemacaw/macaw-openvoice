"""Tests for VoiceSettings Pydantic model validation."""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from macaw.server.models.voice_settings import VoiceSettings


class TestVoiceSettingsDefaults:
    """Verify default values match the ElevenLabs schema."""

    def test_defaults_are_valid(self) -> None:
        settings = VoiceSettings()
        assert settings.stability == 0.5
        assert settings.similarity_boost == 0.75
        assert settings.style == 0.0
        assert settings.use_speaker_boost is True
        assert settings.speed == 1.0

    def test_all_fields_present_in_model_dump(self) -> None:
        settings = VoiceSettings()
        dumped = settings.model_dump()
        expected_keys = {"stability", "similarity_boost", "style", "use_speaker_boost", "speed"}
        assert set(dumped.keys()) == expected_keys

    def test_use_speaker_boost_defaults_true(self) -> None:
        settings = VoiceSettings()
        assert settings.use_speaker_boost is True


class TestVoiceSettingsStabilityValidation:
    """Stability must be in [0.0, 1.0]."""

    def test_stability_above_one_rejected(self) -> None:
        with pytest.raises(ValidationError, match="stability"):
            VoiceSettings(stability=1.1)

    def test_stability_below_zero_rejected(self) -> None:
        with pytest.raises(ValidationError, match="stability"):
            VoiceSettings(stability=-0.1)

    def test_stability_boundary_values_accepted(self) -> None:
        low = VoiceSettings(stability=0.0)
        high = VoiceSettings(stability=1.0)
        assert low.stability == 0.0
        assert high.stability == 1.0


class TestVoiceSettingsSimilarityBoostValidation:
    """similarity_boost must be in [0.0, 1.0]."""

    def test_similarity_boost_above_one_rejected(self) -> None:
        with pytest.raises(ValidationError, match="similarity_boost"):
            VoiceSettings(similarity_boost=1.5)

    def test_similarity_boost_below_zero_rejected(self) -> None:
        with pytest.raises(ValidationError, match="similarity_boost"):
            VoiceSettings(similarity_boost=-0.01)


class TestVoiceSettingsStyleValidation:
    """style must be in [0.0, 1.0]."""

    def test_style_above_one_rejected(self) -> None:
        with pytest.raises(ValidationError, match="style"):
            VoiceSettings(style=2.0)

    def test_style_below_zero_rejected(self) -> None:
        with pytest.raises(ValidationError, match="style"):
            VoiceSettings(style=-1.0)


class TestVoiceSettingsSpeedValidation:
    """speed must be in [0.25, 4.0] — matches SpeechRequest range."""

    def test_speed_above_max_rejected(self) -> None:
        with pytest.raises(ValidationError, match="speed"):
            VoiceSettings(speed=4.1)

    def test_speed_below_min_rejected(self) -> None:
        with pytest.raises(ValidationError, match="speed"):
            VoiceSettings(speed=0.2)

    def test_speed_boundary_values_accepted(self) -> None:
        low = VoiceSettings(speed=0.25)
        high = VoiceSettings(speed=4.0)
        assert low.speed == 0.25
        assert high.speed == 4.0
