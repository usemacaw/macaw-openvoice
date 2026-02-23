"""Tests for engine-specific voice_settings mapping."""

from __future__ import annotations

import pytest

from macaw.workers.tts.chatterbox import ChatterboxTurboBackend
from macaw.workers.tts.interface import TTSBackend
from macaw.workers.tts.kokoro import KokoroBackend
from macaw.workers.tts.qwen3 import Qwen3TTSBackend


class TestKokoroMapping:
    """KokoroBackend.map_voice_settings()."""

    def test_maps_speed(self) -> None:
        backend = KokoroBackend()
        result = backend.map_voice_settings({"speed": 1.5})
        assert result == {"speed": 1.5}

    def test_default_speed_returns_empty(self) -> None:
        backend = KokoroBackend()
        result = backend.map_voice_settings({"speed": 1.0})
        assert result == {}

    def test_ignores_stability(self) -> None:
        backend = KokoroBackend()
        result = backend.map_voice_settings({"stability": 0.3, "speed": 1.0})
        # stability is ignored, speed is default -> empty result
        assert result == {}

    def test_empty_settings_returns_empty(self) -> None:
        backend = KokoroBackend()
        result = backend.map_voice_settings({})
        assert result == {}

    @pytest.mark.asyncio
    async def test_capabilities_supports_voice_settings(self) -> None:
        backend = KokoroBackend()
        caps = await backend.capabilities()
        assert caps.supports_voice_settings is True


class TestQwen3Mapping:
    """Qwen3TTSBackend.map_voice_settings()."""

    def test_maps_stability_to_temperature_inversion(self) -> None:
        backend = Qwen3TTSBackend()
        result = backend.map_voice_settings({"stability": 0.5})
        assert result["temperature"] == pytest.approx(0.5)

    def test_stability_zero_gives_temperature_one(self) -> None:
        backend = Qwen3TTSBackend()
        result = backend.map_voice_settings({"stability": 0.0})
        assert result["temperature"] == pytest.approx(1.0)

    def test_stability_one_gives_temperature_zero(self) -> None:
        backend = Qwen3TTSBackend()
        result = backend.map_voice_settings({"stability": 1.0})
        assert result["temperature"] == pytest.approx(0.0)

    def test_maps_speed(self) -> None:
        backend = Qwen3TTSBackend()
        result = backend.map_voice_settings({"speed": 2.0})
        assert result["speed"] == 2.0

    def test_empty_settings_returns_empty(self) -> None:
        backend = Qwen3TTSBackend()
        result = backend.map_voice_settings({})
        assert result == {}

    @pytest.mark.asyncio
    async def test_capabilities_supports_voice_settings(self) -> None:
        backend = Qwen3TTSBackend()
        caps = await backend.capabilities()
        assert caps.supports_voice_settings is True


class TestChatterboxMapping:
    """ChatterboxTurboBackend.map_voice_settings()."""

    def test_maps_speed(self) -> None:
        backend = ChatterboxTurboBackend()
        result = backend.map_voice_settings({"speed": 0.5})
        assert result == {"speed": 0.5}

    def test_default_speed_returns_empty(self) -> None:
        backend = ChatterboxTurboBackend()
        result = backend.map_voice_settings({"speed": 1.0})
        assert result == {}

    def test_empty_settings_returns_empty(self) -> None:
        backend = ChatterboxTurboBackend()
        result = backend.map_voice_settings({})
        assert result == {}

    @pytest.mark.asyncio
    async def test_capabilities_supports_voice_settings(self) -> None:
        backend = ChatterboxTurboBackend()
        caps = await backend.capabilities()
        assert caps.supports_voice_settings is True


class TestTTSBackendDefault:
    """TTSBackend default map_voice_settings returns empty dict."""

    def test_default_returns_empty(self) -> None:
        # TTSBackend is abstract, but map_voice_settings is concrete.
        # Test via a concrete subclass (KokoroBackend inherits the default
        # but overrides it; use TTSBackend directly via __dict__ check).
        assert TTSBackend.map_voice_settings(TTSBackend, {}) == {}  # type: ignore[arg-type]
