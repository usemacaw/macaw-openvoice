"""Tests for TTS types, TTSBackend interface, and TTS exceptions.

M9-01: Validates that TTS types are frozen, interface is not instantiable,
and exception hierarchy is correct.
"""

from __future__ import annotations

import dataclasses
from typing import TYPE_CHECKING

import pytest

from macaw._types import TTSEngineCapabilities, VoiceInfo
from macaw.exceptions import MacawError, TTSEngineError, TTSError, TTSSynthesisError
from macaw.workers.tts.interface import TTSBackend

if TYPE_CHECKING:
    from collections.abc import AsyncIterator


# --- VoiceInfo ---


class TestVoiceInfo:
    def test_creation(self) -> None:
        voice = VoiceInfo(voice_id="v1", name="Default", language="en")
        assert voice.voice_id == "v1"
        assert voice.name == "Default"
        assert voice.language == "en"
        assert voice.gender is None

    def test_creation_with_gender(self) -> None:
        voice = VoiceInfo(voice_id="v1", name="Sara", language="pt", gender="female")
        assert voice.gender == "female"

    def test_frozen(self) -> None:
        voice = VoiceInfo(voice_id="v1", name="Default", language="en")
        with pytest.raises(dataclasses.FrozenInstanceError):
            voice.name = "other"  # type: ignore[misc]

    def test_slots(self) -> None:
        voice = VoiceInfo(voice_id="v1", name="Default", language="en")
        assert not hasattr(voice, "__dict__")

    def test_replace(self) -> None:
        voice = VoiceInfo(voice_id="v1", name="Default", language="en")
        updated = dataclasses.replace(voice, language="pt")
        assert updated.language == "pt"
        assert voice.language == "en"


# --- TTSEngineCapabilities ---


class TestTTSEngineCapabilities:
    def test_defaults(self) -> None:
        caps = TTSEngineCapabilities()
        assert caps.supports_streaming is False
        assert caps.supports_voice_cloning is False
        assert caps.supports_instruct is False
        assert caps.max_text_length is None

    def test_custom_values(self) -> None:
        caps = TTSEngineCapabilities(
            supports_streaming=True,
            supports_voice_cloning=True,
            supports_instruct=True,
            max_text_length=500,
        )
        assert caps.supports_streaming is True
        assert caps.supports_voice_cloning is True
        assert caps.supports_instruct is True
        assert caps.max_text_length == 500

    def test_frozen(self) -> None:
        caps = TTSEngineCapabilities()
        with pytest.raises(dataclasses.FrozenInstanceError):
            caps.supports_streaming = True  # type: ignore[misc]

    def test_slots(self) -> None:
        caps = TTSEngineCapabilities()
        assert not hasattr(caps, "__dict__")

    def test_replace(self) -> None:
        caps = TTSEngineCapabilities()
        updated = dataclasses.replace(caps, supports_streaming=True, max_text_length=1000)
        assert updated.supports_streaming is True
        assert updated.max_text_length == 1000
        assert caps.supports_streaming is False


# --- TTSError / TTSSynthesisError / TTSEngineError ---


class TestTTSExceptions:
    def test_tts_error_is_macaw_error(self) -> None:
        assert issubclass(TTSError, MacawError)

    def test_tts_synthesis_error_is_tts_error(self) -> None:
        assert issubclass(TTSSynthesisError, TTSError)

    def test_tts_synthesis_error_attributes(self) -> None:
        err = TTSSynthesisError(model_name="kokoro-v1", reason="out of memory")
        assert err.model_name == "kokoro-v1"
        assert err.reason == "out of memory"
        assert "kokoro-v1" in str(err)
        assert "out of memory" in str(err)

    def test_tts_error_catchable_as_macaw_error(self) -> None:
        with pytest.raises(MacawError):
            raise TTSSynthesisError("model", "reason")


class TestTTSEngineError:
    def test_tts_engine_error_is_tts_error(self) -> None:
        assert issubclass(TTSEngineError, TTSError)

    def test_tts_engine_error_is_not_synthesis_error(self) -> None:
        assert not issubclass(TTSEngineError, TTSSynthesisError)

    def test_tts_engine_error_attributes(self) -> None:
        err = TTSEngineError(model_name="kokoro-v1", reason="GPU OOM")
        assert err.model_name == "kokoro-v1"
        assert err.reason == "GPU OOM"
        assert "kokoro-v1" in str(err)
        assert "GPU OOM" in str(err)

    def test_tts_engine_error_catchable_as_macaw_error(self) -> None:
        with pytest.raises(MacawError):
            raise TTSEngineError("model", "reason")


# --- TTSBackend ABC ---


class TestTTSBackendABC:
    def test_cannot_instantiate_directly(self) -> None:
        with pytest.raises(TypeError, match="abstract"):
            TTSBackend()  # type: ignore[abstract]

    def test_concrete_subclass_can_be_created(self) -> None:
        """A complete concrete implementation can be instantiated."""

        class _ConcreteTTS(TTSBackend):
            async def load(self, model_path: str, config: dict[str, object]) -> None: ...

            async def capabilities(self) -> TTSEngineCapabilities:
                return TTSEngineCapabilities()

            async def synthesize(
                self,
                text: str,
                voice: str = "default",
                *,
                sample_rate: int = 24000,
                speed: float = 1.0,
            ) -> AsyncIterator[bytes]:
                yield b""  # pragma: no cover

            async def voices(self) -> list[VoiceInfo]:
                return []

            async def unload(self) -> None: ...

            async def health(self) -> dict[str, str]:
                return {"status": "ok"}

        instance = _ConcreteTTS()
        assert instance is not None

    def test_partial_implementation_raises(self) -> None:
        """Missing a single method should prevent instantiation."""

        class _IncompleteTTS(TTSBackend):
            async def load(self, model_path: str, config: dict[str, object]) -> None: ...

            # Missing: synthesize, voices, unload, health

        with pytest.raises(TypeError, match="abstract"):
            _IncompleteTTS()  # type: ignore[abstract]

    def test_has_six_abstract_methods(self) -> None:
        abstract_methods = TTSBackend.__abstractmethods__
        assert len(abstract_methods) == 6
        expected = {"load", "capabilities", "synthesize", "voices", "unload", "health"}
        assert abstract_methods == expected
