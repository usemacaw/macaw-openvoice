"""Tests for voice changer backend interface and capabilities."""

from __future__ import annotations

from typing import TYPE_CHECKING

import pytest

from macaw._types import VoiceChangerCapabilities
from macaw.workers.voice_changer.interface import VoiceChangerBackend

if TYPE_CHECKING:
    from collections.abc import AsyncIterator


# ---------------------------------------------------------------------------
# Helpers — concrete stubs
# ---------------------------------------------------------------------------


class _ConcreteVC(VoiceChangerBackend):
    """Minimal valid implementation for testing non-abstract methods."""

    async def load(self, model_path: str, config: dict[str, object]) -> None:
        pass

    async def convert(  # type: ignore[override]
        self,
        source_audio: bytes,
        voice: str = "default",
        **kwargs: object,
    ) -> AsyncIterator[bytes]:
        yield b""

    async def capabilities(self) -> VoiceChangerCapabilities:
        return VoiceChangerCapabilities()


# ---------------------------------------------------------------------------
# VoiceChangerCapabilities
# ---------------------------------------------------------------------------


class TestVoiceChangerCapabilities:
    def test_defaults(self) -> None:
        caps = VoiceChangerCapabilities()
        assert caps.supports_streaming is False
        assert caps.supports_voice_reference is False
        assert caps.max_audio_duration_s == 300.0
        assert caps.supported_sample_rates == (16000,)

    def test_custom_values(self) -> None:
        caps = VoiceChangerCapabilities(
            supports_streaming=True,
            supports_voice_reference=True,
            max_audio_duration_s=600.0,
            supported_sample_rates=(16000, 24000, 48000),
        )
        assert caps.supports_streaming is True
        assert caps.supports_voice_reference is True
        assert caps.max_audio_duration_s == 600.0
        assert 48000 in caps.supported_sample_rates

    def test_frozen(self) -> None:
        caps = VoiceChangerCapabilities()
        with pytest.raises(AttributeError):
            caps.supports_streaming = True  # type: ignore[misc]


# ---------------------------------------------------------------------------
# VoiceChangerBackend ABC
# ---------------------------------------------------------------------------


class TestVoiceChangerBackendABC:
    def test_cannot_instantiate_directly(self) -> None:
        with pytest.raises(TypeError, match="abstract"):
            VoiceChangerBackend()  # type: ignore[abstract]

    def test_subclass_must_implement_load(self) -> None:
        class _NoLoad(VoiceChangerBackend):
            async def convert(  # type: ignore[override]
                self,
                source_audio: bytes,
                voice: str = "default",
                **kwargs: object,
            ) -> AsyncIterator[bytes]:
                yield b""

            async def capabilities(self) -> VoiceChangerCapabilities:
                return VoiceChangerCapabilities()

        with pytest.raises(TypeError, match="abstract"):
            _NoLoad()  # type: ignore[abstract]

    def test_subclass_must_implement_convert(self) -> None:
        class _NoConvert(VoiceChangerBackend):
            async def load(self, model_path: str, config: dict[str, object]) -> None:
                pass

            async def capabilities(self) -> VoiceChangerCapabilities:
                return VoiceChangerCapabilities()

        with pytest.raises(TypeError, match="abstract"):
            _NoConvert()  # type: ignore[abstract]

    def test_subclass_must_implement_capabilities(self) -> None:
        class _NoCaps(VoiceChangerBackend):
            async def load(self, model_path: str, config: dict[str, object]) -> None:
                pass

            async def convert(  # type: ignore[override]
                self,
                source_audio: bytes,
                voice: str = "default",
                **kwargs: object,
            ) -> AsyncIterator[bytes]:
                yield b""

        with pytest.raises(TypeError, match="abstract"):
            _NoCaps()  # type: ignore[abstract]


# ---------------------------------------------------------------------------
# post_load_hook default
# ---------------------------------------------------------------------------


class TestPostLoadHook:
    async def test_default_post_load_hook_is_noop(self) -> None:
        """Default post_load_hook does nothing (no-op)."""
        backend = _ConcreteVC()
        # Should not raise
        await backend.post_load_hook()


# ---------------------------------------------------------------------------
# Engine loader compatibility
# ---------------------------------------------------------------------------


class TestEngineLoaderCompatibility:
    def test_external_loader_finds_vc_backend(self) -> None:
        """load_external_backend can discover VoiceChangerBackend subclasses."""
        import sys
        import types

        from macaw.workers._engine_loader import load_external_backend

        fake_module = types.ModuleType("fake_vc_engine")

        class FakeVCEngine(VoiceChangerBackend):
            async def load(self, model_path: str, config: dict[str, object]) -> None:
                pass

            async def convert(  # type: ignore[override]
                self,
                source_audio: bytes,
                voice: str = "default",
                **kwargs: object,
            ) -> AsyncIterator[bytes]:
                yield b""

            async def capabilities(self) -> VoiceChangerCapabilities:
                return VoiceChangerCapabilities()

        fake_module.FakeVCEngine = FakeVCEngine  # type: ignore[attr-defined]

        sys.modules["fake_vc_engine"] = fake_module
        try:
            backend = load_external_backend(
                "fake_vc_engine",
                VoiceChangerBackend,  # type: ignore[type-abstract]
            )
            assert isinstance(backend, FakeVCEngine)
        finally:
            del sys.modules["fake_vc_engine"]
