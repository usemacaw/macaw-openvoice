"""Tests for sound effect backend interface and capabilities."""

from __future__ import annotations

from typing import TYPE_CHECKING

import pytest

from macaw._types import SoundEffectCapabilities
from macaw.workers.sound_effects.interface import SoundEffectBackend

if TYPE_CHECKING:
    from collections.abc import AsyncIterator


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


class _ConcreteSFX(SoundEffectBackend):
    """Minimal valid implementation for testing non-abstract methods."""

    async def load(self, model_path: str, config: dict[str, object]) -> None:
        pass

    async def generate(  # type: ignore[override]
        self,
        text: str,
        *,
        duration_s: float = 5.0,
        prompt_influence: float = 0.3,
        loop: bool = False,
        options: dict[str, object] | None = None,
    ) -> AsyncIterator[bytes]:
        yield b""

    async def capabilities(self) -> SoundEffectCapabilities:
        return SoundEffectCapabilities()


# ---------------------------------------------------------------------------
# SoundEffectCapabilities
# ---------------------------------------------------------------------------


class TestSoundEffectCapabilities:
    def test_defaults(self) -> None:
        caps = SoundEffectCapabilities()
        assert caps.max_duration_s == 30.0
        assert caps.min_duration_s == 0.5
        assert caps.supports_loop is False
        assert caps.supports_prompt_influence is False

    def test_custom_values(self) -> None:
        caps = SoundEffectCapabilities(
            max_duration_s=60.0,
            min_duration_s=1.0,
            supports_loop=True,
            supports_prompt_influence=True,
        )
        assert caps.max_duration_s == 60.0
        assert caps.min_duration_s == 1.0
        assert caps.supports_loop is True
        assert caps.supports_prompt_influence is True

    def test_frozen(self) -> None:
        caps = SoundEffectCapabilities()
        with pytest.raises(AttributeError):
            caps.supports_loop = True  # type: ignore[misc]


# ---------------------------------------------------------------------------
# SoundEffectBackend ABC
# ---------------------------------------------------------------------------


class TestSoundEffectBackendABC:
    def test_cannot_instantiate_directly(self) -> None:
        with pytest.raises(TypeError, match="abstract"):
            SoundEffectBackend()  # type: ignore[abstract]

    def test_subclass_must_implement_load(self) -> None:
        class _NoLoad(SoundEffectBackend):
            async def generate(  # type: ignore[override]
                self,
                text: str,
                **kwargs: object,
            ) -> AsyncIterator[bytes]:
                yield b""

            async def capabilities(self) -> SoundEffectCapabilities:
                return SoundEffectCapabilities()

        with pytest.raises(TypeError, match="abstract"):
            _NoLoad()  # type: ignore[abstract]

    def test_subclass_must_implement_generate(self) -> None:
        class _NoGenerate(SoundEffectBackend):
            async def load(self, model_path: str, config: dict[str, object]) -> None:
                pass

            async def capabilities(self) -> SoundEffectCapabilities:
                return SoundEffectCapabilities()

        with pytest.raises(TypeError, match="abstract"):
            _NoGenerate()  # type: ignore[abstract]

    def test_subclass_must_implement_capabilities(self) -> None:
        class _NoCaps(SoundEffectBackend):
            async def load(self, model_path: str, config: dict[str, object]) -> None:
                pass

            async def generate(  # type: ignore[override]
                self,
                text: str,
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
        backend = _ConcreteSFX()
        await backend.post_load_hook()


# ---------------------------------------------------------------------------
# Engine loader compatibility
# ---------------------------------------------------------------------------


class TestEngineLoaderCompatibility:
    def test_external_loader_finds_sfx_backend(self) -> None:
        """load_external_backend can discover SoundEffectBackend subclasses."""
        import sys
        import types

        from macaw.workers._engine_loader import load_external_backend

        fake_module = types.ModuleType("fake_sfx_engine")

        class FakeSFXEngine(SoundEffectBackend):
            async def load(self, model_path: str, config: dict[str, object]) -> None:
                pass

            async def generate(  # type: ignore[override]
                self,
                text: str,
                **kwargs: object,
            ) -> AsyncIterator[bytes]:
                yield b""

            async def capabilities(self) -> SoundEffectCapabilities:
                return SoundEffectCapabilities()

        fake_module.FakeSFXEngine = FakeSFXEngine  # type: ignore[attr-defined]

        sys.modules["fake_sfx_engine"] = fake_module
        try:
            backend = load_external_backend(
                "fake_sfx_engine",
                SoundEffectBackend,  # type: ignore[type-abstract]
            )
            assert isinstance(backend, FakeSFXEngine)
        finally:
            del sys.modules["fake_sfx_engine"]
