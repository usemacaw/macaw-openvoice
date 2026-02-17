"""Tests for post_load_hook() lifecycle method in STT and TTS backend ABCs."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any
from unittest.mock import AsyncMock, patch

import pytest

from macaw._types import (
    BatchResult,
    EngineCapabilities,
    STTArchitecture,
    TTSEngineCapabilities,
)
from macaw.workers.stt.interface import STTBackend
from macaw.workers.tts.interface import TTSBackend

if TYPE_CHECKING:
    from collections.abc import AsyncIterator


# ---------------------------------------------------------------------------
# Concrete stubs for testing (ABCs cannot be instantiated directly)
# ---------------------------------------------------------------------------


class _StubTTSBackend(TTSBackend):
    """Minimal concrete TTSBackend for testing the default hook."""

    async def load(self, model_path: str, config: dict[str, object]) -> None:
        pass

    async def capabilities(self) -> TTSEngineCapabilities:
        return TTSEngineCapabilities()

    async def synthesize(  # type: ignore[override]
        self,
        text: str,
        voice: str = "default",
        *,
        sample_rate: int = 24000,
        speed: float = 1.0,
        options: dict[str, object] | None = None,
    ) -> AsyncIterator[bytes]:
        yield b""

    async def voices(self) -> list[Any]:
        return []

    async def unload(self) -> None:
        pass

    async def health(self) -> dict[str, str]:
        return {"status": "ok"}


class _StubSTTBackend(STTBackend):
    """Minimal concrete STTBackend for testing the default hook."""

    @property
    def architecture(self) -> STTArchitecture:
        return STTArchitecture.ENCODER_DECODER

    async def load(self, model_path: str, config: dict[str, object]) -> None:
        pass

    async def capabilities(self) -> EngineCapabilities:
        return EngineCapabilities()

    async def transcribe_file(
        self,
        audio_data: bytes,
        language: str | None = None,
        initial_prompt: str | None = None,
        hot_words: list[str] | None = None,
        temperature: float = 0.0,
        word_timestamps: bool = False,
    ) -> BatchResult:
        return BatchResult(text="", language="en", duration=0.0, segments=())

    async def transcribe_stream(  # type: ignore[override]
        self,
        audio_chunks: AsyncIterator[bytes],
        language: str | None = None,
        initial_prompt: str | None = None,
        hot_words: list[str] | None = None,
    ) -> AsyncIterator[Any]:
        return  # empty generator
        yield  # pragma: no cover — makes this an async generator

    async def unload(self) -> None:
        pass

    async def health(self) -> dict[str, str]:
        return {"status": "ok"}


# ---------------------------------------------------------------------------
# Custom hook stubs
# ---------------------------------------------------------------------------


class _TTSWithCustomHook(_StubTTSBackend):
    """TTS backend that overrides post_load_hook for testing."""

    def __init__(self) -> None:
        self.hook_called = False

    async def post_load_hook(self) -> None:
        self.hook_called = True


class _STTWithCustomHook(_StubSTTBackend):
    """STT backend that overrides post_load_hook for testing."""

    def __init__(self) -> None:
        self.hook_called = False

    async def post_load_hook(self) -> None:
        self.hook_called = True


class _TTSWithFailingHook(_StubTTSBackend):
    """TTS backend whose post_load_hook raises."""

    async def post_load_hook(self) -> None:
        msg = "vocoder failed to load"
        raise RuntimeError(msg)


class _STTWithFailingHook(_StubSTTBackend):
    """STT backend whose post_load_hook raises."""

    async def post_load_hook(self) -> None:
        msg = "punctuation model failed to load"
        raise RuntimeError(msg)


# ===================================================================
# 1. Default no-op behaviour
# ===================================================================


async def test_tts_backend_post_load_hook_is_noop_by_default() -> None:
    """TTSBackend.post_load_hook() completes without side effects."""
    backend = _StubTTSBackend()

    await backend.post_load_hook()  # should not raise


async def test_stt_backend_post_load_hook_is_noop_by_default() -> None:
    """STTBackend.post_load_hook() completes without side effects."""
    backend = _StubSTTBackend()

    await backend.post_load_hook()  # should not raise


# ===================================================================
# 2. Custom hook is called
# ===================================================================


async def test_tts_custom_hook_is_called() -> None:
    """A TTS backend that overrides post_load_hook has it invoked."""
    backend = _TTSWithCustomHook()
    assert not backend.hook_called

    await backend.post_load_hook()

    assert backend.hook_called


async def test_stt_custom_hook_is_called() -> None:
    """An STT backend that overrides post_load_hook has it invoked."""
    backend = _STTWithCustomHook()
    assert not backend.hook_called

    await backend.post_load_hook()

    assert backend.hook_called


# ===================================================================
# 3. Hook failure propagates
# ===================================================================


async def test_tts_hook_failure_propagates() -> None:
    """If post_load_hook raises, the exception propagates to the caller."""
    backend = _TTSWithFailingHook()

    with pytest.raises(RuntimeError, match="vocoder failed to load"):
        await backend.post_load_hook()


async def test_stt_hook_failure_propagates() -> None:
    """If post_load_hook raises, the exception propagates to the caller."""
    backend = _STTWithFailingHook()

    with pytest.raises(RuntimeError, match="punctuation model failed to load"):
        await backend.post_load_hook()


# ===================================================================
# 4. Correct call sequence: load -> hook -> warmup
# ===================================================================


async def test_tts_serve_calls_hook_between_load_and_warmup() -> None:
    """TTS serve() calls post_load_hook after load and before warmup.

    The serve() function is aborted after warmup via a side_effect
    exception to avoid signal handler registration complexity in tests.
    """
    call_order: list[str] = []

    backend = AsyncMock(spec=_StubTTSBackend)
    backend.load = AsyncMock(side_effect=lambda *a, **kw: call_order.append("load"))
    backend.post_load_hook = AsyncMock(side_effect=lambda: call_order.append("post_load_hook"))

    class _AbortAfterWarmupError(Exception):
        pass

    async def _fake_warmup(*args: object, **kwargs: object) -> None:
        call_order.append("warmup")
        raise _AbortAfterWarmupError

    with (
        patch("macaw.workers.tts.main._create_backend", return_value=backend),
        patch("macaw.workers.tts.main.configure_torch_inference"),
        patch("macaw.workers.tts.main._warmup_backend", side_effect=_fake_warmup),
    ):
        from macaw.workers.tts.main import serve

        with pytest.raises(_AbortAfterWarmupError):
            await serve(port=50052, engine="kokoro", model_path="/tmp/model", engine_config={})

    assert call_order == ["load", "post_load_hook", "warmup"]


async def test_stt_serve_calls_hook_between_load_and_warmup() -> None:
    """STT serve() calls post_load_hook after load and before warmup."""
    call_order: list[str] = []

    backend = AsyncMock(spec=_StubSTTBackend)
    backend.load = AsyncMock(side_effect=lambda *a, **kw: call_order.append("load"))
    backend.post_load_hook = AsyncMock(side_effect=lambda: call_order.append("post_load_hook"))

    class _AbortAfterWarmupError(Exception):
        pass

    async def _fake_warmup(*args: object, **kwargs: object) -> None:
        call_order.append("warmup")
        raise _AbortAfterWarmupError

    with (
        patch("macaw.workers.stt.main._create_backend", return_value=backend),
        patch("macaw.workers.stt.main.configure_torch_inference"),
        patch("macaw.workers.stt.main._warmup_backend", side_effect=_fake_warmup),
    ):
        from macaw.workers.stt.main import serve

        with pytest.raises(_AbortAfterWarmupError):
            await serve(
                port=50051, engine="faster-whisper", model_path="/tmp/model", engine_config={}
            )

    assert call_order == ["load", "post_load_hook", "warmup"]


# ===================================================================
# 5. Hook is not abstract — existing engines work without override
# ===================================================================


async def test_tts_backend_hook_not_abstract() -> None:
    """post_load_hook is not abstract — instantiation does not require override."""
    # If it were abstract, _StubTTSBackend would fail to instantiate
    # because it does not override post_load_hook.
    backend = _StubTTSBackend()
    assert hasattr(backend, "post_load_hook")
    assert callable(backend.post_load_hook)


async def test_stt_backend_hook_not_abstract() -> None:
    """post_load_hook is not abstract — instantiation does not require override."""
    backend = _StubSTTBackend()
    assert hasattr(backend, "post_load_hook")
    assert callable(backend.post_load_hook)
