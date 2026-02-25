"""Tests for the STTBackend contract.

Verifies that the interface is correctly defined and that
concrete implementations are forced to implement all methods.
"""

from collections.abc import AsyncIterator

import pytest

from macaw._types import (
    BatchResult,
    EngineCapabilities,
    STTArchitecture,
    TranscriptSegment,
)
from macaw.workers.stt.interface import STTBackend


class IncompleteBackend(STTBackend):  # type: ignore[abstract]
    """Backend that does not implement any abstract methods."""


class MinimalBackend(STTBackend):
    """Backend with minimal implementation to validate the contract."""

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
        return BatchResult(text="", language="pt", duration=0.0, segments=())

    async def transcribe_stream(
        self,
        audio_chunks: AsyncIterator[bytes],
        language: str | None = None,
        initial_prompt: str | None = None,
        hot_words: list[str] | None = None,
    ) -> AsyncIterator[TranscriptSegment]:
        return
        yield

    async def unload(self) -> None:
        pass

    async def health(self) -> dict[str, str]:
        return {"status": "ok"}


class TestSTTBackendContract:
    def test_cannot_instantiate_incomplete(self) -> None:
        with pytest.raises(TypeError):
            IncompleteBackend()

    def test_can_instantiate_complete(self) -> None:
        backend = MinimalBackend()
        assert backend.architecture == STTArchitecture.ENCODER_DECODER

    async def test_health_returns_dict(self) -> None:
        backend = MinimalBackend()
        result = await backend.health()
        assert "status" in result

    async def test_capabilities_returns_engine_capabilities(self) -> None:
        backend = MinimalBackend()
        caps = await backend.capabilities()
        assert isinstance(caps, EngineCapabilities)

    async def test_transcribe_file_returns_batch_result(self) -> None:
        backend = MinimalBackend()
        result = await backend.transcribe_file(audio_data=b"fake_audio")
        assert isinstance(result, BatchResult)

    async def test_load_accepts_config(self) -> None:
        backend = MinimalBackend()
        await backend.load("/path/to/model", {"beam_size": 5})

    async def test_unload_succeeds(self) -> None:
        backend = MinimalBackend()
        await backend.unload()
