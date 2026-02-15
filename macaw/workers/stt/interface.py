"""Abstract interface for STT backends.

Every STT backend (Faster-Whisper, Paraformer, etc.) must implement
this interface to plug into the Macaw runtime.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from collections.abc import AsyncIterator

    from macaw._types import (
        BatchResult,
        EngineCapabilities,
        STTArchitecture,
        TranscriptSegment,
    )


class STTBackend(ABC):
    """Contract that every STT engine must implement.

    The runtime interacts with engines exclusively through this interface.
    Adding a new engine requires:
    1. Implement STTBackend
    2. Create a macaw.yaml manifest
    3. Register it in the Model Registry
    Zero changes to the runtime core.
    """

    @property
    @abstractmethod
    def architecture(self) -> STTArchitecture:
        """Model architecture (encoder-decoder, ctc, streaming-native).

        Determines how the runtime adapts the streaming pipeline.
        """
        ...

    @abstractmethod
    async def load(self, model_path: str, config: dict[str, object]) -> None:
        """Load the model into memory.

        Args:
            model_path: Path to the model files.
            config: engine_config from the macaw.yaml manifest.

        Raises:
            ModelLoadError: If the model cannot be loaded.
        """
        ...

    @abstractmethod
    async def capabilities(self) -> EngineCapabilities:
        """Runtime capabilities reported by the engine.

        May differ from the manifest if the engine discovers additional
        capabilities after load.
        """
        ...

    @abstractmethod
    async def transcribe_file(
        self,
        audio_data: bytes,
        language: str | None = None,
        initial_prompt: str | None = None,
        hot_words: list[str] | None = None,
        temperature: float = 0.0,
        word_timestamps: bool = False,
    ) -> BatchResult:
        """Transcribe full audio (batch).

        Args:
            audio_data: 16-bit PCM audio, 16kHz, mono (already preprocessed).
            language: ISO 639-1 code, "auto", or "mixed".
            initial_prompt: Context to guide transcription.
            hot_words: Words for keyword boosting.
            temperature: Sampling temperature (0.0-1.0).
            word_timestamps: If True, return per-word timestamps.

        Returns:
            BatchResult with text, language, duration, and segments.
        """
        ...

    @abstractmethod
    async def transcribe_stream(
        self,
        audio_chunks: AsyncIterator[bytes],
        language: str | None = None,
        initial_prompt: str | None = None,
        hot_words: list[str] | None = None,
    ) -> AsyncIterator[TranscriptSegment]:
        """Transcribe audio in streaming mode.

        Receives chunks of 16-bit PCM audio, 16kHz, mono (already preprocessed
        by the runtime Audio Preprocessing Pipeline).

        For encoder-decoder: runtime uses LocalAgreement for partials.
        For CTC: engine produces native partials frame-by-frame.
        For streaming-native: engine manages internal state.

        Args:
            audio_chunks: Async iterator of PCM chunks.
            language: ISO 639-1 code, "auto", or "mixed".
            initial_prompt: Context from the previous segment.
            hot_words: Words for keyword boosting.

        Yields:
            TranscriptSegment (partial and final).
        """
        ...

    @abstractmethod
    async def unload(self) -> None:
        """Unload the model from memory.

        Frees resources (GPU memory, buffers). After unload, load() must
        be called again before transcribe.
        """
        ...

    @abstractmethod
    async def health(self) -> dict[str, str]:
        """Backend status.

        Returns:
            Dict with at least {"status": "ok"|"loading"|"error"}.
        """
        ...
