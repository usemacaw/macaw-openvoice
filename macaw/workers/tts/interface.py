"""Abstract interface for TTS backends.

Every TTS backend (Kokoro, Qwen3-TTS, etc.) must implement this interface
to plug into the Macaw runtime.
"""

from __future__ import annotations

import inspect
from abc import ABC, abstractmethod
from typing import TYPE_CHECKING, Literal, TypeVar

from macaw._audio_constants import TTS_DEFAULT_SAMPLE_RATE

if TYPE_CHECKING:
    from collections.abc import AsyncIterator

    from macaw._types import TTSChunkResult, TTSEngineCapabilities, VoiceInfo

_T = TypeVar("_T")


async def _resolve_stream(result: object) -> AsyncIterator[_T]:
    """Resolve a backend method result that may be a coroutine or async iterator.

    TTSBackend methods declared as returning ``AsyncIterator`` may actually
    return an ``AsyncGenerator`` (when using ``yield``) or a coroutine
    wrapping one.  This helper normalises both cases into an
    ``AsyncIterator``.
    """
    if inspect.iscoroutine(result):
        return await result  # type: ignore[no-any-return]
    return result  # type: ignore[return-value]


class TTSBackend(ABC):
    """Contract that every TTS engine must implement.

    The runtime interacts with TTS engines exclusively through this interface.
    Adding a new engine requires:
    1. Implement TTSBackend
    2. Create a macaw.yaml manifest with type: tts
    3. Register in the _create_backend() factory
    Zero changes to the runtime core.

    The key difference from STTBackend is that synthesize() returns an
    AsyncIterator of audio chunks (streaming), enabling low TTFB TTS — the
    first chunk can be sent to the client before synthesis is complete.
    """

    @abstractmethod
    async def load(self, model_path: str, config: dict[str, object]) -> None:
        """Load the TTS model into memory.

        Args:
            model_path: Path to the model files.
            config: engine_config from the macaw.yaml manifest.

        Raises:
            ModelLoadError: If the model cannot be loaded.
        """
        ...

    @abstractmethod
    async def capabilities(self) -> TTSEngineCapabilities:
        """Return engine capabilities for runtime decision-making.

        Used by the runtime to determine whether the engine supports
        streaming, voice cloning, instruct mode, or has text length limits.
        """
        ...

    @abstractmethod
    async def synthesize(
        self,
        text: str,
        voice: str = "default",
        *,
        sample_rate: int = TTS_DEFAULT_SAMPLE_RATE,
        speed: float = 1.0,
        options: dict[str, object] | None = None,
    ) -> AsyncIterator[bytes]:
        """Synthesize text into audio (streaming PCM chunks).

        Returns 16-bit PCM audio chunks as the engine synthesizes.
        This enables low-TTFB streaming — the first chunk can be ready
        in <50ms for fast engines.

        Args:
            text: Text to synthesize.
            voice: Voice identifier (default: "default").
            sample_rate: Output audio sample rate.
            speed: Synthesis speed (0.25-4.0, default 1.0).
            options: Engine-specific options (e.g., language, ref_audio,
                ref_text, instruction for LLM-based TTS). Backends that
                do not use extended options should ignore this parameter.

        Yields:
            16-bit PCM audio chunks.

        Raises:
            TTSSynthesisError: If synthesis fails.
        """
        ...

    @abstractmethod
    async def voices(self) -> list[VoiceInfo]:
        """List available voices for the model.

        Returns:
            List of VoiceInfo with details for each voice.
        """
        ...

    async def synthesize_with_alignment(
        self,
        text: str,
        voice: str = "default",
        *,
        sample_rate: int = TTS_DEFAULT_SAMPLE_RATE,
        speed: float = 1.0,
        alignment_granularity: Literal["word", "character"] = "word",
        options: dict[str, object] | None = None,
    ) -> AsyncIterator[TTSChunkResult]:
        """Synthesize text with per-chunk alignment data.

        Default: wraps synthesize() with no alignment. Engines with native
        alignment (e.g., Kokoro) override this to include timing data.

        Args:
            text: Text to synthesize.
            voice: Voice identifier (default: "default").
            sample_rate: Output audio sample rate.
            speed: Synthesis speed (0.25-4.0, default 1.0).
            alignment_granularity: "word" (default) or "character".
            options: Engine-specific options.

        Yields:
            TTSChunkResult with audio and optional alignment.
        """
        from macaw._types import TTSChunkResult as _TTSChunkResult

        result = self.synthesize(
            text=text,
            voice=voice,
            sample_rate=sample_rate,
            speed=speed,
            options=options,
        )

        stream: AsyncIterator[bytes] = await _resolve_stream(result)

        async for audio_chunk in stream:
            yield _TTSChunkResult(audio=audio_chunk)

    async def post_load_hook(self) -> None:  # noqa: B027
        """Optional hook called after load() and before warmup.

        Override this method to load auxiliary models that depend on the
        main model being in memory (e.g., vocoder, speaker embedding
        extractor, language-specific adapters).

        The default implementation is a no-op — engines opt in by
        overriding, existing engines are unaffected.
        """

    @abstractmethod
    async def unload(self) -> None:
        """Unload the TTS model from memory.

        Frees resources (GPU memory, buffers). After unload, load() must
        be called again before synthesize.
        """
        ...

    @abstractmethod
    async def health(self) -> dict[str, str]:
        """TTS backend status.

        Returns:
            Dict with at least {"status": "ok"|"loading"|"error"}.
        """
        ...
