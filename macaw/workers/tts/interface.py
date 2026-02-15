"""Abstract interface for TTS backends.

Every TTS backend (Kokoro, Qwen3-TTS, etc.) must implement this interface
to plug into the Macaw runtime.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from collections.abc import AsyncIterator

    from macaw._types import VoiceInfo


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
    async def synthesize(
        self,
        text: str,
        voice: str = "default",
        *,
        sample_rate: int = 24000,
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
