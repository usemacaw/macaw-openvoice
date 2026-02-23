"""Abstract interface for voice changer (speech-to-speech) backends.

Every voice changer engine must implement this interface to plug into
the Macaw runtime. Voice conversion transforms source audio to a
target voice while preserving timing, emotion, and intonation.

Adding an **external** engine (separate package):
1. Implement VoiceChangerBackend in your own package
2. Create a macaw.yaml manifest with ``python_package: your_module``
   and ``type: voice_changer``
3. Install your package in the same environment

The runtime loads external VC engines via ``load_external_backend()``.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import TYPE_CHECKING

from macaw._audio_constants import TTS_DEFAULT_SAMPLE_RATE

if TYPE_CHECKING:
    from collections.abc import AsyncIterator

    from macaw._types import VoiceChangerCapabilities


class VoiceChangerBackend(ABC):
    """Contract that every voice changer engine must implement.

    The key difference from TTSBackend is that convert() accepts source
    audio bytes (not text) and returns transformed audio with the target
    voice applied.
    """

    @abstractmethod
    async def load(self, model_path: str, config: dict[str, object]) -> None:
        """Load the voice changer model into memory.

        Args:
            model_path: Path to the model files.
            config: engine_config from the macaw.yaml manifest.

        Raises:
            ModelLoadError: If the model cannot be loaded.
        """
        ...

    @abstractmethod
    async def convert(
        self,
        source_audio: bytes,
        voice: str = "default",
        *,
        sample_rate: int = TTS_DEFAULT_SAMPLE_RATE,
        options: dict[str, object] | None = None,
    ) -> AsyncIterator[bytes]:
        """Convert source audio to target voice (streaming PCM chunks).

        Preserves timing, emotion, and intonation from the source audio
        while replacing the voice identity with the target voice.

        Args:
            source_audio: Source audio as PCM 16-bit 16kHz mono bytes.
            voice: Target voice identifier.
            sample_rate: Output audio sample rate.
            options: Engine-specific options (e.g., voice_settings,
                ref_audio for target voice reference, pitch_shift).

        Yields:
            16-bit PCM audio chunks with the target voice.

        Raises:
            TTSSynthesisError: If voice conversion fails.
        """
        ...

    @abstractmethod
    async def capabilities(self) -> VoiceChangerCapabilities:
        """Return engine capabilities for runtime decision-making."""
        ...

    async def post_load_hook(self) -> None:  # noqa: B027
        """Optional hook called after load() and before first convert().

        Override to perform engine-specific warmup (e.g., dummy inference
        to prime GPU caches).
        """
