"""Abstract interface for sound effect generation backends.

Every sound effect engine must implement this interface to plug into
the Macaw runtime. Sound generation transforms a text description into
audio (e.g., "rain on a tin roof", "explosion in a cave").

Adding an **external** engine (separate package):
1. Implement SoundEffectBackend in your own package
2. Create a macaw.yaml manifest with ``python_package: your_module``
   and ``type: sound_effect``
3. Install your package in the same environment

The runtime loads external SFX engines via ``load_external_backend()``.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from collections.abc import AsyncIterator

    from macaw._types import SoundEffectCapabilities


class SoundEffectBackend(ABC):
    """Contract that every sound effect engine must implement.

    Unlike TTSBackend (text -> speech with voice identity), this
    produces non-speech audio from a descriptive text prompt.
    """

    @abstractmethod
    async def load(self, model_path: str, config: dict[str, object]) -> None:
        """Load the sound effect model into memory.

        Args:
            model_path: Path to the model files.
            config: engine_config from the macaw.yaml manifest.

        Raises:
            ModelLoadError: If the model cannot be loaded.
        """
        ...

    @abstractmethod
    async def generate(
        self,
        text: str,
        *,
        duration_s: float = 5.0,
        prompt_influence: float = 0.3,
        loop: bool = False,
        options: dict[str, object] | None = None,
    ) -> AsyncIterator[bytes]:
        """Generate sound effect audio from a text description.

        Args:
            text: Descriptive prompt (e.g., "thunder rolling in the distance").
            duration_s: Desired output duration in seconds.
            prompt_influence: How closely to follow the prompt (0-1).
            loop: Whether to generate seamless looping audio.
            options: Engine-specific options.

        Yields:
            16-bit PCM audio chunks.

        Raises:
            TTSSynthesisError: If sound generation fails.
        """
        ...

    @abstractmethod
    async def capabilities(self) -> SoundEffectCapabilities:
        """Return engine capabilities for runtime decision-making."""
        ...

    async def post_load_hook(self) -> None:  # noqa: B027
        """Optional hook called after load() and before first generate().

        Override to perform engine-specific warmup (e.g., dummy inference
        to prime GPU caches).
        """
