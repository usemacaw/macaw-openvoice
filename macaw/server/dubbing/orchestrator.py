"""Dubbing orchestrator: STT -> translate -> TTS pipeline.

The orchestrator chains internal STT/TTS capabilities with a
pluggable translation backend to produce dubbed audio.
"""

from __future__ import annotations

from abc import ABC, abstractmethod

from macaw.logging import get_logger

logger = get_logger("dubbing.orchestrator")


class TranslationBackend(ABC):
    """Contract for pluggable text translation.

    Users must provide their own implementation (e.g., wrapper
    around Google Translate, DeepL, or an LLM).
    """

    @abstractmethod
    async def translate(
        self,
        text: str,
        source_lang: str,
        target_lang: str,
    ) -> str:
        """Translate text from source to target language.

        Args:
            text: Source text to translate.
            source_lang: BCP-47 source language code (e.g., "en").
            target_lang: BCP-47 target language code (e.g., "pt").

        Returns:
            Translated text.

        Raises:
            TranslationError: If translation fails.
        """
        ...


class DubbingOrchestrator:
    """Coordinates the dubbing pipeline.

    Requires a translation backend to be injected. STT and TTS
    are accessed via the Macaw runtime (scheduler/worker manager).

    MVP scope:
    - Audio input only (no video extraction)
    - Single target language per request
    - Sequential segment processing
    """

    def __init__(self, translation_backend: TranslationBackend) -> None:
        self._translator = translation_backend

    @property
    def translator(self) -> TranslationBackend:
        """Return the configured translation backend."""
        return self._translator
