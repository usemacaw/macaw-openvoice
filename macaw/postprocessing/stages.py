"""Base interface for text post-processing pipeline stages."""

from __future__ import annotations

from abc import ABC, abstractmethod


class TextStage(ABC):
    """Text post-processing pipeline stage.

    Each stage receives raw (or partially processed) text and returns
    transformed text. Stages are composed in sequence by the
    PostProcessingPipeline.
    """

    @property
    @abstractmethod
    def name(self) -> str:
        """Identifier name for the stage (e.g. 'itn', 'entity_formatting')."""
        ...

    @abstractmethod
    def process(self, text: str, *, language: str | None = None) -> str:
        """Process text and return transformed text.

        Args:
            text: Input text (may be empty).
            language: ISO 639-1 language code (optional). Stages that are
                language-aware (e.g. ITN) use this to select the correct
                normalizer. Stages that are language-agnostic ignore it.

        Returns:
            Processed text.
        """
        ...
