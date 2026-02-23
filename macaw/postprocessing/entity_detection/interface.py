"""Abstract interface for entity detection in transcription text."""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class DetectedEntity:
    """A detected entity in transcription text.

    Attributes:
        text: The matched text span.
        entity_type: Specific type (e.g. "email_address", "credit_card").
        category: Broad category: "pii", "phi", or "pci".
        start_char: Start character offset (inclusive).
        end_char: End character offset (exclusive).
    """

    text: str
    entity_type: str
    category: str
    start_char: int
    end_char: int


class EntityDetector(ABC):
    """Abstract base class for entity detection."""

    @abstractmethod
    def detect(self, text: str, categories: list[str] | None = None) -> list[DetectedEntity]:
        """Detect entities in text, optionally filtered by category.

        Args:
            text: Input text to scan.
            categories: If provided, only return entities whose category
                is in this list. None or ["all"] means all categories.

        Returns:
            List of detected entities sorted by start_char.
        """
