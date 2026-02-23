"""Transcript persistence interface.

Defines the abstract store contract and the ``StoredTranscript`` data model
for persisting transcription results across requests.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field


@dataclass(frozen=True, slots=True)
class StoredTranscript:
    """A persisted transcription result."""

    transcription_id: str
    text: str
    language: str | None = None
    duration: float | None = None
    model: str | None = None
    created_at: str = ""  # ISO 8601
    metadata: dict[str, object] = field(default_factory=dict)


class TranscriptStore(ABC):
    """Abstract base class for transcript persistence."""

    @abstractmethod
    async def save(self, transcript: StoredTranscript) -> str:
        """Save transcript, return transcription_id."""

    @abstractmethod
    async def get(self, transcription_id: str) -> StoredTranscript | None:
        """Get transcript by ID, return None if not found."""

    @abstractmethod
    async def delete(self, transcription_id: str) -> bool:
        """Delete transcript, return True if deleted."""

    @abstractmethod
    async def list_all(self, limit: int = 100, offset: int = 0) -> list[StoredTranscript]:
        """List transcripts with pagination."""
