"""Abstract interface for speaker diarization backends.

Every diarization backend must implement this interface to plug into the
Macaw STT worker as a post-processing step.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from macaw._types import SpeakerSegment


class DiarizationBackend(ABC):
    """Contract for speaker diarization backends.

    Diarization is a post-processing step that identifies speakers in
    pre-transcribed audio. The backend receives raw PCM audio and returns
    time-aligned segments with speaker identifiers.
    """

    @abstractmethod
    async def diarize(
        self,
        audio: bytes,
        sample_rate: int,
        *,
        max_speakers: int | None = None,
    ) -> tuple[SpeakerSegment, ...]:
        """Identify speakers in audio and return time-aligned segments.

        Args:
            audio: 16-bit PCM audio bytes.
            sample_rate: Sample rate of the audio (e.g. 16000).
            max_speakers: Maximum expected speakers (None = auto-detect).

        Returns:
            Tuple of SpeakerSegment with speaker_id, start, end, text.
        """
        ...

    @abstractmethod
    async def load(self) -> None:
        """Load diarization model into memory.

        Called explicitly during worker startup or lazily on first
        ``diarize()`` call.
        """
        ...

    @abstractmethod
    async def health(self) -> dict[str, str]:
        """Health check for the diarization backend.

        Returns:
            Dict with at least ``{"status": "ok"|"loading"|"error"}``.
        """
        ...
