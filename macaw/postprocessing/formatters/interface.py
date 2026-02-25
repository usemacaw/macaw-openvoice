"""Abstract base class for transcript formatters."""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from macaw._types import BatchResult


@dataclass(frozen=True, slots=True)
class FormatOptions:
    """Options controlling transcript format output.

    Attributes:
        include_speakers: Include speaker labels (requires diarization).
        include_timestamps: Include segment timestamps in text output.
        max_chars_per_segment: Maximum characters per subtitle/segment line.
    """

    include_speakers: bool = False
    include_timestamps: bool = False
    max_chars_per_segment: int = 80


@dataclass(frozen=True, slots=True)
class FormattedOutput:
    """Result of formatting a transcript.

    Attributes:
        content: Formatted transcript bytes.
        content_type: MIME type of the output (e.g., "text/plain").
        file_extension: Suggested file extension (e.g., ".srt").
    """

    content: bytes
    content_type: str
    file_extension: str


class TranscriptFormatter(ABC):
    """ABC for transcript formatters.

    Implementations convert a BatchResult into a specific output format
    (SRT, TXT, HTML, etc.). Each formatter handles its own content type
    and file extension.
    """

    @property
    @abstractmethod
    def format_name(self) -> str:
        """Short name for this format (e.g., 'srt', 'txt', 'html')."""

    @abstractmethod
    def format(self, result: BatchResult, options: FormatOptions | None = None) -> FormattedOutput:
        """Format a transcript into the target format.

        Args:
            result: Transcription result with segments, words, speakers.
            options: Formatting options. Uses defaults if None.

        Returns:
            FormattedOutput with content bytes, content_type, and extension.
        """
