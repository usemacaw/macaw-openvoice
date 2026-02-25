"""Transcript formatter package — converts BatchResult to export formats."""

from __future__ import annotations

from macaw.postprocessing.formatters.interface import (
    FormatOptions,
    FormattedOutput,
    TranscriptFormatter,
)

__all__ = [
    "FormatOptions",
    "FormattedOutput",
    "TranscriptFormatter",
    "create_formatter",
]

#: Supported export format names.
SUPPORTED_EXPORT_FORMATS: frozenset[str] = frozenset({"srt", "txt", "html"})


def create_formatter(format_name: str) -> TranscriptFormatter:
    """Create a transcript formatter by name.

    Args:
        format_name: Format identifier ("srt", "txt", "html").

    Returns:
        TranscriptFormatter instance.

    Raises:
        ValueError: If the format name is not recognized.
    """
    name = format_name.strip().lower()

    if name == "srt":
        from macaw.postprocessing.formatters.srt import SRTFormatter

        return SRTFormatter()

    if name == "txt":
        from macaw.postprocessing.formatters.txt import TXTFormatter

        return TXTFormatter()

    if name == "html":
        from macaw.postprocessing.formatters.html import HTMLFormatter

        return HTMLFormatter()

    msg = (
        f"Unknown export format '{format_name}'. "
        f"Supported: {', '.join(sorted(SUPPORTED_EXPORT_FORMATS))}."
    )
    raise ValueError(msg)
