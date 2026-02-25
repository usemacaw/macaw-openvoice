"""Plain text transcript formatter."""

from __future__ import annotations

from typing import TYPE_CHECKING

from macaw.postprocessing.formatters.interface import (
    FormatOptions,
    FormattedOutput,
    TranscriptFormatter,
)

if TYPE_CHECKING:
    from macaw._types import BatchResult


def _format_timestamp_short(seconds: float) -> str:
    """Format seconds as [MM:SS] for compact text display."""
    total_secs = max(0, round(seconds))
    minutes, secs = divmod(total_secs, 60)
    return f"[{minutes:02d}:{secs:02d}]"


class TXTFormatter(TranscriptFormatter):
    """Format transcript as plain text.

    Supports optional timestamps and speaker labels.
    """

    @property
    def format_name(self) -> str:
        return "txt"

    def format(self, result: BatchResult, options: FormatOptions | None = None) -> FormattedOutput:
        opts = options or FormatOptions()
        lines: list[str] = []

        for seg in result.segments:
            text = seg.text.strip()
            if not text:
                continue

            parts: list[str] = []

            if opts.include_timestamps:
                parts.append(_format_timestamp_short(seg.start))

            if opts.include_speakers and result.speaker_segments:
                speaker = _find_speaker_for_time(result, seg.start)
                if speaker:
                    parts.append(f"{speaker}:")

            parts.append(text)
            lines.append(" ".join(parts))

        content = "\n".join(lines)
        return FormattedOutput(
            content=content.encode("utf-8"),
            content_type="text/plain",
            file_extension=".txt",
        )


def _find_speaker_for_time(result: BatchResult, time_s: float) -> str | None:
    """Find the speaker label for a given timestamp."""
    if result.speaker_segments is None:
        return None
    for spk in result.speaker_segments:
        if spk.start <= time_s <= spk.end:
            return spk.speaker_id
    return None
