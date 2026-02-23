"""SRT subtitle formatter for transcripts."""

from __future__ import annotations

from typing import TYPE_CHECKING

from macaw.postprocessing.formatters.interface import (
    FormatOptions,
    FormattedOutput,
    TranscriptFormatter,
)

if TYPE_CHECKING:
    from macaw._types import BatchResult


def _format_timestamp_srt(seconds: float) -> str:
    """Format seconds to SRT timestamp (HH:MM:SS,mmm)."""
    total_ms = max(0, round(seconds * 1000))
    hours, remainder = divmod(total_ms, 3_600_000)
    minutes, remainder = divmod(remainder, 60_000)
    secs, millis = divmod(remainder, 1000)
    return f"{hours:02d}:{minutes:02d}:{secs:02d},{millis:03d}"


class SRTFormatter(TranscriptFormatter):
    """Format transcript as SRT subtitles."""

    @property
    def format_name(self) -> str:
        return "srt"

    def format(self, result: BatchResult, options: FormatOptions | None = None) -> FormattedOutput:
        opts = options or FormatOptions()
        lines: list[str] = []
        counter = 0

        for seg in result.segments:
            text = seg.text.strip()
            if not text:
                continue

            # Prepend speaker label if available and requested
            if opts.include_speakers and result.speaker_segments:
                speaker = _find_speaker_for_time(result, seg.start)
                if speaker:
                    text = f"[{speaker}] {text}"

            # Split long lines
            chunks = _split_text(text, opts.max_chars_per_segment)
            for chunk in chunks:
                counter += 1
                start = _format_timestamp_srt(seg.start)
                end = _format_timestamp_srt(seg.end)
                lines.append(str(counter))
                lines.append(f"{start} --> {end}")
                lines.append(chunk)
                lines.append("")

        content = "\n".join(lines)
        return FormattedOutput(
            content=content.encode("utf-8"),
            content_type="text/srt",
            file_extension=".srt",
        )


def _find_speaker_for_time(result: BatchResult, time_s: float) -> str | None:
    """Find the speaker label for a given timestamp."""
    if result.speaker_segments is None:
        return None
    for spk in result.speaker_segments:
        if spk.start <= time_s <= spk.end:
            return spk.speaker_id
    return None


def _split_text(text: str, max_chars: int) -> list[str]:
    """Split text into lines not exceeding max_chars.

    Splits on word boundaries. Returns the original text as a single-item
    list if it fits.
    """
    if len(text) <= max_chars:
        return [text]

    chunks: list[str] = []
    words = text.split()
    current_line: list[str] = []
    current_len = 0

    for word in words:
        word_len = len(word)
        separator_len = 1 if current_line else 0
        if current_len + separator_len + word_len > max_chars and current_line:
            chunks.append(" ".join(current_line))
            current_line = [word]
            current_len = word_len
        else:
            current_line.append(word)
            current_len += separator_len + word_len

    if current_line:
        chunks.append(" ".join(current_line))
    return chunks
