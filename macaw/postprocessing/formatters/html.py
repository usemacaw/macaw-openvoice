"""HTML transcript formatter.

Generates a styled HTML document from a transcript. Uses simple string
templating (no external dependencies). The output is self-contained with
inline CSS.
"""

from __future__ import annotations

from html import escape
from typing import TYPE_CHECKING

from macaw.postprocessing.formatters.interface import (
    FormatOptions,
    FormattedOutput,
    TranscriptFormatter,
)

if TYPE_CHECKING:
    from macaw._types import BatchResult


_HTML_TEMPLATE_START = """\
<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<title>Transcript</title>
<style>
body { font-family: system-ui, sans-serif; max-width: 800px; margin: 2em auto; padding: 0 1em; line-height: 1.6; }
.segment { margin-bottom: 0.5em; }
.timestamp { color: #666; font-size: 0.85em; font-family: monospace; }
.speaker { font-weight: bold; color: #2563eb; }
</style>
</head>
<body>
<h1>Transcript</h1>
"""

_HTML_TEMPLATE_END = """\
</body>
</html>
"""


class HTMLFormatter(TranscriptFormatter):
    """Format transcript as a styled HTML document.

    Self-contained: uses inline CSS, no external dependencies.
    """

    @property
    def format_name(self) -> str:
        return "html"

    def format(self, result: BatchResult, options: FormatOptions | None = None) -> FormattedOutput:
        opts = options or FormatOptions()
        parts: list[str] = [_HTML_TEMPLATE_START]

        for seg in result.segments:
            text = seg.text.strip()
            if not text:
                continue

            parts.append('<div class="segment">')

            if opts.include_timestamps:
                ts = _format_timestamp_html(seg.start)
                parts.append(f'<span class="timestamp">{ts}</span> ')

            if opts.include_speakers and result.speaker_segments:
                speaker = _find_speaker_for_time(result, seg.start)
                if speaker:
                    parts.append(f'<span class="speaker">{escape(speaker)}:</span> ')

            parts.append(f"<span>{escape(text)}</span>")
            parts.append("</div>")

        parts.append(_HTML_TEMPLATE_END)
        content = "\n".join(parts)
        return FormattedOutput(
            content=content.encode("utf-8"),
            content_type="text/html",
            file_extension=".html",
        )


def _format_timestamp_html(seconds: float) -> str:
    """Format seconds as [MM:SS] for HTML display."""
    total_secs = max(0, round(seconds))
    minutes, secs = divmod(total_secs, 60)
    return f"[{minutes:02d}:{secs:02d}]"


def _find_speaker_for_time(result: BatchResult, time_s: float) -> str | None:
    """Find the speaker label for a given timestamp."""
    if result.speaker_segments is None:
        return None
    for spk in result.speaker_segments:
        if spk.start <= time_s <= spk.end:
            return spk.speaker_id
    return None
