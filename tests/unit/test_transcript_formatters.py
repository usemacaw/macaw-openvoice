"""Tests for macaw.postprocessing.formatters — transcript export formatters.

Sprint K: Additional STT Output Formats.
"""

from __future__ import annotations

import pytest

from macaw._types import BatchResult, SegmentDetail, SpeakerSegment
from macaw.postprocessing.formatters import (
    SUPPORTED_EXPORT_FORMATS,
    create_formatter,
)
from macaw.postprocessing.formatters.html import HTMLFormatter
from macaw.postprocessing.formatters.interface import FormatOptions, FormattedOutput
from macaw.postprocessing.formatters.srt import SRTFormatter, _split_text
from macaw.postprocessing.formatters.txt import TXTFormatter

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


def _make_result(
    text: str = "Hello world. How are you?",
    n_segments: int = 2,
    with_speakers: bool = False,
) -> BatchResult:
    """Build a BatchResult for testing."""
    segments = tuple(
        SegmentDetail(
            id=i,
            start=i * 5.0,
            end=(i + 1) * 5.0 - 0.1,
            text=f"Segment {i} text." if n_segments > 1 else text,
        )
        for i in range(n_segments)
    )

    speakers = None
    if with_speakers:
        speakers = tuple(
            SpeakerSegment(
                speaker_id=f"SPEAKER_{i % 2}",
                start=i * 5.0,
                end=(i + 1) * 5.0 - 0.1,
                text=f"Segment {i} text.",
            )
            for i in range(n_segments)
        )

    return BatchResult(
        text=text,
        language="en",
        duration=n_segments * 5.0,
        segments=segments,
        speaker_segments=speakers,
    )


# ---------------------------------------------------------------------------
# Factory
# ---------------------------------------------------------------------------


class TestCreateFormatter:
    """Tests for the create_formatter factory."""

    def test_create_srt(self) -> None:
        assert isinstance(create_formatter("srt"), SRTFormatter)

    def test_create_txt(self) -> None:
        assert isinstance(create_formatter("txt"), TXTFormatter)

    def test_create_html(self) -> None:
        assert isinstance(create_formatter("html"), HTMLFormatter)

    def test_case_insensitive(self) -> None:
        assert isinstance(create_formatter("SRT"), SRTFormatter)

    def test_whitespace_stripped(self) -> None:
        assert isinstance(create_formatter("  txt  "), TXTFormatter)

    def test_unknown_raises(self) -> None:
        with pytest.raises(ValueError, match="Unknown export format"):
            create_formatter("pdf")

    def test_supported_formats_constant(self) -> None:
        assert frozenset({"srt", "txt", "html"}) == SUPPORTED_EXPORT_FORMATS


# ---------------------------------------------------------------------------
# SRT Formatter
# ---------------------------------------------------------------------------


class TestSRTFormatter:
    """Tests for SRTFormatter."""

    def test_format_name(self) -> None:
        assert SRTFormatter().format_name == "srt"

    def test_basic_output(self) -> None:
        result = _make_result(n_segments=2)
        output = SRTFormatter().format(result)

        assert isinstance(output, FormattedOutput)
        assert output.content_type == "text/srt"
        assert output.file_extension == ".srt"

        text = output.content.decode("utf-8")
        assert "1\n" in text
        assert "00:00:00,000 --> " in text
        assert "Segment 0 text." in text

    def test_two_segments(self) -> None:
        result = _make_result(n_segments=2)
        output = SRTFormatter().format(result)
        text = output.content.decode("utf-8")
        assert "2\n" in text
        assert "Segment 1 text." in text

    def test_empty_segments_skipped(self) -> None:
        result = BatchResult(
            text="",
            language="en",
            duration=5.0,
            segments=(
                SegmentDetail(id=0, start=0.0, end=5.0, text=""),
                SegmentDetail(id=1, start=5.0, end=10.0, text="Hello"),
            ),
        )
        output = SRTFormatter().format(result)
        text = output.content.decode("utf-8")
        # Only one numbered entry (the non-empty one)
        assert text.startswith("1\n")
        assert "2\n" not in text

    def test_with_speakers(self) -> None:
        result = _make_result(n_segments=2, with_speakers=True)
        output = SRTFormatter().format(result, FormatOptions(include_speakers=True))
        text = output.content.decode("utf-8")
        assert "[SPEAKER_0]" in text

    def test_without_speakers_option(self) -> None:
        result = _make_result(n_segments=2, with_speakers=True)
        output = SRTFormatter().format(result, FormatOptions(include_speakers=False))
        text = output.content.decode("utf-8")
        assert "[SPEAKER_0]" not in text

    def test_timestamp_format(self) -> None:
        result = BatchResult(
            text="Test",
            language="en",
            duration=3661.5,
            segments=(SegmentDetail(id=0, start=3661.5, end=3665.0, text="Test"),),
        )
        output = SRTFormatter().format(result)
        text = output.content.decode("utf-8")
        # 3661.5s = 01:01:01,500
        assert "01:01:01,500" in text


class TestSplitText:
    """Tests for the _split_text helper."""

    def test_short_text_unchanged(self) -> None:
        assert _split_text("short text", 80) == ["short text"]

    def test_long_text_splits(self) -> None:
        text = "word " * 20  # 100 chars
        chunks = _split_text(text.strip(), 50)
        assert len(chunks) > 1
        for chunk in chunks:
            assert len(chunk) <= 50


# ---------------------------------------------------------------------------
# TXT Formatter
# ---------------------------------------------------------------------------


class TestTXTFormatter:
    """Tests for TXTFormatter."""

    def test_format_name(self) -> None:
        assert TXTFormatter().format_name == "txt"

    def test_basic_output(self) -> None:
        result = _make_result(n_segments=2)
        output = TXTFormatter().format(result)

        assert output.content_type == "text/plain"
        assert output.file_extension == ".txt"

        text = output.content.decode("utf-8")
        assert "Segment 0 text." in text
        assert "Segment 1 text." in text

    def test_with_timestamps(self) -> None:
        result = _make_result(n_segments=1)
        output = TXTFormatter().format(result, FormatOptions(include_timestamps=True))
        text = output.content.decode("utf-8")
        assert "[00:00]" in text

    def test_with_speakers(self) -> None:
        result = _make_result(n_segments=2, with_speakers=True)
        output = TXTFormatter().format(result, FormatOptions(include_speakers=True))
        text = output.content.decode("utf-8")
        assert "SPEAKER_0:" in text

    def test_with_timestamps_and_speakers(self) -> None:
        result = _make_result(n_segments=2, with_speakers=True)
        output = TXTFormatter().format(
            result, FormatOptions(include_timestamps=True, include_speakers=True)
        )
        text = output.content.decode("utf-8")
        assert "[00:00]" in text
        assert "SPEAKER_0:" in text

    def test_no_options_plain_text(self) -> None:
        result = _make_result(n_segments=1, text="Hello world")
        output = TXTFormatter().format(result)
        text = output.content.decode("utf-8")
        # No timestamps, no speakers — just the text
        assert "[" not in text
        assert ":" not in text or "Hello world" in text


# ---------------------------------------------------------------------------
# HTML Formatter
# ---------------------------------------------------------------------------


class TestHTMLFormatter:
    """Tests for HTMLFormatter."""

    def test_format_name(self) -> None:
        assert HTMLFormatter().format_name == "html"

    def test_basic_output(self) -> None:
        result = _make_result(n_segments=2)
        output = HTMLFormatter().format(result)

        assert output.content_type == "text/html"
        assert output.file_extension == ".html"

        text = output.content.decode("utf-8")
        assert "<!DOCTYPE html>" in text
        assert "<h1>Transcript</h1>" in text
        assert "Segment 0 text." in text

    def test_html_escaping(self) -> None:
        result = BatchResult(
            text="<script>alert(1)</script>",
            language="en",
            duration=5.0,
            segments=(
                SegmentDetail(
                    id=0,
                    start=0.0,
                    end=5.0,
                    text="<script>alert(1)</script>",
                ),
            ),
        )
        output = HTMLFormatter().format(result)
        text = output.content.decode("utf-8")
        # Script tags should be escaped
        assert "<script>" not in text
        assert "&lt;script&gt;" in text

    def test_with_timestamps(self) -> None:
        result = _make_result(n_segments=1)
        output = HTMLFormatter().format(result, FormatOptions(include_timestamps=True))
        text = output.content.decode("utf-8")
        assert 'class="timestamp"' in text
        assert "[00:00]" in text

    def test_with_speakers(self) -> None:
        result = _make_result(n_segments=2, with_speakers=True)
        output = HTMLFormatter().format(result, FormatOptions(include_speakers=True))
        text = output.content.decode("utf-8")
        assert 'class="speaker"' in text
        assert "SPEAKER_0" in text

    def test_self_contained_css(self) -> None:
        result = _make_result(n_segments=1)
        output = HTMLFormatter().format(result)
        text = output.content.decode("utf-8")
        assert "<style>" in text
        assert "font-family" in text


# ---------------------------------------------------------------------------
# FormatOptions defaults
# ---------------------------------------------------------------------------


class TestFormatOptions:
    """Tests for FormatOptions defaults."""

    def test_defaults(self) -> None:
        opts = FormatOptions()
        assert opts.include_speakers is False
        assert opts.include_timestamps is False
        assert opts.max_chars_per_segment == 80

    def test_frozen(self) -> None:
        opts = FormatOptions()
        with pytest.raises(AttributeError):
            opts.include_speakers = True  # type: ignore[misc]
