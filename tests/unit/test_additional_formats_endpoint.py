"""Tests for additional_formats parameter on transcription endpoints.

Sprint K: Additional STT Output Formats — API wiring.
"""

from __future__ import annotations

import base64

import pytest

from macaw._types import BatchResult, ResponseFormat, SegmentDetail
from macaw.server.formatters import _build_additional_formats, format_response

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


def _make_result() -> BatchResult:
    return BatchResult(
        text="Hello world",
        language="en",
        duration=5.0,
        segments=(SegmentDetail(id=0, start=0.0, end=5.0, text="Hello world"),),
    )


# ---------------------------------------------------------------------------
# format_response with additional_formats
# ---------------------------------------------------------------------------


class TestFormatResponseAdditionalFormats:
    """Tests for format_response with additional_formats parameter."""

    def test_verbose_json_includes_additional_formats(self) -> None:
        result = _make_result()
        response = format_response(
            result,
            ResponseFormat.VERBOSE_JSON,
            additional_formats=[{"format": "srt"}],
        )
        assert "additional_formats" in response
        assert len(response["additional_formats"]) == 1
        fmt = response["additional_formats"][0]
        assert fmt["format"] == "srt"
        assert fmt["content_type"] == "text/srt"
        assert fmt["file_extension"] == ".srt"
        # Content is base64-encoded
        decoded = base64.b64decode(fmt["content"]).decode("utf-8")
        assert "Hello world" in decoded

    def test_multiple_formats(self) -> None:
        result = _make_result()
        response = format_response(
            result,
            ResponseFormat.VERBOSE_JSON,
            additional_formats=[
                {"format": "srt"},
                {"format": "txt"},
                {"format": "html"},
            ],
        )
        assert len(response["additional_formats"]) == 3
        formats = {f["format"] for f in response["additional_formats"]}
        assert formats == {"srt", "txt", "html"}

    def test_no_additional_formats_omits_key(self) -> None:
        result = _make_result()
        response = format_response(result, ResponseFormat.VERBOSE_JSON)
        assert "additional_formats" not in response

    def test_empty_list_omits_key(self) -> None:
        result = _make_result()
        response = format_response(
            result,
            ResponseFormat.VERBOSE_JSON,
            additional_formats=[],
        )
        # Empty list is falsy, so additional_formats should not be in response
        assert "additional_formats" not in response

    def test_json_format_ignores_additional_formats(self) -> None:
        """additional_formats only works with verbose_json."""
        result = _make_result()
        response = format_response(
            result,
            ResponseFormat.JSON,
            additional_formats=[{"format": "srt"}],
        )
        # JSON format returns simple dict without additional_formats
        assert "additional_formats" not in response

    def test_txt_format_with_timestamps(self) -> None:
        result = _make_result()
        response = format_response(
            result,
            ResponseFormat.VERBOSE_JSON,
            additional_formats=[{"format": "txt", "include_timestamps": "true"}],
        )
        decoded = base64.b64decode(response["additional_formats"][0]["content"]).decode("utf-8")
        assert "[00:00]" in decoded

    def test_srt_with_speakers(self) -> None:
        from macaw._types import SpeakerSegment

        result = BatchResult(
            text="Hello",
            language="en",
            duration=5.0,
            segments=(SegmentDetail(id=0, start=0.0, end=5.0, text="Hello"),),
            speaker_segments=(
                SpeakerSegment(speaker_id="SPK_0", start=0.0, end=5.0, text="Hello"),
            ),
        )
        response = format_response(
            result,
            ResponseFormat.VERBOSE_JSON,
            additional_formats=[{"format": "srt", "include_speakers": "true"}],
        )
        decoded = base64.b64decode(response["additional_formats"][0]["content"]).decode("utf-8")
        assert "[SPK_0]" in decoded


# ---------------------------------------------------------------------------
# _build_additional_formats
# ---------------------------------------------------------------------------


class TestBuildAdditionalFormats:
    """Tests for the _build_additional_formats helper."""

    def test_single_format(self) -> None:
        result = _make_result()
        formats = _build_additional_formats(result, [{"format": "txt"}])
        assert len(formats) == 1
        assert formats[0]["format"] == "txt"
        assert formats[0]["content_type"] == "text/plain"

    def test_base64_content_valid(self) -> None:
        result = _make_result()
        formats = _build_additional_formats(result, [{"format": "srt"}])
        # Should not raise
        decoded = base64.b64decode(formats[0]["content"])
        assert len(decoded) > 0

    def test_unknown_format_raises(self) -> None:
        result = _make_result()
        with pytest.raises(ValueError, match="Unknown export format"):
            _build_additional_formats(result, [{"format": "pdf"}])

    def test_include_speakers_bool(self) -> None:
        """Test that include_speakers can be passed as bool."""
        result = _make_result()
        formats = _build_additional_formats(
            result,
            [{"format": "txt", "include_speakers": True}],  # type: ignore[dict-item]
        )
        assert len(formats) == 1

    def test_html_content_type(self) -> None:
        result = _make_result()
        formats = _build_additional_formats(result, [{"format": "html"}])
        assert formats[0]["content_type"] == "text/html"
        assert formats[0]["file_extension"] == ".html"


# ---------------------------------------------------------------------------
# Pydantic models
# ---------------------------------------------------------------------------


class TestAdditionalFormatResponse:
    """Tests for the AdditionalFormatResponse Pydantic model."""

    def test_model_fields(self) -> None:
        from macaw.server.models.responses import AdditionalFormatResponse

        resp = AdditionalFormatResponse(
            format="srt",
            content="SGVsbG8=",
            content_type="text/srt",
            file_extension=".srt",
        )
        assert resp.format == "srt"
        assert resp.content == "SGVsbG8="

    def test_verbose_response_with_additional_formats(self) -> None:
        from macaw.server.models.responses import (
            AdditionalFormatResponse,
            VerboseTranscriptionResponse,
        )

        resp = VerboseTranscriptionResponse(
            language="en",
            duration=5.0,
            text="Hello",
            additional_formats=[
                AdditionalFormatResponse(
                    format="srt",
                    content="SGVsbG8=",
                    content_type="text/srt",
                    file_extension=".srt",
                )
            ],
        )
        assert resp.additional_formats is not None
        assert len(resp.additional_formats) == 1

    def test_verbose_response_without_additional_formats(self) -> None:
        from macaw.server.models.responses import VerboseTranscriptionResponse

        resp = VerboseTranscriptionResponse(
            language="en",
            duration=5.0,
            text="Hello",
        )
        assert resp.additional_formats is None
