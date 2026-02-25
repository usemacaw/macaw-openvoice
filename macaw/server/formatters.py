"""Response formatters for BatchResult -> API response format."""

from __future__ import annotations

import base64
from typing import TYPE_CHECKING, Any

from fastapi.responses import PlainTextResponse

from macaw._types import ResponseFormat

if TYPE_CHECKING:
    from macaw._types import BatchResult


def format_response(
    result: BatchResult,
    fmt: ResponseFormat,
    task: str = "transcribe",
    entity_detection: list[str] | None = None,
    additional_formats: list[dict[str, str]] | None = None,
) -> Any:
    """Format BatchResult for the requested response format.

    Args:
        result: Transcription result.
        fmt: Desired response format.
        task: Task type ("transcribe" or "translate").
        entity_detection: Category filter for entity detection (None = disabled).
        additional_formats: List of export format requests (e.g., [{"format":"srt"}]).

    Returns:
        Response appropriate for the requested format.
    """
    match fmt:
        case ResponseFormat.JSON:
            return {"text": result.text}
        case ResponseFormat.VERBOSE_JSON:
            return _to_verbose_json(
                result,
                task,
                entity_detection=entity_detection,
                additional_formats=additional_formats,
            )
        case ResponseFormat.TEXT:
            return PlainTextResponse(result.text)
        case ResponseFormat.SRT:
            return PlainTextResponse(_to_srt(result), media_type="text/plain")
        case ResponseFormat.VTT:
            return PlainTextResponse(_to_vtt(result), media_type="text/plain")
        case _:  # pragma: no cover
            return {"text": result.text}


def _to_verbose_json(
    result: BatchResult,
    task: str,
    entity_detection: list[str] | None = None,
    additional_formats: list[dict[str, str]] | None = None,
) -> dict[str, Any]:
    """Convert BatchResult to verbose_json format."""
    segments = [
        {
            "id": seg.id,
            "start": seg.start,
            "end": seg.end,
            "text": seg.text,
            "avg_logprob": seg.avg_logprob,
            "compression_ratio": seg.compression_ratio,
            "no_speech_prob": seg.no_speech_prob,
        }
        for seg in result.segments
    ]

    response: dict[str, Any] = {
        "task": task,
        "language": result.language,
        "duration": result.duration,
        "text": result.text,
        "segments": segments,
    }

    if result.words is not None:
        response["words"] = [
            {"word": w.word, "start": w.start, "end": w.end} for w in result.words
        ]

    if result.speaker_segments is not None:
        response["speaker_segments"] = [
            {
                "speaker_id": s.speaker_id,
                "start": s.start,
                "end": s.end,
                "text": s.text,
            }
            for s in result.speaker_segments
        ]

    if entity_detection is not None:
        from macaw.postprocessing.entity_detection import create_entity_detector

        detector = create_entity_detector()
        entities = detector.detect(result.text, categories=entity_detection)
        response["entities"] = [
            {
                "text": e.text,
                "entity_type": e.entity_type,
                "category": e.category,
                "start_char": e.start_char,
                "end_char": e.end_char,
            }
            for e in entities
        ]

    if additional_formats:
        response["additional_formats"] = _build_additional_formats(result, additional_formats)

    return response


def _format_timestamp_srt(seconds: float) -> str:
    """Format timestamp to SRT format (HH:MM:SS,mmm)."""
    total_ms = max(0, round(seconds * 1000))
    hours, remainder = divmod(total_ms, 3_600_000)
    minutes, remainder = divmod(remainder, 60_000)
    secs, millis = divmod(remainder, 1000)
    return f"{hours:02d}:{minutes:02d}:{secs:02d},{millis:03d}"


def _format_timestamp_vtt(seconds: float) -> str:
    """Format timestamp to VTT format (HH:MM:SS.mmm)."""
    total_ms = max(0, round(seconds * 1000))
    hours, remainder = divmod(total_ms, 3_600_000)
    minutes, remainder = divmod(remainder, 60_000)
    secs, millis = divmod(remainder, 1000)
    return f"{hours:02d}:{minutes:02d}:{secs:02d}.{millis:03d}"


def _to_srt(result: BatchResult) -> str:
    """Convert BatchResult to SRT format."""
    lines: list[str] = []
    counter = 0
    for seg in result.segments:
        text = seg.text.strip()
        if not text:
            continue
        counter += 1
        start = _format_timestamp_srt(seg.start)
        end = _format_timestamp_srt(seg.end)
        lines.append(f"{counter}")
        lines.append(f"{start} --> {end}")
        lines.append(text)
        lines.append("")
    return "\n".join(lines)


def _to_vtt(result: BatchResult) -> str:
    """Convert BatchResult to WebVTT format."""
    lines: list[str] = ["WEBVTT", ""]
    for seg in result.segments:
        text = seg.text.strip()
        if not text:
            continue
        start = _format_timestamp_vtt(seg.start)
        end = _format_timestamp_vtt(seg.end)
        lines.append(f"{start} --> {end}")
        lines.append(text)
        lines.append("")
    return "\n".join(lines)


def _build_additional_formats(
    result: BatchResult,
    format_requests: list[dict[str, str]],
) -> list[dict[str, str]]:
    """Run transcript formatters and return base64-encoded results.

    Each entry in format_requests should have at minimum a ``format`` key
    (e.g., ``{"format": "srt"}``). Optional keys ``include_speakers`` and
    ``include_timestamps`` control formatting.

    Returns a list of dicts with: format, content (base64), content_type,
    file_extension.
    """
    from macaw.postprocessing.formatters import create_formatter
    from macaw.postprocessing.formatters.interface import FormatOptions

    results: list[dict[str, str]] = []
    for req in format_requests:
        fmt_name = req.get("format", "")
        formatter = create_formatter(fmt_name)

        options = FormatOptions(
            include_speakers=req.get("include_speakers", "false").lower() == "true"
            if isinstance(req.get("include_speakers"), str)
            else bool(req.get("include_speakers", False)),
            include_timestamps=req.get("include_timestamps", "false").lower() == "true"
            if isinstance(req.get("include_timestamps"), str)
            else bool(req.get("include_timestamps", False)),
        )

        output = formatter.format(result, options)
        results.append(
            {
                "format": fmt_name,
                "content": base64.b64encode(output.content).decode("ascii"),
                "content_type": output.content_type,
                "file_extension": output.file_extension,
            }
        )

    return results
