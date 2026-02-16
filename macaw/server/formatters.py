"""Response formatters for BatchResult -> API response format."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from fastapi.responses import PlainTextResponse

from macaw._types import ResponseFormat

if TYPE_CHECKING:
    from macaw._types import BatchResult


def format_response(result: BatchResult, fmt: ResponseFormat, task: str = "transcribe") -> Any:
    """Format BatchResult for the requested response format.

    Args:
        result: Transcription result.
        fmt: Desired response format.
        task: Task type ("transcribe" or "translate").

    Returns:
        Response appropriate for the requested format.
    """
    match fmt:
        case ResponseFormat.JSON:
            return {"text": result.text}
        case ResponseFormat.VERBOSE_JSON:
            return _to_verbose_json(result, task)
        case ResponseFormat.TEXT:
            return PlainTextResponse(result.text)
        case ResponseFormat.SRT:
            return PlainTextResponse(_to_srt(result), media_type="text/plain")
        case ResponseFormat.VTT:
            return PlainTextResponse(_to_vtt(result), media_type="text/plain")
        case _:  # pragma: no cover
            return {"text": result.text}


def _to_verbose_json(result: BatchResult, task: str) -> dict[str, Any]:
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
