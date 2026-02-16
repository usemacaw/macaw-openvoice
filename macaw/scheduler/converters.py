"""Conversion between TranscribeRequest/BatchResult and gRPC protobuf messages.

Functions used by the Scheduler for communication with workers via gRPC.
These are the inverse of the converters in workers/stt/converters.py.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from macaw._types import BatchResult, SegmentDetail, WordTimestamp
from macaw.proto import TranscribeFileRequest, TranscribeFileResponse

if TYPE_CHECKING:
    from macaw.server.models.requests import TranscribeRequest


def build_proto_request(request: TranscribeRequest) -> TranscribeFileRequest:
    """Convert internal TranscribeRequest to TranscribeFileRequest protobuf."""
    return TranscribeFileRequest(
        request_id=request.request_id,
        audio_data=request.audio_data,
        language=request.language or "",
        response_format=request.response_format.value,
        temperature=request.temperature,
        timestamp_granularities=request.timestamp_granularities,
        initial_prompt=request.initial_prompt or "",
        hot_words=request.hot_words if request.hot_words else [],
        task=request.task,
    )


def proto_response_to_batch_result(response: TranscribeFileResponse) -> BatchResult:
    """Convert TranscribeFileResponse protobuf to Macaw BatchResult."""
    segments = tuple(
        SegmentDetail(
            id=s.id,
            start=s.start,
            end=s.end,
            text=s.text,
            avg_logprob=s.avg_logprob,
            no_speech_prob=s.no_speech_prob,
            compression_ratio=s.compression_ratio,
        )
        for s in response.segments
    )

    words: tuple[WordTimestamp, ...] | None = None
    if response.words:
        words = tuple(
            WordTimestamp(
                word=w.word,
                start=w.start,
                end=w.end,
                probability=w.probability if w.probability != 0.0 else None,
            )
            for w in response.words
        )

    return BatchResult(
        text=response.text,
        language=response.language,
        duration=response.duration,
        segments=segments,
        words=words,
    )
