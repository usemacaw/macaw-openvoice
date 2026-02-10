"""Estruturas auxiliares para acompanhar jobs no demo."""

from __future__ import annotations

import asyncio
import time
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from macaw._types import BatchResult, SegmentDetail, WordTimestamp
    from macaw.scheduler.queue import RequestPriority


@dataclass(slots=True)
class DemoJob:
    """Representa uma transcricao submetida pelo demo."""

    request_id: str
    model_name: str
    priority: RequestPriority
    language: str | None = None
    status: str = "queued"
    created_at: float = field(default_factory=time.time)
    updated_at: float = field(default_factory=time.time)
    result: BatchResult | None = None
    error: str | None = None

    def as_dict(self) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "request_id": self.request_id,
            "model": self.model_name,
            "priority": self.priority.name,
            "language": self.language,
            "status": self.status,
            "created_at": self.created_at,
            "updated_at": self.updated_at,
        }
        if self.error is not None:
            payload["error"] = self.error
        if self.result is not None:
            payload["result"] = _batch_result_as_dict(self.result)
        return payload


class DemoJobStore:
    """Armazena jobs de forma thread-safe com asyncio.Lock."""

    def __init__(self) -> None:
        self._jobs: dict[str, DemoJob] = {}
        self._lock = asyncio.Lock()

    async def add(self, job: DemoJob) -> None:
        async with self._lock:
            self._jobs[job.request_id] = job

    async def get(self, request_id: str) -> DemoJob | None:
        async with self._lock:
            job = self._jobs.get(request_id)
            return None if job is None else _clone(job)

    async def list(self) -> list[DemoJob]:
        async with self._lock:
            return [_clone(job) for job in self._jobs.values()]

    async def update(self, request_id: str, **changes: Any) -> DemoJob | None:
        async with self._lock:
            job = self._jobs.get(request_id)
            if job is None:
                return None
            for key, value in changes.items():
                setattr(job, key, value)
            job.updated_at = time.time()
            return _clone(job)

    async def remove(self, request_id: str) -> None:
        async with self._lock:
            self._jobs.pop(request_id, None)


def _clone(job: DemoJob) -> DemoJob:
    copied = DemoJob(
        request_id=job.request_id,
        model_name=job.model_name,
        priority=job.priority,
        language=job.language,
        status=job.status,
        created_at=job.created_at,
        updated_at=job.updated_at,
        result=job.result,
        error=job.error,
    )
    return copied


def _batch_result_as_dict(result: BatchResult) -> dict[str, Any]:
    return {
        "text": result.text,
        "language": result.language,
        "duration": result.duration,
        "segments": [_segment_as_dict(seg) for seg in result.segments],
        "words": None
        if result.words is None
        else [_word_as_dict(word) for word in result.words],
    }


def _segment_as_dict(segment: SegmentDetail) -> dict[str, Any]:
    return {
        "id": segment.id,
        "start": segment.start,
        "end": segment.end,
        "text": segment.text,
        "avg_logprob": segment.avg_logprob,
        "no_speech_prob": segment.no_speech_prob,
        "compression_ratio": segment.compression_ratio,
    }


def _word_as_dict(word: WordTimestamp) -> dict[str, Any]:
    return {
        "word": word.word,
        "start": word.start,
        "end": word.end,
        "probability": word.probability,
    }
