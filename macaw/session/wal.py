"""WAL In-Memory -- Write-Ahead Log for session recovery.

Records checkpoints after each transcript.final emission. Enables recovery
without duplication after worker crash.

Deliberately simple: in-memory, single record, overwrite.
Not an append-only log -- it's a pointer to "where we are".
The WAL is consulted during recovery (M6-07) to determine the last
confirmed segment, ring buffer offset, and timestamp.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class WALCheckpoint:
    """Checkpoint recorded in the WAL.

    Represents the confirmed session state at the moment
    a transcript.final was emitted to the client.

    Attributes:
        segment_id: ID of the confirmed segment.
        buffer_offset: Absolute offset in the Ring Buffer up to where audio
            was processed (total_written at commit time).
        timestamp_ms: Monotonic timestamp in ms at the time of the checkpoint.
    """

    segment_id: int
    buffer_offset: int
    timestamp_ms: int


class SessionWAL:
    """In-memory Write-Ahead Log for streaming sessions.

    Records checkpoints after each transcript.final emission.
    Enables recovery without duplication after worker crash.

    Deliberately simple: in-memory, single record, overwrite.
    Not an append-only log -- it's a pointer to "where we are".

    Thread-safety: not required -- single-threaded in the asyncio event loop,
    same guarantee as RingBuffer.
    """

    __slots__ = ("_checkpoint",)

    def __init__(self) -> None:
        self._checkpoint = WALCheckpoint(segment_id=0, buffer_offset=0, timestamp_ms=0)

    @property
    def last_committed_segment_id(self) -> int:
        """ID of the last confirmed segment."""
        return self._checkpoint.segment_id

    @property
    def last_committed_buffer_offset(self) -> int:
        """Ring Buffer offset of the last commit."""
        return self._checkpoint.buffer_offset

    @property
    def last_committed_timestamp_ms(self) -> int:
        """Monotonic timestamp in ms of the last commit."""
        return self._checkpoint.timestamp_ms

    @property
    def checkpoint(self) -> WALCheckpoint:
        """Current checkpoint (last recorded)."""
        return self._checkpoint

    def record_checkpoint(
        self,
        segment_id: int,
        buffer_offset: int,
        timestamp_ms: int,
    ) -> None:
        """Record atomic checkpoint after transcript.final.

        Each checkpoint overwrites the previous (in-memory WAL, not append-only).
        Atomicity is guaranteed by reference assignment in Python
        (single assignment in the event loop, no threading).

        Args:
            segment_id: ID of the confirmed segment.
            buffer_offset: Ring Buffer offset up to where audio was processed.
            timestamp_ms: Monotonic timestamp in ms at the checkpoint moment.
        """
        self._checkpoint = WALCheckpoint(
            segment_id=segment_id,
            buffer_offset=buffer_offset,
            timestamp_ms=timestamp_ms,
        )
