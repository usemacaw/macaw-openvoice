"""gRPC streaming client for runtime -> STT worker communication.

Manages bidirectional gRPC streams for real-time transcription.
Each session opens a stream with the worker, sends AudioFrames, and receives
TranscriptEvents.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import grpc
import grpc.aio

from macaw._types import TranscriptSegment, WordTimestamp
from macaw.exceptions import WorkerCrashError, WorkerTimeoutError
from macaw.logging import get_logger
from macaw.proto.stt_worker_pb2 import AudioFrame, TranscriptEvent

if TYPE_CHECKING:
    from collections.abc import AsyncIterator

    from macaw.proto.stt_worker_pb2_grpc import STTWorkerStub

logger = get_logger("scheduler.streaming")

# gRPC channel options for streaming — aggressive keepalive to detect
# worker crash via stream break in <100ms (RULE 1: P99, not average)
_GRPC_STREAMING_CHANNEL_OPTIONS = [
    ("grpc.max_send_message_length", 10 * 1024 * 1024),
    ("grpc.max_receive_message_length", 10 * 1024 * 1024),
    ("grpc.keepalive_time_ms", 10_000),
    ("grpc.keepalive_timeout_ms", 5_000),
    ("grpc.keepalive_permit_without_calls", 1),
    ("grpc.http2.min_recv_ping_interval_without_data_ms", 5_000),
    # Allow unlimited keepalive pings without data to prevent silent
    # connection death during mute-on-speak (no frames sent while TTS active).
    ("grpc.http2.max_pings_without_data", 0),
]


class StreamHandle:
    """Handle for a bidirectional gRPC stream with the worker.

    Manages sending AudioFrames and receiving TranscriptEvents
    for a streaming session.

    Typical lifecycle:
        1. open_stream() creates the StreamHandle
        2. send_frame() sends audio to the worker
        3. receive_events() consumes transcript events
        4. close() shuts down gracefully (is_last=True + done_writing)

    If the worker crashes, send_frame() and receive_events() raise
    WorkerCrashError. The runtime detects via stream break and initiates recovery.
    """

    def __init__(
        self,
        session_id: str,
        call: grpc.aio.StreamStreamCall,  # type: ignore[type-arg]
    ) -> None:
        self._session_id = session_id
        self._call = call
        self._closed = False

    @property
    def session_id(self) -> str:
        """Session ID associated with this stream."""
        return self._session_id

    @property
    def is_closed(self) -> bool:
        """True if the stream was closed (graceful or by error)."""
        return self._closed

    async def send_frame(
        self,
        pcm_data: bytes,
        initial_prompt: str | None = None,
        hot_words: list[str] | None = None,
    ) -> None:
        """Send an audio frame to the worker.

        Args:
            pcm_data: PCM 16-bit 16kHz mono bytes.
            initial_prompt: Context for conditioning (typically on first frame).
            hot_words: Words for keyword boosting.

        Raises:
            WorkerCrashError: If the stream was closed by the worker.
        """
        if self._closed:
            raise WorkerCrashError(self._session_id)

        frame = AudioFrame(
            session_id=self._session_id,
            data=pcm_data,
            is_last=False,
            initial_prompt=initial_prompt or "",
            hot_words=hot_words or [],
        )
        try:
            await self._call.write(frame)
        except grpc.aio.AioRpcError as e:
            self._closed = True
            logger.error(
                "stream_write_error",
                session_id=self._session_id,
                grpc_code=e.code().name if e.code() else "UNKNOWN",
            )
            raise WorkerCrashError(self._session_id) from e

    async def receive_events(self) -> AsyncIterator[TranscriptSegment]:
        """Receive TranscriptEvents from the worker as TranscriptSegments.

        Yields:
            TranscriptSegment converted from proto TranscriptEvent.

        Raises:
            WorkerCrashError: If the stream broke unexpectedly.
            WorkerTimeoutError: If timeout receiving events.
        """
        try:
            async for event in self._call:
                yield _proto_event_to_transcript_segment(event)
        except grpc.aio.AioRpcError as e:
            self._closed = True
            if e.code() == grpc.StatusCode.DEADLINE_EXCEEDED:
                raise WorkerTimeoutError(self._session_id, 0.0) from e
            logger.error(
                "stream_receive_error",
                session_id=self._session_id,
                grpc_code=e.code().name if e.code() else "UNKNOWN",
            )
            raise WorkerCrashError(self._session_id) from e

    async def close(self) -> None:
        """Close the stream gracefully, sending is_last=True.

        Sends an empty AudioFrame with is_last=True to signal the worker
        that there will be no more frames, and calls done_writing() to close
        the write side of the stream.

        Idempotent -- calls on already closed stream are no-op.
        """
        if self._closed:
            return

        try:
            frame = AudioFrame(
                session_id=self._session_id,
                data=b"",
                is_last=True,
            )
            await self._call.write(frame)
            await self._call.done_writing()
        except grpc.aio.AioRpcError:
            logger.debug(
                "stream_close_write_error",
                session_id=self._session_id,
            )
        finally:
            self._closed = True

    async def cancel(self) -> None:
        """Cancel the stream immediately.

        Does not wait for pending data flush. Used for fast cancellation
        (target: <=50ms per PRD).
        """
        self._closed = True
        self._call.cancel()


class StreamingGRPCClient:
    """gRPC client for bidirectional streaming with STT workers.

    Manages the gRPC channel and stream opening. A channel is reused
    for multiple streams (one session = one stream).

    Typical lifecycle:
        client = StreamingGRPCClient("localhost:50051")
        await client.connect()
        handle = await client.open_stream("sess_123")
        # ... use handle ...
        await client.close()
    """

    def __init__(self, worker_address: str) -> None:
        """Initialize the client.

        Args:
            worker_address: gRPC worker address (e.g. "localhost:50051").
        """
        self._worker_address = worker_address
        self._channel: grpc.aio.Channel | None = None
        self._stub: STTWorkerStub | None = None

    async def connect(self) -> None:
        """Open gRPC channel with the worker.

        The channel is reused for all streams opened via open_stream().
        """
        self._channel = grpc.aio.insecure_channel(
            self._worker_address,
            options=_GRPC_STREAMING_CHANNEL_OPTIONS,
        )
        from macaw.proto.stt_worker_pb2_grpc import STTWorkerStub

        self._stub = STTWorkerStub(self._channel)  # type: ignore[no-untyped-call]
        logger.info("grpc_streaming_connected", worker_address=self._worker_address)

    async def open_stream(self, session_id: str) -> StreamHandle:
        """Open a bidirectional stream for a session.

        Args:
            session_id: Streaming session ID.

        Returns:
            StreamHandle for sending/receiving messages.

        Raises:
            WorkerCrashError: If the channel is not connected or failed to open stream.
        """
        if self._stub is None:
            raise WorkerCrashError(session_id)

        try:
            call = self._stub.TranscribeStream()
            return StreamHandle(session_id=session_id, call=call)
        except grpc.aio.AioRpcError as e:
            logger.error(
                "stream_open_error",
                session_id=session_id,
                grpc_code=e.code().name if e.code() else "UNKNOWN",
            )
            raise WorkerCrashError(session_id) from e

    async def close(self) -> None:
        """Close the gRPC channel.

        All streams opened via this channel will be invalidated.
        """
        if self._channel is not None:
            await self._channel.close()
            self._channel = None
            self._stub = None
            logger.info(
                "grpc_streaming_disconnected",
                worker_address=self._worker_address,
            )


def _proto_event_to_transcript_segment(event: TranscriptEvent) -> TranscriptSegment:
    """Convert TranscriptEvent proto to Macaw TranscriptSegment.

    Pure function -- no side effects, no IO.

    Args:
        event: TranscriptEvent received from the worker via gRPC.

    Returns:
        Immutable TranscriptSegment with converted data.
    """
    words: tuple[WordTimestamp, ...] | None = None
    if event.words:
        words = tuple(
            WordTimestamp(
                word=w.word,
                start=w.start,
                end=w.end,
                probability=w.probability if w.probability != 0.0 else None,
            )
            for w in event.words
        )

    return TranscriptSegment(
        text=event.text,
        is_final=(event.event_type == "final"),
        segment_id=event.segment_id,
        start_ms=event.start_ms,
        end_ms=event.end_ms,
        language=event.language if event.language else None,
        confidence=event.confidence if event.confidence != 0.0 else None,
        words=words,
    )
