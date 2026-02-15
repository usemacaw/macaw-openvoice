"""gRPC servicer for the STT worker.

Implements the STTWorker service defined in stt_worker.proto.
Delegates transcription to the injected STTBackend.
"""

from __future__ import annotations

import asyncio
import inspect
import threading
from typing import TYPE_CHECKING

import grpc

from macaw.logging import get_logger
from macaw.proto.stt_worker_pb2 import (
    CancelResponse,
    TranscribeFileResponse,
    TranscriptEvent,
)
from macaw.workers.stt.converters import (
    batch_result_to_proto_response,
    health_dict_to_proto_response,
    proto_request_to_transcribe_params,
    transcript_segment_to_proto_event,
)

if TYPE_CHECKING:
    from collections.abc import AsyncIterator

    from macaw._types import TranscriptSegment
    from macaw.proto.stt_worker_pb2 import (
        AudioFrame,
        CancelRequest,
        HealthRequest,
        TranscribeFileRequest,
    )
    from macaw.workers.stt.interface import STTBackend

from macaw.proto.stt_worker_pb2_grpc import STTWorkerServicer as _BaseServicer

logger = get_logger("worker.stt.servicer")


_MAX_CANCELLED_REQUESTS = 10_000


class STTWorkerServicer(_BaseServicer):
    """Implementation of the STTWorker gRPC service.

    Receives gRPC requests, delegates to STTBackend, returns proto responses.
    """

    def __init__(
        self,
        backend: STTBackend,
        model_name: str,
        engine: str,
        *,
        max_concurrent: int = 1,
    ) -> None:
        self._backend = backend
        self._model_name = model_name
        self._engine = engine
        self._max_concurrent = max(1, max_concurrent)
        self._inference_semaphore = asyncio.Semaphore(self._max_concurrent)
        self._cancel_lock = threading.Lock()
        self._cancelled_requests: set[str] = set()
        self._active_request_ids: set[str] = set()

    def is_cancelled(self, request_id: str) -> bool:
        """Check whether a request was cancelled.

        Thread-safe — can be called from executor threads during inference.
        """
        with self._cancel_lock:
            return request_id in self._cancelled_requests

    async def TranscribeFile(  # noqa: N802  # type: ignore[override]
        self,
        request: TranscribeFileRequest,
        context: grpc.aio.ServicerContext[TranscribeFileRequest, TranscribeFileResponse],
    ) -> TranscribeFileResponse:
        """Batch transcription for an audio file.

        Uses a semaphore to limit inference concurrency on the worker.
        Multiple requests can arrive via asyncio.gather() from the scheduler
        (M8-06 batch dispatch). The semaphore ensures at most
        ``max_concurrent`` inferences run in parallel.
        """
        params = proto_request_to_transcribe_params(request)
        request_id = request.request_id

        with self._cancel_lock:
            self._active_request_ids.add(request_id)

        logger.info(
            "transcribe_file_start",
            request_id=request_id,
            language=params.get("language"),
            audio_bytes=len(request.audio_data),
        )

        try:
            # Check cancellation before acquiring the semaphore
            if self.is_cancelled(request_id):
                logger.info("transcribe_file_cancelled_before_start", request_id=request_id)
                await context.abort(grpc.StatusCode.CANCELLED, "Request cancelled")
                return TranscribeFileResponse()  # pragma: no cover

            async with self._inference_semaphore:
                # Check cancellation after acquiring the semaphore (may have waited)
                if self.is_cancelled(request_id):
                    logger.info(
                        "transcribe_file_cancelled_after_semaphore",
                        request_id=request_id,
                    )
                    await context.abort(grpc.StatusCode.CANCELLED, "Request cancelled")
                    return TranscribeFileResponse()  # pragma: no cover

                result = await self._backend.transcribe_file(**params)  # type: ignore[arg-type]

            # Check cancellation after inference
            if self.is_cancelled(request_id):
                logger.info("transcribe_file_cancelled_after_inference", request_id=request_id)
                await context.abort(grpc.StatusCode.CANCELLED, "Request cancelled")
                return TranscribeFileResponse()  # pragma: no cover
        except Exception as exc:
            logger.error("transcribe_file_error", request_id=request_id, error=str(exc))
            await context.abort(grpc.StatusCode.INTERNAL, str(exc))
            return TranscribeFileResponse()  # pragma: no cover — unreachable in real gRPC
        finally:
            with self._cancel_lock:
                self._active_request_ids.discard(request_id)
                self._cancelled_requests.discard(request_id)

        logger.info(
            "transcribe_file_done",
            request_id=request_id,
            text_length=len(result.text),
            segments=len(result.segments),
        )

        return batch_result_to_proto_response(result)

    async def TranscribeStream(  # noqa: N802  # type: ignore[override]
        self,
        request_iterator: AsyncIterator[AudioFrame],
        context: grpc.aio.ServicerContext[AudioFrame, TranscriptEvent],
    ) -> AsyncIterator[TranscriptEvent]:
        """Bidirectional streaming STT transcription.

        Receives a stream of AudioFrames, delegates to the backend via
        transcribe_stream, and returns a stream of TranscriptEvents.

        Metadata (session_id, initial_prompt, hot_words) is extracted from
        the first AudioFrame before starting the backend.
        """
        # Read first frame to extract session metadata
        first_frame: AudioFrame | None = None
        async for frame in request_iterator:
            first_frame = frame
            break

        if first_frame is None:
            # Empty stream — no frame received
            return

        session_id = first_frame.session_id
        initial_prompt: str | None = (
            first_frame.initial_prompt if first_frame.initial_prompt else None
        )
        hot_words: list[str] | None = (
            list(first_frame.hot_words) if first_frame.hot_words else None
        )

        # If the first frame already signals end, there is no audio to process
        if first_frame.is_last:
            logger.info("transcribe_stream_empty", session_id=session_id)
            return

        async def audio_chunk_generator() -> AsyncIterator[bytes]:
            """Extract PCM bytes from the AudioFrame stream.

            Emits data from the first frame (already read) and then consumes
            the rest of the request_iterator.
            """
            # Emit data from the first frame
            yield bytes(first_frame.data)

            # Consume remaining frames
            async for frame in request_iterator:
                if context.cancelled():
                    return
                if frame.is_last:
                    return
                yield bytes(frame.data)

        logger.info("transcribe_stream_start", session_id=session_id)

        try:
            # transcribe_stream() can be:
            # 1. An async generator (uses yield) — returns AsyncGenerator directly
            # 2. An async coroutine that returns AsyncIterator — requires await
            # We detect coroutines and await them, matching the TTS servicer pattern.
            result = self._backend.transcribe_stream(
                audio_chunks=audio_chunk_generator(),
                language=None,
                initial_prompt=initial_prompt,
                hot_words=hot_words,
            )

            stream: AsyncIterator[TranscriptSegment]
            if inspect.iscoroutine(result):
                stream = await result
            else:
                stream = result  # type: ignore[assignment]

            async for segment in stream:
                if context.cancelled():
                    logger.info("transcribe_stream_cancelled", session_id=session_id)
                    return
                event = transcript_segment_to_proto_event(segment, session_id)
                yield event
        except Exception as exc:
            logger.error(
                "transcribe_stream_error",
                session_id=session_id,
                error=str(exc),
            )
            await context.abort(grpc.StatusCode.INTERNAL, str(exc))
            return  # pragma: no cover — unreachable in real gRPC

        logger.info("transcribe_stream_done", session_id=session_id)

    async def Cancel(  # noqa: N802  # type: ignore[override]
        self,
        request: CancelRequest,
        context: grpc.aio.ServicerContext[CancelRequest, CancelResponse],
    ) -> CancelResponse:
        """Cooperative cancellation of a running batch request.

        Sets an internal flag checked between inference segments.
        Cancellation is cooperative — it does not interrupt running CUDA kernels.

        For streaming, cancellation continues via stream break (gRPC call.cancel()).
        """
        request_id = request.request_id

        with self._cancel_lock:
            if len(self._cancelled_requests) >= _MAX_CANCELLED_REQUESTS:
                self._cancelled_requests.clear()
            self._cancelled_requests.add(request_id)
            is_current = request_id in self._active_request_ids

        logger.info(
            "cancel_received",
            request_id=request_id,
            is_current_request=is_current,
        )

        return CancelResponse(acknowledged=True)

    async def Health(  # noqa: N802  # type: ignore[override]
        self,
        request: HealthRequest,
        context: grpc.aio.ServicerContext[HealthRequest, object],
    ) -> object:
        """Worker health check."""
        health = await self._backend.health()
        return health_dict_to_proto_response(health, self._model_name, self._engine)
