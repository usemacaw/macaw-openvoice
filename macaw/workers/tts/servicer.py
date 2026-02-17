"""gRPC servicer for the TTS worker.

Implements the TTSWorker service defined in tts_worker.proto.
Delegates synthesis to the injected TTSBackend.
"""

from __future__ import annotations

import inspect
from typing import TYPE_CHECKING

import grpc

from macaw.logging import get_logger
from macaw.proto.tts_worker_pb2 import (
    ListVoicesResponse,
)
from macaw.workers.tts.converters import (
    audio_chunk_to_proto,
    health_dict_to_proto_response,
    proto_request_to_synthesize_params,
    voices_to_proto_response,
)

if TYPE_CHECKING:
    from collections.abc import AsyncIterator

    from macaw.proto.tts_worker_pb2 import (
        HealthRequest,
        HealthResponse,
        ListVoicesRequest,
        SynthesizeChunk,
        SynthesizeRequest,
    )
    from macaw.workers.tts.interface import TTSBackend

from macaw.proto.tts_worker_pb2_grpc import TTSWorkerServicer as _BaseServicer

logger = get_logger("worker.tts.servicer")


class TTSWorkerServicer(_BaseServicer):
    """Implementation of the TTSWorker gRPC service.

    Receives gRPC requests, delegates to TTSBackend, returns proto responses.
    Synthesize is server-streaming: text in, audio chunks out.
    """

    def __init__(
        self,
        backend: TTSBackend,
        model_name: str,
        engine: str,
    ) -> None:
        self._backend = backend
        self._model_name = model_name
        self._engine = engine

    async def Synthesize(  # noqa: N802  # type: ignore[override]
        self,
        request: SynthesizeRequest,
        context: grpc.aio.ServicerContext[SynthesizeRequest, SynthesizeChunk],
    ) -> AsyncIterator[SynthesizeChunk]:
        """Speech synthesis via server-streaming.

        Receives SynthesizeRequest with text, delegates to TTSBackend.synthesize(),
        and yields SynthesizeChunk with PCM audio as the engine synthesizes.
        """
        params = proto_request_to_synthesize_params(request)
        request_id = request.request_id
        text = params.text

        if not text.strip():
            logger.warning("synthesize_empty_text", request_id=request_id)
            await context.abort(
                grpc.StatusCode.INVALID_ARGUMENT,
                "Text must not be empty",
            )
            return  # pragma: no cover

        logger.info(
            "synthesize_start",
            request_id=request_id,
            voice=params.voice,
            text_length=len(text),
        )

        try:
            # synthesize() can be:
            # 1. An async generator (uses yield) — returns AsyncGenerator directly
            # 2. An async coroutine that returns AsyncIterator — requires await
            # We use the same heuristic as STT: try async for directly,
            # and if it fails (coroutine), await first.
            result = self._backend.synthesize(
                text=text,
                voice=params.voice,
                sample_rate=params.sample_rate,
                speed=params.speed,
                options=params.options,
            )

            # If it is a coroutine (async def without yield), await to get the iterator
            stream: AsyncIterator[bytes]
            if inspect.iscoroutine(result):
                stream = await result
            else:
                stream = result  # type: ignore[assignment]

            # Create codec encoder if requested
            from macaw.codec import create_encoder

            codec_name = request.codec
            encoder = create_encoder(codec_name, params.sample_rate) if codec_name else None

            accumulated_duration = 0.0
            chunk_count = 0

            async for audio_chunk in stream:
                if context.cancelled():
                    logger.info("synthesize_cancelled", request_id=request_id)
                    return

                chunk_count += 1
                # Estimate chunk duration from PCM (before encoding)
                chunk_duration = (
                    len(audio_chunk) / (params.sample_rate * 2) if params.sample_rate > 0 else 0.0
                )
                accumulated_duration += chunk_duration

                if encoder is not None:
                    encoded = encoder.encode(audio_chunk)
                    if encoded:
                        yield audio_chunk_to_proto(
                            audio_data=encoded,
                            is_last=False,
                            duration=accumulated_duration,
                            codec=codec_name,
                        )
                else:
                    yield audio_chunk_to_proto(
                        audio_data=audio_chunk,
                        is_last=False,
                        duration=accumulated_duration,
                    )

            # Flush encoder and send final chunk
            if encoder is not None:
                flushed = encoder.flush()
                yield audio_chunk_to_proto(
                    audio_data=flushed,
                    is_last=True,
                    duration=accumulated_duration,
                    codec=codec_name,
                )
            else:
                # Send empty final chunk signaling end of stream
                yield audio_chunk_to_proto(
                    audio_data=b"",
                    is_last=True,
                    duration=accumulated_duration,
                )

        except Exception as exc:
            from macaw.exceptions import TTSSynthesisError

            logger.error(
                "synthesize_error",
                request_id=request_id,
                error=str(exc),
            )
            # Client errors (bad input, missing ref_audio, etc.) → INVALID_ARGUMENT
            # Server errors (OOM, unexpected) → INTERNAL
            status = (
                grpc.StatusCode.INVALID_ARGUMENT
                if isinstance(exc, TTSSynthesisError)
                else grpc.StatusCode.INTERNAL
            )
            await context.abort(status, str(exc))
            return  # pragma: no cover

        logger.info(
            "synthesize_done",
            request_id=request_id,
            chunks=chunk_count,
            duration=accumulated_duration,
        )

    async def ListVoices(  # noqa: N802  # type: ignore[override]
        self,
        request: ListVoicesRequest,
        context: grpc.aio.ServicerContext[ListVoicesRequest, ListVoicesResponse],
    ) -> ListVoicesResponse:
        """List available voices from the loaded TTS backend."""
        try:
            voice_list = await self._backend.voices()
        except Exception as exc:
            logger.error("list_voices_error", error=str(exc))
            await context.abort(grpc.StatusCode.INTERNAL, str(exc))
            return ListVoicesResponse()  # pragma: no cover
        return voices_to_proto_response(voice_list)

    async def Health(  # noqa: N802  # type: ignore[override]
        self,
        request: HealthRequest,
        context: grpc.aio.ServicerContext[HealthRequest, HealthResponse],
    ) -> HealthResponse:
        """Health check for the TTS worker."""
        health = await self._backend.health()
        return health_dict_to_proto_response(health, self._model_name, self._engine)
