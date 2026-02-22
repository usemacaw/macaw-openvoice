"""gRPC servicer for the TTS worker.

Implements the TTSWorker service defined in tts_worker.proto.
Delegates synthesis to the injected TTSBackend.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import grpc

from macaw.logging import get_logger
from macaw.proto.tts_worker_pb2 import (
    ListVoicesResponse,
)
from macaw.workers.tts.converters import (
    SynthesizeParams,
    audio_chunk_to_proto,
    health_dict_to_proto_response,
    proto_request_to_synthesize_params,
    voices_to_proto_response,
)
from macaw.workers.tts.interface import _resolve_stream

if TYPE_CHECKING:
    from collections.abc import AsyncIterator

    from macaw._types import TTSChunkResult
    from macaw.alignment.interface import Aligner
    from macaw.codec.interface import CodecEncoder
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


def _encode_and_build_chunk(
    audio_data: bytes,
    encoder: CodecEncoder | None,
    codec_name: str,
    duration: float,
    alignment: tuple[object, ...] | None = None,
    normalized_alignment: tuple[object, ...] | None = None,
    alignment_granularity: str = "word",
) -> SynthesizeChunk | None:
    """Encode audio (if codec) and build a SynthesizeChunk proto.

    Returns None when the encoder absorbs the chunk (frame buffering).
    """
    if encoder is not None:
        encoded = encoder.encode(audio_data)
        if not encoded:
            return None
        return audio_chunk_to_proto(
            audio_data=encoded,
            is_last=False,
            duration=duration,
            codec=codec_name,
            alignment=alignment,  # type: ignore[arg-type]
            normalized_alignment=normalized_alignment,  # type: ignore[arg-type]
            alignment_granularity=alignment_granularity,  # type: ignore[arg-type]
        )
    return audio_chunk_to_proto(
        audio_data=audio_data,
        is_last=False,
        duration=duration,
        alignment=alignment,  # type: ignore[arg-type]
        normalized_alignment=normalized_alignment,  # type: ignore[arg-type]
        alignment_granularity=alignment_granularity,  # type: ignore[arg-type]
    )


class TTSWorkerServicer(_BaseServicer):
    """Implementation of the TTSWorker gRPC service.

    Receives gRPC requests, delegates to TTSBackend, returns proto responses.
    Synthesize is server-streaming: text in, audio chunks out.

    The ``Synthesize`` RPC dispatches to one of three private methods based
    on request parameters and engine capabilities:

    - ``_synthesize_standard`` — no alignment, direct streaming
    - ``_synthesize_native_alignment`` — engine provides timing data inline
    - ``_synthesize_forced_alignment`` — CTC fallback (full-buffer, then align)
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
        self._aligner: Aligner | None = None
        self._aligner_checked: bool = False

    def _get_aligner(self) -> Aligner | None:
        """Get or create the forced alignment backend (lazy).

        The aligner is created on first use and cached for subsequent
        requests.  Returns ``None`` when torchaudio is not installed.
        """
        if not self._aligner_checked:
            from macaw.alignment import create_aligner

            self._aligner = create_aligner()
            self._aligner_checked = True
            if self._aligner is not None:
                logger.info("forced_aligner_ready")
            else:
                logger.info("forced_aligner_unavailable")
        return self._aligner

    async def _accumulate_audio(
        self,
        params: SynthesizeParams,
        context: grpc.aio.ServicerContext[SynthesizeRequest, SynthesizeChunk],
        request_id: str,
    ) -> tuple[bytearray, int, float]:
        """Run standard synthesis and accumulate all audio into a buffer.

        Returns:
            Tuple of (accumulated_audio, chunk_count, accumulated_duration).
        """
        accumulated = bytearray()
        chunk_count = 0
        accumulated_duration = 0.0

        result = self._backend.synthesize(
            text=params.text,
            voice=params.voice,
            sample_rate=params.sample_rate,
            speed=params.speed,
            options=params.options,
        )

        stream: AsyncIterator[bytes] = await _resolve_stream(result)

        async for audio_chunk in stream:
            if context.cancelled():
                logger.info("synthesize_cancelled", request_id=request_id)
                break
            accumulated.extend(audio_chunk)
            chunk_count += 1
            chunk_duration = (
                len(audio_chunk) / (params.sample_rate * 2) if params.sample_rate > 0 else 0.0
            )
            accumulated_duration += chunk_duration

        return accumulated, chunk_count, accumulated_duration

    async def _synthesize_forced_alignment(
        self,
        params: SynthesizeParams,
        context: grpc.aio.ServicerContext[SynthesizeRequest, SynthesizeChunk],
        request_id: str,
        encoder: CodecEncoder | None,
        codec_name: str,
    ) -> AsyncIterator[SynthesizeChunk]:
        """Forced alignment fallback (full-buffer, then CTC align).

        TRADE-OFF: CTC-based forced alignment (wav2vec2) requires the complete
        audio waveform, so this path accumulates ALL synthesized audio before
        emitting any chunks.  For long texts this eliminates streaming latency
        benefits — the client receives zero audio until synthesis + alignment
        complete.  Native alignment (Kokoro) does NOT have this limitation.
        See ADR-009 for the design rationale.
        """
        logger.info(
            "forced_alignment_full_buffer",
            request_id=request_id,
            text_length=len(params.text),
            engine=type(self._backend).__name__,
        )
        aligner = self._get_aligner()
        accumulated_audio, _chunk_count, accumulated_duration = await self._accumulate_audio(
            params,
            context,
            request_id,
        )

        alignment = None
        if aligner is not None and accumulated_audio:
            language = "en"
            if params.options and params.options.get("language"):
                language = str(params.options["language"])
            try:
                alignment = await aligner.align(
                    audio=bytes(accumulated_audio),
                    text=params.text,
                    sample_rate=params.sample_rate,
                    language=language,
                    granularity=params.alignment_granularity,
                )
            except Exception as align_exc:
                logger.warning(
                    "forced_alignment_failed",
                    request_id=request_id,
                    error=str(align_exc),
                )

        # Re-yield accumulated audio in chunks with alignment on the first chunk.
        from macaw.workers.tts.audio_utils import CHUNK_SIZE_BYTES

        audio_bytes = bytes(accumulated_audio)
        first_yielded = False
        for i in range(0, max(len(audio_bytes), 1), CHUNK_SIZE_BYTES):
            chunk_data = audio_bytes[i : i + CHUNK_SIZE_BYTES]
            if not chunk_data:
                break
            chunk_alignment = alignment if not first_yielded else None
            first_yielded = True
            proto_chunk = _encode_and_build_chunk(
                chunk_data,
                encoder,
                codec_name,
                accumulated_duration,
                alignment=chunk_alignment,
                alignment_granularity=params.alignment_granularity,
            )
            if proto_chunk is not None:
                yield proto_chunk

    async def _synthesize_native_alignment(
        self,
        params: SynthesizeParams,
        context: grpc.aio.ServicerContext[SynthesizeRequest, SynthesizeChunk],
        request_id: str,
        encoder: CodecEncoder | None,
        codec_name: str,
    ) -> AsyncIterator[SynthesizeChunk]:
        """Native alignment path — engine provides timing data inline."""
        align_result = self._backend.synthesize_with_alignment(
            text=params.text,
            voice=params.voice,
            sample_rate=params.sample_rate,
            speed=params.speed,
            alignment_granularity=params.alignment_granularity,
            options=params.options,
        )

        align_stream: AsyncIterator[TTSChunkResult] = await _resolve_stream(
            align_result,
        )

        accumulated_duration = 0.0

        async for chunk_result in align_stream:
            if context.cancelled():
                logger.info("synthesize_cancelled", request_id=request_id)
                return

            chunk_duration = (
                len(chunk_result.audio) / (params.sample_rate * 2)
                if params.sample_rate > 0
                else 0.0
            )
            accumulated_duration += chunk_duration

            proto_chunk = _encode_and_build_chunk(
                chunk_result.audio,
                encoder,
                codec_name,
                accumulated_duration,
                alignment=chunk_result.alignment,
                normalized_alignment=chunk_result.normalized_alignment,
                alignment_granularity=chunk_result.alignment_granularity,
            )
            if proto_chunk is not None:
                yield proto_chunk

    async def _synthesize_standard(
        self,
        params: SynthesizeParams,
        context: grpc.aio.ServicerContext[SynthesizeRequest, SynthesizeChunk],
        request_id: str,
        encoder: CodecEncoder | None,
        codec_name: str,
    ) -> AsyncIterator[SynthesizeChunk]:
        """Standard synthesis — no alignment, direct streaming."""
        result = self._backend.synthesize(
            text=params.text,
            voice=params.voice,
            sample_rate=params.sample_rate,
            speed=params.speed,
            options=params.options,
        )

        stream: AsyncIterator[bytes] = await _resolve_stream(result)

        accumulated_duration = 0.0

        async for audio_chunk in stream:
            if context.cancelled():
                logger.info("synthesize_cancelled", request_id=request_id)
                return

            chunk_duration = (
                len(audio_chunk) / (params.sample_rate * 2) if params.sample_rate > 0 else 0.0
            )
            accumulated_duration += chunk_duration

            proto_chunk = _encode_and_build_chunk(
                audio_chunk,
                encoder,
                codec_name,
                accumulated_duration,
            )
            if proto_chunk is not None:
                yield proto_chunk

    async def Synthesize(  # noqa: N802  # type: ignore[override]
        self,
        request: SynthesizeRequest,
        context: grpc.aio.ServicerContext[SynthesizeRequest, SynthesizeChunk],
    ) -> AsyncIterator[SynthesizeChunk]:
        """Speech synthesis via server-streaming.

        Routes to one of three synthesis strategies based on
        ``include_alignment`` and engine capabilities, then flushes the
        codec encoder (if any) and yields the final chunk.
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
            # Create codec encoder if requested (shared by all paths).
            from macaw.codec import create_encoder

            codec_name = request.codec
            if codec_name:
                from macaw.config.settings import get_settings

                bitrate = get_settings().codec.opus_bitrate
                encoder = create_encoder(codec_name, params.sample_rate, bitrate=bitrate)
            else:
                encoder = None

            accumulated_duration = 0.0
            chunk_count = 0

            # Select synthesis strategy.
            if params.include_alignment:
                caps = await self._backend.capabilities()
                if not caps.supports_alignment:
                    strategy = self._synthesize_forced_alignment(
                        params,
                        context,
                        request_id,
                        encoder,
                        codec_name,
                    )
                else:
                    strategy = self._synthesize_native_alignment(
                        params,
                        context,
                        request_id,
                        encoder,
                        codec_name,
                    )
            else:
                strategy = self._synthesize_standard(
                    params,
                    context,
                    request_id,
                    encoder,
                    codec_name,
                )

            async for chunk in strategy:
                chunk_count += 1
                accumulated_duration = chunk.duration
                yield chunk

            # If the stream was cancelled mid-flight, skip flush/final.
            if context.cancelled():
                return

            # Flush encoder and send final chunk (shared).
            if encoder is not None:
                flushed = encoder.flush()
                yield audio_chunk_to_proto(
                    audio_data=flushed,
                    is_last=True,
                    duration=accumulated_duration,
                    codec=codec_name,
                )
            else:
                yield audio_chunk_to_proto(
                    audio_data=b"",
                    is_last=True,
                    duration=accumulated_duration,
                )

        except Exception as exc:
            from macaw.exceptions import CodecUnavailableError, TTSSynthesisError

            logger.error(
                "synthesize_error",
                request_id=request_id,
                error=str(exc),
            )
            # Client errors (bad input, missing ref_audio, unavailable codec) → INVALID_ARGUMENT
            # Server errors (OOM, unexpected) → INTERNAL
            status = (
                grpc.StatusCode.INVALID_ARGUMENT
                if isinstance(exc, TTSSynthesisError | CodecUnavailableError)
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
