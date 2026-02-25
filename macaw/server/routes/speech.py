"""POST /v1/audio/speech — speech synthesis (TTS)."""

from __future__ import annotations

import asyncio
import base64
import io
import struct
import uuid
from typing import TYPE_CHECKING, Any

import grpc.aio
from fastapi import APIRouter, Depends, Request
from fastapi.responses import Response, StreamingResponse

from macaw.codec.output_format import OutputFormat, parse_output_format
from macaw.exceptions import (
    InvalidRequestError,
    VoiceNotFoundError,
    WorkerCrashError,
    WorkerTimeoutError,
    WorkerUnavailableError,
)
from macaw.logging import get_logger
from macaw.proto.tts_worker_pb2_grpc import TTSWorkerStub
from macaw.registry.registry import ModelRegistry  # noqa: TC001
from macaw.scheduler.tts_converters import build_tts_proto_request
from macaw.server.constants import (
    DEFAULT_VOICE_NAME,
    SAVED_VOICE_PREFIX,
    TTS_DEFAULT_SAMPLE_RATE,
    TTS_GRPC_TIMEOUT,
)
from macaw.server.dependencies import get_registry, get_worker_manager
from macaw.server.grpc_channels import get_or_create_tts_channel
from macaw.server.models.speech import SpeechRequest  # noqa: TC001
from macaw.server.tts_service import resolve_tts_resources
from macaw.workers.manager import WorkerManager  # noqa: TC001

if TYPE_CHECKING:
    from collections.abc import AsyncIterator

    from macaw.audio_effects.chain import AudioEffectChain
    from macaw.proto.tts_worker_pb2 import SynthesizeChunk, SynthesizeRequest

router = APIRouter(tags=["Audio"])

logger = get_logger("server.routes.speech")


def _read_file_bytes(path: str) -> bytes:
    """Read file contents (blocking). Run via asyncio.to_thread."""
    with open(path, "rb") as f:
        return f.read()


@router.post("/v1/audio/speech")
async def create_speech(
    body: SpeechRequest,
    request: Request,
    registry: ModelRegistry = Depends(get_registry),  # noqa: B008
    worker_manager: WorkerManager = Depends(get_worker_manager),  # noqa: B008
) -> Response:
    """Synthesize audio from text.

    Compatible with OpenAI Audio API POST /v1/audio/speech.
    Returns binary audio in the body (not JSON).

    Uses StreamingResponse to reduce TTFB: audio chunks are sent
    to the client as they arrive from the TTS worker, instead of
    accumulating all chunks before responding.
    """
    request_id = str(uuid.uuid4())

    logger.info(
        "speech_request",
        request_id=request_id,
        model=body.model,
        voice=body.voice,
        text_length=len(body.input),
        response_format=body.response_format,
        output_format=body.output_format,
    )

    # Validate non-empty text (whitespace-only not caught by min_length)
    if not body.input.strip():
        raise InvalidRequestError("The 'input' field cannot be empty.")

    # Resolve output format: output_format takes precedence over response_format
    output_fmt = _resolve_output_format(body)

    # Alignment forces NDJSON output — reject incompatible formats
    # before opening any gRPC stream (fail-fast, avoid resource leaks).
    if body.include_alignment and output_fmt.codec not in ("wav", "pcm"):
        raise InvalidRequestError(
            f"include_alignment=true is not compatible with output_format='{output_fmt.codec}'. "
            "Use 'wav' or 'pcm' with alignment."
        )

    # Pre-flight: check codec availability before opening gRPC stream
    if output_fmt.needs_encoding:
        from macaw.codec import is_codec_available

        if not is_codec_available(output_fmt.codec):
            codec = output_fmt.codec
            extra = "codec" if codec == "opus" else codec
            raise InvalidRequestError(
                f"Codec '{codec}' is not available. "
                f"Install with: pip install macaw-openvoice[{extra}]"
            )

    # Resolve TTS model + worker
    _manifest, worker, worker_address = resolve_tts_resources(registry, worker_manager, body.model)

    # Decode base64 ref_audio if present
    ref_audio_bytes: bytes | None = None
    if body.ref_audio:
        try:
            ref_audio_bytes = base64.b64decode(body.ref_audio)
        except Exception as exc:
            raise InvalidRequestError(f"Invalid base64 in 'ref_audio': {exc}") from exc

    # Resolve saved voice if voice starts with "voice_"
    voice = body.voice
    ref_text = body.ref_text
    instruction = body.instruction
    language = body.language

    if voice.startswith(SAVED_VOICE_PREFIX):
        saved_voice_id = voice[len(SAVED_VOICE_PREFIX) :]
        voice_store = request.app.state.voice_store
        if voice_store is None:
            raise InvalidRequestError("VoiceStore not configured. Cannot resolve saved voice.")

        saved = await voice_store.get(saved_voice_id)
        if saved is None:
            raise VoiceNotFoundError(saved_voice_id)

        # Conflict: inline ref_audio + saved voice with ref_audio
        if ref_audio_bytes is not None and saved.ref_audio_path is not None:
            raise InvalidRequestError(
                "Cannot provide both inline ref_audio and a saved cloned voice."
            )

        # Inject saved voice params (saved values fill gaps, don't override inline)
        if saved.ref_audio_path is not None and ref_audio_bytes is None:
            ref_audio_bytes = await asyncio.to_thread(
                _read_file_bytes,
                saved.ref_audio_path,
            )
        if saved.ref_text is not None and ref_text is None:
            ref_text = saved.ref_text
        if saved.instruction is not None and instruction is None:
            instruction = saved.instruction
        if saved.language is not None and language is None:
            language = saved.language

        # Use "default" as voice for the engine (saved voice provides params)
        voice = DEFAULT_VOICE_NAME

    # Parse SSML tags if enabled (before pronunciation rules)
    ssml_result = None
    synthesis_text = body.input
    if body.enable_ssml_parsing:
        from macaw.postprocessing.ssml.parser import SSMLParseError, SSMLParser
        from macaw.workers.tts.ssml_mapper import map_ssml_directives

        try:
            parse_result = SSMLParser().parse(body.input)
        except SSMLParseError as exc:
            raise InvalidRequestError(str(exc)) from exc

        ssml_result = map_ssml_directives(
            parse_result.text,
            parse_result.directives,
            sample_rate=TTS_DEFAULT_SAMPLE_RATE,
        )
        synthesis_text = ssml_result.text
    if body.pronunciation_dictionary_locators:
        from macaw.server.pronunciation.applicator import apply_pronunciation_rules

        pron_store = getattr(request.app.state, "pronunciation_store", None)
        if pron_store is None:
            raise InvalidRequestError("PronunciationStore not configured on this server.")

        try:
            synthesis_text = await apply_pronunciation_rules(
                synthesis_text,
                body.pronunciation_dictionary_locators,
                pron_store,
            )
        except ValueError as exc:
            raise InvalidRequestError(str(exc)) from exc
        except KeyError as exc:
            raise InvalidRequestError(str(exc)) from exc

    # Apply SSML speed override (prosody/emphasis → speed multiplier)
    effective_speed = body.speed
    if ssml_result is not None and ssml_result.speed_override is not None:
        effective_speed = ssml_result.speed_override

    # Build proto request
    proto_request = build_tts_proto_request(
        request_id=request_id,
        text=synthesis_text,
        voice=voice,
        sample_rate=TTS_DEFAULT_SAMPLE_RATE,
        speed=effective_speed,
        language=language,
        ref_audio=ref_audio_bytes,
        ref_text=ref_text,
        instruction=instruction,
        include_alignment=body.include_alignment,
        alignment_granularity=body.alignment_granularity,
        seed=body.seed,
        text_normalization=body.text_normalization,
        temperature=body.temperature,
        top_k=body.top_k,
        top_p=body.top_p,
        voice_settings=body.voice_settings.model_dump() if body.voice_settings else None,
        previous_text=body.previous_text,
        next_text=body.next_text,
        previous_request_ids=body.previous_request_ids,
        next_request_ids=body.next_request_ids,
    )

    # Get pooled TTS channel (reused across requests)
    tts_channels: dict[str, grpc.aio.Channel] = request.app.state.tts_channels
    channel = get_or_create_tts_channel(tts_channels, worker_address)

    # Pre-flight: open gRPC stream and fetch first chunk.
    # This validates the connection and request params BEFORE starting
    # the StreamingResponse (so we can still return proper HTTP errors).
    response_stream, first_chunk = await _open_tts_stream(
        channel=channel,
        proto_request=proto_request,
        worker_id=worker.worker_id,
    )

    # Build effect chain from request params (None = no effects)
    effect_chain = None
    if body.effects is not None:
        from macaw.audio_effects import create_effect_chain

        effect_chain = create_effect_chain(
            pitch_shift_semitones=body.effects.pitch_shift_semitones,
            reverb_room_size=body.effects.reverb_room_size,
            reverb_damping=body.effects.reverb_damping,
            reverb_wet_dry_mix=body.effects.reverb_wet_dry_mix,
            sample_rate=TTS_DEFAULT_SAMPLE_RATE,
        )

    # Alignment mode: NDJSON response with audio + timing data per chunk.
    if body.include_alignment:
        return StreamingResponse(
            _stream_tts_audio_with_alignment(
                response_stream=response_stream,
                first_chunk=first_chunk,
                sample_rate=TTS_DEFAULT_SAMPLE_RATE,
                request_id=request_id,
                effect_chain=effect_chain,
            ),
            media_type="application/x-ndjson",
        )

    # Standard mode: binary audio stream
    first_audio_chunk = first_chunk.audio_data if first_chunk else b""

    return StreamingResponse(
        _stream_tts_audio(
            response_stream=response_stream,
            first_audio_chunk=first_audio_chunk,
            output_fmt=output_fmt,
            source_sample_rate=TTS_DEFAULT_SAMPLE_RATE,
            request_id=request_id,
            effect_chain=effect_chain,
        ),
        media_type=output_fmt.content_type,
    )


async def _open_tts_stream(
    *,
    channel: grpc.aio.Channel,
    proto_request: SynthesizeRequest,
    worker_id: str,
) -> tuple[grpc.aio.UnaryStreamCall[Any, Any], SynthesizeChunk | None]:
    """Open gRPC stream and fetch first chunk (pre-flight validation).

    Uses a pooled channel (not closed on error — gRPC channels handle
    reconnection automatically).

    Returns the response stream iterator and the first SynthesizeChunk
    proto (or None if the stream is empty). The caller extracts
    .audio_data and optionally .alignment from the first chunk.

    Raises domain exceptions on gRPC errors so the HTTP layer can return
    proper status codes before the StreamingResponse starts.
    """
    try:
        stub = TTSWorkerStub(channel)  # type: ignore[no-untyped-call]
        response_stream = stub.Synthesize(proto_request, timeout=TTS_GRPC_TIMEOUT)

        # Read chunks until we get one with audio_data (pre-flight)
        first_chunk: SynthesizeChunk | None = None
        async for chunk in response_stream:
            if chunk.audio_data:
                first_chunk = chunk
                break
            if chunk.is_last:
                break

        return response_stream, first_chunk

    except grpc.aio.AioRpcError as exc:
        code = exc.code()
        if code == grpc.StatusCode.DEADLINE_EXCEEDED:
            raise WorkerTimeoutError(worker_id, TTS_GRPC_TIMEOUT) from exc
        if code == grpc.StatusCode.UNAVAILABLE:
            raise WorkerUnavailableError("tts") from exc
        if code == grpc.StatusCode.INVALID_ARGUMENT:
            detail = exc.details() or "Synthesis failed"
            raise InvalidRequestError(detail) from exc
        raise WorkerCrashError(worker_id) from exc


async def _stream_tts_audio(
    *,
    response_stream: grpc.aio.UnaryStreamCall[Any, Any],
    first_audio_chunk: bytes,
    output_fmt: OutputFormat,
    source_sample_rate: int,
    request_id: str,
    effect_chain: AudioEffectChain | None = None,
) -> AsyncIterator[bytes]:
    """Async generator that yields TTS audio chunks for StreamingResponse.

    For WAV format: yields a WAV header (with max data size placeholder)
    followed by raw PCM chunks. For encoded formats (opus, mp3, mulaw, alaw):
    resamples PCM to target sample rate (if needed) and encodes via
    CodecEncoder. For PCM format: resamples and yields raw PCM directly.

    The gRPC channel is pooled and NOT closed here — it is reused across
    requests and closed on server shutdown via close_tts_channels().
    """
    from macaw.codec import create_encoder

    encoder = None
    if output_fmt.needs_encoding:
        encoder = create_encoder(
            output_fmt.codec,
            sample_rate=output_fmt.sample_rate,
            bitrate=output_fmt.bitrate_bps,
        )

    # Determine if PCM resampling is needed before encoding
    needs_resample = output_fmt.sample_rate != source_sample_rate

    total_audio_bytes = 0
    try:
        # WAV header with streaming-compatible max size placeholder
        if output_fmt.codec == "wav":
            yield _wav_streaming_header(output_fmt.sample_rate)

        # Yield pre-fetched first chunk
        if first_audio_chunk:
            audio_data = first_audio_chunk
            total_audio_bytes += len(audio_data)
            if effect_chain is not None:
                audio_data = effect_chain.process_bytes(audio_data, source_sample_rate)
            if needs_resample:
                audio_data = _resample_chunk(
                    audio_data, source_sample_rate, output_fmt.sample_rate
                )
            if encoder is not None:
                encoded = encoder.encode(audio_data)
                if encoded:
                    yield encoded
            else:
                yield audio_data

        # Stream remaining chunks
        async for chunk in response_stream:
            if chunk.audio_data:
                audio_data = chunk.audio_data
                total_audio_bytes += len(audio_data)
                if effect_chain is not None:
                    audio_data = effect_chain.process_bytes(audio_data, source_sample_rate)
                if needs_resample:
                    audio_data = _resample_chunk(
                        audio_data, source_sample_rate, output_fmt.sample_rate
                    )
                if encoder is not None:
                    encoded = encoder.encode(audio_data)
                    if encoded:
                        yield encoded
                else:
                    yield audio_data
            if chunk.is_last:
                break

        # Flush remaining encoder buffer
        if encoder is not None:
            flushed = encoder.flush()
            if flushed:
                yield flushed

        logger.info(
            "speech_done",
            request_id=request_id,
            audio_bytes=total_audio_bytes,
        )

    except grpc.aio.AioRpcError as exc:
        # Error after streaming started — can't change HTTP status.
        # Log and let the client detect truncation.
        logger.error(
            "tts_stream_error_mid_response",
            request_id=request_id,
            grpc_code=str(exc.code()),
            audio_bytes_sent=total_audio_bytes,
        )
    except Exception:
        logger.exception(
            "tts_stream_unexpected_error",
            request_id=request_id,
            audio_bytes_sent=total_audio_bytes,
        )


async def _stream_tts_audio_with_alignment(
    *,
    response_stream: grpc.aio.UnaryStreamCall[Any, Any],
    first_chunk: SynthesizeChunk | None,
    sample_rate: int,
    request_id: str,
    effect_chain: AudioEffectChain | None = None,
) -> AsyncIterator[bytes]:
    """Async generator that yields NDJSON lines with audio + alignment.

    Each line is a JSON object encoded as UTF-8 followed by newline.
    Audio data is base64-encoded. Alignment data (when present) includes
    per-word/character timing information.

    Format:
        {"type":"audio","audio":"<base64>","alignment":{...}}
        {"type":"audio","audio":"<base64>"}
        {"type":"done","duration":1.5}
    """
    from macaw.server.models.alignment import (
        AlignmentItemResponse,
        AlignmentStreamDone,
        AudioChunkWithAlignment,
        ChunkAlignmentResponse,
    )

    accumulated_duration = 0.0
    total_audio_bytes = 0
    has_alignment_data = False

    try:

        def _chunk_to_ndjson(chunk: SynthesizeChunk) -> bytes | None:
            """Convert a SynthesizeChunk proto to an NDJSON line (bytes)."""
            nonlocal accumulated_duration, total_audio_bytes, has_alignment_data

            if not chunk.audio_data:
                return None

            audio_data = chunk.audio_data
            total_audio_bytes += len(audio_data)

            # Apply effects if configured
            if effect_chain is not None:
                audio_data = effect_chain.process_bytes(audio_data, sample_rate)

            # Estimate duration from PCM
            chunk_duration = len(chunk.audio_data) / (sample_rate * 2) if sample_rate > 0 else 0.0
            accumulated_duration += chunk_duration

            # Build alignment response if present
            alignment_resp = None
            if chunk.alignment and chunk.alignment.items:
                has_alignment_data = True
                raw_gran = chunk.alignment.granularity or "word"
                gran = raw_gran if raw_gran in ("word", "character") else "word"
                alignment_resp = ChunkAlignmentResponse(
                    items=[
                        AlignmentItemResponse(
                            text=item.text,
                            start_ms=item.start_ms,
                            duration_ms=item.duration_ms,
                        )
                        for item in chunk.alignment.items
                    ],
                    granularity=gran,  # type: ignore[arg-type]
                )

            # Build normalized alignment response if present
            norm_alignment_resp = None
            if chunk.normalized_alignment and chunk.normalized_alignment.items:
                raw_ngran = chunk.normalized_alignment.granularity or "word"
                ngran = raw_ngran if raw_ngran in ("word", "character") else "word"
                norm_alignment_resp = ChunkAlignmentResponse(
                    items=[
                        AlignmentItemResponse(
                            text=item.text,
                            start_ms=item.start_ms,
                            duration_ms=item.duration_ms,
                        )
                        for item in chunk.normalized_alignment.items
                    ],
                    granularity=ngran,  # type: ignore[arg-type]
                )

            line = AudioChunkWithAlignment(
                audio=base64.b64encode(audio_data).decode("ascii"),
                alignment=alignment_resp,
                normalized_alignment=norm_alignment_resp,
            )
            return line.model_dump_json(exclude_none=True).encode("utf-8") + b"\n"

        # Yield pre-fetched first chunk
        if first_chunk is not None:
            ndjson_line = _chunk_to_ndjson(first_chunk)
            if ndjson_line:
                yield ndjson_line

        # Stream remaining chunks
        async for chunk in response_stream:
            if chunk.audio_data:
                ndjson_line = _chunk_to_ndjson(chunk)
                if ndjson_line:
                    yield ndjson_line
            if chunk.is_last:
                break

        # Done marker with alignment availability feedback
        done_line = AlignmentStreamDone(
            duration=accumulated_duration,
            alignment_available=has_alignment_data,
        )
        yield done_line.model_dump_json().encode("utf-8") + b"\n"

        logger.info(
            "speech_alignment_done",
            request_id=request_id,
            audio_bytes=total_audio_bytes,
            duration=accumulated_duration,
        )

    except grpc.aio.AioRpcError as exc:
        logger.error(
            "tts_alignment_stream_error",
            request_id=request_id,
            grpc_code=str(exc.code()),
            audio_bytes_sent=total_audio_bytes,
        )
    except Exception:
        logger.exception(
            "tts_alignment_stream_unexpected_error",
            request_id=request_id,
            audio_bytes_sent=total_audio_bytes,
        )


def _wav_streaming_header(sample_rate: int) -> bytes:
    """Build a WAV header for streaming (unknown total size).

    Uses 0x7FFFFFFF as data_size — a well-known convention for
    streaming WAV that most audio players handle correctly.
    """
    num_channels = 1
    bits_per_sample = 16
    byte_rate = sample_rate * num_channels * bits_per_sample // 8
    block_align = num_channels * bits_per_sample // 8
    # Max size placeholder for streaming
    data_size = 0x7FFFFFFF

    buf = io.BytesIO()
    # RIFF header
    buf.write(b"RIFF")
    buf.write(struct.pack("<I", 36 + data_size))
    buf.write(b"WAVE")
    # fmt subchunk
    buf.write(b"fmt ")
    buf.write(struct.pack("<I", 16))
    buf.write(struct.pack("<H", 1))  # PCM format
    buf.write(struct.pack("<H", num_channels))
    buf.write(struct.pack("<I", sample_rate))
    buf.write(struct.pack("<I", byte_rate))
    buf.write(struct.pack("<H", block_align))
    buf.write(struct.pack("<H", bits_per_sample))
    # data subchunk
    buf.write(b"data")
    buf.write(struct.pack("<I", data_size))

    return buf.getvalue()


def _resolve_output_format(body: SpeechRequest) -> OutputFormat:
    """Resolve the output format from the request body.

    ``output_format`` (compound string) takes precedence.
    Falls back to ``response_format`` (simple codec name).

    Raises:
        InvalidRequestError: If the format string is invalid.
    """
    if body.output_format is not None:
        try:
            return parse_output_format(body.output_format)
        except ValueError as exc:
            raise InvalidRequestError(str(exc)) from exc
    # Backward compat: simple response_format → OutputFormat with defaults
    return parse_output_format(body.response_format)


def _resample_chunk(pcm_data: bytes, from_rate: int, to_rate: int) -> bytes:
    """Resample a PCM chunk from one sample rate to another.

    Returns the data unchanged when rates match.
    """
    from macaw._dsp import resample_pcm_bytes

    return resample_pcm_bytes(pcm_data, from_rate, to_rate)
