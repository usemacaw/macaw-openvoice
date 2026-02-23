"""POST /v1/text-to-dialogue — multi-speaker dialogue synthesis."""

from __future__ import annotations

import asyncio
import io
import struct
import uuid
from typing import TYPE_CHECKING

import grpc.aio
from fastapi import APIRouter, Depends, Request
from fastapi.responses import StreamingResponse

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
from macaw.server.models.dialogue import (
    MAX_DIALOGUE_VOICES,
    DialogueRequest,
)
from macaw.server.tts_service import resolve_tts_resources
from macaw.workers.manager import WorkerManager  # noqa: TC001

if TYPE_CHECKING:
    from collections.abc import AsyncIterator

router = APIRouter(tags=["Dialogue"])

logger = get_logger("server.routes.dialogue")


def _read_file_bytes(path: str) -> bytes:
    """Read file contents (blocking). Run via asyncio.to_thread."""
    with open(path, "rb") as f:
        return f.read()


@router.post("/v1/text-to-dialogue")
async def create_dialogue(
    body: DialogueRequest,
    request: Request,
    registry: ModelRegistry = Depends(get_registry),  # noqa: B008
    worker_manager: WorkerManager = Depends(get_worker_manager),  # noqa: B008
) -> StreamingResponse:
    """Synthesize multi-speaker dialogue.

    Accepts an array of (text, voice_id) segments, synthesizes each
    sequentially with the correct voice, and streams concatenated audio.
    """
    request_id = str(uuid.uuid4())

    logger.info(
        "dialogue_request",
        request_id=request_id,
        model=body.model,
        segments=len(body.inputs),
    )

    # Validate: max unique voices
    unique_voices = {seg.voice_id for seg in body.inputs}
    if len(unique_voices) > MAX_DIALOGUE_VOICES:
        raise InvalidRequestError(
            f"Too many unique voices ({len(unique_voices)}). Maximum is {MAX_DIALOGUE_VOICES}."
        )

    # Validate non-empty text per segment
    for i, seg in enumerate(body.inputs):
        if not seg.text.strip():
            raise InvalidRequestError(f"Segment {i}: 'text' cannot be empty or whitespace-only.")

    # Resolve output format
    output_fmt = _resolve_output_format(body)

    # Pre-flight: check codec availability
    if output_fmt.needs_encoding:
        from macaw.codec import is_codec_available

        if not is_codec_available(output_fmt.codec):
            raise InvalidRequestError(
                f"Codec '{output_fmt.codec}' is not available. Install the required library."
            )

    # Resolve TTS model + worker (all segments share the same model/worker)
    _manifest, worker, worker_address = resolve_tts_resources(
        registry,
        worker_manager,
        body.model,
    )

    # Get pooled TTS channel
    tts_channels: dict[str, grpc.aio.Channel] = request.app.state.tts_channels
    channel = get_or_create_tts_channel(tts_channels, worker_address)

    # Resolve voices for all segments up front (fail-fast before streaming)
    resolved_voices = await _resolve_all_voices(body, request)

    return StreamingResponse(
        _stream_dialogue(
            channel=channel,
            body=body,
            resolved_voices=resolved_voices,
            output_fmt=output_fmt,
            worker_id=worker.worker_id,
            request_id=request_id,
        ),
        media_type=output_fmt.content_type,
    )


async def _resolve_all_voices(
    body: DialogueRequest,
    request: Request,
) -> list[_ResolvedVoice]:
    """Resolve all voice_ids before streaming begins (fail-fast).

    Returns a list of ResolvedVoice objects (one per segment) containing
    the resolved voice name and any saved-voice parameters.
    """
    voice_store = getattr(request.app.state, "voice_store", None)
    results: list[_ResolvedVoice] = []

    for i, seg in enumerate(body.inputs):
        if seg.voice_id.startswith(SAVED_VOICE_PREFIX):
            saved_voice_id = seg.voice_id[len(SAVED_VOICE_PREFIX) :]

            if voice_store is None:
                raise InvalidRequestError(
                    f"Segment {i}: VoiceStore not configured. Cannot resolve saved voice."
                )

            saved = await voice_store.get(saved_voice_id)
            if saved is None:
                raise VoiceNotFoundError(saved_voice_id)

            # Load ref_audio bytes if present
            ref_audio_bytes: bytes | None = None
            if saved.ref_audio_path is not None:
                ref_audio_bytes = await asyncio.to_thread(
                    _read_file_bytes,
                    saved.ref_audio_path,
                )

            results.append(
                _ResolvedVoice(
                    voice=DEFAULT_VOICE_NAME,
                    ref_audio=ref_audio_bytes,
                    ref_text=saved.ref_text,
                    instruction=saved.instruction,
                    language=saved.language,
                )
            )
        else:
            results.append(_ResolvedVoice(voice=seg.voice_id))

    return results


class _ResolvedVoice:
    """Resolved voice parameters for a dialogue segment."""

    __slots__ = ("instruction", "language", "ref_audio", "ref_text", "voice")

    def __init__(
        self,
        voice: str,
        ref_audio: bytes | None = None,
        ref_text: str | None = None,
        instruction: str | None = None,
        language: str | None = None,
    ) -> None:
        self.voice = voice
        self.ref_audio = ref_audio
        self.ref_text = ref_text
        self.instruction = instruction
        self.language = language


async def _stream_dialogue(
    *,
    channel: grpc.aio.Channel,
    body: DialogueRequest,
    resolved_voices: list[_ResolvedVoice],
    output_fmt: OutputFormat,
    worker_id: str,
    request_id: str,
) -> AsyncIterator[bytes]:
    """Stream concatenated audio from sequential TTS synthesis per segment.

    Emits a WAV header first (for WAV format), then PCM chunks from each
    segment in order. For encoded formats, encodes via codec encoder.
    """
    from macaw.codec import create_encoder

    encoder = None
    if output_fmt.needs_encoding:
        encoder = create_encoder(
            output_fmt.codec,
            sample_rate=output_fmt.sample_rate,
            bitrate=output_fmt.bitrate_bps,
        )

    needs_resample = output_fmt.sample_rate != TTS_DEFAULT_SAMPLE_RATE
    total_audio_bytes = 0

    try:
        # WAV header with streaming max-size placeholder
        if output_fmt.codec == "wav":
            yield _wav_streaming_header(output_fmt.sample_rate)

        for seg_idx, seg in enumerate(body.inputs):
            rv = resolved_voices[seg_idx]

            # Use segment-level language override or global language
            seg_language = rv.language or body.language

            proto_request = build_tts_proto_request(
                request_id=f"{request_id}_seg{seg_idx}",
                text=seg.text,
                voice=rv.voice,
                sample_rate=TTS_DEFAULT_SAMPLE_RATE,
                speed=body.speed,
                language=seg_language,
                ref_audio=rv.ref_audio,
                ref_text=rv.ref_text,
                instruction=rv.instruction,
            )

            stub = TTSWorkerStub(channel)  # type: ignore[no-untyped-call]

            try:
                response_stream = stub.Synthesize(
                    proto_request,
                    timeout=TTS_GRPC_TIMEOUT,
                )
                async for chunk in response_stream:
                    if chunk.audio_data:
                        audio_data: bytes = chunk.audio_data
                        total_audio_bytes += len(audio_data)
                        if needs_resample:
                            audio_data = _resample_chunk(
                                audio_data,
                                TTS_DEFAULT_SAMPLE_RATE,
                                output_fmt.sample_rate,
                            )
                        if encoder is not None:
                            encoded = encoder.encode(audio_data)
                            if encoded:
                                yield encoded
                        else:
                            yield audio_data
                    if chunk.is_last:
                        break
            except grpc.aio.AioRpcError as exc:
                code = exc.code()
                if code == grpc.StatusCode.DEADLINE_EXCEEDED:
                    raise WorkerTimeoutError(
                        worker_id,
                        TTS_GRPC_TIMEOUT,
                    ) from exc
                if code == grpc.StatusCode.UNAVAILABLE:
                    raise WorkerUnavailableError("tts") from exc
                if code == grpc.StatusCode.INVALID_ARGUMENT:
                    detail = exc.details() or "Synthesis failed"
                    raise InvalidRequestError(
                        f"Segment {seg_idx}: {detail}",
                    ) from exc
                raise WorkerCrashError(worker_id) from exc

            logger.debug(
                "dialogue_segment_done",
                request_id=request_id,
                segment=seg_idx,
                voice=rv.voice,
            )

        # Flush remaining encoder buffer
        if encoder is not None:
            flushed = encoder.flush()
            if flushed:
                yield flushed

        logger.info(
            "dialogue_done",
            request_id=request_id,
            segments=len(body.inputs),
            audio_bytes=total_audio_bytes,
        )

    except (WorkerTimeoutError, WorkerUnavailableError, WorkerCrashError):
        # Re-raise domain exceptions for HTTP error handler
        # (only works if raised before first yield)
        raise
    except grpc.aio.AioRpcError as exc:
        logger.error(
            "dialogue_stream_error",
            request_id=request_id,
            grpc_code=str(exc.code()),
            audio_bytes_sent=total_audio_bytes,
        )
    except Exception:
        logger.exception(
            "dialogue_stream_unexpected_error",
            request_id=request_id,
            audio_bytes_sent=total_audio_bytes,
        )


def _wav_streaming_header(sample_rate: int) -> bytes:
    """Build a WAV header for streaming (unknown total size)."""
    num_channels = 1
    bits_per_sample = 16
    byte_rate = sample_rate * num_channels * bits_per_sample // 8
    block_align = num_channels * bits_per_sample // 8
    data_size = 0x7FFFFFFF

    buf = io.BytesIO()
    buf.write(b"RIFF")
    buf.write(struct.pack("<I", 36 + data_size))
    buf.write(b"WAVE")
    buf.write(b"fmt ")
    buf.write(struct.pack("<I", 16))
    buf.write(struct.pack("<H", 1))  # PCM format
    buf.write(struct.pack("<H", num_channels))
    buf.write(struct.pack("<I", sample_rate))
    buf.write(struct.pack("<I", byte_rate))
    buf.write(struct.pack("<H", block_align))
    buf.write(struct.pack("<H", bits_per_sample))
    buf.write(b"data")
    buf.write(struct.pack("<I", data_size))
    return buf.getvalue()


def _resolve_output_format(body: DialogueRequest) -> OutputFormat:
    """Resolve output format from the request body."""
    if body.output_format is not None:
        try:
            return parse_output_format(body.output_format)
        except ValueError as exc:
            raise InvalidRequestError(str(exc)) from exc
    return parse_output_format(body.response_format)


def _resample_chunk(pcm_data: bytes, from_rate: int, to_rate: int) -> bytes:
    """Resample a PCM chunk."""
    from macaw._dsp import resample_pcm_bytes

    return resample_pcm_bytes(pcm_data, from_rate, to_rate)
