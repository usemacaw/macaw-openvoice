"""Conversion between Macaw types and gRPC protobuf messages for TTS.

Pure functions without side effects -- easier testing and reuse.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Literal

from macaw._audio_constants import TTS_DEFAULT_SAMPLE_RATE
from macaw.proto.tts_worker_pb2 import (
    AlignmentItem,
    ChunkAlignment,
    HealthResponse,
    ListVoicesResponse,
    SynthesizeChunk,
    VoiceInfoProto,
)
from macaw.workers.proto_utils import build_health_response

if TYPE_CHECKING:
    from macaw._types import TTSAlignmentItem, VoiceInfo
    from macaw.proto.tts_worker_pb2 import SynthesizeRequest


@dataclass(frozen=True, slots=True)
class SynthesizeParams:
    """Typed parameters extracted from SynthesizeRequest.

    ``options`` carries engine-specific parameters (language, ref_audio,
    instruction, seed, temperature, top_k, top_p, etc.) as an untyped dict.
    This is intentional: the runtime layer is engine-agnostic, so sampling
    params stay in the opaque dict and each backend extracts what it needs.

    ``include_alignment`` and ``alignment_granularity`` are typed fields
    because alignment is a runtime-level concern (the servicer decides the
    alignment path), not an engine-specific one.
    """

    text: str
    voice: str
    sample_rate: int
    speed: float
    options: dict[str, object] | None = None
    include_alignment: bool = False
    alignment_granularity: Literal["word", "character"] = "word"


def proto_request_to_synthesize_params(
    request: SynthesizeRequest,
) -> SynthesizeParams:
    """Convert gRPC SynthesizeRequest to params for TTSBackend.synthesize.

    Treat empty strings as defaults (protobuf default for strings is empty).
    Extended fields (language, ref_audio, ref_text, instruction) are packed
    into an options dict only when at least one is present.
    """
    voice: str = request.voice if request.voice else "default"
    sample_rate: int = request.sample_rate if request.sample_rate > 0 else TTS_DEFAULT_SAMPLE_RATE
    speed: float = request.speed if request.speed > 0.0 else 1.0

    # Proto3 scalars: 0/0.0/"" are wire defaults.  We use explicit != 0
    # checks instead of truthiness so the intent is visible: zero means
    # "engine default / not set" for all numeric params.
    options: dict[str, object] | None = None
    has_extended = (
        request.language != ""
        or request.ref_audio != b""
        or request.ref_text != ""
        or request.instruction != ""
        or request.seed != 0
        or request.text_normalization != ""
        or request.temperature != 0
        or request.top_k != 0
        or request.top_p != 0
    )
    if has_extended:
        options = {}
        if request.language:
            options["language"] = request.language
        if request.ref_audio:
            options["ref_audio"] = request.ref_audio
        if request.ref_text:
            options["ref_text"] = request.ref_text
        if request.instruction:
            options["instruction"] = request.instruction
        if request.seed != 0:
            options["seed"] = request.seed
        if request.text_normalization:
            options["text_normalization"] = request.text_normalization
        if request.temperature != 0:
            options["temperature"] = request.temperature
        if request.top_k != 0:
            options["top_k"] = request.top_k
        if request.top_p != 0:
            options["top_p"] = request.top_p

    include_alignment: bool = request.include_alignment
    raw_granularity = request.alignment_granularity or "word"
    alignment_granularity: Literal["word", "character"] = (
        "character" if raw_granularity == "character" else "word"
    )

    return SynthesizeParams(
        text=request.text,
        voice=voice,
        sample_rate=sample_rate,
        speed=speed,
        options=options,
        include_alignment=include_alignment,
        alignment_granularity=alignment_granularity,
    )


def _alignment_items_to_proto(
    items: tuple[TTSAlignmentItem, ...],
    granularity: str,
) -> ChunkAlignment:
    """Convert alignment items to a ChunkAlignment protobuf message."""
    return ChunkAlignment(
        items=[
            AlignmentItem(
                text=item.text,
                start_ms=item.start_ms,
                duration_ms=item.duration_ms,
            )
            for item in items
        ],
        granularity=granularity,
    )


def audio_chunk_to_proto(
    audio_data: bytes,
    is_last: bool,
    duration: float,
    codec: str = "",
    alignment: tuple[TTSAlignmentItem, ...] | None = None,
    normalized_alignment: tuple[TTSAlignmentItem, ...] | None = None,
    alignment_granularity: Literal["word", "character"] = "word",
) -> SynthesizeChunk:
    """Convert audio chunk to SynthesizeChunk protobuf."""
    chunk_alignment = (
        _alignment_items_to_proto(alignment, alignment_granularity) if alignment else None
    )
    chunk_norm_alignment = (
        _alignment_items_to_proto(normalized_alignment, alignment_granularity)
        if normalized_alignment
        else None
    )
    return SynthesizeChunk(
        audio_data=audio_data,
        is_last=is_last,
        duration=duration,
        codec=codec,
        alignment=chunk_alignment,
        normalized_alignment=chunk_norm_alignment,
    )


def voices_to_proto_response(
    voices: list[VoiceInfo],
) -> ListVoicesResponse:
    """Convert list of VoiceInfo to ListVoicesResponse protobuf."""
    protos = [
        VoiceInfoProto(
            voice_id=v.voice_id,
            name=v.name,
            language=v.language,
            gender=v.gender or "",
        )
        for v in voices
    ]
    return ListVoicesResponse(voices=protos)


def health_dict_to_proto_response(
    health: dict[str, str],
    model_name: str,
    engine: str,
) -> HealthResponse:
    """Convert backend health dict to HealthResponse protobuf."""
    return build_health_response(HealthResponse, health, model_name, engine)  # type: ignore[no-any-return]
