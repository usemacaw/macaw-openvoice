"""Conversion between Macaw types and gRPC protobuf messages for TTS.

Pure functions without side effects -- easier testing and reuse.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

from macaw._audio_constants import TTS_DEFAULT_SAMPLE_RATE
from macaw.proto.tts_worker_pb2 import (
    HealthResponse,
    ListVoicesResponse,
    SynthesizeChunk,
    VoiceInfoProto,
)
from macaw.workers.proto_utils import build_health_response

if TYPE_CHECKING:
    from macaw._types import VoiceInfo
    from macaw.proto.tts_worker_pb2 import SynthesizeRequest


@dataclass(frozen=True, slots=True)
class SynthesizeParams:
    """Typed parameters extracted from SynthesizeRequest."""

    text: str
    voice: str
    sample_rate: int
    speed: float
    options: dict[str, object] | None = None


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

    options: dict[str, object] | None = None
    if request.language or request.ref_audio or request.ref_text or request.instruction:
        options = {}
        if request.language:
            options["language"] = request.language
        if request.ref_audio:
            options["ref_audio"] = request.ref_audio
        if request.ref_text:
            options["ref_text"] = request.ref_text
        if request.instruction:
            options["instruction"] = request.instruction

    return SynthesizeParams(
        text=request.text,
        voice=voice,
        sample_rate=sample_rate,
        speed=speed,
        options=options,
    )


def audio_chunk_to_proto(
    audio_data: bytes,
    is_last: bool,
    duration: float,
) -> SynthesizeChunk:
    """Convert audio chunk to SynthesizeChunk protobuf."""
    return SynthesizeChunk(
        audio_data=audio_data,
        is_last=is_last,
        duration=duration,
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
