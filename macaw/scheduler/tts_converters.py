"""TTS converters: proto <-> domain for the REST endpoint.

Pure functions that build SynthesizeRequest proto from REST API parameters,
and accumulate SynthesizeChunk proto into domain TTSSpeechResult.
"""

from __future__ import annotations

from typing import Literal

from macaw.proto.tts_worker_pb2 import SynthesizeRequest


def build_tts_proto_request(
    *,
    request_id: str,
    text: str,
    voice: str,
    sample_rate: int,
    speed: float,
    language: str | None = None,
    ref_audio: bytes | None = None,
    ref_text: str | None = None,
    instruction: str | None = None,
    codec: str | None = None,
    include_alignment: bool = False,
    alignment_granularity: Literal["word", "character"] = "word",
    seed: int | None = None,
    text_normalization: str = "auto",
    temperature: float | None = None,
    top_k: int | None = None,
    top_p: float | None = None,
) -> SynthesizeRequest:
    """Build SynthesizeRequest proto from REST API parameters."""
    req = SynthesizeRequest(
        request_id=request_id,
        text=text,
        voice=voice,
        sample_rate=sample_rate,
        speed=speed,
    )
    if language:
        req.language = language
    if ref_audio:
        req.ref_audio = ref_audio
    if ref_text:
        req.ref_text = ref_text
    if instruction:
        req.instruction = instruction
    if codec:
        req.codec = codec
    if include_alignment:
        req.include_alignment = True
        req.alignment_granularity = alignment_granularity
    # Proto3 convention: scalar 0 is the wire default, so we only set
    # fields that differ from 0.  Zero values mean "engine default" for
    # all sampling params (documented in SpeechRequest descriptions).
    if seed is not None and seed > 0:
        req.seed = seed
    if text_normalization and text_normalization != "auto":
        req.text_normalization = text_normalization
    if temperature is not None and temperature > 0:
        req.temperature = temperature
    if top_k is not None and top_k > 0:
        req.top_k = top_k
    if top_p is not None and top_p > 0:
        req.top_p = top_p
    return req
