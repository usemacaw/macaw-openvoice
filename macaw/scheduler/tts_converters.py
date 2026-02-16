"""Conversores TTS: proto <-> dominio para o endpoint REST.

Funcoes puras que constroem SynthesizeRequest proto a partir dos parametros
da API REST, e acumulam SynthesizeChunk proto em TTSSpeechResult de dominio.
"""

from __future__ import annotations

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
) -> SynthesizeRequest:
    """Constroi SynthesizeRequest proto a partir dos parametros da API REST."""
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
    return req
