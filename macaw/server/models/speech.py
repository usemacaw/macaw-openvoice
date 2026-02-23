"""Pydantic models for the TTS endpoint POST /v1/audio/speech."""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field

from macaw.server.constants import TTS_MAX_TEXT_LENGTH
from macaw.server.models.effects import AudioEffectsParams
from macaw.server.models.voice_settings import VoiceSettings
from macaw.server.pronunciation.models import PronDictLocator


class SpeechRequest(BaseModel):
    """Request body for POST /v1/audio/speech.

    Compatible with the OpenAI Audio API contract.
    """

    model: str = Field(description="TTS model name in the registry.")
    input: str = Field(
        min_length=1, max_length=TTS_MAX_TEXT_LENGTH, description="Text to be synthesized."
    )
    voice: str = Field(
        default="default",
        description="Voice identifier.",
    )
    response_format: Literal["wav", "pcm", "opus", "mp3", "mulaw", "alaw"] = Field(
        default="wav",
        description="Output audio format (simple codec name).",
    )
    output_format: str | None = Field(
        default=None,
        description="Output format string with optional sample rate and bitrate "
        "(e.g., 'mp3_44100_128', 'pcm_16000', 'opus_48000_64'). "
        "Takes precedence over response_format when provided.",
    )
    speed: float = Field(
        default=1.0,
        ge=0.25,
        le=4.0,
        description="Synthesis speed (0.25-4.0).",
    )
    # Extended options for LLM-based TTS engines (e.g., Qwen3-TTS)
    language: str | None = Field(
        default=None,
        description="Target language (e.g., 'English', 'Chinese').",
    )
    ref_audio: str | None = Field(
        default=None,
        description="Base64-encoded reference audio for voice cloning.",
    )
    ref_text: str | None = Field(
        default=None,
        description="Transcription of the reference audio.",
    )
    instruction: str | None = Field(
        default=None,
        description="Style/voice instruction for voice design.",
    )
    # SSML parsing (parse SSML tags in input text before synthesis)
    enable_ssml_parsing: bool = Field(
        default=False,
        description="When true, parse input text as SSML. Supported tags: "
        "<break>, <prosody>, <emphasis>, <say-as>, <phoneme>. "
        "Unsupported tags are stripped. Invalid SSML returns 422.",
    )
    # Post-synthesis audio effects (applied server-side before transport)
    effects: AudioEffectsParams | None = Field(
        default=None,
        description="Audio effects applied after synthesis (pitch shift, reverb).",
    )
    # Alignment options
    include_alignment: bool = Field(
        default=False,
        description="Include word-level timing alignment in response. "
        "When true, response switches to NDJSON (application/x-ndjson) "
        "regardless of response_format. Not compatible with response_format='opus'.",
    )
    alignment_granularity: Literal["word", "character"] = Field(
        default="word",
        description="Alignment granularity: 'word' or 'character'.",
    )
    # Seed and generation control
    seed: int | None = Field(
        default=None,
        ge=1,
        description="Random seed for reproducible generation (must be >= 1). "
        "0 is reserved as 'not set' (proto3 convention). "
        "No-op for deterministic engines (e.g. Kokoro).",
    )
    text_normalization: Literal["auto", "on", "off"] = Field(
        default="auto",
        description="Text normalization mode: 'auto' (engine decides), "
        "'on' (force normalization), 'off' (skip normalization).",
    )
    # Sampling parameters (LLM-based TTS engines like Qwen3-TTS)
    temperature: float | None = Field(
        default=None,
        ge=0.0,
        le=2.0,
        description="Sampling temperature for LLM-based TTS (0 = engine default).",
    )
    top_k: int | None = Field(
        default=None,
        ge=0,
        description="Top-k sampling for LLM-based TTS (0 = engine default).",
    )
    top_p: float | None = Field(
        default=None,
        ge=0.0,
        le=1.0,
        description="Nucleus sampling threshold for LLM-based TTS (0 = engine default).",
    )
    # Voice settings (ElevenLabs-compatible abstract voice characteristics)
    voice_settings: VoiceSettings | None = Field(
        default=None,
        description="Abstract voice characteristics (stability, similarity_boost, style, speed). "
        "Mapped to engine-specific parameters by the TTS backend.",
    )
    # Pronunciation dictionaries (up to 3, applied in order before synthesis)
    pronunciation_dictionary_locators: list[PronDictLocator] | None = Field(
        default=None,
        max_length=3,
        description="Ordered list of pronunciation dictionary references (max 3). "
        "Alias rules perform text replacement before synthesis.",
    )
    # Cross-generation continuity (ElevenLabs-compatible)
    previous_text: str | None = Field(
        default=None,
        description="Text generated in the previous TTS request. "
        "Used by engines for prosody continuity across generations.",
    )
    next_text: str | None = Field(
        default=None,
        description="Text that will be generated in the next TTS request. "
        "Used by engines for anticipatory prosody.",
    )
    previous_request_ids: list[str] | None = Field(
        default=None,
        max_length=3,
        description="Request IDs of up to 3 previous TTS requests "
        "for cross-generation conditioning.",
    )
    next_request_ids: list[str] | None = Field(
        default=None,
        max_length=3,
        description="Request IDs of up to 3 next TTS requests for cross-generation conditioning.",
    )
