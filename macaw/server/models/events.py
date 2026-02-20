"""Pydantic models for the WebSocket STT streaming protocol.

Defines all server->client events and client->server commands
as per the protocol defined in the PRD (section 9 — WS /v1/realtime).
"""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field

from macaw._types import VADSensitivity
from macaw.server.constants import TTS_MAX_TEXT_LENGTH
from macaw.server.models.effects import AudioEffectsParams

# ---------------------------------------------------------------------------
# Shared models
# ---------------------------------------------------------------------------


class PreprocessingOverrides(BaseModel):
    """Per-session configurable preprocessing overrides."""

    # Allow fields beginning with "model_" (eg. `model_tts`) without
    # triggering Pydantic's protected namespace warning during model
    # construction. These models are frozen to be safely hashable and
    # lightweight for message passing.
    model_config = ConfigDict(frozen=True, protected_namespaces=())

    denoise: bool = False
    denoise_engine: str = "rnnoise"


class SessionConfig(BaseModel):
    """Session configuration, returned in session.created.

    Timeout defaults are derived from ``SessionSettings`` so the values
    reported to the client in ``session.created`` match the actual
    server-side behaviour.
    """

    model_config = ConfigDict(frozen=True, protected_namespaces=())

    vad_sensitivity: VADSensitivity = VADSensitivity.NORMAL
    silence_timeout_ms: int = 30_000
    hold_timeout_ms: int = 300_000
    max_segment_duration_ms: int = 30_000
    language: str | None = None
    enable_partial_transcripts: bool = True
    enable_itn: bool = True
    preprocessing: PreprocessingOverrides = PreprocessingOverrides()
    input_sample_rate: int | None = None
    model_tts: str | None = None


class WordEvent(BaseModel):
    """Word with timestamps for transcript.final."""

    model_config = ConfigDict(frozen=True, protected_namespaces=())

    word: str
    start: float
    end: float


# ---------------------------------------------------------------------------
# Server -> Client events
# ---------------------------------------------------------------------------


class SessionCreatedEvent(BaseModel):
    """Emitted when the WebSocket session is created."""

    model_config = ConfigDict(frozen=True, protected_namespaces=())

    type: Literal["session.created"] = "session.created"
    session_id: str
    model: str
    config: SessionConfig


class VADSpeechStartEvent(BaseModel):
    """Emitted when VAD detects speech start."""

    model_config = ConfigDict(frozen=True, protected_namespaces=())

    type: Literal["vad.speech_start"] = "vad.speech_start"
    timestamp_ms: int


class VADSpeechEndEvent(BaseModel):
    """Emitted when VAD detects speech end."""

    model_config = ConfigDict(frozen=True, protected_namespaces=())

    type: Literal["vad.speech_end"] = "vad.speech_end"
    timestamp_ms: int


class TranscriptPartialEvent(BaseModel):
    """Intermediate transcription hypothesis (may change)."""

    model_config = ConfigDict(frozen=True, protected_namespaces=())

    type: Literal["transcript.partial"] = "transcript.partial"
    text: str
    segment_id: int
    timestamp_ms: int


class TranscriptFinalEvent(BaseModel):
    """Confirmed transcription segment (immutable)."""

    model_config = ConfigDict(frozen=True, protected_namespaces=())

    type: Literal["transcript.final"] = "transcript.final"
    text: str
    segment_id: int
    start_ms: int
    end_ms: int
    language: str | None = None
    confidence: float | None = None
    words: list[WordEvent] | None = None


class SessionHoldEvent(BaseModel):
    """Emitted when the session transitions to HOLD state."""

    model_config = ConfigDict(frozen=True, protected_namespaces=())

    type: Literal["session.hold"] = "session.hold"
    timestamp_ms: int
    hold_timeout_ms: int


class SessionRateLimitEvent(BaseModel):
    """Backpressure: client sending audio faster than real-time."""

    model_config = ConfigDict(frozen=True, protected_namespaces=())

    type: Literal["session.rate_limit"] = "session.rate_limit"
    delay_ms: int
    message: str


class SessionFramesDroppedEvent(BaseModel):
    """Frames dropped due to excessive backlog (>10s)."""

    model_config = ConfigDict(frozen=True, protected_namespaces=())

    type: Literal["session.frames_dropped"] = "session.frames_dropped"
    dropped_ms: int
    message: str


class StreamingErrorEvent(BaseModel):
    """Error during streaming (with recoverability flag)."""

    model_config = ConfigDict(frozen=True, protected_namespaces=())

    type: Literal["error"] = "error"
    code: str
    message: str
    recoverable: bool
    resume_segment_id: int | None = None


class SessionClosedEvent(BaseModel):
    """Emitted when the session is terminated."""

    model_config = ConfigDict(frozen=True, protected_namespaces=())

    type: Literal["session.closed"] = "session.closed"
    reason: str
    total_duration_ms: int
    segments_transcribed: int


class TTSSpeakingStartEvent(BaseModel):
    """Emitted when TTS starts producing audio."""

    model_config = ConfigDict(frozen=True, protected_namespaces=())

    type: Literal["tts.speaking_start"] = "tts.speaking_start"
    request_id: str
    timestamp_ms: int


class TTSSpeakingEndEvent(BaseModel):
    """Emitted when TTS finishes producing audio."""

    model_config = ConfigDict(frozen=True, protected_namespaces=())

    type: Literal["tts.speaking_end"] = "tts.speaking_end"
    request_id: str
    timestamp_ms: int
    duration_ms: int
    cancelled: bool = False


class TTSAlignmentItemEvent(BaseModel):
    """Single word/character timing in a TTS alignment event.

    Named ``TTSAlignmentItemEvent`` to avoid collision with the runtime
    dataclass ``macaw._types.TTSAlignmentItem``.
    """

    model_config = ConfigDict(frozen=True, protected_namespaces=())

    text: str
    start_ms: int
    duration_ms: int


class TTSAlignmentEvent(BaseModel):
    """Emitted before each TTS binary audio frame when alignment is requested.

    Contains per-word or per-character timing for the upcoming audio chunk.
    Sent as a JSON text frame immediately before the corresponding binary frame.

    ``normalized_items`` carries the same timing against normalized/phonemized
    text (only present for engines that expose phoneme data, e.g. Kokoro).
    """

    model_config = ConfigDict(frozen=True, protected_namespaces=())

    type: Literal["tts.alignment"] = "tts.alignment"
    request_id: str
    items: list[TTSAlignmentItemEvent]
    normalized_items: list[TTSAlignmentItemEvent] | None = None
    granularity: Literal["word", "character"] = "word"


# ---------------------------------------------------------------------------
# Client -> Server commands
# ---------------------------------------------------------------------------


class SessionConfigureCommand(BaseModel):
    """Configures streaming session parameters."""

    model_config = ConfigDict(frozen=True, protected_namespaces=())

    type: Literal["session.configure"] = "session.configure"
    vad_sensitivity: VADSensitivity | None = None
    silence_timeout_ms: int | None = Field(default=None, gt=0)
    hold_timeout_ms: int | None = Field(default=None, gt=0)
    max_segment_duration_ms: int | None = Field(default=None, gt=0)
    language: str | None = None
    hot_words: list[str] | None = None
    hot_word_boost: float | None = None
    enable_partial_transcripts: bool | None = None
    enable_itn: bool | None = None
    preprocessing: PreprocessingOverrides | None = None
    input_sample_rate: int | None = Field(default=None, gt=0)
    model_tts: str | None = None


class SessionCancelCommand(BaseModel):
    """Cancels the streaming session."""

    model_config = ConfigDict(frozen=True, protected_namespaces=())

    type: Literal["session.cancel"] = "session.cancel"


class InputAudioBufferCommitCommand(BaseModel):
    """Forces manual commit of the current audio segment."""

    model_config = ConfigDict(frozen=True, protected_namespaces=())

    type: Literal["input_audio_buffer.commit"] = "input_audio_buffer.commit"


class SessionCloseCommand(BaseModel):
    """Terminates the streaming session gracefully."""

    model_config = ConfigDict(frozen=True, protected_namespaces=())

    type: Literal["session.close"] = "session.close"


class TTSSpeakCommand(BaseModel):
    """Command to synthesize voice via TTS on full-duplex WebSocket.

    The client sends text and receives TTS audio as binary frames.
    """

    model_config = ConfigDict(frozen=True)

    type: Literal["tts.speak"] = "tts.speak"
    text: str = Field(min_length=1, max_length=TTS_MAX_TEXT_LENGTH)
    voice: str = "default"
    speed: float = Field(default=1.0, ge=0.25, le=4.0)
    request_id: str | None = None
    # Extended options for LLM-based TTS engines (e.g., Qwen3-TTS)
    language: str | None = None
    ref_audio: str | None = None  # base64
    ref_text: str | None = None
    instruction: str | None = None
    # Audio codec for TTS output; None = raw PCM
    codec: Literal["opus"] | None = None
    # Post-synthesis audio effects (applied server-side before transport)
    effects: AudioEffectsParams | None = None
    # Alignment options (opt-in per-word/character timing)
    include_alignment: bool = False
    alignment_granularity: Literal["word", "character"] = "word"
    # Seed and generation control
    seed: int | None = Field(default=None, ge=1)
    text_normalization: Literal["auto", "on", "off"] = "auto"
    # Sampling parameters (LLM-based TTS engines)
    temperature: float | None = Field(default=None, ge=0.0, le=2.0)
    top_k: int | None = Field(default=None, ge=0)
    top_p: float | None = Field(default=None, ge=0.0, le=1.0)


class TTSCancelCommand(BaseModel):
    """Cancels in-progress TTS synthesis.

    If request_id is provided, cancels only that synthesis.
    If omitted, cancels any active synthesis.
    """

    model_config = ConfigDict(frozen=True)

    type: Literal["tts.cancel"] = "tts.cancel"
    request_id: str | None = None


# ---------------------------------------------------------------------------
# Union types for dispatch
# ---------------------------------------------------------------------------

ServerEvent = (
    SessionCreatedEvent
    | VADSpeechStartEvent
    | VADSpeechEndEvent
    | TranscriptPartialEvent
    | TranscriptFinalEvent
    | SessionHoldEvent
    | SessionRateLimitEvent
    | SessionFramesDroppedEvent
    | StreamingErrorEvent
    | SessionClosedEvent
    | TTSSpeakingStartEvent
    | TTSSpeakingEndEvent
    | TTSAlignmentEvent
)

ClientCommand = (
    SessionConfigureCommand
    | SessionCancelCommand
    | InputAudioBufferCommitCommand
    | SessionCloseCommand
    | TTSSpeakCommand
    | TTSCancelCommand
)
