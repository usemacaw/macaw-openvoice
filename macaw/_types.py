"""Core types for Macaw OpenVoice.

This module defines enums, dataclasses, and type aliases used by all runtime
components. Changes here affect the entire system.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum


class STTArchitecture(Enum):
    """STT model architecture.

    Determines how the runtime adapts the streaming pipeline:
    - ENCODER_DECODER: window accumulation, LocalAgreement for partials (e.g., Whisper)
    - CTC: frame-by-frame, native partials (CTC-based models)
    - STREAMING_NATIVE: true streaming, engine manages state (e.g., Paraformer)
    """

    ENCODER_DECODER = "encoder-decoder"
    CTC = "ctc"
    STREAMING_NATIVE = "streaming-native"


class ModelType(Enum):
    """Model type in the registry."""

    STT = "stt"
    TTS = "tts"


class SessionState(Enum):
    """State of an STT streaming session.

    Valid transitions:
        INIT -> ACTIVE (first audio with speech)
        INIT -> CLOSED (30s timeout without audio)
        ACTIVE -> SILENCE (VAD detects silence)
        SILENCE -> ACTIVE (VAD detects speech)
        SILENCE -> HOLD (30s timeout without speech)
        HOLD -> ACTIVE (VAD detects speech)
        HOLD -> CLOSING (5min timeout)
        CLOSING -> CLOSED (full flush or 2s timeout)
        Any -> CLOSED (unrecoverable error)
    """

    INIT = "init"
    ACTIVE = "active"
    SILENCE = "silence"
    HOLD = "hold"
    CLOSING = "closing"
    CLOSED = "closed"


class VADSensitivity(Enum):
    """VAD sensitivity level.

    Adjusts the Silero VAD threshold and energy pre-filter together.
    """

    HIGH = "high"  # threshold=0.3, energy=-50dBFS (whisper, banking)
    NORMAL = "normal"  # threshold=0.5, energy=-40dBFS (normal conversation)
    LOW = "low"  # threshold=0.7, energy=-30dBFS (noisy environment)


class ResponseFormat(Enum):
    """Response format for the transcription API."""

    JSON = "json"
    VERBOSE_JSON = "verbose_json"
    TEXT = "text"
    SRT = "srt"
    VTT = "vtt"


@dataclass(frozen=True, slots=True)
class WordTimestamp:
    """Timestamp for a single word."""

    word: str
    start: float
    end: float
    probability: float | None = None


@dataclass(frozen=True, slots=True)
class TranscriptSegment:
    """Transcription segment (partial or final).

    Emitted by the worker via gRPC and forwarded to the client via WebSocket.
    """

    text: str
    is_final: bool
    segment_id: int
    start_ms: int | None = None
    end_ms: int | None = None
    language: str | None = None
    confidence: float | None = None
    words: tuple[WordTimestamp, ...] | None = None


@dataclass(frozen=True, slots=True)
class SegmentDetail:
    """Segment details in verbose_json format."""

    id: int
    start: float
    end: float
    text: str
    avg_logprob: float = 0.0
    no_speech_prob: float = 0.0
    compression_ratio: float = 0.0


@dataclass(frozen=True, slots=True)
class BatchResult:
    """Batch transcription result (full file)."""

    text: str
    language: str
    duration: float
    segments: tuple[SegmentDetail, ...]
    words: tuple[WordTimestamp, ...] | None = None


@dataclass(frozen=True, slots=True)
class EngineCapabilities:
    """Capabilities reported by the STT engine at runtime.

    May differ from the manifest (macaw.yaml) if the engine discovers
    additional capabilities after load.
    """

    supports_hot_words: bool = False
    supports_initial_prompt: bool = False
    supports_batch: bool = False
    supports_word_timestamps: bool = False
    max_concurrent_sessions: int = 1


# --- TTS ---


@dataclass(frozen=True, slots=True)
class VoiceInfo:
    """Information about an available voice in the TTS backend."""

    voice_id: str
    name: str
    language: str
    gender: str | None = None


@dataclass(frozen=True, slots=True)
class TTSSpeechResult:
    """TTS synthesis result (full audio).

    Used by the REST endpoint POST /v1/audio/speech.
    For streaming, TTSBackend.synthesize() returns AsyncIterator[bytes].
    """

    audio_data: bytes
    sample_rate: int
    duration: float
    voice: str
