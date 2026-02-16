"""Modelos Pydantic para o protocolo WebSocket de streaming STT.

Define todos os eventos server->client e comandos client->server
conforme o protocolo definido no PRD (secao 9 — WS /v1/realtime).
"""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field

from macaw._types import VADSensitivity
from macaw.server.constants import TTS_MAX_TEXT_LENGTH

# ---------------------------------------------------------------------------
# Shared models
# ---------------------------------------------------------------------------


class PreprocessingOverrides(BaseModel):
    """Overrides de preprocessing configuráveis por sessão."""

    # Allow fields beginning with "model_" (eg. `model_tts`) without
    # triggering Pydantic's protected namespace warning during model
    # construction. These models are frozen to be safely hashable and
    # lightweight for message passing.
    model_config = ConfigDict(frozen=True, protected_namespaces=())

    denoise: bool = False
    denoise_engine: str = "rnnoise"


class SessionConfig(BaseModel):
    """Configuracao da sessao, retornada em session.created."""

    model_config = ConfigDict(frozen=True, protected_namespaces=())

    vad_sensitivity: VADSensitivity = VADSensitivity.NORMAL
    silence_timeout_ms: int = 300
    hold_timeout_ms: int = 300_000
    max_segment_duration_ms: int = 30_000
    language: str | None = None
    enable_partial_transcripts: bool = True
    enable_itn: bool = True
    preprocessing: PreprocessingOverrides = PreprocessingOverrides()
    input_sample_rate: int | None = None
    model_tts: str | None = None


class WordEvent(BaseModel):
    """Palavra com timestamps para transcript.final."""

    model_config = ConfigDict(frozen=True, protected_namespaces=())

    word: str
    start: float
    end: float


# ---------------------------------------------------------------------------
# Server -> Client events
# ---------------------------------------------------------------------------


class SessionCreatedEvent(BaseModel):
    """Emitido quando a sessao WebSocket e criada."""

    model_config = ConfigDict(frozen=True, protected_namespaces=())

    type: Literal["session.created"] = "session.created"
    session_id: str
    model: str
    config: SessionConfig


class VADSpeechStartEvent(BaseModel):
    """Emitido quando o VAD detecta inicio de fala."""

    model_config = ConfigDict(frozen=True, protected_namespaces=())

    type: Literal["vad.speech_start"] = "vad.speech_start"
    timestamp_ms: int


class VADSpeechEndEvent(BaseModel):
    """Emitido quando o VAD detecta fim de fala."""

    model_config = ConfigDict(frozen=True, protected_namespaces=())

    type: Literal["vad.speech_end"] = "vad.speech_end"
    timestamp_ms: int


class TranscriptPartialEvent(BaseModel):
    """Hipotese intermediaria de transcricao (pode mudar)."""

    model_config = ConfigDict(frozen=True, protected_namespaces=())

    type: Literal["transcript.partial"] = "transcript.partial"
    text: str
    segment_id: int
    timestamp_ms: int


class TranscriptFinalEvent(BaseModel):
    """Segmento confirmado de transcricao (nao muda)."""

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
    """Emitido quando a sessao transita para estado HOLD."""

    model_config = ConfigDict(frozen=True, protected_namespaces=())

    type: Literal["session.hold"] = "session.hold"
    timestamp_ms: int
    hold_timeout_ms: int


class SessionRateLimitEvent(BaseModel):
    """Backpressure: cliente enviando audio mais rapido que real-time."""

    model_config = ConfigDict(frozen=True, protected_namespaces=())

    type: Literal["session.rate_limit"] = "session.rate_limit"
    delay_ms: int
    message: str


class SessionFramesDroppedEvent(BaseModel):
    """Frames descartados por excesso de backlog (>10s)."""

    model_config = ConfigDict(frozen=True, protected_namespaces=())

    type: Literal["session.frames_dropped"] = "session.frames_dropped"
    dropped_ms: int
    message: str


class StreamingErrorEvent(BaseModel):
    """Erro durante streaming (com flag de recuperabilidade)."""

    model_config = ConfigDict(frozen=True, protected_namespaces=())

    type: Literal["error"] = "error"
    code: str
    message: str
    recoverable: bool
    resume_segment_id: int | None = None


class SessionClosedEvent(BaseModel):
    """Emitido quando a sessao e encerrada."""

    model_config = ConfigDict(frozen=True, protected_namespaces=())

    type: Literal["session.closed"] = "session.closed"
    reason: str
    total_duration_ms: int
    segments_transcribed: int


class TTSSpeakingStartEvent(BaseModel):
    """Emitido quando o TTS comeca a produzir audio."""

    model_config = ConfigDict(frozen=True, protected_namespaces=())

    type: Literal["tts.speaking_start"] = "tts.speaking_start"
    request_id: str
    timestamp_ms: int


class TTSSpeakingEndEvent(BaseModel):
    """Emitido quando o TTS termina de produzir audio."""

    model_config = ConfigDict(frozen=True, protected_namespaces=())

    type: Literal["tts.speaking_end"] = "tts.speaking_end"
    request_id: str
    timestamp_ms: int
    duration_ms: int
    cancelled: bool = False


# ---------------------------------------------------------------------------
# Client -> Server commands
# ---------------------------------------------------------------------------


class SessionConfigureCommand(BaseModel):
    """Configura parametros da sessao de streaming."""

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
    """Cancela a sessao de streaming."""

    model_config = ConfigDict(frozen=True, protected_namespaces=())

    type: Literal["session.cancel"] = "session.cancel"


class InputAudioBufferCommitCommand(BaseModel):
    """Forca commit manual do segmento de audio atual."""

    model_config = ConfigDict(frozen=True, protected_namespaces=())

    type: Literal["input_audio_buffer.commit"] = "input_audio_buffer.commit"


class SessionCloseCommand(BaseModel):
    """Encerra a sessao de streaming gracefully."""

    model_config = ConfigDict(frozen=True, protected_namespaces=())

    type: Literal["session.close"] = "session.close"


class TTSSpeakCommand(BaseModel):
    """Comando para sintetizar voz via TTS no WebSocket full-duplex.

    O cliente envia texto e recebe audio TTS como binary frames.
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


class TTSCancelCommand(BaseModel):
    """Cancela a sintese TTS em andamento.

    Se request_id e fornecido, cancela apenas essa sintese.
    Se omitido, cancela qualquer sintese ativa.
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
)

ClientCommand = (
    SessionConfigureCommand
    | SessionCancelCommand
    | InputAudioBufferCommitCommand
    | SessionCloseCommand
    | TTSSpeakCommand
    | TTSCancelCommand
)
