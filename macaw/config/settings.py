"""Centralized configuration via pydantic-settings.

All ``MACAW_*`` environment variables are read, validated, and exposed here.
Logging env vars (``MACAW_LOG_FORMAT``, ``MACAW_LOG_LEVEL``) are intentionally
excluded — they stay in ``macaw.logging`` for bootstrap-safety.

Usage::

    from macaw.config.settings import get_settings

    settings = get_settings()
    print(settings.server.port)        # int, validated
    print(settings.worker.models_path) # Path, expanded

``.env`` files in the working directory are loaded automatically.
"""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path
from typing import TYPE_CHECKING

from pydantic import AliasChoices, Field, model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

from macaw._audio_constants import DEFAULT_DC_CUTOFF_HZ, DEFAULT_ITN_LANGUAGE, DEFAULT_TARGET_DBFS

if TYPE_CHECKING:
    from macaw._types import VADSensitivity


class ServerSettings(BaseSettings):
    """HTTP server and WebSocket settings."""

    model_config = SettingsConfigDict(extra="ignore", populate_by_name=True)

    host: str = Field(default="127.0.0.1", validation_alias="MACAW_HOST")
    port: int = Field(default=8000, ge=1, le=65535, validation_alias="MACAW_PORT")
    max_file_size_mb: int = Field(
        default=25, ge=1, le=500, validation_alias="MACAW_MAX_FILE_SIZE_MB"
    )
    retry_after_s: str = Field(default="5", validation_alias="MACAW_RETRY_AFTER_S")
    ws_max_frame_size_bytes: int = Field(
        default=1_048_576, ge=1024, validation_alias="MACAW_MAX_WS_FRAME_SIZE_BYTES"
    )
    ws_inactivity_timeout_s: float = Field(
        default=60.0, gt=0, validation_alias="MACAW_WS_INACTIVITY_TIMEOUT_S"
    )
    ws_heartbeat_interval_s: float = Field(
        default=10.0, gt=0, validation_alias="MACAW_WS_HEARTBEAT_INTERVAL_S"
    )
    ws_check_interval_s: float = Field(
        default=5.0, gt=0, validation_alias="MACAW_WS_CHECK_INTERVAL_S"
    )
    cors_origins: str = Field(
        default="",
        validation_alias="MACAW_CORS_ORIGINS",
    )
    voice_dir: str | None = Field(
        default=None,
        validation_alias="MACAW_VOICE_DIR",
    )

    @property
    def cors_origins_list(self) -> list[str]:
        """CORS origins as a list (parsed from comma-separated string)."""
        return [o.strip() for o in self.cors_origins.split(",") if o.strip()]

    @property
    def max_file_size_bytes(self) -> int:
        """Maximum upload size in bytes (derived from MB setting)."""
        return self.max_file_size_mb * 1024 * 1024

    @model_validator(mode="after")
    def _heartbeat_lt_inactivity(self) -> ServerSettings:
        if self.ws_heartbeat_interval_s >= self.ws_inactivity_timeout_s:
            msg = "ws_heartbeat_interval_s must be < ws_inactivity_timeout_s"
            raise ValueError(msg)
        return self


class TTSSettings(BaseSettings):
    """TTS gRPC timeout and streaming settings."""

    model_config = SettingsConfigDict(extra="ignore", populate_by_name=True)

    grpc_timeout_s: float = Field(default=60.0, gt=0, validation_alias="MACAW_TTS_GRPC_TIMEOUT_S")
    list_voices_timeout_s: float = Field(
        default=10.0, gt=0, validation_alias="MACAW_TTS_LIST_VOICES_TIMEOUT_S"
    )
    chunk_size_bytes: int = Field(
        default=4096,
        ge=512,
        le=65536,
        validation_alias="MACAW_TTS_CHUNK_SIZE_BYTES",
    )
    max_text_length: int = Field(
        default=4096,
        ge=1,
        le=1_000_000,
        validation_alias="MACAW_TTS_MAX_TEXT_LENGTH",
    )


class CLISettings(BaseSettings):
    """CLI client settings."""

    model_config = SettingsConfigDict(extra="ignore", populate_by_name=True)

    server_url: str = Field(default="http://localhost:8000", validation_alias="MACAW_SERVER_URL")
    http_timeout_s: float = Field(default=120.0, gt=0, validation_alias="MACAW_HTTP_TIMEOUT_S")


class WorkerSettings(BaseSettings):
    """Worker subprocess settings."""

    model_config = SettingsConfigDict(extra="ignore", populate_by_name=True)

    models_dir: str = Field(default="~/.macaw/models", validation_alias="MACAW_MODELS_DIR")
    worker_base_port: int = Field(
        default=50051, ge=1024, le=65535, validation_alias="MACAW_WORKER_BASE_PORT"
    )
    worker_host: str = Field(
        default="localhost",
        validation_alias="MACAW_WORKER_HOST",
    )
    stt_max_concurrent: int = Field(
        default=1,
        ge=1,
        le=16,
        validation_alias="MACAW_STT_WORKER_MAX_CONCURRENT",
    )
    stt_accumulation_threshold_s: float = Field(
        default=5.0,
        gt=0,
        le=30.0,
        validation_alias="MACAW_STT_ACCUMULATION_THRESHOLD_S",
    )
    stt_max_cancelled_requests: int = Field(
        default=10_000,
        ge=100,
        le=1_000_000,
        validation_alias="MACAW_STT_MAX_CANCELLED_REQUESTS",
    )

    @property
    def models_path(self) -> Path:
        """Expanded models directory as a Path object."""
        return Path(self.models_dir).expanduser()


class WorkerLifecycleSettings(BaseSettings):
    """Worker subprocess lifecycle tuning (crash recovery, health probing)."""

    model_config = SettingsConfigDict(extra="ignore", populate_by_name=True)

    max_crashes_in_window: int = Field(
        default=3, ge=1, le=100, validation_alias="MACAW_WORKER_MAX_CRASHES"
    )
    crash_window_s: float = Field(
        default=60.0, gt=0, le=3600, validation_alias="MACAW_WORKER_CRASH_WINDOW_S"
    )
    health_probe_initial_delay_s: float = Field(
        default=0.5, gt=0, le=60, validation_alias="MACAW_WORKER_HEALTH_PROBE_INITIAL_DELAY_S"
    )
    health_probe_max_delay_s: float = Field(
        default=5.0, gt=0, le=300, validation_alias="MACAW_WORKER_HEALTH_PROBE_MAX_DELAY_S"
    )
    health_probe_timeout_s: float = Field(
        default=120.0, gt=0, le=600, validation_alias="MACAW_WORKER_HEALTH_PROBE_TIMEOUT_S"
    )
    monitor_interval_s: float = Field(
        default=1.0, gt=0, le=60, validation_alias="MACAW_WORKER_MONITOR_INTERVAL_S"
    )
    stop_grace_period_s: float = Field(
        default=5.0, gt=0, le=60, validation_alias="MACAW_WORKER_STOP_GRACE_PERIOD_S"
    )
    default_warmup_steps: int = Field(
        default=3, ge=0, le=20, validation_alias="MACAW_WORKER_WARMUP_STEPS"
    )
    health_probe_rpc_timeout_s: float = Field(
        default=2.0,
        gt=0,
        le=30,
        validation_alias="MACAW_WORKER_HEALTH_PROBE_RPC_TIMEOUT_S",
    )

    @model_validator(mode="after")
    def _initial_lt_max_delay(self) -> WorkerLifecycleSettings:
        if self.health_probe_initial_delay_s >= self.health_probe_max_delay_s:
            msg = "health_probe_initial_delay_s must be < health_probe_max_delay_s"
            raise ValueError(msg)
        return self


class GRPCSettings(BaseSettings):
    """gRPC channel message size limits."""

    model_config = SettingsConfigDict(extra="ignore", populate_by_name=True)

    max_batch_message_mb: int = Field(
        default=30, ge=1, le=500, validation_alias="MACAW_GRPC_MAX_BATCH_MESSAGE_MB"
    )
    max_streaming_message_mb: int = Field(
        default=10, ge=1, le=100, validation_alias="MACAW_GRPC_MAX_STREAMING_MESSAGE_MB"
    )

    @property
    def max_batch_message_bytes(self) -> int:
        """Maximum batch gRPC message size in bytes (derived from MB setting)."""
        return self.max_batch_message_mb * 1024 * 1024

    @property
    def max_streaming_message_bytes(self) -> int:
        """Maximum streaming gRPC message size in bytes (derived from MB setting)."""
        return self.max_streaming_message_mb * 1024 * 1024


class VADSettings(BaseSettings):
    """Voice Activity Detection tuning.

    Controls sensitivity (coordinated preset for energy + silero thresholds)
    and debounce durations for speech start/end detection.
    """

    model_config = SettingsConfigDict(extra="ignore", populate_by_name=True)

    sensitivity: str = Field(
        default="normal",
        validation_alias="MACAW_VAD_SENSITIVITY",
    )
    min_speech_duration_ms: int = Field(
        default=250,
        ge=50,
        le=5000,
        validation_alias="MACAW_VAD_MIN_SPEECH_DURATION_MS",
    )
    min_silence_duration_ms: int = Field(
        default=300,
        ge=50,
        le=5000,
        validation_alias="MACAW_VAD_MIN_SILENCE_DURATION_MS",
    )
    max_speech_duration_ms: int = Field(
        default=30_000,
        ge=1000,
        le=600_000,
        validation_alias="MACAW_VAD_MAX_SPEECH_DURATION_MS",
    )
    energy_threshold_dbfs: float | None = Field(
        default=None,
        ge=-80.0,
        le=0.0,
        validation_alias="MACAW_VAD_ENERGY_THRESHOLD_DBFS",
    )
    silero_threshold: float | None = Field(
        default=None,
        gt=0.0,
        lt=1.0,
        validation_alias="MACAW_VAD_SILERO_THRESHOLD",
    )

    @model_validator(mode="after")
    def _validate_sensitivity(self) -> VADSettings:
        valid = {"high", "normal", "low"}
        normalized = self.sensitivity.lower()
        if normalized not in valid:
            msg = f"sensitivity must be one of {valid}, got {self.sensitivity!r}"
            raise ValueError(msg)
        object.__setattr__(self, "sensitivity", normalized)
        return self

    @model_validator(mode="after")
    def _min_speech_lt_max(self) -> VADSettings:
        if self.min_speech_duration_ms >= self.max_speech_duration_ms:
            msg = "min_speech_duration_ms must be < max_speech_duration_ms"
            raise ValueError(msg)
        return self

    @property
    def vad_sensitivity(self) -> VADSensitivity:
        """Return the VADSensitivity enum for use in VAD components."""
        from macaw._types import VADSensitivity

        return VADSensitivity(self.sensitivity)


class SessionSettings(BaseSettings):
    """Streaming session tuning (timeouts, ring buffer, backpressure)."""

    model_config = SettingsConfigDict(extra="ignore", populate_by_name=True)

    ring_buffer_duration_s: float = Field(
        default=60.0, gt=0, le=600, validation_alias="MACAW_SESSION_RING_BUFFER_DURATION_S"
    )
    recovery_timeout_s: float = Field(
        default=10.0, gt=0, le=120, validation_alias="MACAW_SESSION_RECOVERY_TIMEOUT_S"
    )
    drain_stream_timeout_s: float = Field(
        default=5.0, gt=0, le=60, validation_alias="MACAW_SESSION_DRAIN_STREAM_TIMEOUT_S"
    )
    flush_and_close_timeout_s: float = Field(
        default=2.0, gt=0, le=30, validation_alias="MACAW_SESSION_FLUSH_AND_CLOSE_TIMEOUT_S"
    )
    backpressure_max_backlog_s: float = Field(
        default=10.0, gt=0, le=120, validation_alias="MACAW_SESSION_BACKPRESSURE_MAX_BACKLOG_S"
    )
    backpressure_rate_limit_threshold: float = Field(
        default=1.2,
        gt=1.0,
        le=5.0,
        validation_alias="MACAW_SESSION_BACKPRESSURE_RATE_LIMIT_THRESHOLD",
    )
    init_timeout_s: float = Field(
        default=30.0,
        ge=1.0,
        le=600,
        validation_alias="MACAW_SESSION_INIT_TIMEOUT_S",
    )
    silence_timeout_s: float = Field(
        default=30.0,
        ge=1.0,
        le=600,
        validation_alias="MACAW_SESSION_SILENCE_TIMEOUT_S",
    )
    hold_timeout_s: float = Field(
        default=300.0,
        ge=1.0,
        le=3600,
        validation_alias="MACAW_SESSION_HOLD_TIMEOUT_S",
    )
    closing_timeout_s: float = Field(
        default=2.0,
        ge=1.0,
        le=60,
        validation_alias="MACAW_SESSION_CLOSING_TIMEOUT_S",
    )
    ring_buffer_force_commit_threshold: float = Field(
        default=0.90,
        gt=0.5,
        lt=1.0,
        validation_alias="MACAW_SESSION_RING_BUFFER_FORCE_COMMIT_THRESHOLD",
    )
    cross_segment_max_tokens: int = Field(
        default=224,
        ge=1,
        le=2048,
        validation_alias="MACAW_SESSION_CROSS_SEGMENT_MAX_TOKENS",
    )


class SchedulerSettings(BaseSettings):
    """Scheduler dispatch tuning (timeouts, batching, aging)."""

    model_config = SettingsConfigDict(extra="ignore", populate_by_name=True)

    min_grpc_timeout_s: float = Field(
        default=30.0, gt=0, le=600, validation_alias="MACAW_SCHEDULER_MIN_GRPC_TIMEOUT_S"
    )
    timeout_factor: float = Field(
        default=2.0, gt=0, le=10, validation_alias="MACAW_SCHEDULER_TIMEOUT_FACTOR"
    )
    shutdown_timeout_s: float = Field(
        default=10.0, gt=0, le=120, validation_alias="MACAW_SCHEDULER_SHUTDOWN_TIMEOUT_S"
    )
    aging_threshold_s: float = Field(
        default=30.0, gt=0, le=300, validation_alias="MACAW_SCHEDULER_AGING_THRESHOLD_S"
    )
    batch_accumulate_ms: float = Field(
        default=50.0, gt=0, le=5000, validation_alias="MACAW_SCHEDULER_BATCH_ACCUMULATE_MS"
    )
    batch_max_size: int = Field(
        default=8, ge=1, le=64, validation_alias="MACAW_SCHEDULER_BATCH_MAX_SIZE"
    )
    no_worker_backoff_s: float = Field(
        default=0.1, gt=0, le=10, validation_alias="MACAW_SCHEDULER_NO_WORKER_BACKOFF_S"
    )
    dequeue_poll_interval_s: float = Field(
        default=0.5, gt=0, le=10, validation_alias="MACAW_SCHEDULER_DEQUEUE_POLL_INTERVAL_S"
    )
    latency_cleanup_interval_s: float = Field(
        default=30.0, gt=0, le=600, validation_alias="MACAW_SCHEDULER_LATENCY_CLEANUP_INTERVAL_S"
    )
    latency_ttl_s: float = Field(
        default=60.0, gt=0, le=600, validation_alias="MACAW_SCHEDULER_LATENCY_TTL_S"
    )
    cancel_propagation_timeout_s: float = Field(
        default=0.1, gt=0, le=10, validation_alias="MACAW_SCHEDULER_CANCEL_PROPAGATION_TIMEOUT_S"
    )


class PreprocessingSettings(BaseSettings):
    """Audio preprocessing pipeline tuning.

    Controls DC-remove cutoff frequency and gain normalization target.
    These defaults match the pipeline config in ``macaw.config.preprocessing``.
    """

    model_config = SettingsConfigDict(extra="ignore", populate_by_name=True)

    dc_cutoff_hz: int = Field(
        default=DEFAULT_DC_CUTOFF_HZ,
        ge=1,
        le=500,
        validation_alias="MACAW_PREPROCESSING_DC_CUTOFF_HZ",
    )
    target_dbfs: float = Field(
        default=DEFAULT_TARGET_DBFS,
        ge=-60.0,
        le=0.0,
        validation_alias="MACAW_PREPROCESSING_TARGET_DBFS",
    )


class PostProcessingSettings(BaseSettings):
    """Text post-processing pipeline tuning."""

    model_config = SettingsConfigDict(extra="ignore", populate_by_name=True)

    itn_default_language: str = Field(
        default=DEFAULT_ITN_LANGUAGE,
        validation_alias=AliasChoices("MACAW_ITN_DEFAULT_LANGUAGE", "MACAW_ITN_LANGUAGE"),
    )


class CodecSettings(BaseSettings):
    """Audio codec encoding settings."""

    model_config = SettingsConfigDict(extra="ignore", populate_by_name=True)

    opus_bitrate: int = Field(
        default=64000,
        ge=6000,
        le=512000,
        validation_alias="MACAW_CODEC_OPUS_BITRATE",
    )


class EffectsSettings(BaseSettings):
    """Post-synthesis audio effects settings."""

    model_config = SettingsConfigDict(extra="ignore", populate_by_name=True)

    pitch_shift_max_semitones: float = Field(
        default=12.0,
        ge=1.0,
        le=24.0,
        validation_alias="MACAW_EFFECTS_PITCH_SHIFT_MAX_SEMITONES",
    )


class MacawSettings(BaseSettings):
    """Root settings — aggregates all subsystem settings.

    Loads ``.env`` from the current directory when present.
    """

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    server: ServerSettings = Field(default_factory=ServerSettings)
    tts: TTSSettings = Field(default_factory=TTSSettings)
    cli: CLISettings = Field(default_factory=CLISettings)
    worker: WorkerSettings = Field(default_factory=WorkerSettings)
    worker_lifecycle: WorkerLifecycleSettings = Field(default_factory=WorkerLifecycleSettings)
    grpc: GRPCSettings = Field(default_factory=GRPCSettings)
    session: SessionSettings = Field(default_factory=SessionSettings)
    scheduler: SchedulerSettings = Field(default_factory=SchedulerSettings)
    vad: VADSettings = Field(default_factory=VADSettings)
    preprocessing: PreprocessingSettings = Field(default_factory=PreprocessingSettings)
    postprocessing: PostProcessingSettings = Field(default_factory=PostProcessingSettings)
    codec: CodecSettings = Field(default_factory=CodecSettings)
    effects: EffectsSettings = Field(default_factory=EffectsSettings)


@lru_cache(maxsize=1)
def get_settings() -> MacawSettings:
    """Return the singleton ``MacawSettings`` instance.

    The result is cached — subsequent calls return the same object.
    Call ``get_settings.cache_clear()`` in tests to reset.
    """
    return MacawSettings()
