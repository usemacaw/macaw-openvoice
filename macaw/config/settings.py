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

from pydantic import Field, model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


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
    """TTS gRPC timeout settings."""

    model_config = SettingsConfigDict(extra="ignore", populate_by_name=True)

    grpc_timeout_s: float = Field(default=60.0, gt=0, validation_alias="MACAW_TTS_GRPC_TIMEOUT_S")
    list_voices_timeout_s: float = Field(
        default=10.0, gt=0, validation_alias="MACAW_TTS_LIST_VOICES_TIMEOUT_S"
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
    scheduler: SchedulerSettings = Field(default_factory=SchedulerSettings)


@lru_cache(maxsize=1)
def get_settings() -> MacawSettings:
    """Return the singleton ``MacawSettings`` instance.

    The result is cached — subsequent calls return the same object.
    Call ``get_settings.cache_clear()`` in tests to reset.
    """
    return MacawSettings()
