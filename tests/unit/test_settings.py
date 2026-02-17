"""Tests for macaw.config.settings — centralized pydantic-settings configuration."""

from __future__ import annotations

from pathlib import Path

import pytest
from pydantic import ValidationError

from macaw.config.settings import (
    CLISettings,
    GRPCSettings,
    MacawSettings,
    SchedulerSettings,
    ServerSettings,
    SessionSettings,
    TTSSettings,
    VADSettings,
    WorkerLifecycleSettings,
    WorkerSettings,
    get_settings,
)


class TestServerSettingsDefaults:
    """Verify ServerSettings defaults match the previous os.environ.get defaults."""

    def test_default_host(self) -> None:
        s = ServerSettings()
        assert s.host == "127.0.0.1"

    def test_default_port(self) -> None:
        s = ServerSettings()
        assert s.port == 8000

    def test_default_max_file_size_mb(self) -> None:
        s = ServerSettings()
        assert s.max_file_size_mb == 25

    def test_max_file_size_bytes_property(self) -> None:
        s = ServerSettings()
        assert s.max_file_size_bytes == 25 * 1024 * 1024

    def test_default_retry_after(self) -> None:
        s = ServerSettings()
        assert s.retry_after_s == "5"

    def test_default_ws_max_frame_size(self) -> None:
        s = ServerSettings()
        assert s.ws_max_frame_size_bytes == 1_048_576

    def test_default_ws_inactivity_timeout(self) -> None:
        s = ServerSettings()
        assert s.ws_inactivity_timeout_s == 60.0

    def test_default_ws_heartbeat_interval(self) -> None:
        s = ServerSettings()
        assert s.ws_heartbeat_interval_s == 10.0

    def test_default_ws_check_interval(self) -> None:
        s = ServerSettings()
        assert s.ws_check_interval_s == 5.0

    def test_default_cors_origins_empty(self) -> None:
        s = ServerSettings()
        assert s.cors_origins == ""

    def test_cors_origins_list_empty_string(self) -> None:
        s = ServerSettings()
        assert s.cors_origins_list == []

    def test_cors_origins_list_single(self) -> None:
        s = ServerSettings(cors_origins="http://localhost:3000")  # type: ignore[call-arg]
        assert s.cors_origins_list == ["http://localhost:3000"]

    def test_cors_origins_list_multiple(self) -> None:
        s = ServerSettings(cors_origins="http://localhost:3000, https://example.com")  # type: ignore[call-arg]
        assert s.cors_origins_list == ["http://localhost:3000", "https://example.com"]

    def test_cors_origins_list_strips_whitespace(self) -> None:
        s = ServerSettings(cors_origins=" http://a.com , http://b.com ")  # type: ignore[call-arg]
        assert s.cors_origins_list == ["http://a.com", "http://b.com"]

    def test_cors_origins_list_ignores_empty_segments(self) -> None:
        s = ServerSettings(cors_origins="http://a.com,,http://b.com,")  # type: ignore[call-arg]
        assert s.cors_origins_list == ["http://a.com", "http://b.com"]


class TestTTSSettingsDefaults:
    def test_default_grpc_timeout(self) -> None:
        s = TTSSettings()
        assert s.grpc_timeout_s == 60.0

    def test_default_list_voices_timeout(self) -> None:
        s = TTSSettings()
        assert s.list_voices_timeout_s == 10.0


class TestCLISettingsDefaults:
    def test_default_server_url(self) -> None:
        s = CLISettings()
        assert s.server_url == "http://localhost:8000"

    def test_default_http_timeout(self) -> None:
        s = CLISettings()
        assert s.http_timeout_s == 120.0


class TestWorkerSettingsDefaults:
    def test_default_models_dir(self) -> None:
        s = WorkerSettings()
        assert s.models_dir == "~/.macaw/models"

    def test_default_worker_base_port(self) -> None:
        s = WorkerSettings()
        assert s.worker_base_port == 50051

    def test_models_path_expands_tilde(self) -> None:
        s = WorkerSettings()
        assert s.models_path == Path("~/.macaw/models").expanduser()
        assert "~" not in str(s.models_path)

    def test_default_stt_max_cancelled_requests(self) -> None:
        s = WorkerSettings()
        assert s.stt_max_cancelled_requests == 10_000


class TestServerSettingsValidation:
    def test_port_too_low_raises(self) -> None:
        with pytest.raises(ValidationError, match="port"):
            ServerSettings(port=0)  # type: ignore[call-arg]

    def test_port_too_high_raises(self) -> None:
        with pytest.raises(ValidationError, match="port"):
            ServerSettings(port=70000)  # type: ignore[call-arg]

    def test_max_file_size_mb_too_low_raises(self) -> None:
        with pytest.raises(ValidationError, match="max_file_size_mb"):
            ServerSettings(max_file_size_mb=0)  # type: ignore[call-arg]

    def test_ws_heartbeat_must_be_lt_inactivity(self) -> None:
        with pytest.raises(
            ValidationError,
            match="ws_heartbeat_interval_s must be < ws_inactivity_timeout_s",
        ):
            ServerSettings(
                ws_heartbeat_interval_s=60.0,  # type: ignore[call-arg]
                ws_inactivity_timeout_s=60.0,  # type: ignore[call-arg]
            )

    def test_ws_heartbeat_greater_than_inactivity_raises(self) -> None:
        with pytest.raises(ValidationError):
            ServerSettings(
                ws_heartbeat_interval_s=120.0,  # type: ignore[call-arg]
                ws_inactivity_timeout_s=60.0,  # type: ignore[call-arg]
            )


class TestWorkerSettingsValidation:
    def test_worker_base_port_too_low_raises(self) -> None:
        with pytest.raises(ValidationError, match="worker_base_port"):
            WorkerSettings(worker_base_port=80)  # type: ignore[call-arg]

    def test_worker_base_port_too_high_raises(self) -> None:
        with pytest.raises(ValidationError, match="worker_base_port"):
            WorkerSettings(worker_base_port=70000)  # type: ignore[call-arg]


class TestTTSSettingsValidation:
    def test_grpc_timeout_zero_raises(self) -> None:
        with pytest.raises(ValidationError, match="grpc_timeout_s"):
            TTSSettings(grpc_timeout_s=0)  # type: ignore[call-arg]

    def test_grpc_timeout_negative_raises(self) -> None:
        with pytest.raises(ValidationError, match="grpc_timeout_s"):
            TTSSettings(grpc_timeout_s=-1.0)  # type: ignore[call-arg]


class TestEnvOverrides:
    """Verify env vars override defaults via monkeypatch."""

    def test_port_from_env(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("MACAW_PORT", "9999")
        s = ServerSettings()
        assert s.port == 9999

    def test_host_from_env(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("MACAW_HOST", "0.0.0.0")
        s = ServerSettings()
        assert s.host == "0.0.0.0"

    def test_models_dir_from_env(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("MACAW_MODELS_DIR", "/opt/models")
        s = WorkerSettings()
        assert s.models_dir == "/opt/models"
        assert s.models_path == Path("/opt/models")

    def test_tts_grpc_timeout_from_env(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("MACAW_TTS_GRPC_TIMEOUT_S", "30.0")
        s = TTSSettings()
        assert s.grpc_timeout_s == 30.0

    def test_cli_server_url_from_env(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("MACAW_SERVER_URL", "https://api.example.com")
        s = CLISettings()
        assert s.server_url == "https://api.example.com"

    def test_invalid_port_from_env_raises(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("MACAW_PORT", "not_a_number")
        with pytest.raises(ValidationError):
            ServerSettings()

    def test_max_file_size_bytes_reflects_env(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("MACAW_MAX_FILE_SIZE_MB", "100")
        s = ServerSettings()
        assert s.max_file_size_bytes == 100 * 1024 * 1024


class TestMacawSettingsRoot:
    """Test the root MacawSettings aggregator."""

    def test_subsettings_are_populated(self) -> None:
        settings = MacawSettings()
        assert isinstance(settings.server, ServerSettings)
        assert isinstance(settings.tts, TTSSettings)
        assert isinstance(settings.cli, CLISettings)
        assert isinstance(settings.worker, WorkerSettings)
        assert isinstance(settings.worker_lifecycle, WorkerLifecycleSettings)
        assert isinstance(settings.grpc, GRPCSettings)
        assert isinstance(settings.session, SessionSettings)
        assert isinstance(settings.scheduler, SchedulerSettings)
        assert isinstance(settings.vad, VADSettings)

    def test_extra_env_vars_ignored(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("MACAW_UNKNOWN_VAR", "ignored")
        settings = MacawSettings()
        assert not hasattr(settings, "MACAW_UNKNOWN_VAR")


class TestWorkerLifecycleSettingsDefaults:
    def test_default_max_crashes_in_window(self) -> None:
        s = WorkerLifecycleSettings()
        assert s.max_crashes_in_window == 3

    def test_default_crash_window_s(self) -> None:
        s = WorkerLifecycleSettings()
        assert s.crash_window_s == 60.0

    def test_default_health_probe_initial_delay_s(self) -> None:
        s = WorkerLifecycleSettings()
        assert s.health_probe_initial_delay_s == 0.5

    def test_default_health_probe_max_delay_s(self) -> None:
        s = WorkerLifecycleSettings()
        assert s.health_probe_max_delay_s == 5.0

    def test_default_health_probe_timeout_s(self) -> None:
        s = WorkerLifecycleSettings()
        assert s.health_probe_timeout_s == 120.0

    def test_default_monitor_interval_s(self) -> None:
        s = WorkerLifecycleSettings()
        assert s.monitor_interval_s == 1.0

    def test_default_stop_grace_period_s(self) -> None:
        s = WorkerLifecycleSettings()
        assert s.stop_grace_period_s == 5.0

    def test_default_warmup_steps(self) -> None:
        s = WorkerLifecycleSettings()
        assert s.default_warmup_steps == 3


class TestWorkerLifecycleSettingsValidation:
    def test_max_crashes_zero_raises(self) -> None:
        with pytest.raises(ValidationError, match="max_crashes_in_window"):
            WorkerLifecycleSettings(max_crashes_in_window=0)  # type: ignore[call-arg]

    def test_max_crashes_over_limit_raises(self) -> None:
        with pytest.raises(ValidationError, match="max_crashes_in_window"):
            WorkerLifecycleSettings(max_crashes_in_window=101)  # type: ignore[call-arg]

    def test_crash_window_zero_raises(self) -> None:
        with pytest.raises(ValidationError, match="crash_window_s"):
            WorkerLifecycleSettings(crash_window_s=0)  # type: ignore[call-arg]

    def test_crash_window_over_limit_raises(self) -> None:
        with pytest.raises(ValidationError, match="crash_window_s"):
            WorkerLifecycleSettings(crash_window_s=3601)  # type: ignore[call-arg]

    def test_warmup_steps_negative_raises(self) -> None:
        with pytest.raises(ValidationError, match="default_warmup_steps"):
            WorkerLifecycleSettings(default_warmup_steps=-1)  # type: ignore[call-arg]

    def test_warmup_steps_over_limit_raises(self) -> None:
        with pytest.raises(ValidationError, match="default_warmup_steps"):
            WorkerLifecycleSettings(default_warmup_steps=21)  # type: ignore[call-arg]

    def test_initial_delay_must_be_lt_max_delay(self) -> None:
        with pytest.raises(
            ValidationError,
            match="health_probe_initial_delay_s must be < health_probe_max_delay_s",
        ):
            WorkerLifecycleSettings(
                health_probe_initial_delay_s=5.0,  # type: ignore[call-arg]
                health_probe_max_delay_s=5.0,  # type: ignore[call-arg]
            )

    def test_initial_delay_greater_than_max_raises(self) -> None:
        with pytest.raises(ValidationError):
            WorkerLifecycleSettings(
                health_probe_initial_delay_s=10.0,  # type: ignore[call-arg]
                health_probe_max_delay_s=5.0,  # type: ignore[call-arg]
            )


class TestWorkerLifecycleSettingsEnvOverrides:
    def test_max_crashes_from_env(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("MACAW_WORKER_MAX_CRASHES", "5")
        s = WorkerLifecycleSettings()
        assert s.max_crashes_in_window == 5

    def test_stop_grace_period_from_env(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("MACAW_WORKER_STOP_GRACE_PERIOD_S", "10.0")
        s = WorkerLifecycleSettings()
        assert s.stop_grace_period_s == 10.0

    def test_warmup_steps_from_env(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("MACAW_WORKER_WARMUP_STEPS", "0")
        s = WorkerLifecycleSettings()
        assert s.default_warmup_steps == 0


class TestGRPCSettingsDefaults:
    def test_default_max_batch_message_mb(self) -> None:
        s = GRPCSettings()
        assert s.max_batch_message_mb == 30

    def test_default_max_streaming_message_mb(self) -> None:
        s = GRPCSettings()
        assert s.max_streaming_message_mb == 10

    def test_max_batch_message_bytes_property(self) -> None:
        s = GRPCSettings()
        assert s.max_batch_message_bytes == 30 * 1024 * 1024

    def test_max_streaming_message_bytes_property(self) -> None:
        s = GRPCSettings()
        assert s.max_streaming_message_bytes == 10 * 1024 * 1024


class TestGRPCSettingsValidation:
    def test_batch_mb_zero_raises(self) -> None:
        with pytest.raises(ValidationError, match="max_batch_message_mb"):
            GRPCSettings(max_batch_message_mb=0)  # type: ignore[call-arg]

    def test_batch_mb_over_limit_raises(self) -> None:
        with pytest.raises(ValidationError, match="max_batch_message_mb"):
            GRPCSettings(max_batch_message_mb=501)  # type: ignore[call-arg]

    def test_streaming_mb_zero_raises(self) -> None:
        with pytest.raises(ValidationError, match="max_streaming_message_mb"):
            GRPCSettings(max_streaming_message_mb=0)  # type: ignore[call-arg]

    def test_streaming_mb_over_limit_raises(self) -> None:
        with pytest.raises(ValidationError, match="max_streaming_message_mb"):
            GRPCSettings(max_streaming_message_mb=101)  # type: ignore[call-arg]


class TestGRPCSettingsEnvOverrides:
    def test_batch_mb_from_env(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("MACAW_GRPC_MAX_BATCH_MESSAGE_MB", "50")
        s = GRPCSettings()
        assert s.max_batch_message_mb == 50
        assert s.max_batch_message_bytes == 50 * 1024 * 1024

    def test_streaming_mb_from_env(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("MACAW_GRPC_MAX_STREAMING_MESSAGE_MB", "20")
        s = GRPCSettings()
        assert s.max_streaming_message_mb == 20
        assert s.max_streaming_message_bytes == 20 * 1024 * 1024


class TestSchedulerSettingsDefaults:
    def test_default_min_grpc_timeout(self) -> None:
        s = SchedulerSettings()
        assert s.min_grpc_timeout_s == 30.0

    def test_default_timeout_factor(self) -> None:
        s = SchedulerSettings()
        assert s.timeout_factor == 2.0

    def test_default_shutdown_timeout(self) -> None:
        s = SchedulerSettings()
        assert s.shutdown_timeout_s == 10.0

    def test_default_aging_threshold(self) -> None:
        s = SchedulerSettings()
        assert s.aging_threshold_s == 30.0

    def test_default_batch_accumulate_ms(self) -> None:
        s = SchedulerSettings()
        assert s.batch_accumulate_ms == 50.0

    def test_default_batch_max_size(self) -> None:
        s = SchedulerSettings()
        assert s.batch_max_size == 8


class TestSchedulerSettingsValidation:
    def test_min_grpc_timeout_zero_raises(self) -> None:
        with pytest.raises(ValidationError, match="min_grpc_timeout_s"):
            SchedulerSettings(min_grpc_timeout_s=0)  # type: ignore[call-arg]

    def test_timeout_factor_zero_raises(self) -> None:
        with pytest.raises(ValidationError, match="timeout_factor"):
            SchedulerSettings(timeout_factor=0)  # type: ignore[call-arg]

    def test_timeout_factor_over_limit_raises(self) -> None:
        with pytest.raises(ValidationError, match="timeout_factor"):
            SchedulerSettings(timeout_factor=11)  # type: ignore[call-arg]

    def test_batch_max_size_zero_raises(self) -> None:
        with pytest.raises(ValidationError, match="batch_max_size"):
            SchedulerSettings(batch_max_size=0)  # type: ignore[call-arg]

    def test_batch_max_size_over_limit_raises(self) -> None:
        with pytest.raises(ValidationError, match="batch_max_size"):
            SchedulerSettings(batch_max_size=65)  # type: ignore[call-arg]


class TestSchedulerSettingsEnvOverrides:
    def test_min_grpc_timeout_from_env(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("MACAW_SCHEDULER_MIN_GRPC_TIMEOUT_S", "60.0")
        s = SchedulerSettings()
        assert s.min_grpc_timeout_s == 60.0

    def test_aging_threshold_from_env(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("MACAW_SCHEDULER_AGING_THRESHOLD_S", "15.0")
        s = SchedulerSettings()
        assert s.aging_threshold_s == 15.0

    def test_batch_max_size_from_env(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("MACAW_SCHEDULER_BATCH_MAX_SIZE", "16")
        s = SchedulerSettings()
        assert s.batch_max_size == 16


class TestSchedulerSettingsExtendedDefaults:
    """Defaults for the new scheduler dispatch tuning fields."""

    def test_default_no_worker_backoff(self) -> None:
        s = SchedulerSettings()
        assert s.no_worker_backoff_s == 0.1

    def test_default_dequeue_poll_interval(self) -> None:
        s = SchedulerSettings()
        assert s.dequeue_poll_interval_s == 0.5

    def test_default_latency_cleanup_interval(self) -> None:
        s = SchedulerSettings()
        assert s.latency_cleanup_interval_s == 30.0

    def test_default_latency_ttl(self) -> None:
        s = SchedulerSettings()
        assert s.latency_ttl_s == 60.0

    def test_default_cancel_propagation_timeout(self) -> None:
        s = SchedulerSettings()
        assert s.cancel_propagation_timeout_s == 0.1


class TestSchedulerSettingsExtendedValidation:
    def test_no_worker_backoff_zero_raises(self) -> None:
        with pytest.raises(ValidationError, match="no_worker_backoff_s"):
            SchedulerSettings(no_worker_backoff_s=0)  # type: ignore[call-arg]

    def test_dequeue_poll_interval_zero_raises(self) -> None:
        with pytest.raises(ValidationError, match="dequeue_poll_interval_s"):
            SchedulerSettings(dequeue_poll_interval_s=0)  # type: ignore[call-arg]

    def test_latency_ttl_zero_raises(self) -> None:
        with pytest.raises(ValidationError, match="latency_ttl_s"):
            SchedulerSettings(latency_ttl_s=0)  # type: ignore[call-arg]

    def test_cancel_propagation_timeout_over_limit_raises(self) -> None:
        with pytest.raises(ValidationError, match="cancel_propagation_timeout_s"):
            SchedulerSettings(cancel_propagation_timeout_s=11)  # type: ignore[call-arg]


class TestSchedulerSettingsExtendedEnvOverrides:
    def test_no_worker_backoff_from_env(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("MACAW_SCHEDULER_NO_WORKER_BACKOFF_S", "0.5")
        s = SchedulerSettings()
        assert s.no_worker_backoff_s == 0.5

    def test_latency_ttl_from_env(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("MACAW_SCHEDULER_LATENCY_TTL_S", "120.0")
        s = SchedulerSettings()
        assert s.latency_ttl_s == 120.0

    def test_cancel_propagation_timeout_from_env(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("MACAW_SCHEDULER_CANCEL_PROPAGATION_TIMEOUT_S", "0.5")
        s = SchedulerSettings()
        assert s.cancel_propagation_timeout_s == 0.5


class TestSessionSettingsDefaults:
    def test_default_ring_buffer_duration(self) -> None:
        s = SessionSettings()
        assert s.ring_buffer_duration_s == 60.0

    def test_default_recovery_timeout(self) -> None:
        s = SessionSettings()
        assert s.recovery_timeout_s == 10.0

    def test_default_drain_stream_timeout(self) -> None:
        s = SessionSettings()
        assert s.drain_stream_timeout_s == 5.0

    def test_default_flush_and_close_timeout(self) -> None:
        s = SessionSettings()
        assert s.flush_and_close_timeout_s == 2.0

    def test_default_backpressure_max_backlog(self) -> None:
        s = SessionSettings()
        assert s.backpressure_max_backlog_s == 10.0

    def test_default_backpressure_rate_limit_threshold(self) -> None:
        s = SessionSettings()
        assert s.backpressure_rate_limit_threshold == 1.2


class TestSessionSettingsValidation:
    def test_ring_buffer_duration_zero_raises(self) -> None:
        with pytest.raises(ValidationError, match="ring_buffer_duration_s"):
            SessionSettings(ring_buffer_duration_s=0)  # type: ignore[call-arg]

    def test_ring_buffer_duration_over_limit_raises(self) -> None:
        with pytest.raises(ValidationError, match="ring_buffer_duration_s"):
            SessionSettings(ring_buffer_duration_s=601)  # type: ignore[call-arg]

    def test_recovery_timeout_zero_raises(self) -> None:
        with pytest.raises(ValidationError, match="recovery_timeout_s"):
            SessionSettings(recovery_timeout_s=0)  # type: ignore[call-arg]

    def test_drain_stream_timeout_zero_raises(self) -> None:
        with pytest.raises(ValidationError, match="drain_stream_timeout_s"):
            SessionSettings(drain_stream_timeout_s=0)  # type: ignore[call-arg]

    def test_flush_and_close_timeout_zero_raises(self) -> None:
        with pytest.raises(ValidationError, match="flush_and_close_timeout_s"):
            SessionSettings(flush_and_close_timeout_s=0)  # type: ignore[call-arg]

    def test_backpressure_rate_limit_threshold_at_one_raises(self) -> None:
        with pytest.raises(ValidationError, match="backpressure_rate_limit_threshold"):
            SessionSettings(backpressure_rate_limit_threshold=1.0)  # type: ignore[call-arg]

    def test_backpressure_rate_limit_threshold_over_limit_raises(self) -> None:
        with pytest.raises(ValidationError, match="backpressure_rate_limit_threshold"):
            SessionSettings(backpressure_rate_limit_threshold=5.1)  # type: ignore[call-arg]


class TestSessionSettingsEnvOverrides:
    def test_ring_buffer_duration_from_env(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("MACAW_SESSION_RING_BUFFER_DURATION_S", "120.0")
        s = SessionSettings()
        assert s.ring_buffer_duration_s == 120.0

    def test_recovery_timeout_from_env(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("MACAW_SESSION_RECOVERY_TIMEOUT_S", "30.0")
        s = SessionSettings()
        assert s.recovery_timeout_s == 30.0

    def test_drain_stream_timeout_from_env(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("MACAW_SESSION_DRAIN_STREAM_TIMEOUT_S", "10.0")
        s = SessionSettings()
        assert s.drain_stream_timeout_s == 10.0

    def test_backpressure_max_backlog_from_env(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("MACAW_SESSION_BACKPRESSURE_MAX_BACKLOG_S", "20.0")
        s = SessionSettings()
        assert s.backpressure_max_backlog_s == 20.0

    def test_backpressure_rate_limit_from_env(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("MACAW_SESSION_BACKPRESSURE_RATE_LIMIT_THRESHOLD", "1.5")
        s = SessionSettings()
        assert s.backpressure_rate_limit_threshold == 1.5


class TestGetSettingsSingleton:
    """Test the lru_cache singleton."""

    def test_returns_same_instance(self) -> None:
        get_settings.cache_clear()
        s1 = get_settings()
        s2 = get_settings()
        assert s1 is s2

    def test_cache_clear_creates_new_instance(self) -> None:
        get_settings.cache_clear()
        s1 = get_settings()
        get_settings.cache_clear()
        s2 = get_settings()
        assert s1 is not s2


class TestVADSettingsDefaults:
    def test_default_sensitivity(self) -> None:
        s = VADSettings()
        assert s.sensitivity == "normal"

    def test_default_min_speech_duration_ms(self) -> None:
        s = VADSettings()
        assert s.min_speech_duration_ms == 250

    def test_default_min_silence_duration_ms(self) -> None:
        s = VADSettings()
        assert s.min_silence_duration_ms == 300

    def test_default_max_speech_duration_ms(self) -> None:
        s = VADSettings()
        assert s.max_speech_duration_ms == 30_000

    def test_default_energy_threshold_dbfs_none(self) -> None:
        s = VADSettings()
        assert s.energy_threshold_dbfs is None


class TestVADSettingsValidation:
    def test_sensitivity_invalid_raises(self) -> None:
        with pytest.raises(ValidationError, match="sensitivity must be one of"):
            VADSettings(sensitivity="ultra")  # type: ignore[call-arg]

    def test_sensitivity_case_insensitive_upper(self) -> None:
        s = VADSettings(sensitivity="HIGH")  # type: ignore[call-arg]
        assert s.sensitivity == "high"

    def test_sensitivity_case_insensitive_mixed(self) -> None:
        s = VADSettings(sensitivity="Normal")  # type: ignore[call-arg]
        assert s.sensitivity == "normal"

    def test_sensitivity_case_insensitive_low(self) -> None:
        s = VADSettings(sensitivity="LOW")  # type: ignore[call-arg]
        assert s.sensitivity == "low"

    def test_min_speech_duration_ms_too_low_raises(self) -> None:
        with pytest.raises(ValidationError, match="min_speech_duration_ms"):
            VADSettings(min_speech_duration_ms=49)  # type: ignore[call-arg]

    def test_min_speech_duration_ms_too_high_raises(self) -> None:
        with pytest.raises(ValidationError, match="min_speech_duration_ms"):
            VADSettings(min_speech_duration_ms=5001)  # type: ignore[call-arg]

    def test_min_silence_duration_ms_too_low_raises(self) -> None:
        with pytest.raises(ValidationError, match="min_silence_duration_ms"):
            VADSettings(min_silence_duration_ms=49)  # type: ignore[call-arg]

    def test_min_silence_duration_ms_too_high_raises(self) -> None:
        with pytest.raises(ValidationError, match="min_silence_duration_ms"):
            VADSettings(min_silence_duration_ms=5001)  # type: ignore[call-arg]

    def test_max_speech_duration_ms_too_low_raises(self) -> None:
        with pytest.raises(ValidationError, match="max_speech_duration_ms"):
            VADSettings(max_speech_duration_ms=999)  # type: ignore[call-arg]

    def test_max_speech_duration_ms_too_high_raises(self) -> None:
        with pytest.raises(ValidationError, match="max_speech_duration_ms"):
            VADSettings(max_speech_duration_ms=600_001)  # type: ignore[call-arg]

    def test_min_speech_gte_max_raises(self) -> None:
        with pytest.raises(
            ValidationError,
            match="min_speech_duration_ms must be < max_speech_duration_ms",
        ):
            VADSettings(
                min_speech_duration_ms=1000,  # type: ignore[call-arg]
                max_speech_duration_ms=1000,  # type: ignore[call-arg]
            )

    def test_min_speech_greater_than_max_raises(self) -> None:
        with pytest.raises(ValidationError):
            VADSettings(
                min_speech_duration_ms=2000,  # type: ignore[call-arg]
                max_speech_duration_ms=1000,  # type: ignore[call-arg]
            )

    def test_vad_sensitivity_property_returns_enum(self) -> None:
        from macaw._types import VADSensitivity

        s = VADSettings()
        assert s.vad_sensitivity == VADSensitivity.NORMAL

    def test_vad_sensitivity_property_high(self) -> None:
        from macaw._types import VADSensitivity

        s = VADSettings(sensitivity="high")  # type: ignore[call-arg]
        assert s.vad_sensitivity == VADSensitivity.HIGH


class TestVADSettingsEnvOverrides:
    def test_sensitivity_from_env(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("MACAW_VAD_SENSITIVITY", "HIGH")
        s = VADSettings()
        assert s.sensitivity == "high"

    def test_min_speech_from_env(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("MACAW_VAD_MIN_SPEECH_DURATION_MS", "500")
        s = VADSettings()
        assert s.min_speech_duration_ms == 500

    def test_min_silence_from_env(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("MACAW_VAD_MIN_SILENCE_DURATION_MS", "600")
        s = VADSettings()
        assert s.min_silence_duration_ms == 600

    def test_max_speech_from_env(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("MACAW_VAD_MAX_SPEECH_DURATION_MS", "60000")
        s = VADSettings()
        assert s.max_speech_duration_ms == 60_000

    def test_energy_threshold_dbfs_from_env(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("MACAW_VAD_ENERGY_THRESHOLD_DBFS", "-45.0")
        s = VADSettings()
        assert s.energy_threshold_dbfs == -45.0

    def test_energy_threshold_dbfs_too_low_raises(self) -> None:
        with pytest.raises(ValidationError, match="energy_threshold_dbfs"):
            VADSettings(energy_threshold_dbfs=-81.0)  # type: ignore[call-arg]

    def test_energy_threshold_dbfs_too_high_raises(self) -> None:
        with pytest.raises(ValidationError, match="energy_threshold_dbfs"):
            VADSettings(energy_threshold_dbfs=1.0)  # type: ignore[call-arg]

    def test_stt_max_cancelled_requests_from_env(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("MACAW_STT_MAX_CANCELLED_REQUESTS", "50000")
        s = WorkerSettings()
        assert s.stt_max_cancelled_requests == 50_000


class TestWorkerHostSetting:
    def test_worker_host_default(self) -> None:
        s = WorkerSettings()
        assert s.worker_host == "localhost"

    def test_worker_host_from_env(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("MACAW_WORKER_HOST", "10.0.0.5")
        s = WorkerSettings()
        assert s.worker_host == "10.0.0.5"

    def test_worker_host_hostname(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("MACAW_WORKER_HOST", "worker.internal.svc")
        s = WorkerSettings()
        assert s.worker_host == "worker.internal.svc"


class TestCrossSegmentMaxTokensSetting:
    """Tests for MACAW_SESSION_CROSS_SEGMENT_MAX_TOKENS."""

    def test_default_is_224(self) -> None:
        s = SessionSettings()
        assert s.cross_segment_max_tokens == 224

    def test_from_env(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("MACAW_SESSION_CROSS_SEGMENT_MAX_TOKENS", "448")
        s = SessionSettings()
        assert s.cross_segment_max_tokens == 448

    def test_zero_raises(self) -> None:
        with pytest.raises(ValidationError, match="cross_segment_max_tokens"):
            SessionSettings(cross_segment_max_tokens=0)  # type: ignore[call-arg]

    def test_over_limit_raises(self) -> None:
        with pytest.raises(ValidationError, match="cross_segment_max_tokens"):
            SessionSettings(cross_segment_max_tokens=2049)  # type: ignore[call-arg]


class TestSileroThresholdSetting:
    """Tests for MACAW_VAD_SILERO_THRESHOLD."""

    def test_default_is_none(self) -> None:
        s = VADSettings()
        assert s.silero_threshold is None

    def test_from_env(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("MACAW_VAD_SILERO_THRESHOLD", "0.4")
        s = VADSettings()
        assert s.silero_threshold == pytest.approx(0.4)

    def test_zero_raises(self) -> None:
        with pytest.raises(ValidationError, match="silero_threshold"):
            VADSettings(silero_threshold=0.0)  # type: ignore[call-arg]

    def test_one_raises(self) -> None:
        with pytest.raises(ValidationError, match="silero_threshold"):
            VADSettings(silero_threshold=1.0)  # type: ignore[call-arg]
