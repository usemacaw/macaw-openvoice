"""Tests for macaw.config.settings — centralized pydantic-settings configuration."""

from __future__ import annotations

from pathlib import Path

import pytest
from pydantic import ValidationError

from macaw.config.settings import (
    CLISettings,
    MacawSettings,
    ServerSettings,
    TTSSettings,
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

    def test_extra_env_vars_ignored(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("MACAW_UNKNOWN_VAR", "ignored")
        settings = MacawSettings()
        assert not hasattr(settings, "MACAW_UNKNOWN_VAR")


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
