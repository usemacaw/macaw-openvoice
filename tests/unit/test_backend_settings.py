"""Tests for BackendSettings in macaw.config.settings."""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

from macaw.config.settings import BackendSettings, MacawSettings, get_settings

if TYPE_CHECKING:
    import pytest


class TestBackendSettingsDefaults:
    def test_default_venv_dir(self) -> None:
        s = BackendSettings()
        assert s.venv_dir == "~/.cache/macaw/venvs"

    def test_default_auto_provision(self) -> None:
        s = BackendSettings()
        assert s.auto_provision is True

    def test_default_uv_path(self) -> None:
        s = BackendSettings()
        assert s.uv_path == "uv"

    def test_venv_base_path_expands_tilde(self) -> None:
        s = BackendSettings()
        result = s.venv_base_path
        assert isinstance(result, Path)
        assert "~" not in str(result)
        assert str(result).endswith("/.cache/macaw/venvs")


class TestBackendSettingsEnvOverrides:
    def test_venv_dir_from_env(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("MACAW_VENV_DIR", "/opt/macaw/venvs")
        s = BackendSettings()
        assert s.venv_dir == "/opt/macaw/venvs"
        assert s.venv_base_path == Path("/opt/macaw/venvs")

    def test_auto_provision_from_env(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("MACAW_BACKEND_AUTO_PROVISION", "false")
        s = BackendSettings()
        assert s.auto_provision is False

    def test_uv_path_from_env(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("MACAW_UV_PATH", "/usr/local/bin/uv")
        s = BackendSettings()
        assert s.uv_path == "/usr/local/bin/uv"


class TestBackendSettingsOnMacawSettings:
    def test_accessor_returns_valid_instance(self) -> None:
        settings = MacawSettings()
        assert isinstance(settings.backend, BackendSettings)

    def test_get_settings_includes_backend(self) -> None:
        get_settings.cache_clear()
        try:
            settings = get_settings()
            assert isinstance(settings.backend, BackendSettings)
            assert settings.backend.venv_dir == "~/.cache/macaw/venvs"
        finally:
            get_settings.cache_clear()
