"""Tests for `macaw backends` CLI commands."""

from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock, patch

from click.testing import CliRunner

from macaw.cli.main import cli
from macaw.exceptions import VenvProvisionError

# Patch targets — these are lazy-imported inside click commands,
# so we patch at the source module level.
_VENV_MANAGER = "macaw.backends.venv_manager.VenvManager"
_GET_SETTINGS = "macaw.config.settings.get_settings"


def _mock_settings(monkeypatch: MagicMock | None = None) -> MagicMock:
    """Create a mock get_settings() returning valid BackendSettings."""
    mock = MagicMock()
    mock.return_value.backend.venv_base_path = Path("/tmp/venvs")
    mock.return_value.backend.uv_path = "uv"
    return mock


class TestBackendsInstall:
    def test_install_unknown_engine(self) -> None:
        runner = CliRunner()
        result = runner.invoke(cli, ["backends", "install", "nonexistent-engine"])
        assert result.exit_code != 0

    @patch(_GET_SETTINGS)
    @patch(_VENV_MANAGER)
    def test_install_skips_existing(self, mock_cls: MagicMock, mock_get: MagicMock) -> None:
        mock_get.return_value.backend.venv_base_path = Path("/tmp/venvs")
        mock_get.return_value.backend.uv_path = "uv"
        mock_cls.return_value.exists.return_value = True

        runner = CliRunner()
        result = runner.invoke(cli, ["backends", "install", "kokoro"])
        assert result.exit_code == 0
        assert "already exists" in result.output

    @patch(_GET_SETTINGS)
    @patch(_VENV_MANAGER)
    def test_install_provisions_new(self, mock_cls: MagicMock, mock_get: MagicMock) -> None:
        mock_get.return_value.backend.venv_base_path = Path("/tmp/venvs")
        mock_get.return_value.backend.uv_path = "uv"
        mock_cls.return_value.exists.return_value = False
        mock_cls.return_value.provision.return_value = Path("/tmp/venvs/kokoro/bin/python")

        runner = CliRunner()
        result = runner.invoke(cli, ["backends", "install", "kokoro"])
        assert result.exit_code == 0
        assert "provisioned" in result.output.lower()

    @patch(_GET_SETTINGS)
    @patch(_VENV_MANAGER)
    def test_install_handles_error(self, mock_cls: MagicMock, mock_get: MagicMock) -> None:
        mock_get.return_value.backend.venv_base_path = Path("/tmp/venvs")
        mock_get.return_value.backend.uv_path = "uv"
        mock_cls.return_value.exists.return_value = False
        mock_cls.return_value.provision.side_effect = VenvProvisionError("kokoro", "uv not found")

        runner = CliRunner()
        result = runner.invoke(cli, ["backends", "install", "kokoro"])
        assert result.exit_code != 0


class TestBackendsList:
    @patch(_GET_SETTINGS)
    @patch(_VENV_MANAGER)
    def test_list_shows_engines(self, mock_cls: MagicMock, mock_get: MagicMock) -> None:
        mock_get.return_value.backend.venv_base_path = Path("/tmp/venvs")
        mock_get.return_value.backend.uv_path = "uv"
        mock_cls.return_value.exists.return_value = False
        mock_cls.return_value.venv_dir.return_value = Path("/tmp/venvs/test")

        runner = CliRunner()
        result = runner.invoke(cli, ["backends", "list"])
        assert result.exit_code == 0
        assert "faster-whisper" in result.output
        assert "kokoro" in result.output


class TestBackendsRemove:
    def test_remove_unknown_engine(self) -> None:
        """Remove must validate engine against ENGINE_EXTRAS."""
        runner = CliRunner()
        result = runner.invoke(cli, ["backends", "remove", "nonexistent-engine", "-y"])
        assert result.exit_code != 0
        assert "unknown engine" in result.output.lower()

    @patch(_GET_SETTINGS)
    @patch(_VENV_MANAGER)
    def test_remove_nonexistent(self, mock_cls: MagicMock, mock_get: MagicMock) -> None:
        mock_get.return_value.backend.venv_base_path = Path("/tmp/venvs")
        mock_get.return_value.backend.uv_path = "uv"
        mock_cls.return_value.exists.return_value = False

        runner = CliRunner()
        result = runner.invoke(cli, ["backends", "remove", "kokoro", "-y"])
        assert result.exit_code == 0
        assert "no venv found" in result.output.lower()

    @patch(_GET_SETTINGS)
    @patch(_VENV_MANAGER)
    def test_remove_existing_with_yes(self, mock_cls: MagicMock, mock_get: MagicMock) -> None:
        mock_get.return_value.backend.venv_base_path = Path("/tmp/venvs")
        mock_get.return_value.backend.uv_path = "uv"
        mock_cls.return_value.exists.return_value = True

        runner = CliRunner()
        result = runner.invoke(cli, ["backends", "remove", "kokoro", "-y"])
        assert result.exit_code == 0
        assert "removed" in result.output.lower()
        mock_cls.return_value.remove.assert_called_once_with("kokoro")


class TestBackendsStatus:
    def test_status_unknown_engine(self) -> None:
        """Status must validate engine against ENGINE_EXTRAS."""
        runner = CliRunner()
        result = runner.invoke(cli, ["backends", "status", "nonexistent-engine"])
        assert result.exit_code != 0
        assert "unknown engine" in result.output.lower()

    @patch(_GET_SETTINGS)
    @patch(_VENV_MANAGER)
    def test_status_missing(self, mock_cls: MagicMock, mock_get: MagicMock) -> None:
        mock_get.return_value.backend.venv_base_path = Path("/tmp/venvs")
        mock_get.return_value.backend.uv_path = "uv"
        mock_cls.return_value.exists.return_value = False

        runner = CliRunner()
        result = runner.invoke(cli, ["backends", "status", "kokoro"])
        assert result.exit_code == 0
        assert "no provisioned venv" in result.output.lower()

    @patch(_GET_SETTINGS)
    @patch(_VENV_MANAGER)
    def test_status_provisioned(self, mock_cls: MagicMock, mock_get: MagicMock) -> None:
        mock_get.return_value.backend.venv_base_path = Path("/tmp/venvs")
        mock_get.return_value.backend.uv_path = "uv"
        mock_cls.return_value.exists.return_value = True
        mock_cls.return_value.venv_dir.return_value = Path("/tmp/venvs/kokoro")
        mock_cls.return_value.venv_python.return_value = Path("/tmp/venvs/kokoro/bin/python")
        mock_cls.return_value.read_marker.return_value = {
            "engine": "kokoro",
            "extra": "kokoro",
            "provisioned_at": "2026-01-01T00:00:00Z",
        }
        mock_cls.return_value.is_engine_available_in_venv.return_value = True

        runner = CliRunner()
        result = runner.invoke(cli, ["backends", "status", "kokoro"])
        assert result.exit_code == 0
        assert "kokoro" in result.output
        assert "Available:  yes" in result.output
