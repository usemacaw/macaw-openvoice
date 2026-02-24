"""Tests for auto-installation of engine dependencies in `macaw pull`."""

from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock, patch

from macaw.cli.pull import _install_engine_deps

# Patch targets match lazy imports inside _install_engine_deps
_VENV_MANAGER = "macaw.backends.venv_manager.VenvManager"
_GET_SETTINGS = "macaw.config.settings.get_settings"


class TestInstallEngineDeps:
    @patch(_GET_SETTINGS)
    @patch(_VENV_MANAGER)
    def test_install_skipped_when_venv_exists(
        self, mock_cls: MagicMock, mock_get: MagicMock
    ) -> None:
        """Engine venv already exists → returns True, no provisioning."""
        mock_get.return_value.backend.venv_base_path = Path("/tmp/venvs")
        mock_get.return_value.backend.uv_path = "uv"
        mock_cls.return_value.exists.return_value = True

        result = _install_engine_deps("faster-whisper")

        assert result is True
        mock_cls.return_value.exists.assert_called_once_with("faster-whisper")
        mock_cls.return_value.provision.assert_not_called()

    @patch(_GET_SETTINGS)
    @patch(_VENV_MANAGER)
    def test_install_provisions_when_missing(
        self, mock_cls: MagicMock, mock_get: MagicMock
    ) -> None:
        """Engine venv missing → provisions and returns True."""
        mock_get.return_value.backend.venv_base_path = Path("/tmp/venvs")
        mock_get.return_value.backend.uv_path = "uv"
        mock_cls.return_value.exists.return_value = False
        mock_cls.return_value.provision.return_value = Path("/tmp/venvs/kokoro/bin/python")

        result = _install_engine_deps("kokoro")

        assert result is True
        mock_cls.return_value.provision.assert_called_once_with("kokoro")

    @patch(_GET_SETTINGS)
    @patch(_VENV_MANAGER)
    def test_install_returns_false_on_provision_failure(
        self, mock_cls: MagicMock, mock_get: MagicMock
    ) -> None:
        """Provisioning failure → returns False, does not raise."""
        from macaw.exceptions import VenvProvisionError

        mock_get.return_value.backend.venv_base_path = Path("/tmp/venvs")
        mock_get.return_value.backend.uv_path = "uv"
        mock_cls.return_value.exists.return_value = False
        mock_cls.return_value.provision.side_effect = VenvProvisionError(
            "faster-whisper", "uv not found"
        )

        result = _install_engine_deps("faster-whisper")

        assert result is False

    @patch(_GET_SETTINGS)
    @patch(_VENV_MANAGER)
    def test_install_unknown_engine_provisions(
        self, mock_cls: MagicMock, mock_get: MagicMock
    ) -> None:
        """Unknown engine → still attempts provision (VenvManager handles extras)."""
        mock_get.return_value.backend.venv_base_path = Path("/tmp/venvs")
        mock_get.return_value.backend.uv_path = "uv"
        mock_cls.return_value.exists.return_value = False
        mock_cls.return_value.provision.return_value = Path(
            "/tmp/venvs/some-future-engine/bin/python"
        )

        result = _install_engine_deps("some-future-engine")

        assert result is True

    def test_already_installed_model_still_installs_deps(self) -> None:
        """Model already downloaded but engine missing → installs deps."""
        from click.testing import CliRunner

        from macaw.cli.pull import pull

        runner = CliRunner()
        mock_catalog = MagicMock()
        mock_entry = MagicMock()
        mock_entry.engine = "kokoro"
        mock_catalog.get.return_value = mock_entry

        mock_downloader = MagicMock()
        mock_downloader.is_installed.return_value = True

        with (
            patch("macaw.registry.catalog.ModelCatalog", return_value=mock_catalog),
            patch("macaw.registry.downloader.ModelDownloader", return_value=mock_downloader),
            patch("macaw.cli.pull._install_engine_deps") as mock_install,
        ):
            result = runner.invoke(pull, ["kokoro-v1"])

        assert result.exit_code == 0
        mock_install.assert_called_once_with("kokoro")
