"""Tests for auto-installation of engine dependencies in `macaw pull`."""

from __future__ import annotations

from unittest.mock import patch

from macaw.cli.pull import _install_engine_deps


class TestInstallEngineDeps:
    def test_install_skipped_when_engine_available(self) -> None:
        """Engine already installed → no subprocess call, returns True."""
        with (
            patch("macaw.cli.pull.is_engine_available", return_value=True) as mock_avail,
            patch("macaw.cli.pull.subprocess.run") as mock_run,
        ):
            result = _install_engine_deps("faster-whisper")

        assert result is True
        mock_avail.assert_called_once_with("faster-whisper")
        mock_run.assert_not_called()

    def test_install_runs_pip_when_engine_missing(self) -> None:
        """Engine not installed → runs pip install with correct extra."""
        with (
            patch("macaw.cli.pull.is_engine_available", return_value=False),
            patch("macaw.cli.pull.subprocess.run") as mock_run,
        ):
            mock_run.return_value.returncode = 0
            result = _install_engine_deps("kokoro")

        assert result is True
        mock_run.assert_called_once()
        cmd = mock_run.call_args[0][0]
        assert "macaw-openvoice[kokoro]" in cmd[-1]

    def test_install_returns_false_on_pip_failure(self) -> None:
        """pip failure → returns False, does not raise."""
        with (
            patch("macaw.cli.pull.is_engine_available", return_value=False),
            patch("macaw.cli.pull.subprocess.run") as mock_run,
        ):
            mock_run.return_value.returncode = 1
            result = _install_engine_deps("faster-whisper")

        assert result is False

    def test_install_unknown_engine_skipped(self) -> None:
        """Unknown engine → is_engine_available returns True, no subprocess."""
        with patch("macaw.cli.pull.subprocess.run") as mock_run:
            result = _install_engine_deps("some-future-engine")

        assert result is True
        mock_run.assert_not_called()

    def test_already_installed_model_still_installs_deps(self) -> None:
        """Model already downloaded but engine missing → installs deps."""
        from unittest.mock import MagicMock

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
