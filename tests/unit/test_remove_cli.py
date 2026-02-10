"""Tests for `Macaw remove` CLI command."""

from __future__ import annotations

from typing import TYPE_CHECKING
from unittest.mock import MagicMock, patch

if TYPE_CHECKING:
    from pathlib import Path

from click.testing import CliRunner

from macaw.cli.main import cli


@patch("macaw.registry.downloader.ModelDownloader")
class TestRemoveCommand:
    def test_remove_success_with_confirm(
        self, mock_downloader_cls: MagicMock, tmp_path: Path
    ) -> None:
        mock_downloader = MagicMock()
        mock_downloader_cls.return_value = mock_downloader
        mock_downloader.is_installed.return_value = True
        mock_downloader.remove.return_value = True

        runner = CliRunner()
        result = runner.invoke(
            cli,
            ["remove", "faster-whisper-tiny", "--models-dir", str(tmp_path)],
            input="y\n",
        )
        assert result.exit_code == 0
        assert "removido" in result.output.lower()

    def test_remove_success_with_yes_flag(
        self, mock_downloader_cls: MagicMock, tmp_path: Path
    ) -> None:
        mock_downloader = MagicMock()
        mock_downloader_cls.return_value = mock_downloader
        mock_downloader.is_installed.return_value = True
        mock_downloader.remove.return_value = True

        runner = CliRunner()
        result = runner.invoke(
            cli,
            ["remove", "faster-whisper-tiny", "-y", "--models-dir", str(tmp_path)],
        )
        assert result.exit_code == 0
        assert "removido" in result.output.lower()

    def test_remove_cancelled_by_user(
        self, mock_downloader_cls: MagicMock, tmp_path: Path
    ) -> None:
        mock_downloader = MagicMock()
        mock_downloader_cls.return_value = mock_downloader
        mock_downloader.is_installed.return_value = True

        runner = CliRunner()
        result = runner.invoke(
            cli,
            ["remove", "faster-whisper-tiny", "--models-dir", str(tmp_path)],
            input="n\n",
        )
        assert result.exit_code == 0
        assert "Cancelado" in result.output
        mock_downloader.remove.assert_not_called()

    def test_remove_not_installed(self, mock_downloader_cls: MagicMock, tmp_path: Path) -> None:
        mock_downloader = MagicMock()
        mock_downloader_cls.return_value = mock_downloader
        mock_downloader.is_installed.return_value = False

        runner = CliRunner()
        result = runner.invoke(
            cli,
            ["remove", "nonexistent-model", "--models-dir", str(tmp_path)],
        )
        assert result.exit_code == 1
        assert "nao esta instalado" in result.output

    def test_remove_fails(self, mock_downloader_cls: MagicMock, tmp_path: Path) -> None:
        mock_downloader = MagicMock()
        mock_downloader_cls.return_value = mock_downloader
        mock_downloader.is_installed.return_value = True
        mock_downloader.remove.return_value = False

        runner = CliRunner()
        result = runner.invoke(
            cli,
            ["remove", "failing-model", "-y", "--models-dir", str(tmp_path)],
        )
        assert result.exit_code == 1
        assert "Erro ao remover" in result.output
