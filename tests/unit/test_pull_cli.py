"""Tests for `Macaw pull` CLI command."""

from __future__ import annotations

from typing import TYPE_CHECKING
from unittest.mock import MagicMock, patch

if TYPE_CHECKING:
    from pathlib import Path

from click.testing import CliRunner

from macaw.cli.main import cli


@patch("macaw.registry.downloader.ModelDownloader")
@patch("macaw.registry.catalog.ModelCatalog")
class TestPullCommand:
    def test_pull_success(
        self, mock_catalog_cls: MagicMock, mock_downloader_cls: MagicMock, tmp_path: Path
    ) -> None:
        mock_catalog = MagicMock()
        mock_catalog_cls.return_value = mock_catalog
        mock_entry = MagicMock()
        mock_entry.name = "faster-whisper-tiny"
        mock_entry.repo = "Systran/faster-whisper-tiny"
        mock_catalog.get.return_value = mock_entry

        mock_downloader = MagicMock()
        mock_downloader_cls.return_value = mock_downloader
        mock_downloader.is_installed.return_value = False
        mock_downloader.download.return_value = tmp_path / "faster-whisper-tiny"

        runner = CliRunner()
        result = runner.invoke(cli, ["pull", "faster-whisper-tiny", "--models-dir", str(tmp_path)])
        assert result.exit_code == 0
        assert "Downloading" in result.output
        assert "Model installed" in result.output or "installed" in result.output.lower()

    def test_pull_model_not_in_catalog(
        self, mock_catalog_cls: MagicMock, mock_downloader_cls: MagicMock
    ) -> None:
        mock_catalog = MagicMock()
        mock_catalog_cls.return_value = mock_catalog
        mock_catalog.get.return_value = None
        mock_catalog.list_models.return_value = []

        runner = CliRunner()
        result = runner.invoke(cli, ["pull", "nonexistent-model"])
        assert result.exit_code == 1
        assert "not found" in result.output

    def test_pull_already_installed_without_force(
        self, mock_catalog_cls: MagicMock, mock_downloader_cls: MagicMock
    ) -> None:
        mock_catalog = MagicMock()
        mock_catalog_cls.return_value = mock_catalog
        mock_entry = MagicMock()
        mock_entry.name = "faster-whisper-tiny"
        mock_catalog.get.return_value = mock_entry

        mock_downloader = MagicMock()
        mock_downloader_cls.return_value = mock_downloader
        mock_downloader.is_installed.return_value = True

        runner = CliRunner()
        result = runner.invoke(cli, ["pull", "faster-whisper-tiny"])
        assert result.exit_code == 0
        assert "is already installed" in result.output

    def test_pull_with_force_reinstalls(
        self, mock_catalog_cls: MagicMock, mock_downloader_cls: MagicMock, tmp_path: Path
    ) -> None:
        mock_catalog = MagicMock()
        mock_catalog_cls.return_value = mock_catalog
        mock_entry = MagicMock()
        mock_entry.name = "faster-whisper-tiny"
        mock_entry.repo = "Systran/faster-whisper-tiny"
        mock_catalog.get.return_value = mock_entry

        mock_downloader = MagicMock()
        mock_downloader_cls.return_value = mock_downloader
        mock_downloader.is_installed.return_value = True
        mock_downloader.download.return_value = tmp_path / "faster-whisper-tiny"

        runner = CliRunner()
        result = runner.invoke(
            cli,
            ["pull", "faster-whisper-tiny", "--force", "--models-dir", str(tmp_path)],
        )
        assert result.exit_code == 0
        mock_downloader.download.assert_called_once()

    def test_pull_catalog_load_error(
        self, mock_catalog_cls: MagicMock, mock_downloader_cls: MagicMock
    ) -> None:
        mock_catalog = MagicMock()
        mock_catalog_cls.return_value = mock_catalog
        mock_catalog.load.side_effect = FileNotFoundError("Catalogo nao encontrado")

        runner = CliRunner()
        result = runner.invoke(cli, ["pull", "faster-whisper-tiny"])
        assert result.exit_code == 1
        assert "catalogo" in result.output.lower()

    def test_pull_download_runtime_error(
        self, mock_catalog_cls: MagicMock, mock_downloader_cls: MagicMock
    ) -> None:
        mock_catalog = MagicMock()
        mock_catalog_cls.return_value = mock_catalog
        mock_entry = MagicMock()
        mock_entry.name = "faster-whisper-tiny"
        mock_entry.repo = "Systran/faster-whisper-tiny"
        mock_catalog.get.return_value = mock_entry

        mock_downloader = MagicMock()
        mock_downloader_cls.return_value = mock_downloader
        mock_downloader.is_installed.return_value = False
        mock_downloader.download.side_effect = RuntimeError("huggingface_hub nao esta instalado")

        runner = CliRunner()
        result = runner.invoke(cli, ["pull", "faster-whisper-tiny"])
        assert result.exit_code == 1
        assert "huggingface_hub" in result.output

    def test_pull_lists_available_models_on_not_found(
        self, mock_catalog_cls: MagicMock, mock_downloader_cls: MagicMock
    ) -> None:
        mock_catalog = MagicMock()
        mock_catalog_cls.return_value = mock_catalog
        mock_catalog.get.return_value = None

        mock_entry1 = MagicMock()
        mock_entry1.name = "faster-whisper-tiny"
        mock_entry1.description = "Tiny model"
        mock_entry2 = MagicMock()
        mock_entry2.name = "kokoro-v1"
        mock_entry2.description = "TTS model"
        mock_catalog.list_models.return_value = [mock_entry1, mock_entry2]

        runner = CliRunner()
        result = runner.invoke(cli, ["pull", "nonexistent-model"])
        assert result.exit_code == 1
        assert "faster-whisper-tiny" in result.output
        assert "kokoro-v1" in result.output
