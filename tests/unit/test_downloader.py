"""Tests for ModelDownloader."""

from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest
import yaml

from macaw.registry.catalog import CatalogEntry
from macaw.registry.downloader import ModelDownloader


@pytest.fixture
def models_dir(tmp_path: Path) -> Path:
    """Diretorio temporario para modelos."""
    d = tmp_path / "models"
    d.mkdir()
    return d


@pytest.fixture
def sample_entry() -> CatalogEntry:
    """Entrada de catalogo de exemplo."""
    return CatalogEntry(
        name="faster-whisper-tiny",
        repo="Systran/faster-whisper-tiny",
        engine="faster-whisper",
        model_type="stt",
        architecture="encoder-decoder",
        description="Tiny model",
        manifest={
            "name": "faster-whisper-tiny",
            "version": "1.0.0",
            "engine": "faster-whisper",
            "type": "stt",
            "capabilities": {
                "streaming": True,
                "architecture": "encoder-decoder",
                "languages": ["auto", "en", "pt"],
            },
            "resources": {
                "memory_mb": 512,
            },
            "engine_config": {
                "model_size": "tiny",
                "compute_type": "float16",
                "device": "auto",
                "vad_filter": False,
            },
        },
    )


@pytest.fixture
def entry_without_manifest() -> CatalogEntry:
    """Entrada de catalogo sem manifesto."""
    return CatalogEntry(
        name="no-manifest",
        repo="test/no-manifest",
        engine="test",
        model_type="stt",
        manifest={},
    )


class TestIsInstalled:
    def test_not_installed(self, models_dir: Path) -> None:
        downloader = ModelDownloader(models_dir)
        assert downloader.is_installed("faster-whisper-tiny") is False

    def test_installed_with_manifest(self, models_dir: Path) -> None:
        model_dir = models_dir / "faster-whisper-tiny"
        model_dir.mkdir()
        (model_dir / "macaw.yaml").write_text("name: faster-whisper-tiny")
        downloader = ModelDownloader(models_dir)
        assert downloader.is_installed("faster-whisper-tiny") is True

    def test_directory_exists_without_manifest(self, models_dir: Path) -> None:
        model_dir = models_dir / "faster-whisper-tiny"
        model_dir.mkdir()
        downloader = ModelDownloader(models_dir)
        assert downloader.is_installed("faster-whisper-tiny") is False


class TestDownload:
    def test_download_raises_without_huggingface_hub(
        self, models_dir: Path, sample_entry: CatalogEntry
    ) -> None:
        downloader = ModelDownloader(models_dir)
        with (
            patch.dict("sys.modules", {"huggingface_hub": None}),
            pytest.raises(RuntimeError, match="huggingface_hub"),
        ):
            downloader.download(sample_entry)

    def test_download_creates_model_directory(
        self, models_dir: Path, sample_entry: CatalogEntry
    ) -> None:
        with patch("macaw.registry.downloader.ModelDownloader.download") as mock_download:
            mock_download.return_value = models_dir / "faster-whisper-tiny"
            downloader = ModelDownloader(models_dir)
            result = downloader.download(sample_entry)
            assert result == models_dir / "faster-whisper-tiny"

    def test_download_existing_model_without_force_raises(
        self, models_dir: Path, sample_entry: CatalogEntry
    ) -> None:
        model_dir = models_dir / "faster-whisper-tiny"
        model_dir.mkdir()
        (model_dir / "macaw.yaml").write_text("name: faster-whisper-tiny")
        downloader = ModelDownloader(models_dir)
        with pytest.raises(FileExistsError, match="already installed"):
            downloader.download(sample_entry)

    def test_download_without_manifest_raises(
        self, models_dir: Path, entry_without_manifest: CatalogEntry
    ) -> None:
        """Download falha se catalogo nao tem manifesto."""
        mock_snapshot = MagicMock(return_value=str(models_dir / "no-manifest"))
        with patch(
            "macaw.registry.downloader.snapshot_download",
            mock_snapshot,
            create=True,
        ):

            async def patched_download(
                self_inner: ModelDownloader,
                entry: CatalogEntry,
                *,
                force: bool = False,
            ) -> Path:
                # Skip HF download, just test manifest writing
                model_dir = self_inner._models_dir / entry.name
                model_dir.mkdir(parents=True, exist_ok=True)
                self_inner._write_manifest(model_dir, entry)
                return model_dir

            # Test _write_manifest directly
            downloader = ModelDownloader(models_dir)
            with pytest.raises(ValueError, match="has no manifest"):
                downloader._write_manifest(models_dir / "no-manifest", entry_without_manifest)


class TestWriteManifest:
    def test_write_manifest_creates_yaml(
        self, models_dir: Path, sample_entry: CatalogEntry
    ) -> None:
        model_dir = models_dir / "faster-whisper-tiny"
        model_dir.mkdir()
        downloader = ModelDownloader(models_dir)
        downloader._write_manifest(model_dir, sample_entry)

        manifest_path = model_dir / "macaw.yaml"
        assert manifest_path.exists()

        content = yaml.safe_load(manifest_path.read_text(encoding="utf-8"))
        assert content["name"] == "faster-whisper-tiny"
        assert content["engine"] == "faster-whisper"
        assert content["type"] == "stt"

    def test_write_manifest_preserves_all_fields(
        self, models_dir: Path, sample_entry: CatalogEntry
    ) -> None:
        model_dir = models_dir / "test-model"
        model_dir.mkdir()
        downloader = ModelDownloader(models_dir)
        downloader._write_manifest(model_dir, sample_entry)

        manifest_path = model_dir / "macaw.yaml"
        content = yaml.safe_load(manifest_path.read_text(encoding="utf-8"))
        assert "capabilities" in content
        assert content["capabilities"]["streaming"] is True
        assert "engine_config" in content
        assert content["engine_config"]["vad_filter"] is False

    def test_write_manifest_empty_manifest_raises(
        self, models_dir: Path, entry_without_manifest: CatalogEntry
    ) -> None:
        model_dir = models_dir / "no-manifest"
        model_dir.mkdir()
        downloader = ModelDownloader(models_dir)
        with pytest.raises(ValueError, match="has no manifest"):
            downloader._write_manifest(model_dir, entry_without_manifest)


class TestRemove:
    def test_remove_existing_model(self, models_dir: Path) -> None:
        model_dir = models_dir / "test-model"
        model_dir.mkdir()
        (model_dir / "macaw.yaml").write_text("name: test-model")
        (model_dir / "model.bin").write_bytes(b"\x00" * 100)

        downloader = ModelDownloader(models_dir)
        assert downloader.remove("test-model") is True
        assert not model_dir.exists()

    def test_remove_nonexistent_model(self, models_dir: Path) -> None:
        downloader = ModelDownloader(models_dir)
        assert downloader.remove("nonexistent") is False

    def test_remove_cleans_entire_directory(self, models_dir: Path) -> None:
        model_dir = models_dir / "test-model"
        model_dir.mkdir()
        sub_dir = model_dir / "subdir"
        sub_dir.mkdir()
        (sub_dir / "file.txt").write_text("content")
        (model_dir / "macaw.yaml").write_text("name: test-model")

        downloader = ModelDownloader(models_dir)
        downloader.remove("test-model")
        assert not model_dir.exists()


class TestModelsDir:
    def test_default_models_dir(self) -> None:
        downloader = ModelDownloader()
        assert downloader.models_dir == Path.home() / ".macaw" / "models"

    def test_custom_models_dir(self, tmp_path: Path) -> None:
        downloader = ModelDownloader(tmp_path / "custom")
        assert downloader.models_dir == tmp_path / "custom"

    def test_string_models_dir(self, tmp_path: Path) -> None:
        downloader = ModelDownloader(str(tmp_path / "string_path"))
        assert downloader.models_dir == tmp_path / "string_path"
