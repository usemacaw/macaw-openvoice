"""Tests for ModelCatalog."""

from __future__ import annotations

from textwrap import dedent
from typing import TYPE_CHECKING

import pytest

if TYPE_CHECKING:
    from pathlib import Path

from macaw.registry.catalog import CatalogEntry, ModelCatalog


@pytest.fixture
def catalog_yaml(tmp_path: Path) -> Path:
    """Cria catalog.yaml valido para testes."""
    content = dedent("""\
        models:
          faster-whisper-tiny:
            repo: "Systran/faster-whisper-tiny"
            engine: faster-whisper
            type: stt
            architecture: encoder-decoder
            description: "Tiny model for testing"
            manifest:
              name: faster-whisper-tiny
              version: "1.0.0"
              engine: faster-whisper
              type: stt

          kokoro-v1:
            repo: "hexgrad/Kokoro-82M"
            engine: kokoro
            type: tts
            description: "TTS model"
            manifest:
              name: kokoro-v1
              version: "1.0.0"
              engine: kokoro
              type: tts
    """)
    path = tmp_path / "catalog.yaml"
    path.write_text(content, encoding="utf-8")
    return path


@pytest.fixture
def empty_catalog_yaml(tmp_path: Path) -> Path:
    """Cria catalog.yaml com lista vazia."""
    content = dedent("""\
        models: {}
    """)
    path = tmp_path / "catalog.yaml"
    path.write_text(content, encoding="utf-8")
    return path


@pytest.fixture
def invalid_catalog_yaml(tmp_path: Path) -> Path:
    """Cria catalog.yaml invalido (sem campo models)."""
    content = dedent("""\
        version: 1
        entries:
          - name: foo
    """)
    path = tmp_path / "catalog.yaml"
    path.write_text(content, encoding="utf-8")
    return path


@pytest.fixture
def invalid_models_type_yaml(tmp_path: Path) -> Path:
    """Cria catalog.yaml com models como lista em vez de dict."""
    content = dedent("""\
        models:
          - name: foo
          - name: bar
    """)
    path = tmp_path / "catalog.yaml"
    path.write_text(content, encoding="utf-8")
    return path


class TestCatalogLoad:
    def test_load_valid_catalog(self, catalog_yaml: Path) -> None:
        catalog = ModelCatalog(catalog_yaml)
        catalog.load()
        assert len(catalog.list_models()) == 2

    def test_load_empty_catalog(self, empty_catalog_yaml: Path) -> None:
        catalog = ModelCatalog(empty_catalog_yaml)
        catalog.load()
        assert len(catalog.list_models()) == 0

    def test_load_nonexistent_file_raises(self, tmp_path: Path) -> None:
        catalog = ModelCatalog(tmp_path / "nonexistent.yaml")
        with pytest.raises(FileNotFoundError, match="Catalog not found"):
            catalog.load()

    def test_load_invalid_catalog_raises(self, invalid_catalog_yaml: Path) -> None:
        catalog = ModelCatalog(invalid_catalog_yaml)
        with pytest.raises(ValueError, match="campo 'models' ausente"):
            catalog.load()

    def test_load_invalid_models_type_raises(self, invalid_models_type_yaml: Path) -> None:
        catalog = ModelCatalog(invalid_models_type_yaml)
        with pytest.raises(ValueError, match="deve ser mapeamento"):
            catalog.load()


class TestCatalogGet:
    def test_get_existing_model(self, catalog_yaml: Path) -> None:
        catalog = ModelCatalog(catalog_yaml)
        catalog.load()
        entry = catalog.get("faster-whisper-tiny")
        assert entry is not None
        assert entry.name == "faster-whisper-tiny"
        assert entry.repo == "Systran/faster-whisper-tiny"
        assert entry.engine == "faster-whisper"
        assert entry.model_type == "stt"
        assert entry.architecture == "encoder-decoder"

    def test_get_nonexistent_model_returns_none(self, catalog_yaml: Path) -> None:
        catalog = ModelCatalog(catalog_yaml)
        catalog.load()
        assert catalog.get("nonexistent") is None

    def test_get_tts_model(self, catalog_yaml: Path) -> None:
        catalog = ModelCatalog(catalog_yaml)
        catalog.load()
        entry = catalog.get("kokoro-v1")
        assert entry is not None
        assert entry.engine == "kokoro"
        assert entry.model_type == "tts"


class TestCatalogHasModel:
    def test_has_existing_model(self, catalog_yaml: Path) -> None:
        catalog = ModelCatalog(catalog_yaml)
        catalog.load()
        assert catalog.has_model("faster-whisper-tiny") is True

    def test_has_nonexistent_model(self, catalog_yaml: Path) -> None:
        catalog = ModelCatalog(catalog_yaml)
        catalog.load()
        assert catalog.has_model("nonexistent") is False


class TestCatalogListModels:
    def test_list_returns_all_entries(self, catalog_yaml: Path) -> None:
        catalog = ModelCatalog(catalog_yaml)
        catalog.load()
        models = catalog.list_models()
        assert len(models) == 2
        names = {m.name for m in models}
        assert names == {"faster-whisper-tiny", "kokoro-v1"}


class TestCatalogEntry:
    def test_entry_is_frozen(self) -> None:
        entry = CatalogEntry(
            name="test",
            repo="test/repo",
            engine="test-engine",
            model_type="stt",
        )
        with pytest.raises(AttributeError):
            entry.name = "modified"  # type: ignore[misc]

    def test_entry_defaults(self) -> None:
        entry = CatalogEntry(
            name="test",
            repo="test/repo",
            engine="test-engine",
            model_type="stt",
        )
        assert entry.architecture is None
        assert entry.description == ""
        assert entry.manifest == {}

    def test_entry_with_manifest(self) -> None:
        manifest = {"name": "test", "version": "1.0.0"}
        entry = CatalogEntry(
            name="test",
            repo="test/repo",
            engine="test-engine",
            model_type="stt",
            manifest=manifest,
        )
        assert entry.manifest == manifest


class TestDefaultCatalog:
    """Tests that the built-in catalog.yaml loads without errors."""

    def test_builtin_catalog_loads(self) -> None:
        catalog = ModelCatalog()
        catalog.load()
        models = catalog.list_models()
        assert len(models) >= 1

    def test_builtin_catalog_has_faster_whisper_tiny(self) -> None:
        catalog = ModelCatalog()
        catalog.load()
        assert catalog.has_model("faster-whisper-tiny")

    def test_builtin_catalog_has_kokoro(self) -> None:
        catalog = ModelCatalog()
        catalog.load()
        assert catalog.has_model("kokoro-v1")

    def test_all_entries_have_required_fields(self) -> None:
        catalog = ModelCatalog()
        catalog.load()
        for entry in catalog.list_models():
            assert entry.name, "Entry missing name"
            assert entry.repo, f"Entry {entry.name} missing repo"
            assert entry.engine, f"Entry {entry.name} missing engine"
            assert entry.model_type in ("stt", "tts"), (
                f"Entry {entry.name} has invalid type: {entry.model_type}"
            )
            assert entry.manifest, f"Entry {entry.name} missing manifest"

    def test_all_entries_have_manifest_with_name(self) -> None:
        catalog = ModelCatalog()
        catalog.load()
        for entry in catalog.list_models():
            assert "name" in entry.manifest, f"Entry {entry.name} manifest missing 'name' field"
