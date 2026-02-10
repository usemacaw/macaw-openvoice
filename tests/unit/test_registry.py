"""Testes do Model Registry."""

from __future__ import annotations

from typing import TYPE_CHECKING

import pytest

if TYPE_CHECKING:
    from pathlib import Path

from macaw.exceptions import ModelNotFoundError
from macaw.registry.registry import ModelRegistry

VALID_STT_MANIFEST = """\
name: faster-whisper-tiny
version: "1.0.0"
engine: faster-whisper
type: stt
resources:
  memory_mb: 150
engine_config:
  model_size: tiny
  compute_type: float16
  device: auto
"""

VALID_STT_MANIFEST_2 = """\
name: faster-whisper-large-v3
version: "3.0.0"
engine: faster-whisper
type: stt
resources:
  memory_mb: 3072
engine_config:
  model_size: large-v3
  compute_type: float16
  device: auto
"""

INVALID_MANIFEST = """\
name: 123
this_is: not_a_valid_manifest
"""


def _create_model_dir(base: Path, dir_name: str, manifest_yaml: str) -> Path:
    model_dir = base / dir_name
    model_dir.mkdir()
    (model_dir / "macaw.yaml").write_text(manifest_yaml)
    return model_dir


async def test_scan_finds_models_in_directory(tmp_path: Path) -> None:
    _create_model_dir(tmp_path, "whisper-tiny", VALID_STT_MANIFEST)
    _create_model_dir(tmp_path, "whisper-large", VALID_STT_MANIFEST_2)

    registry = ModelRegistry(tmp_path)
    await registry.scan()

    assert len(registry.list_models()) == 2
    assert registry.has_model("faster-whisper-tiny")
    assert registry.has_model("faster-whisper-large-v3")


async def test_scan_ignores_dirs_without_manifest(tmp_path: Path) -> None:
    _create_model_dir(tmp_path, "whisper-tiny", VALID_STT_MANIFEST)
    (tmp_path / "no-manifest-dir").mkdir()

    registry = ModelRegistry(tmp_path)
    await registry.scan()

    assert len(registry.list_models()) == 1


async def test_scan_ignores_invalid_manifests(tmp_path: Path) -> None:
    _create_model_dir(tmp_path, "whisper-tiny", VALID_STT_MANIFEST)
    _create_model_dir(tmp_path, "broken-model", INVALID_MANIFEST)

    registry = ModelRegistry(tmp_path)
    await registry.scan()

    assert len(registry.list_models()) == 1
    assert registry.has_model("faster-whisper-tiny")


async def test_scan_empty_directory(tmp_path: Path) -> None:
    registry = ModelRegistry(tmp_path)
    await registry.scan()

    assert registry.list_models() == []


async def test_scan_nonexistent_directory(tmp_path: Path) -> None:
    registry = ModelRegistry(tmp_path / "nonexistent")
    await registry.scan()

    assert registry.list_models() == []


async def test_get_manifest_returns_correct_model(tmp_path: Path) -> None:
    _create_model_dir(tmp_path, "whisper-tiny", VALID_STT_MANIFEST)

    registry = ModelRegistry(tmp_path)
    await registry.scan()

    manifest = registry.get_manifest("faster-whisper-tiny")
    assert manifest.name == "faster-whisper-tiny"
    assert manifest.engine == "faster-whisper"


async def test_get_manifest_raises_model_not_found(tmp_path: Path) -> None:
    registry = ModelRegistry(tmp_path)
    await registry.scan()

    with pytest.raises(ModelNotFoundError):
        registry.get_manifest("inexistente")


async def test_list_models_returns_all(tmp_path: Path) -> None:
    _create_model_dir(tmp_path, "whisper-tiny", VALID_STT_MANIFEST)
    _create_model_dir(tmp_path, "whisper-large", VALID_STT_MANIFEST_2)

    registry = ModelRegistry(tmp_path)
    await registry.scan()

    models = registry.list_models()
    names = {m.name for m in models}
    assert names == {"faster-whisper-tiny", "faster-whisper-large-v3"}


async def test_has_model_true_when_exists(tmp_path: Path) -> None:
    _create_model_dir(tmp_path, "whisper-tiny", VALID_STT_MANIFEST)

    registry = ModelRegistry(tmp_path)
    await registry.scan()

    assert registry.has_model("faster-whisper-tiny") is True


async def test_has_model_false_when_missing(tmp_path: Path) -> None:
    registry = ModelRegistry(tmp_path)
    await registry.scan()

    assert registry.has_model("inexistente") is False


async def test_get_model_path_returns_correct_path(tmp_path: Path) -> None:
    model_dir = _create_model_dir(tmp_path, "whisper-tiny", VALID_STT_MANIFEST)

    registry = ModelRegistry(tmp_path)
    await registry.scan()

    path = registry.get_model_path("faster-whisper-tiny")
    assert path == model_dir


async def test_get_model_path_raises_model_not_found(tmp_path: Path) -> None:
    registry = ModelRegistry(tmp_path)
    await registry.scan()

    with pytest.raises(ModelNotFoundError):
        registry.get_model_path("inexistente")


async def test_scan_indexes_by_manifest_name_not_dir_name(tmp_path: Path) -> None:
    """O registry indexa por manifest.name, nao pelo nome do diretorio."""
    _create_model_dir(tmp_path, "my-custom-dir-name", VALID_STT_MANIFEST)

    registry = ModelRegistry(tmp_path)
    await registry.scan()

    assert registry.has_model("faster-whisper-tiny")
    assert not registry.has_model("my-custom-dir-name")


async def test_scan_ignores_files_in_models_dir(tmp_path: Path) -> None:
    """Arquivos soltos no models_dir sao ignorados (nao sao subdiretorios)."""
    _create_model_dir(tmp_path, "whisper-tiny", VALID_STT_MANIFEST)
    (tmp_path / "random_file.txt").write_text("not a model")

    registry = ModelRegistry(tmp_path)
    await registry.scan()

    assert len(registry.list_models()) == 1
