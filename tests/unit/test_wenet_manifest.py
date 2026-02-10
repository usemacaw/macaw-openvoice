"""Testes do manifesto WeNet e registro na factory."""

from __future__ import annotations

from typing import TYPE_CHECKING

import pytest

from macaw._types import ModelType, STTArchitecture
from macaw.config.manifest import ModelManifest
from macaw.workers.stt.main import _create_backend

if TYPE_CHECKING:
    from pathlib import Path


class TestWeNetManifest:
    """Testes de parsing e validacao do manifesto WeNet."""

    def test_parse_wenet_manifest(self, valid_stt_wenet_manifest_path: Path) -> None:
        manifest = ModelManifest.from_yaml_path(valid_stt_wenet_manifest_path)
        assert manifest.name == "wenet-ctc"
        assert manifest.version == "1.0.0"
        assert manifest.engine == "wenet"
        assert manifest.model_type == ModelType.STT

    def test_architecture_is_ctc(self, valid_stt_wenet_manifest_path: Path) -> None:
        manifest = ModelManifest.from_yaml_path(valid_stt_wenet_manifest_path)
        assert manifest.capabilities.architecture == STTArchitecture.CTC

    def test_capabilities(self, valid_stt_wenet_manifest_path: Path) -> None:
        manifest = ModelManifest.from_yaml_path(valid_stt_wenet_manifest_path)
        assert manifest.capabilities.streaming is True
        assert manifest.capabilities.hot_words is True
        assert manifest.capabilities.partial_transcripts is True
        assert manifest.capabilities.translation is False
        assert manifest.capabilities.initial_prompt is False
        assert manifest.capabilities.word_timestamps is True

    def test_languages(self, valid_stt_wenet_manifest_path: Path) -> None:
        manifest = ModelManifest.from_yaml_path(valid_stt_wenet_manifest_path)
        assert "zh" in manifest.capabilities.languages
        assert "en" in manifest.capabilities.languages

    def test_resources(self, valid_stt_wenet_manifest_path: Path) -> None:
        manifest = ModelManifest.from_yaml_path(valid_stt_wenet_manifest_path)
        assert manifest.resources.memory_mb == 1024
        assert manifest.resources.gpu_required is False

    def test_engine_config(self, valid_stt_wenet_manifest_path: Path) -> None:
        manifest = ModelManifest.from_yaml_path(valid_stt_wenet_manifest_path)
        assert manifest.engine_config.device == "auto"
        assert manifest.engine_config.vad_filter is False

    def test_from_yaml_string(self) -> None:
        yaml_str = """
name: wenet-test
version: 0.1.0
engine: wenet
type: stt
capabilities:
  architecture: ctc
  streaming: true
  hot_words: true
resources:
  memory_mb: 512
"""
        manifest = ModelManifest.from_yaml_string(yaml_str)
        assert manifest.name == "wenet-test"
        assert manifest.engine == "wenet"
        assert manifest.capabilities.architecture == STTArchitecture.CTC


class TestCreateBackendFactory:
    """Testes da factory _create_backend para WeNet."""

    def test_create_wenet_backend(self) -> None:
        from macaw.workers.stt.wenet import WeNetBackend

        backend = _create_backend("wenet")
        assert isinstance(backend, WeNetBackend)

    def test_create_faster_whisper_backend(self) -> None:
        from macaw.workers.stt.faster_whisper import FasterWhisperBackend

        backend = _create_backend("faster-whisper")
        assert isinstance(backend, FasterWhisperBackend)

    def test_create_unknown_engine_raises(self) -> None:
        with pytest.raises(ValueError, match="Engine STT nao suportada"):
            _create_backend("nao-existe")

    def test_wenet_backend_architecture(self) -> None:
        backend = _create_backend("wenet")
        assert backend.architecture == STTArchitecture.CTC

    def test_faster_whisper_backend_architecture(self) -> None:
        backend = _create_backend("faster-whisper")
        assert backend.architecture == STTArchitecture.ENCODER_DECODER
