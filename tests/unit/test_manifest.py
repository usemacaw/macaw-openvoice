"""Testes do parsing e validacao de manifestos macaw.yaml."""

from pathlib import Path

import pytest

from macaw._types import ModelType, STTArchitecture
from macaw.config.manifest import ModelManifest
from macaw.exceptions import ManifestParseError, ManifestValidationError


class TestManifestFromYamlPath:
    def test_valid_stt_manifest(self, valid_stt_manifest_path: Path) -> None:
        manifest = ModelManifest.from_yaml_path(valid_stt_manifest_path)
        assert manifest.name == "faster-whisper-large-v3"
        assert manifest.version == "3.0.0"
        assert manifest.engine == "faster-whisper"
        assert manifest.model_type == ModelType.STT

    def test_valid_stt_capabilities(self, valid_stt_manifest_path: Path) -> None:
        manifest = ModelManifest.from_yaml_path(valid_stt_manifest_path)
        assert manifest.capabilities.streaming is True
        assert manifest.capabilities.architecture == STTArchitecture.ENCODER_DECODER
        assert "pt" in manifest.capabilities.languages
        assert manifest.capabilities.word_timestamps is True

    def test_valid_stt_resources(self, valid_stt_manifest_path: Path) -> None:
        manifest = ModelManifest.from_yaml_path(valid_stt_manifest_path)
        assert manifest.resources.memory_mb == 3072
        assert manifest.resources.gpu_required is False
        assert manifest.resources.gpu_recommended is True

    def test_valid_stt_engine_config(self, valid_stt_manifest_path: Path) -> None:
        manifest = ModelManifest.from_yaml_path(valid_stt_manifest_path)
        assert manifest.engine_config.model_size == "large-v3"
        assert manifest.engine_config.vad_filter is False

    def test_valid_tts_manifest(self, valid_tts_manifest_path: Path) -> None:
        manifest = ModelManifest.from_yaml_path(valid_tts_manifest_path)
        assert manifest.name == "kokoro-v1"
        assert manifest.model_type == ModelType.TTS

    def test_minimal_manifest(self, minimal_manifest_path: Path) -> None:
        manifest = ModelManifest.from_yaml_path(minimal_manifest_path)
        assert manifest.name == "minimal-model"
        assert manifest.capabilities.streaming is False
        assert manifest.engine_config.compute_type == "auto"

    def test_invalid_manifest_raises_validation_error(self, invalid_manifest_path: Path) -> None:
        with pytest.raises(ManifestValidationError):
            ModelManifest.from_yaml_path(invalid_manifest_path)

    def test_nonexistent_file_raises_parse_error(self) -> None:
        with pytest.raises(ManifestParseError, match="File not found"):
            ModelManifest.from_yaml_path("/nonexistent/path/macaw.yaml")


class TestManifestFromYamlString:
    def test_type_normalized_to_model_type(self) -> None:
        yaml_str = """
name: test-model
version: 1.0.0
engine: test
type: stt
resources:
  memory_mb: 512
"""
        manifest = ModelManifest.from_yaml_string(yaml_str)
        assert manifest.model_type == ModelType.STT

    def test_invalid_yaml_raises_parse_error(self) -> None:
        with pytest.raises(ManifestParseError, match="Invalid YAML"):
            ModelManifest.from_yaml_string("{{invalid yaml")

    def test_non_dict_yaml_raises_parse_error(self) -> None:
        with pytest.raises(ManifestParseError, match="must be a mapping"):
            ModelManifest.from_yaml_string("- item1\n- item2")

    def test_invalid_model_name_raises_error(self) -> None:
        yaml_str = """
name: "invalid name with spaces"
version: 1.0.0
engine: test
type: stt
resources:
  memory_mb: 512
"""
        with pytest.raises(ManifestValidationError):
            ModelManifest.from_yaml_string(yaml_str)

    def test_architecture_parses_to_enum(self) -> None:
        yaml_str = """
name: test-ctc
version: 1.0.0
engine: test-stt
type: stt
capabilities:
  architecture: ctc
resources:
  memory_mb: 512
"""
        manifest = ModelManifest.from_yaml_string(yaml_str)
        assert manifest.capabilities.architecture == STTArchitecture.CTC

    def test_engine_config_accepts_extra_fields(self) -> None:
        yaml_str = """
name: test-model
version: 1.0.0
engine: custom
type: stt
resources:
  memory_mb: 512
engine_config:
  custom_param: "value"
  another_param: 42
"""
        manifest = ModelManifest.from_yaml_string(yaml_str)
        extra = manifest.engine_config.model_extra
        assert extra is not None
        assert extra["custom_param"] == "value"
        assert extra["another_param"] == 42


class TestPythonPackageValidation:
    """Verify python_package field validation on ModelManifest."""

    _BASE_YAML = """
name: test-model
version: 1.0.0
engine: custom
type: stt
resources:
  memory_mb: 512
"""

    def test_python_package_none_by_default(self) -> None:
        manifest = ModelManifest.from_yaml_string(self._BASE_YAML)
        assert manifest.python_package is None

    def test_valid_simple_module(self) -> None:
        yaml_str = self._BASE_YAML + "python_package: my_engine\n"
        manifest = ModelManifest.from_yaml_string(yaml_str)
        assert manifest.python_package == "my_engine"

    def test_valid_dotted_module(self) -> None:
        yaml_str = self._BASE_YAML + "python_package: my_company.engines.whisper_cpp\n"
        manifest = ModelManifest.from_yaml_string(yaml_str)
        assert manifest.python_package == "my_company.engines.whisper_cpp"

    def test_empty_string_rejected(self) -> None:
        yaml_str = self._BASE_YAML + 'python_package: ""\n'
        with pytest.raises(ManifestValidationError):
            ModelManifest.from_yaml_string(yaml_str)

    def test_invalid_starts_with_digit(self) -> None:
        yaml_str = self._BASE_YAML + "python_package: 3rd_party.engine\n"
        with pytest.raises(ManifestValidationError):
            ModelManifest.from_yaml_string(yaml_str)

    def test_invalid_contains_hyphen(self) -> None:
        yaml_str = self._BASE_YAML + "python_package: my-engine\n"
        with pytest.raises(ManifestValidationError):
            ModelManifest.from_yaml_string(yaml_str)

    def test_invalid_contains_spaces(self) -> None:
        yaml_str = self._BASE_YAML + "python_package: my engine\n"
        with pytest.raises(ManifestValidationError):
            ModelManifest.from_yaml_string(yaml_str)

    def test_invalid_trailing_dot(self) -> None:
        yaml_str = self._BASE_YAML + "python_package: my_engine.\n"
        with pytest.raises(ManifestValidationError):
            ModelManifest.from_yaml_string(yaml_str)

    def test_valid_underscore_prefix(self) -> None:
        yaml_str = self._BASE_YAML + "python_package: _private.engine\n"
        manifest = ModelManifest.from_yaml_string(yaml_str)
        assert manifest.python_package == "_private.engine"


class TestModelCapabilitiesStrictSchema:
    """Verify ModelCapabilities rejects unknown fields (extra='forbid')."""

    def test_unknown_capability_field_raises_validation_error(self) -> None:
        yaml_str = """
name: test-model
version: 1.0.0
engine: custom
type: stt
capabilities:
  streaming: true
  supports_foo: true
resources:
  memory_mb: 512
"""
        with pytest.raises(ManifestValidationError):
            ModelManifest.from_yaml_string(yaml_str)

    def test_known_capability_fields_accepted(self) -> None:
        yaml_str = """
name: test-model
version: 1.0.0
engine: custom
type: stt
capabilities:
  streaming: true
  architecture: encoder-decoder
  languages: ["en"]
  word_timestamps: true
  translation: false
  partial_transcripts: true
  hot_words: false
  batch_inference: true
  language_detection: true
  initial_prompt: true
resources:
  memory_mb: 512
"""
        manifest = ModelManifest.from_yaml_string(yaml_str)
        assert manifest.capabilities.streaming is True
        assert manifest.capabilities.word_timestamps is True

    def test_empty_capabilities_accepted(self) -> None:
        yaml_str = """
name: test-model
version: 1.0.0
engine: custom
type: stt
resources:
  memory_mb: 512
"""
        manifest = ModelManifest.from_yaml_string(yaml_str)
        assert manifest.capabilities.streaming is False

    def test_engine_config_still_allows_extra(self) -> None:
        """Confirm EngineConfig extra='allow' is not affected."""
        yaml_str = """
name: test-model
version: 1.0.0
engine: custom
type: stt
resources:
  memory_mb: 512
engine_config:
  custom_engine_param: 42
"""
        manifest = ModelManifest.from_yaml_string(yaml_str)
        assert manifest.engine_config.model_extra is not None
        assert manifest.engine_config.model_extra["custom_engine_param"] == 42
