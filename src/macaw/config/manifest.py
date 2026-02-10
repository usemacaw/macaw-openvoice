"""Parsing and validation for macaw.yaml manifests."""

from __future__ import annotations

from pathlib import Path

import yaml
from pydantic import BaseModel, field_validator

from macaw._types import ModelType, STTArchitecture  # noqa: TC001 - Pydantic needs at runtime
from macaw.exceptions import ManifestParseError, ManifestValidationError


class ModelCapabilities(BaseModel):
    """Capabilities declared in the manifest."""

    streaming: bool = False
    architecture: STTArchitecture | None = None
    languages: list[str] = []
    word_timestamps: bool = False
    translation: bool = False
    partial_transcripts: bool = False
    hot_words: bool = False
    batch_inference: bool = False
    language_detection: bool = False
    initial_prompt: bool = False


class ModelResources(BaseModel):
    """Resources required for the model."""

    memory_mb: int
    gpu_required: bool = False
    gpu_recommended: bool = False
    load_time_seconds: int = 10


class EngineConfig(BaseModel, extra="allow"):
    """Engine-specific configuration.

    Allows extra fields because each engine has its own parameters.
    """

    model_size: str | None = None
    compute_type: str = "float16"
    device: str = "auto"
    beam_size: int = 5
    vad_filter: bool = False


class ModelManifest(BaseModel):
    """Macaw model manifest (macaw.yaml).

    Describes capabilities, resources, and configuration for a model
    installed in the local registry.
    """

    name: str
    version: str
    engine: str
    model_type: ModelType
    description: str = ""
    capabilities: ModelCapabilities = ModelCapabilities()
    resources: ModelResources
    engine_config: EngineConfig = EngineConfig()

    @field_validator("name")
    @classmethod
    def name_must_be_valid(cls, v: str) -> str:
        if not v or not v.replace("-", "").replace("_", "").isalnum():
            msg = f"Invalid model name: '{v}'. Use only alphanumerics, hyphens, and underscores."
            raise ValueError(msg)
        return v

    @classmethod
    def from_yaml_path(cls, path: str | Path) -> ModelManifest:
        """Load manifest from a YAML file."""
        path = Path(path)
        if not path.exists():
            raise ManifestParseError(str(path), "File not found")

        try:
            raw = path.read_text(encoding="utf-8")
        except OSError as e:
            raise ManifestParseError(str(path), f"Error reading file: {e}") from e

        return cls.from_yaml_string(raw, source_path=str(path))

    @classmethod
    def from_yaml_string(cls, raw: str, source_path: str = "<string>") -> ModelManifest:
        """Load manifest from a YAML string."""
        try:
            data = yaml.safe_load(raw)
        except yaml.YAMLError as e:
            raise ManifestParseError(source_path, f"Invalid YAML: {e}") from e

        if not isinstance(data, dict):
            raise ManifestParseError(source_path, "YAML content must be a mapping")

        # Normalize 'type' field -> 'model_type' (macaw.yaml uses 'type')
        if "type" in data and "model_type" not in data:
            data["model_type"] = data.pop("type")

        try:
            return cls.model_validate(data)
        except Exception as e:
            errors = [str(e)]
            raise ManifestValidationError(source_path, errors) from e
