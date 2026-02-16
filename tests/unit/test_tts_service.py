"""Tests for TTS service facade (resolve_tts_resources, find_default_tts_model)."""

from __future__ import annotations

from unittest.mock import MagicMock

import pytest

from macaw._types import ModelType
from macaw.exceptions import ModelNotFoundError, WorkerUnavailableError
from macaw.server.tts_service import find_default_tts_model, resolve_tts_resources


def _make_registry(
    *,
    model_name: str = "kokoro-v1",
    model_type: ModelType = ModelType.TTS,
    models: list[MagicMock] | None = None,
) -> MagicMock:
    """Create a mock ModelRegistry."""
    registry = MagicMock()
    manifest = MagicMock()
    manifest.name = model_name
    manifest.model_type = model_type
    registry.get_manifest.return_value = manifest

    if models is not None:
        registry.list_models.return_value = models
    else:
        registry.list_models.return_value = [manifest]

    return registry


def _make_worker_manager(*, has_worker: bool = True, port: int = 50052) -> MagicMock:
    """Create a mock WorkerManager."""
    manager = MagicMock()
    if has_worker:
        worker = MagicMock()
        worker.port = port
        worker.worker_id = "tts-worker-1"
        manager.get_ready_worker.return_value = worker
    else:
        manager.get_ready_worker.return_value = None
    return manager


class TestResolveTTSResources:
    def test_valid_tts_model_returns_tuple(self) -> None:
        registry = _make_registry()
        manager = _make_worker_manager()

        manifest, worker, address = resolve_tts_resources(registry, manager, "kokoro-v1")

        assert manifest.model_type == ModelType.TTS
        assert worker.port == 50052
        assert address == "localhost:50052"

    def test_stt_model_raises_model_not_found(self) -> None:
        registry = _make_registry(model_type=ModelType.STT)
        manager = _make_worker_manager()

        with pytest.raises(ModelNotFoundError):
            resolve_tts_resources(registry, manager, "faster-whisper-tiny")

    def test_no_ready_worker_raises_worker_unavailable(self) -> None:
        registry = _make_registry()
        manager = _make_worker_manager(has_worker=False)

        with pytest.raises(WorkerUnavailableError):
            resolve_tts_resources(registry, manager, "kokoro-v1")

    def test_nonexistent_model_raises_model_not_found(self) -> None:
        registry = MagicMock()
        registry.get_manifest.side_effect = ModelNotFoundError("unknown")
        manager = _make_worker_manager()

        with pytest.raises(ModelNotFoundError):
            resolve_tts_resources(registry, manager, "unknown")


class TestFindDefaultTTSModel:
    def test_returns_first_tts_model(self) -> None:
        tts_model = MagicMock()
        tts_model.model_type = ModelType.TTS
        tts_model.name = "kokoro-v1"
        stt_model = MagicMock()
        stt_model.model_type = ModelType.STT
        stt_model.name = "whisper-tiny"

        registry = _make_registry(models=[stt_model, tts_model])

        result = find_default_tts_model(registry)

        assert result == "kokoro-v1"

    def test_returns_none_when_no_tts_models(self) -> None:
        stt_model = MagicMock()
        stt_model.model_type = ModelType.STT
        stt_model.name = "whisper-tiny"

        registry = _make_registry(models=[stt_model])

        result = find_default_tts_model(registry)

        assert result is None
