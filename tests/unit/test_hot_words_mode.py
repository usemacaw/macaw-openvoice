"""Tests for hot_words_mode field on EngineCapabilities and related models."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest

from macaw._types import EngineCapabilities, ModelType
from macaw.config.manifest import ModelCapabilities
from macaw.server.models.models import ModelCapabilitiesResponse


class TestEngineCapabilitiesHotWordsMode:
    """EngineCapabilities.hot_words_mode field."""

    def test_default_is_none(self) -> None:
        caps = EngineCapabilities()
        assert caps.hot_words_mode == "none"

    def test_prompt_injection(self) -> None:
        caps = EngineCapabilities(hot_words_mode="prompt_injection")
        assert caps.hot_words_mode == "prompt_injection"

    def test_beam_bias(self) -> None:
        caps = EngineCapabilities(hot_words_mode="beam_bias")
        assert caps.hot_words_mode == "beam_bias"

    def test_native(self) -> None:
        caps = EngineCapabilities(hot_words_mode="native")
        assert caps.hot_words_mode == "native"

    def test_frozen_immutable(self) -> None:
        caps = EngineCapabilities(hot_words_mode="native")
        with pytest.raises(AttributeError):
            caps.hot_words_mode = "none"  # type: ignore[misc]


class TestModelCapabilitiesHotWordsMode:
    """ModelCapabilities (manifest) hot_words_mode field."""

    def test_default_is_none(self) -> None:
        caps = ModelCapabilities()
        assert caps.hot_words_mode == "none"

    def test_custom_mode(self) -> None:
        caps = ModelCapabilities(hot_words_mode="prompt_injection")
        assert caps.hot_words_mode == "prompt_injection"


class TestModelCapabilitiesResponseHotWordsMode:
    """ModelCapabilitiesResponse (API) hot_words_mode field."""

    def test_default_is_none(self) -> None:
        resp = ModelCapabilitiesResponse()
        assert resp.hot_words_mode == "none"

    def test_custom_mode(self) -> None:
        resp = ModelCapabilitiesResponse(hot_words_mode="beam_bias")
        assert resp.hot_words_mode == "beam_bias"

    def test_serialization(self) -> None:
        resp = ModelCapabilitiesResponse(hot_words_mode="native")
        data = resp.model_dump()
        assert data["hot_words_mode"] == "native"


class TestListModelsHotWordsMode:
    """GET /v1/models response includes hot_words_mode from manifest."""

    @pytest.mark.asyncio
    async def test_list_models_includes_hot_words_mode(self) -> None:
        from macaw.config.manifest import (
            EngineConfig,
            ModelCapabilities,
            ModelManifest,
            ModelResources,
        )
        from macaw.server.routes.health import list_models

        manifest = ModelManifest(
            name="test-model",
            version="1.0",
            engine="faster-whisper",
            model_type=ModelType.STT,
            capabilities=ModelCapabilities(
                hot_words=True,
                hot_words_mode="prompt_injection",
            ),
            resources=ModelResources(memory_mb=500),
            engine_config=EngineConfig(),
        )

        mock_registry = MagicMock()
        mock_registry.list_models.return_value = [manifest]

        mock_request = MagicMock()
        mock_request.app.state.registry = mock_registry
        mock_request.app.state.worker_manager = None

        result = await list_models(mock_request)

        assert result["data"][0]["capabilities"]["hot_words_mode"] == "prompt_injection"

    @pytest.mark.asyncio
    async def test_list_models_default_hot_words_mode(self) -> None:
        from macaw.config.manifest import (
            EngineConfig,
            ModelCapabilities,
            ModelManifest,
            ModelResources,
        )
        from macaw.server.routes.health import list_models

        manifest = ModelManifest(
            name="test-model",
            version="1.0",
            engine="qwen3-asr",
            model_type=ModelType.STT,
            capabilities=ModelCapabilities(),
            resources=ModelResources(memory_mb=500),
            engine_config=EngineConfig(),
        )

        mock_registry = MagicMock()
        mock_registry.list_models.return_value = [manifest]

        mock_request = MagicMock()
        mock_request.app.state.registry = mock_registry
        mock_request.app.state.worker_manager = None

        result = await list_models(mock_request)

        assert result["data"][0]["capabilities"]["hot_words_mode"] == "none"


class TestFasterWhisperHotWordsMode:
    """FasterWhisperBackend reports prompt_injection hot words mode."""

    @pytest.mark.asyncio
    async def test_capabilities_hot_words_mode(self) -> None:
        with patch.dict("sys.modules", {"faster_whisper": MagicMock()}):
            from macaw.workers.stt.faster_whisper import FasterWhisperBackend

            backend = FasterWhisperBackend()
            caps = await backend.capabilities()
            assert caps.hot_words_mode == "prompt_injection"
