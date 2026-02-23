"""Tests for GET /v1/models with per-engine capabilities."""

from __future__ import annotations

import httpx
from httpx import ASGITransport

from macaw._types import ModelType
from macaw.config.manifest import ModelCapabilities, ModelManifest, ModelResources
from macaw.server.app import create_app
from macaw.server.models.models import ModelCapabilitiesResponse, ModelInfo


def _make_registry(*manifests: ModelManifest) -> object:
    """Create a minimal mock registry that returns the given manifests."""
    from unittest.mock import MagicMock

    registry = MagicMock()
    registry.list_models.return_value = list(manifests)
    return registry


def _make_stt_manifest(
    name: str = "faster-whisper-tiny",
    engine: str = "faster-whisper",
    *,
    streaming: bool = True,
    languages: list[str] | None = None,
    word_timestamps: bool = True,
    translation: bool = True,
    hot_words: bool = True,
    batch_inference: bool = True,
    language_detection: bool = True,
) -> ModelManifest:
    """Build a realistic STT manifest with explicit capabilities."""
    return ModelManifest(
        name=name,
        version="1.0.0",
        engine=engine,
        model_type=ModelType.STT,
        capabilities=ModelCapabilities(
            streaming=streaming,
            languages=languages or ["en", "pt", "es"],
            word_timestamps=word_timestamps,
            translation=translation,
            hot_words=hot_words,
            batch_inference=batch_inference,
            language_detection=language_detection,
        ),
        resources=ModelResources(memory_mb=512),
    )


def _make_tts_manifest(
    name: str = "kokoro-v1",
    engine: str = "kokoro",
    *,
    voice_cloning: bool = False,
    instruct_mode: bool = False,
    alignment: bool = True,
    character_alignment: bool = True,
    voice_design: bool = False,
) -> ModelManifest:
    """Build a realistic TTS manifest with explicit capabilities."""
    return ModelManifest(
        name=name,
        version="1.0.0",
        engine=engine,
        model_type=ModelType.TTS,
        capabilities=ModelCapabilities(
            voice_cloning=voice_cloning,
            instruct_mode=instruct_mode,
            alignment=alignment,
            character_alignment=character_alignment,
            voice_design=voice_design,
        ),
        resources=ModelResources(memory_mb=1024),
    )


# --- STT capabilities ---


async def test_list_models_returns_stt_capabilities() -> None:
    """GET /v1/models includes STT capability fields from the manifest."""
    manifest = _make_stt_manifest()
    registry = _make_registry(manifest)
    app = create_app(registry=registry)  # type: ignore[arg-type]

    async with httpx.AsyncClient(
        transport=ASGITransport(app=app),
        base_url="http://test",
    ) as client:
        response = await client.get("/v1/models")

    assert response.status_code == 200
    body = response.json()
    assert body["object"] == "list"
    assert len(body["data"]) == 1

    model = body["data"][0]
    caps = model["capabilities"]
    assert caps["streaming"] is True
    assert caps["languages"] == ["en", "pt", "es"]
    assert caps["word_timestamps"] is True
    assert caps["translation"] is True
    assert caps["hot_words"] is True
    assert caps["batch_inference"] is True
    assert caps["language_detection"] is True


# --- TTS capabilities ---


async def test_list_models_returns_tts_capabilities() -> None:
    """GET /v1/models includes TTS capability fields from the manifest."""
    manifest = _make_tts_manifest(
        name="qwen3-tts",
        engine="qwen3-tts",
        voice_cloning=True,
        instruct_mode=True,
        alignment=True,
        character_alignment=False,
        voice_design=True,
    )
    registry = _make_registry(manifest)
    app = create_app(registry=registry)  # type: ignore[arg-type]

    async with httpx.AsyncClient(
        transport=ASGITransport(app=app),
        base_url="http://test",
    ) as client:
        response = await client.get("/v1/models")

    assert response.status_code == 200
    model = response.json()["data"][0]
    caps = model["capabilities"]
    assert caps["voice_cloning"] is True
    assert caps["instruct_mode"] is True
    assert caps["alignment"] is True
    assert caps["character_alignment"] is False
    assert caps["voice_design"] is True


# --- Empty / None registry ---


async def test_list_models_empty_when_no_registry() -> None:
    """GET /v1/models returns empty list when registry is None."""
    app = create_app(registry=None)

    async with httpx.AsyncClient(
        transport=ASGITransport(app=app),
        base_url="http://test",
    ) as client:
        response = await client.get("/v1/models")

    assert response.status_code == 200
    body = response.json()
    assert body["object"] == "list"
    assert body["data"] == []


# --- Defaults ---


async def test_list_models_default_capabilities() -> None:
    """Capabilities without explicit fields have correct defaults (all False/empty)."""
    manifest = ModelManifest(
        name="minimal-model",
        version="1.0.0",
        engine="test-engine",
        model_type=ModelType.STT,
        capabilities=ModelCapabilities(),
        resources=ModelResources(memory_mb=256),
    )
    registry = _make_registry(manifest)
    app = create_app(registry=registry)  # type: ignore[arg-type]

    async with httpx.AsyncClient(
        transport=ASGITransport(app=app),
        base_url="http://test",
    ) as client:
        response = await client.get("/v1/models")

    caps = response.json()["data"][0]["capabilities"]
    # STT defaults
    assert caps["streaming"] is False
    assert caps["languages"] == []
    assert caps["word_timestamps"] is False
    assert caps["translation"] is False
    assert caps["hot_words"] is False
    assert caps["batch_inference"] is False
    assert caps["language_detection"] is False
    # TTS defaults
    assert caps["voice_cloning"] is False
    assert caps["instruct_mode"] is False
    assert caps["alignment"] is False
    assert caps["character_alignment"] is False
    assert caps["voice_design"] is False


# --- Backward compatibility ---


async def test_list_models_backward_compat_response_shape() -> None:
    """Response preserves existing fields: id, object, owned_by, type, engine."""
    manifest = _make_stt_manifest(name="whisper-large")
    registry = _make_registry(manifest)
    app = create_app(registry=registry)  # type: ignore[arg-type]

    async with httpx.AsyncClient(
        transport=ASGITransport(app=app),
        base_url="http://test",
    ) as client:
        response = await client.get("/v1/models")

    model = response.json()["data"][0]
    assert model["id"] == "whisper-large"
    assert model["object"] == "model"
    assert model["owned_by"] == "macaw"
    assert model["created"] == 0
    assert model["type"] == "stt"
    assert model["engine"] == "faster-whisper"
    # capabilities is present as a new field
    assert "capabilities" in model
    assert isinstance(model["capabilities"], dict)


# --- Internal fields excluded ---


async def test_capabilities_response_excludes_internal_fields() -> None:
    """Internal-only fields (architecture, partial_transcripts, initial_prompt)
    must not appear in the API response."""
    manifest = _make_stt_manifest()
    registry = _make_registry(manifest)
    app = create_app(registry=registry)  # type: ignore[arg-type]

    async with httpx.AsyncClient(
        transport=ASGITransport(app=app),
        base_url="http://test",
    ) as client:
        response = await client.get("/v1/models")

    caps = response.json()["data"][0]["capabilities"]
    assert "architecture" not in caps
    assert "partial_transcripts" not in caps
    assert "initial_prompt" not in caps


# --- Multiple models ---


async def test_list_models_multiple_models() -> None:
    """GET /v1/models returns all registered models with their respective capabilities."""
    stt = _make_stt_manifest(name="whisper-tiny", streaming=True, translation=True)
    tts = _make_tts_manifest(name="kokoro-v1", voice_cloning=False, alignment=True)
    registry = _make_registry(stt, tts)
    app = create_app(registry=registry)  # type: ignore[arg-type]

    async with httpx.AsyncClient(
        transport=ASGITransport(app=app),
        base_url="http://test",
    ) as client:
        response = await client.get("/v1/models")

    body = response.json()
    assert len(body["data"]) == 2

    stt_model = body["data"][0]
    assert stt_model["id"] == "whisper-tiny"
    assert stt_model["type"] == "stt"
    assert stt_model["capabilities"]["streaming"] is True
    assert stt_model["capabilities"]["translation"] is True

    tts_model = body["data"][1]
    assert tts_model["id"] == "kokoro-v1"
    assert tts_model["type"] == "tts"
    assert tts_model["capabilities"]["alignment"] is True
    assert tts_model["capabilities"]["voice_cloning"] is False


# --- Pydantic model unit tests ---


class TestModelCapabilitiesResponse:
    """Unit tests for the ModelCapabilitiesResponse Pydantic model."""

    def test_default_values(self) -> None:
        caps = ModelCapabilitiesResponse()
        assert caps.streaming is False
        assert caps.languages == []
        assert caps.word_timestamps is False
        assert caps.voice_cloning is False
        assert caps.alignment is False

    def test_explicit_values(self) -> None:
        caps = ModelCapabilitiesResponse(
            streaming=True,
            languages=["en", "fr"],
            voice_cloning=True,
        )
        assert caps.streaming is True
        assert caps.languages == ["en", "fr"]
        assert caps.voice_cloning is True

    def test_model_dump_has_all_client_fields(self) -> None:
        dump = ModelCapabilitiesResponse().model_dump()
        expected_fields = {
            "streaming",
            "languages",
            "word_timestamps",
            "translation",
            "hot_words",
            "hot_words_mode",
            "batch_inference",
            "diarization",
            "language_detection",
            "voice_cloning",
            "instruct_mode",
            "alignment",
            "character_alignment",
            "voice_design",
        }
        assert set(dump.keys()) == expected_fields

    def test_model_dump_excludes_internal_fields(self) -> None:
        dump = ModelCapabilitiesResponse().model_dump()
        assert "architecture" not in dump
        assert "partial_transcripts" not in dump
        assert "initial_prompt" not in dump


class TestModelInfo:
    """Unit tests for the ModelInfo Pydantic model."""

    def test_required_fields(self) -> None:
        info = ModelInfo(
            id="my-model",
            type="stt",
            engine="faster-whisper",
            capabilities=ModelCapabilitiesResponse(),
        )
        assert info.id == "my-model"
        assert info.object == "model"
        assert info.owned_by == "macaw"
        assert info.created == 0
        assert info.type == "stt"
        assert info.engine == "faster-whisper"
        assert isinstance(info.capabilities, ModelCapabilitiesResponse)

    def test_model_dump_shape(self) -> None:
        info = ModelInfo(
            id="test",
            type="tts",
            engine="kokoro",
            capabilities=ModelCapabilitiesResponse(alignment=True),
        )
        dump = info.model_dump()
        assert dump["id"] == "test"
        assert dump["type"] == "tts"
        assert dump["engine"] == "kokoro"
        assert dump["capabilities"]["alignment"] is True


# --- ModelCapabilities TTS fields (manifest-level) ---


class TestModelCapabilitiesTTSFields:
    """Verify the new TTS fields on the manifest ModelCapabilities class."""

    def test_tts_fields_default_false(self) -> None:
        caps = ModelCapabilities()
        assert caps.voice_cloning is False
        assert caps.instruct_mode is False
        assert caps.alignment is False
        assert caps.character_alignment is False
        assert caps.voice_design is False

    def test_tts_fields_set_true(self) -> None:
        caps = ModelCapabilities(
            voice_cloning=True,
            instruct_mode=True,
            alignment=True,
            character_alignment=True,
            voice_design=True,
        )
        assert caps.voice_cloning is True
        assert caps.instruct_mode is True
        assert caps.alignment is True
        assert caps.character_alignment is True
        assert caps.voice_design is True

    def test_extra_forbid_rejects_unknown_tts_field(self) -> None:
        """extra='forbid' still rejects unknown fields after adding TTS fields."""
        import pytest

        with pytest.raises(Exception, match="extra_forbidden"):
            ModelCapabilities(supports_video=True)  # type: ignore[call-arg]
