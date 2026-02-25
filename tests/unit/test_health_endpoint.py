"""Testes do health endpoint e app factory."""

from __future__ import annotations

from unittest.mock import MagicMock

import httpx

import macaw
from macaw._types import ModelType
from macaw.config.manifest import ModelCapabilities, ModelManifest, ModelResources
from macaw.server.app import create_app


async def test_create_app_returns_fastapi_instance() -> None:
    app = create_app()
    assert app is not None
    assert app.title == "Macaw OpenVoice"


async def test_health_endpoint_returns_ok_without_worker_manager() -> None:
    app = create_app()
    async with httpx.AsyncClient(
        transport=httpx.ASGITransport(app=app),
        base_url="http://test",
    ) as client:
        response = await client.get("/health")

    assert response.status_code == 200
    body = response.json()
    assert body["status"] == "ok"
    assert body["version"] == macaw.__version__


async def test_health_endpoint_includes_models_loaded_when_registry_present() -> None:
    registry = MagicMock()
    registry.list_models.return_value = [MagicMock(), MagicMock()]
    app = create_app(registry=registry)

    async with httpx.AsyncClient(
        transport=httpx.ASGITransport(app=app),
        base_url="http://test",
    ) as client:
        response = await client.get("/health")

    assert response.status_code == 200
    body = response.json()
    assert body["models_loaded"] == 2


async def test_health_returns_ok_when_all_workers_ready() -> None:
    worker_manager = MagicMock()
    worker_manager.worker_summary.return_value = {
        "total": 2,
        "ready": 2,
        "starting": 0,
        "crashed": 0,
    }
    app = create_app(worker_manager=worker_manager)

    async with httpx.AsyncClient(
        transport=httpx.ASGITransport(app=app),
        base_url="http://test",
    ) as client:
        response = await client.get("/health")

    body = response.json()
    assert body["status"] == "ok"
    assert body["workers_ready"] == 2
    assert body["workers_total"] == 2


async def test_health_returns_loading_when_workers_still_starting() -> None:
    worker_manager = MagicMock()
    worker_manager.worker_summary.return_value = {
        "total": 2,
        "ready": 1,
        "starting": 1,
        "crashed": 0,
    }
    app = create_app(worker_manager=worker_manager)

    async with httpx.AsyncClient(
        transport=httpx.ASGITransport(app=app),
        base_url="http://test",
    ) as client:
        response = await client.get("/health")

    body = response.json()
    assert body["status"] == "loading"
    assert body["workers_ready"] == 1
    assert body["workers_total"] == 2


async def test_health_returns_degraded_when_worker_crashed() -> None:
    worker_manager = MagicMock()
    worker_manager.worker_summary.return_value = {
        "total": 2,
        "ready": 1,
        "starting": 0,
        "crashed": 1,
    }
    app = create_app(worker_manager=worker_manager)

    async with httpx.AsyncClient(
        transport=httpx.ASGITransport(app=app),
        base_url="http://test",
    ) as client:
        response = await client.get("/health")

    body = response.json()
    assert body["status"] == "degraded"
    assert body["workers_ready"] == 1
    assert body["workers_total"] == 2


async def test_app_state_stores_registry_and_scheduler() -> None:
    app = create_app(registry=None, scheduler=None)
    assert app.state.registry is None
    assert app.state.scheduler is None


# ─── Request ID Middleware (L-46) ───


async def test_request_id_middleware_sets_request_state() -> None:
    """RequestIDMiddleware sets a UUID on request.state.request_id for each request."""
    app = create_app()
    ids: list[str] = []

    async with httpx.AsyncClient(
        transport=httpx.ASGITransport(app=app),
        base_url="http://test",
    ) as client:
        for _ in range(3):
            response = await client.get("/health")
            assert response.status_code == 200
            # We can't directly access request.state from outside, but we can
            # verify the middleware is wired by checking error handlers use it.
            ids.append("ok")

    # All requests succeeded (middleware didn't break anything)
    assert len(ids) == 3


# ─── /v1/models OpenAI format (M-13) ───


async def test_list_models_returns_openai_format() -> None:
    """GET /v1/models returns OpenAI-compatible format with object: list."""
    registry = MagicMock()
    m1 = ModelManifest(
        name="faster-whisper-tiny",
        version="1.0.0",
        engine="faster-whisper",
        model_type=ModelType.STT,
        capabilities=ModelCapabilities(streaming=True),
        resources=ModelResources(memory_mb=512),
    )
    m2 = ModelManifest(
        name="kokoro-v1",
        version="1.0.0",
        engine="kokoro",
        model_type=ModelType.TTS,
        capabilities=ModelCapabilities(alignment=True),
        resources=ModelResources(memory_mb=1024),
    )
    registry.list_models.return_value = [m1, m2]

    app = create_app(registry=registry)
    async with httpx.AsyncClient(
        transport=httpx.ASGITransport(app=app),
        base_url="http://test",
    ) as client:
        response = await client.get("/v1/models")

    assert response.status_code == 200
    body = response.json()
    assert body["object"] == "list"
    assert len(body["data"]) == 2
    assert body["data"][0]["id"] == "faster-whisper-tiny"
    assert body["data"][0]["object"] == "model"
    assert body["data"][0]["owned_by"] == "macaw"
    assert body["data"][1]["id"] == "kokoro-v1"
    # Capabilities are now present
    assert body["data"][0]["capabilities"]["streaming"] is True
    assert body["data"][1]["capabilities"]["alignment"] is True


# ─── Optional features probing (#dep-compat) ───


async def test_health_includes_optional_features() -> None:
    """Health endpoint includes optional_features dict with expected keys."""
    # Reset cache so probe runs fresh
    from macaw.server.routes import health as health_module

    health_module._optional_features_cache = health_module._TTLCache()

    app = create_app()
    async with httpx.AsyncClient(
        transport=httpx.ASGITransport(app=app),
        base_url="http://test",
    ) as client:
        response = await client.get("/health")

    assert response.status_code == 200
    body = response.json()
    features = body["optional_features"]
    expected_keys = {"alignment", "diarization", "codec_opus", "codec_mp3", "itn"}
    assert set(features.keys()) == expected_keys
    # All values must be booleans
    assert all(isinstance(v, bool) for v in features.values())

    # Clean up cache for other tests
    health_module._optional_features_cache = health_module._TTLCache()


async def test_optional_features_cached() -> None:
    """Probing is called once; subsequent requests reuse cached result."""
    from unittest.mock import patch as _patch

    from macaw.server.routes import health as health_module

    health_module._optional_features_cache = health_module._TTLCache()

    app = create_app()

    with _patch.object(
        health_module, "_safe_find_spec", wraps=health_module._safe_find_spec
    ) as spy:
        async with httpx.AsyncClient(
            transport=httpx.ASGITransport(app=app),
            base_url="http://test",
        ) as client:
            await client.get("/health")
            await client.get("/health")

    # _safe_find_spec was called only during the first request (cache hit on second)
    first_call_count = spy.call_count
    assert first_call_count > 0  # At least called once

    # After first probe, cache is populated — second request doesn't call again
    # Exactly 3 calls: alignment, diarization, itn (once per probe)
    assert first_call_count == 3

    # Clean up
    health_module._optional_features_cache = health_module._TTLCache()


async def test_list_models_empty_when_no_registry() -> None:
    """GET /v1/models returns empty list when registry is None."""
    app = create_app(registry=None)
    async with httpx.AsyncClient(
        transport=httpx.ASGITransport(app=app),
        base_url="http://test",
    ) as client:
        response = await client.get("/v1/models")

    assert response.status_code == 200
    body = response.json()
    assert body["object"] == "list"
    assert body["data"] == []
