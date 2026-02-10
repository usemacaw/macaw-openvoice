"""Testes do health endpoint e app factory."""

from __future__ import annotations

from unittest.mock import MagicMock

import httpx

import macaw
from macaw.server.app import create_app


async def test_create_app_returns_fastapi_instance() -> None:
    app = create_app()
    assert app is not None
    assert app.title == "Macaw OpenVoice"


async def test_health_endpoint_returns_ok() -> None:
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


async def test_app_state_stores_registry_and_scheduler() -> None:
    app = create_app(registry=None, scheduler=None)
    assert app.state.registry is None
    assert app.state.scheduler is None
