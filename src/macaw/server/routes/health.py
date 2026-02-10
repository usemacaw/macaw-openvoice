"""Health check and models list endpoints."""

from __future__ import annotations

from typing import Any

from fastapi import APIRouter, Request

import macaw

router = APIRouter()


@router.get("/health")
async def health(request: Request) -> dict[str, Any]:
    """Runtime health check.

    Returns basic status (liveness) and readiness information
    based on registry and scheduler state.
    """
    response: dict[str, Any] = {
        "status": "ok",
        "version": macaw.__version__,
    }

    registry = getattr(request.app.state, "registry", None)
    if registry is not None:
        models = registry.list_models()
        response["models_loaded"] = len(models)

    return response


@router.get("/v1/models")
async def list_models(request: Request) -> dict[str, Any]:
    """List models loaded on the server.

    Used by the `macaw ps` command.
    """
    registry = getattr(request.app.state, "registry", None)
    if registry is None:
        return {"models": []}

    manifests = registry.list_models()
    models: list[dict[str, Any]] = []
    for m in manifests:
        models.append(
            {
                "name": m.name,
                "type": m.model_type.value,
                "engine": m.engine,
                "status": "loaded",
            }
        )

    return {"models": models}
