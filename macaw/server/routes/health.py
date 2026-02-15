"""Health check and models list endpoints."""

from __future__ import annotations

from typing import Any

from fastapi import APIRouter, Request

import macaw

router = APIRouter()


@router.get("/health")
async def health(request: Request) -> dict[str, Any]:
    """Runtime health check.

    Returns status based on both registry and worker readiness:
    - ``"ok"`` when all spawned workers are READY.
    - ``"loading"`` when workers are still starting.
    - ``"degraded"`` when one or more workers have crashed.
    """
    response: dict[str, Any] = {
        "status": "ok",
        "version": macaw.__version__,
    }

    registry = getattr(request.app.state, "registry", None)
    if registry is not None:
        models = registry.list_models()
        response["models_loaded"] = len(models)

    worker_manager = getattr(request.app.state, "worker_manager", None)
    if worker_manager is not None:
        summary = worker_manager.worker_summary()
        response["workers_ready"] = summary["ready"]
        response["workers_total"] = summary["total"]
        if summary["starting"] > 0:
            response["status"] = "loading"
        elif summary["crashed"] > 0:
            response["status"] = "degraded"

    return response


@router.get("/v1/models")
async def list_models(request: Request) -> dict[str, Any]:
    """List models loaded on the server.

    Returns OpenAI-compatible format with ``object: "list"`` and ``data`` array.
    Used by the `macaw ps` command.
    """
    registry = getattr(request.app.state, "registry", None)
    if registry is None:
        return {"object": "list", "data": []}

    manifests = registry.list_models()
    data: list[dict[str, Any]] = []
    for m in manifests:
        data.append(
            {
                "id": m.name,
                "object": "model",
                "owned_by": "macaw",
                "created": 0,
            }
        )

    return {"object": "list", "data": data}
