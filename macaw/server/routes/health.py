"""Health check and models list endpoints."""

from __future__ import annotations

import time
from importlib.util import find_spec
from typing import TYPE_CHECKING, Any, Generic, TypeVar

if TYPE_CHECKING:
    from collections.abc import Callable

from fastapi import APIRouter, Request

import macaw
from macaw.server.models.models import ModelCapabilitiesResponse, ModelInfo

router = APIRouter(tags=["System"])

_T = TypeVar("_T")


class _TTLCache(Generic[_T]):
    """Single-value cache with optional TTL for health check probes.

    Assumes single-threaded async (uvicorn default).  When *ttl_s* is
    ``None``, the value is cached permanently (optional features that
    never change at runtime).
    """

    __slots__ = ("_ts", "_ttl_s", "_value")

    def __init__(self, ttl_s: float | None = None) -> None:
        self._ttl_s = ttl_s
        self._value: _T | None = None
        self._ts: float = 0.0

    def get(self, compute: Callable[[], _T]) -> _T:
        """Return cached value, recomputing if expired or missing."""
        if self._value is not None and (
            self._ttl_s is None or (time.monotonic() - self._ts) < self._ttl_s
        ):
            return self._value
        self._value = compute()
        self._ts = time.monotonic()
        return self._value


_optional_features_cache: _TTLCache[dict[str, bool]] = _TTLCache()
_backend_venvs_cache: _TTLCache[dict[str, dict[str, bool]]] = _TTLCache(ttl_s=30.0)


def _safe_find_spec(module_name: str) -> bool:
    """Check if a module is importable without actually importing it.

    ``find_spec`` raises ``ModuleNotFoundError`` for dotted names when the
    parent package is missing (e.g. ``pyannote.audio`` when ``pyannote`` is
    not installed).  This helper catches that and returns ``False``.
    """
    try:
        return find_spec(module_name) is not None
    except (ModuleNotFoundError, ValueError):
        return False


def _compute_optional_features() -> dict[str, bool]:
    """Probe availability of optional features using lightweight import checks."""
    from macaw.codec import is_codec_available

    return {
        "alignment": _safe_find_spec("torchaudio"),
        "diarization": _safe_find_spec("pyannote.audio"),
        "codec_opus": is_codec_available("opus"),
        "codec_mp3": is_codec_available("mp3"),
        "itn": _safe_find_spec("nemo_text_processing"),
    }


def _probe_optional_features() -> dict[str, bool]:
    """Return cached optional feature availability (permanent cache)."""
    return _optional_features_cache.get(_compute_optional_features)


def _compute_backend_venvs() -> dict[str, dict[str, bool]]:
    """Probe per-engine venv provisioning status."""
    from macaw.backends.venv_manager import VenvManager
    from macaw.config.settings import get_settings
    from macaw.engines import ENGINE_EXTRAS

    settings = get_settings().backend
    manager = VenvManager(settings.venv_base_path, uv_path=settings.uv_path)
    return {engine: {"provisioned": manager.exists(engine)} for engine in sorted(ENGINE_EXTRAS)}


def _probe_backend_venvs() -> dict[str, dict[str, bool]]:
    """Return cached venv status (30s TTL)."""
    return _backend_venvs_cache.get(_compute_backend_venvs)


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

    response["optional_features"] = _probe_optional_features()
    response["backend_venvs"] = _probe_backend_venvs()

    return response


@router.get("/v1/models")
async def list_models(request: Request) -> dict[str, Any]:
    """List models loaded on the server.

    Returns OpenAI-compatible format with ``object: "list"`` and ``data`` array.
    Each model entry includes a ``capabilities`` object with client-facing
    capability flags from the manifest.  Used by the ``macaw ps`` command.
    """
    registry = getattr(request.app.state, "registry", None)
    if registry is None:
        return {"object": "list", "data": []}

    manifests = registry.list_models()
    data: list[dict[str, Any]] = []
    for m in manifests:
        caps = m.capabilities
        capabilities = ModelCapabilitiesResponse(
            streaming=caps.streaming,
            languages=caps.languages,
            word_timestamps=caps.word_timestamps,
            translation=caps.translation,
            hot_words=caps.hot_words,
            hot_words_mode=caps.hot_words_mode,
            batch_inference=caps.batch_inference,
            diarization=caps.diarization,
            language_detection=caps.language_detection,
            voice_cloning=caps.voice_cloning,
            instruct_mode=caps.instruct_mode,
            alignment=caps.alignment,
            character_alignment=caps.character_alignment,
            voice_design=caps.voice_design,
        )
        info = ModelInfo(
            id=m.name,
            type=m.model_type.value,
            engine=m.engine,
            capabilities=capabilities,
        )
        data.append(info.model_dump())

    return {"object": "list", "data": data}
