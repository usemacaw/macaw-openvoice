"""Shared protobuf converter utilities for STT and TTS workers.

Contains converter functions whose logic is identical across worker types.
"""

from __future__ import annotations

from typing import Any


def build_health_response(
    health_response_cls: type[Any],
    health: dict[str, str],
    model_name: str,
    engine: str,
) -> Any:
    """Build a HealthResponse protobuf from a backend health dict.

    Accepts either ``stt_worker_pb2.HealthResponse`` or
    ``tts_worker_pb2.HealthResponse`` — both have identical constructors.

    Args:
        health_response_cls: The HealthResponse protobuf class to instantiate.
        health: Backend health dict (must contain ``status`` key).
        model_name: Model name for the response.
        engine: Engine name for the response.

    Returns:
        An instance of health_response_cls.
    """
    return health_response_cls(
        status=health.get("status", "unknown"),
        model_name=model_name,
        engine=engine,
    )
