"""Shared constants for STT and TTS worker subprocesses.

Single source of truth for worker lifecycle parameters and gRPC server
options that are identical between STT and TTS workers.

``STOP_GRACE_PERIOD`` and ``DEFAULT_WARMUP_STEPS`` are resolved lazily via
PEP 562 ``__getattr__`` from ``WorkerLifecycleSettings``.
"""

from __future__ import annotations

# gRPC server-side options for worker subprocesses.
# Matches the client-side streaming keepalive expectations.
# Protocol-level — not user-tunable.
GRPC_WORKER_SERVER_OPTIONS: list[tuple[str, int]] = [
    ("grpc.http2.min_recv_ping_interval_without_data_ms", 5_000),
    ("grpc.keepalive_permit_without_calls", 1),
]

# Real-time factor threshold for warmup readiness.
# RTFx < 1.0 means the system cannot keep up with real-time audio.
MIN_REALTIME_RTF: float = 1.0


# --- PEP 562 lazy resolution for configurable constants ---
_SETTINGS_ATTRS: dict[str, str] = {
    "STOP_GRACE_PERIOD": "stop_grace_period_s",
    "DEFAULT_WARMUP_STEPS": "default_warmup_steps",
}


def __getattr__(name: str) -> float | int:
    settings_field = _SETTINGS_ATTRS.get(name)
    if settings_field is not None:
        from macaw.config.settings import get_settings

        return getattr(get_settings().worker_lifecycle, settings_field)  # type: ignore[no-any-return]
    msg = f"module {__name__!r} has no attribute {name!r}"
    raise AttributeError(msg)
