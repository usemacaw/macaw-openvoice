"""Centralized gRPC channel and server options for the Macaw runtime.

Single source of truth for message size limits, keepalive settings, and
other gRPC tuning parameters used by the scheduler, streaming client,
and TTS service.

Message size limits are configurable via ``MACAW_GRPC_MAX_BATCH_MESSAGE_MB``
and ``MACAW_GRPC_MAX_STREAMING_MESSAGE_MB`` environment variables.
Keepalive settings are protocol-level and not user-tunable.

Legacy names (``GRPC_BATCH_CHANNEL_OPTIONS``, etc.) are supported via
PEP 562 ``__getattr__`` for backward compatibility.
"""

from __future__ import annotations


def get_batch_channel_options() -> list[tuple[str, int]]:
    """Build gRPC channel options for batch (Scheduler -> STT worker).

    Message sizes read from ``GRPCSettings`` at call time; keepalive
    settings are hardcoded (protocol-level, not user-tunable).
    """
    from macaw.config.settings import get_settings

    max_bytes = get_settings().grpc.max_batch_message_bytes
    return [
        ("grpc.max_send_message_length", max_bytes),
        ("grpc.max_receive_message_length", max_bytes),
        ("grpc.keepalive_time_ms", 30_000),
        ("grpc.keepalive_timeout_ms", 10_000),
        ("grpc.keepalive_permit_without_calls", 1),
    ]


def get_streaming_channel_options() -> list[tuple[str, int]]:
    """Build gRPC channel options for streaming (StreamingGRPCClient -> STT worker).

    Aggressive keepalive to detect worker crash via stream break promptly.
    10s keepalive + 5s timeout = crash detection within ~15s.
    Unlimited pings without data: prevents silent connection death during
    mute-on-speak (no frames sent while TTS active).
    """
    from macaw.config.settings import get_settings

    max_bytes = get_settings().grpc.max_streaming_message_bytes
    return [
        ("grpc.max_send_message_length", max_bytes),
        ("grpc.max_receive_message_length", max_bytes),
        ("grpc.keepalive_time_ms", 10_000),
        ("grpc.keepalive_timeout_ms", 5_000),
        ("grpc.keepalive_permit_without_calls", 1),
        ("grpc.http2.min_recv_ping_interval_without_data_ms", 5_000),
        ("grpc.http2.max_pings_without_data", 0),
    ]


def get_tts_channel_options() -> list[tuple[str, int]]:
    """Build gRPC channel options for TTS (Server -> TTS worker).

    Same message limits as batch (TTS responses can be large for long texts).
    No custom keepalive: TTS RPCs are request-response, not long-lived streams.
    """
    from macaw.config.settings import get_settings

    max_bytes = get_settings().grpc.max_batch_message_bytes
    return [
        ("grpc.max_send_message_length", max_bytes),
        ("grpc.max_receive_message_length", max_bytes),
    ]


# --- PEP 562 backward compatibility ---
# Old imports like ``from macaw._grpc_constants import GRPC_BATCH_CHANNEL_OPTIONS``
# resolve to a function call at access time.
_COMPAT_NAMES: dict[str, str] = {
    "GRPC_BATCH_CHANNEL_OPTIONS": "get_batch_channel_options",
    "GRPC_STREAMING_CHANNEL_OPTIONS": "get_streaming_channel_options",
    "GRPC_TTS_CHANNEL_OPTIONS": "get_tts_channel_options",
}


def __getattr__(name: str) -> list[tuple[str, int]]:
    func_name = _COMPAT_NAMES.get(name)
    if func_name is not None:
        func = globals()[func_name]
        result: list[tuple[str, int]] = func()
        return result
    msg = f"module {__name__!r} has no attribute {name!r}"
    raise AttributeError(msg)
