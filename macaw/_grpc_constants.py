"""Centralized gRPC channel and server options for the Macaw runtime.

Single source of truth for message size limits, keepalive settings, and
other gRPC tuning parameters used by the scheduler, streaming client,
and TTS service.
"""

from __future__ import annotations

# --- Message size limits ---
# gRPC default is 4MB, insufficient for 25MB audio files.
_MAX_BATCH_MESSAGE_BYTES = 30 * 1024 * 1024  # 30MB (batch audio files)
_MAX_STREAMING_MESSAGE_BYTES = 10 * 1024 * 1024  # 10MB (streaming frames)

# --- Batch channel (Scheduler -> STT worker for TranscribeFile) ---
# Moderate keepalive: batch requests are short-lived, worker crash is detected
# by the RPC failing rather than keepalive timeout.
GRPC_BATCH_CHANNEL_OPTIONS: list[tuple[str, int]] = [
    ("grpc.max_send_message_length", _MAX_BATCH_MESSAGE_BYTES),
    ("grpc.max_receive_message_length", _MAX_BATCH_MESSAGE_BYTES),
    ("grpc.keepalive_time_ms", 30_000),
    ("grpc.keepalive_timeout_ms", 10_000),
    ("grpc.keepalive_permit_without_calls", 1),
]

# --- Streaming channel (StreamingGRPCClient -> STT worker for TranscribeStream) ---
# Aggressive keepalive to detect worker crash via stream break promptly.
# 10s keepalive + 5s timeout = crash detection within ~15s.
# Unlimited pings without data: prevents silent connection death during
# mute-on-speak (no frames sent while TTS active).
GRPC_STREAMING_CHANNEL_OPTIONS: list[tuple[str, int]] = [
    ("grpc.max_send_message_length", _MAX_STREAMING_MESSAGE_BYTES),
    ("grpc.max_receive_message_length", _MAX_STREAMING_MESSAGE_BYTES),
    ("grpc.keepalive_time_ms", 10_000),
    ("grpc.keepalive_timeout_ms", 5_000),
    ("grpc.keepalive_permit_without_calls", 1),
    ("grpc.http2.min_recv_ping_interval_without_data_ms", 5_000),
    ("grpc.http2.max_pings_without_data", 0),
]

# --- TTS channel (Server -> TTS worker for Synthesize/ListVoices) ---
# Same message limits as batch (TTS responses can be large for long texts).
# No custom keepalive: TTS RPCs are request-response, not long-lived streams.
GRPC_TTS_CHANNEL_OPTIONS: list[tuple[str, int]] = [
    ("grpc.max_send_message_length", _MAX_BATCH_MESSAGE_BYTES),
    ("grpc.max_receive_message_length", _MAX_BATCH_MESSAGE_BYTES),
]
