"""Shared constants for STT and TTS worker subprocesses.

Single source of truth for worker lifecycle parameters and gRPC server
options that are identical between STT and TTS workers.
"""

from __future__ import annotations

# Grace period (seconds) for gRPC server.stop() before forceful termination.
# Used by both worker mains and the WorkerManager's stop logic.
STOP_GRACE_PERIOD: float = 5.0

# Default number of warmup inference passes at worker startup.
DEFAULT_WARMUP_STEPS: int = 3

# gRPC server-side options for worker subprocesses.
# Matches the client-side streaming keepalive expectations.
GRPC_WORKER_SERVER_OPTIONS: list[tuple[str, int]] = [
    ("grpc.http2.min_recv_ping_interval_without_data_ms", 5_000),
    ("grpc.keepalive_permit_without_calls", 1),
]
