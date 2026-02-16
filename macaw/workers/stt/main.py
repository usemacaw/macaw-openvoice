"""Entry point for the STT worker as a gRPC subprocess.

Usage:
    python -m macaw.workers.stt --port 50051 --engine faster-whisper \
        --model-path /models/large-v3 --model-size large-v3
"""

from __future__ import annotations

# NB: CUDA env must be configured before importing torch.
from macaw.workers.torch_utils import configure_cuda_env

configure_cuda_env()

import argparse  # noqa: E402
import asyncio  # noqa: E402
import signal  # noqa: E402
import sys  # noqa: E402
import time  # noqa: E402
from typing import TYPE_CHECKING  # noqa: E402

import grpc.aio  # noqa: E402

from macaw.logging import configure_logging, get_logger  # noqa: E402
from macaw.proto import add_STTWorkerServicer_to_server  # noqa: E402
from macaw.workers.stt.servicer import STTWorkerServicer  # noqa: E402
from macaw.workers.torch_utils import configure_torch_inference  # noqa: E402

if TYPE_CHECKING:
    from macaw.workers.stt.interface import STTBackend

logger = get_logger("worker.stt.main")

STOP_GRACE_PERIOD = 5.0


def _create_backend(engine: str) -> STTBackend:
    """Create an STTBackend instance based on the engine name.

    Raises:
        ValueError: If the engine is not supported.
    """
    if engine == "faster-whisper":
        from macaw.workers.stt.faster_whisper import FasterWhisperBackend

        return FasterWhisperBackend()

    msg = f"Unsupported STT engine: {engine}"
    raise ValueError(msg)


async def serve(
    port: int,
    engine: str,
    model_path: str,
    engine_config: dict[str, object],
) -> None:
    """Start the gRPC server for the STT worker.

    Args:
        port: Port to listen on.
        engine: Engine name (e.g., "faster-whisper").
        model_path: Path to model files.
        engine_config: Engine configuration (compute_type, device, etc).
    """
    configure_torch_inference()

    backend = _create_backend(engine)

    logger.info("loading_model", engine=engine, model_path=model_path)
    await backend.load(model_path, engine_config)
    logger.info("model_loaded", engine=engine)

    warmup_steps = int(engine_config.get("warmup_steps", 3))  # type: ignore[call-overload]
    await _warmup_backend(backend, warmup_steps=warmup_steps)

    model_name = str(engine_config.get("model_size", "unknown"))
    servicer = STTWorkerServicer(
        backend=backend,
        model_name=model_name,
        engine=engine,
    )

    server = grpc.aio.server(
        options=[
            ("grpc.http2.min_recv_ping_interval_without_data_ms", 5_000),
            ("grpc.keepalive_permit_without_calls", 1),
        ]
    )
    add_STTWorkerServicer_to_server(servicer, server)  # type: ignore[no-untyped-call]
    listen_addr = f"[::]:{port}"
    server.add_insecure_port(listen_addr)

    loop = asyncio.get_running_loop()
    shutting_down = False
    shutdown_task: asyncio.Task[None] | None = None

    async def _shutdown() -> None:
        nonlocal shutting_down
        if shutting_down:
            return
        shutting_down = True
        logger.info("shutdown_start", grace_period=STOP_GRACE_PERIOD)
        await server.stop(STOP_GRACE_PERIOD)
        await backend.unload()
        logger.info("shutdown_complete")

    def _signal_handler() -> None:
        nonlocal shutdown_task
        shutdown_task = asyncio.ensure_future(_shutdown())
        shutdown_task.add_done_callback(lambda t: t.exception() if not t.cancelled() else None)

    for sig in (signal.SIGTERM, signal.SIGINT):
        loop.add_signal_handler(sig, _signal_handler)

    await server.start()
    logger.info("worker_started", port=port, engine=engine)

    await server.wait_for_termination()


_WARMUP_AUDIO_DURATIONS = (1.0, 3.0, 5.0)
_SAMPLE_RATE = 16000


async def _warmup_backend(backend: STTBackend, *, warmup_steps: int = 3) -> None:
    """Run warmup inference passes to prime GPU caches, JIT, and memory pools.

    Multiple passes with varied audio lengths exercise different CUDA kernel
    configurations. RTFx is measured on the final pass as a readiness signal.

    Args:
        backend: Loaded STT backend.
        warmup_steps: Number of warmup passes (default 3). Set to 0 to skip.
    """
    if warmup_steps <= 0:
        logger.info("warmup_skipped", warmup_steps=warmup_steps)
        return

    rtfx: float | None = None

    for step in range(warmup_steps):
        duration_s = _WARMUP_AUDIO_DURATIONS[step % len(_WARMUP_AUDIO_DURATIONS)]
        num_samples = int(duration_s * _SAMPLE_RATE)
        silence = b"\x00\x00" * num_samples

        is_last = step == warmup_steps - 1

        try:
            start = time.monotonic()
            await backend.transcribe_file(silence)
            elapsed = time.monotonic() - start

            if is_last and elapsed > 0:
                rtfx = duration_s / elapsed

            logger.info(
                "warmup_step",
                step=step + 1,
                total=warmup_steps,
                audio_duration_s=duration_s,
                elapsed_s=round(elapsed, 3),
            )
        except Exception as exc:
            logger.warning("warmup_step_failed", step=step + 1, error=str(exc))
            return

    if rtfx is not None:
        if rtfx < 1.0:
            logger.warning(
                "warmup_rtfx_below_realtime",
                rtfx=round(rtfx, 2),
                message="System cannot keep up with real-time audio input",
            )
        logger.info("warmup_complete", rtfx=round(rtfx, 2), warmup_steps=warmup_steps)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(description="Macaw STT Worker (gRPC)")
    parser.add_argument("--port", type=int, default=50051, help="gRPC port (default: 50051)")
    parser.add_argument(
        "--engine", type=str, default="faster-whisper", help="Engine STT (default: faster-whisper)"
    )
    parser.add_argument("--model-path", type=str, required=True, help="Model path")
    parser.add_argument(
        "--engine-config",
        type=str,
        default="{}",
        help="Engine config as JSON string",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    """Main entry point for the STT worker."""
    import json

    configure_logging()
    args = parse_args(argv)

    engine_config: dict[str, object] = json.loads(args.engine_config)

    try:
        asyncio.run(
            serve(
                port=args.port,
                engine=args.engine,
                model_path=args.model_path,
                engine_config=engine_config,
            )
        )
    except KeyboardInterrupt:
        logger.info("worker_interrupted")
        sys.exit(0)
