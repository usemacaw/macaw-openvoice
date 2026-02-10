"""Entry point for the STT worker as a gRPC subprocess.

Usage:
    python -m macaw.workers.stt --port 50051 --engine faster-whisper \
        --model-path /models/large-v3 --model-size large-v3
"""

from __future__ import annotations

import argparse
import asyncio
import signal
import sys
from typing import TYPE_CHECKING

import grpc.aio

from macaw.logging import configure_logging, get_logger
from macaw.proto import add_STTWorkerServicer_to_server
from macaw.workers.stt.servicer import STTWorkerServicer

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

    if engine == "wenet":
        from macaw.workers.stt.wenet import WeNetBackend

        return WeNetBackend()

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
    backend = _create_backend(engine)

    logger.info("loading_model", engine=engine, model_path=model_path)
    await backend.load(model_path, engine_config)
    logger.info("model_loaded", engine=engine)

    model_name = str(engine_config.get("model_size", "unknown"))
    servicer = STTWorkerServicer(
        backend=backend,
        model_name=model_name,
        engine=engine,
    )

    server = grpc.aio.server()
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

    for sig in (signal.SIGTERM, signal.SIGINT):
        loop.add_signal_handler(sig, _signal_handler)

    await server.start()
    logger.info("worker_started", port=port, engine=engine)

    await server.wait_for_termination()


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(description="Macaw STT Worker (gRPC)")
    parser.add_argument("--port", type=int, default=50051, help="gRPC port (default: 50051)")
    parser.add_argument(
        "--engine", type=str, default="faster-whisper", help="Engine STT (default: faster-whisper)"
    )
    parser.add_argument("--model-path", type=str, required=True, help="Model path")
    parser.add_argument(
        "--compute-type", type=str, default="float16", help="Compute type (default: float16)"
    )
    parser.add_argument(
        "--device", type=str, default="auto", help="Device: auto, cpu, cuda (default: auto)"
    )
    parser.add_argument(
        "--model-size", type=str, default="large-v3", help="Model size (default: large-v3)"
    )
    parser.add_argument(
        "--beam-size", type=int, default=5, help="Beam size for decoding (default: 5)"
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    """Main entry point for the STT worker."""
    configure_logging()
    args = parse_args(argv)

    engine_config: dict[str, object] = {
        "model_size": args.model_size,
        "compute_type": args.compute_type,
        "device": args.device,
        "beam_size": args.beam_size,
    }

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
