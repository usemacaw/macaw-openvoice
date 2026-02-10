"""Entry point for the TTS worker as a gRPC subprocess.

Usage:
    python -m macaw.workers.tts --port 50052 --engine kokoro \
        --model-path /models/kokoro-v1
"""

from __future__ import annotations

import argparse
import asyncio
import signal
import sys
from typing import TYPE_CHECKING

import grpc.aio

from macaw.logging import configure_logging, get_logger
from macaw.proto import add_TTSWorkerServicer_to_server
from macaw.workers.tts.servicer import TTSWorkerServicer

if TYPE_CHECKING:
    from macaw.workers.tts.interface import TTSBackend

logger = get_logger("worker.tts.main")

STOP_GRACE_PERIOD = 5.0


def _create_backend(engine: str) -> TTSBackend:
    """Create a TTSBackend instance based on the engine name.

    Raises:
        ValueError: If the engine is not supported.
    """
    if engine == "kokoro":
        from macaw.workers.tts.kokoro import KokoroBackend

        return KokoroBackend()

    msg = f"Engine TTS nao suportada: {engine}"
    raise ValueError(msg)


async def serve(
    port: int,
    engine: str,
    model_path: str,
    engine_config: dict[str, object],
) -> None:
    """Start the gRPC server for the TTS worker.

    Args:
        port: Port to listen on.
        engine: Engine name (e.g., "kokoro").
        model_path: Path to model files.
        engine_config: Engine configuration (device, etc).
    """
    backend = _create_backend(engine)

    logger.info("loading_model", engine=engine, model_path=model_path)
    await backend.load(model_path, engine_config)
    logger.info("model_loaded", engine=engine)

    model_name = str(engine_config.get("model_name", "unknown"))
    servicer = TTSWorkerServicer(
        backend=backend,
        model_name=model_name,
        engine=engine,
    )

    server = grpc.aio.server()
    add_TTSWorkerServicer_to_server(servicer, server)  # type: ignore[no-untyped-call]
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
    parser = argparse.ArgumentParser(description="Macaw TTS Worker (gRPC)")
    parser.add_argument("--port", type=int, default=50052, help="gRPC port (default: 50052)")
    parser.add_argument(
        "--engine", type=str, default="kokoro", help="Engine TTS (default: kokoro)"
    )
    parser.add_argument("--model-path", type=str, required=True, help="Model path")
    parser.add_argument(
        "--device", type=str, default="auto", help="Device: auto, cpu, cuda (default: auto)"
    )
    parser.add_argument(
        "--model-name", type=str, default="kokoro-v1", help="Model name (default: kokoro-v1)"
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    """Main entry point for the TTS worker."""
    configure_logging()
    args = parse_args(argv)

    engine_config: dict[str, object] = {
        "model_name": args.model_name,
        "device": args.device,
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


if __name__ == "__main__":
    main()
