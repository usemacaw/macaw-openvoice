"""Entry point for the TTS worker as a gRPC subprocess.

Usage:
    python -m macaw.workers.tts --port 50052 --engine kokoro \
        --model-path /models/kokoro-v1
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

from macaw._audio_constants import BYTES_PER_SAMPLE_INT16, TTS_DEFAULT_SAMPLE_RATE  # noqa: E402
from macaw.logging import configure_logging, get_logger  # noqa: E402
from macaw.proto import add_TTSWorkerServicer_to_server  # noqa: E402
from macaw.workers._constants import (  # noqa: E402
    DEFAULT_WARMUP_STEPS,
    GRPC_WORKER_SERVER_OPTIONS,
    STOP_GRACE_PERIOD,
)
from macaw.workers.torch_utils import configure_torch_inference  # noqa: E402
from macaw.workers.tts.servicer import TTSWorkerServicer  # noqa: E402

if TYPE_CHECKING:
    from macaw.workers.tts.interface import TTSBackend

logger = get_logger("worker.tts.main")


def _create_backend(engine: str) -> TTSBackend:
    """Create a TTSBackend instance based on the engine name.

    Raises:
        ValueError: If the engine is not supported.
    """
    if engine == "kokoro":
        from macaw.workers.tts.kokoro import KokoroBackend

        return KokoroBackend()

    if engine == "qwen3-tts":
        from macaw.workers.tts.qwen3 import Qwen3TTSBackend

        return Qwen3TTSBackend()

    msg = f"Unsupported TTS engine: {engine}"
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
    configure_torch_inference()

    backend = _create_backend(engine)

    logger.info("loading_model", engine=engine, model_path=model_path)
    await backend.load(model_path, engine_config)
    logger.info("model_loaded", engine=engine)

    warmup_steps = int(engine_config.get("warmup_steps", DEFAULT_WARMUP_STEPS))  # type: ignore[call-overload]
    await _warmup_backend(backend, warmup_steps=warmup_steps)

    model_name = str(engine_config.get("model_name", "unknown"))
    servicer = TTSWorkerServicer(
        backend=backend,
        model_name=model_name,
        engine=engine,
    )

    server = grpc.aio.server(options=GRPC_WORKER_SERVER_OPTIONS)
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
        shutdown_task.add_done_callback(lambda t: t.exception() if not t.cancelled() else None)

    for sig in (signal.SIGTERM, signal.SIGINT):
        loop.add_signal_handler(sig, _signal_handler)

    await server.start()
    logger.info("worker_started", port=port, engine=engine)

    await server.wait_for_termination()


_WARMUP_TEXTS = (
    "Hello.",
    "This is a warmup sentence for the text to speech engine.",
    "The quick brown fox jumps over the lazy dog. Pack my box with five dozen liquor jugs.",
)


async def _warmup_backend(
    backend: TTSBackend, *, warmup_steps: int = DEFAULT_WARMUP_STEPS
) -> None:
    """Run warmup synthesis passes to prime GPU caches, JIT, and memory pools.

    Multiple passes with varied text lengths exercise different decoder
    configurations. RTFx is measured on the final pass as a readiness signal.

    Args:
        backend: Loaded TTS backend.
        warmup_steps: Number of warmup passes (default 3). Set to 0 to skip.
    """
    if warmup_steps <= 0:
        logger.info("warmup_skipped", warmup_steps=warmup_steps)
        return

    rtfx: float | None = None

    for step in range(warmup_steps):
        text = _WARMUP_TEXTS[step % len(_WARMUP_TEXTS)]
        is_last = step == warmup_steps - 1

        try:
            start = time.monotonic()
            total_bytes = 0
            # synthesize() is an async generator in concrete backends
            async for chunk in backend.synthesize(text):  # type: ignore[attr-defined]
                total_bytes += len(chunk)
            elapsed = time.monotonic() - start

            if is_last and elapsed > 0 and total_bytes > 0:
                audio_duration_s = total_bytes / (TTS_DEFAULT_SAMPLE_RATE * BYTES_PER_SAMPLE_INT16)
                rtfx = audio_duration_s / elapsed

            logger.info(
                "warmup_step",
                step=step + 1,
                total=warmup_steps,
                text_len=len(text),
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
                message="TTS system cannot keep up with real-time synthesis",
            )
        logger.info("warmup_complete", rtfx=round(rtfx, 2), warmup_steps=warmup_steps)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(description="Macaw TTS Worker (gRPC)")
    parser.add_argument("--port", type=int, default=50052, help="gRPC port (default: 50052)")
    parser.add_argument(
        "--engine", type=str, default="kokoro", help="Engine TTS (default: kokoro)"
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
    """Main entry point for the TTS worker."""
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


if __name__ == "__main__":
    main()
