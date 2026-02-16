"""`macaw serve` command — starts API Server + workers."""

from __future__ import annotations

import asyncio
import signal
import sys
from pathlib import Path
from typing import TYPE_CHECKING

import click

from macaw.cli.main import cli
from macaw.config.settings import get_settings
from macaw.engines import is_engine_available
from macaw.logging import configure_logging, get_logger

if TYPE_CHECKING:
    from macaw.config.manifest import ModelManifest
    from macaw.postprocessing.pipeline import PostProcessingPipeline
    from macaw.preprocessing.pipeline import AudioPreprocessingPipeline
    from macaw.registry.registry import ModelRegistry
    from macaw.workers.manager import WorkerManager

logger = get_logger("cli.serve")

_s = get_settings()
DEFAULT_HOST = _s.server.host
DEFAULT_PORT = _s.server.port
DEFAULT_MODELS_DIR = _s.worker.models_dir
DEFAULT_WORKER_BASE_PORT = _s.worker.worker_base_port


@cli.command()
@click.option("--host", default=DEFAULT_HOST, show_default=True, help="API Server host.")
@click.option("--port", default=DEFAULT_PORT, type=int, show_default=True, help="HTTP port.")
@click.option(
    "--models-dir",
    default=DEFAULT_MODELS_DIR,
    show_default=True,
    help="Directory with installed models.",
)
@click.option(
    "--cors-origins",
    default="",
    help="CORS origins (comma-separated). Ex: http://localhost:3000",
)
@click.option(
    "--log-format",
    type=click.Choice(["console", "json"]),
    default="console",
    show_default=True,
    help="Log format.",
)
@click.option(
    "--log-level",
    type=click.Choice(["DEBUG", "INFO", "WARNING", "ERROR"]),
    default="INFO",
    show_default=True,
    help="Log level.",
)
def serve(
    host: str,
    port: int,
    models_dir: str,
    cors_origins: str,
    log_format: str,
    log_level: str,
) -> None:
    """Starts the Macaw API Server with workers for installed models."""
    configure_logging(log_format=log_format, level=log_level)
    origins = [o.strip() for o in cors_origins.split(",") if o.strip()] if cors_origins else []
    asyncio.run(_serve(host, port, models_dir, cors_origins=origins))


async def _spawn_all_workers(
    registry: ModelRegistry,
    worker_manager: WorkerManager,
    models: list[ModelManifest],
    base_port: int,
) -> tuple[list[ModelManifest], list[ModelManifest]]:
    """Spawn gRPC workers for all discovered models.

    Returns (stt_models, tts_models) manifests that were processed.
    """
    from macaw._types import ModelType

    port_counter = base_port

    stt_models = [m for m in models if m.model_type == ModelType.STT]
    for manifest in stt_models:
        if not is_engine_available(manifest.engine):
            logger.warning(
                "engine_not_installed",
                model=manifest.name,
                engine=manifest.engine,
                hint=f"pip install macaw-openvoice[{manifest.engine}]",
            )
            continue
        model_path = str(registry.get_model_path(manifest.name))
        await worker_manager.spawn_worker(
            model_name=manifest.name,
            port=port_counter,
            engine=manifest.engine,
            model_path=model_path,
            engine_config=manifest.engine_config.model_dump(),
            worker_type="stt",
        )
        logger.info(
            "worker_spawned",
            model=manifest.name,
            engine=manifest.engine,
            port=port_counter,
            worker_type="stt",
        )
        port_counter += 1

    tts_models = [m for m in models if m.model_type == ModelType.TTS]
    for manifest in tts_models:
        if not is_engine_available(manifest.engine):
            logger.warning(
                "engine_not_installed",
                model=manifest.name,
                engine=manifest.engine,
                hint=f"pip install macaw-openvoice[{manifest.engine}]",
            )
            continue
        model_path = str(registry.get_model_path(manifest.name))
        tts_engine_config = manifest.engine_config.model_dump()
        tts_engine_config["model_name"] = manifest.name
        await worker_manager.spawn_worker(
            model_name=manifest.name,
            port=port_counter,
            engine=manifest.engine,
            model_path=model_path,
            engine_config=tts_engine_config,
            worker_type="tts",
        )
        logger.info(
            "worker_spawned",
            model=manifest.name,
            engine=manifest.engine,
            port=port_counter,
            worker_type="tts",
        )
        port_counter += 1

    return stt_models, tts_models


def _build_pipelines() -> tuple[AudioPreprocessingPipeline, PostProcessingPipeline]:
    """Build preprocessing and postprocessing pipelines from default config."""
    from macaw.config.postprocessing import PostProcessingConfig
    from macaw.config.preprocessing import PreprocessingConfig
    from macaw.postprocessing.itn import ITNStage
    from macaw.postprocessing.pipeline import PostProcessingPipeline as PostPipeline
    from macaw.postprocessing.stages import TextStage  # noqa: TC001
    from macaw.preprocessing.dc_remove import DCRemoveStage
    from macaw.preprocessing.gain_normalize import GainNormalizeStage
    from macaw.preprocessing.pipeline import AudioPreprocessingPipeline as PrePipeline
    from macaw.preprocessing.resample import ResampleStage
    from macaw.preprocessing.stages import AudioStage  # noqa: TC001

    pre_config = PreprocessingConfig()
    pre_stages: list[AudioStage] = []
    if pre_config.resample:
        pre_stages.append(ResampleStage(pre_config.target_sample_rate))
    if pre_config.dc_remove:
        pre_stages.append(DCRemoveStage(pre_config.dc_remove_cutoff_hz))
    if pre_config.gain_normalize:
        pre_stages.append(GainNormalizeStage(pre_config.target_dbfs))
    preprocessing = PrePipeline(pre_config, pre_stages)

    post_config = PostProcessingConfig()
    post_stages: list[TextStage] = []
    if post_config.itn.enabled:
        post_stages.append(ITNStage(post_config.itn.language))
    postprocessing = PostPipeline(post_config, post_stages)

    logger.info(
        "pipelines_configured",
        preprocessing_stages=[s.name for s in pre_stages],
        postprocessing_stages=[s.name for s in post_stages],
    )

    return preprocessing, postprocessing


async def _serve(
    host: str,
    port: int,
    models_dir: str,
    *,
    cors_origins: list[str] | None = None,
) -> None:
    """Main async flow for serve."""
    import uvicorn

    from macaw.registry.registry import ModelRegistry
    from macaw.scheduler.scheduler import Scheduler
    from macaw.server.app import create_app
    from macaw.workers.manager import WorkerManager

    models_path = Path(models_dir).expanduser()

    # 1. Scan registry
    registry = ModelRegistry(models_path)
    await registry.scan()

    models = registry.list_models()
    if not models:
        logger.error("no_models_found", models_dir=str(models_path))
        click.echo(f"Error: no models found in {models_path}", err=True)
        click.echo("Run 'macaw pull <model>' to install a model.", err=True)
        sys.exit(1)

    # 2. Spawn workers
    worker_manager = WorkerManager()
    stt_models, tts_models = await _spawn_all_workers(
        registry, worker_manager, models, DEFAULT_WORKER_BASE_PORT
    )

    if worker_manager.worker_summary()["total"] == 0:
        logger.warning("no_workers_spawned", hint="Install engine packages. See: macaw list")

    logger.info(
        "server_starting",
        host=host,
        port=port,
        models_count=len(models),
        stt_workers=len(stt_models),
        tts_workers=len(tts_models),
    )

    # 3. Build pipelines
    preprocessing_pipeline, postprocessing_pipeline = _build_pipelines()

    # 4. Create app
    scheduler = Scheduler(worker_manager, registry)
    app = create_app(
        registry=registry,
        scheduler=scheduler,
        preprocessing_pipeline=preprocessing_pipeline,
        postprocessing_pipeline=postprocessing_pipeline,
        worker_manager=worker_manager,
        cors_origins=cors_origins,
    )

    # 4.5 Wire StreamingGRPCClient for WebSocket /v1/realtime
    # NOTE: Only the first STT model gets WebSocket streaming support
    stt_worker = worker_manager.get_ready_worker(stt_models[0].name) if stt_models else None
    stt_worker_port = (
        stt_worker.port if stt_worker else (DEFAULT_WORKER_BASE_PORT if stt_models else None)
    )
    if stt_worker_port is not None:
        from macaw.scheduler.streaming import StreamingGRPCClient

        worker_host = get_settings().worker.worker_host
        streaming_client = StreamingGRPCClient(f"{worker_host}:{stt_worker_port}")
        await streaming_client.connect()
        app.state.streaming_grpc_client = streaming_client
        logger.info("streaming_grpc_connected", worker_port=stt_worker_port)

    # 5. Setup shutdown
    shutdown_event = asyncio.Event()
    loop = asyncio.get_running_loop()

    def _handle_signal(s: signal.Signals) -> None:
        logger.info("shutdown_signal", signal=s.name)
        shutdown_event.set()

    for sig in (signal.SIGTERM, signal.SIGINT):
        loop.add_signal_handler(sig, _handle_signal, sig)

    # 6. Run uvicorn
    config = uvicorn.Config(app, host=host, port=port, log_level="warning")
    server = uvicorn.Server(config)

    server_task = asyncio.create_task(server.serve())

    # Wait for shutdown signal or server to stop
    _done, _ = await asyncio.wait(
        [server_task, asyncio.create_task(shutdown_event.wait())],
        return_when=asyncio.FIRST_COMPLETED,
    )

    # 7. Graceful shutdown
    if not server_task.done():
        server.should_exit = True
        await server_task

    # Close streaming gRPC client
    streaming_grpc = getattr(app.state, "streaming_grpc_client", None)
    if streaming_grpc is not None:
        await streaming_grpc.close()

    logger.info("stopping_workers")
    await worker_manager.stop_all()
    logger.info("server_stopped")
