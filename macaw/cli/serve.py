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
    default=None,
    help="CORS origins (comma-separated). Overrides MACAW_CORS_ORIGINS env var.",
)
@click.option(
    "--voice-dir",
    default=None,
    help="Directory for saved voices (enables Voice CRUD). Overrides MACAW_VOICE_DIR.",
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
    cors_origins: str | None,
    voice_dir: str | None,
    log_format: str,
    log_level: str,
) -> None:
    """Starts the Macaw API Server with workers for installed models."""
    configure_logging(log_format=log_format, level=log_level)
    # CLI flag overrides env var; if neither set, no CORS.
    if cors_origins is not None:
        origins = [o.strip() for o in cors_origins.split(",") if o.strip()]
    else:
        origins = get_settings().server.cors_origins_list
    # CLI flag overrides env var; if neither set, no VoiceStore.
    if voice_dir is None:
        voice_dir = get_settings().server.voice_dir
    asyncio.run(
        _serve(
            host, port, models_dir, cors_origins=origins, voice_dir=voice_dir, log_level=log_level
        )
    )


async def _spawn_or_register_worker(
    manifest: ModelManifest,
    worker_type: str,
    registry: ModelRegistry,
    worker_manager: WorkerManager,
    remote_workers: dict[str, str],
    port: int,
) -> bool:
    """Spawn a local worker or register a remote one for *manifest*.

    Returns ``True`` if a local worker was spawned (port consumed),
    ``False`` if the worker was remote or the engine is unavailable.
    """
    from macaw.backends.venv_manager import resolve_python_for_engine

    remote_ep = remote_workers.get(manifest.engine)
    if remote_ep:
        await worker_manager.register_remote_worker(
            model_name=manifest.name,
            engine=manifest.engine,
            remote_endpoint=remote_ep,
            worker_type=worker_type,
        )
        logger.info(
            "remote_worker_registered",
            model=manifest.name,
            engine=manifest.engine,
            remote_endpoint=remote_ep,
            worker_type=worker_type,
        )
        return False

    # Run in thread — resolve_python_for_engine() may call provision()
    # which runs blocking subprocess commands (up to 600s).
    venv_python = await asyncio.to_thread(resolve_python_for_engine, manifest.engine)
    effective_venv = venv_python if venv_python != sys.executable else None
    if not is_engine_available(
        manifest.engine,
        python_package=manifest.python_package,
        venv_python=effective_venv,
    ):
        # Venv check failed — try the main Python as ultimate fallback.
        # This handles the case where the user installed deps in the main
        # environment (e.g., ``pip install macaw-openvoice[kokoro]``) but
        # the per-engine venv is stale, corrupt, or was never provisioned.
        if effective_venv is not None and is_engine_available(
            manifest.engine,
            python_package=manifest.python_package,
        ):
            logger.warning(
                "engine_available_in_main_python",
                model=manifest.name,
                engine=manifest.engine,
            )
            venv_python = sys.executable
            effective_venv = None
        else:
            logger.warning(
                "engine_not_installed",
                model=manifest.name,
                engine=manifest.engine,
                hint=f"pip install macaw-openvoice[{manifest.engine}]",
            )
            return False

    model_path = str(registry.get_model_path(manifest.name))
    engine_config = manifest.engine_config.model_dump()
    if worker_type == "tts":
        engine_config["model_name"] = manifest.name

    await worker_manager.spawn_worker(
        model_name=manifest.name,
        port=port,
        engine=manifest.engine,
        model_path=model_path,
        engine_config=engine_config,
        worker_type=worker_type,
        python_package=manifest.python_package,
        venv_python=venv_python,
    )
    logger.info(
        "worker_spawned",
        model=manifest.name,
        engine=manifest.engine,
        port=port,
        worker_type=worker_type,
        venv_python=venv_python,
    )
    return True


async def _spawn_all_workers(
    registry: ModelRegistry,
    worker_manager: WorkerManager,
    models: list[ModelManifest],
    base_port: int,
) -> tuple[list[ModelManifest], list[ModelManifest], int, int]:
    """Spawn gRPC workers for all discovered models.

    Checks ``BackendSettings.remote_workers`` first — if an engine has a
    remote endpoint configured, a remote worker handle is registered instead
    of spawning a local subprocess. This enables horizontal scaling with
    engine containers in Docker/K8s.

    Returns (stt_models, tts_models, stt_spawned, tts_spawned).
    """
    from macaw._types import ModelType

    remote_workers = get_settings().backend.remote_workers
    port_counter = base_port

    stt_spawned = 0
    stt_models = [m for m in models if m.model_type == ModelType.STT]
    for manifest in stt_models:
        spawned = await _spawn_or_register_worker(
            manifest,
            "stt",
            registry,
            worker_manager,
            remote_workers,
            port_counter,
        )
        if spawned:
            port_counter += 1
            stt_spawned += 1

    tts_spawned = 0
    tts_models = [m for m in models if m.model_type == ModelType.TTS]
    for manifest in tts_models:
        spawned = await _spawn_or_register_worker(
            manifest,
            "tts",
            registry,
            worker_manager,
            remote_workers,
            port_counter,
        )
        if spawned:
            port_counter += 1
            tts_spawned += 1

    return stt_models, tts_models, stt_spawned, tts_spawned


def _build_pipelines() -> tuple[AudioPreprocessingPipeline, PostProcessingPipeline]:
    """Build preprocessing and postprocessing pipelines from default config."""
    from macaw.config.postprocessing import ITNConfig, PostProcessingConfig
    from macaw.config.preprocessing import PreprocessingConfig
    from macaw.config.settings import get_settings
    from macaw.postprocessing.itn import ITNStage
    from macaw.postprocessing.pipeline import PostProcessingPipeline as PostPipeline
    from macaw.postprocessing.stages import TextStage  # noqa: TC001
    from macaw.preprocessing.dc_remove import DCRemoveStage
    from macaw.preprocessing.gain_normalize import GainNormalizeStage
    from macaw.preprocessing.pipeline import AudioPreprocessingPipeline as PrePipeline
    from macaw.preprocessing.resample import ResampleStage
    from macaw.preprocessing.stages import AudioStage  # noqa: TC001

    settings = get_settings()
    pre_config = PreprocessingConfig(
        dc_remove_cutoff_hz=settings.preprocessing.dc_cutoff_hz,
        target_dbfs=settings.preprocessing.target_dbfs,
    )
    pre_stages: list[AudioStage] = []
    if pre_config.resample:
        pre_stages.append(ResampleStage(pre_config.target_sample_rate))
    if pre_config.dc_remove:
        pre_stages.append(DCRemoveStage(pre_config.dc_remove_cutoff_hz))
    if pre_config.gain_normalize:
        pre_stages.append(GainNormalizeStage(pre_config.target_dbfs))
    preprocessing = PrePipeline(pre_config, pre_stages)

    post_config = PostProcessingConfig(
        itn=ITNConfig(default_language=settings.postprocessing.itn_default_language),
    )
    post_stages: list[TextStage] = []
    if post_config.itn.enabled:
        post_stages.append(ITNStage(default_language=post_config.itn.default_language))
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
    voice_dir: str | None = None,
    log_level: str = "WARNING",
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
    stt_models, tts_models, stt_spawned, tts_spawned = await _spawn_all_workers(
        registry, worker_manager, models, DEFAULT_WORKER_BASE_PORT
    )

    if worker_manager.worker_summary()["total"] == 0:
        logger.warning("no_workers_spawned", hint="Install engine packages. See: macaw list")

    logger.info(
        "server_starting",
        host=host,
        port=port,
        models_count=len(models),
        stt_models=len(stt_models),
        tts_models=len(tts_models),
        stt_workers_spawned=stt_spawned,
        tts_workers_spawned=tts_spawned,
    )

    # 3. Build pipelines
    preprocessing_pipeline, postprocessing_pipeline = _build_pipelines()

    # 4. Create app
    scheduler = Scheduler(worker_manager, registry)

    # VoiceStore: enable when voice_dir is provided via CLI or env var
    voice_store = None
    if voice_dir is not None:
        from macaw.server.voice_store import FileSystemVoiceStore

        voice_store = FileSystemVoiceStore(voice_dir)
        logger.info("voice_store_enabled", voice_dir=voice_dir)

    app = create_app(
        registry=registry,
        scheduler=scheduler,
        preprocessing_pipeline=preprocessing_pipeline,
        postprocessing_pipeline=postprocessing_pipeline,
        worker_manager=worker_manager,
        voice_store=voice_store,
        cors_origins=cors_origins,
    )

    # 4.5 Wire StreamingGRPCClient for WebSocket /v1/realtime
    # NOTE: Only the first STT model gets WebSocket streaming support
    stt_worker = worker_manager.get_ready_worker(stt_models[0].name) if stt_models else None
    if stt_worker is None and stt_models:
        # Worker may still be starting (e.g., remote worker health probing)
        # Fall back to getting any worker handle for the first STT model
        stt_worker = worker_manager.get_worker_for_model(stt_models[0].name)
    if stt_worker is not None:
        from macaw.scheduler.streaming import StreamingGRPCClient

        streaming_address = stt_worker.worker_address
        streaming_client = StreamingGRPCClient(streaming_address)
        await streaming_client.connect()
        app.state.streaming_grpc_client = streaming_client
        logger.info("streaming_grpc_connected", worker_address=streaming_address)
    elif stt_models:
        # Fallback: use default port for backward compatibility
        from macaw.scheduler.streaming import StreamingGRPCClient

        worker_host = get_settings().worker.worker_host
        fallback_address = f"{worker_host}:{DEFAULT_WORKER_BASE_PORT}"
        streaming_client = StreamingGRPCClient(fallback_address)
        await streaming_client.connect()
        app.state.streaming_grpc_client = streaming_client
        logger.info("streaming_grpc_connected", worker_address=fallback_address)

    # 5. Setup shutdown
    shutdown_event = asyncio.Event()
    loop = asyncio.get_running_loop()

    def _handle_signal(s: signal.Signals) -> None:
        logger.info("shutdown_signal", signal=s.name)
        shutdown_event.set()

    for sig in (signal.SIGTERM, signal.SIGINT):
        loop.add_signal_handler(sig, _handle_signal, sig)

    # 6. Run uvicorn
    config = uvicorn.Config(app, host=host, port=port, log_level=log_level.lower())
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
