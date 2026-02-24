"""Worker Manager — manages the lifecycle of STT/TTS workers as subprocesses.

Responsibilities:
- Spawn workers as gRPC subprocesses
- Health probing with exponential backoff
- Process monitoring (crash detection)
- Auto-restart with rate limiting
- Graceful shutdown
"""

from __future__ import annotations

import asyncio
import contextlib
import os
import subprocess
import sys
import time
from collections.abc import Callable, Coroutine
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any

from macaw.logging import get_logger

logger = get_logger("worker.manager")


class WorkerType(Enum):
    """Worker type — STT or TTS."""

    STT = "stt"
    TTS = "tts"


class WorkerState(Enum):
    """Worker state in the lifecycle."""

    STARTING = "starting"
    READY = "ready"
    BUSY = "busy"
    STOPPING = "stopping"
    STOPPED = "stopped"
    CRASHED = "crashed"


@dataclass
class WorkerHandle:
    """Handle for a running worker."""

    worker_id: str
    port: int
    model_name: str
    engine: str
    process: subprocess.Popen[bytes] | None = None
    state: WorkerState = WorkerState.STARTING
    crash_count: int = 0
    crash_timestamps: list[float] = field(default_factory=list)
    last_started_at: float = field(default_factory=time.monotonic)
    model_path: str = ""
    engine_config: dict[str, object] = field(default_factory=dict)
    worker_type: WorkerType = WorkerType.STT
    python_package: str | None = None
    venv_python: str | None = None
    remote_endpoint: str | None = None

    @property
    def worker_address(self) -> str:
        """Return gRPC address (host:port) for this worker.

        For remote workers, returns the configured endpoint.
        For local workers, combines worker_host setting with the port.
        """
        if self.remote_endpoint is not None:
            return self.remote_endpoint
        from macaw.config.settings import get_settings

        return f"{get_settings().worker.worker_host}:{self.port}"


class WorkerManager:
    """Manage lifecycle of gRPC workers as subprocesses.

    Each worker is a separate process running the STT gRPC servicer.
    The manager handles spawn, health checks, crash detection, and restart.
    """

    def __init__(self) -> None:
        from macaw.config.settings import get_settings

        self._workers: dict[str, WorkerHandle] = {}
        self._tasks: dict[str, list[asyncio.Task[None]]] = {}
        self._lifecycle = get_settings().worker_lifecycle

    async def register_remote_worker(
        self,
        model_name: str,
        engine: str,
        remote_endpoint: str,
        worker_type: WorkerType | str = WorkerType.STT,
    ) -> WorkerHandle:
        """Register a remote worker (no subprocess spawning).

        The worker process is managed externally (e.g., Docker, K8s).
        Only health probing is performed locally — no process monitoring.

        Args:
            model_name: Model name (for identification).
            engine: Engine name (e.g., "faster-whisper", "kokoro").
            remote_endpoint: gRPC address as "host:port".
            worker_type: Worker type (WorkerType enum or "stt"/"tts" string).

        Returns:
            WorkerHandle with worker information.

        Raises:
            ValueError: If remote_endpoint format is invalid.
        """
        if isinstance(worker_type, str):
            worker_type = WorkerType(worker_type)

        # Validate endpoint format
        if ":" not in remote_endpoint:
            msg = f"Invalid remote endpoint format: {remote_endpoint!r} (expected 'host:port')"
            raise ValueError(msg)

        _host, port_str = remote_endpoint.rsplit(":", 1)
        if not _host or not port_str.isdigit():
            msg = f"Invalid remote endpoint format: {remote_endpoint!r} (expected 'host:port')"
            raise ValueError(msg)

        port = int(port_str)
        worker_id = f"{engine}-remote-{port}"

        logger.info(
            "registering_remote_worker",
            worker_id=worker_id,
            engine=engine,
            remote_endpoint=remote_endpoint,
            worker_type=worker_type.value,
        )

        handle = WorkerHandle(
            worker_id=worker_id,
            port=port,
            model_name=model_name,
            engine=engine,
            process=None,
            state=WorkerState.STARTING,
            worker_type=worker_type,
            remote_endpoint=remote_endpoint,
        )

        self._workers[worker_id] = handle

        # Only start health probe — no monitor task (no local process to poll)
        health_task = asyncio.create_task(self._health_probe(worker_id))
        self._tasks[worker_id] = [health_task]

        return handle

    async def spawn_worker(
        self,
        model_name: str,
        port: int,
        engine: str,
        model_path: str,
        engine_config: dict[str, object],
        worker_type: WorkerType | str = WorkerType.STT,
        *,
        python_package: str | None = None,
        venv_python: str | None = None,
    ) -> WorkerHandle:
        """Start a new worker as a subprocess.

        Args:
            model_name: Model name (for identification).
            port: gRPC port for the worker to listen on.
            engine: Engine name (e.g., "faster-whisper", "kokoro").
            model_path: Path to model files.
            engine_config: Engine configuration.
            worker_type: Worker type (WorkerType enum or "stt"/"tts" string).
            python_package: Dotted module path for external engines (optional).
            venv_python: Path to isolated venv Python executable (optional).

        Returns:
            WorkerHandle with worker information.
        """
        if isinstance(worker_type, str):
            worker_type = WorkerType(worker_type)

        worker_id = f"{engine}-{port}"

        logger.info(
            "spawning_worker",
            worker_id=worker_id,
            port=port,
            engine=engine,
            worker_type=worker_type.value,
        )

        process = _spawn_worker_process(
            port,
            engine,
            model_path,
            engine_config,
            worker_type=worker_type,
            python_package=python_package,
            venv_python=venv_python,
        )

        handle = WorkerHandle(
            worker_id=worker_id,
            port=port,
            model_name=model_name,
            engine=engine,
            process=process,
            state=WorkerState.STARTING,
            model_path=model_path,
            engine_config=engine_config,
            worker_type=worker_type,
            python_package=python_package,
            venv_python=venv_python,
        )

        self._workers[worker_id] = handle
        self._start_background_tasks(worker_id)

        return handle

    def _start_background_tasks(self, worker_id: str) -> None:
        """Start health probe and monitoring tasks for a worker."""
        health_task = asyncio.create_task(self._health_probe(worker_id))
        monitor_task = asyncio.create_task(self._monitor_worker(worker_id))
        self._tasks[worker_id] = [health_task, monitor_task]

    async def _cancel_background_tasks(self, worker_id: str) -> None:
        """Cancel and await completion of background tasks for a worker."""
        tasks = self._tasks.pop(worker_id, [])
        for task in tasks:
            task.cancel()
        if tasks:
            await asyncio.gather(*tasks, return_exceptions=True)

    async def stop_worker(self, worker_id: str) -> None:
        """Stop a worker gracefully (SIGTERM, wait, SIGKILL if needed).

        For remote workers, skips process termination (no local process).

        Args:
            worker_id: Worker ID.
        """
        handle = self._workers.get(worker_id)
        if handle is None:
            return

        handle.state = WorkerState.STOPPING
        logger.info("stopping_worker", worker_id=worker_id)

        await self._cancel_background_tasks(worker_id)

        # Skip process termination for remote workers (no local process)
        if handle.remote_endpoint is None:
            process = handle.process
            if process is not None and process.poll() is None:
                process.terminate()
                try:
                    await asyncio.wait_for(
                        asyncio.get_running_loop().run_in_executor(None, process.wait),
                        timeout=self._lifecycle.stop_grace_period_s,
                    )
                except asyncio.TimeoutError:  # noqa: UP041
                    logger.warning("worker_force_kill", worker_id=worker_id)
                    process.kill()
                    try:
                        await asyncio.wait_for(
                            asyncio.get_running_loop().run_in_executor(None, process.wait),
                            timeout=self._lifecycle.stop_grace_period_s,
                        )
                    except asyncio.TimeoutError:  # noqa: UP041
                        logger.warning("worker_force_kill_timeout", worker_id=worker_id)

        handle.state = WorkerState.STOPPED
        logger.info("worker_stopped", worker_id=worker_id)

    async def stop_all(self) -> None:
        """Stop all workers in parallel."""
        worker_ids = list(self._workers.keys())
        if worker_ids:
            await asyncio.gather(*(self.stop_worker(wid) for wid in worker_ids))

    def get_worker(self, worker_id: str) -> WorkerHandle | None:
        """Return worker handle by ID."""
        return self._workers.get(worker_id)

    def get_ready_worker(self, model_name: str) -> WorkerHandle | None:
        """Return the first READY worker for a given model."""
        for handle in self._workers.values():
            if handle.model_name == model_name and handle.state == WorkerState.READY:
                return handle
        return None

    def get_worker_for_model(self, model_name: str) -> WorkerHandle | None:
        """Return any worker handle for a model, regardless of state.

        Useful when a READY worker is not yet available (e.g., remote worker
        still health-probing) but the worker address is needed.
        """
        for handle in self._workers.values():
            if handle.model_name == model_name:
                return handle
        return None

    def worker_summary(self) -> dict[str, int]:
        """Return a summary of worker states.

        Returns:
            Dict with counts: total, ready, starting, crashed.
        """
        total = len(self._workers)
        ready = sum(1 for h in self._workers.values() if h.state == WorkerState.READY)
        starting = sum(1 for h in self._workers.values() if h.state == WorkerState.STARTING)
        crashed = sum(1 for h in self._workers.values() if h.state == WorkerState.CRASHED)
        return {"total": total, "ready": ready, "starting": starting, "crashed": crashed}

    async def _health_probe(self, worker_id: str) -> None:
        """Check worker health with exponential backoff after spawn.

        Transitions from STARTING -> READY when health returns ok.
        """
        handle = self._workers.get(worker_id)
        if handle is None:
            return

        delay = self._lifecycle.health_probe_initial_delay_s
        start = time.monotonic()

        # Extract host for remote workers
        remote_host: str | None = None
        if handle.remote_endpoint is not None:
            remote_host = handle.remote_endpoint.rsplit(":", 1)[0]

        while handle.state == WorkerState.STARTING:
            elapsed = time.monotonic() - start
            if elapsed > self._lifecycle.health_probe_timeout_s:
                logger.error("health_probe_timeout", worker_id=worker_id)
                handle.state = WorkerState.CRASHED
                return

            try:
                await asyncio.sleep(delay)
                result = await _check_worker_health(
                    handle.port,
                    timeout=self._lifecycle.health_probe_rpc_timeout_s,
                    worker_type=handle.worker_type,
                    host=remote_host,
                )
                if result.get("status") == "ok":
                    handle.state = WorkerState.READY
                    logger.info("worker_ready", worker_id=worker_id, elapsed_s=round(elapsed, 2))
                    return
            except asyncio.CancelledError:
                return
            except Exception:
                logger.debug("health_probe_error", worker_id=worker_id, exc_info=True)

            delay = min(delay * 2, self._lifecycle.health_probe_max_delay_s)

    async def _monitor_worker(self, worker_id: str) -> None:
        """Monitor the worker process and detect crashes.

        Returns immediately for remote workers (no local process to poll).
        """
        handle = self._workers.get(worker_id)
        if handle is None:
            return

        # Remote workers have no local process to monitor
        if handle.remote_endpoint is not None:
            return

        try:
            while handle.state not in (WorkerState.STOPPING, WorkerState.STOPPED):
                await asyncio.sleep(self._lifecycle.monitor_interval_s)

                process = handle.process
                if process is None:
                    continue

                exit_code = process.poll()
                if exit_code is not None and handle.state not in (
                    WorkerState.STOPPING,
                    WorkerState.STOPPED,
                ):
                    stderr_output = ""
                    if process.stderr is not None:
                        with contextlib.suppress(Exception):
                            stderr_output = process.stderr.read().decode(errors="replace")[-2000:]
                    logger.error(
                        "worker_crashed",
                        worker_id=worker_id,
                        exit_code=exit_code,
                        stderr=stderr_output or "(empty)",
                    )
                    handle.state = WorkerState.CRASHED
                    await self._attempt_restart(worker_id)
                    return
        except asyncio.CancelledError:
            return

    async def _attempt_restart(self, worker_id: str) -> None:
        """Attempt to restart a worker with rate limiting.

        Do not restart if max_crashes_in_window is exceeded within crash_window_s.

        For remote workers, no subprocess is spawned — only health probing
        resumes to detect when the remote host restarts the worker.
        """
        handle = self._workers.get(worker_id)
        if handle is None:
            return

        now = time.monotonic()
        handle.crash_count += 1
        handle.crash_timestamps.append(now)

        crash_window = self._lifecycle.crash_window_s

        # Clear timestamps outside the window
        handle.crash_timestamps = [ts for ts in handle.crash_timestamps if now - ts < crash_window]

        if len(handle.crash_timestamps) >= self._lifecycle.max_crashes_in_window:
            logger.error(
                "worker_max_crashes_exceeded",
                worker_id=worker_id,
                crashes=handle.crash_count,
                window_seconds=crash_window,
            )
            handle.state = WorkerState.CRASHED
            return

        # Backoff based on the number of recent crashes
        backoff = self._lifecycle.health_probe_initial_delay_s * (
            2 ** len(handle.crash_timestamps)
        )
        logger.info(
            "worker_restarting",
            worker_id=worker_id,
            backoff_s=backoff,
            crash_count=handle.crash_count,
        )

        await asyncio.sleep(backoff)

        if handle.remote_endpoint is not None:
            # Remote workers: no subprocess to spawn.
            # Transition to STARTING and resume health probing.
            handle.state = WorkerState.STARTING
            handle.last_started_at = time.monotonic()
            self._tasks.pop(worker_id, None)
            health_task = asyncio.create_task(self._health_probe(worker_id))
            self._tasks[worker_id] = [health_task]
            return

        process = _spawn_worker_process(
            handle.port,
            handle.engine,
            handle.model_path,
            handle.engine_config,
            worker_type=handle.worker_type,
            python_package=handle.python_package,
            venv_python=handle.venv_python,
        )

        handle.process = process
        handle.state = WorkerState.STARTING
        handle.last_started_at = time.monotonic()

        # Do not call _cancel_background_tasks here — we are executing INSIDE
        # _monitor_worker, which is one of the tasks in the dict. Cancelling itself
        # would cause CancelledError and _start_background_tasks would never run.
        # Old tasks have already finished (health probe completed, monitor is
        # about to return after this method). Just replace in the dict.
        self._tasks.pop(worker_id, None)
        self._start_background_tasks(worker_id)


def _build_worker_cmd(
    port: int,
    engine: str,
    model_path: str,
    engine_config: dict[str, object],
    worker_type: WorkerType = WorkerType.STT,
    *,
    python_package: str | None = None,
    venv_python: str | None = None,
) -> list[str]:
    """Build CLI command to start a worker as a subprocess.

    Args:
        port: gRPC port for the worker to listen on.
        engine: Engine name (e.g., "faster-whisper", "kokoro").
        model_path: Path to model files.
        engine_config: Engine configuration.
        worker_type: Worker type (WorkerType enum).
        python_package: Dotted module path for external engines (optional).
        venv_python: Path to isolated venv Python executable (optional).

    Returns:
        List of arguments for subprocess.Popen.
    """
    import json

    python_exe = venv_python if venv_python is not None else sys.executable
    module = f"macaw.workers.{worker_type.value}"
    cmd = [
        python_exe,
        "-m",
        module,
        "--port",
        str(port),
        "--engine",
        engine,
        "--model-path",
        model_path,
        "--engine-config",
        json.dumps(engine_config),
    ]

    if python_package:
        cmd.extend(["--python-package", python_package])

    return cmd


def _spawn_worker_process(
    port: int,
    engine: str,
    model_path: str,
    engine_config: dict[str, object],
    worker_type: WorkerType = WorkerType.STT,
    *,
    python_package: str | None = None,
    venv_python: str | None = None,
) -> subprocess.Popen[bytes]:
    """Create worker subprocess.

    Args:
        port: gRPC port for the worker to listen on.
        engine: Engine name.
        model_path: Path to model files.
        engine_config: Engine configuration.
        worker_type: Worker type (WorkerType enum).
        python_package: Dotted module path for external engines (optional).
        venv_python: Path to isolated venv Python executable (optional).

    Returns:
        Popen handle for the created process.
    """
    cmd = _build_worker_cmd(
        port,
        engine,
        model_path,
        engine_config,
        worker_type=worker_type,
        python_package=python_package,
        venv_python=venv_python,
    )
    # Flat layout: macaw/workers/manager.py → parents[2] = repo root
    # (macaw/ is directly under the repo root, not under src/)
    repo_root = Path(__file__).resolve().parents[2]
    env = os.environ.copy()
    existing_pythonpath = env.get("PYTHONPATH")
    if existing_pythonpath:
        env["PYTHONPATH"] = f"{repo_root}{os.pathsep}{existing_pythonpath}"
    else:
        env["PYTHONPATH"] = str(repo_root)
    return subprocess.Popen(
        cmd,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.PIPE,
        env=env,
    )


async def _check_stt_health(channel: Any, timeout: float) -> dict[str, str]:
    """Perform STT health check via gRPC."""
    from macaw.proto import HealthRequest
    from macaw.proto.stt_worker_pb2_grpc import STTWorkerStub

    stub = STTWorkerStub(channel)  # type: ignore[no-untyped-call]
    response = await asyncio.wait_for(
        stub.Health(HealthRequest()),
        timeout=timeout,
    )
    return {
        "status": response.status,
        "model_name": response.model_name,
        "engine": response.engine,
    }


async def _check_tts_health(channel: Any, timeout: float) -> dict[str, str]:
    """Perform TTS health check via gRPC."""
    from macaw.proto import TTSHealthRequest, TTSWorkerStub

    stub = TTSWorkerStub(channel)  # type: ignore[no-untyped-call]
    response = await asyncio.wait_for(
        stub.Health(TTSHealthRequest()),
        timeout=timeout,
    )
    return {
        "status": response.status,
        "model_name": response.model_name,
        "engine": response.engine,
    }


_HealthChecker = Callable[[Any, float], Coroutine[Any, Any, dict[str, str]]]

# Dispatch table: WorkerType → health check function(channel, timeout) → dict
_HEALTH_CHECKERS: dict[WorkerType, _HealthChecker] = {
    WorkerType.STT: _check_stt_health,
    WorkerType.TTS: _check_tts_health,
}


async def _check_worker_health(
    port: int,
    timeout: float = 2.0,
    worker_type: WorkerType = WorkerType.STT,
    host: str | None = None,
) -> dict[str, str]:
    """Check worker health via gRPC Health RPC.

    Uses a dispatch table keyed by WorkerType. Adding a new worker type
    only requires adding an entry to ``_HEALTH_CHECKERS``.

    Args:
        port: Worker gRPC port.
        timeout: Timeout in seconds.
        worker_type: Worker type (WorkerType enum).
        host: Override host for remote workers. Uses worker_host setting if None.

    Returns:
        Dict with worker status.
    """
    import grpc.aio

    if host is None:
        from macaw.config.settings import get_settings

        host = get_settings().worker.worker_host
    checker = _HEALTH_CHECKERS[worker_type]
    channel = grpc.aio.insecure_channel(f"{host}:{port}")
    try:
        return await checker(channel, timeout)
    finally:
        await channel.close()
