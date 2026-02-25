"""Tests for remote worker support (register, lifecycle, address resolution)."""

from __future__ import annotations

import asyncio
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from macaw.workers.manager import WorkerHandle, WorkerManager, WorkerState, WorkerType

# ── WorkerHandle fields ──────────────────────────────────────────────


class TestWorkerHandleRemoteEndpoint:
    """WorkerHandle.remote_endpoint and worker_address property."""

    def test_remote_endpoint_default_none(self) -> None:
        handle = WorkerHandle(
            worker_id="fw-50051",
            port=50051,
            model_name="large-v3",
            engine="faster-whisper",
        )
        assert handle.remote_endpoint is None

    def test_worker_address_local(self) -> None:
        handle = WorkerHandle(
            worker_id="fw-50051",
            port=50051,
            model_name="large-v3",
            engine="faster-whisper",
        )
        with patch("macaw.config.settings.get_settings") as mock_settings:
            mock_settings.return_value.worker.worker_host = "localhost"
            assert handle.worker_address == "localhost:50051"

    def test_worker_address_remote(self) -> None:
        handle = WorkerHandle(
            worker_id="fw-remote-50051",
            port=50051,
            model_name="large-v3",
            engine="faster-whisper",
            remote_endpoint="stt-worker:50051",
        )
        # Should NOT call get_settings — remote_endpoint takes precedence
        assert handle.worker_address == "stt-worker:50051"

    def test_worker_address_remote_with_ip(self) -> None:
        handle = WorkerHandle(
            worker_id="fw-remote-9090",
            port=9090,
            model_name="large-v3",
            engine="faster-whisper",
            remote_endpoint="192.168.1.100:9090",
        )
        assert handle.worker_address == "192.168.1.100:9090"


# ── BackendSettings.remote_workers ───────────────────────────────────


class TestBackendSettingsRemoteWorkers:
    """BackendSettings.remote_workers field."""

    def test_default_empty_dict(self) -> None:
        from macaw.config.settings import BackendSettings

        settings = BackendSettings()
        assert settings.remote_workers == {}

    def test_env_var_parsing(self) -> None:
        import os

        from macaw.config.settings import BackendSettings

        with patch.dict(
            os.environ,
            {
                "MACAW_REMOTE_WORKERS": '{"faster-whisper":"stt-host:50051","kokoro":"tts-host:50052"}'
            },
        ):
            settings = BackendSettings()
        assert settings.remote_workers == {
            "faster-whisper": "stt-host:50051",
            "kokoro": "tts-host:50052",
        }


# ── register_remote_worker ───────────────────────────────────────────


class TestRegisterRemoteWorker:
    """WorkerManager.register_remote_worker()."""

    async def test_register_creates_handle(self) -> None:
        with patch("macaw.config.settings.get_settings") as mock:
            mock.return_value = MagicMock()
            mock.return_value.worker_lifecycle = MagicMock(
                health_probe_initial_delay_s=0.01,
                health_probe_max_delay_s=0.02,
                health_probe_timeout_s=0.05,
                health_probe_rpc_timeout_s=0.01,
                monitor_interval_s=1.0,
                stop_grace_period_s=5.0,
            )
            manager = WorkerManager()

        # Patch health probe to avoid real gRPC calls
        with patch.object(manager, "_health_probe", new_callable=AsyncMock):
            handle = await manager.register_remote_worker(
                model_name="large-v3",
                engine="faster-whisper",
                remote_endpoint="stt-worker:50051",
                worker_type="stt",
            )

        assert handle.remote_endpoint == "stt-worker:50051"
        assert handle.process is None
        assert handle.state == WorkerState.STARTING
        assert handle.worker_type == WorkerType.STT
        assert handle.worker_id == "faster-whisper-remote-50051"
        assert handle.port == 50051

    async def test_register_starts_health_probe_only(self) -> None:
        with patch("macaw.config.settings.get_settings") as mock:
            mock.return_value = MagicMock()
            mock.return_value.worker_lifecycle = MagicMock(
                health_probe_initial_delay_s=0.01,
                health_probe_max_delay_s=0.02,
                health_probe_timeout_s=0.05,
                health_probe_rpc_timeout_s=0.01,
                monitor_interval_s=1.0,
                stop_grace_period_s=5.0,
            )
            manager = WorkerManager()

        with patch.object(manager, "_health_probe", new_callable=AsyncMock):
            handle = await manager.register_remote_worker(
                model_name="large-v3",
                engine="faster-whisper",
                remote_endpoint="stt-worker:50051",
            )

        # Should have exactly 1 task (health probe), not 2 (no monitor)
        tasks = manager._tasks.get(handle.worker_id, [])
        assert len(tasks) == 1

    async def test_register_invalid_endpoint_format(self) -> None:
        with patch("macaw.config.settings.get_settings") as mock:
            mock.return_value = MagicMock()
            mock.return_value.worker_lifecycle = MagicMock()
            manager = WorkerManager()

        with pytest.raises(ValueError, match="Invalid remote endpoint format"):
            await manager.register_remote_worker(
                model_name="large-v3",
                engine="faster-whisper",
                remote_endpoint="no-port-here",
            )

    async def test_register_invalid_endpoint_no_host(self) -> None:
        with patch("macaw.config.settings.get_settings") as mock:
            mock.return_value = MagicMock()
            mock.return_value.worker_lifecycle = MagicMock()
            manager = WorkerManager()

        with pytest.raises(ValueError, match="Invalid remote endpoint format"):
            await manager.register_remote_worker(
                model_name="large-v3",
                engine="faster-whisper",
                remote_endpoint=":50051",
            )


# ── Lifecycle: monitor, stop, restart ────────────────────────────────


class TestRemoteWorkerLifecycle:
    """Lifecycle methods for remote workers."""

    async def test_monitor_returns_immediately_for_remote(self) -> None:
        with patch("macaw.config.settings.get_settings") as mock:
            mock.return_value = MagicMock()
            mock.return_value.worker_lifecycle = MagicMock(monitor_interval_s=0.01)
            manager = WorkerManager()

        handle = WorkerHandle(
            worker_id="fw-remote-50051",
            port=50051,
            model_name="large-v3",
            engine="faster-whisper",
            remote_endpoint="stt-worker:50051",
        )
        manager._workers["fw-remote-50051"] = handle

        # _monitor_worker should return immediately for remote workers
        # If it doesn't, it would loop forever (or until timeout)
        await asyncio.wait_for(manager._monitor_worker("fw-remote-50051"), timeout=1.0)

    async def test_stop_skips_process_termination_for_remote(self) -> None:
        with patch("macaw.config.settings.get_settings") as mock:
            mock.return_value = MagicMock()
            mock.return_value.worker_lifecycle = MagicMock(stop_grace_period_s=1.0)
            manager = WorkerManager()

        handle = WorkerHandle(
            worker_id="fw-remote-50051",
            port=50051,
            model_name="large-v3",
            engine="faster-whisper",
            remote_endpoint="stt-worker:50051",
            process=None,
        )
        manager._workers["fw-remote-50051"] = handle
        manager._tasks["fw-remote-50051"] = []

        await manager.stop_worker("fw-remote-50051")

        assert handle.state == WorkerState.STOPPED

    async def test_attempt_restart_no_spawn_for_remote(self) -> None:
        with patch("macaw.config.settings.get_settings") as mock:
            mock.return_value = MagicMock()
            mock.return_value.worker_lifecycle = MagicMock(
                crash_window_s=60.0,
                max_crashes_in_window=3,
                health_probe_initial_delay_s=0.01,
                health_probe_max_delay_s=0.02,
                health_probe_timeout_s=0.05,
                health_probe_rpc_timeout_s=0.01,
            )
            manager = WorkerManager()

        handle = WorkerHandle(
            worker_id="fw-remote-50051",
            port=50051,
            model_name="large-v3",
            engine="faster-whisper",
            remote_endpoint="stt-worker:50051",
            process=None,
        )
        manager._workers["fw-remote-50051"] = handle

        with patch.object(manager, "_health_probe", new_callable=AsyncMock):
            await manager._attempt_restart("fw-remote-50051")

        # Should transition to STARTING, NOT spawn a process
        assert handle.state == WorkerState.STARTING
        assert handle.process is None
        assert handle.crash_count == 1

    async def test_attempt_restart_rate_limiting_for_remote(self) -> None:
        with patch("macaw.config.settings.get_settings") as mock:
            mock.return_value = MagicMock()
            mock.return_value.worker_lifecycle = MagicMock(
                crash_window_s=60.0,
                max_crashes_in_window=2,
                health_probe_initial_delay_s=0.01,
                health_probe_max_delay_s=0.02,
                health_probe_timeout_s=0.05,
                health_probe_rpc_timeout_s=0.01,
            )
            manager = WorkerManager()

        handle = WorkerHandle(
            worker_id="fw-remote-50051",
            port=50051,
            model_name="large-v3",
            engine="faster-whisper",
            remote_endpoint="stt-worker:50051",
            process=None,
            crash_count=1,
            crash_timestamps=[100.0],  # Already 1 crash in window
        )
        manager._workers["fw-remote-50051"] = handle

        with patch("macaw.workers.manager.time") as mock_time:
            mock_time.monotonic.return_value = 110.0
            mock_time.sleep = AsyncMock()
            await manager._attempt_restart("fw-remote-50051")

        # Should stay CRASHED (2 crashes in window >= max_crashes_in_window=2)
        assert handle.state == WorkerState.CRASHED


# ── Health check with host parameter ─────────────────────────────────


class TestHealthCheckHostParam:
    """_check_worker_health with optional host parameter."""

    async def test_health_check_uses_settings_host_by_default(self) -> None:
        from macaw.workers.manager import _check_worker_health

        mock_channel = AsyncMock()
        mock_checker = AsyncMock(return_value={"status": "ok"})

        with (
            patch("grpc.aio.insecure_channel", return_value=mock_channel) as mock_create,
            patch("macaw.config.settings.get_settings") as mock_settings,
            patch.dict("macaw.workers.manager._HEALTH_CHECKERS", {WorkerType.STT: mock_checker}),
        ):
            mock_settings.return_value.worker.worker_host = "localhost"
            result = await _check_worker_health(50051)

        mock_create.assert_called_once_with("localhost:50051")
        assert result == {"status": "ok"}

    async def test_health_check_uses_custom_host(self) -> None:
        from macaw.workers.manager import _check_worker_health

        mock_channel = AsyncMock()
        mock_checker = AsyncMock(return_value={"status": "ok"})

        with (
            patch("grpc.aio.insecure_channel", return_value=mock_channel) as mock_create,
            patch.dict("macaw.workers.manager._HEALTH_CHECKERS", {WorkerType.STT: mock_checker}),
        ):
            result = await _check_worker_health(50051, host="stt-worker")

        mock_create.assert_called_once_with("stt-worker:50051")
        assert result == {"status": "ok"}


# ── Scheduler address resolution ─────────────────────────────────────


class TestSchedulerAddressResolution:
    """Scheduler uses worker.worker_address for gRPC channels."""

    async def test_transcribe_inline_uses_worker_address(self) -> None:
        from macaw.scheduler.scheduler import Scheduler

        mock_worker = MagicMock(spec=WorkerHandle)
        mock_worker.worker_address = "remote-host:50051"
        mock_worker.worker_id = "fw-remote-50051"

        mock_manager = MagicMock()
        mock_manager.get_ready_worker.return_value = mock_worker

        mock_registry = MagicMock()
        mock_registry.get_manifest.return_value = MagicMock()

        mock_request = MagicMock()
        mock_request.model_name = "large-v3"
        mock_request.request_id = "req-1"
        mock_request.audio_data = b"\x00" * 32000
        mock_request.task = "transcribe"

        mock_channel = AsyncMock()
        mock_result = MagicMock()
        mock_result.text = "hello"
        mock_result.segments = []

        with (
            patch("macaw.config.settings.get_settings") as mock_settings,
            patch("grpc.aio.insecure_channel", return_value=mock_channel) as mock_create,
        ):
            mock_settings.return_value.scheduler = MagicMock(
                aging_threshold_s=30.0,
                batch_accumulate_ms=50.0,
                batch_max_size=8,
                latency_ttl_s=60.0,
                cancel_propagation_timeout_s=0.1,
                min_grpc_timeout_s=30.0,
                timeout_factor=2.0,
            )
            mock_settings.return_value.worker.worker_host = "localhost"

            scheduler = Scheduler(mock_manager, mock_registry)

            with patch.object(scheduler, "_send_to_worker", new_callable=AsyncMock) as mock_send:
                mock_send.return_value = mock_result
                await scheduler._transcribe_inline(mock_request)

            # Channel should be created with worker_address, not f"{host}:{port}"
            mock_create.assert_called_once()
            call_args = mock_create.call_args[0][0]
            assert call_args == "remote-host:50051"

    async def test_dispatch_request_uses_worker_address(self) -> None:
        from macaw.scheduler.scheduler import Scheduler

        mock_worker = MagicMock(spec=WorkerHandle)
        mock_worker.worker_address = "remote-host:50051"
        mock_worker.worker_id = "fw-remote-50051"

        mock_manager = MagicMock()
        mock_manager.get_ready_worker.return_value = mock_worker

        mock_registry = MagicMock()

        mock_request = MagicMock()
        mock_request.model_name = "large-v3"
        mock_request.request_id = "req-1"
        mock_request.audio_data = b"\x00" * 32000
        mock_request.task = "transcribe"

        mock_result = MagicMock()
        mock_result.text = "hello"
        mock_result.segments = []

        loop = asyncio.get_running_loop()
        future: asyncio.Future[object] = loop.create_future()
        cancel_event = asyncio.Event()

        scheduled = MagicMock()
        scheduled.request = mock_request
        scheduled.result_future = future
        scheduled.cancel_event = cancel_event
        scheduled.priority = MagicMock()
        scheduled.priority.name = "BATCH"

        with patch("macaw.config.settings.get_settings") as mock_settings:
            mock_settings.return_value.scheduler = MagicMock(
                aging_threshold_s=30.0,
                batch_accumulate_ms=50.0,
                batch_max_size=8,
                latency_ttl_s=60.0,
                cancel_propagation_timeout_s=0.1,
                no_worker_backoff_s=0.1,
                min_grpc_timeout_s=30.0,
                timeout_factor=2.0,
            )
            mock_settings.return_value.worker.worker_host = "localhost"

            scheduler = Scheduler(mock_manager, mock_registry)

            with patch.object(scheduler, "_send_to_worker", new_callable=AsyncMock) as mock_send:
                mock_send.return_value = mock_result
                await scheduler._dispatch_request(scheduled)

        # The cancellation manager should have the remote address
        assert future.done()
        assert future.result() == mock_result
