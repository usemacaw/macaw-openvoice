"""Testes para WorkerManager.

Usa mocks para subprocess.Popen e _check_worker_health.
"""

from __future__ import annotations

import asyncio
from typing import TYPE_CHECKING
from unittest.mock import AsyncMock, MagicMock, patch

from macaw.config.settings import get_settings
from macaw.workers.manager import (
    WorkerManager,
    WorkerState,
)

if TYPE_CHECKING:
    import pytest


def _make_mock_process(poll_return: int | None = None) -> MagicMock:
    """Cria mock de subprocess.Popen."""
    proc = MagicMock()
    proc.poll.return_value = poll_return
    proc.wait.return_value = 0
    proc.terminate.return_value = None
    proc.kill.return_value = None
    proc.stdout = MagicMock()
    proc.stderr = MagicMock()
    return proc


class TestSpawnWorker:
    @patch("macaw.workers.manager.subprocess.Popen")
    async def test_creates_worker_handle(self, mock_popen: MagicMock) -> None:
        mock_popen.return_value = _make_mock_process()
        manager = WorkerManager()

        with patch(
            "macaw.workers.manager._check_worker_health", new_callable=AsyncMock
        ) as mock_health:
            mock_health.return_value = {"status": "ok"}
            handle = await manager.spawn_worker(
                model_name="large-v3",
                port=50051,
                engine="faster-whisper",
                model_path="/models/test",
                engine_config={"model_size": "large-v3", "device": "cpu"},
            )

            assert handle.worker_id == "faster-whisper-50051"
            assert handle.model_name == "large-v3"
            assert handle.port == 50051

            # Cleanup
            await manager.stop_all()

    @patch("macaw.workers.manager.subprocess.Popen")
    async def test_worker_starts_in_starting_state(self, mock_popen: MagicMock) -> None:
        mock_popen.return_value = _make_mock_process()
        manager = WorkerManager()

        with patch(
            "macaw.workers.manager._check_worker_health", new_callable=AsyncMock
        ) as mock_health:
            mock_health.side_effect = Exception("not ready yet")
            handle = await manager.spawn_worker(
                model_name="large-v3",
                port=50052,
                engine="faster-whisper",
                model_path="/models/test",
                engine_config={},
            )
            # Immediately after spawn, state should be STARTING
            assert handle.state == WorkerState.STARTING

            await manager.stop_all()

    @patch("macaw.workers.manager.subprocess.Popen")
    async def test_get_worker_returns_handle(self, mock_popen: MagicMock) -> None:
        mock_popen.return_value = _make_mock_process()
        manager = WorkerManager()

        with patch(
            "macaw.workers.manager._check_worker_health", new_callable=AsyncMock
        ) as mock_health:
            mock_health.return_value = {"status": "ok"}
            await manager.spawn_worker(
                model_name="large-v3",
                port=50053,
                engine="faster-whisper",
                model_path="/models/test",
                engine_config={},
            )

            handle = manager.get_worker("faster-whisper-50053")
            assert handle is not None
            assert handle.port == 50053

            await manager.stop_all()


class TestHealthProbe:
    @patch("macaw.workers.manager.subprocess.Popen")
    async def test_transitions_to_ready(self, mock_popen: MagicMock) -> None:
        mock_popen.return_value = _make_mock_process()
        manager = WorkerManager()

        with patch(
            "macaw.workers.manager._check_worker_health", new_callable=AsyncMock
        ) as mock_health:
            mock_health.return_value = {"status": "ok"}
            handle = await manager.spawn_worker(
                model_name="large-v3",
                port=50054,
                engine="faster-whisper",
                model_path="/models/test",
                engine_config={},
            )

            # Give the health probe time to run
            await asyncio.sleep(1.0)
            assert handle.state == WorkerState.READY

            await manager.stop_all()

    @patch("macaw.workers.manager.subprocess.Popen")
    async def test_stays_starting_when_health_fails(self, mock_popen: MagicMock) -> None:
        mock_popen.return_value = _make_mock_process()
        manager = WorkerManager()

        with patch(
            "macaw.workers.manager._check_worker_health", new_callable=AsyncMock
        ) as mock_health:
            mock_health.side_effect = Exception("Connection refused")
            handle = await manager.spawn_worker(
                model_name="large-v3",
                port=50055,
                engine="faster-whisper",
                model_path="/models/test",
                engine_config={},
            )

            # Give a small window — should still be starting
            await asyncio.sleep(0.3)
            assert handle.state == WorkerState.STARTING

            await manager.stop_all()

    @patch("macaw.workers.manager.subprocess.Popen")
    async def test_crashes_on_timeout(
        self, mock_popen: MagicMock, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        mock_popen.return_value = _make_mock_process()
        monkeypatch.setenv("MACAW_WORKER_HEALTH_PROBE_TIMEOUT_S", "1.0")
        get_settings.cache_clear()
        manager = WorkerManager()

        with patch(
            "macaw.workers.manager._check_worker_health", new_callable=AsyncMock
        ) as mock_health:
            mock_health.side_effect = Exception("Connection refused")
            handle = await manager.spawn_worker(
                model_name="large-v3",
                port=50056,
                engine="faster-whisper",
                model_path="/models/test",
                engine_config={},
            )

            # Wait for health probe timeout
            await asyncio.sleep(2.0)
            assert handle.state == WorkerState.CRASHED

            await manager.stop_all()
        get_settings.cache_clear()


class TestMonitorWorker:
    @patch("macaw.workers.manager.subprocess.Popen")
    async def test_detects_process_crash(self, mock_popen: MagicMock) -> None:
        proc = _make_mock_process(poll_return=None)
        mock_popen.return_value = proc
        manager = WorkerManager()

        with patch(
            "macaw.workers.manager._check_worker_health", new_callable=AsyncMock
        ) as mock_health:
            mock_health.return_value = {"status": "ok"}

            # Fill crash history so restart is blocked (avoids re-spawn complexity)
            handle = await manager.spawn_worker(
                model_name="large-v3",
                port=50057,
                engine="faster-whisper",
                model_path="/models/test",
                engine_config={},
            )

            # Wait for health probe
            await asyncio.sleep(1.0)
            assert handle.state == WorkerState.READY

            # Pre-fill crash timestamps to exceed max
            import time

            handle.crash_timestamps = [
                time.monotonic()
            ] * get_settings().worker_lifecycle.max_crashes_in_window

            # Simulate crash
            proc.poll.return_value = 1

            # Wait for monitor to detect and attempt restart (which will fail due to max crashes)
            await asyncio.sleep(2.0)
            assert handle.state == WorkerState.CRASHED

            await manager.stop_all()

    @patch("macaw.workers.manager.subprocess.Popen")
    async def test_does_not_trigger_on_graceful_stop(self, mock_popen: MagicMock) -> None:
        proc = _make_mock_process(poll_return=None)
        mock_popen.return_value = proc
        manager = WorkerManager()

        with patch(
            "macaw.workers.manager._check_worker_health", new_callable=AsyncMock
        ) as mock_health:
            mock_health.return_value = {"status": "ok"}
            handle = await manager.spawn_worker(
                model_name="large-v3",
                port=50058,
                engine="faster-whisper",
                model_path="/models/test",
                engine_config={},
            )

            await asyncio.sleep(1.0)
            # Graceful stop sets state to STOPPING before process exits
            await manager.stop_worker("faster-whisper-50058")
            assert handle.state == WorkerState.STOPPED


class TestAutoRestart:
    @patch("macaw.workers.manager.subprocess.Popen")
    async def test_restart_increments_crash_count(self, mock_popen: MagicMock) -> None:
        proc = _make_mock_process(poll_return=None)
        mock_popen.return_value = proc
        manager = WorkerManager()

        with patch(
            "macaw.workers.manager._check_worker_health", new_callable=AsyncMock
        ) as mock_health:
            mock_health.return_value = {"status": "ok"}
            handle = await manager.spawn_worker(
                model_name="large-v3",
                port=50059,
                engine="faster-whisper",
                model_path="/models/test",
                engine_config={},
            )

            await asyncio.sleep(1.0)
            assert handle.state == WorkerState.READY

            # Simulate crash
            proc.poll.return_value = 1
            await asyncio.sleep(3.0)

            assert handle.crash_count >= 1

            await manager.stop_all()

    @patch("macaw.workers.manager.subprocess.Popen")
    async def test_exceeding_max_crashes_stays_crashed(self, mock_popen: MagicMock) -> None:
        proc = _make_mock_process(poll_return=None)
        mock_popen.return_value = proc
        manager = WorkerManager()

        with patch(
            "macaw.workers.manager._check_worker_health", new_callable=AsyncMock
        ) as mock_health:
            mock_health.return_value = {"status": "ok"}
            handle = await manager.spawn_worker(
                model_name="large-v3",
                port=50060,
                engine="faster-whisper",
                model_path="/models/test",
                engine_config={},
            )

            await asyncio.sleep(1.0)

            # Pre-fill crash timestamps to simulate max crashes reached
            import time

            handle.crash_timestamps = [
                time.monotonic()
            ] * get_settings().worker_lifecycle.max_crashes_in_window

            # Call _attempt_restart directly
            await manager._attempt_restart("faster-whisper-50060")
            assert handle.state == WorkerState.CRASHED

            await manager.stop_all()


class TestWorkerSummary:
    def test_empty_manager_returns_all_zeros(self) -> None:
        manager = WorkerManager()
        summary = manager.worker_summary()
        assert summary == {"total": 0, "ready": 0, "starting": 0, "crashed": 0}

    @patch("macaw.workers.manager.subprocess.Popen")
    async def test_counts_starting_workers(self, mock_popen: MagicMock) -> None:
        mock_popen.return_value = _make_mock_process()
        manager = WorkerManager()

        with patch(
            "macaw.workers.manager._check_worker_health", new_callable=AsyncMock
        ) as mock_health:
            mock_health.side_effect = Exception("not ready")
            await manager.spawn_worker(
                model_name="test-model",
                port=50070,
                engine="faster-whisper",
                model_path="/models/test",
                engine_config={},
            )

            summary = manager.worker_summary()
            assert summary["total"] == 1
            assert summary["starting"] == 1
            assert summary["ready"] == 0

            await manager.stop_all()

    @patch("macaw.workers.manager.subprocess.Popen")
    async def test_counts_ready_workers(self, mock_popen: MagicMock) -> None:
        mock_popen.return_value = _make_mock_process()
        manager = WorkerManager()

        with patch(
            "macaw.workers.manager._check_worker_health", new_callable=AsyncMock
        ) as mock_health:
            mock_health.return_value = {"status": "ok"}
            await manager.spawn_worker(
                model_name="test-model",
                port=50071,
                engine="faster-whisper",
                model_path="/models/test",
                engine_config={},
            )

            await asyncio.sleep(1.0)
            summary = manager.worker_summary()
            assert summary["total"] == 1
            assert summary["ready"] == 1
            assert summary["starting"] == 0

            await manager.stop_all()


class TestStopWorker:
    @patch("macaw.workers.manager.subprocess.Popen")
    async def test_sets_stopped_state(self, mock_popen: MagicMock) -> None:
        mock_popen.return_value = _make_mock_process()
        manager = WorkerManager()

        with patch(
            "macaw.workers.manager._check_worker_health", new_callable=AsyncMock
        ) as mock_health:
            mock_health.return_value = {"status": "ok"}
            await manager.spawn_worker(
                model_name="large-v3",
                port=50061,
                engine="faster-whisper",
                model_path="/models/test",
                engine_config={},
            )

            await manager.stop_worker("faster-whisper-50061")
            handle = manager.get_worker("faster-whisper-50061")
            assert handle is not None
            assert handle.state == WorkerState.STOPPED

    @patch("macaw.workers.manager.subprocess.Popen")
    async def test_stop_nonexistent_worker_is_noop(self, mock_popen: MagicMock) -> None:
        manager = WorkerManager()
        await manager.stop_worker("nonexistent-99999")

    @patch("macaw.workers.manager.subprocess.Popen")
    async def test_force_kills_on_timeout(
        self, mock_popen: MagicMock, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        proc = _make_mock_process(poll_return=None)
        # Make wait() hang forever to simulate stuck process
        import time as time_mod

        proc.wait.side_effect = lambda: time_mod.sleep(10)
        mock_popen.return_value = proc

        # Use a very short grace period so the test is fast
        monkeypatch.setenv("MACAW_WORKER_STOP_GRACE_PERIOD_S", "0.2")
        get_settings.cache_clear()
        manager = WorkerManager()

        with patch(
            "macaw.workers.manager._check_worker_health", new_callable=AsyncMock
        ) as mock_health:
            mock_health.return_value = {"status": "ok"}
            await manager.spawn_worker(
                model_name="large-v3",
                port=50062,
                engine="faster-whisper",
                model_path="/models/test",
                engine_config={},
            )

            await manager.stop_worker("faster-whisper-50062")

        proc.kill.assert_called_once()
        get_settings.cache_clear()


class TestStopAll:
    @patch("macaw.workers.manager.subprocess.Popen")
    async def test_stops_all_workers(self, mock_popen: MagicMock) -> None:
        mock_popen.return_value = _make_mock_process()
        manager = WorkerManager()

        with patch(
            "macaw.workers.manager._check_worker_health", new_callable=AsyncMock
        ) as mock_health:
            mock_health.return_value = {"status": "ok"}
            await manager.spawn_worker(
                model_name="large-v3",
                port=50063,
                engine="faster-whisper",
                model_path="/models/test1",
                engine_config={},
            )
            await manager.spawn_worker(
                model_name="tiny",
                port=50064,
                engine="faster-whisper",
                model_path="/models/test2",
                engine_config={},
            )

            await manager.stop_all()

            for wid in ["faster-whisper-50063", "faster-whisper-50064"]:
                handle = manager.get_worker(wid)
                assert handle is not None
                assert handle.state == WorkerState.STOPPED
