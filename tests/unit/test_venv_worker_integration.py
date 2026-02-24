"""Tests for venv_python integration in worker manager and engines."""

from __future__ import annotations

import sys
from types import ModuleType
from unittest.mock import MagicMock, patch

from macaw.engines import is_engine_available
from macaw.workers.manager import (
    WorkerHandle,
    _build_worker_cmd,
)


class TestWorkerHandleVenvPython:
    def test_default_is_none(self) -> None:
        handle = WorkerHandle(
            worker_id="test-1",
            port=50051,
            model_name="test",
            engine="faster-whisper",
        )
        assert handle.venv_python is None


class TestBuildWorkerCmdVenvPython:
    def test_default_uses_sys_executable(self) -> None:
        cmd = _build_worker_cmd(
            port=50051,
            engine="faster-whisper",
            model_path="/models/test",
            engine_config={},
        )
        assert cmd[0] == sys.executable

    def test_venv_python_overrides_sys_executable(self) -> None:
        cmd = _build_worker_cmd(
            port=50051,
            engine="faster-whisper",
            model_path="/models/test",
            engine_config={},
            venv_python="/venvs/faster-whisper/bin/python",
        )
        assert cmd[0] == "/venvs/faster-whisper/bin/python"
        assert sys.executable not in cmd

    def test_venv_python_with_python_package(self) -> None:
        cmd = _build_worker_cmd(
            port=50051,
            engine="my-engine",
            model_path="/models/test",
            engine_config={},
            venv_python="/venvs/my-engine/bin/python",
            python_package="my_company.stt",
        )
        assert cmd[0] == "/venvs/my-engine/bin/python"
        assert "--python-package" in cmd
        assert "my_company.stt" in cmd


class TestIsEngineAvailableVenvPython:
    def test_existing_behavior_unchanged_when_venv_python_none(self) -> None:
        """When venv_python is None, existing behavior is preserved."""
        fake_spec = ModuleType("fake_spec")
        with patch("macaw.engines.importlib.util.find_spec", return_value=fake_spec):
            assert is_engine_available("faster-whisper") is True

    @patch("macaw.engines.subprocess.run")
    def test_venv_python_checks_via_subprocess(self, mock_run: MagicMock) -> None:
        mock_run.return_value = MagicMock(returncode=0)
        result = is_engine_available(
            "faster-whisper",
            venv_python="/venvs/fw/bin/python",
        )
        assert result is True
        mock_run.assert_called_once()
        cmd = mock_run.call_args[0][0]
        assert cmd[0] == "/venvs/fw/bin/python"
        assert "import faster_whisper" in cmd[2]

    @patch("macaw.engines.subprocess.run")
    def test_venv_python_rejects_unsafe_package(self, mock_run: MagicMock) -> None:
        """Package name with shell metacharacters must be rejected."""
        result = is_engine_available(
            "kokoro",
            python_package="os; rm -rf /",
            venv_python="/venvs/kokoro/bin/python",
        )
        assert result is False
        mock_run.assert_not_called()

    @patch("macaw.engines.subprocess.run")
    def test_venv_python_returns_false_on_failure(self, mock_run: MagicMock) -> None:
        mock_run.return_value = MagicMock(returncode=1)
        result = is_engine_available(
            "kokoro",
            venv_python="/venvs/kokoro/bin/python",
        )
        assert result is False

    @patch("macaw.engines.subprocess.run")
    def test_venv_python_returns_false_on_timeout(self, mock_run: MagicMock) -> None:
        import subprocess

        mock_run.side_effect = subprocess.TimeoutExpired(cmd="test", timeout=10)
        result = is_engine_available(
            "kokoro",
            venv_python="/venvs/kokoro/bin/python",
        )
        assert result is False

    @patch("macaw.engines.subprocess.run")
    def test_venv_python_unknown_engine_returns_true(self, mock_run: MagicMock) -> None:
        """Unknown engine with venv_python returns True (no package to check)."""
        result = is_engine_available(
            "future-engine",
            venv_python="/venvs/future/bin/python",
        )
        assert result is True
        mock_run.assert_not_called()


class TestSpawnWorkerVenvPython:
    @patch("macaw.workers.manager.subprocess.Popen")
    async def test_spawn_stores_venv_python_on_handle(self, mock_popen: MagicMock) -> None:
        from unittest.mock import AsyncMock

        from macaw.workers.manager import WorkerManager

        mock_popen.return_value = MagicMock(
            poll=MagicMock(return_value=None),
            wait=MagicMock(return_value=0),
            terminate=MagicMock(),
            kill=MagicMock(),
            stderr=MagicMock(),
        )

        manager = WorkerManager()
        with patch(
            "macaw.workers.manager._check_worker_health", new_callable=AsyncMock
        ) as mock_health:
            mock_health.return_value = {"status": "ok"}
            handle = await manager.spawn_worker(
                model_name="test",
                port=50090,
                engine="faster-whisper",
                model_path="/models/test",
                engine_config={},
                venv_python="/venvs/fw/bin/python",
            )

            assert handle.venv_python == "/venvs/fw/bin/python"
            await manager.stop_all()

    @patch("macaw.workers.manager.subprocess.Popen")
    async def test_spawn_passes_venv_to_popen(self, mock_popen: MagicMock) -> None:
        from unittest.mock import AsyncMock

        from macaw.workers.manager import WorkerManager

        mock_popen.return_value = MagicMock(
            poll=MagicMock(return_value=None),
            wait=MagicMock(return_value=0),
            terminate=MagicMock(),
            kill=MagicMock(),
            stderr=MagicMock(),
        )

        manager = WorkerManager()
        with patch(
            "macaw.workers.manager._check_worker_health", new_callable=AsyncMock
        ) as mock_health:
            mock_health.return_value = {"status": "ok"}
            await manager.spawn_worker(
                model_name="test",
                port=50091,
                engine="faster-whisper",
                model_path="/models/test",
                engine_config={},
                venv_python="/venvs/fw/bin/python",
            )

            # Check Popen was called with the venv python
            popen_cmd = mock_popen.call_args[0][0]
            assert popen_cmd[0] == "/venvs/fw/bin/python"
            await manager.stop_all()

    @patch("macaw.workers.manager.subprocess.Popen")
    async def test_restart_preserves_venv_python(self, mock_popen: MagicMock) -> None:
        """_attempt_restart passes handle.venv_python to the new process."""
        from unittest.mock import AsyncMock

        from macaw.workers.manager import WorkerManager

        mock_popen.return_value = MagicMock(
            poll=MagicMock(return_value=None),
            wait=MagicMock(return_value=0),
            terminate=MagicMock(),
            kill=MagicMock(),
            stderr=MagicMock(),
        )

        manager = WorkerManager()
        with patch(
            "macaw.workers.manager._check_worker_health", new_callable=AsyncMock
        ) as mock_health:
            mock_health.return_value = {"status": "ok"}
            handle = await manager.spawn_worker(
                model_name="test",
                port=50092,
                engine="faster-whisper",
                model_path="/models/test",
                engine_config={},
                venv_python="/venvs/fw/bin/python",
            )

            # Reset Popen mock to track the restart call
            mock_popen.reset_mock()
            mock_popen.return_value = MagicMock(
                poll=MagicMock(return_value=None),
                wait=MagicMock(return_value=0),
                terminate=MagicMock(),
                kill=MagicMock(),
                stderr=MagicMock(),
            )

            # Trigger restart
            handle.crash_timestamps = []  # Clear so restart is allowed
            await manager._attempt_restart("faster-whisper-50092")

            # Verify Popen was called with the venv python
            assert mock_popen.called
            restart_cmd = mock_popen.call_args[0][0]
            assert restart_cmd[0] == "/venvs/fw/bin/python"

            await manager.stop_all()
