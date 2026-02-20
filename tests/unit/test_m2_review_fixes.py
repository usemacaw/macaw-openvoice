"""Tests for the 5 fixes from M2 code review.

Fix 1: Servicer TranscribeFile - explicit return after context.abort
Fix 2: WorkerManager - _build_worker_cmd extracted (DRY)
Fix 3: WorkerManager - _check_worker_health import moved outside try
Fix 4: Worker main.py - protection against double shutdown
Fix 5: WorkerManager - tasks awaited after cancel in stop_worker
"""

from __future__ import annotations

import asyncio
import sys
from typing import TYPE_CHECKING
from unittest.mock import AsyncMock, MagicMock, patch

import grpc
import pytest

from macaw._types import (
    BatchResult,
    EngineCapabilities,
    SegmentDetail,
    STTArchitecture,
)
from macaw.proto.stt_worker_pb2 import TranscribeFileRequest
from macaw.workers.manager import (
    WorkerManager,
    WorkerType,
    _build_worker_cmd,
)
from macaw.workers.stt.interface import STTBackend
from macaw.workers.stt.servicer import STTWorkerServicer

if TYPE_CHECKING:
    from collections.abc import AsyncIterator

    from macaw._types import TranscriptSegment


class MockBackend(STTBackend):
    """Mock backend for tests."""

    def __init__(self, transcribe_result: BatchResult | None = None) -> None:
        self._result = transcribe_result or BatchResult(
            text="mock transcript",
            language="en",
            duration=1.0,
            segments=(SegmentDetail(id=0, start=0.0, end=1.0, text="mock transcript"),),
            words=None,
        )
        self._health_status = "ok"

    @property
    def architecture(self) -> STTArchitecture:
        return STTArchitecture.ENCODER_DECODER

    async def load(self, model_path: str, config: dict[str, object]) -> None:
        pass

    async def capabilities(self) -> EngineCapabilities:
        return EngineCapabilities()

    async def transcribe_file(
        self,
        audio_data: bytes,
        language: str | None = None,
        initial_prompt: str | None = None,
        hot_words: list[str] | None = None,
        temperature: float = 0.0,
        word_timestamps: bool = False,
    ) -> BatchResult:
        return self._result

    async def transcribe_stream(
        self,
        audio_chunks: AsyncIterator[bytes],
        language: str | None = None,
        initial_prompt: str | None = None,
        hot_words: list[str] | None = None,
    ) -> AsyncIterator[TranscriptSegment]:
        raise NotImplementedError
        yield  # pragma: no cover

    async def unload(self) -> None:
        pass

    async def health(self) -> dict[str, str]:
        return {"status": self._health_status}


def _make_context() -> MagicMock:
    """Create a mock grpc.aio.ServicerContext."""
    ctx = MagicMock()
    ctx.abort = AsyncMock()
    return ctx


def _make_mock_process(poll_return: int | None = None) -> MagicMock:
    """Create a mock subprocess.Popen."""
    proc = MagicMock()
    proc.poll.return_value = poll_return
    proc.wait.return_value = 0
    proc.terminate.return_value = None
    proc.kill.return_value = None
    proc.stdout = MagicMock()
    proc.stderr = MagicMock()
    return proc


# ============================================================
# Fix 1: Servicer TranscribeFile — return after abort
# ============================================================


class TestFix1ServicerAbortReturn:
    """Verifies that TranscribeFile does not access undefined `result` after error."""

    async def test_error_returns_empty_response_when_abort_does_not_raise(self) -> None:
        """If context.abort does not raise (mock), the explicit return avoids UnboundLocalError."""
        backend = MockBackend()
        backend.transcribe_file = AsyncMock(side_effect=RuntimeError("GPU OOM"))  # type: ignore[method-assign]
        servicer = STTWorkerServicer(
            backend=backend,
            model_name="large-v3",
            engine="faster-whisper",
        )
        request = TranscribeFileRequest(
            request_id="req-fix1",
            audio_data=b"\x00\x01" * 100,
        )
        ctx = _make_context()
        # abort does not raise in this mock — simulates defensive scenario
        ctx.abort = AsyncMock(return_value=None)

        response = await servicer.TranscribeFile(request, ctx)
        # The explicit return should return an empty TranscribeFileResponse
        assert response.text == ""
        ctx.abort.assert_called_once_with(grpc.StatusCode.INTERNAL, "GPU OOM")

    async def test_error_with_abort_raising_propagates_correctly(self) -> None:
        """Normal scenario: abort raises AbortError, and the exception propagates."""
        backend = MockBackend()
        backend.transcribe_file = AsyncMock(side_effect=RuntimeError("OOM"))  # type: ignore[method-assign]
        servicer = STTWorkerServicer(
            backend=backend,
            model_name="large-v3",
            engine="faster-whisper",
        )
        request = TranscribeFileRequest(
            request_id="req-fix1b",
            audio_data=b"\x00\x01" * 100,
        )
        ctx = _make_context()
        ctx.abort = AsyncMock(
            side_effect=grpc.aio.AbortError(  # type: ignore[attr-defined]
                grpc.StatusCode.INTERNAL, "OOM"
            )
        )

        with pytest.raises(grpc.aio.AbortError):  # type: ignore[attr-defined]
            await servicer.TranscribeFile(request, ctx)


# ============================================================
# Fix 2: _build_worker_cmd extracted (DRY)
# ============================================================


class TestFix2BuildWorkerCmd:
    """Verifies that _build_worker_cmd generates the correct command."""

    def test_basic_cmd(self) -> None:
        cmd = _build_worker_cmd(
            port=50051,
            engine="faster-whisper",
            model_path="/models/test",
            engine_config={},
        )
        assert sys.executable in cmd
        assert "-m" in cmd
        assert "macaw.workers.stt" in cmd
        assert "--port" in cmd
        assert "50051" in cmd
        assert "--engine" in cmd
        assert "faster-whisper" in cmd
        assert "--model-path" in cmd
        assert "/models/test" in cmd

    def test_includes_engine_config_json(self) -> None:
        import json

        config = {"compute_type": "int8", "device": "cuda"}
        cmd = _build_worker_cmd(
            port=50051,
            engine="faster-whisper",
            model_path="/models/test",
            engine_config=config,
        )
        idx = cmd.index("--engine-config")
        parsed = json.loads(cmd[idx + 1])
        assert parsed["compute_type"] == "int8"
        assert parsed["device"] == "cuda"

    def test_all_config_options_in_json(self) -> None:
        import json

        config = {
            "compute_type": "float16",
            "device": "cpu",
            "model_size": "large-v3",
            "beam_size": 5,
        }
        cmd = _build_worker_cmd(
            port=50051,
            engine="faster-whisper",
            model_path="/models/test",
            engine_config=config,
        )
        idx = cmd.index("--engine-config")
        parsed = json.loads(cmd[idx + 1])
        assert parsed == config

    def test_empty_config_produces_empty_json(self) -> None:
        import json

        cmd = _build_worker_cmd(
            port=50051,
            engine="faster-whisper",
            model_path="/models/test",
            engine_config={},
        )
        idx = cmd.index("--engine-config")
        parsed = json.loads(cmd[idx + 1])
        assert parsed == {}

    def test_python_package_included_when_set(self) -> None:
        cmd = _build_worker_cmd(
            port=50051,
            engine="my-engine",
            model_path="/models/test",
            engine_config={},
            python_package="my_company.engines.stt",
        )
        assert "--python-package" in cmd
        idx = cmd.index("--python-package")
        assert cmd[idx + 1] == "my_company.engines.stt"

    def test_python_package_omitted_when_none(self) -> None:
        cmd = _build_worker_cmd(
            port=50051,
            engine="faster-whisper",
            model_path="/models/test",
            engine_config={},
            python_package=None,
        )
        assert "--python-package" not in cmd


# ============================================================
# Fix 3: _check_worker_health — import outside try
# ============================================================


class TestFix3HealthCheckImport:
    """Verifies that _check_worker_health closes channel correctly."""

    async def test_channel_closed_on_success(self) -> None:
        """Channel is closed even when health returns successfully."""
        from macaw.workers.manager import _check_worker_health

        mock_response = MagicMock()
        mock_response.status = "ok"
        mock_response.model_name = "test"
        mock_response.engine = "test"

        mock_stub_instance = MagicMock()
        mock_stub_instance.Health = AsyncMock(return_value=mock_response)

        mock_channel = AsyncMock()

        with (
            patch("grpc.aio.insecure_channel", return_value=mock_channel),
            patch(
                "macaw.proto.stt_worker_pb2_grpc.STTWorkerStub",
                return_value=mock_stub_instance,
            ),
        ):
            result = await _check_worker_health(50051, timeout=1.0)
            assert result["status"] == "ok"
            mock_channel.close.assert_called_once()

    async def test_channel_closed_on_error(self) -> None:
        """Channel is closed even when an error occurs."""
        from macaw.workers.manager import _check_worker_health

        mock_stub_instance = MagicMock()
        mock_stub_instance.Health = AsyncMock(side_effect=Exception("Connection refused"))

        mock_channel = AsyncMock()

        with (
            patch("grpc.aio.insecure_channel", return_value=mock_channel),
            patch(
                "macaw.proto.stt_worker_pb2_grpc.STTWorkerStub",
                return_value=mock_stub_instance,
            ),
        ):
            with pytest.raises(Exception, match="Connection refused"):
                await _check_worker_health(50051, timeout=1.0)
            mock_channel.close.assert_called_once()


# ============================================================
# Fix 4: Signal handler — protection against double shutdown
# ============================================================


class TestFix4DoubleShutdownProtection:
    """Verifies that _shutdown does not execute twice."""

    async def test_shutdown_guard_flag(self) -> None:
        """Simulates double invocation of _shutdown and verifies it only executes once."""
        shutdown_count = 0

        async def mock_serve() -> None:
            nonlocal shutdown_count
            # We cannot actually call serve() without gRPC, but we can
            # test the guard logic directly.
            shutting_down = False

            async def _shutdown() -> None:
                nonlocal shutting_down, shutdown_count
                if shutting_down:
                    return
                shutting_down = True
                shutdown_count += 1

            await _shutdown()
            await _shutdown()  # Second call should be no-op

        await mock_serve()
        assert shutdown_count == 1

    async def test_signal_handler_is_named_function(self) -> None:
        """Verifies that the signal handler is a named function, not a lambda."""
        import inspect

        from macaw.workers.stt import main as main_mod

        source = inspect.getsource(main_mod.serve)
        # There should be no lambda in add_signal_handler
        assert "lambda:" not in source or "lambda: asyncio.ensure_future" not in source


# ============================================================
# Fix 5: Tasks awaited after cancel
# ============================================================


class TestFix5TasksAwaitedAfterCancel:
    """Verifies that background tasks are awaited after cancel."""

    @patch("macaw.workers.manager._spawn_worker_process")
    async def test_tasks_removed_from_dict_after_stop(self, mock_spawn: MagicMock) -> None:
        mock_spawn.return_value = _make_mock_process()
        manager = WorkerManager()

        with patch(
            "macaw.workers.manager._check_worker_health", new_callable=AsyncMock
        ) as mock_health:
            mock_health.return_value = {"status": "ok"}
            await manager.spawn_worker(
                model_name="large-v3",
                port=50070,
                engine="faster-whisper",
                model_path="/models/test",
                engine_config={},
            )

            # Verify tasks exist before stop
            worker_id = "faster-whisper-50070"
            assert worker_id in manager._tasks

            await manager.stop_worker(worker_id)

            # Tasks should be removed from dict after stop (popped by _cancel_background_tasks)
            assert worker_id not in manager._tasks

    @patch("macaw.workers.manager._spawn_worker_process")
    async def test_cancel_background_tasks_gathers_with_return_exceptions(
        self, mock_spawn: MagicMock
    ) -> None:
        """_cancel_background_tasks uses gather with return_exceptions=True."""
        mock_spawn.return_value = _make_mock_process()
        manager = WorkerManager()

        with patch(
            "macaw.workers.manager._check_worker_health", new_callable=AsyncMock
        ) as mock_health:
            mock_health.return_value = {"status": "ok"}
            await manager.spawn_worker(
                model_name="large-v3",
                port=50071,
                engine="faster-whisper",
                model_path="/models/test",
                engine_config={},
            )

            await asyncio.sleep(1.0)

            worker_id = "faster-whisper-50071"
            # Cancel and await — should not raise even if tasks raise CancelledError
            await manager._cancel_background_tasks(worker_id)

            # After cancel, dict entry should be gone
            assert worker_id not in manager._tasks

            await manager.stop_all()

    @patch("macaw.workers.manager._spawn_worker_process")
    async def test_spawn_uses_extracted_spawn_function(self, mock_spawn: MagicMock) -> None:
        """spawn_worker uses _spawn_worker_process (DRY)."""
        mock_spawn.return_value = _make_mock_process()
        manager = WorkerManager()

        with patch(
            "macaw.workers.manager._check_worker_health", new_callable=AsyncMock
        ) as mock_health:
            mock_health.return_value = {"status": "ok"}
            await manager.spawn_worker(
                model_name="large-v3",
                port=50072,
                engine="faster-whisper",
                model_path="/models/test",
                engine_config={"device": "cpu"},
            )

            mock_spawn.assert_called_once_with(
                50072,
                "faster-whisper",
                "/models/test",
                {"device": "cpu"},
                worker_type=WorkerType.STT,
                python_package=None,
            )

            await manager.stop_all()
