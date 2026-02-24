"""Tests verifying that worker modules don't import server-only packages.

The core deps (9 packages) should be sufficient for workers. Server-only
packages (fastapi, uvicorn, click, httpx, huggingface_hub, python-multipart,
defusedxml) should NOT be imported by worker code paths.
"""

from __future__ import annotations

import sys
from typing import Any
from unittest.mock import patch

_SERVER_PACKAGES = frozenset(
    {
        "fastapi",
        "uvicorn",
        "click",
        "httpx",
        "huggingface_hub",
        "multipart",
        "defusedxml",
    }
)


class TestCoreDepsOnly:
    """Workers should not import server-only packages."""

    def test_worker_manager_no_server_imports(self) -> None:
        """macaw.workers.manager should not import server packages."""
        # The module is already imported, check its loaded modules
        # We check the module's source file for direct imports (not sys.modules,
        # which includes test infrastructure imports)
        import importlib

        import macaw.workers.manager  # noqa: F401

        mod = importlib.import_module("macaw.workers.manager")
        source = mod.__file__ or ""
        assert "fastapi" not in source
        # If any server package is in the module source, that's a problem
        with open(source) as f:
            content = f.read()
        for pkg in _SERVER_PACKAGES:
            # Allow TYPE_CHECKING imports and string references
            lines = [
                line
                for line in content.split("\n")
                if f"import {pkg}" in line
                and "TYPE_CHECKING" not in line
                and line.strip().startswith(("import", "from"))
            ]
            assert not lines, f"Server package {pkg!r} imported in manager.py: {lines}"

    def test_worker_handle_core_deps_only(self) -> None:
        """WorkerHandle dataclass uses only core deps (pydantic not required)."""
        from macaw.workers.manager import WorkerHandle

        handle = WorkerHandle(
            worker_id="test-1",
            port=50051,
            model_name="test",
            engine="test-engine",
        )
        assert handle.worker_id == "test-1"
        assert handle.remote_endpoint is None

    def test_pyproject_core_deps_count(self) -> None:
        """Core dependencies should be exactly 9 packages."""
        import tomllib
        from pathlib import Path

        pyproject = Path(__file__).parents[2] / "pyproject.toml"
        with open(pyproject, "rb") as f:
            data = tomllib.load(f)

        core_deps = data["project"]["dependencies"]
        assert len(core_deps) == 9, f"Expected 9 core deps, got {len(core_deps)}: {core_deps}"

    def test_server_extra_exists(self) -> None:
        """[server] optional extra should exist with 7 packages."""
        import tomllib
        from pathlib import Path

        pyproject = Path(__file__).parents[2] / "pyproject.toml"
        with open(pyproject, "rb") as f:
            data = tomllib.load(f)

        extras = data["project"]["optional-dependencies"]
        assert "server" in extras, "Missing [server] extra in pyproject.toml"
        server_deps = extras["server"]
        assert len(server_deps) == 7, (
            f"Expected 7 server deps, got {len(server_deps)}: {server_deps}"
        )

    def test_all_extra_includes_server(self) -> None:
        """[all] extra should include server."""
        import tomllib
        from pathlib import Path

        pyproject = Path(__file__).parents[2] / "pyproject.toml"
        with open(pyproject, "rb") as f:
            data = tomllib.load(f)

        all_deps = data["project"]["optional-dependencies"]["all"]
        assert any("server" in dep for dep in all_deps), "[all] extra does not include server"

    def test_opuslib_not_in_core(self) -> None:
        """opuslib should be in [codec] extra only, not in core deps."""
        import tomllib
        from pathlib import Path

        pyproject = Path(__file__).parents[2] / "pyproject.toml"
        with open(pyproject, "rb") as f:
            data = tomllib.load(f)

        core_deps_str = str(data["project"]["dependencies"])
        assert "opuslib" not in core_deps_str, "opuslib should not be in core dependencies"

        codec_deps = data["project"]["optional-dependencies"].get("codec", [])
        assert any("opuslib" in d for d in codec_deps), "opuslib should be in [codec] extra"

    def test_nemo_not_in_core(self) -> None:
        """nemo_text_processing should be in [itn] extra only, not in core deps."""
        import tomllib
        from pathlib import Path

        pyproject = Path(__file__).parents[2] / "pyproject.toml"
        with open(pyproject, "rb") as f:
            data = tomllib.load(f)

        core_deps_str = str(data["project"]["dependencies"])
        assert "nemo" not in core_deps_str, (
            "nemo_text_processing should not be in core dependencies"
        )

        itn_deps = data["project"]["optional-dependencies"].get("itn", [])
        assert any("nemo" in d for d in itn_deps), "nemo_text_processing should be in [itn] extra"


class TestSpawnAllWorkersRemoteDispatch:
    """_spawn_all_workers dispatches to remote vs local based on config."""

    async def test_remote_worker_registered_for_configured_engine(self) -> None:
        from macaw.cli.serve import _spawn_all_workers

        mock_manifest = _make_manifest("stt", "faster-whisper", "large-v3")
        mock_registry = _make_registry()

        mock_manager = _make_manager()

        with patch("macaw.cli.serve.get_settings") as mock_settings:
            mock_settings.return_value.backend.remote_workers = {
                "faster-whisper": "stt-host:50051",
            }
            _stt_models, _tts_models = await _spawn_all_workers(
                mock_registry, mock_manager, [mock_manifest], 50051
            )

        mock_manager.register_remote_worker.assert_called_once()
        call_kwargs = mock_manager.register_remote_worker.call_args
        assert call_kwargs.kwargs["remote_endpoint"] == "stt-host:50051"

    async def test_local_spawn_when_no_remote_configured(self) -> None:
        from macaw.cli.serve import _spawn_all_workers

        mock_manifest = _make_manifest("stt", "faster-whisper", "large-v3")
        mock_registry = _make_registry()
        mock_manager = _make_manager()

        with (
            patch("macaw.cli.serve.get_settings") as mock_settings,
            patch("macaw.cli.serve.is_engine_available", return_value=True),
            patch(
                "macaw.backends.venv_manager.resolve_python_for_engine",
                return_value=sys.executable,
            ),
        ):
            mock_settings.return_value.backend.remote_workers = {}
            _stt_models, _tts_models = await _spawn_all_workers(
                mock_registry, mock_manager, [mock_manifest], 50051
            )

        mock_manager.spawn_worker.assert_called_once()
        mock_manager.register_remote_worker.assert_not_called()

    async def test_mixed_local_and_remote(self) -> None:
        from macaw.cli.serve import _spawn_all_workers

        stt_manifest = _make_manifest("stt", "faster-whisper", "large-v3")
        tts_manifest = _make_manifest("tts", "kokoro", "kokoro-v1")
        mock_registry = _make_registry()
        mock_manager = _make_manager()

        with (
            patch("macaw.cli.serve.get_settings") as mock_settings,
            patch("macaw.cli.serve.is_engine_available", return_value=True),
            patch(
                "macaw.backends.venv_manager.resolve_python_for_engine",
                return_value=sys.executable,
            ),
        ):
            mock_settings.return_value.backend.remote_workers = {
                "faster-whisper": "stt-host:50051",
                # kokoro NOT in remote_workers → local spawn
            }
            _stt_models, _tts_models = await _spawn_all_workers(
                mock_registry, mock_manager, [stt_manifest, tts_manifest], 50051
            )

        # STT: remote, TTS: local
        mock_manager.register_remote_worker.assert_called_once()
        mock_manager.spawn_worker.assert_called_once()

    async def test_tts_remote_worker(self) -> None:
        from macaw.cli.serve import _spawn_all_workers

        tts_manifest = _make_manifest("tts", "kokoro", "kokoro-v1")
        mock_registry = _make_registry()
        mock_manager = _make_manager()

        with patch("macaw.cli.serve.get_settings") as mock_settings:
            mock_settings.return_value.backend.remote_workers = {
                "kokoro": "tts-host:50052",
            }
            _stt_models, _tts_models = await _spawn_all_workers(
                mock_registry, mock_manager, [tts_manifest], 50051
            )

        mock_manager.register_remote_worker.assert_called_once()
        call_args = mock_manager.register_remote_worker.call_args
        assert "tts" in str(call_args)


# ── Helpers ──────────────────────────────────────────────────────────


def _make_manifest(model_type_str: str, engine: str, name: str) -> Any:
    from unittest.mock import MagicMock

    from macaw._types import ModelType

    manifest = MagicMock()
    manifest.model_type = ModelType(model_type_str)
    manifest.engine = engine
    manifest.name = name
    manifest.python_package = None
    manifest.engine_config = MagicMock()
    manifest.engine_config.model_dump.return_value = {}
    return manifest


def _make_registry() -> Any:
    from unittest.mock import MagicMock

    registry = MagicMock()
    registry.get_model_path.return_value = "/models/test"
    return registry


def _make_manager() -> Any:
    from unittest.mock import AsyncMock, MagicMock

    manager = MagicMock()
    manager.register_remote_worker = AsyncMock()
    manager.spawn_worker = AsyncMock()
    return manager
