"""VenvManager — per-engine venv provisioning and resolution.

Each ML engine (faster-whisper, kokoro, qwen3-tts, etc.) gets its own
Python venv under ``~/.cache/macaw/venvs/{engine}/``.  This prevents
dependency conflicts between engines sharing the same process.

Uses ``uv`` for venv creation (hardlinks avoid torch duplication).
"""

from __future__ import annotations

import asyncio
import json
import re
import shutil
import subprocess
import sys
from datetime import UTC, datetime
from pathlib import Path

from macaw.engines import _SAFE_MODULE_RE, ENGINE_EXTRAS
from macaw.exceptions import VenvProvisionError
from macaw.logging import get_logger

logger = get_logger("backends.venv_manager")

_MARKER_FILENAME = ".macaw-engine"

# Subprocess timeouts (seconds).
_VENV_CREATION_TIMEOUT_S = 120
_DEP_INSTALL_TIMEOUT_S = 600

# Engine names must be alphanumeric with hyphens, dots, or underscores.
# Guards against path traversal (e.g., "../../etc") and shell injection.
_SAFE_ENGINE_RE = re.compile(r"^[a-zA-Z0-9]([a-zA-Z0-9._-]*[a-zA-Z0-9])?$")

# Extra names follow the same constraints as engine names — they are
# interpolated into ``uv pip install macaw-openvoice[{extra}]``.
_SAFE_EXTRA_RE = re.compile(r"^[a-zA-Z0-9]([a-zA-Z0-9._-]*[a-zA-Z0-9])?$")


def _validate_engine_name(engine: str) -> None:
    """Raise ``ValueError`` if *engine* contains unsafe characters.

    Engine names are used to construct filesystem paths and subprocess
    arguments.  This validation prevents path traversal and injection.
    """
    if not _SAFE_ENGINE_RE.match(engine):
        raise ValueError(
            f"Invalid engine name: {engine!r}. "
            "Must be alphanumeric with hyphens, dots, or underscores."
        )


def _validate_extra_name(extra: str) -> None:
    """Raise ``ValueError`` if *extra* contains unsafe characters.

    Extra names are interpolated into ``uv pip install macaw-openvoice[{extra}]``
    subprocess commands.  This validation prevents injection.
    """
    if not _SAFE_EXTRA_RE.match(extra):
        raise ValueError(
            f"Invalid extra name: {extra!r}. "
            "Must be alphanumeric with hyphens, dots, or underscores."
        )


class VenvManager:
    """Manage per-engine Python virtual environments.

    Each engine gets an isolated venv at ``{base_dir}/{engine}/``.
    A ``.macaw-engine`` marker file validates ownership.
    """

    def __init__(self, base_dir: Path, *, uv_path: str = "uv") -> None:
        self._base_dir = base_dir
        self._uv_path = uv_path

    def _engine_dir(self, engine: str) -> Path:
        """Return engine directory path after validating the name.

        All public methods delegate to this to ensure validation happens
        exactly once per call chain.
        """
        _validate_engine_name(engine)
        return self._base_dir / engine

    def venv_dir(self, engine: str) -> Path:
        """Return the venv directory path for *engine*."""
        return self._engine_dir(engine)

    def venv_python(self, engine: str) -> Path:
        """Return the Python executable path inside *engine*'s venv."""
        return self._engine_dir(engine) / "bin" / "python"

    def exists(self, engine: str) -> bool:
        """Check if a provisioned venv exists for *engine*.

        Requires BOTH the venv directory and the ``.macaw-engine`` marker
        to exist (guards against stale or manually-created directories).
        """
        vdir = self._engine_dir(engine)
        marker = vdir / _MARKER_FILENAME
        return vdir.is_dir() and marker.is_file()

    def provision(self, engine: str, *, extra: str | None = None) -> Path:
        """Create a venv for *engine* and install its dependencies.

        Args:
            engine: Engine name (e.g., "faster-whisper").
            extra: pyproject.toml extra name. Defaults to ``ENGINE_EXTRAS[engine]``.

        Returns:
            Path to the venv Python executable.

        Raises:
            VenvProvisionError: If ``uv`` is not found or install fails.
                On failure after venv creation, the orphan directory is
                cleaned up automatically to avoid stale state on disk.
            RuntimeError: If called from inside an async event loop.

        Note:
            This method runs blocking subprocess calls (up to 600s for
            dependency install).  Currently called before the event loop
            in ``macaw serve`` and from the synchronous CLI.  If called
            from async context, wrap with ``asyncio.to_thread()``.
        """
        # Fail-fast if called from inside a running event loop — blocking
        # subprocess calls (up to 720s) would freeze the server.
        try:
            asyncio.get_running_loop()
        except RuntimeError:
            pass  # No running loop — safe to proceed
        else:
            raise RuntimeError(
                "provision() is blocking and must not be called from async context. "
                "Use asyncio.to_thread(manager.provision, engine)."
            )

        vdir = self._engine_dir(engine)
        if extra is None:
            extra = ENGINE_EXTRAS.get(engine, engine)
        _validate_extra_name(extra)

        python_path = vdir / "bin" / "python"

        # Lock venv Python to the runtime version.  If the system does not
        # have this version installed, ``uv venv`` will fail with a clear
        # message — no silent fallback to a different version.
        py_version = f"{sys.version_info.major}.{sys.version_info.minor}"

        # Create directory chain including the venv dir.  ``uv venv``
        # handles pre-existing directories gracefully (overwrites).
        vdir.mkdir(parents=True, exist_ok=True)
        try:
            self._run_create_venv(engine, vdir, py_version)
            self._run_install_deps(engine, python_path, extra)
        except VenvProvisionError:
            # Clean up orphan directory (venv created but install failed,
            # so no marker was written).
            marker = vdir / _MARKER_FILENAME
            if vdir.is_dir() and not marker.is_file():
                try:
                    shutil.rmtree(vdir)
                    logger.info("venv_orphan_cleaned", engine=engine, path=str(vdir))
                except OSError as exc:
                    logger.warning(
                        "venv_orphan_cleanup_failed",
                        engine=engine,
                        path=str(vdir),
                        error=str(exc),
                    )
            raise

        # Step 3: Write marker — only reached when both steps succeed
        marker = vdir / _MARKER_FILENAME
        # Wall clock is correct here — this is persisted metadata for human
        # inspection, not a timeout.  monotonic() is only for deadlines.
        marker_data = {
            "engine": engine,
            "extra": extra,
            "python_version": sys.version,
            "provisioned_at": datetime.now(tz=UTC).isoformat(),
        }
        marker.write_text(json.dumps(marker_data, indent=2))

        return python_path

    def _run_create_venv(self, engine: str, vdir: Path, py_version: str) -> None:
        """Run ``uv venv`` to create the virtual environment."""
        try:
            result = subprocess.run(
                [self._uv_path, "venv", str(vdir), "--python", py_version],
                capture_output=True,
                text=True,
                timeout=_VENV_CREATION_TIMEOUT_S,
            )
        except FileNotFoundError:
            raise VenvProvisionError(
                engine, f"'{self._uv_path}' not found — install uv first"
            ) from None
        except subprocess.TimeoutExpired:
            raise VenvProvisionError(
                engine, f"venv creation timed out ({_VENV_CREATION_TIMEOUT_S}s)"
            ) from None

        if result.returncode != 0:
            raise VenvProvisionError(engine, f"venv creation failed: {result.stderr.strip()}")

        logger.info("venv_created", engine=engine, path=str(vdir))

    @staticmethod
    def _editable_source_dir() -> Path | None:
        """Return the local source directory if macaw-openvoice is editable.

        Uses ``importlib.metadata`` to read ``direct_url.json`` — the
        standard PEP 610 mechanism for detecting direct-URL and editable
        installs.  Returns ``None`` for regular PyPI installs.
        """
        import json as _json
        from importlib.metadata import distribution

        try:
            dist = distribution("macaw-openvoice")
        except Exception:  # PackageNotFoundError
            return None

        raw = dist.read_text("direct_url.json")
        if raw is None:
            return None

        try:
            data = _json.loads(raw)
        except (ValueError, TypeError):
            return None

        if not data.get("dir_info", {}).get("editable", False):
            return None

        url: str = data.get("url", "")
        if url.startswith("file://"):
            return Path(url.removeprefix("file://"))
        return None

    def _run_install_deps(self, engine: str, python_path: Path, extra: str) -> None:
        """Run ``uv pip install`` to install engine dependencies.

        When running from an editable (dev) install, installs from the
        local source directory so that the venv matches the working tree.
        Otherwise installs from PyPI.
        """
        editable_dir = self._editable_source_dir()
        if editable_dir is not None:
            install_spec = f"{editable_dir}[{extra}]"
            logger.debug("venv_install_from_source", engine=engine, source=str(editable_dir))
        else:
            install_spec = f"macaw-openvoice[{extra}]"

        try:
            result = subprocess.run(
                [
                    self._uv_path,
                    "pip",
                    "install",
                    "--python",
                    str(python_path),
                    install_spec,
                ],
                capture_output=True,
                text=True,
                timeout=_DEP_INSTALL_TIMEOUT_S,
            )
        except FileNotFoundError:
            raise VenvProvisionError(
                engine, f"'{self._uv_path}' not found — install uv first"
            ) from None
        except subprocess.TimeoutExpired:
            raise VenvProvisionError(
                engine, f"dependency install timed out ({_DEP_INSTALL_TIMEOUT_S}s)"
            ) from None

        if result.returncode != 0:
            raise VenvProvisionError(engine, f"dependency install failed: {result.stderr.strip()}")

        logger.info("venv_deps_installed", engine=engine, extra=extra)

    def remove(self, engine: str) -> None:
        """Delete the venv directory for *engine*.

        Validates that the resolved path is inside ``_base_dir`` to
        prevent path traversal attacks (e.g., ``engine="../../data"``).
        """
        vdir = self._engine_dir(engine).resolve()
        if not vdir.is_relative_to(self._base_dir.resolve()):
            raise ValueError(f"Invalid engine name: {engine!r}")
        if vdir.is_dir():
            shutil.rmtree(vdir)
            logger.info("venv_removed", engine=engine, path=str(vdir))

    def is_engine_available_in_venv(self, engine: str) -> bool:
        """Check if the engine's package is importable inside its venv."""
        from macaw.engines import ENGINE_PACKAGE

        python_path = self._engine_dir(engine) / "bin" / "python"
        if not python_path.is_file():
            return False

        package = ENGINE_PACKAGE.get(engine)
        if package is None:
            return True

        # Validate package name before interpolating into subprocess command
        # to prevent command injection (e.g., package="os; rm -rf /").
        if not _SAFE_MODULE_RE.match(package):
            logger.warning("unsafe_package_name", engine=engine, package=package)
            return False

        try:
            result = subprocess.run(
                [str(python_path), "-c", f"import {package}"],
                capture_output=True,
                timeout=10,
            )
            return result.returncode == 0
        except (subprocess.TimeoutExpired, FileNotFoundError, OSError) as exc:
            logger.warning("venv_availability_check_failed", engine=engine, error=str(exc))
            return False

    def read_marker(self, engine: str) -> dict[str, str] | None:
        """Read the marker file for *engine*, or ``None`` if missing/corrupt."""
        marker = self._engine_dir(engine) / _MARKER_FILENAME
        if not marker.is_file():
            return None
        try:
            data = json.loads(marker.read_text())
            if isinstance(data, dict):
                return data
            return None
        except (json.JSONDecodeError, OSError):
            return None


def resolve_python_for_engine(engine: str) -> str:
    """Return the Python executable path for an engine's worker.

    Resolution order:
    1. Venv exists -> venv Python
    2. auto_provision=True and venv missing -> provision, return venv Python
    3. Fallback -> sys.executable (backward-compatible)

    Never crashes — logs warning and falls back on any error.
    """
    from macaw.config.settings import get_settings

    settings = get_settings().backend
    manager = VenvManager(settings.venv_base_path, uv_path=settings.uv_path)

    # 1. Venv already exists — validate it's healthy
    if manager.exists(engine):
        python_path = str(manager.venv_python(engine))
        if manager.is_engine_available_in_venv(engine):
            logger.debug("venv_resolved_existing", engine=engine, python=python_path)
            return python_path
        # Stale venv: marker exists but engine is not importable.
        # Remove and fall through to re-provision (or fallback).
        logger.warning("venv_stale_detected", engine=engine, python=python_path)
        try:
            manager.remove(engine)
        except OSError as exc:
            logger.warning(
                "venv_stale_removal_failed",
                engine=engine,
                error=str(exc),
            )
            return sys.executable

    # 2. Auto-provision if enabled
    if settings.auto_provision:
        try:
            python_path = str(manager.provision(engine))
            logger.info("venv_auto_provisioned", engine=engine, python=python_path)
            return python_path
        except VenvProvisionError as exc:
            logger.warning(
                "venv_provision_failed_fallback",
                engine=engine,
                reason=str(exc),
            )
            return sys.executable

    # 3. Fallback
    logger.debug("venv_fallback_sys_executable", engine=engine)
    return sys.executable
