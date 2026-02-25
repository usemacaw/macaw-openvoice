"""Tests for VenvManager and resolve_python_for_engine."""

from __future__ import annotations

import json
import sys
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from macaw.backends.venv_manager import (
    VenvManager,
    _validate_engine_name,
    _validate_extra_name,
    resolve_python_for_engine,
)
from macaw.exceptions import VenvProvisionError


class TestValidateEngineName:
    """Validation rejects path traversal and shell injection characters."""

    def test_valid_simple_name(self) -> None:
        _validate_engine_name("kokoro")  # Should not raise

    def test_valid_hyphenated_name(self) -> None:
        _validate_engine_name("faster-whisper")

    def test_valid_dotted_name(self) -> None:
        _validate_engine_name("qwen3.tts")

    def test_valid_underscored_name(self) -> None:
        _validate_engine_name("my_engine")

    def test_valid_single_char(self) -> None:
        _validate_engine_name("x")

    def test_rejects_path_traversal(self) -> None:
        with pytest.raises(ValueError, match="Invalid engine name"):
            _validate_engine_name("../../etc")

    def test_rejects_slash(self) -> None:
        with pytest.raises(ValueError, match="Invalid engine name"):
            _validate_engine_name("foo/bar")

    def test_rejects_semicolon(self) -> None:
        with pytest.raises(ValueError, match="Invalid engine name"):
            _validate_engine_name("foo;rm -rf /")

    def test_rejects_space(self) -> None:
        with pytest.raises(ValueError, match="Invalid engine name"):
            _validate_engine_name("foo bar")

    def test_rejects_empty(self) -> None:
        with pytest.raises(ValueError, match="Invalid engine name"):
            _validate_engine_name("")

    def test_rejects_leading_hyphen(self) -> None:
        with pytest.raises(ValueError, match="Invalid engine name"):
            _validate_engine_name("-engine")

    def test_rejects_trailing_hyphen(self) -> None:
        with pytest.raises(ValueError, match="Invalid engine name"):
            _validate_engine_name("engine-")


class TestVenvManagerPaths:
    def test_venv_dir_returns_correct_path(self, tmp_path: Path) -> None:
        manager = VenvManager(tmp_path)
        assert manager.venv_dir("kokoro") == tmp_path / "kokoro"

    def test_venv_python_returns_bin_python(self, tmp_path: Path) -> None:
        manager = VenvManager(tmp_path)
        expected = tmp_path / "kokoro" / "bin" / "python"
        assert manager.venv_python("kokoro") == expected


class TestVenvManagerExists:
    def test_false_when_dir_missing(self, tmp_path: Path) -> None:
        manager = VenvManager(tmp_path)
        assert manager.exists("kokoro") is False

    def test_false_when_dir_exists_but_no_marker(self, tmp_path: Path) -> None:
        (tmp_path / "kokoro").mkdir()
        manager = VenvManager(tmp_path)
        assert manager.exists("kokoro") is False

    def test_true_when_dir_and_marker_exist(self, tmp_path: Path) -> None:
        venv_dir = tmp_path / "kokoro"
        venv_dir.mkdir()
        (venv_dir / ".macaw-engine").write_text('{"engine": "kokoro"}')
        manager = VenvManager(tmp_path)
        assert manager.exists("kokoro") is True


class TestVenvManagerMarker:
    def test_read_marker_returns_data(self, tmp_path: Path) -> None:
        venv_dir = tmp_path / "kokoro"
        venv_dir.mkdir()
        data = {"engine": "kokoro", "extra": "kokoro", "provisioned_at": "2026-01-01T00:00:00Z"}
        (venv_dir / ".macaw-engine").write_text(json.dumps(data))
        manager = VenvManager(tmp_path)
        result = manager.read_marker("kokoro")
        assert result is not None
        assert result["engine"] == "kokoro"

    def test_read_marker_returns_none_when_missing(self, tmp_path: Path) -> None:
        manager = VenvManager(tmp_path)
        assert manager.read_marker("kokoro") is None

    def test_read_marker_returns_none_for_non_dict_json(self, tmp_path: Path) -> None:
        """A valid JSON file that is not a dict should return None."""
        venv_dir = tmp_path / "kokoro"
        venv_dir.mkdir()
        (venv_dir / ".macaw-engine").write_text("[1, 2, 3]")
        manager = VenvManager(tmp_path)
        assert manager.read_marker("kokoro") is None

    def test_read_marker_returns_none_for_corrupt_json(self, tmp_path: Path) -> None:
        venv_dir = tmp_path / "kokoro"
        venv_dir.mkdir()
        (venv_dir / ".macaw-engine").write_text("not json at all")
        manager = VenvManager(tmp_path)
        assert manager.read_marker("kokoro") is None


class TestVenvManagerProvision:
    @patch("macaw.backends.venv_manager.subprocess.run")
    def test_provision_success(self, mock_run: MagicMock, tmp_path: Path) -> None:
        mock_run.return_value = MagicMock(returncode=0, stderr="", stdout="")
        manager = VenvManager(tmp_path, uv_path="/usr/bin/uv")

        python_path = manager.provision("kokoro")

        assert python_path == tmp_path / "kokoro" / "bin" / "python"
        # Two subprocess calls: venv creation + pip install
        assert mock_run.call_count == 2

        # Marker file written
        marker = tmp_path / "kokoro" / ".macaw-engine"
        assert marker.is_file()
        marker_data = json.loads(marker.read_text())
        assert marker_data["engine"] == "kokoro"
        assert marker_data["extra"] == "kokoro"
        assert "provisioned_at" in marker_data

    @patch("macaw.backends.venv_manager.subprocess.run")
    def test_provision_uv_not_found(self, mock_run: MagicMock, tmp_path: Path) -> None:
        mock_run.side_effect = FileNotFoundError("uv not found")
        manager = VenvManager(tmp_path)

        with pytest.raises(VenvProvisionError, match="not found"):
            manager.provision("kokoro")

    @patch("macaw.backends.venv_manager.subprocess.run")
    def test_provision_venv_creation_fails(self, mock_run: MagicMock, tmp_path: Path) -> None:
        mock_run.return_value = MagicMock(returncode=1, stderr="some error")
        manager = VenvManager(tmp_path)

        with pytest.raises(VenvProvisionError, match="venv creation failed"):
            manager.provision("kokoro")

    @patch("macaw.backends.venv_manager.subprocess.run")
    def test_provision_install_fails(self, mock_run: MagicMock, tmp_path: Path) -> None:
        # First call (venv creation) succeeds, second (install) fails
        mock_run.side_effect = [
            MagicMock(returncode=0, stderr=""),
            MagicMock(returncode=1, stderr="install error"),
        ]
        manager = VenvManager(tmp_path)

        with pytest.raises(VenvProvisionError, match="dependency install failed"):
            manager.provision("kokoro")

    @patch("macaw.backends.venv_manager.subprocess.run")
    def test_provision_custom_extra(self, mock_run: MagicMock, tmp_path: Path) -> None:
        mock_run.return_value = MagicMock(returncode=0, stderr="", stdout="")
        manager = VenvManager(tmp_path, uv_path="uv")

        manager.provision("kokoro", extra="kokoro-gpu")

        # Verify the install command used custom extra
        install_call = mock_run.call_args_list[1]
        cmd = install_call[0][0]
        install_spec = cmd[-1]  # Last arg is the install specifier
        assert install_spec.endswith("[kokoro-gpu]")

    @patch("macaw.backends.venv_manager.subprocess.run")
    def test_provision_cleans_orphan_on_install_failure(
        self, mock_run: MagicMock, tmp_path: Path
    ) -> None:
        """When install fails after venv creation, orphan dir is removed."""
        mock_run.side_effect = [
            MagicMock(returncode=0, stderr=""),  # venv creation OK
            MagicMock(returncode=1, stderr="install error"),  # install fails
        ]
        manager = VenvManager(tmp_path)

        with pytest.raises(VenvProvisionError, match="dependency install failed"):
            manager.provision("kokoro")

        # Orphan directory should have been cleaned up
        venv_dir = tmp_path / "kokoro"
        assert not venv_dir.exists()

    @patch("macaw.backends.venv_manager.subprocess.run")
    def test_provision_rejects_invalid_engine_name(
        self, mock_run: MagicMock, tmp_path: Path
    ) -> None:
        manager = VenvManager(tmp_path)
        with pytest.raises(ValueError, match="Invalid engine name"):
            manager.provision("../malicious")


class TestVenvManagerRemove:
    def test_remove_existing(self, tmp_path: Path) -> None:
        venv_dir = tmp_path / "kokoro"
        venv_dir.mkdir()
        (venv_dir / ".macaw-engine").write_text("{}")
        manager = VenvManager(tmp_path)

        manager.remove("kokoro")
        assert not venv_dir.exists()

    def test_remove_nonexistent_is_noop(self, tmp_path: Path) -> None:
        manager = VenvManager(tmp_path)
        manager.remove("kokoro")  # Should not raise

    def test_remove_rejects_path_traversal(self, tmp_path: Path) -> None:
        """Engine name with '../' must be rejected."""
        manager = VenvManager(tmp_path)
        with pytest.raises(ValueError, match="Invalid engine name"):
            manager.remove("../../etc")


class TestVenvManagerAvailability:
    @patch("macaw.backends.venv_manager.subprocess.run")
    def test_available_when_import_succeeds(self, mock_run: MagicMock, tmp_path: Path) -> None:
        mock_run.return_value = MagicMock(returncode=0)
        # Create the fake python binary path so is_file() passes
        python_path = tmp_path / "kokoro" / "bin" / "python"
        python_path.parent.mkdir(parents=True)
        python_path.touch()

        manager = VenvManager(tmp_path)
        assert manager.is_engine_available_in_venv("kokoro") is True

    def test_unavailable_when_no_python(self, tmp_path: Path) -> None:
        manager = VenvManager(tmp_path)
        assert manager.is_engine_available_in_venv("kokoro") is False

    @patch("macaw.backends.venv_manager.subprocess.run")
    def test_rejects_unsafe_package_name(self, mock_run: MagicMock, tmp_path: Path) -> None:
        """Package name with shell metacharacters should be rejected."""
        python_path = tmp_path / "kokoro" / "bin" / "python"
        python_path.parent.mkdir(parents=True)
        python_path.touch()

        manager = VenvManager(tmp_path)
        with patch("macaw.engines.ENGINE_PACKAGE", {"kokoro": "os; rm -rf /"}):
            result = manager.is_engine_available_in_venv("kokoro")
            assert result is False
            mock_run.assert_not_called()


class TestResolvePythonForEngine:
    @patch(
        "macaw.backends.venv_manager.VenvManager.is_engine_available_in_venv", return_value=True
    )
    @patch("macaw.backends.venv_manager.VenvManager.exists", return_value=True)
    @patch("macaw.backends.venv_manager.VenvManager.venv_python")
    def test_returns_existing_venv_python(
        self,
        mock_venv_python: MagicMock,
        mock_exists: MagicMock,
        mock_available: MagicMock,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        from macaw.config.settings import get_settings

        mock_venv_python.return_value = Path("/venvs/kokoro/bin/python")
        monkeypatch.setenv("MACAW_VENV_DIR", "/venvs")
        get_settings.cache_clear()

        try:
            result = resolve_python_for_engine("kokoro")
            assert result == "/venvs/kokoro/bin/python"
        finally:
            get_settings.cache_clear()

    @patch("macaw.backends.venv_manager.VenvManager.remove")
    @patch("macaw.backends.venv_manager.VenvManager.provision")
    @patch(
        "macaw.backends.venv_manager.VenvManager.is_engine_available_in_venv", return_value=False
    )
    @patch("macaw.backends.venv_manager.VenvManager.exists", return_value=True)
    @patch("macaw.backends.venv_manager.VenvManager.venv_python")
    def test_stale_venv_detected_and_reprovisioned(
        self,
        mock_venv_python: MagicMock,
        mock_exists: MagicMock,
        mock_available: MagicMock,
        mock_provision: MagicMock,
        mock_remove: MagicMock,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """Stale venv (marker exists but engine not importable) is removed and reprovisioned."""
        from macaw.config.settings import get_settings

        mock_venv_python.return_value = Path("/venvs/kokoro/bin/python")
        mock_provision.return_value = Path("/venvs/kokoro/bin/python")
        # After remove, exists returns False so provision path is taken
        mock_exists.side_effect = [True, False]
        monkeypatch.setenv("MACAW_VENV_DIR", "/venvs")
        monkeypatch.setenv("MACAW_BACKEND_AUTO_PROVISION", "true")
        get_settings.cache_clear()

        try:
            result = resolve_python_for_engine("kokoro")
            mock_remove.assert_called_once_with("kokoro")
            mock_provision.assert_called_once_with("kokoro")
            assert result == "/venvs/kokoro/bin/python"
        finally:
            get_settings.cache_clear()

    @patch("macaw.backends.venv_manager.VenvManager.remove")
    @patch(
        "macaw.backends.venv_manager.VenvManager.provision",
        side_effect=VenvProvisionError("kokoro", "network error"),
    )
    @patch(
        "macaw.backends.venv_manager.VenvManager.is_engine_available_in_venv", return_value=False
    )
    @patch("macaw.backends.venv_manager.VenvManager.exists", return_value=True)
    @patch("macaw.backends.venv_manager.VenvManager.venv_python")
    def test_stale_venv_fallback_when_reprovision_fails(
        self,
        mock_venv_python: MagicMock,
        mock_exists: MagicMock,
        mock_available: MagicMock,
        mock_provision: MagicMock,
        mock_remove: MagicMock,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """Stale venv removed, reprovision fails — falls back to sys.executable."""
        from macaw.config.settings import get_settings

        mock_venv_python.return_value = Path("/venvs/kokoro/bin/python")
        mock_exists.side_effect = [True, False]
        monkeypatch.setenv("MACAW_VENV_DIR", "/venvs")
        monkeypatch.setenv("MACAW_BACKEND_AUTO_PROVISION", "true")
        get_settings.cache_clear()

        try:
            result = resolve_python_for_engine("kokoro")
            assert result == sys.executable
            mock_remove.assert_called_once_with("kokoro")
        finally:
            get_settings.cache_clear()

    @patch(
        "macaw.backends.venv_manager.VenvManager.remove",
        side_effect=OSError("permission denied"),
    )
    @patch(
        "macaw.backends.venv_manager.VenvManager.is_engine_available_in_venv", return_value=False
    )
    @patch("macaw.backends.venv_manager.VenvManager.exists", return_value=True)
    @patch("macaw.backends.venv_manager.VenvManager.venv_python")
    def test_stale_venv_removal_failure_falls_back(
        self,
        mock_venv_python: MagicMock,
        mock_exists: MagicMock,
        mock_available: MagicMock,
        mock_remove: MagicMock,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """Cannot remove stale venv — falls back to sys.executable."""
        from macaw.config.settings import get_settings

        mock_venv_python.return_value = Path("/venvs/kokoro/bin/python")
        monkeypatch.setenv("MACAW_VENV_DIR", "/venvs")
        get_settings.cache_clear()

        try:
            result = resolve_python_for_engine("kokoro")
            assert result == sys.executable
        finally:
            get_settings.cache_clear()

    @patch("macaw.backends.venv_manager.VenvManager.exists", return_value=False)
    @patch("macaw.backends.venv_manager.VenvManager.provision")
    def test_auto_provisions_when_missing(
        self,
        mock_provision: MagicMock,
        mock_exists: MagicMock,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        from macaw.config.settings import get_settings

        mock_provision.return_value = Path("/venvs/kokoro/bin/python")
        monkeypatch.setenv("MACAW_VENV_DIR", "/venvs")
        monkeypatch.setenv("MACAW_BACKEND_AUTO_PROVISION", "true")
        get_settings.cache_clear()

        try:
            result = resolve_python_for_engine("kokoro")
            assert result == "/venvs/kokoro/bin/python"
            mock_provision.assert_called_once_with("kokoro")
        finally:
            get_settings.cache_clear()

    @patch("macaw.backends.venv_manager.VenvManager.exists", return_value=False)
    def test_fallback_when_auto_provision_disabled(
        self,
        mock_exists: MagicMock,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        from macaw.config.settings import get_settings

        monkeypatch.setenv("MACAW_BACKEND_AUTO_PROVISION", "false")
        get_settings.cache_clear()

        try:
            result = resolve_python_for_engine("kokoro")
            assert result == sys.executable
        finally:
            get_settings.cache_clear()

    @patch("macaw.backends.venv_manager.VenvManager.exists", return_value=False)
    @patch(
        "macaw.backends.venv_manager.VenvManager.provision",
        side_effect=VenvProvisionError("kokoro", "uv not found"),
    )
    def test_fallback_on_provision_error(
        self,
        mock_provision: MagicMock,
        mock_exists: MagicMock,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        from macaw.config.settings import get_settings

        monkeypatch.setenv("MACAW_BACKEND_AUTO_PROVISION", "true")
        get_settings.cache_clear()

        try:
            result = resolve_python_for_engine("kokoro")
            assert result == sys.executable
        finally:
            get_settings.cache_clear()

    @patch("macaw.backends.venv_manager.VenvManager.exists", return_value=False)
    @patch(
        "macaw.backends.venv_manager.VenvManager.provision",
        side_effect=VenvProvisionError("kokoro", "disk full"),
    )
    def test_logs_warning_on_fallback(
        self,
        mock_provision: MagicMock,
        mock_exists: MagicMock,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """Fallback logs a warning with engine and reason."""
        from macaw.config.settings import get_settings

        monkeypatch.setenv("MACAW_BACKEND_AUTO_PROVISION", "true")
        get_settings.cache_clear()

        try:
            with patch("macaw.backends.venv_manager.logger") as mock_logger:
                resolve_python_for_engine("kokoro")
                mock_logger.warning.assert_called_once()
                call_kwargs = mock_logger.warning.call_args
                assert "kokoro" in str(call_kwargs)
        finally:
            get_settings.cache_clear()


class TestValidateExtraName:
    """Extra name validation prevents injection in uv pip install commands."""

    def test_valid_simple_extra(self) -> None:
        _validate_extra_name("kokoro")

    def test_valid_hyphenated_extra(self) -> None:
        _validate_extra_name("kokoro-gpu")

    def test_valid_dotted_extra(self) -> None:
        _validate_extra_name("qwen3.tts")

    def test_rejects_semicolon_injection(self) -> None:
        with pytest.raises(ValueError, match="Invalid extra name"):
            _validate_extra_name("]; pip install evil; #")

    def test_rejects_brackets(self) -> None:
        with pytest.raises(ValueError, match="Invalid extra name"):
            _validate_extra_name("foo[bar]")

    def test_rejects_spaces(self) -> None:
        with pytest.raises(ValueError, match="Invalid extra name"):
            _validate_extra_name("foo bar")

    def test_rejects_empty(self) -> None:
        with pytest.raises(ValueError, match="Invalid extra name"):
            _validate_extra_name("")


class TestProvisionAsyncGuard:
    """provision() must not be called from inside a running event loop."""

    async def test_raises_when_called_from_async(self, tmp_path: Path) -> None:
        """Calling provision() from async context raises RuntimeError."""
        manager = VenvManager(tmp_path)
        with pytest.raises(RuntimeError, match="must not be called from async context"):
            manager.provision("kokoro")

    def test_succeeds_outside_event_loop(self, tmp_path: Path) -> None:
        """provision() works normally when no event loop is running."""
        with patch("macaw.backends.venv_manager.subprocess.run") as mock_run:
            mock_run.return_value = MagicMock(returncode=0, stderr="", stdout="")
            manager = VenvManager(tmp_path, uv_path="uv")
            python_path = manager.provision("kokoro")
            assert python_path == tmp_path / "kokoro" / "bin" / "python"


class TestProvisionExtraValidation:
    """provision() validates extra name before interpolation."""

    def test_rejects_malicious_extra(self, tmp_path: Path) -> None:
        """Injected extra name is caught before reaching subprocess."""
        manager = VenvManager(tmp_path)
        with pytest.raises(ValueError, match="Invalid extra name"):
            manager.provision("kokoro", extra="]; pip install evil; #")

    @patch("macaw.backends.venv_manager.subprocess.run")
    def test_accepts_valid_custom_extra(self, mock_run: MagicMock, tmp_path: Path) -> None:
        mock_run.return_value = MagicMock(returncode=0, stderr="", stdout="")
        manager = VenvManager(tmp_path, uv_path="uv")
        manager.provision("kokoro", extra="kokoro-gpu")
        # Verify custom extra was used
        install_call = mock_run.call_args_list[1]
        cmd = install_call[0][0]
        install_spec = cmd[-1]  # Last arg is the install specifier
        assert install_spec.endswith("[kokoro-gpu]")


class TestProvisionMarkerPythonVersion:
    """Marker file includes full Python version for diagnostics."""

    @patch("macaw.backends.venv_manager.subprocess.run")
    def test_marker_includes_python_version(self, mock_run: MagicMock, tmp_path: Path) -> None:
        mock_run.return_value = MagicMock(returncode=0, stderr="", stdout="")
        manager = VenvManager(tmp_path, uv_path="uv")
        manager.provision("kokoro")

        marker = tmp_path / "kokoro" / ".macaw-engine"
        data = json.loads(marker.read_text())
        assert "python_version" in data
        assert data["python_version"] == sys.version


class TestOrphanCleanupLogging:
    """Orphan cleanup logs warning when rmtree fails."""

    @patch("macaw.backends.venv_manager.subprocess.run")
    @patch("macaw.backends.venv_manager.shutil.rmtree", side_effect=OSError("permission denied"))
    def test_logs_warning_on_cleanup_failure(
        self, mock_rmtree: MagicMock, mock_run: MagicMock, tmp_path: Path
    ) -> None:
        mock_run.side_effect = [
            MagicMock(returncode=0, stderr=""),  # venv creation OK
            MagicMock(returncode=1, stderr="install error"),  # install fails
        ]
        manager = VenvManager(tmp_path)

        with (
            pytest.raises(VenvProvisionError, match="dependency install failed"),
            patch("macaw.backends.venv_manager.logger") as mock_logger,
        ):
            manager.provision("kokoro")

        mock_logger.warning.assert_called_once()
        call_kwargs = mock_logger.warning.call_args
        assert "venv_orphan_cleanup_failed" in str(call_kwargs)
        assert "permission denied" in str(call_kwargs)


class TestAvailabilityCheckLogging:
    """is_engine_available_in_venv logs warning on suppressed exceptions."""

    @patch("macaw.backends.venv_manager.subprocess.run")
    def test_logs_warning_on_timeout(self, mock_run: MagicMock, tmp_path: Path) -> None:
        import subprocess as sp

        mock_run.side_effect = sp.TimeoutExpired("python", 10)
        python_path = tmp_path / "kokoro" / "bin" / "python"
        python_path.parent.mkdir(parents=True)
        python_path.touch()

        manager = VenvManager(tmp_path)
        with patch("macaw.backends.venv_manager.logger") as mock_logger:
            result = manager.is_engine_available_in_venv("kokoro")

        assert result is False
        mock_logger.warning.assert_called_once()
        assert "venv_availability_check_failed" in str(mock_logger.warning.call_args)

    def test_logs_warning_on_file_not_found(self, tmp_path: Path) -> None:
        python_path = tmp_path / "kokoro" / "bin" / "python"
        python_path.parent.mkdir(parents=True)
        python_path.touch()

        manager = VenvManager(tmp_path)
        with (
            patch(
                "macaw.backends.venv_manager.subprocess.run",
                side_effect=FileNotFoundError("no such file"),
            ),
            patch("macaw.backends.venv_manager.logger") as mock_logger,
        ):
            result = manager.is_engine_available_in_venv("kokoro")

        assert result is False
        mock_logger.warning.assert_called_once()
        assert "venv_availability_check_failed" in str(mock_logger.warning.call_args)
