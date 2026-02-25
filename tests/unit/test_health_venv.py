"""Tests for backend_venvs section in /health endpoint."""

from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock, patch

from macaw.server.routes import health as health_module
from macaw.server.routes.health import _probe_backend_venvs, _TTLCache

# Patch targets match the lazy imports inside _probe_backend_venvs
_VENV_MANAGER = "macaw.backends.venv_manager.VenvManager"
_GET_SETTINGS = "macaw.config.settings.get_settings"


def _reset_venv_cache() -> None:
    """Clear the TTL cache between tests."""
    health_module._backend_venvs_cache = _TTLCache(ttl_s=30.0)


class TestProbeBackendVenvs:
    def setup_method(self) -> None:
        _reset_venv_cache()

    def teardown_method(self) -> None:
        _reset_venv_cache()

    @patch(_GET_SETTINGS)
    @patch(_VENV_MANAGER)
    def test_returns_per_engine_status(self, mock_cls: MagicMock, mock_get: MagicMock) -> None:
        mock_get.return_value.backend.venv_base_path = Path("/tmp/venvs")
        mock_get.return_value.backend.uv_path = "uv"
        mock_cls.return_value.exists.return_value = False

        result = _probe_backend_venvs()
        assert isinstance(result, dict)
        # Should contain all engines from ENGINE_EXTRAS
        assert "faster-whisper" in result
        assert "kokoro" in result
        assert result["kokoro"] == {"provisioned": False}

    @patch(_GET_SETTINGS)
    @patch(_VENV_MANAGER)
    def test_reports_provisioned_engine(self, mock_cls: MagicMock, mock_get: MagicMock) -> None:
        mock_get.return_value.backend.venv_base_path = Path("/tmp/venvs")
        mock_get.return_value.backend.uv_path = "uv"

        def exists_side_effect(engine: str) -> bool:
            return engine == "kokoro"

        mock_cls.return_value.exists.side_effect = exists_side_effect

        result = _probe_backend_venvs()
        assert result["kokoro"] == {"provisioned": True}
        assert result["faster-whisper"] == {"provisioned": False}

    @patch(_GET_SETTINGS)
    @patch(_VENV_MANAGER)
    def test_ttl_cache_avoids_repeated_io(self, mock_cls: MagicMock, mock_get: MagicMock) -> None:
        """Second call within TTL returns cached result without new VenvManager."""
        mock_get.return_value.backend.venv_base_path = Path("/tmp/venvs")
        mock_get.return_value.backend.uv_path = "uv"
        mock_cls.return_value.exists.return_value = False

        _probe_backend_venvs()
        call_count_1 = mock_cls.call_count

        _probe_backend_venvs()
        call_count_2 = mock_cls.call_count

        # Second call should NOT create a new VenvManager (cache hit)
        assert call_count_2 == call_count_1

    @patch("macaw.server.routes.health.time.monotonic", return_value=100.0)
    @patch(_GET_SETTINGS)
    @patch(_VENV_MANAGER)
    def test_ttl_cache_expires(
        self, mock_cls: MagicMock, mock_get: MagicMock, mock_time: MagicMock
    ) -> None:
        """After TTL expires, result is refreshed."""
        mock_get.return_value.backend.venv_base_path = Path("/tmp/venvs")
        mock_get.return_value.backend.uv_path = "uv"
        mock_cls.return_value.exists.return_value = False

        _probe_backend_venvs()
        call_count_1 = mock_cls.call_count

        # Advance time past TTL
        mock_time.return_value = 200.0
        _probe_backend_venvs()
        call_count_2 = mock_cls.call_count

        # Should have created a new VenvManager
        assert call_count_2 > call_count_1
