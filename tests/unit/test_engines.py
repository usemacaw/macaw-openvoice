"""Tests for engine availability checks in `macaw.engines`."""

from __future__ import annotations

from types import ModuleType
from unittest.mock import patch

from macaw.engines import is_engine_available


class TestIsEngineAvailable:
    def test_known_engine_available(self) -> None:
        """Known engine whose package is installed returns True."""
        fake_spec = ModuleType("fake_spec")
        with patch("macaw.engines.importlib.util.find_spec", return_value=fake_spec):
            assert is_engine_available("faster-whisper") is True

    def test_known_engine_not_available(self) -> None:
        """Known engine whose package is missing returns False."""
        with patch("macaw.engines.importlib.util.find_spec", return_value=None):
            assert is_engine_available("kokoro") is False

    def test_unknown_engine_returns_true(self) -> None:
        """Engine not in ENGINE_PACKAGE mapping returns True (pass-through)."""
        assert is_engine_available("some-future-engine") is True

    def test_external_engine_available(self) -> None:
        """External engine with python_package checks that package instead."""
        fake_spec = ModuleType("fake_spec")
        with patch("macaw.engines.importlib.util.find_spec", return_value=fake_spec) as mock_spec:
            result = is_engine_available("my-engine", python_package="my_company.stt")

        assert result is True
        mock_spec.assert_called_once_with("my_company.stt")

    def test_external_engine_not_available(self) -> None:
        """External engine whose python_package is not importable returns False."""
        with patch("macaw.engines.importlib.util.find_spec", return_value=None):
            assert is_engine_available("my-engine", python_package="missing.pkg") is False

    def test_external_engine_bypasses_builtin_check(self) -> None:
        """When python_package is set, ENGINE_PACKAGE is not consulted."""
        # kokoro is a known built-in engine, but python_package takes precedence
        fake_spec = ModuleType("fake_spec")
        with patch("macaw.engines.importlib.util.find_spec", return_value=fake_spec) as mock_spec:
            result = is_engine_available("kokoro", python_package="custom_kokoro")

        assert result is True
        mock_spec.assert_called_once_with("custom_kokoro")
