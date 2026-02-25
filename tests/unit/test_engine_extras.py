"""Tests for ENGINE_EXTRAS and VenvProvisionError."""

from __future__ import annotations

from macaw.engines import ENGINE_EXTRAS, ENGINE_PACKAGE
from macaw.exceptions import MacawError, VenvProvisionError


class TestEngineExtras:
    def test_keys_match_engine_package(self) -> None:
        """ENGINE_EXTRAS keys must match ENGINE_PACKAGE keys exactly."""
        assert set(ENGINE_EXTRAS.keys()) == set(ENGINE_PACKAGE.keys())

    def test_all_values_are_strings(self) -> None:
        for engine, extra in ENGINE_EXTRAS.items():
            assert isinstance(extra, str), f"{engine} extra is not a string"
            assert len(extra) > 0, f"{engine} extra is empty"


class TestVenvProvisionError:
    def test_inherits_from_macaw_error(self) -> None:
        assert issubclass(VenvProvisionError, MacawError)

    def test_stores_engine_and_reason(self) -> None:
        err = VenvProvisionError("kokoro", "uv not found")
        assert err.engine == "kokoro"
        assert err.reason == "uv not found"

    def test_message_format(self) -> None:
        err = VenvProvisionError("faster-whisper", "install failed")
        assert str(err) == "Venv provisioning failed for engine 'faster-whisper': install failed"
