"""Testes das exceptions tipadas do Macaw."""

import pytest

from macaw.exceptions import (
    AudioFormatError,
    AudioTooLargeError,
    ConfigError,
    MacawError,
    ManifestParseError,
    ManifestValidationError,
    ModelLoadError,
    ModelNotFoundError,
    SessionClosedError,
    SessionNotFoundError,
    WorkerCrashError,
    WorkerTimeoutError,
    WorkerUnavailableError,
)


class TestHierarchy:
    def test_all_exceptions_inherit_from_macaw_error(self) -> None:
        exceptions = [
            ManifestParseError("f", "r"),
            ManifestValidationError("f", ["e"]),
            ModelNotFoundError("m"),
            ModelLoadError("m", "r"),
            WorkerCrashError("w"),
            WorkerTimeoutError("w", 5.0),
            WorkerUnavailableError("m"),
            AudioFormatError("d"),
            AudioTooLargeError(100, 50),
            SessionNotFoundError("s"),
            SessionClosedError("s"),
        ]
        for exc in exceptions:
            assert isinstance(exc, MacawError)

    def test_config_errors_are_config_error(self) -> None:
        assert isinstance(ManifestParseError("f", "r"), ConfigError)
        assert isinstance(ManifestValidationError("f", ["e"]), ConfigError)


class TestExceptionMessages:
    def test_model_not_found_has_model_name(self) -> None:
        exc = ModelNotFoundError("faster-whisper-large-v3")
        assert "faster-whisper-large-v3" in str(exc)
        assert exc.model_name == "faster-whisper-large-v3"

    def test_worker_crash_with_exit_code(self) -> None:
        exc = WorkerCrashError("worker-1", exit_code=137)
        assert "137" in str(exc)
        assert exc.exit_code == 137

    def test_worker_crash_without_exit_code(self) -> None:
        exc = WorkerCrashError("worker-1")
        assert exc.exit_code is None

    def test_audio_too_large_shows_mb(self) -> None:
        exc = AudioTooLargeError(
            size_bytes=30 * 1024 * 1024,
            max_bytes=25 * 1024 * 1024,
        )
        assert "30.0MB" in str(exc)
        assert "25.0MB" in str(exc)

    def test_manifest_validation_joins_errors(self) -> None:
        exc = ManifestValidationError(
            "macaw.yaml",
            ["campo 'name' faltando", "campo 'type' invalido"],
        )
        assert "campo 'name' faltando" in str(exc)
        assert "campo 'type' invalido" in str(exc)

    def test_exceptions_are_catchable_by_base(self) -> None:
        with pytest.raises(MacawError):
            raise ModelNotFoundError("test")

    def test_worker_timeout_message(self) -> None:
        exc = WorkerTimeoutError("worker-2", 10.0)
        assert "worker-2" in str(exc)
        assert "10.0s" in str(exc)

    def test_session_not_found_message(self) -> None:
        exc = SessionNotFoundError("sess_abc123")
        assert "sess_abc123" in str(exc)

    def test_session_closed_message(self) -> None:
        exc = SessionClosedError("sess_abc123")
        assert "sess_abc123" in str(exc)

    def test_audio_format_error_message(self) -> None:
        exc = AudioFormatError("formato OGG nao suportado")
        assert "formato OGG nao suportado" in str(exc)

    def test_model_load_error_message(self) -> None:
        exc = ModelLoadError("fw-large-v3", "CUDA out of memory")
        assert "fw-large-v3" in str(exc)
        assert "CUDA out of memory" in str(exc)
