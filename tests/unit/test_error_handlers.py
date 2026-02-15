"""Tests for HTTP exception handlers in macaw.server.error_handlers.

Each handler maps a typed Macaw exception to a specific HTTP status code
and OpenAI-compatible response format.
"""

from __future__ import annotations

from unittest.mock import MagicMock

from fastapi import FastAPI

from macaw.exceptions import (
    AudioFormatError,
    AudioTooLargeError,
    InvalidRequestError,
    MacawError,
    ModelNotFoundError,
    VoiceNotFoundError,
    WorkerCrashError,
    WorkerTimeoutError,
    WorkerUnavailableError,
)
from macaw.server.error_handlers import (
    _handle_audio_format_error,
    _handle_audio_too_large,
    _handle_invalid_request,
    _handle_macaw_error,
    _handle_model_not_found,
    _handle_unexpected_error,
    _handle_voice_not_found,
    _handle_worker_crash,
    _handle_worker_timeout,
    _handle_worker_unavailable,
    register_error_handlers,
)


def _make_request(*, request_id: str | None = None) -> MagicMock:
    """Create a mock Request with optional request_id on state."""
    request = MagicMock()
    if request_id is not None:
        request.state.request_id = request_id
    else:
        # getattr(request.state, "request_id", None) should return None
        del request.state.request_id
    return request


class TestHandleVoiceNotFound:
    async def test_returns_404(self) -> None:
        # Arrange
        request = _make_request()
        exc = VoiceNotFoundError(voice_id="my-voice")

        # Act
        response = await _handle_voice_not_found(request, exc)

        # Assert
        assert response.status_code == 404
        body = response.body.decode()
        assert "voice_not_found_error" in body
        assert "voice_not_found" in body

    async def test_includes_exception_message(self) -> None:
        request = _make_request()
        exc = VoiceNotFoundError(voice_id="missing-voice")

        response = await _handle_voice_not_found(request, exc)

        assert response.status_code == 404
        body = response.body.decode()
        assert "missing-voice" in body


class TestHandleModelNotFound:
    async def test_returns_404(self) -> None:
        request = _make_request()
        exc = ModelNotFoundError(model_name="whisper-large-v4")

        response = await _handle_model_not_found(request, exc)

        assert response.status_code == 404
        body = response.body.decode()
        assert "model_not_found_error" in body
        assert "model_not_found" in body
        assert "whisper-large-v4" in body


class TestHandleInvalidRequest:
    async def test_returns_400(self) -> None:
        request = _make_request()
        exc = InvalidRequestError(detail="Missing required field 'model'")

        response = await _handle_invalid_request(request, exc)

        assert response.status_code == 400
        body = response.body.decode()
        assert "invalid_request_error" in body
        assert "invalid_request" in body


class TestHandleAudioFormatError:
    async def test_returns_400(self) -> None:
        request = _make_request()
        exc = AudioFormatError(detail="Unsupported codec: opus")

        response = await _handle_audio_format_error(request, exc)

        assert response.status_code == 400
        body = response.body.decode()
        assert "audio_format_error" in body
        assert "invalid_audio" in body


class TestHandleAudioTooLarge:
    async def test_returns_413(self) -> None:
        request = _make_request()
        exc = AudioTooLargeError(size_bytes=50_000_000, max_bytes=25_000_000)

        response = await _handle_audio_too_large(request, exc)

        assert response.status_code == 413
        body = response.body.decode()
        assert "audio_too_large_error" in body
        assert "file_too_large" in body


class TestHandleWorkerUnavailable:
    async def test_returns_503_with_retry_after(self) -> None:
        request = _make_request()
        exc = WorkerUnavailableError(model_name="whisper-large-v3")

        response = await _handle_worker_unavailable(request, exc)

        assert response.status_code == 503
        assert response.headers.get("retry-after") == "5"
        body = response.body.decode()
        assert "worker_unavailable_error" in body
        assert "service_unavailable" in body


class TestHandleWorkerTimeout:
    async def test_returns_504(self) -> None:
        request = _make_request()
        exc = WorkerTimeoutError(worker_id="stt-0", timeout_seconds=30.0)

        response = await _handle_worker_timeout(request, exc)

        assert response.status_code == 504
        body = response.body.decode()
        assert "worker_timeout_error" in body
        assert "gateway_timeout" in body
        assert "30s" in body


class TestHandleWorkerCrash:
    async def test_returns_502_with_retry_after(self) -> None:
        request = _make_request()
        exc = WorkerCrashError(worker_id="stt-0", exit_code=137)

        response = await _handle_worker_crash(request, exc)

        assert response.status_code == 502
        assert response.headers.get("retry-after") == "5"
        body = response.body.decode()
        assert "worker_crash_error" in body
        assert "bad_gateway" in body


class TestHandleMacawError:
    async def test_returns_500(self) -> None:
        request = _make_request()
        exc = MacawError("Something went wrong in domain logic")

        response = await _handle_macaw_error(request, exc)

        assert response.status_code == 500
        body = response.body.decode()
        assert "internal_error" in body

    async def test_does_not_leak_internal_details(self) -> None:
        request = _make_request()
        exc = MacawError("secret internal detail")

        response = await _handle_macaw_error(request, exc)

        body = response.body.decode()
        assert "secret internal detail" not in body
        assert "Internal server error" in body


class TestHandleUnexpectedError:
    async def test_returns_500(self) -> None:
        request = _make_request()
        exc = RuntimeError("unexpected crash")

        response = await _handle_unexpected_error(request, exc)

        assert response.status_code == 500
        body = response.body.decode()
        assert "internal_error" in body

    async def test_does_not_leak_internal_details(self) -> None:
        request = _make_request()
        exc = ValueError("should not be visible")

        response = await _handle_unexpected_error(request, exc)

        body = response.body.decode()
        assert "should not be visible" not in body
        assert "Internal server error" in body


class TestRegisterErrorHandlers:
    def test_registers_all_handlers(self) -> None:
        app = FastAPI()

        register_error_handlers(app)

        registered = app.exception_handlers
        assert InvalidRequestError in registered
        assert VoiceNotFoundError in registered
        assert ModelNotFoundError in registered
        assert AudioFormatError in registered
        assert AudioTooLargeError in registered
        assert WorkerUnavailableError in registered
        assert WorkerTimeoutError in registered
        assert WorkerCrashError in registered
        assert MacawError in registered
        assert Exception in registered

    def test_registers_exactly_ten_handlers(self) -> None:
        app = FastAPI()
        # FastAPI registers some default handlers (e.g., HTTPException, RequestValidationError)
        default_count = len(app.exception_handlers)

        register_error_handlers(app)

        added = len(app.exception_handlers) - default_count
        assert added == 10
