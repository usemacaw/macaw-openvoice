"""Tests for cloud_storage_url on POST /v1/audio/transcriptions (Sprint M, Task M2).

Validates:
 - Exactly one of file/cloud_storage_url required (400 if both or neither)
 - cloud_storage_url downloads and transcribes audio
 - STTDownloadSettings defaults and env var overrides
 - AudioDownloadError maps to 502 in error handler
 - Endpoint still works with file upload (backward compat)
"""

from __future__ import annotations

from io import BytesIO
from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from fastapi import UploadFile

from macaw.config.settings import STTDownloadSettings
from macaw.exceptions import AudioDownloadError
from macaw.server.error_handlers import _handle_audio_download_error

# ---------------------------------------------------------------------------
# STTDownloadSettings
# ---------------------------------------------------------------------------


class TestSTTDownloadSettings:
    """STTDownloadSettings Pydantic model."""

    def test_defaults(self) -> None:
        settings = STTDownloadSettings()
        assert settings.max_size_bytes == 2 * 1024 * 1024 * 1024  # 2 GB
        assert settings.timeout_s == 120.0

    def test_custom_values(self) -> None:
        settings = STTDownloadSettings(max_size_bytes=500_000_000, timeout_s=60.0)
        assert settings.max_size_bytes == 500_000_000
        assert settings.timeout_s == 60.0

    def test_min_size_validation(self) -> None:
        import pydantic

        with pytest.raises(pydantic.ValidationError):
            STTDownloadSettings(max_size_bytes=100)

    def test_max_timeout_validation(self) -> None:
        import pydantic

        with pytest.raises(pydantic.ValidationError):
            STTDownloadSettings(timeout_s=9999)

    def test_env_var_aliases(self) -> None:
        with patch.dict(
            "os.environ",
            {
                "MACAW_STT_MAX_DOWNLOAD_SIZE_BYTES": "1000000",
                "MACAW_STT_DOWNLOAD_TIMEOUT_S": "30",
            },
        ):
            settings = STTDownloadSettings()
            assert settings.max_size_bytes == 1_000_000
            assert settings.timeout_s == 30.0


# ---------------------------------------------------------------------------
# AudioDownloadError handler
# ---------------------------------------------------------------------------


class TestAudioDownloadErrorHandler:
    """AudioDownloadError maps to HTTP 502."""

    @pytest.mark.asyncio
    async def test_returns_502(self) -> None:
        exc = AudioDownloadError("Connection refused")
        request = MagicMock()
        request.state = MagicMock(spec=[])

        response = await _handle_audio_download_error(request, exc)

        assert response.status_code == 502
        assert b"bad_gateway" in response.body

    @pytest.mark.asyncio
    async def test_error_message_in_response(self) -> None:
        exc = AudioDownloadError("HTTP 404 from remote server")
        request = MagicMock()
        request.state = MagicMock(spec=[])

        response = await _handle_audio_download_error(request, exc)

        assert b"Audio download failed" in response.body


# ---------------------------------------------------------------------------
# Endpoint validation — exactly one of file/url
# ---------------------------------------------------------------------------


class TestEndpointValidation:
    """Exactly one of file/cloud_storage_url must be provided."""

    @pytest.mark.asyncio
    async def test_both_file_and_url_returns_400(self) -> None:
        """Providing both file and cloud_storage_url returns 400."""
        from macaw.exceptions import InvalidRequestError
        from macaw.server.routes.transcriptions import create_transcription

        file = UploadFile(file=BytesIO(b"audio"), filename="test.wav")
        request = _make_mock_request()

        with pytest.raises(InvalidRequestError, match="not both"):
            await create_transcription(
                request=request,
                file=file,
                cloud_storage_url="https://example.com/audio.wav",
                model="whisper",
                **_default_form_params(),
            )

    @pytest.mark.asyncio
    async def test_neither_file_nor_url_returns_400(self) -> None:
        """Providing neither file nor cloud_storage_url returns 400."""
        from macaw.exceptions import InvalidRequestError
        from macaw.server.routes.transcriptions import create_transcription

        request = _make_mock_request()

        with pytest.raises(InvalidRequestError, match="required"):
            await create_transcription(
                request=request,
                file=None,
                cloud_storage_url=None,
                model="whisper",
                **_default_form_params(),
            )


# ---------------------------------------------------------------------------
# Endpoint with cloud_storage_url
# ---------------------------------------------------------------------------


class TestEndpointWithUrl:
    """Endpoint downloads audio from URL and transcribes it."""

    @pytest.mark.asyncio
    async def test_url_downloads_and_transcribes(self) -> None:
        """cloud_storage_url triggers download_audio then handle_audio_request."""
        from macaw.server.routes.transcriptions import create_transcription

        request = _make_mock_request()
        audio_bytes = b"RIFF" + b"\x00" * 100

        mock_settings = MagicMock()
        mock_settings.stt_download.max_size_bytes = 1024 * 1024
        mock_settings.stt_download.timeout_s = 30.0

        with (
            patch(
                "macaw.server.audio_downloader.download_audio",
                new_callable=AsyncMock,
                return_value=audio_bytes,
            ) as mock_download,
            patch(
                "macaw.config.settings.get_settings",
                return_value=mock_settings,
            ),
            patch(
                "macaw.server.routes.transcriptions.handle_audio_request",
                new_callable=AsyncMock,
                return_value={"text": "hello world"},
            ) as mock_handle,
        ):
            result = await create_transcription(
                request=request,
                file=None,
                cloud_storage_url="https://s3.amazonaws.com/bucket/audio.wav",
                model="whisper",
                **_default_form_params(),
            )

        assert result == {"text": "hello world"}
        mock_download.assert_awaited_once_with(
            "https://s3.amazonaws.com/bucket/audio.wav",
            max_size_bytes=1024 * 1024,
            timeout_s=30.0,
        )
        # Verify handle_audio_request was called with an UploadFile
        call_kwargs = mock_handle.call_args.kwargs
        assert isinstance(call_kwargs["file"], UploadFile)
        assert call_kwargs["model"] == "whisper"

    @pytest.mark.asyncio
    async def test_url_download_error_propagates(self) -> None:
        """AudioDownloadError from download_audio propagates to caller."""
        from macaw.server.routes.transcriptions import create_transcription

        request = _make_mock_request()

        mock_settings = MagicMock()
        mock_settings.stt_download.max_size_bytes = 1024
        mock_settings.stt_download.timeout_s = 10.0

        with (
            patch(
                "macaw.server.audio_downloader.download_audio",
                new_callable=AsyncMock,
                side_effect=AudioDownloadError("HTTP 403 from remote server"),
            ),
            patch(
                "macaw.config.settings.get_settings",
                return_value=mock_settings,
            ),
            pytest.raises(AudioDownloadError, match="403"),
        ):
            await create_transcription(
                request=request,
                file=None,
                cloud_storage_url="https://example.com/private.wav",
                model="whisper",
                **_default_form_params(),
            )


# ---------------------------------------------------------------------------
# Backward compatibility — file upload still works
# ---------------------------------------------------------------------------


class TestEndpointBackwardCompat:
    """File upload path unchanged (backward compat)."""

    @pytest.mark.asyncio
    async def test_file_upload_still_works(self) -> None:
        """File upload without cloud_storage_url works as before."""
        from macaw.server.routes.transcriptions import create_transcription

        request = _make_mock_request()
        file = UploadFile(file=BytesIO(b"audio"), filename="test.wav")

        with patch(
            "macaw.server.routes.transcriptions.handle_audio_request",
            new_callable=AsyncMock,
            return_value={"text": "transcribed"},
        ) as mock_handle:
            result = await create_transcription(
                request=request,
                file=file,
                cloud_storage_url=None,
                model="whisper",
                **_default_form_params(),
            )

        assert result == {"text": "transcribed"}
        # Verify the original file was passed through
        call_kwargs = mock_handle.call_args.kwargs
        assert call_kwargs["file"] is file


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_mock_request() -> MagicMock:
    """Create a mock FastAPI Request with app.state."""
    request = MagicMock()
    request.app.state = MagicMock()
    return request


def _default_form_params() -> dict[str, Any]:
    """Default Form params for calling create_transcription directly.

    When calling the endpoint function directly (not through FastAPI),
    Form() defaults are not resolved. This provides the actual defaults.
    """
    return {
        "language": None,
        "prompt": None,
        "response_format": "json",
        "temperature": 0.0,
        "timestamp_granularities": ["segment"],
        "hot_words": None,
        "diarize": False,
        "max_speakers": None,
        "webhook_url": None,
        "webhook_secret": None,
        "webhook_metadata": None,
        "itn": True,
        "entity_detection": None,
        "additional_formats": None,
    }
