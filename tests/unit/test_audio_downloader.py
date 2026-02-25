"""Tests for the async audio downloader (Sprint M, Task M1).

Validates:
 - HTTPS-only URL validation
 - SSRF prevention (private/loopback/reserved IPs rejected)
 - Download with size limit enforcement
 - Timeout handling
 - HTTP error mapping
 - Content-Length early rejection
 - Streaming download with chunked accumulation
"""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock, patch

import httpx
import pytest

from macaw.exceptions import AudioDownloadError, InvalidRequestError
from macaw.server.audio_downloader import (
    _validate_hostname_not_private,
    _validate_url,
    download_audio,
)

# ---------------------------------------------------------------------------
# URL validation
# ---------------------------------------------------------------------------


class TestValidateUrl:
    """URL scheme and format validation."""

    def test_https_url_accepted(self) -> None:
        result = _validate_url("https://example.com/audio.wav")
        assert result == "https://example.com/audio.wav"

    def test_http_url_rejected(self) -> None:
        with pytest.raises(InvalidRequestError, match="Only HTTPS"):
            _validate_url("http://example.com/audio.wav")

    def test_ftp_url_rejected(self) -> None:
        with pytest.raises(InvalidRequestError, match="Only HTTPS"):
            _validate_url("ftp://example.com/audio.wav")

    def test_file_url_rejected(self) -> None:
        with pytest.raises(InvalidRequestError, match="Only HTTPS"):
            _validate_url("file:///etc/passwd")

    def test_empty_scheme_rejected(self) -> None:
        with pytest.raises(InvalidRequestError):
            _validate_url("://example.com/audio.wav")

    def test_missing_hostname_rejected(self) -> None:
        with pytest.raises(InvalidRequestError, match="missing hostname"):
            _validate_url("https:///audio.wav")


# ---------------------------------------------------------------------------
# SSRF prevention — hostname validation
# ---------------------------------------------------------------------------


class TestValidateHostnameNotPrivate:
    """Private/loopback/reserved IP address rejection."""

    def test_public_hostname_allowed(self) -> None:
        _validate_hostname_not_private("s3.amazonaws.com")

    def test_public_ip_allowed(self) -> None:
        _validate_hostname_not_private("8.8.8.8")

    def test_loopback_ipv4_rejected(self) -> None:
        with pytest.raises(InvalidRequestError, match="private/reserved"):
            _validate_hostname_not_private("127.0.0.1")

    def test_loopback_ipv6_rejected(self) -> None:
        with pytest.raises(InvalidRequestError, match="private/reserved"):
            _validate_hostname_not_private("::1")

    def test_private_10_x_rejected(self) -> None:
        with pytest.raises(InvalidRequestError, match="private/reserved"):
            _validate_hostname_not_private("10.0.0.1")

    def test_private_172_16_rejected(self) -> None:
        with pytest.raises(InvalidRequestError, match="private/reserved"):
            _validate_hostname_not_private("172.16.0.1")

    def test_private_192_168_rejected(self) -> None:
        with pytest.raises(InvalidRequestError, match="private/reserved"):
            _validate_hostname_not_private("192.168.1.1")

    def test_link_local_rejected(self) -> None:
        with pytest.raises(InvalidRequestError, match="private/reserved"):
            _validate_hostname_not_private("169.254.0.1")

    def test_ipv6_link_local_rejected(self) -> None:
        with pytest.raises(InvalidRequestError, match="private/reserved"):
            _validate_hostname_not_private("fe80::1")

    def test_regular_hostname_allowed(self) -> None:
        _validate_hostname_not_private("storage.googleapis.com")


# ---------------------------------------------------------------------------
# download_audio — success path
# ---------------------------------------------------------------------------


class TestDownloadAudioSuccess:
    """Successful download scenarios."""

    @pytest.mark.asyncio
    async def test_downloads_audio_bytes(self) -> None:
        audio_data = b"RIFF" + b"\x00" * 100

        mock_response = AsyncMock()
        mock_response.headers = {}
        mock_response.raise_for_status = MagicMock()
        mock_response.aiter_bytes = _make_aiter([audio_data])

        with patch("macaw.server.audio_downloader.httpx.AsyncClient") as mock_client_cls:
            _setup_mock_client(mock_client_cls, mock_response)

            result = await download_audio(
                "https://example.com/audio.wav",
                max_size_bytes=1024 * 1024,
                timeout_s=30.0,
            )

        assert result == audio_data

    @pytest.mark.asyncio
    async def test_multi_chunk_download(self) -> None:
        chunks = [b"chunk1", b"chunk2", b"chunk3"]

        mock_response = AsyncMock()
        mock_response.headers = {}
        mock_response.raise_for_status = MagicMock()
        mock_response.aiter_bytes = _make_aiter(chunks)

        with patch("macaw.server.audio_downloader.httpx.AsyncClient") as mock_client_cls:
            _setup_mock_client(mock_client_cls, mock_response)

            result = await download_audio(
                "https://example.com/audio.wav",
                max_size_bytes=1024 * 1024,
                timeout_s=30.0,
            )

        assert result == b"chunk1chunk2chunk3"


# ---------------------------------------------------------------------------
# download_audio — validation errors
# ---------------------------------------------------------------------------


class TestDownloadAudioValidation:
    """URL validation errors in download_audio."""

    @pytest.mark.asyncio
    async def test_rejects_http_url(self) -> None:
        with pytest.raises(InvalidRequestError, match="Only HTTPS"):
            await download_audio(
                "http://example.com/audio.wav",
                max_size_bytes=1024,
                timeout_s=10.0,
            )

    @pytest.mark.asyncio
    async def test_rejects_private_ip(self) -> None:
        with pytest.raises(InvalidRequestError, match="private/reserved"):
            await download_audio(
                "https://192.168.1.1/audio.wav",
                max_size_bytes=1024,
                timeout_s=10.0,
            )


# ---------------------------------------------------------------------------
# download_audio — size limit enforcement
# ---------------------------------------------------------------------------


class TestDownloadAudioSizeLimit:
    """Size limit enforcement via Content-Length and streaming."""

    @pytest.mark.asyncio
    async def test_rejects_via_content_length_header(self) -> None:
        mock_response = AsyncMock()
        mock_response.headers = {"content-length": "2000"}
        mock_response.raise_for_status = MagicMock()

        with patch("macaw.server.audio_downloader.httpx.AsyncClient") as mock_client_cls:
            _setup_mock_client(mock_client_cls, mock_response)

            with pytest.raises(AudioDownloadError, match="Remote file too large"):
                await download_audio(
                    "https://example.com/large.wav",
                    max_size_bytes=1000,
                    timeout_s=30.0,
                )

    @pytest.mark.asyncio
    async def test_rejects_via_streaming_size_guard(self) -> None:
        chunks = [b"x" * 600, b"x" * 600]  # 1200 total, limit 1000

        mock_response = AsyncMock()
        mock_response.headers = {}  # No Content-Length
        mock_response.raise_for_status = MagicMock()
        mock_response.aiter_bytes = _make_aiter(chunks)

        with patch("macaw.server.audio_downloader.httpx.AsyncClient") as mock_client_cls:
            _setup_mock_client(mock_client_cls, mock_response)

            with pytest.raises(AudioDownloadError, match="exceeds size limit"):
                await download_audio(
                    "https://example.com/large.wav",
                    max_size_bytes=1000,
                    timeout_s=30.0,
                )

    @pytest.mark.asyncio
    async def test_invalid_content_length_ignored(self) -> None:
        """Non-numeric Content-Length doesn't cause a crash."""
        mock_response = AsyncMock()
        mock_response.headers = {"content-length": "not-a-number"}
        mock_response.raise_for_status = MagicMock()
        mock_response.aiter_bytes = _make_aiter([b"audio"])

        with patch("macaw.server.audio_downloader.httpx.AsyncClient") as mock_client_cls:
            _setup_mock_client(mock_client_cls, mock_response)

            result = await download_audio(
                "https://example.com/audio.wav",
                max_size_bytes=1024,
                timeout_s=30.0,
            )

        assert result == b"audio"


# ---------------------------------------------------------------------------
# download_audio — error handling
# ---------------------------------------------------------------------------


class TestDownloadAudioErrors:
    """Network error, timeout, and HTTP error handling."""

    @pytest.mark.asyncio
    async def test_timeout_raises_download_error(self) -> None:
        with patch("macaw.server.audio_downloader.httpx.AsyncClient") as mock_client_cls:
            mock_client = AsyncMock()
            mock_client.__aenter__ = AsyncMock(return_value=mock_client)
            mock_client.__aexit__ = AsyncMock(return_value=False)
            mock_client.stream = MagicMock(side_effect=httpx.TimeoutException("timed out"))
            mock_client_cls.return_value = mock_client

            with pytest.raises(AudioDownloadError, match="timed out"):
                await download_audio(
                    "https://example.com/audio.wav",
                    max_size_bytes=1024,
                    timeout_s=5.0,
                )

    @pytest.mark.asyncio
    async def test_http_status_error_raises_download_error(self) -> None:
        mock_response = AsyncMock()
        mock_response.headers = {}
        mock_response.status_code = 404
        mock_response.raise_for_status = MagicMock(
            side_effect=httpx.HTTPStatusError(
                "Not Found",
                request=httpx.Request("GET", "https://example.com/audio.wav"),
                response=httpx.Response(404),
            )
        )

        with patch("macaw.server.audio_downloader.httpx.AsyncClient") as mock_client_cls:
            _setup_mock_client(mock_client_cls, mock_response)

            with pytest.raises(AudioDownloadError, match="HTTP 404"):
                await download_audio(
                    "https://example.com/audio.wav",
                    max_size_bytes=1024,
                    timeout_s=30.0,
                )

    @pytest.mark.asyncio
    async def test_network_error_raises_download_error(self) -> None:
        with patch("macaw.server.audio_downloader.httpx.AsyncClient") as mock_client_cls:
            mock_client = AsyncMock()
            mock_client.__aenter__ = AsyncMock(return_value=mock_client)
            mock_client.__aexit__ = AsyncMock(return_value=False)
            mock_client.stream = MagicMock(side_effect=httpx.ConnectError("Connection refused"))
            mock_client_cls.return_value = mock_client

            with pytest.raises(AudioDownloadError, match="Network error"):
                await download_audio(
                    "https://example.com/audio.wav",
                    max_size_bytes=1024,
                    timeout_s=30.0,
                )

    @pytest.mark.asyncio
    async def test_too_many_redirects_raises_download_error(self) -> None:
        with patch("macaw.server.audio_downloader.httpx.AsyncClient") as mock_client_cls:
            mock_client = AsyncMock()
            mock_client.__aenter__ = AsyncMock(return_value=mock_client)
            mock_client.__aexit__ = AsyncMock(return_value=False)
            mock_client.stream = MagicMock(
                side_effect=httpx.TooManyRedirects(
                    "too many redirects",
                    request=httpx.Request("GET", "https://example.com/audio.wav"),
                )
            )
            mock_client_cls.return_value = mock_client

            with pytest.raises(AudioDownloadError, match="Too many redirects"):
                await download_audio(
                    "https://example.com/audio.wav",
                    max_size_bytes=1024,
                    timeout_s=30.0,
                )


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_aiter(chunks: list[bytes]):  # type: ignore[no-untyped-def]
    """Create a callable that returns an async iterator over chunks."""

    async def _aiter(**_kwargs: object):  # type: ignore[no-untyped-def]
        for chunk in chunks:
            yield chunk

    return _aiter


def _setup_mock_client(mock_client_cls: MagicMock, mock_response: AsyncMock) -> None:
    """Wire up mock httpx.AsyncClient with a stream context manager."""
    mock_stream_ctx = AsyncMock()
    mock_stream_ctx.__aenter__ = AsyncMock(return_value=mock_response)
    mock_stream_ctx.__aexit__ = AsyncMock(return_value=False)

    mock_client = AsyncMock()
    mock_client.__aenter__ = AsyncMock(return_value=mock_client)
    mock_client.__aexit__ = AsyncMock(return_value=False)
    mock_client.stream = MagicMock(return_value=mock_stream_ctx)

    mock_client_cls.return_value = mock_client
