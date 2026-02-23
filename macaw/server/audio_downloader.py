"""Async audio downloader for cloud_storage_url transcription.

Downloads audio from HTTPS URLs (S3 pre-signed, GCS, any HTTPS) with:
- HTTPS-only enforcement (security)
- SSRF prevention (rejects private/loopback IP literals)
- Streaming download with size limit enforcement
- Configurable timeout
"""

from __future__ import annotations

import ipaddress
from urllib.parse import urlparse

import httpx

from macaw.exceptions import AudioDownloadError, InvalidRequestError
from macaw.logging import get_logger

logger = get_logger("server.audio_downloader")

_DOWNLOAD_CHUNK_SIZE = 64 * 1024  # 64 KB chunks
_MAX_REDIRECTS = 5


def _validate_url(url: str) -> str:
    """Validate URL scheme and hostname for security.

    Returns the validated URL unchanged.

    Raises:
        InvalidRequestError: If URL scheme is not HTTPS or hostname is a
            private/loopback IP address.
    """
    parsed = urlparse(url)

    if parsed.scheme != "https":
        raise InvalidRequestError(f"Only HTTPS URLs are supported, got '{parsed.scheme}://'")

    if not parsed.hostname:
        raise InvalidRequestError("Invalid URL: missing hostname")

    _validate_hostname_not_private(parsed.hostname)

    return url


def _validate_hostname_not_private(hostname: str) -> None:
    """Reject private/loopback/reserved IP address literals (SSRF prevention).

    Non-IP hostnames (e.g., ``s3.amazonaws.com``) are allowed through — DNS
    resolution is handled by httpx at connection time.

    Raises:
        InvalidRequestError: If hostname is a private, loopback, reserved,
            or link-local IP address.
    """
    try:
        addr = ipaddress.ip_address(hostname)
    except ValueError:
        # Not an IP literal (regular hostname) — allow
        return

    if addr.is_private or addr.is_loopback or addr.is_reserved or addr.is_link_local:
        raise InvalidRequestError(
            f"URL points to a private/reserved address ({hostname}). "
            "Only public HTTPS URLs are allowed."
        )


async def download_audio(
    url: str,
    *,
    max_size_bytes: int,
    timeout_s: float,
) -> bytes:
    """Download audio from an HTTPS URL.

    Streams the response body in chunks, enforcing a maximum size limit.
    Validates the URL scheme (HTTPS-only) and rejects private IP addresses
    to prevent SSRF attacks.

    Args:
        url: The HTTPS URL to download from.
        max_size_bytes: Maximum allowed download size in bytes.
        timeout_s: Total timeout for the download in seconds.

    Returns:
        Raw audio bytes.

    Raises:
        InvalidRequestError: If the URL is invalid (non-HTTPS, private IP).
        AudioDownloadError: If the download fails (timeout, HTTP error,
            network error, or size limit exceeded).
    """
    _validate_url(url)

    logger.info("audio_download_start", url=url, max_size_bytes=max_size_bytes)

    try:
        async with (
            httpx.AsyncClient(
                follow_redirects=True,
                max_redirects=_MAX_REDIRECTS,
                timeout=httpx.Timeout(timeout_s),
            ) as client,
            client.stream("GET", url) as response,
        ):
            response.raise_for_status()

            # Early rejection via Content-Length header (avoids wasting bandwidth)
            content_length = response.headers.get("content-length")
            if content_length is not None:
                try:
                    declared_size = int(content_length)
                except ValueError:
                    declared_size = 0
                if declared_size > max_size_bytes:
                    raise AudioDownloadError(
                        f"Remote file too large ({declared_size} bytes, max {max_size_bytes})"
                    )

            # Stream download with size guard
            chunks: list[bytes] = []
            total = 0
            async for chunk in response.aiter_bytes(chunk_size=_DOWNLOAD_CHUNK_SIZE):
                total += len(chunk)
                if total > max_size_bytes:
                    raise AudioDownloadError(
                        f"Download exceeds size limit of {max_size_bytes} bytes"
                    )
                chunks.append(chunk)

    except AudioDownloadError:
        raise
    except httpx.TimeoutException as exc:
        raise AudioDownloadError(f"Download timed out after {timeout_s}s") from exc
    except httpx.HTTPStatusError as exc:
        raise AudioDownloadError(f"HTTP {exc.response.status_code} from remote server") from exc
    except httpx.TooManyRedirects as exc:
        raise AudioDownloadError(f"Too many redirects (max {_MAX_REDIRECTS})") from exc
    except httpx.RequestError as exc:
        raise AudioDownloadError(f"Network error: {exc}") from exc

    audio_data = b"".join(chunks)

    logger.info("audio_download_complete", url=url, size_bytes=len(audio_data))

    return audio_data
