"""Tests for POST /v1/audio/align endpoint.

Validates:
- Successful word and character alignment
- Input validation (empty text, empty audio)
- 503 when torchaudio is not available
- Audio duration calculation
- Default parameters (language, granularity)
"""

from __future__ import annotations

import io
import struct
import wave
from dataclasses import dataclass
from typing import TYPE_CHECKING
from unittest.mock import AsyncMock

import httpx

from macaw.server.app import create_app

if TYPE_CHECKING:
    from fastapi import FastAPI

# --- Helpers ---


def _make_wav(sample_rate: int = 16000, duration_s: float = 1.0) -> bytes:
    """Generate a minimal WAV file with a constant-value tone."""
    n_samples = int(sample_rate * duration_s)
    buf = io.BytesIO()
    with wave.open(buf, "wb") as f:
        f.setnchannels(1)
        f.setsampwidth(2)  # 16-bit
        f.setframerate(sample_rate)
        f.writeframes(struct.pack(f"<{n_samples}h", *([1000] * n_samples)))
    return buf.getvalue()


@dataclass(frozen=True)
class _FakeAlignmentItem:
    """Mimics TTSAlignmentItem for test assertions."""

    text: str
    start_ms: int
    duration_ms: int


def _make_mock_aligner(
    items: tuple[_FakeAlignmentItem, ...] | None = None,
) -> AsyncMock:
    """Create an AsyncMock aligner that returns the given items."""
    if items is None:
        items = (
            _FakeAlignmentItem(text="Hello", start_ms=0, duration_ms=450),
            _FakeAlignmentItem(text="world", start_ms=500, duration_ms=500),
        )
    aligner = AsyncMock()
    aligner.align = AsyncMock(return_value=items)
    return aligner


def _make_app(*, aligner: AsyncMock | None = None) -> FastAPI:
    """Create a test app with optional aligner on app.state."""
    app = create_app()
    if aligner is not None:
        app.state.aligner = aligner
    return app


# --- Success Cases ---


class TestAlignWordItems:
    async def test_align_word_items(self) -> None:
        """POST with valid WAV + text returns alignment items."""
        items = (
            _FakeAlignmentItem(text="Hello", start_ms=0, duration_ms=450),
            _FakeAlignmentItem(text="world", start_ms=500, duration_ms=500),
        )
        aligner = _make_mock_aligner(items)
        app = _make_app(aligner=aligner)

        async with httpx.AsyncClient(
            transport=httpx.ASGITransport(app=app),
            base_url="http://test",
        ) as client:
            wav_bytes = _make_wav()
            response = await client.post(
                "/v1/audio/align",
                files={"file": ("test.wav", wav_bytes, "audio/wav")},
                data={"text": "Hello world"},
            )

        assert response.status_code == 200
        body = response.json()
        assert len(body["items"]) == 2
        assert body["items"][0]["text"] == "Hello"
        assert body["items"][0]["start_ms"] == 0
        assert body["items"][0]["duration_ms"] == 450
        assert body["items"][1]["text"] == "world"
        assert body["items"][1]["start_ms"] == 500
        assert body["items"][1]["duration_ms"] == 500

        aligner.align.assert_awaited_once()
        call_kwargs = aligner.align.call_args
        assert call_kwargs.kwargs["text"] == "Hello world"
        assert call_kwargs.kwargs["language"] == "en"
        assert call_kwargs.kwargs["granularity"] == "word"


class TestAlignCharacterGranularity:
    async def test_granularity_forwarded_to_aligner(self) -> None:
        """granularity=character is forwarded to the aligner."""
        items = (
            _FakeAlignmentItem(text="H", start_ms=0, duration_ms=100),
            _FakeAlignmentItem(text="i", start_ms=100, duration_ms=100),
        )
        aligner = _make_mock_aligner(items)
        app = _make_app(aligner=aligner)

        async with httpx.AsyncClient(
            transport=httpx.ASGITransport(app=app),
            base_url="http://test",
        ) as client:
            wav_bytes = _make_wav()
            response = await client.post(
                "/v1/audio/align",
                files={"file": ("test.wav", wav_bytes, "audio/wav")},
                data={"text": "Hi", "granularity": "character"},
            )

        assert response.status_code == 200
        body = response.json()
        assert body["granularity"] == "character"
        assert len(body["items"]) == 2

        call_kwargs = aligner.align.call_args
        assert call_kwargs.kwargs["granularity"] == "character"


# --- Validation Errors ---


class TestAlignEmptyText:
    async def test_empty_text_returns_400(self) -> None:
        """Empty text returns 400."""
        aligner = _make_mock_aligner()
        app = _make_app(aligner=aligner)

        async with httpx.AsyncClient(
            transport=httpx.ASGITransport(app=app, raise_app_exceptions=False),
            base_url="http://test",
        ) as client:
            wav_bytes = _make_wav()
            response = await client.post(
                "/v1/audio/align",
                files={"file": ("test.wav", wav_bytes, "audio/wav")},
                data={"text": ""},
            )

        assert response.status_code == 400

    async def test_whitespace_only_text_returns_400(self) -> None:
        """Whitespace-only text returns 400."""
        aligner = _make_mock_aligner()
        app = _make_app(aligner=aligner)

        async with httpx.AsyncClient(
            transport=httpx.ASGITransport(app=app, raise_app_exceptions=False),
            base_url="http://test",
        ) as client:
            wav_bytes = _make_wav()
            response = await client.post(
                "/v1/audio/align",
                files={"file": ("test.wav", wav_bytes, "audio/wav")},
                data={"text": "   "},
            )

        assert response.status_code == 400


class TestAlignEmptyAudio:
    async def test_empty_audio_returns_400(self) -> None:
        """Empty audio file returns 400."""
        aligner = _make_mock_aligner()
        app = _make_app(aligner=aligner)

        async with httpx.AsyncClient(
            transport=httpx.ASGITransport(app=app, raise_app_exceptions=False),
            base_url="http://test",
        ) as client:
            response = await client.post(
                "/v1/audio/align",
                files={"file": ("test.wav", b"", "audio/wav")},
                data={"text": "Hello world"},
            )

        assert response.status_code == 400


# --- Service Unavailable ---


class TestAlignNoTorchaudio:
    async def test_no_torchaudio_returns_503(self) -> None:
        """When aligner is None (torchaudio not installed), returns 503."""
        from unittest.mock import patch

        app = create_app()

        # Patch create_aligner to return None (simulates torchaudio missing).
        # The route does lazy-init via `from macaw.alignment import create_aligner`,
        # so we patch the canonical location.
        with patch("macaw.alignment.create_aligner", return_value=None):
            async with httpx.AsyncClient(
                transport=httpx.ASGITransport(app=app, raise_app_exceptions=False),
                base_url="http://test",
            ) as client:
                wav_bytes = _make_wav()
                response = await client.post(
                    "/v1/audio/align",
                    files={"file": ("test.wav", wav_bytes, "audio/wav")},
                    data={"text": "Hello"},
                )

        assert response.status_code == 503
        body = response.json()
        assert "torchaudio" in body["error"]["message"]


# --- Metadata ---


class TestAlignResponseDuration:
    async def test_duration_calculated_from_pcm(self) -> None:
        """audio_duration_ms is calculated from PCM byte length."""
        aligner = _make_mock_aligner(items=())
        app = _make_app(aligner=aligner)

        # 1 second at 16kHz, 16-bit = 32000 bytes PCM
        wav_bytes = _make_wav(sample_rate=16000, duration_s=1.0)

        async with httpx.AsyncClient(
            transport=httpx.ASGITransport(app=app),
            base_url="http://test",
        ) as client:
            response = await client.post(
                "/v1/audio/align",
                files={"file": ("test.wav", wav_bytes, "audio/wav")},
                data={"text": "Hello"},
            )

        assert response.status_code == 200
        body = response.json()
        assert body["audio_duration_ms"] == 1000

    async def test_duration_half_second(self) -> None:
        """audio_duration_ms is correct for 0.5s audio."""
        aligner = _make_mock_aligner(items=())
        app = _make_app(aligner=aligner)

        wav_bytes = _make_wav(sample_rate=16000, duration_s=0.5)

        async with httpx.AsyncClient(
            transport=httpx.ASGITransport(app=app),
            base_url="http://test",
        ) as client:
            response = await client.post(
                "/v1/audio/align",
                files={"file": ("test.wav", wav_bytes, "audio/wav")},
                data={"text": "Hello"},
            )

        assert response.status_code == 200
        body = response.json()
        assert body["audio_duration_ms"] == 500


class TestAlignDefaultLanguage:
    async def test_default_language_is_en(self) -> None:
        """Default language is 'en'."""
        aligner = _make_mock_aligner()
        app = _make_app(aligner=aligner)

        async with httpx.AsyncClient(
            transport=httpx.ASGITransport(app=app),
            base_url="http://test",
        ) as client:
            wav_bytes = _make_wav()
            response = await client.post(
                "/v1/audio/align",
                files={"file": ("test.wav", wav_bytes, "audio/wav")},
                data={"text": "Hello"},
            )

        assert response.status_code == 200
        body = response.json()
        assert body["language"] == "en"

        call_kwargs = aligner.align.call_args
        assert call_kwargs.kwargs["language"] == "en"

    async def test_custom_language_forwarded(self) -> None:
        """Custom language is forwarded to the aligner."""
        aligner = _make_mock_aligner()
        app = _make_app(aligner=aligner)

        async with httpx.AsyncClient(
            transport=httpx.ASGITransport(app=app),
            base_url="http://test",
        ) as client:
            wav_bytes = _make_wav()
            response = await client.post(
                "/v1/audio/align",
                files={"file": ("test.wav", wav_bytes, "audio/wav")},
                data={"text": "Ola", "language": "pt"},
            )

        assert response.status_code == 200
        body = response.json()
        assert body["language"] == "pt"

        call_kwargs = aligner.align.call_args
        assert call_kwargs.kwargs["language"] == "pt"


class TestAlignDefaultGranularity:
    async def test_default_granularity_is_word(self) -> None:
        """Default granularity is 'word'."""
        aligner = _make_mock_aligner()
        app = _make_app(aligner=aligner)

        async with httpx.AsyncClient(
            transport=httpx.ASGITransport(app=app),
            base_url="http://test",
        ) as client:
            wav_bytes = _make_wav()
            response = await client.post(
                "/v1/audio/align",
                files={"file": ("test.wav", wav_bytes, "audio/wav")},
                data={"text": "Hello"},
            )

        assert response.status_code == 200
        body = response.json()
        assert body["granularity"] == "word"


class TestAlignRawPcm:
    async def test_raw_pcm_treated_as_16khz(self) -> None:
        """Non-WAV data is treated as raw PCM 16kHz 16-bit mono."""
        aligner = _make_mock_aligner(items=())
        app = _make_app(aligner=aligner)

        # 0.5s of raw PCM at 16kHz = 16000 bytes
        raw_pcm = b"\x00\x01" * 8000

        async with httpx.AsyncClient(
            transport=httpx.ASGITransport(app=app),
            base_url="http://test",
        ) as client:
            response = await client.post(
                "/v1/audio/align",
                files={"file": ("audio.pcm", raw_pcm, "audio/pcm")},
                data={"text": "Hello"},
            )

        assert response.status_code == 200
        body = response.json()
        assert body["audio_duration_ms"] == 500

        call_kwargs = aligner.align.call_args
        assert call_kwargs.kwargs["sample_rate"] == 16000


class TestAlignLazyInit:
    async def test_lazy_init_aligner_on_first_request(self) -> None:
        """Aligner is lazy-initialized on first request if not set."""
        from unittest.mock import patch

        mock_aligner = _make_mock_aligner()
        app = create_app()
        # Do NOT set app.state.aligner -- force lazy init path.

        with patch(
            "macaw.alignment.create_aligner",
            return_value=mock_aligner,
        ):
            async with httpx.AsyncClient(
                transport=httpx.ASGITransport(app=app),
                base_url="http://test",
            ) as client:
                wav_bytes = _make_wav()
                response = await client.post(
                    "/v1/audio/align",
                    files={"file": ("test.wav", wav_bytes, "audio/wav")},
                    data={"text": "Hello"},
                )

        assert response.status_code == 200
        # Aligner should now be cached on app state.
        assert app.state.aligner is mock_aligner


class TestAlignResponseShape:
    async def test_response_has_all_fields(self) -> None:
        """Response includes items, language, granularity, audio_duration_ms."""
        aligner = _make_mock_aligner()
        app = _make_app(aligner=aligner)

        async with httpx.AsyncClient(
            transport=httpx.ASGITransport(app=app),
            base_url="http://test",
        ) as client:
            wav_bytes = _make_wav()
            response = await client.post(
                "/v1/audio/align",
                files={"file": ("test.wav", wav_bytes, "audio/wav")},
                data={"text": "Hello world"},
            )

        assert response.status_code == 200
        body = response.json()
        assert "items" in body
        assert "language" in body
        assert "granularity" in body
        assert "audio_duration_ms" in body
        assert isinstance(body["items"], list)
        assert isinstance(body["audio_duration_ms"], int)
