"""Tests for voice changer REST endpoint."""

from __future__ import annotations

from unittest.mock import MagicMock

from httpx import ASGITransport, AsyncClient

from macaw.server.app import create_app

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_app(*, vc_backend: object | None = None) -> object:
    """Create a minimal app for VC endpoint tests."""
    app = create_app()
    app.state.vc_backend = vc_backend
    return app


# ---------------------------------------------------------------------------
# 503 — No VC engine configured
# ---------------------------------------------------------------------------


class TestVoiceChangerNoEngine:
    async def test_returns_503_when_no_engine(self) -> None:
        """Endpoint returns 503 when no VC backend is configured."""
        app = _make_app(vc_backend=None)

        transport = ASGITransport(app=app, raise_app_exceptions=False)  # type: ignore[arg-type]
        async with AsyncClient(transport=transport, base_url="http://test") as client:
            response = await client.post(
                "/v1/speech-to-speech/alice",
                files={"audio": ("test.wav", b"\x00" * 100, "audio/wav")},
            )

        assert response.status_code == 503

    async def test_503_message_mentions_vc_engine(self) -> None:
        """503 response body mentions voice changer engine."""
        app = _make_app(vc_backend=None)

        transport = ASGITransport(app=app, raise_app_exceptions=False)  # type: ignore[arg-type]
        async with AsyncClient(transport=transport, base_url="http://test") as client:
            response = await client.post(
                "/v1/speech-to-speech/bob",
                files={"audio": ("test.wav", b"\x00" * 100, "audio/wav")},
            )

        body = response.json()
        assert "voice changer" in body.get("error", {}).get("message", "").lower()


# ---------------------------------------------------------------------------
# Input validation
# ---------------------------------------------------------------------------


class TestVoiceChangerValidation:
    async def test_empty_audio_rejected(self) -> None:
        """Empty audio file returns 422/400."""
        app = _make_app(vc_backend=MagicMock())

        transport = ASGITransport(app=app, raise_app_exceptions=False)  # type: ignore[arg-type]
        async with AsyncClient(transport=transport, base_url="http://test") as client:
            response = await client.post(
                "/v1/speech-to-speech/alice",
                files={"audio": ("test.wav", b"", "audio/wav")},
            )

        assert response.status_code in (400, 422)

    async def test_missing_audio_file_rejected(self) -> None:
        """Request without audio file returns 422."""
        app = _make_app(vc_backend=MagicMock())

        transport = ASGITransport(app=app, raise_app_exceptions=False)  # type: ignore[arg-type]
        async with AsyncClient(transport=transport, base_url="http://test") as client:
            response = await client.post("/v1/speech-to-speech/alice")

        assert response.status_code == 422

    async def test_invalid_output_format_rejected(self) -> None:
        """Invalid output_format returns 400."""
        app = _make_app(vc_backend=MagicMock())

        transport = ASGITransport(app=app, raise_app_exceptions=False)  # type: ignore[arg-type]
        async with AsyncClient(transport=transport, base_url="http://test") as client:
            response = await client.post(
                "/v1/speech-to-speech/alice",
                files={"audio": ("test.wav", b"\x00" * 100, "audio/wav")},
                data={"output_format": "invalid_codec"},
            )

        assert response.status_code in (400, 422)


# ---------------------------------------------------------------------------
# Endpoint contract with mock backend
# ---------------------------------------------------------------------------


class TestVoiceChangerWithBackend:
    async def test_accepts_valid_request(self) -> None:
        """Valid request with VC backend returns 200."""
        app = _make_app(vc_backend=MagicMock())

        transport = ASGITransport(app=app, raise_app_exceptions=False)  # type: ignore[arg-type]
        async with AsyncClient(transport=transport, base_url="http://test") as client:
            response = await client.post(
                "/v1/speech-to-speech/alice",
                files={"audio": ("test.wav", b"\x00" * 100, "audio/wav")},
            )

        assert response.status_code == 200

    async def test_wav_response_has_header(self) -> None:
        """WAV output format returns response with RIFF header."""
        app = _make_app(vc_backend=MagicMock())

        transport = ASGITransport(app=app, raise_app_exceptions=False)  # type: ignore[arg-type]
        async with AsyncClient(transport=transport, base_url="http://test") as client:
            response = await client.post(
                "/v1/speech-to-speech/alice",
                files={"audio": ("test.wav", b"\x00" * 100, "audio/wav")},
                data={"output_format": "wav"},
            )

        assert response.status_code == 200
        assert response.content[:4] == b"RIFF"

    async def test_voice_id_in_path(self) -> None:
        """Voice ID is extracted from URL path."""
        app = _make_app(vc_backend=MagicMock())

        transport = ASGITransport(app=app, raise_app_exceptions=False)  # type: ignore[arg-type]
        async with AsyncClient(transport=transport, base_url="http://test") as client:
            response = await client.post(
                "/v1/speech-to-speech/my-custom-voice",
                files={"audio": ("test.wav", b"\x00" * 100, "audio/wav")},
            )

        assert response.status_code == 200


# ---------------------------------------------------------------------------
# Route model tests
# ---------------------------------------------------------------------------


class TestVoiceChangerModels:
    def test_response_model(self) -> None:
        from macaw.server.models.voice_changer import VoiceChangerResponse

        resp = VoiceChangerResponse(
            request_id="req-1",
            voice_id="alice",
            output_format="wav",
        )
        assert resp.request_id == "req-1"
        assert resp.voice_id == "alice"

    def test_form_params_defaults(self) -> None:
        from macaw.server.models.voice_changer import VoiceChangerFormParams

        params = VoiceChangerFormParams()
        assert params.output_format == "wav"
        assert params.voice_settings is None
