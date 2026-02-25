"""Tests for sound effects generation REST endpoint."""

from __future__ import annotations

from unittest.mock import MagicMock

from httpx import ASGITransport, AsyncClient

from macaw.server.app import create_app

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_app(*, sfx_backend: object | None = None) -> object:
    """Create a minimal app for SFX endpoint tests."""
    app = create_app()
    app.state.sfx_backend = sfx_backend
    return app


# ---------------------------------------------------------------------------
# 503 — No SFX engine configured
# ---------------------------------------------------------------------------


class TestSoundEffectsNoEngine:
    async def test_returns_503_when_no_engine(self) -> None:
        """Endpoint returns 503 when no SFX backend is configured."""
        app = _make_app(sfx_backend=None)

        transport = ASGITransport(app=app, raise_app_exceptions=False)  # type: ignore[arg-type]
        async with AsyncClient(transport=transport, base_url="http://test") as client:
            response = await client.post(
                "/v1/sound-generation",
                json={"text": "rain on a tin roof"},
            )

        assert response.status_code == 503

    async def test_503_message_mentions_sound_effect(self) -> None:
        """503 response body mentions sound effect engine."""
        app = _make_app(sfx_backend=None)

        transport = ASGITransport(app=app, raise_app_exceptions=False)  # type: ignore[arg-type]
        async with AsyncClient(transport=transport, base_url="http://test") as client:
            response = await client.post(
                "/v1/sound-generation",
                json={"text": "thunder"},
            )

        body = response.json()
        assert "sound effect" in body.get("error", {}).get("message", "").lower()


# ---------------------------------------------------------------------------
# Input validation
# ---------------------------------------------------------------------------


class TestSoundEffectsValidation:
    async def test_empty_text_rejected(self) -> None:
        """Empty text returns 422."""
        app = _make_app(sfx_backend=MagicMock())

        transport = ASGITransport(app=app, raise_app_exceptions=False)  # type: ignore[arg-type]
        async with AsyncClient(transport=transport, base_url="http://test") as client:
            response = await client.post(
                "/v1/sound-generation",
                json={"text": ""},
            )

        assert response.status_code == 422

    async def test_missing_text_rejected(self) -> None:
        """Request without text returns 422."""
        app = _make_app(sfx_backend=MagicMock())

        transport = ASGITransport(app=app, raise_app_exceptions=False)  # type: ignore[arg-type]
        async with AsyncClient(transport=transport, base_url="http://test") as client:
            response = await client.post(
                "/v1/sound-generation",
                json={},
            )

        assert response.status_code == 422

    async def test_duration_too_short_rejected(self) -> None:
        """Duration below 0.5s returns 422."""
        app = _make_app(sfx_backend=MagicMock())

        transport = ASGITransport(app=app, raise_app_exceptions=False)  # type: ignore[arg-type]
        async with AsyncClient(transport=transport, base_url="http://test") as client:
            response = await client.post(
                "/v1/sound-generation",
                json={"text": "rain", "duration_seconds": 0.1},
            )

        assert response.status_code == 422

    async def test_duration_too_long_rejected(self) -> None:
        """Duration above 30s returns 422."""
        app = _make_app(sfx_backend=MagicMock())

        transport = ASGITransport(app=app, raise_app_exceptions=False)  # type: ignore[arg-type]
        async with AsyncClient(transport=transport, base_url="http://test") as client:
            response = await client.post(
                "/v1/sound-generation",
                json={"text": "rain", "duration_seconds": 60.0},
            )

        assert response.status_code == 422

    async def test_invalid_prompt_influence_rejected(self) -> None:
        """prompt_influence outside 0-1 range returns 422."""
        app = _make_app(sfx_backend=MagicMock())

        transport = ASGITransport(app=app, raise_app_exceptions=False)  # type: ignore[arg-type]
        async with AsyncClient(transport=transport, base_url="http://test") as client:
            response = await client.post(
                "/v1/sound-generation",
                json={"text": "rain", "prompt_influence": 1.5},
            )

        assert response.status_code == 422

    async def test_invalid_output_format_rejected(self) -> None:
        """Invalid output_format returns 400."""
        app = _make_app(sfx_backend=MagicMock())

        transport = ASGITransport(app=app, raise_app_exceptions=False)  # type: ignore[arg-type]
        async with AsyncClient(transport=transport, base_url="http://test") as client:
            response = await client.post(
                "/v1/sound-generation",
                json={"text": "rain", "output_format": "invalid_codec"},
            )

        assert response.status_code in (400, 422)


# ---------------------------------------------------------------------------
# Endpoint contract with mock backend
# ---------------------------------------------------------------------------


class TestSoundEffectsWithBackend:
    async def test_accepts_valid_request(self) -> None:
        """Valid request with SFX backend returns 200."""
        app = _make_app(sfx_backend=MagicMock())

        transport = ASGITransport(app=app, raise_app_exceptions=False)  # type: ignore[arg-type]
        async with AsyncClient(transport=transport, base_url="http://test") as client:
            response = await client.post(
                "/v1/sound-generation",
                json={"text": "rain on a tin roof"},
            )

        assert response.status_code == 200

    async def test_wav_response_has_header(self) -> None:
        """WAV output format returns response with RIFF header."""
        app = _make_app(sfx_backend=MagicMock())

        transport = ASGITransport(app=app, raise_app_exceptions=False)  # type: ignore[arg-type]
        async with AsyncClient(transport=transport, base_url="http://test") as client:
            response = await client.post(
                "/v1/sound-generation",
                json={"text": "rain", "output_format": "wav"},
            )

        assert response.status_code == 200
        assert response.content[:4] == b"RIFF"

    async def test_custom_duration(self) -> None:
        """Custom duration is accepted."""
        app = _make_app(sfx_backend=MagicMock())

        transport = ASGITransport(app=app, raise_app_exceptions=False)  # type: ignore[arg-type]
        async with AsyncClient(transport=transport, base_url="http://test") as client:
            response = await client.post(
                "/v1/sound-generation",
                json={"text": "explosion", "duration_seconds": 10.0},
            )

        assert response.status_code == 200

    async def test_loop_parameter(self) -> None:
        """Loop parameter is accepted."""
        app = _make_app(sfx_backend=MagicMock())

        transport = ASGITransport(app=app, raise_app_exceptions=False)  # type: ignore[arg-type]
        async with AsyncClient(transport=transport, base_url="http://test") as client:
            response = await client.post(
                "/v1/sound-generation",
                json={"text": "ocean waves", "loop": True},
            )

        assert response.status_code == 200


# ---------------------------------------------------------------------------
# Model tests
# ---------------------------------------------------------------------------


class TestSoundEffectsModels:
    def test_request_defaults(self) -> None:
        from macaw.server.models.sound_effects import SoundGenerationRequest

        req = SoundGenerationRequest(text="rain")
        assert req.duration_seconds == 5.0
        assert req.prompt_influence == 0.3
        assert req.loop is False
        assert req.output_format == "wav"

    def test_response_model(self) -> None:
        from macaw.server.models.sound_effects import SoundGenerationResponse

        resp = SoundGenerationResponse(
            request_id="req-1",
            duration_seconds=5.0,
            output_format="wav",
        )
        assert resp.request_id == "req-1"
        assert resp.duration_seconds == 5.0
