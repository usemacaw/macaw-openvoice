"""Tests for dubbing endpoints and orchestrator."""

from __future__ import annotations

from unittest.mock import MagicMock

from httpx import ASGITransport, AsyncClient

from macaw.server.app import create_app

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_app(*, dubbing_orchestrator: object | None = None) -> object:
    """Create a minimal app for dubbing endpoint tests."""
    app = create_app()
    app.state.dubbing_orchestrator = dubbing_orchestrator
    return app


# ---------------------------------------------------------------------------
# 503 — No dubbing orchestrator configured
# ---------------------------------------------------------------------------


class TestDubbingNoOrchestrator:
    async def test_create_returns_503_when_no_orchestrator(self) -> None:
        """POST /v1/dubbing returns 503 when no orchestrator configured."""
        app = _make_app(dubbing_orchestrator=None)

        transport = ASGITransport(app=app, raise_app_exceptions=False)  # type: ignore[arg-type]
        async with AsyncClient(transport=transport, base_url="http://test") as client:
            response = await client.post(
                "/v1/dubbing",
                data={"target_lang": "pt"},
                files={"audio": ("test.wav", b"\x00" * 100, "audio/wav")},
            )

        assert response.status_code == 503

    async def test_503_message_mentions_dubbing(self) -> None:
        """503 response body mentions dubbing orchestrator."""
        app = _make_app(dubbing_orchestrator=None)

        transport = ASGITransport(app=app, raise_app_exceptions=False)  # type: ignore[arg-type]
        async with AsyncClient(transport=transport, base_url="http://test") as client:
            response = await client.post(
                "/v1/dubbing",
                data={"target_lang": "es"},
                files={"audio": ("test.wav", b"\x00" * 100, "audio/wav")},
            )

        body = response.json()
        assert "dubbing" in body.get("error", {}).get("message", "").lower()

    async def test_status_returns_503_when_no_orchestrator(self) -> None:
        """GET /v1/dubbing/{id} returns 503 when no orchestrator configured."""
        app = _make_app(dubbing_orchestrator=None)

        transport = ASGITransport(app=app, raise_app_exceptions=False)  # type: ignore[arg-type]
        async with AsyncClient(transport=transport, base_url="http://test") as client:
            response = await client.get("/v1/dubbing/fake-id")

        assert response.status_code == 503

    async def test_audio_returns_503_when_no_orchestrator(self) -> None:
        """GET /v1/dubbing/{id}/audio/{lang} returns 503 when no orchestrator."""
        app = _make_app(dubbing_orchestrator=None)

        transport = ASGITransport(app=app, raise_app_exceptions=False)  # type: ignore[arg-type]
        async with AsyncClient(transport=transport, base_url="http://test") as client:
            response = await client.get("/v1/dubbing/fake-id/audio/pt")

        assert response.status_code == 503

    async def test_delete_returns_503_when_no_orchestrator(self) -> None:
        """DELETE /v1/dubbing/{id} returns 503 when no orchestrator."""
        app = _make_app(dubbing_orchestrator=None)

        transport = ASGITransport(app=app, raise_app_exceptions=False)  # type: ignore[arg-type]
        async with AsyncClient(transport=transport, base_url="http://test") as client:
            response = await client.delete("/v1/dubbing/fake-id")

        assert response.status_code == 503


# ---------------------------------------------------------------------------
# Input validation
# ---------------------------------------------------------------------------


class TestDubbingValidation:
    async def test_empty_audio_rejected(self) -> None:
        """Empty audio file returns 400."""
        app = _make_app(dubbing_orchestrator=MagicMock())

        transport = ASGITransport(app=app, raise_app_exceptions=False)  # type: ignore[arg-type]
        async with AsyncClient(transport=transport, base_url="http://test") as client:
            response = await client.post(
                "/v1/dubbing",
                data={"target_lang": "pt"},
                files={"audio": ("test.wav", b"", "audio/wav")},
            )

        assert response.status_code in (400, 422)

    async def test_missing_audio_rejected(self) -> None:
        """Request without audio file returns 422."""
        app = _make_app(dubbing_orchestrator=MagicMock())

        transport = ASGITransport(app=app, raise_app_exceptions=False)  # type: ignore[arg-type]
        async with AsyncClient(transport=transport, base_url="http://test") as client:
            response = await client.post(
                "/v1/dubbing",
                data={"target_lang": "pt"},
            )

        assert response.status_code == 422

    async def test_missing_target_lang_rejected(self) -> None:
        """Request without target_lang returns 422."""
        app = _make_app(dubbing_orchestrator=MagicMock())

        transport = ASGITransport(app=app, raise_app_exceptions=False)  # type: ignore[arg-type]
        async with AsyncClient(transport=transport, base_url="http://test") as client:
            response = await client.post(
                "/v1/dubbing",
                files={"audio": ("test.wav", b"\x00" * 100, "audio/wav")},
            )

        assert response.status_code == 422

    async def test_invalid_output_format_rejected(self) -> None:
        """Invalid output_format returns 400."""
        app = _make_app(dubbing_orchestrator=MagicMock())

        transport = ASGITransport(app=app, raise_app_exceptions=False)  # type: ignore[arg-type]
        async with AsyncClient(transport=transport, base_url="http://test") as client:
            response = await client.post(
                "/v1/dubbing",
                data={"target_lang": "pt", "output_format": "invalid_codec"},
                files={"audio": ("test.wav", b"\x00" * 100, "audio/wav")},
            )

        assert response.status_code in (400, 422)


# ---------------------------------------------------------------------------
# Create dubbing job
# ---------------------------------------------------------------------------


class TestDubbingCreate:
    async def test_returns_202_with_job_id(self) -> None:
        """Valid request returns 202 with dubbing_id."""
        app = _make_app(dubbing_orchestrator=MagicMock())

        transport = ASGITransport(app=app, raise_app_exceptions=False)  # type: ignore[arg-type]
        async with AsyncClient(transport=transport, base_url="http://test") as client:
            response = await client.post(
                "/v1/dubbing",
                data={"target_lang": "pt"},
                files={"audio": ("test.wav", b"\x00" * 100, "audio/wav")},
            )

        assert response.status_code == 202
        body = response.json()
        assert "dubbing_id" in body
        assert body["status"] == "pending"

    async def test_accepts_optional_params(self) -> None:
        """Request with all optional params returns 202."""
        app = _make_app(dubbing_orchestrator=MagicMock())

        transport = ASGITransport(app=app, raise_app_exceptions=False)  # type: ignore[arg-type]
        async with AsyncClient(transport=transport, base_url="http://test") as client:
            response = await client.post(
                "/v1/dubbing",
                data={
                    "target_lang": "fr",
                    "source_lang": "en",
                    "voice_id": "alice",
                    "output_format": "wav",
                },
                files={"audio": ("test.wav", b"\x00" * 100, "audio/wav")},
            )

        assert response.status_code == 202


# ---------------------------------------------------------------------------
# Status, audio, delete — 404 (MVP: no job store)
# ---------------------------------------------------------------------------


class TestDubbingJobOperations:
    async def test_status_returns_404(self) -> None:
        """GET /v1/dubbing/{id} returns 404 for MVP (no job store)."""
        app = _make_app(dubbing_orchestrator=MagicMock())

        transport = ASGITransport(app=app, raise_app_exceptions=False)  # type: ignore[arg-type]
        async with AsyncClient(transport=transport, base_url="http://test") as client:
            response = await client.get("/v1/dubbing/nonexistent-id")

        assert response.status_code == 404

    async def test_audio_returns_404(self) -> None:
        """GET /v1/dubbing/{id}/audio/{lang} returns 404 for MVP."""
        app = _make_app(dubbing_orchestrator=MagicMock())

        transport = ASGITransport(app=app, raise_app_exceptions=False)  # type: ignore[arg-type]
        async with AsyncClient(transport=transport, base_url="http://test") as client:
            response = await client.get("/v1/dubbing/nonexistent-id/audio/pt")

        assert response.status_code == 404

    async def test_delete_returns_404(self) -> None:
        """DELETE /v1/dubbing/{id} returns 404 for MVP."""
        app = _make_app(dubbing_orchestrator=MagicMock())

        transport = ASGITransport(app=app, raise_app_exceptions=False)  # type: ignore[arg-type]
        async with AsyncClient(transport=transport, base_url="http://test") as client:
            response = await client.delete("/v1/dubbing/nonexistent-id")

        assert response.status_code == 404


# ---------------------------------------------------------------------------
# Model tests
# ---------------------------------------------------------------------------


class TestDubbingModels:
    def test_request_defaults(self) -> None:
        from macaw.server.models.dubbing import DubbingRequest

        req = DubbingRequest(target_lang="pt")
        assert req.source_lang is None
        assert req.voice_id is None
        assert req.output_format == "wav"

    def test_job_response(self) -> None:
        from macaw.server.models.dubbing import DubbingJobResponse

        resp = DubbingJobResponse(dubbing_id="dub-123")
        assert resp.dubbing_id == "dub-123"
        assert resp.status == "pending"

    def test_status_response(self) -> None:
        from macaw.server.models.dubbing import DubbingStatusResponse

        resp = DubbingStatusResponse(
            dubbing_id="dub-456",
            status="completed",
            target_lang="pt",
        )
        assert resp.status == "completed"
        assert resp.error is None


# ---------------------------------------------------------------------------
# Orchestrator + TranslationBackend
# ---------------------------------------------------------------------------


class TestTranslationBackend:
    def test_cannot_instantiate_directly(self) -> None:
        import pytest

        from macaw.server.dubbing.orchestrator import TranslationBackend

        with pytest.raises(TypeError, match="abstract"):
            TranslationBackend()  # type: ignore[abstract]

    def test_dubbing_orchestrator_stores_translator(self) -> None:
        from macaw.server.dubbing.orchestrator import DubbingOrchestrator

        mock_translator = MagicMock()
        orch = DubbingOrchestrator(translation_backend=mock_translator)
        assert orch.translator is mock_translator
