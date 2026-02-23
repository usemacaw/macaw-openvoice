"""Tests for PVC (Professional Voice Cloning) endpoints and training interface."""

from __future__ import annotations

from unittest.mock import MagicMock

import pytest
from httpx import ASGITransport, AsyncClient

from macaw.server.app import create_app
from macaw.server.models.voice_training import (
    PVC_MAX_SAMPLE_SIZE_BYTES,
    PVC_MAX_SAMPLES_PER_PROJECT,
)
from macaw.workers.tts.training.interface import VoiceTrainingBackend

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_app(*, training_backend: object | None = None) -> object:
    """Create a minimal app for voice training endpoint tests."""
    app = create_app()
    app.state.training_backend = training_backend
    return app


# ---------------------------------------------------------------------------
# VoiceTrainingBackend ABC
# ---------------------------------------------------------------------------


class TestVoiceTrainingBackendABC:
    def test_cannot_instantiate_directly(self) -> None:
        with pytest.raises(TypeError, match="abstract"):
            VoiceTrainingBackend()  # type: ignore[abstract]

    def test_subclass_must_implement_create_project(self) -> None:
        class _NoCreate(VoiceTrainingBackend):
            async def add_samples(self, project_id: str, audio_samples: list[bytes]) -> None:
                pass

            async def train(self, project_id: str, config: dict[str, object] | None = None) -> str:
                return ""

            async def get_training_status(self, job_id: str) -> str:
                return "pending"

            async def get_trained_voice_id(self, job_id: str) -> str | None:
                return None

        with pytest.raises(TypeError, match="abstract"):
            _NoCreate()  # type: ignore[abstract]

    def test_subclass_must_implement_train(self) -> None:
        class _NoTrain(VoiceTrainingBackend):
            async def create_project(self, name: str, description: str = "") -> str:
                return ""

            async def add_samples(self, project_id: str, audio_samples: list[bytes]) -> None:
                pass

            async def get_training_status(self, job_id: str) -> str:
                return "pending"

            async def get_trained_voice_id(self, job_id: str) -> str | None:
                return None

        with pytest.raises(TypeError, match="abstract"):
            _NoTrain()  # type: ignore[abstract]

    def test_subclass_must_implement_add_samples(self) -> None:
        class _NoSamples(VoiceTrainingBackend):
            async def create_project(self, name: str, description: str = "") -> str:
                return ""

            async def train(self, project_id: str, config: dict[str, object] | None = None) -> str:
                return ""

            async def get_training_status(self, job_id: str) -> str:
                return "pending"

            async def get_trained_voice_id(self, job_id: str) -> str | None:
                return None

        with pytest.raises(TypeError, match="abstract"):
            _NoSamples()  # type: ignore[abstract]

    def test_subclass_must_implement_get_training_status(self) -> None:
        class _NoStatus(VoiceTrainingBackend):
            async def create_project(self, name: str, description: str = "") -> str:
                return ""

            async def add_samples(self, project_id: str, audio_samples: list[bytes]) -> None:
                pass

            async def train(self, project_id: str, config: dict[str, object] | None = None) -> str:
                return ""

            async def get_trained_voice_id(self, job_id: str) -> str | None:
                return None

        with pytest.raises(TypeError, match="abstract"):
            _NoStatus()  # type: ignore[abstract]

    def test_subclass_must_implement_get_trained_voice_id(self) -> None:
        class _NoVoiceId(VoiceTrainingBackend):
            async def create_project(self, name: str, description: str = "") -> str:
                return ""

            async def add_samples(self, project_id: str, audio_samples: list[bytes]) -> None:
                pass

            async def train(self, project_id: str, config: dict[str, object] | None = None) -> str:
                return ""

            async def get_training_status(self, job_id: str) -> str:
                return "pending"

        with pytest.raises(TypeError, match="abstract"):
            _NoVoiceId()  # type: ignore[abstract]


# ---------------------------------------------------------------------------
# post_load_hook default
# ---------------------------------------------------------------------------


class TestTrainingPostLoadHook:
    async def test_default_post_load_hook_is_noop(self) -> None:
        """Default post_load_hook does nothing (no-op)."""

        class _Complete(VoiceTrainingBackend):
            async def create_project(self, name: str, description: str = "") -> str:
                return "proj_1"

            async def add_samples(self, project_id: str, audio_samples: list[bytes]) -> None:
                pass

            async def train(self, project_id: str, config: dict[str, object] | None = None) -> str:
                return "job_1"

            async def get_training_status(self, job_id: str) -> str:
                return "pending"

            async def get_trained_voice_id(self, job_id: str) -> str | None:
                return None

        backend = _Complete()
        await backend.post_load_hook()


# ---------------------------------------------------------------------------
# Pydantic models
# ---------------------------------------------------------------------------


class TestPVCModels:
    def test_create_request_defaults(self) -> None:
        from macaw.server.models.voice_training import PVCCreateRequest

        req = PVCCreateRequest(name="My Voice")
        assert req.name == "My Voice"
        assert req.description == ""

    def test_create_request_with_description(self) -> None:
        from macaw.server.models.voice_training import PVCCreateRequest

        req = PVCCreateRequest(name="Test", description="A test project")
        assert req.description == "A test project"

    def test_create_request_name_too_short(self) -> None:
        from pydantic import ValidationError

        from macaw.server.models.voice_training import PVCCreateRequest

        with pytest.raises(ValidationError, match="string_too_short"):
            PVCCreateRequest(name="")

    def test_create_request_name_too_long(self) -> None:
        from pydantic import ValidationError

        from macaw.server.models.voice_training import PVCCreateRequest

        with pytest.raises(ValidationError, match="string_too_long"):
            PVCCreateRequest(name="x" * 201)

    def test_create_response(self) -> None:
        from macaw.server.models.voice_training import PVCCreateResponse

        resp = PVCCreateResponse(project_id="pvc_abc", name="Test Voice")
        assert resp.project_id == "pvc_abc"
        assert resp.status == "created"

    def test_train_response(self) -> None:
        from macaw.server.models.voice_training import PVCTrainResponse

        resp = PVCTrainResponse(project_id="pvc_abc", job_id="train_xyz")
        assert resp.status == "pending"

    def test_status_response_created(self) -> None:
        from macaw.server.models.voice_training import PVCStatusResponse

        resp = PVCStatusResponse(project_id="pvc_abc", name="Test", status="created")
        assert resp.voice_id is None
        assert resp.error is None
        assert resp.sample_count == 0

    def test_status_response_completed_with_voice_id(self) -> None:
        from macaw.server.models.voice_training import PVCStatusResponse

        resp = PVCStatusResponse(
            project_id="pvc_abc",
            name="Test",
            status="completed",
            voice_id="voice_abc",
            sample_count=10,
        )
        assert resp.voice_id == "voice_abc"
        assert resp.sample_count == 10

    def test_security_constants(self) -> None:
        assert PVC_MAX_SAMPLE_SIZE_BYTES == 20 * 1024 * 1024
        assert PVC_MAX_SAMPLES_PER_PROJECT == 25


# ---------------------------------------------------------------------------
# 503 — No training backend configured
# ---------------------------------------------------------------------------


class TestPVCNoBackend:
    async def test_create_returns_503_when_no_backend(self) -> None:
        """POST /v1/voices/pvc returns 503 when no training backend configured."""
        app = _make_app(training_backend=None)

        transport = ASGITransport(app=app, raise_app_exceptions=False)  # type: ignore[arg-type]
        async with AsyncClient(transport=transport, base_url="http://test") as client:
            response = await client.post(
                "/v1/voices/pvc",
                json={"name": "My Voice"},
            )

        assert response.status_code == 503

    async def test_503_message_mentions_training(self) -> None:
        """503 response body mentions training backend."""
        app = _make_app(training_backend=None)

        transport = ASGITransport(app=app, raise_app_exceptions=False)  # type: ignore[arg-type]
        async with AsyncClient(transport=transport, base_url="http://test") as client:
            response = await client.post(
                "/v1/voices/pvc",
                json={"name": "My Voice"},
            )

        body = response.json()
        assert "training" in body.get("error", {}).get("message", "").lower()

    async def test_samples_returns_503_when_no_backend(self) -> None:
        """POST /v1/voices/pvc/{id}/samples returns 503."""
        app = _make_app(training_backend=None)

        transport = ASGITransport(app=app, raise_app_exceptions=False)  # type: ignore[arg-type]
        async with AsyncClient(transport=transport, base_url="http://test") as client:
            response = await client.post(
                "/v1/voices/pvc/fake-id/samples",
                files={"samples": ("sample.wav", b"\x00" * 100, "audio/wav")},
            )

        assert response.status_code == 503

    async def test_train_returns_503_when_no_backend(self) -> None:
        """POST /v1/voices/pvc/{id}/train returns 503."""
        app = _make_app(training_backend=None)

        transport = ASGITransport(app=app, raise_app_exceptions=False)  # type: ignore[arg-type]
        async with AsyncClient(transport=transport, base_url="http://test") as client:
            response = await client.post("/v1/voices/pvc/fake-id/train")

        assert response.status_code == 503

    async def test_status_returns_503_when_no_backend(self) -> None:
        """GET /v1/voices/pvc/{id} returns 503."""
        app = _make_app(training_backend=None)

        transport = ASGITransport(app=app, raise_app_exceptions=False)  # type: ignore[arg-type]
        async with AsyncClient(transport=transport, base_url="http://test") as client:
            response = await client.get("/v1/voices/pvc/fake-id")

        assert response.status_code == 503

    async def test_delete_returns_503_when_no_backend(self) -> None:
        """DELETE /v1/voices/pvc/{id} returns 503."""
        app = _make_app(training_backend=None)

        transport = ASGITransport(app=app, raise_app_exceptions=False)  # type: ignore[arg-type]
        async with AsyncClient(transport=transport, base_url="http://test") as client:
            response = await client.delete("/v1/voices/pvc/fake-id")

        assert response.status_code == 503


# ---------------------------------------------------------------------------
# Create project
# ---------------------------------------------------------------------------


class TestPVCCreateProject:
    async def test_returns_project_id(self) -> None:
        """POST /v1/voices/pvc returns project_id and status=created."""
        app = _make_app(training_backend=MagicMock())

        transport = ASGITransport(app=app, raise_app_exceptions=False)  # type: ignore[arg-type]
        async with AsyncClient(transport=transport, base_url="http://test") as client:
            response = await client.post(
                "/v1/voices/pvc",
                json={"name": "My Voice"},
            )

        assert response.status_code == 200
        body = response.json()
        assert body["project_id"].startswith("pvc_")
        assert body["name"] == "My Voice"
        assert body["status"] == "created"

    async def test_empty_name_rejected(self) -> None:
        """Empty name is rejected by Pydantic validation."""
        app = _make_app(training_backend=MagicMock())

        transport = ASGITransport(app=app, raise_app_exceptions=False)  # type: ignore[arg-type]
        async with AsyncClient(transport=transport, base_url="http://test") as client:
            response = await client.post(
                "/v1/voices/pvc",
                json={"name": ""},
            )

        assert response.status_code == 422


# ---------------------------------------------------------------------------
# Upload samples — validation
# ---------------------------------------------------------------------------


class TestPVCSampleValidation:
    async def test_empty_sample_rejected(self) -> None:
        """Empty audio sample returns 400."""
        app = _make_app(training_backend=MagicMock())

        transport = ASGITransport(app=app, raise_app_exceptions=False)  # type: ignore[arg-type]
        async with AsyncClient(transport=transport, base_url="http://test") as client:
            response = await client.post(
                "/v1/voices/pvc/proj_1/samples",
                files={"samples": ("empty.wav", b"", "audio/wav")},
            )

        assert response.status_code == 400

    async def test_unsupported_content_type_rejected(self) -> None:
        """Non-audio content type returns 400."""
        app = _make_app(training_backend=MagicMock())

        transport = ASGITransport(app=app, raise_app_exceptions=False)  # type: ignore[arg-type]
        async with AsyncClient(transport=transport, base_url="http://test") as client:
            response = await client.post(
                "/v1/voices/pvc/proj_1/samples",
                files={"samples": ("test.txt", b"text content", "text/plain")},
            )

        assert response.status_code == 400

    async def test_valid_sample_accepted(self) -> None:
        """Valid audio sample is accepted."""
        app = _make_app(training_backend=MagicMock())

        transport = ASGITransport(app=app, raise_app_exceptions=False)  # type: ignore[arg-type]
        async with AsyncClient(transport=transport, base_url="http://test") as client:
            response = await client.post(
                "/v1/voices/pvc/proj_1/samples",
                files={"samples": ("sample.wav", b"\x00" * 100, "audio/wav")},
            )

        assert response.status_code == 200
        body = response.json()
        assert body["project_id"] == "proj_1"
        assert body["samples_added"] == 1


# ---------------------------------------------------------------------------
# Start training
# ---------------------------------------------------------------------------


class TestPVCStartTraining:
    async def test_returns_202_with_job_id(self) -> None:
        """POST /v1/voices/pvc/{id}/train returns 202 with job_id."""
        app = _make_app(training_backend=MagicMock())

        transport = ASGITransport(app=app, raise_app_exceptions=False)  # type: ignore[arg-type]
        async with AsyncClient(transport=transport, base_url="http://test") as client:
            response = await client.post("/v1/voices/pvc/proj_1/train")

        assert response.status_code == 202
        body = response.json()
        assert body["project_id"] == "proj_1"
        assert body["job_id"].startswith("train_")
        assert body["status"] == "pending"


# ---------------------------------------------------------------------------
# Get status — MVP returns created
# ---------------------------------------------------------------------------


class TestPVCGetStatus:
    async def test_returns_created_status(self) -> None:
        """GET /v1/voices/pvc/{id} returns status=created (MVP, no persistence)."""
        app = _make_app(training_backend=MagicMock())

        transport = ASGITransport(app=app, raise_app_exceptions=False)  # type: ignore[arg-type]
        async with AsyncClient(transport=transport, base_url="http://test") as client:
            response = await client.get("/v1/voices/pvc/proj_1")

        assert response.status_code == 200
        body = response.json()
        assert body["project_id"] == "proj_1"
        assert body["status"] == "created"


# ---------------------------------------------------------------------------
# Delete project
# ---------------------------------------------------------------------------


class TestPVCDeleteProject:
    async def test_returns_204(self) -> None:
        """DELETE /v1/voices/pvc/{id} returns 204."""
        app = _make_app(training_backend=MagicMock())

        transport = ASGITransport(app=app, raise_app_exceptions=False)  # type: ignore[arg-type]
        async with AsyncClient(transport=transport, base_url="http://test") as client:
            response = await client.delete("/v1/voices/pvc/proj_1")

        assert response.status_code == 204
