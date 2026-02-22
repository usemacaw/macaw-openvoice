"""Tests for the /v1/voices endpoints.

Validates:
- GET /v1/voices -- returns voices from loaded TTS workers via gRPC
- GET /v1/voices -- empty list when no TTS models loaded
- GET /v1/voices -- handles worker unavailable gracefully
- GET /v1/voices -- handles gRPC errors gracefully
- POST /v1/voices -- create saved voice (designed + cloned)
- GET /v1/voices/{id} -- get saved voice
- DELETE /v1/voices/{id} -- delete saved voice
- PUT /v1/voices/{id} -- update saved voice
- Validation errors for invalid voice creation
"""

from __future__ import annotations

import io
from unittest.mock import AsyncMock, MagicMock, patch

import httpx

from macaw._types import ModelType
from macaw.server.app import create_app
from macaw.server.voice_store import FileSystemVoiceStore

# --- Helpers ---


def _make_manifest(*, name: str = "kokoro-v1", model_type: ModelType = ModelType.TTS) -> MagicMock:
    manifest = MagicMock()
    manifest.name = name
    manifest.model_type = model_type
    return manifest


def _make_mock_registry(
    manifests: list[MagicMock] | None = None,
) -> MagicMock:
    registry = MagicMock()
    registry.list_models.return_value = manifests or []
    return registry


def _make_mock_worker_manager(*, has_worker: bool = True) -> MagicMock:
    manager = MagicMock()
    if has_worker:
        worker = MagicMock()
        worker.port = 50052
        worker.worker_id = "tts-worker-1"
        manager.get_ready_worker.return_value = worker
    else:
        manager.get_ready_worker.return_value = None
    return manager


def _make_list_voices_response(voices: list[dict[str, str]]) -> MagicMock:
    """Create a mock ListVoicesResponse with VoiceInfoProto objects."""
    response = MagicMock()
    voice_protos = []
    for v in voices:
        vp = MagicMock()
        vp.voice_id = v.get("voice_id", "")
        vp.name = v.get("name", "")
        vp.language = v.get("language", "")
        vp.gender = v.get("gender", "")
        voice_protos.append(vp)
    response.voices = voice_protos
    return response


# --- GET /v1/voices (preset voices from workers) ---


class TestListVoicesReturnsVoices:
    """GET /v1/voices returns voices from loaded TTS workers."""

    async def test_returns_voices_from_worker(self) -> None:
        manifests = [_make_manifest(name="kokoro-v1")]
        registry = _make_mock_registry(manifests=manifests)
        worker_manager = _make_mock_worker_manager(has_worker=True)

        app = create_app(
            registry=registry,
            worker_manager=worker_manager,
        )

        mock_response = _make_list_voices_response(
            [
                {"voice_id": "af_heart", "name": "af_heart", "language": "en", "gender": "female"},
                {"voice_id": "am_echo", "name": "am_echo", "language": "en", "gender": "male"},
            ]
        )

        mock_stub_instance = MagicMock()
        mock_stub_instance.ListVoices = AsyncMock(return_value=mock_response)

        with patch(
            "macaw.server.routes.voices.TTSWorkerStub",
            return_value=mock_stub_instance,
        ):
            async with httpx.AsyncClient(
                transport=httpx.ASGITransport(app=app),
                base_url="http://test",
            ) as client:
                resp = await client.get("/v1/voices")

        assert resp.status_code == 200
        body = resp.json()
        assert len(body["data"]) == 2
        assert body["data"][0]["voice_id"] == "af_heart"
        assert body["data"][0]["model"] == "kokoro-v1"
        assert body["data"][1]["voice_id"] == "am_echo"
        assert body["data"][1]["gender"] == "male"


class TestListVoicesNoTTSModels:
    """GET /v1/voices returns empty when no TTS models loaded."""

    async def test_returns_empty_list(self) -> None:
        # No TTS manifests, only STT
        manifests = [_make_manifest(name="whisper-v1", model_type=ModelType.STT)]
        registry = _make_mock_registry(manifests=manifests)
        worker_manager = _make_mock_worker_manager(has_worker=True)

        app = create_app(
            registry=registry,
            worker_manager=worker_manager,
        )

        async with httpx.AsyncClient(
            transport=httpx.ASGITransport(app=app),
            base_url="http://test",
        ) as client:
            resp = await client.get("/v1/voices")

        assert resp.status_code == 200
        body = resp.json()
        assert body["data"] == []

    async def test_returns_empty_when_registry_has_no_manifests(self) -> None:
        registry = _make_mock_registry(manifests=[])
        worker_manager = _make_mock_worker_manager()

        app = create_app(
            registry=registry,
            worker_manager=worker_manager,
        )

        async with httpx.AsyncClient(
            transport=httpx.ASGITransport(app=app),
            base_url="http://test",
        ) as client:
            resp = await client.get("/v1/voices")

        assert resp.status_code == 200
        assert resp.json()["data"] == []


class TestListVoicesWorkerUnavailable:
    """GET /v1/voices handles worker unavailable gracefully."""

    async def test_skips_model_with_no_ready_worker(self) -> None:
        manifests = [_make_manifest(name="kokoro-v1")]
        registry = _make_mock_registry(manifests=manifests)
        worker_manager = _make_mock_worker_manager(has_worker=False)

        app = create_app(
            registry=registry,
            worker_manager=worker_manager,
        )

        async with httpx.AsyncClient(
            transport=httpx.ASGITransport(app=app),
            base_url="http://test",
        ) as client:
            resp = await client.get("/v1/voices")

        assert resp.status_code == 200
        assert resp.json()["data"] == []


class TestListVoicesGRPCError:
    """GET /v1/voices handles gRPC errors gracefully."""

    async def test_skips_model_on_grpc_error(self) -> None:
        manifests = [_make_manifest(name="kokoro-v1")]
        registry = _make_mock_registry(manifests=manifests)
        worker_manager = _make_mock_worker_manager(has_worker=True)

        app = create_app(
            registry=registry,
            worker_manager=worker_manager,
        )

        mock_stub_instance = MagicMock()
        mock_stub_instance.ListVoices = AsyncMock(
            side_effect=Exception("Connection refused"),
        )

        with patch(
            "macaw.server.routes.voices.TTSWorkerStub",
            return_value=mock_stub_instance,
        ):
            async with httpx.AsyncClient(
                transport=httpx.ASGITransport(app=app),
                base_url="http://test",
            ) as client:
                resp = await client.get("/v1/voices")

        # Should return 200 with empty list, not 500
        assert resp.status_code == 200
        assert resp.json()["data"] == []


class TestVoicesConverters:
    """Test the voices_to_proto_response converter."""

    def test_converts_voice_list_to_proto(self) -> None:
        from macaw._types import VoiceInfo
        from macaw.workers.tts.converters import voices_to_proto_response

        voices = [
            VoiceInfo(voice_id="af_heart", name="af_heart", language="en", gender="female"),
            VoiceInfo(voice_id="am_echo", name="am_echo", language="en", gender="male"),
        ]
        response = voices_to_proto_response(voices)
        assert len(response.voices) == 2
        assert response.voices[0].voice_id == "af_heart"
        assert response.voices[0].gender == "female"
        assert response.voices[1].voice_id == "am_echo"

    def test_converts_empty_list(self) -> None:
        from macaw.workers.tts.converters import voices_to_proto_response

        response = voices_to_proto_response([])
        assert len(response.voices) == 0

    def test_none_gender_becomes_empty_string(self) -> None:
        from macaw._types import VoiceInfo
        from macaw.workers.tts.converters import voices_to_proto_response

        voices = [VoiceInfo(voice_id="test", name="test", language="en", gender=None)]
        response = voices_to_proto_response(voices)
        assert response.voices[0].gender == ""


# --- POST /v1/voices (create saved voice) ---


class TestCreateVoice:
    """POST /v1/voices creates a saved voice."""

    async def test_create_designed_voice(self, tmp_path: object) -> None:
        voice_store = FileSystemVoiceStore(str(tmp_path))
        registry = _make_mock_registry()
        manager = _make_mock_worker_manager()

        app = create_app(
            registry=registry,
            worker_manager=manager,
            voice_store=voice_store,
        )

        async with httpx.AsyncClient(
            transport=httpx.ASGITransport(app=app),
            base_url="http://test",
        ) as client:
            resp = await client.post(
                "/v1/voices",
                data={
                    "name": "Deep Narrator",
                    "voice_type": "designed",
                    "instruction": "A deep male narrator voice.",
                    "language": "en",
                },
            )

        assert resp.status_code == 201
        body = resp.json()
        assert body["name"] == "Deep Narrator"
        assert body["voice_type"] == "designed"
        assert body["instruction"] == "A deep male narrator voice."
        assert body["has_ref_audio"] is False
        assert "voice_id" in body
        assert body["created_at"] > 0

    async def test_create_cloned_voice_with_ref_audio(self, tmp_path: object) -> None:
        voice_store = FileSystemVoiceStore(str(tmp_path))
        registry = _make_mock_registry()
        manager = _make_mock_worker_manager()

        app = create_app(
            registry=registry,
            worker_manager=manager,
            voice_store=voice_store,
        )

        audio_bytes = b"\x00\x01\x02\x03" * 100

        async with httpx.AsyncClient(
            transport=httpx.ASGITransport(app=app),
            base_url="http://test",
        ) as client:
            resp = await client.post(
                "/v1/voices",
                data={
                    "name": "My Clone",
                    "voice_type": "cloned",
                    "ref_text": "Hello world",
                    "language": "en",
                },
                files={"ref_audio": ("ref.wav", io.BytesIO(audio_bytes), "audio/wav")},
            )

        assert resp.status_code == 201
        body = resp.json()
        assert body["voice_type"] == "cloned"
        assert body["has_ref_audio"] is True
        assert body["ref_text"] == "Hello world"

    async def test_create_voice_invalid_type_returns_400(self, tmp_path: object) -> None:
        voice_store = FileSystemVoiceStore(str(tmp_path))
        registry = _make_mock_registry()
        manager = _make_mock_worker_manager()

        app = create_app(
            registry=registry,
            worker_manager=manager,
            voice_store=voice_store,
        )

        async with httpx.AsyncClient(
            transport=httpx.ASGITransport(app=app, raise_app_exceptions=False),
            base_url="http://test",
        ) as client:
            resp = await client.post(
                "/v1/voices",
                data={"name": "Bad", "voice_type": "invalid"},
            )

        assert resp.status_code == 400
        assert "voice_type" in resp.json()["error"]["message"]

    async def test_create_cloned_without_ref_audio_returns_400(self, tmp_path: object) -> None:
        voice_store = FileSystemVoiceStore(str(tmp_path))
        registry = _make_mock_registry()
        manager = _make_mock_worker_manager()

        app = create_app(
            registry=registry,
            worker_manager=manager,
            voice_store=voice_store,
        )

        async with httpx.AsyncClient(
            transport=httpx.ASGITransport(app=app, raise_app_exceptions=False),
            base_url="http://test",
        ) as client:
            resp = await client.post(
                "/v1/voices",
                data={"name": "Clone", "voice_type": "cloned"},
            )

        assert resp.status_code == 400
        assert "ref_audio" in resp.json()["error"]["message"]

    async def test_create_designed_without_instruction_returns_400(self, tmp_path: object) -> None:
        voice_store = FileSystemVoiceStore(str(tmp_path))
        registry = _make_mock_registry()
        manager = _make_mock_worker_manager()

        app = create_app(
            registry=registry,
            worker_manager=manager,
            voice_store=voice_store,
        )

        async with httpx.AsyncClient(
            transport=httpx.ASGITransport(app=app, raise_app_exceptions=False),
            base_url="http://test",
        ) as client:
            resp = await client.post(
                "/v1/voices",
                data={"name": "Design", "voice_type": "designed"},
            )

        assert resp.status_code == 400
        assert "instruction" in resp.json()["error"]["message"]

    async def test_create_voice_no_store_returns_400(self) -> None:
        registry = _make_mock_registry()
        manager = _make_mock_worker_manager()

        app = create_app(registry=registry, worker_manager=manager)

        async with httpx.AsyncClient(
            transport=httpx.ASGITransport(app=app, raise_app_exceptions=False),
            base_url="http://test",
        ) as client:
            resp = await client.post(
                "/v1/voices",
                data={"name": "Test", "voice_type": "designed", "instruction": "test"},
            )

        assert resp.status_code == 400
        assert "VoiceStore" in resp.json()["error"]["message"]


# --- GET /v1/voices/{id} (get saved voice) ---


class TestGetVoice:
    """GET /v1/voices/{id} retrieves a saved voice."""

    async def test_get_existing_voice(self, tmp_path: object) -> None:
        voice_store = FileSystemVoiceStore(str(tmp_path))
        saved = await voice_store.save(
            voice_id="abc123",
            name="My Voice",
            voice_type="designed",
            instruction="Speak clearly",
        )

        registry = _make_mock_registry()
        manager = _make_mock_worker_manager()

        app = create_app(
            registry=registry,
            worker_manager=manager,
            voice_store=voice_store,
        )

        async with httpx.AsyncClient(
            transport=httpx.ASGITransport(app=app),
            base_url="http://test",
        ) as client:
            resp = await client.get(f"/v1/voices/{saved.voice_id}")

        assert resp.status_code == 200
        body = resp.json()
        assert body["voice_id"] == "abc123"
        assert body["name"] == "My Voice"
        assert body["voice_type"] == "designed"

    async def test_get_nonexistent_returns_404(self, tmp_path: object) -> None:
        voice_store = FileSystemVoiceStore(str(tmp_path))
        registry = _make_mock_registry()
        manager = _make_mock_worker_manager()

        app = create_app(
            registry=registry,
            worker_manager=manager,
            voice_store=voice_store,
        )

        async with httpx.AsyncClient(
            transport=httpx.ASGITransport(app=app, raise_app_exceptions=False),
            base_url="http://test",
        ) as client:
            resp = await client.get("/v1/voices/nonexistent")

        assert resp.status_code == 404
        assert "voice_not_found" in resp.json()["error"]["code"]


# --- DELETE /v1/voices/{id} ---


class TestDeleteVoice:
    """DELETE /v1/voices/{id} removes a saved voice."""

    async def test_delete_existing_voice(self, tmp_path: object) -> None:
        voice_store = FileSystemVoiceStore(str(tmp_path))
        saved = await voice_store.save(
            voice_id="del-me",
            name="Delete Me",
            voice_type="designed",
            instruction="temp",
        )

        registry = _make_mock_registry()
        manager = _make_mock_worker_manager()

        app = create_app(
            registry=registry,
            worker_manager=manager,
            voice_store=voice_store,
        )

        async with httpx.AsyncClient(
            transport=httpx.ASGITransport(app=app),
            base_url="http://test",
        ) as client:
            resp = await client.delete(f"/v1/voices/{saved.voice_id}")

        assert resp.status_code == 204

        # Verify removed from store
        result = await voice_store.get("del-me")
        assert result is None

    async def test_delete_nonexistent_returns_404(self, tmp_path: object) -> None:
        voice_store = FileSystemVoiceStore(str(tmp_path))
        registry = _make_mock_registry()
        manager = _make_mock_worker_manager()

        app = create_app(
            registry=registry,
            worker_manager=manager,
            voice_store=voice_store,
        )

        async with httpx.AsyncClient(
            transport=httpx.ASGITransport(app=app, raise_app_exceptions=False),
            base_url="http://test",
        ) as client:
            resp = await client.delete("/v1/voices/nonexistent")

        assert resp.status_code == 404


# --- PUT /v1/voices/{id} (update saved voice) ---


class TestUpdateVoice:
    """PUT /v1/voices/{id} updates a saved voice's metadata."""

    async def test_update_voice_name(self, tmp_path: object) -> None:
        voice_store = FileSystemVoiceStore(str(tmp_path))
        saved = await voice_store.save(
            voice_id="upd-1",
            name="Old Name",
            voice_type="designed",
            instruction="test instruction",
        )

        registry = _make_mock_registry()
        manager = _make_mock_worker_manager()

        app = create_app(
            registry=registry,
            worker_manager=manager,
            voice_store=voice_store,
        )

        async with httpx.AsyncClient(
            transport=httpx.ASGITransport(app=app),
            base_url="http://test",
        ) as client:
            resp = await client.put(
                f"/v1/voices/{saved.voice_id}",
                data={"name": "New Name"},
            )

        assert resp.status_code == 200
        body = resp.json()
        assert body["name"] == "New Name"
        assert body["instruction"] == "test instruction"
        assert body["voice_id"] == "upd-1"

    async def test_update_voice_multiple_fields(self, tmp_path: object) -> None:
        voice_store = FileSystemVoiceStore(str(tmp_path))
        saved = await voice_store.save(
            voice_id="upd-2",
            name="Original",
            voice_type="designed",
            instruction="old instruction",
            language="en",
        )

        registry = _make_mock_registry()
        manager = _make_mock_worker_manager()

        app = create_app(
            registry=registry,
            worker_manager=manager,
            voice_store=voice_store,
        )

        async with httpx.AsyncClient(
            transport=httpx.ASGITransport(app=app),
            base_url="http://test",
        ) as client:
            resp = await client.put(
                f"/v1/voices/{saved.voice_id}",
                data={"name": "Updated", "language": "pt"},
            )

        assert resp.status_code == 200
        body = resp.json()
        assert body["name"] == "Updated"
        assert body["language"] == "pt"
        assert body["instruction"] == "old instruction"

    async def test_update_nonexistent_returns_404(self, tmp_path: object) -> None:
        voice_store = FileSystemVoiceStore(str(tmp_path))
        registry = _make_mock_registry()
        manager = _make_mock_worker_manager()

        app = create_app(
            registry=registry,
            worker_manager=manager,
            voice_store=voice_store,
        )

        async with httpx.AsyncClient(
            transport=httpx.ASGITransport(app=app, raise_app_exceptions=False),
            base_url="http://test",
        ) as client:
            resp = await client.put(
                "/v1/voices/nonexistent",
                data={"name": "New"},
            )

        assert resp.status_code == 404
        assert "voice_not_found" in resp.json()["error"]["code"]

    async def test_update_voice_no_store_returns_400(self) -> None:
        registry = _make_mock_registry()
        manager = _make_mock_worker_manager()

        app = create_app(registry=registry, worker_manager=manager)

        async with httpx.AsyncClient(
            transport=httpx.ASGITransport(app=app, raise_app_exceptions=False),
            base_url="http://test",
        ) as client:
            resp = await client.put(
                "/v1/voices/some-id",
                data={"name": "Test"},
            )

        assert resp.status_code == 400
        assert "VoiceStore" in resp.json()["error"]["message"]

    async def test_update_preserves_has_ref_audio(self, tmp_path: object) -> None:
        voice_store = FileSystemVoiceStore(str(tmp_path))
        await voice_store.save(
            voice_id="upd-3",
            name="Clone",
            voice_type="cloned",
            ref_audio=b"\x00" * 100,
            ref_text="Original text",
        )

        registry = _make_mock_registry()
        manager = _make_mock_worker_manager()

        app = create_app(
            registry=registry,
            worker_manager=manager,
            voice_store=voice_store,
        )

        async with httpx.AsyncClient(
            transport=httpx.ASGITransport(app=app),
            base_url="http://test",
        ) as client:
            resp = await client.put(
                "/v1/voices/upd-3",
                data={"name": "Updated Clone"},
            )

        assert resp.status_code == 200
        body = resp.json()
        assert body["has_ref_audio"] is True
        assert body["name"] == "Updated Clone"
        assert body["ref_text"] == "Original text"
