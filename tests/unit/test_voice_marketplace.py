"""Tests for voice marketplace endpoints (share/unshare, list shared, copy)."""

from __future__ import annotations

import tempfile

from httpx import ASGITransport, AsyncClient

from macaw.server.app import create_app
from macaw.server.voice_store import FileSystemVoiceStore

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_app_with_store() -> tuple[object, FileSystemVoiceStore]:
    """Create an app with a real FileSystemVoiceStore in a temp dir."""
    tmpdir = tempfile.mkdtemp()
    store = FileSystemVoiceStore(tmpdir)
    app = create_app(voice_store=store)
    return app, store


# ---------------------------------------------------------------------------
# SavedVoice shared field
# ---------------------------------------------------------------------------


class TestSavedVoiceSharedField:
    async def test_new_voice_is_not_shared_by_default(self) -> None:
        _, store = _make_app_with_store()
        voice = await store.save(
            voice_id="v1", name="Test", voice_type="designed", instruction="test"
        )
        assert voice.shared is False

    async def test_set_shared_true(self) -> None:
        _, store = _make_app_with_store()
        await store.save(voice_id="v1", name="Test", voice_type="designed", instruction="test")
        updated = await store.set_shared("v1", shared=True)
        assert updated is not None
        assert updated.shared is True

    async def test_set_shared_false(self) -> None:
        _, store = _make_app_with_store()
        await store.save(voice_id="v1", name="Test", voice_type="designed", instruction="test")
        await store.set_shared("v1", shared=True)
        updated = await store.set_shared("v1", shared=False)
        assert updated is not None
        assert updated.shared is False

    async def test_set_shared_nonexistent_returns_none(self) -> None:
        _, store = _make_app_with_store()
        result = await store.set_shared("nonexistent", shared=True)
        assert result is None

    async def test_shared_persists_across_reads(self) -> None:
        _, store = _make_app_with_store()
        await store.save(voice_id="v1", name="Test", voice_type="designed", instruction="test")
        await store.set_shared("v1", shared=True)
        reloaded = await store.get("v1")
        assert reloaded is not None
        assert reloaded.shared is True

    async def test_list_shared_returns_only_shared(self) -> None:
        _, store = _make_app_with_store()
        await store.save(voice_id="v1", name="Shared", voice_type="designed", instruction="test")
        await store.save(voice_id="v2", name="Private", voice_type="designed", instruction="test")
        await store.set_shared("v1", shared=True)
        shared = await store.list_shared()
        assert len(shared) == 1
        assert shared[0].voice_id == "v1"

    async def test_list_shared_empty_when_none_shared(self) -> None:
        _, store = _make_app_with_store()
        await store.save(voice_id="v1", name="Private", voice_type="designed", instruction="test")
        shared = await store.list_shared()
        assert shared == []


# ---------------------------------------------------------------------------
# Share endpoint
# ---------------------------------------------------------------------------


class TestShareEndpoint:
    async def test_share_voice(self) -> None:
        app, store = _make_app_with_store()
        await store.save(voice_id="v1", name="My Voice", voice_type="designed", instruction="test")

        transport = ASGITransport(app=app, raise_app_exceptions=False)  # type: ignore[arg-type]
        async with AsyncClient(transport=transport, base_url="http://test") as client:
            response = await client.post("/v1/voices/v1/share")

        assert response.status_code == 200
        body = response.json()
        assert body["shared"] is True

    async def test_share_nonexistent_returns_404(self) -> None:
        app, _ = _make_app_with_store()

        transport = ASGITransport(app=app, raise_app_exceptions=False)  # type: ignore[arg-type]
        async with AsyncClient(transport=transport, base_url="http://test") as client:
            response = await client.post("/v1/voices/nonexistent/share")

        assert response.status_code == 404


# ---------------------------------------------------------------------------
# Unshare endpoint
# ---------------------------------------------------------------------------


class TestUnshareEndpoint:
    async def test_unshare_voice(self) -> None:
        app, store = _make_app_with_store()
        await store.save(voice_id="v1", name="My Voice", voice_type="designed", instruction="test")
        await store.set_shared("v1", shared=True)

        transport = ASGITransport(app=app, raise_app_exceptions=False)  # type: ignore[arg-type]
        async with AsyncClient(transport=transport, base_url="http://test") as client:
            response = await client.delete("/v1/voices/v1/share")

        assert response.status_code == 204

    async def test_unshare_nonexistent_returns_404(self) -> None:
        app, _ = _make_app_with_store()

        transport = ASGITransport(app=app, raise_app_exceptions=False)  # type: ignore[arg-type]
        async with AsyncClient(transport=transport, base_url="http://test") as client:
            response = await client.delete("/v1/voices/nonexistent/share")

        assert response.status_code == 404


# ---------------------------------------------------------------------------
# List shared voices
# ---------------------------------------------------------------------------


class TestListSharedVoices:
    async def test_list_shared_endpoint(self) -> None:
        app, store = _make_app_with_store()
        await store.save(voice_id="v1", name="Shared", voice_type="designed", instruction="test")
        await store.save(voice_id="v2", name="Private", voice_type="designed", instruction="test")
        await store.set_shared("v1", shared=True)

        transport = ASGITransport(app=app, raise_app_exceptions=False)  # type: ignore[arg-type]
        async with AsyncClient(transport=transport, base_url="http://test") as client:
            response = await client.get("/v1/shared-voices")

        assert response.status_code == 200
        body = response.json()
        assert len(body) == 1
        assert body[0]["voice_id"] == "v1"
        assert body[0]["shared"] is True

    async def test_list_shared_empty(self) -> None:
        app, _ = _make_app_with_store()

        transport = ASGITransport(app=app, raise_app_exceptions=False)  # type: ignore[arg-type]
        async with AsyncClient(transport=transport, base_url="http://test") as client:
            response = await client.get("/v1/shared-voices")

        assert response.status_code == 200
        assert response.json() == []


# ---------------------------------------------------------------------------
# Copy shared voice
# ---------------------------------------------------------------------------


class TestCopySharedVoice:
    async def test_copy_shared_voice(self) -> None:
        app, store = _make_app_with_store()
        await store.save(
            voice_id="v1", name="Shared Voice", voice_type="designed", instruction="test"
        )
        await store.set_shared("v1", shared=True)

        transport = ASGITransport(app=app, raise_app_exceptions=False)  # type: ignore[arg-type]
        async with AsyncClient(transport=transport, base_url="http://test") as client:
            response = await client.post("/v1/voices/add/v1")

        assert response.status_code == 201
        body = response.json()
        assert body["voice_id"] != "v1"
        assert body["name"] == "Shared Voice"

    async def test_copy_not_shared_returns_400(self) -> None:
        app, store = _make_app_with_store()
        await store.save(
            voice_id="v1", name="Private Voice", voice_type="designed", instruction="test"
        )

        transport = ASGITransport(app=app, raise_app_exceptions=False)  # type: ignore[arg-type]
        async with AsyncClient(transport=transport, base_url="http://test") as client:
            response = await client.post("/v1/voices/add/v1")

        assert response.status_code == 400

    async def test_copy_nonexistent_returns_404(self) -> None:
        app, _ = _make_app_with_store()

        transport = ASGITransport(app=app, raise_app_exceptions=False)  # type: ignore[arg-type]
        async with AsyncClient(transport=transport, base_url="http://test") as client:
            response = await client.post("/v1/voices/add/nonexistent")

        assert response.status_code == 404
