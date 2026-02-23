"""Unit tests for GET/DELETE /v1/speech-to-text/{transcription_id} endpoints."""

from __future__ import annotations

from unittest.mock import AsyncMock

import httpx
from httpx import ASGITransport

from macaw.server.app import create_app
from macaw.server.transcript_store.interface import StoredTranscript, TranscriptStore

_BASE_URL = "http://test"

_SAMPLE_TRANSCRIPT = StoredTranscript(
    transcription_id="test-id-123",
    text="Hello world",
    language="en",
    duration=1.5,
    model="whisper-large-v3",
    created_at="2026-02-23T00:00:00Z",
    metadata={"segments": []},
)


def _make_store(
    *,
    get_return: StoredTranscript | None = None,
    delete_return: bool = True,
) -> AsyncMock:
    store = AsyncMock(spec=TranscriptStore)
    store.get.return_value = get_return
    store.delete.return_value = delete_return
    return store


# ─── GET /v1/speech-to-text/{transcription_id} ───


class TestGetTranscript:
    async def test_get_existing_transcript_returns_200(self) -> None:
        """Store returns a transcript -- endpoint responds with 200."""
        store = _make_store(get_return=_SAMPLE_TRANSCRIPT)
        app = create_app(transcript_store=store)

        async with httpx.AsyncClient(
            transport=ASGITransport(app=app), base_url=_BASE_URL
        ) as client:
            response = await client.get("/v1/speech-to-text/test-id-123")

        assert response.status_code == 200

    async def test_get_transcript_response_fields(self) -> None:
        """All StoredTranscript fields are present in the JSON response."""
        store = _make_store(get_return=_SAMPLE_TRANSCRIPT)
        app = create_app(transcript_store=store)

        async with httpx.AsyncClient(
            transport=ASGITransport(app=app), base_url=_BASE_URL
        ) as client:
            response = await client.get("/v1/speech-to-text/test-id-123")

        body = response.json()
        assert body["transcription_id"] == "test-id-123"
        assert body["text"] == "Hello world"
        assert body["language"] == "en"
        assert body["duration"] == 1.5
        assert body["model"] == "whisper-large-v3"
        assert body["created_at"] == "2026-02-23T00:00:00Z"
        assert body["metadata"] == {"segments": []}

    async def test_get_nonexistent_transcript_returns_404(self) -> None:
        """Store returns None for unknown ID -- endpoint responds with 404."""
        store = _make_store(get_return=None)
        app = create_app(transcript_store=store)

        async with httpx.AsyncClient(
            transport=ASGITransport(app=app, raise_app_exceptions=False),
            base_url=_BASE_URL,
        ) as client:
            response = await client.get("/v1/speech-to-text/nonexistent-id")

        assert response.status_code == 404
        body = response.json()
        assert body["error"]["code"] == "transcript_not_found"

    async def test_get_transcript_response_model(self) -> None:
        """Response conforms to StoredTranscriptResponse schema (exact key set)."""
        transcript_with_defaults = StoredTranscript(
            transcription_id="minimal-id",
            text="Just text",
        )
        store = _make_store(get_return=transcript_with_defaults)
        app = create_app(transcript_store=store)

        async with httpx.AsyncClient(
            transport=ASGITransport(app=app), base_url=_BASE_URL
        ) as client:
            response = await client.get("/v1/speech-to-text/minimal-id")

        body = response.json()
        expected_keys = {
            "transcription_id",
            "text",
            "language",
            "duration",
            "model",
            "created_at",
            "metadata",
        }
        assert set(body.keys()) == expected_keys
        # Default values for optional fields
        assert body["language"] is None
        assert body["duration"] is None
        assert body["model"] is None
        assert body["created_at"] == ""
        assert body["metadata"] == {}

    async def test_get_transcript_calls_store_with_correct_id(self) -> None:
        """The path parameter is forwarded to store.get() as-is."""
        store = _make_store(get_return=_SAMPLE_TRANSCRIPT)
        app = create_app(transcript_store=store)

        async with httpx.AsyncClient(
            transport=ASGITransport(app=app), base_url=_BASE_URL
        ) as client:
            await client.get("/v1/speech-to-text/my-custom-id-789")

        store.get.assert_awaited_once_with("my-custom-id-789")


# ─── DELETE /v1/speech-to-text/{transcription_id} ───


class TestDeleteTranscript:
    async def test_delete_existing_transcript_returns_204(self) -> None:
        """Store.delete returns True -- endpoint responds with 204 No Content."""
        store = _make_store(delete_return=True)
        app = create_app(transcript_store=store)

        async with httpx.AsyncClient(
            transport=ASGITransport(app=app), base_url=_BASE_URL
        ) as client:
            response = await client.delete("/v1/speech-to-text/test-id-123")

        assert response.status_code == 204
        assert response.content == b""

    async def test_delete_nonexistent_transcript_returns_404(self) -> None:
        """Store.delete returns False -- endpoint responds with 404."""
        store = _make_store(delete_return=False)
        app = create_app(transcript_store=store)

        async with httpx.AsyncClient(
            transport=ASGITransport(app=app, raise_app_exceptions=False),
            base_url=_BASE_URL,
        ) as client:
            response = await client.delete("/v1/speech-to-text/nonexistent-id")

        assert response.status_code == 404
        body = response.json()
        assert body["error"]["code"] == "transcript_not_found"

    async def test_delete_calls_store_with_correct_id(self) -> None:
        """The path parameter is forwarded to store.delete() as-is."""
        store = _make_store(delete_return=True)
        app = create_app(transcript_store=store)

        async with httpx.AsyncClient(
            transport=ASGITransport(app=app), base_url=_BASE_URL
        ) as client:
            await client.delete("/v1/speech-to-text/del-abc-456")

        store.delete.assert_awaited_once_with("del-abc-456")


# ─── TranscriptStore not configured ───


class TestTranscriptStoreNotConfigured:
    async def test_get_without_store_returns_400(self) -> None:
        """When no transcript store is configured, GET returns 400 (InvalidRequestError)."""
        app = create_app(transcript_store=None)

        async with httpx.AsyncClient(
            transport=ASGITransport(app=app, raise_app_exceptions=False),
            base_url=_BASE_URL,
        ) as client:
            response = await client.get("/v1/speech-to-text/any-id")

        assert response.status_code == 400
        body = response.json()
        assert body["error"]["code"] == "invalid_request"
        assert "TranscriptStore" in body["error"]["message"]

    async def test_delete_without_store_returns_400(self) -> None:
        """When no transcript store is configured, DELETE returns 400 (InvalidRequestError)."""
        app = create_app(transcript_store=None)

        async with httpx.AsyncClient(
            transport=ASGITransport(app=app, raise_app_exceptions=False),
            base_url=_BASE_URL,
        ) as client:
            response = await client.delete("/v1/speech-to-text/any-id")

        assert response.status_code == 400
        body = response.json()
        assert body["error"]["code"] == "invalid_request"
        assert "TranscriptStore" in body["error"]["message"]
