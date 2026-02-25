"""Unit tests for transcript store (interface + filesystem backend)."""

from __future__ import annotations

import json
import os
import time
from typing import TYPE_CHECKING

import pytest

if TYPE_CHECKING:
    from pathlib import Path

from macaw.exceptions import InvalidRequestError
from macaw.server.transcript_store.filesystem import FileSystemTranscriptStore
from macaw.server.transcript_store.interface import StoredTranscript, TranscriptStore

# ─── StoredTranscript dataclass ───


class TestStoredTranscript:
    def test_creation_with_defaults(self) -> None:
        t = StoredTranscript(transcription_id="abc-123", text="hello world")

        assert t.transcription_id == "abc-123"
        assert t.text == "hello world"
        assert t.language is None
        assert t.duration is None
        assert t.model is None
        assert t.created_at == ""
        assert t.metadata == {}

    def test_creation_with_all_fields(self) -> None:
        meta = {"segments": [{"id": 0, "text": "hello"}]}
        t = StoredTranscript(
            transcription_id="def-456",
            text="hello world",
            language="en",
            duration=2.5,
            model="whisper-large",
            created_at="2026-02-23T10:00:00Z",
            metadata=meta,
        )

        assert t.transcription_id == "def-456"
        assert t.text == "hello world"
        assert t.language == "en"
        assert t.duration == 2.5
        assert t.model == "whisper-large"
        assert t.created_at == "2026-02-23T10:00:00Z"
        assert t.metadata == meta

    def test_frozen_dataclass(self) -> None:
        t = StoredTranscript(transcription_id="abc", text="text")
        with pytest.raises(AttributeError):
            t.text = "other"  # type: ignore[misc]


# ─── TranscriptStore ABC ───


class TestTranscriptStoreABC:
    def test_cannot_instantiate_abc(self) -> None:
        with pytest.raises(TypeError):
            TranscriptStore()  # type: ignore[abstract]


# ─── FileSystemTranscriptStore ───


class TestFileSystemTranscriptStore:
    @pytest.fixture()
    def store(self, tmp_path: Path) -> FileSystemTranscriptStore:
        return FileSystemTranscriptStore(str(tmp_path))

    @pytest.fixture()
    def sample_transcript(self) -> StoredTranscript:
        return StoredTranscript(
            transcription_id="test-id-001",
            text="Hello from the test",
            language="en",
            duration=3.14,
            model="whisper-base",
            created_at="2026-02-23T12:00:00Z",
            metadata={"task": "transcribe"},
        )

    async def test_save_creates_json_file(
        self, store: FileSystemTranscriptStore, sample_transcript: StoredTranscript, tmp_path: Path
    ) -> None:
        result_id = await store.save(sample_transcript)

        assert result_id == "test-id-001"
        json_path = tmp_path / "test-id-001.json"
        assert json_path.exists()

        data = json.loads(json_path.read_text())
        assert data["transcription_id"] == "test-id-001"
        assert data["text"] == "Hello from the test"
        assert data["language"] == "en"
        assert data["duration"] == 3.14
        assert data["model"] == "whisper-base"

    async def test_get_returns_stored_transcript(
        self, store: FileSystemTranscriptStore, sample_transcript: StoredTranscript
    ) -> None:
        await store.save(sample_transcript)
        result = await store.get("test-id-001")

        assert result is not None
        assert result.transcription_id == "test-id-001"
        assert result.text == "Hello from the test"
        assert result.language == "en"
        assert result.duration == 3.14
        assert result.model == "whisper-base"
        assert result.created_at == "2026-02-23T12:00:00Z"
        assert result.metadata == {"task": "transcribe"}

    async def test_get_returns_none_for_nonexistent(
        self, store: FileSystemTranscriptStore
    ) -> None:
        result = await store.get("does-not-exist")
        assert result is None

    async def test_delete_removes_file_and_returns_true(
        self, store: FileSystemTranscriptStore, sample_transcript: StoredTranscript, tmp_path: Path
    ) -> None:
        await store.save(sample_transcript)
        deleted = await store.delete("test-id-001")

        assert deleted is True
        assert not (tmp_path / "test-id-001.json").exists()

    async def test_delete_returns_false_for_nonexistent(
        self, store: FileSystemTranscriptStore
    ) -> None:
        deleted = await store.delete("does-not-exist")
        assert deleted is False

    async def test_list_all_returns_saved_transcripts(
        self, store: FileSystemTranscriptStore
    ) -> None:
        for i in range(3):
            t = StoredTranscript(
                transcription_id=f"id-{i:03d}",
                text=f"text {i}",
                created_at=f"2026-02-23T10:{i:02d}:00Z",
            )
            await store.save(t)
            # Ensure distinct mtime ordering
            time.sleep(0.05)

        results = await store.list_all()
        assert len(results) == 3
        # Newest first (highest mtime)
        assert results[0].transcription_id == "id-002"
        assert results[2].transcription_id == "id-000"

    async def test_list_all_with_limit_and_offset(self, store: FileSystemTranscriptStore) -> None:
        for i in range(5):
            t = StoredTranscript(transcription_id=f"id-{i:03d}", text=f"text {i}")
            await store.save(t)
            time.sleep(0.05)

        # Get second page (items 2-3)
        results = await store.list_all(limit=2, offset=2)
        assert len(results) == 2

    async def test_list_all_empty_directory(self, store: FileSystemTranscriptStore) -> None:
        results = await store.list_all()
        assert results == []

    async def test_ttl_cleanup_removes_expired_files(self, tmp_path: Path) -> None:
        store = FileSystemTranscriptStore(str(tmp_path), ttl_seconds=1)
        t = StoredTranscript(transcription_id="old-one", text="old text")
        await store.save(t)

        # Backdate the file mtime to simulate expiry
        old_path = tmp_path / "old-one.json"
        old_time = time.time() - 10
        os.utime(old_path, (old_time, old_time))

        # Save a fresh one
        t2 = StoredTranscript(transcription_id="new-one", text="new text")
        await store.save(t2)

        results = await store.list_all()
        assert len(results) == 1
        assert results[0].transcription_id == "new-one"
        assert not old_path.exists()

    async def test_validate_transcript_id_rejects_path_traversal(
        self, store: FileSystemTranscriptStore
    ) -> None:
        bad_transcript = StoredTranscript(
            transcription_id="../../etc/passwd",
            text="malicious",
        )
        with pytest.raises(InvalidRequestError, match="Invalid transcription_id"):
            await store.save(bad_transcript)

    async def test_validate_transcript_id_rejects_empty(
        self, store: FileSystemTranscriptStore
    ) -> None:
        with pytest.raises(InvalidRequestError, match="Invalid transcription_id"):
            await store.get("")

    async def test_auto_creates_directory(self, tmp_path: Path) -> None:
        nested = tmp_path / "deep" / "nested" / "path"
        store = FileSystemTranscriptStore(str(nested))
        t = StoredTranscript(transcription_id="auto-dir-test", text="test")
        await store.save(t)

        assert (nested / "auto-dir-test.json").exists()
