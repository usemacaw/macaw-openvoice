"""Filesystem-based transcript store.

Each transcript is stored as a single JSON file named
``{transcription_id}.json`` under the configured directory.
All blocking I/O is offloaded to a thread via ``asyncio.to_thread``.
"""

from __future__ import annotations

import asyncio
import json
import os
import re
import time

from macaw.exceptions import InvalidRequestError
from macaw.server.transcript_store.interface import StoredTranscript, TranscriptStore

_SAFE_TRANSCRIPT_ID = re.compile(r"^[a-zA-Z0-9_-]+$")


class FileSystemTranscriptStore(TranscriptStore):
    """Filesystem-backed transcript persistence.

    Storage layout::

        {store_path}/
            {transcription_id}.json
    """

    def __init__(self, store_path: str, *, ttl_seconds: int | None = None) -> None:
        self._store_path = store_path
        self._ttl_seconds = ttl_seconds

    @staticmethod
    def _validate_transcript_id(transcription_id: str) -> None:
        """Validate transcription_id to prevent path traversal attacks."""
        if not _SAFE_TRANSCRIPT_ID.match(transcription_id):
            raise InvalidRequestError(f"Invalid transcription_id format: {transcription_id!r}")

    def _ensure_directory(self) -> None:
        os.makedirs(self._store_path, exist_ok=True)

    def _file_path(self, transcription_id: str) -> str:
        return os.path.join(self._store_path, f"{transcription_id}.json")

    def _save_sync(self, transcript: StoredTranscript) -> str:
        self._ensure_directory()
        data = {
            "transcription_id": transcript.transcription_id,
            "text": transcript.text,
            "language": transcript.language,
            "duration": transcript.duration,
            "model": transcript.model,
            "created_at": transcript.created_at,
            "metadata": transcript.metadata,
        }
        path = self._file_path(transcript.transcription_id)
        with open(path, "w") as f:
            json.dump(data, f, indent=2)
        return transcript.transcription_id

    async def save(self, transcript: StoredTranscript) -> str:
        self._validate_transcript_id(transcript.transcription_id)
        return await asyncio.to_thread(self._save_sync, transcript)

    def _get_sync(self, transcription_id: str) -> StoredTranscript | None:
        path = self._file_path(transcription_id)
        if not os.path.isfile(path):
            return None
        with open(path) as f:
            data = json.load(f)
        return StoredTranscript(
            transcription_id=data["transcription_id"],
            text=data["text"],
            language=data.get("language"),
            duration=data.get("duration"),
            model=data.get("model"),
            created_at=data.get("created_at", ""),
            metadata=data.get("metadata", {}),
        )

    async def get(self, transcription_id: str) -> StoredTranscript | None:
        self._validate_transcript_id(transcription_id)
        return await asyncio.to_thread(self._get_sync, transcription_id)

    def _delete_sync(self, transcription_id: str) -> bool:
        path = self._file_path(transcription_id)
        if not os.path.isfile(path):
            return False
        os.remove(path)
        return True

    async def delete(self, transcription_id: str) -> bool:
        self._validate_transcript_id(transcription_id)
        return await asyncio.to_thread(self._delete_sync, transcription_id)

    def _cleanup_expired_sync(self) -> None:
        """Remove expired transcript files based on TTL."""
        if self._ttl_seconds is None:
            return
        if not os.path.isdir(self._store_path):
            return
        now = time.time()
        for filename in os.listdir(self._store_path):
            if not filename.endswith(".json"):
                continue
            path = os.path.join(self._store_path, filename)
            try:
                mtime = os.path.getmtime(path)
                if now - mtime > self._ttl_seconds:
                    os.remove(path)
            except OSError:
                continue

    def _list_all_sync(self, limit: int, offset: int) -> list[StoredTranscript]:
        self._cleanup_expired_sync()
        if not os.path.isdir(self._store_path):
            return []

        entries: list[tuple[float, str]] = []
        for filename in os.listdir(self._store_path):
            if not filename.endswith(".json"):
                continue
            path = os.path.join(self._store_path, filename)
            try:
                mtime = os.path.getmtime(path)
                entries.append((mtime, filename))
            except OSError:
                continue

        # Sort by modification time descending (newest first)
        entries.sort(key=lambda e: e[0], reverse=True)

        result: list[StoredTranscript] = []
        for _, filename in entries[offset : offset + limit]:
            transcription_id = filename.removesuffix(".json")
            transcript = self._get_sync(transcription_id)
            if transcript is not None:
                result.append(transcript)
        return result

    async def list_all(self, limit: int = 100, offset: int = 0) -> list[StoredTranscript]:
        return await asyncio.to_thread(self._list_all_sync, limit, offset)
