"""Voice persistence for saved/cloned voices.

Provides a simple filesystem-based store for voice configurations.
Saved voices can be referenced by ``voice_{id}`` in speech requests,
enabling reuse of cloned voices without re-uploading reference audio.
"""

from __future__ import annotations

import asyncio
import dataclasses
import json
import os
import re
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

from macaw.exceptions import InvalidRequestError

if TYPE_CHECKING:
    from macaw._types import VoiceTypeLiteral

_SAFE_VOICE_ID = re.compile(r"^[a-zA-Z0-9_-]+$")


@dataclass(frozen=True, slots=True)
class SavedVoice:
    """A persisted voice configuration."""

    voice_id: str
    name: str
    voice_type: VoiceTypeLiteral
    language: str | None = None
    ref_text: str | None = None
    instruction: str | None = None
    ref_audio_path: str | None = None
    shared: bool = False
    created_at: float = field(default_factory=time.time)


class VoiceStore(ABC):
    """Abstract interface for voice persistence."""

    @abstractmethod
    async def save(
        self,
        voice_id: str,
        name: str,
        voice_type: VoiceTypeLiteral,
        ref_audio: bytes | None = None,
        *,
        language: str | None = None,
        ref_text: str | None = None,
        instruction: str | None = None,
    ) -> SavedVoice:
        """Save a voice configuration.

        Args:
            voice_id: Unique identifier for the voice.
            name: Human-readable name.
            voice_type: "cloned" or "designed".
            ref_audio: Reference audio bytes (only for type=cloned).
            language: Target language.
            ref_text: Reference transcript.
            instruction: Voice design instruction.

        Returns:
            The saved voice.
        """
        ...

    @abstractmethod
    async def get(self, voice_id: str) -> SavedVoice | None:
        """Retrieve a saved voice by ID.

        Returns None if voice not found.
        """
        ...

    @abstractmethod
    async def list_voices(self, type_filter: str | None = None) -> list[SavedVoice]:
        """List all saved voices, optionally filtered by type."""
        ...

    @abstractmethod
    async def delete(self, voice_id: str) -> bool:
        """Delete a saved voice.

        Returns True if the voice was deleted, False if not found.
        """
        ...

    @abstractmethod
    async def update(
        self,
        voice_id: str,
        *,
        name: str | None = None,
        language: str | None = None,
        ref_text: str | None = None,
        instruction: str | None = None,
    ) -> SavedVoice | None:
        """Update a saved voice's metadata fields.

        Only non-None fields are updated. Returns None if voice not found.
        """
        ...

    @abstractmethod
    async def set_shared(self, voice_id: str, *, shared: bool) -> SavedVoice | None:
        """Set the shared flag on a voice.

        Returns the updated voice, or None if not found.
        """
        ...

    @abstractmethod
    async def list_shared(self) -> list[SavedVoice]:
        """List all voices with shared=True."""
        ...


class FileSystemVoiceStore(VoiceStore):
    """Filesystem-based voice store.

    Storage layout::

        {base_dir}/voices/{voice_id}/
            metadata.json  — voice configuration
            ref_audio.wav  — reference audio (only for type=cloned)
    """

    def __init__(self, base_dir: str) -> None:
        self._base_dir = base_dir
        self._voices_dir = os.path.join(base_dir, "voices")

    @staticmethod
    def _validate_voice_id(voice_id: str) -> None:
        """Validate voice_id to prevent path traversal attacks."""
        if not _SAFE_VOICE_ID.match(voice_id):
            raise InvalidRequestError(f"Invalid voice_id format: {voice_id!r}")

    def _save_sync(
        self,
        voice_id: str,
        name: str,
        voice_type: VoiceTypeLiteral,
        ref_audio: bytes | None,
        *,
        language: str | None,
        ref_text: str | None,
        instruction: str | None,
    ) -> SavedVoice:
        voice_dir = os.path.join(self._voices_dir, voice_id)
        os.makedirs(voice_dir, exist_ok=True)

        ref_audio_path: str | None = None
        if ref_audio is not None:
            ref_audio_path = os.path.join(voice_dir, "ref_audio.wav")
            with open(ref_audio_path, "wb") as f:
                f.write(ref_audio)

        created_at = time.time()
        metadata = {
            "voice_id": voice_id,
            "name": name,
            "voice_type": voice_type,
            "language": language,
            "ref_text": ref_text,
            "instruction": instruction,
            "has_ref_audio": ref_audio is not None,
            "shared": False,
            "created_at": created_at,
        }

        metadata_path = os.path.join(voice_dir, "metadata.json")
        with open(metadata_path, "w") as f:
            json.dump(metadata, f, indent=2)

        return SavedVoice(
            voice_id=voice_id,
            name=name,
            voice_type=voice_type,
            language=language,
            ref_text=ref_text,
            instruction=instruction,
            ref_audio_path=ref_audio_path,
            created_at=created_at,
        )

    async def save(
        self,
        voice_id: str,
        name: str,
        voice_type: VoiceTypeLiteral,
        ref_audio: bytes | None = None,
        *,
        language: str | None = None,
        ref_text: str | None = None,
        instruction: str | None = None,
    ) -> SavedVoice:
        self._validate_voice_id(voice_id)
        return await asyncio.to_thread(
            self._save_sync,
            voice_id,
            name,
            voice_type,
            ref_audio,
            language=language,
            ref_text=ref_text,
            instruction=instruction,
        )

    def _get_sync(self, voice_id: str) -> SavedVoice | None:
        metadata_path = os.path.join(self._voices_dir, voice_id, "metadata.json")
        if not os.path.isfile(metadata_path):
            return None

        with open(metadata_path) as f:
            metadata = json.load(f)

        ref_audio_path: str | None = None
        if metadata.get("has_ref_audio"):
            candidate = os.path.join(self._voices_dir, voice_id, "ref_audio.wav")
            if os.path.isfile(candidate):
                ref_audio_path = candidate

        return SavedVoice(
            voice_id=metadata["voice_id"],
            name=metadata["name"],
            voice_type=metadata["voice_type"],
            language=metadata.get("language"),
            ref_text=metadata.get("ref_text"),
            instruction=metadata.get("instruction"),
            ref_audio_path=ref_audio_path,
            shared=metadata.get("shared", False),
            created_at=metadata.get("created_at", 0.0),
        )

    async def get(self, voice_id: str) -> SavedVoice | None:
        self._validate_voice_id(voice_id)
        return await asyncio.to_thread(self._get_sync, voice_id)

    async def list_voices(self, type_filter: str | None = None) -> list[SavedVoice]:
        return await asyncio.to_thread(self._list_voices_sync, type_filter)

    def _list_voices_sync(self, type_filter: str | None) -> list[SavedVoice]:
        entries = self._list_entries()
        result: list[SavedVoice] = []
        for entry in entries:
            voice = self._get_sync(entry)
            if voice is None:
                continue
            if type_filter is not None and voice.voice_type != type_filter:
                continue
            result.append(voice)
        return result

    def _list_entries(self) -> list[str]:
        if not os.path.isdir(self._voices_dir):
            return []
        return sorted(os.listdir(self._voices_dir))

    def _delete_sync(self, voice_id: str) -> bool:
        voice_dir = os.path.join(self._voices_dir, voice_id)
        if not os.path.isdir(voice_dir):
            return False

        import shutil

        shutil.rmtree(voice_dir)
        return True

    async def delete(self, voice_id: str) -> bool:
        self._validate_voice_id(voice_id)
        return await asyncio.to_thread(self._delete_sync, voice_id)

    def _update_sync(
        self,
        voice_id: str,
        *,
        name: str | None,
        language: str | None,
        ref_text: str | None,
        instruction: str | None,
    ) -> SavedVoice | None:
        existing = self._get_sync(voice_id)
        if existing is None:
            return None

        updated = dataclasses.replace(
            existing,
            name=name if name is not None else existing.name,
            language=language if language is not None else existing.language,
            ref_text=ref_text if ref_text is not None else existing.ref_text,
            instruction=instruction if instruction is not None else existing.instruction,
        )
        if updated == existing:
            return existing

        self._write_metadata(voice_id, updated)
        return updated

    def _write_metadata(self, voice_id: str, voice: SavedVoice) -> None:
        """Write voice metadata JSON to disk."""
        metadata = {
            "voice_id": voice.voice_id,
            "name": voice.name,
            "voice_type": voice.voice_type,
            "language": voice.language,
            "ref_text": voice.ref_text,
            "instruction": voice.instruction,
            "has_ref_audio": voice.ref_audio_path is not None,
            "shared": voice.shared,
            "created_at": voice.created_at,
        }
        metadata_path = os.path.join(self._voices_dir, voice_id, "metadata.json")
        with open(metadata_path, "w") as f:
            json.dump(metadata, f, indent=2)

    async def update(
        self,
        voice_id: str,
        *,
        name: str | None = None,
        language: str | None = None,
        ref_text: str | None = None,
        instruction: str | None = None,
    ) -> SavedVoice | None:
        self._validate_voice_id(voice_id)
        return await asyncio.to_thread(
            self._update_sync,
            voice_id,
            name=name,
            language=language,
            ref_text=ref_text,
            instruction=instruction,
        )

    def _set_shared_sync(self, voice_id: str, *, shared: bool) -> SavedVoice | None:
        existing = self._get_sync(voice_id)
        if existing is None:
            return None

        updated = dataclasses.replace(existing, shared=shared)
        if updated == existing:
            return existing

        self._write_metadata(voice_id, updated)
        return updated

    async def set_shared(self, voice_id: str, *, shared: bool) -> SavedVoice | None:
        self._validate_voice_id(voice_id)
        return await asyncio.to_thread(self._set_shared_sync, voice_id, shared=shared)

    def _list_shared_sync(self) -> list[SavedVoice]:
        entries = self._list_entries()
        result: list[SavedVoice] = []
        for entry in entries:
            voice = self._get_sync(entry)
            if voice is not None and voice.shared:
                result.append(voice)
        return result

    async def list_shared(self) -> list[SavedVoice]:
        return await asyncio.to_thread(self._list_shared_sync)
