"""Voice persistence for saved/cloned voices.

Provides a simple filesystem-based store for voice configurations.
Saved voices can be referenced by ``voice_{id}`` in speech requests,
enabling reuse of cloned voices without re-uploading reference audio.
"""

from __future__ import annotations

import asyncio
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
