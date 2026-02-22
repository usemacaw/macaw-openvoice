"""Tests for the VoiceStore (filesystem-based voice persistence).

Validates:
- Save and retrieve voices (round-trip)
- Save cloned voice with ref_audio
- Save designed voice without ref_audio
- Get returns None for non-existent voice
- List voices with and without type filter
- Delete voice removes from disk
- Delete non-existent returns False
- Path traversal prevention (C-02)
- Update voice metadata fields
"""

from __future__ import annotations

import json
import os

import pytest

from macaw.exceptions import InvalidRequestError
from macaw.server.voice_store import FileSystemVoiceStore


class TestFileSystemVoiceStoreSave:
    """VoiceStore.save persists voice metadata and optional ref_audio."""

    async def test_save_designed_voice(self, tmp_path: object) -> None:
        store = FileSystemVoiceStore(str(tmp_path))

        saved = await store.save(
            voice_id="v1",
            name="Deep Narrator",
            voice_type="designed",
            instruction="A deep male narrator voice.",
            language="en",
        )

        assert saved.voice_id == "v1"
        assert saved.name == "Deep Narrator"
        assert saved.voice_type == "designed"
        assert saved.instruction == "A deep male narrator voice."
        assert saved.language == "en"
        assert saved.ref_audio_path is None
        assert saved.created_at > 0

    async def test_save_cloned_voice_with_ref_audio(self, tmp_path: object) -> None:
        store = FileSystemVoiceStore(str(tmp_path))
        audio_data = b"\x00\x01\x02\x03" * 100

        saved = await store.save(
            voice_id="v2",
            name="My Clone",
            voice_type="cloned",
            ref_audio=audio_data,
            ref_text="Hello world",
            language="en",
        )

        assert saved.voice_type == "cloned"
        assert saved.ref_audio_path is not None
        assert saved.ref_text == "Hello world"
        # Verify audio file written to disk
        assert os.path.isfile(saved.ref_audio_path)
        with open(saved.ref_audio_path, "rb") as f:
            assert f.read() == audio_data

    async def test_save_writes_metadata_json(self, tmp_path: object) -> None:
        store = FileSystemVoiceStore(str(tmp_path))

        await store.save(
            voice_id="v3",
            name="Test Voice",
            voice_type="designed",
            instruction="Test instruction",
        )

        metadata_path = os.path.join(str(tmp_path), "voices", "v3", "metadata.json")
        assert os.path.isfile(metadata_path)
        with open(metadata_path) as f:
            metadata = json.load(f)
        assert metadata["voice_id"] == "v3"
        assert metadata["name"] == "Test Voice"
        assert metadata["voice_type"] == "designed"
        assert metadata["has_ref_audio"] is False


class TestFileSystemVoiceStoreGet:
    """VoiceStore.get retrieves saved voices by ID."""

    async def test_get_existing_voice(self, tmp_path: object) -> None:
        store = FileSystemVoiceStore(str(tmp_path))
        await store.save(
            voice_id="v1",
            name="My Voice",
            voice_type="designed",
            instruction="Speak softly",
        )

        result = await store.get("v1")

        assert result is not None
        assert result.voice_id == "v1"
        assert result.name == "My Voice"
        assert result.instruction == "Speak softly"

    async def test_get_nonexistent_returns_none(self, tmp_path: object) -> None:
        store = FileSystemVoiceStore(str(tmp_path))

        result = await store.get("nonexistent")

        assert result is None

    async def test_get_cloned_voice_has_ref_audio_path(self, tmp_path: object) -> None:
        store = FileSystemVoiceStore(str(tmp_path))
        await store.save(
            voice_id="v1",
            name="Clone",
            voice_type="cloned",
            ref_audio=b"\x00" * 100,
            ref_text="Test transcript",
        )

        result = await store.get("v1")

        assert result is not None
        assert result.ref_audio_path is not None
        assert os.path.isfile(result.ref_audio_path)


class TestFileSystemVoiceStoreList:
    """VoiceStore.list_voices returns filtered voice lists."""

    async def test_list_empty_store(self, tmp_path: object) -> None:
        store = FileSystemVoiceStore(str(tmp_path))

        result = await store.list_voices()

        assert result == []

    async def test_list_all_voices(self, tmp_path: object) -> None:
        store = FileSystemVoiceStore(str(tmp_path))
        await store.save(voice_id="v1", name="Voice 1", voice_type="designed", instruction="i1")
        await store.save(voice_id="v2", name="Voice 2", voice_type="cloned", ref_audio=b"\x00")

        result = await store.list_voices()

        assert len(result) == 2
        ids = {v.voice_id for v in result}
        assert ids == {"v1", "v2"}

    async def test_list_with_type_filter(self, tmp_path: object) -> None:
        store = FileSystemVoiceStore(str(tmp_path))
        await store.save(voice_id="v1", name="Designed", voice_type="designed", instruction="i1")
        await store.save(voice_id="v2", name="Cloned", voice_type="cloned", ref_audio=b"\x00")

        cloned = await store.list_voices(type_filter="cloned")
        designed = await store.list_voices(type_filter="designed")

        assert len(cloned) == 1
        assert cloned[0].voice_id == "v2"
        assert len(designed) == 1
        assert designed[0].voice_id == "v1"


class TestFileSystemVoiceStoreDelete:
    """VoiceStore.delete removes voices from disk."""

    async def test_delete_existing_voice(self, tmp_path: object) -> None:
        store = FileSystemVoiceStore(str(tmp_path))
        await store.save(voice_id="v1", name="Delete Me", voice_type="designed", instruction="i1")

        deleted = await store.delete("v1")

        assert deleted is True
        # Verify removed from disk
        voice_dir = os.path.join(str(tmp_path), "voices", "v1")
        assert not os.path.exists(voice_dir)

    async def test_delete_nonexistent_returns_false(self, tmp_path: object) -> None:
        store = FileSystemVoiceStore(str(tmp_path))

        deleted = await store.delete("nonexistent")

        assert deleted is False

    async def test_delete_then_get_returns_none(self, tmp_path: object) -> None:
        store = FileSystemVoiceStore(str(tmp_path))
        await store.save(voice_id="v1", name="Temp", voice_type="designed", instruction="i1")

        await store.delete("v1")
        result = await store.get("v1")

        assert result is None


# --- Path Traversal Prevention (C-02) ---


class TestVoiceStorePathTraversal:
    """Validates that path traversal attacks are blocked by voice_id validation."""

    @pytest.mark.parametrize(
        "malicious_id",
        [
            "../../../etc/passwd",
            "..%2F..%2Fetc%2Fpasswd",
            "voice/../../../secret",
            "/absolute/path",
            "voice id with spaces",
            "voice;rm -rf /",
            "voice\x00null",
            "",
        ],
    )
    async def test_save_rejects_malicious_voice_id(
        self, tmp_path: object, malicious_id: str
    ) -> None:
        store = FileSystemVoiceStore(str(tmp_path))

        with pytest.raises(InvalidRequestError, match="Invalid voice_id"):
            await store.save(
                voice_id=malicious_id,
                name="Evil",
                voice_type="designed",
                instruction="test",
            )

    @pytest.mark.parametrize(
        "malicious_id",
        ["../secret", "../../etc/passwd", "voice/../data"],
    )
    async def test_get_rejects_malicious_voice_id(
        self, tmp_path: object, malicious_id: str
    ) -> None:
        store = FileSystemVoiceStore(str(tmp_path))

        with pytest.raises(InvalidRequestError, match="Invalid voice_id"):
            await store.get(malicious_id)

    @pytest.mark.parametrize(
        "malicious_id",
        ["../secret", "../../etc/passwd", "voice/../data"],
    )
    async def test_delete_rejects_malicious_voice_id(
        self, tmp_path: object, malicious_id: str
    ) -> None:
        store = FileSystemVoiceStore(str(tmp_path))

        with pytest.raises(InvalidRequestError, match="Invalid voice_id"):
            await store.delete(malicious_id)

    async def test_valid_voice_id_formats_accepted(self, tmp_path: object) -> None:
        """Legitimate voice IDs with alphanumeric, hyphens, underscores are accepted."""
        store = FileSystemVoiceStore(str(tmp_path))

        for valid_id in ["abc", "voice-1", "my_voice", "A123-B456_C"]:
            saved = await store.save(
                voice_id=valid_id,
                name="Test",
                voice_type="designed",
                instruction="test",
            )
            assert saved.voice_id == valid_id


# --- Update ---


class TestFileSystemVoiceStoreUpdate:
    """VoiceStore.update modifies metadata of existing voices."""

    async def test_update_name(self, tmp_path: object) -> None:
        store = FileSystemVoiceStore(str(tmp_path))
        await store.save(
            voice_id="v1",
            name="Old Name",
            voice_type="designed",
            instruction="test",
        )

        result = await store.update("v1", name="New Name")

        assert result is not None
        assert result.name == "New Name"
        assert result.instruction == "test"

    async def test_update_language(self, tmp_path: object) -> None:
        store = FileSystemVoiceStore(str(tmp_path))
        await store.save(
            voice_id="v1",
            name="Voice",
            voice_type="designed",
            instruction="test",
            language="en",
        )

        result = await store.update("v1", language="pt")

        assert result is not None
        assert result.language == "pt"
        assert result.name == "Voice"

    async def test_update_ref_text(self, tmp_path: object) -> None:
        store = FileSystemVoiceStore(str(tmp_path))
        await store.save(
            voice_id="v1",
            name="Voice",
            voice_type="cloned",
            ref_audio=b"\x00" * 100,
            ref_text="Old text",
        )

        result = await store.update("v1", ref_text="New text")

        assert result is not None
        assert result.ref_text == "New text"

    async def test_update_instruction(self, tmp_path: object) -> None:
        store = FileSystemVoiceStore(str(tmp_path))
        await store.save(
            voice_id="v1",
            name="Voice",
            voice_type="designed",
            instruction="Old instruction",
        )

        result = await store.update("v1", instruction="New instruction")

        assert result is not None
        assert result.instruction == "New instruction"

    async def test_update_nonexistent_returns_none(self, tmp_path: object) -> None:
        store = FileSystemVoiceStore(str(tmp_path))

        result = await store.update("nonexistent", name="New Name")

        assert result is None

    async def test_update_preserves_unmodified_fields(self, tmp_path: object) -> None:
        store = FileSystemVoiceStore(str(tmp_path))
        await store.save(
            voice_id="v1",
            name="Original",
            voice_type="designed",
            instruction="Keep this",
            language="en",
        )

        result = await store.update("v1", name="Updated")

        assert result is not None
        assert result.name == "Updated"
        assert result.instruction == "Keep this"
        assert result.language == "en"
        assert result.voice_type == "designed"

    async def test_update_persists_to_disk(self, tmp_path: object) -> None:
        store = FileSystemVoiceStore(str(tmp_path))
        await store.save(
            voice_id="v1",
            name="Old",
            voice_type="designed",
            instruction="test",
        )

        await store.update("v1", name="New")

        # Read from disk with a new store instance
        fresh_store = FileSystemVoiceStore(str(tmp_path))
        result = await fresh_store.get("v1")
        assert result is not None
        assert result.name == "New"

    async def test_update_multiple_fields_at_once(self, tmp_path: object) -> None:
        store = FileSystemVoiceStore(str(tmp_path))
        await store.save(
            voice_id="v1",
            name="Old",
            voice_type="designed",
            instruction="old instruction",
            language="en",
        )

        result = await store.update("v1", name="New", language="pt", instruction="new instruction")

        assert result is not None
        assert result.name == "New"
        assert result.language == "pt"
        assert result.instruction == "new instruction"

    async def test_update_no_fields_returns_existing(self, tmp_path: object) -> None:
        """Calling update without any non-None fields returns existing voice unchanged."""
        store = FileSystemVoiceStore(str(tmp_path))
        await store.save(
            voice_id="v1",
            name="Existing",
            voice_type="designed",
            instruction="test",
        )

        result = await store.update("v1")

        assert result is not None
        assert result.name == "Existing"

    @pytest.mark.parametrize(
        "malicious_id",
        ["../secret", "../../etc/passwd", "voice/../data"],
    )
    async def test_update_rejects_malicious_voice_id(
        self, tmp_path: object, malicious_id: str
    ) -> None:
        store = FileSystemVoiceStore(str(tmp_path))

        with pytest.raises(InvalidRequestError, match="Invalid voice_id"):
            await store.update(malicious_id, name="Evil")
