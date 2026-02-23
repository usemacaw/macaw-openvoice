"""Tests for the PronunciationStore (filesystem-based pronunciation dictionary persistence).

Validates:
- ABC cannot be instantiated directly
- Create stores a JSON file with UUID dictionary_id and version_id
- Get returns stored dictionary or None for nonexistent
- List all returns all dictionaries or empty list
- Delete removes file and returns True, or False for nonexistent
- Add rules appends rules and regenerates version_id
- Remove rules filters by string_to_match and regenerates version_id
"""

from __future__ import annotations

import json
import os
import uuid

import pytest

from macaw.server.pronunciation.store import (
    FileSystemPronunciationStore,
    PronunciationStore,
)

# --- ABC ---


class TestPronunciationStoreABC:
    """PronunciationStore ABC enforces implementation of all abstract methods."""

    def test_cannot_instantiate_abc(self) -> None:
        with pytest.raises(TypeError, match="abstract method"):
            PronunciationStore()  # type: ignore[abstract]


# --- Create ---


class TestFileSystemPronunciationStoreCreate:
    """FileSystemPronunciationStore.create persists dictionaries as JSON files."""

    async def test_create_stores_json_file(self, tmp_path: object) -> None:
        store = FileSystemPronunciationStore(str(tmp_path))
        rules = [{"string_to_match": "NATO", "type": "alias", "alias": "N.A.T.O."}]

        result = await store.create("Military Terms", rules, description="Military acronyms")

        # Verify file exists on disk
        dictionary_id = str(result["dictionary_id"])
        file_path = os.path.join(str(tmp_path), f"{dictionary_id}.json")
        assert os.path.isfile(file_path)

        # Verify file content matches returned dict
        with open(file_path) as f:
            on_disk = json.load(f)
        assert on_disk["name"] == "Military Terms"
        assert on_disk["description"] == "Military acronyms"
        assert on_disk["rules"] == rules

    async def test_create_assigns_uuid_and_version(self, tmp_path: object) -> None:
        store = FileSystemPronunciationStore(str(tmp_path))

        result = await store.create("Test Dict", [])

        # Both dictionary_id and version_id must be valid UUIDs
        dictionary_id = str(result["dictionary_id"])
        version_id = str(result["version_id"])
        uuid.UUID(dictionary_id)  # Raises ValueError if not a valid UUID
        uuid.UUID(version_id)
        # They should be different UUIDs
        assert dictionary_id != version_id


# --- Get ---


class TestFileSystemPronunciationStoreGet:
    """FileSystemPronunciationStore.get retrieves dictionaries by ID."""

    async def test_get_returns_stored_dictionary(self, tmp_path: object) -> None:
        store = FileSystemPronunciationStore(str(tmp_path))
        created = await store.create(
            "Phonetics", [{"string_to_match": "gif", "type": "phoneme", "phoneme": "jif"}]
        )
        dictionary_id = str(created["dictionary_id"])

        result = await store.get(dictionary_id)

        assert result is not None
        assert result["dictionary_id"] == dictionary_id
        assert result["name"] == "Phonetics"
        assert len(result["rules"]) == 1  # type: ignore[arg-type]

    async def test_get_returns_none_for_nonexistent(self, tmp_path: object) -> None:
        store = FileSystemPronunciationStore(str(tmp_path))

        result = await store.get("00000000-0000-0000-0000-000000000000")

        assert result is None


# --- List All ---


class TestFileSystemPronunciationStoreListAll:
    """FileSystemPronunciationStore.list_all returns all stored dictionaries."""

    async def test_list_all_returns_all_dictionaries(self, tmp_path: object) -> None:
        store = FileSystemPronunciationStore(str(tmp_path))
        await store.create("Dict A", [{"string_to_match": "a", "type": "alias", "alias": "alpha"}])
        await store.create("Dict B", [{"string_to_match": "b", "type": "alias", "alias": "bravo"}])
        await store.create("Dict C", [])

        result = await store.list_all()

        assert len(result) == 3
        names = {str(d["name"]) for d in result}
        assert names == {"Dict A", "Dict B", "Dict C"}

    async def test_list_all_empty_directory(self, tmp_path: object) -> None:
        # Point to a subdirectory that does not exist yet
        store_dir = os.path.join(str(tmp_path), "nonexistent_subdir")
        store = FileSystemPronunciationStore(store_dir)

        result = await store.list_all()

        assert result == []


# --- Delete ---


class TestFileSystemPronunciationStoreDelete:
    """FileSystemPronunciationStore.delete removes dictionary files from disk."""

    async def test_delete_removes_file_and_returns_true(self, tmp_path: object) -> None:
        store = FileSystemPronunciationStore(str(tmp_path))
        created = await store.create("Temporary", [])
        dictionary_id = str(created["dictionary_id"])

        deleted = await store.delete(dictionary_id)

        assert deleted is True
        # File should no longer exist
        file_path = os.path.join(str(tmp_path), f"{dictionary_id}.json")
        assert not os.path.exists(file_path)
        # Get should return None
        assert await store.get(dictionary_id) is None

    async def test_delete_returns_false_for_nonexistent(self, tmp_path: object) -> None:
        store = FileSystemPronunciationStore(str(tmp_path))

        deleted = await store.delete("00000000-0000-0000-0000-000000000000")

        assert deleted is False


# --- Add Rules ---


class TestFileSystemPronunciationStoreAddRules:
    """FileSystemPronunciationStore.add_rules appends rules and rotates version_id."""

    async def test_add_rules_appends_and_updates_version(self, tmp_path: object) -> None:
        store = FileSystemPronunciationStore(str(tmp_path))
        initial_rules = [
            {"string_to_match": "SQL", "type": "alias", "alias": "sequel"},
        ]
        created = await store.create("Tech Terms", initial_rules)
        dictionary_id = str(created["dictionary_id"])
        original_version = str(created["version_id"])

        new_rules = [
            {"string_to_match": "API", "type": "alias", "alias": "A.P.I."},
        ]
        updated = await store.add_rules(dictionary_id, new_rules)

        assert updated is not None
        # Rules should be appended
        rules = updated["rules"]
        assert isinstance(rules, list)
        assert len(rules) == 2
        match_strings = [str(r["string_to_match"]) for r in rules]
        assert match_strings == ["SQL", "API"]
        # Version must have changed
        new_version = str(updated["version_id"])
        assert new_version != original_version
        uuid.UUID(new_version)  # Must still be a valid UUID

    async def test_add_rules_returns_none_for_nonexistent(self, tmp_path: object) -> None:
        store = FileSystemPronunciationStore(str(tmp_path))

        result = await store.add_rules(
            "00000000-0000-0000-0000-000000000000",
            [{"string_to_match": "test", "type": "alias", "alias": "t"}],
        )

        assert result is None


# --- Remove Rules ---


class TestFileSystemPronunciationStoreRemoveRules:
    """FileSystemPronunciationStore.remove_rules filters by string_to_match."""

    async def test_remove_rules_by_string_to_match(self, tmp_path: object) -> None:
        store = FileSystemPronunciationStore(str(tmp_path))
        rules = [
            {"string_to_match": "SQL", "type": "alias", "alias": "sequel"},
            {"string_to_match": "API", "type": "alias", "alias": "A.P.I."},
            {"string_to_match": "URL", "type": "alias", "alias": "U.R.L."},
        ]
        created = await store.create("Tech Terms", rules)
        dictionary_id = str(created["dictionary_id"])
        original_version = str(created["version_id"])

        updated = await store.remove_rules(dictionary_id, ["SQL", "URL"])

        assert updated is not None
        remaining_rules = updated["rules"]
        assert isinstance(remaining_rules, list)
        assert len(remaining_rules) == 1
        assert remaining_rules[0]["string_to_match"] == "API"
        # Version must have changed
        new_version = str(updated["version_id"])
        assert new_version != original_version
        uuid.UUID(new_version)

    async def test_remove_rules_returns_none_for_nonexistent(self, tmp_path: object) -> None:
        store = FileSystemPronunciationStore(str(tmp_path))

        result = await store.remove_rules(
            "00000000-0000-0000-0000-000000000000",
            ["anything"],
        )

        assert result is None
