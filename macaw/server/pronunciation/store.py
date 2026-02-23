"""Pronunciation dictionary persistence.

Provides a filesystem-based store following the VoiceStore pattern.
Each dictionary is stored as a single JSON file in the configured directory.
"""

from __future__ import annotations

import asyncio
import json
import os
import uuid
from abc import ABC, abstractmethod


class PronunciationStore(ABC):
    """Abstract interface for pronunciation dictionary persistence."""

    @abstractmethod
    async def create(
        self,
        name: str,
        rules: list[dict[str, object]],
        *,
        description: str = "",
    ) -> dict[str, object]:
        """Create a new pronunciation dictionary.

        Returns the full dictionary as a dict (JSON-serializable).
        """
        ...

    @abstractmethod
    async def get(self, dictionary_id: str) -> dict[str, object] | None:
        """Retrieve a dictionary by ID. Returns None if not found."""
        ...

    @abstractmethod
    async def list_all(self) -> list[dict[str, object]]:
        """List all dictionaries."""
        ...

    @abstractmethod
    async def delete(self, dictionary_id: str) -> bool:
        """Delete a dictionary. Returns True if deleted, False if not found."""
        ...

    @abstractmethod
    async def add_rules(
        self,
        dictionary_id: str,
        rules: list[dict[str, object]],
    ) -> dict[str, object] | None:
        """Add rules to an existing dictionary.

        Regenerates version_id. Returns updated dictionary or None if not found.
        """
        ...

    @abstractmethod
    async def remove_rules(
        self,
        dictionary_id: str,
        rule_strings: list[str],
    ) -> dict[str, object] | None:
        """Remove rules by string_to_match from a dictionary.

        Regenerates version_id. Returns updated dictionary or None if not found.
        """
        ...


class FileSystemPronunciationStore(PronunciationStore):
    """Filesystem-based pronunciation store.

    Storage layout::

        {base_dir}/{dictionary_id}.json
    """

    def __init__(self, base_dir: str) -> None:
        self._base_dir = base_dir

    def _ensure_dir(self) -> None:
        os.makedirs(self._base_dir, exist_ok=True)

    def _path_for(self, dictionary_id: str) -> str:
        return os.path.join(self._base_dir, f"{dictionary_id}.json")

    def _create_sync(
        self,
        name: str,
        rules: list[dict[str, object]],
        description: str,
    ) -> dict[str, object]:
        self._ensure_dir()
        from datetime import UTC, datetime

        dictionary: dict[str, object] = {
            "dictionary_id": str(uuid.uuid4()),
            "name": name,
            "description": description,
            "rules": rules,
            "version_id": str(uuid.uuid4()),
            "created_at": datetime.now(UTC).isoformat(),
        }
        path = self._path_for(str(dictionary["dictionary_id"]))
        with open(path, "w", encoding="utf-8") as f:
            json.dump(dictionary, f, indent=2)
        return dictionary

    async def create(
        self,
        name: str,
        rules: list[dict[str, object]],
        *,
        description: str = "",
    ) -> dict[str, object]:
        return await asyncio.to_thread(self._create_sync, name, rules, description)

    def _get_sync(self, dictionary_id: str) -> dict[str, object] | None:
        path = self._path_for(dictionary_id)
        if not os.path.isfile(path):
            return None
        with open(path, encoding="utf-8") as f:
            return json.load(f)  # type: ignore[no-any-return]

    async def get(self, dictionary_id: str) -> dict[str, object] | None:
        return await asyncio.to_thread(self._get_sync, dictionary_id)

    def _list_all_sync(self) -> list[dict[str, object]]:
        if not os.path.isdir(self._base_dir):
            return []
        result: list[dict[str, object]] = []
        for filename in sorted(os.listdir(self._base_dir)):
            if not filename.endswith(".json"):
                continue
            path = os.path.join(self._base_dir, filename)
            with open(path, encoding="utf-8") as f:
                result.append(json.load(f))
        return result

    async def list_all(self) -> list[dict[str, object]]:
        return await asyncio.to_thread(self._list_all_sync)

    def _delete_sync(self, dictionary_id: str) -> bool:
        path = self._path_for(dictionary_id)
        if not os.path.isfile(path):
            return False
        os.remove(path)
        return True

    async def delete(self, dictionary_id: str) -> bool:
        return await asyncio.to_thread(self._delete_sync, dictionary_id)

    def _add_rules_sync(
        self,
        dictionary_id: str,
        rules: list[dict[str, object]],
    ) -> dict[str, object] | None:
        data = self._get_sync(dictionary_id)
        if data is None:
            return None
        raw_rules = data.get("rules")
        existing_rules: list[dict[str, object]] = (
            list(raw_rules) if isinstance(raw_rules, list) else []
        )
        existing_rules.extend(rules)
        data["rules"] = existing_rules
        data["version_id"] = str(uuid.uuid4())
        path = self._path_for(dictionary_id)
        with open(path, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2)
        return data

    async def add_rules(
        self,
        dictionary_id: str,
        rules: list[dict[str, object]],
    ) -> dict[str, object] | None:
        return await asyncio.to_thread(self._add_rules_sync, dictionary_id, rules)

    def _remove_rules_sync(
        self,
        dictionary_id: str,
        rule_strings: list[str],
    ) -> dict[str, object] | None:
        data = self._get_sync(dictionary_id)
        if data is None:
            return None
        raw_rules = data.get("rules")
        existing_rules: list[dict[str, object]] = (
            list(raw_rules) if isinstance(raw_rules, list) else []
        )
        to_remove = set(rule_strings)
        data["rules"] = [
            r for r in existing_rules if str(r.get("string_to_match", "")) not in to_remove
        ]
        data["version_id"] = str(uuid.uuid4())
        path = self._path_for(dictionary_id)
        with open(path, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2)
        return data

    async def remove_rules(
        self,
        dictionary_id: str,
        rule_strings: list[str],
    ) -> dict[str, object] | None:
        return await asyncio.to_thread(self._remove_rules_sync, dictionary_id, rule_strings)
