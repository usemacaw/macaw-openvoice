"""Tests for the pronunciation dictionary applicator module.

Covers apply_pronunciation_rules() — alias replacement, phoneme skip,
dictionary ordering, max-dict validation, missing dict error — and the
pronunciation_dictionary_locators fields on SpeechRequest and TTSSpeakCommand.
"""

from __future__ import annotations

from unittest.mock import AsyncMock

import pytest

from macaw.server.pronunciation.applicator import apply_pronunciation_rules
from macaw.server.pronunciation.models import PronDictLocator

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_store(**get_side_effects: object) -> AsyncMock:
    """Return a mock PronunciationStore with configurable .get() behavior."""
    store: AsyncMock = AsyncMock()
    return store


def _alias_rule(string_to_match: str, alias: str) -> dict[str, object]:
    return {"rule_type": "alias", "string_to_match": string_to_match, "alias": alias}


def _phoneme_rule(string_to_match: str, phoneme: str) -> dict[str, object]:
    return {
        "rule_type": "phoneme",
        "string_to_match": string_to_match,
        "phoneme": phoneme,
        "alphabet": "ipa",
    }


def _locator(dictionary_id: str) -> PronDictLocator:
    return PronDictLocator(pronunciation_dictionary_id=dictionary_id)


# ---------------------------------------------------------------------------
# TestApplyPronunciationRules
# ---------------------------------------------------------------------------


class TestApplyPronunciationRules:
    """Behavior of apply_pronunciation_rules() for various rule combinations."""

    async def test_single_alias_rule_replaces_text(self) -> None:
        # Arrange
        store = AsyncMock()
        store.get.return_value = {
            "dictionary_id": "dict-1",
            "rules": [_alias_rule("NYC", "New York City")],
        }
        locators = [_locator("dict-1")]

        # Act
        result = await apply_pronunciation_rules("Fly to NYC today", locators, store)

        # Assert
        assert result == "Fly to New York City today"
        store.get.assert_awaited_once_with("dict-1")

    async def test_multiple_alias_rules_applied_in_order(self) -> None:
        """Rules within a single dictionary are applied sequentially."""
        store = AsyncMock()
        store.get.return_value = {
            "dictionary_id": "dict-1",
            "rules": [
                _alias_rule("Dr.", "Doctor"),
                _alias_rule("St.", "Street"),
            ],
        }
        locators = [_locator("dict-1")]

        result = await apply_pronunciation_rules("Dr. Smith on St. Elm", locators, store)

        assert result == "Doctor Smith on Street Elm"

    async def test_phoneme_rules_are_skipped(self) -> None:
        """Phoneme rules log a warning but do not modify the text."""
        store = AsyncMock()
        store.get.return_value = {
            "dictionary_id": "dict-1",
            "rules": [_phoneme_rule("tomato", "t@meItoU")],
        }
        locators = [_locator("dict-1")]

        result = await apply_pronunciation_rules("I like tomato soup", locators, store)

        assert result == "I like tomato soup"

    async def test_multiple_dictionaries_applied_in_order(self) -> None:
        """Dictionaries are applied sequentially — dict-1 first, then dict-2."""
        store = AsyncMock()

        async def _get(dictionary_id: str) -> dict[str, object]:
            if dictionary_id == "dict-1":
                return {
                    "dictionary_id": "dict-1",
                    "rules": [_alias_rule("NYC", "New York City")],
                }
            if dictionary_id == "dict-2":
                return {
                    "dictionary_id": "dict-2",
                    "rules": [_alias_rule("New York City", "The Big Apple")],
                }
            return None  # type: ignore[return-value]

        store.get.side_effect = _get
        locators = [_locator("dict-1"), _locator("dict-2")]

        result = await apply_pronunciation_rules("Visit NYC!", locators, store)

        # dict-1 turns "NYC" -> "New York City", then dict-2 turns that -> "The Big Apple"
        assert result == "Visit The Big Apple!"

    async def test_empty_locators_returns_original_text(self) -> None:
        store = AsyncMock()

        result = await apply_pronunciation_rules("Hello world", [], store)

        assert result == "Hello world"
        store.get.assert_not_awaited()

    async def test_max_three_dictionaries_raises_valueerror(self) -> None:
        store = AsyncMock()
        locators = [_locator(f"dict-{i}") for i in range(4)]

        with pytest.raises(ValueError, match="Maximum 3 pronunciation dictionaries"):
            await apply_pronunciation_rules("text", locators, store)

        store.get.assert_not_awaited()

    async def test_dictionary_not_found_raises_keyerror(self) -> None:
        store = AsyncMock()
        store.get.return_value = None
        locators = [_locator("nonexistent")]

        with pytest.raises(KeyError, match="nonexistent"):
            await apply_pronunciation_rules("text", locators, store)

    async def test_empty_rules_returns_original_text(self) -> None:
        """A dictionary with no rules leaves text unchanged."""
        store = AsyncMock()
        store.get.return_value = {
            "dictionary_id": "dict-1",
            "rules": [],
        }
        locators = [_locator("dict-1")]

        result = await apply_pronunciation_rules("unchanged text", locators, store)

        assert result == "unchanged text"


# ---------------------------------------------------------------------------
# TestSpeechRequestPronunciation
# ---------------------------------------------------------------------------


class TestSpeechRequestPronunciation:
    """SpeechRequest.pronunciation_dictionary_locators field behavior."""

    def test_pronunciation_field_defaults_to_none(self) -> None:
        from macaw.server.models.speech import SpeechRequest

        req = SpeechRequest(model="kokoro", input="Hello")

        assert req.pronunciation_dictionary_locators is None

    def test_pronunciation_field_accepts_locators(self) -> None:
        from macaw.server.models.speech import SpeechRequest

        locators = [
            PronDictLocator(pronunciation_dictionary_id="dict-1", version_id="v1"),
            PronDictLocator(pronunciation_dictionary_id="dict-2"),
        ]
        req = SpeechRequest(
            model="kokoro",
            input="Hello",
            pronunciation_dictionary_locators=locators,
        )

        assert req.pronunciation_dictionary_locators is not None
        assert len(req.pronunciation_dictionary_locators) == 2
        assert req.pronunciation_dictionary_locators[0].pronunciation_dictionary_id == "dict-1"
        assert req.pronunciation_dictionary_locators[0].version_id == "v1"
        assert req.pronunciation_dictionary_locators[1].version_id is None


# ---------------------------------------------------------------------------
# TestTTSSpeakCommandPronunciation
# ---------------------------------------------------------------------------


class TestTTSSpeakCommandPronunciation:
    """TTSSpeakCommand.pronunciation_dictionary_locators field behavior."""

    def test_pronunciation_field_defaults_to_none(self) -> None:
        from macaw.server.models.events import TTSSpeakCommand

        cmd = TTSSpeakCommand(text="Hello")

        assert cmd.pronunciation_dictionary_locators is None

    def test_pronunciation_field_accepts_dict_list(self) -> None:
        from macaw.server.models.events import TTSSpeakCommand

        locators = [
            {"pronunciation_dictionary_id": "dict-1", "version_id": "v1"},
            {"pronunciation_dictionary_id": "dict-2"},
        ]
        cmd = TTSSpeakCommand(text="Hello", pronunciation_dictionary_locators=locators)

        assert cmd.pronunciation_dictionary_locators is not None
        assert len(cmd.pronunciation_dictionary_locators) == 2
        assert cmd.pronunciation_dictionary_locators[0]["pronunciation_dictionary_id"] == "dict-1"
