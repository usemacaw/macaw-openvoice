"""Apply pronunciation dictionary rules to text before TTS synthesis."""

from __future__ import annotations

from typing import TYPE_CHECKING

from macaw.logging import get_logger
from macaw.server.pronunciation.models import PronDictLocator, RuleType

if TYPE_CHECKING:
    from macaw.server.pronunciation.store import PronunciationStore

logger = get_logger("server.pronunciation.applicator")

_MAX_DICTIONARIES = 3


async def apply_pronunciation_rules(
    text: str,
    locators: list[PronDictLocator],
    store: PronunciationStore,
) -> str:
    """Apply alias rules from pronunciation dictionaries to text.

    Dictionaries are applied in order. Only alias rules perform string
    replacement; phoneme rules are logged as warnings (not yet supported
    at runtime).

    Args:
        text: Original text to transform.
        locators: Ordered list of dictionary references (max 3).
        store: Pronunciation store for dictionary lookups.

    Returns:
        Text with alias replacements applied.

    Raises:
        ValueError: If more than 3 dictionaries are requested.
        KeyError: If a referenced dictionary is not found.
    """
    if len(locators) > _MAX_DICTIONARIES:
        msg = f"Maximum {_MAX_DICTIONARIES} pronunciation dictionaries per request"
        raise ValueError(msg)

    for locator in locators:
        dictionary = await store.get(locator.pronunciation_dictionary_id)
        if dictionary is None:
            msg = f"Dictionary not found: {locator.pronunciation_dictionary_id}"
            raise KeyError(msg)

        raw_rules = dictionary.get("rules")
        rules: list[dict[str, object]] = list(raw_rules) if isinstance(raw_rules, list) else []
        for rule in rules:
            rule_type = str(rule.get("rule_type", ""))
            if rule_type == RuleType.ALIAS:
                alias = str(rule.get("alias", ""))
                string_to_match = str(rule.get("string_to_match", ""))
                if alias and string_to_match:
                    text = text.replace(string_to_match, alias)
            elif rule_type == RuleType.PHONEME:
                logger.warning(
                    "phoneme_rule_not_supported",
                    dictionary_id=locator.pronunciation_dictionary_id,
                    string_to_match=str(rule.get("string_to_match", "")),
                )

    return text
