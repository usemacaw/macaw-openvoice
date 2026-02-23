"""Pronunciation dictionary CRUD endpoints."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from fastapi import APIRouter, Depends, Request

from macaw.exceptions import InvalidRequestError, PronunciationDictionaryNotFoundError
from macaw.logging import get_logger
from macaw.server.dependencies import require_pronunciation_store
from macaw.server.pronunciation.models import PronunciationRule

if TYPE_CHECKING:
    from macaw.server.pronunciation.store import PronunciationStore

router = APIRouter(tags=["Pronunciation"])

logger = get_logger("server.routes.pronunciation")


@router.post("/v1/pronunciation-dictionaries/add-from-rules", status_code=201)
async def create_dictionary(
    request: Request,
    store: PronunciationStore = Depends(require_pronunciation_store),  # noqa: B008
) -> dict[str, object]:
    """Create a pronunciation dictionary from a list of rules."""
    body: dict[str, Any] = await request.json()

    name = body.get("name")
    if not name or not isinstance(name, str):
        raise InvalidRequestError("'name' is required and must be a non-empty string.")

    if len(name) > 200:
        raise InvalidRequestError("'name' must be at most 200 characters.")

    description = body.get("description", "")
    if not isinstance(description, str):
        raise InvalidRequestError("'description' must be a string.")

    raw_rules = body.get("rules", [])
    if not isinstance(raw_rules, list):
        raise InvalidRequestError("'rules' must be a list.")

    # Validate each rule via Pydantic model
    validated_rules: list[dict[str, object]] = []
    for rule_data in raw_rules:
        rule = PronunciationRule.model_validate(rule_data)
        validated_rules.append(rule.model_dump())

    dictionary = await store.create(name, validated_rules, description=description)

    logger.info(
        "pronunciation_dictionary_created",
        dictionary_id=dictionary["dictionary_id"],
        rule_count=len(validated_rules),
    )

    return dictionary


@router.get("/v1/pronunciation-dictionaries")
async def list_dictionaries(
    store: PronunciationStore = Depends(require_pronunciation_store),  # noqa: B008
) -> list[dict[str, object]]:
    """List all pronunciation dictionaries."""
    return await store.list_all()


@router.get("/v1/pronunciation-dictionaries/{dictionary_id}")
async def get_dictionary(
    dictionary_id: str,
    store: PronunciationStore = Depends(require_pronunciation_store),  # noqa: B008
) -> dict[str, object]:
    """Get a pronunciation dictionary by ID."""
    dictionary = await store.get(dictionary_id)
    if dictionary is None:
        raise PronunciationDictionaryNotFoundError(dictionary_id)
    return dictionary


@router.delete("/v1/pronunciation-dictionaries/{dictionary_id}")
async def delete_dictionary(
    dictionary_id: str,
    store: PronunciationStore = Depends(require_pronunciation_store),  # noqa: B008
) -> dict[str, object]:
    """Delete a pronunciation dictionary."""
    deleted = await store.delete(dictionary_id)
    if not deleted:
        raise PronunciationDictionaryNotFoundError(dictionary_id)

    logger.info("pronunciation_dictionary_deleted", dictionary_id=dictionary_id)

    return {"deleted": True, "dictionary_id": dictionary_id}


@router.post("/v1/pronunciation-dictionaries/{dictionary_id}/rules")
async def add_rules(
    dictionary_id: str,
    request: Request,
    store: PronunciationStore = Depends(require_pronunciation_store),  # noqa: B008
) -> dict[str, object]:
    """Add rules to an existing pronunciation dictionary."""
    body: dict[str, Any] = await request.json()

    raw_rules = body.get("rules", [])
    if not isinstance(raw_rules, list) or not raw_rules:
        raise InvalidRequestError("'rules' must be a non-empty list.")

    validated_rules: list[dict[str, object]] = []
    for rule_data in raw_rules:
        rule = PronunciationRule.model_validate(rule_data)
        validated_rules.append(rule.model_dump())

    updated = await store.add_rules(dictionary_id, validated_rules)
    if updated is None:
        raise PronunciationDictionaryNotFoundError(dictionary_id)

    logger.info(
        "pronunciation_rules_added",
        dictionary_id=dictionary_id,
        rule_count=len(validated_rules),
    )

    return updated


@router.delete("/v1/pronunciation-dictionaries/{dictionary_id}/rules")
async def remove_rules(
    dictionary_id: str,
    request: Request,
    store: PronunciationStore = Depends(require_pronunciation_store),  # noqa: B008
) -> dict[str, object]:
    """Remove rules by string_to_match from a pronunciation dictionary."""
    body: dict[str, Any] = await request.json()

    rule_strings = body.get("rule_strings", [])
    if not isinstance(rule_strings, list) or not rule_strings:
        raise InvalidRequestError("'rule_strings' must be a non-empty list.")

    for s in rule_strings:
        if not isinstance(s, str):
            raise InvalidRequestError("Each entry in 'rule_strings' must be a string.")

    updated = await store.remove_rules(dictionary_id, rule_strings)
    if updated is None:
        raise PronunciationDictionaryNotFoundError(dictionary_id)

    logger.info(
        "pronunciation_rules_removed",
        dictionary_id=dictionary_id,
        removed_strings=rule_strings,
    )

    return updated
