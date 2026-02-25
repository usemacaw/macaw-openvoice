"""Pydantic models for pronunciation dictionaries."""

from __future__ import annotations

import enum
import uuid
from datetime import UTC, datetime

from pydantic import BaseModel, Field, model_validator


class RuleType(enum.StrEnum):
    """Type of pronunciation rule."""

    ALIAS = "alias"
    PHONEME = "phoneme"


class PronunciationRule(BaseModel):
    """A single pronunciation replacement rule.

    For ``alias`` rules, ``string_to_match`` is replaced by ``alias`` in the text.
    For ``phoneme`` rules, ``phoneme`` and ``alphabet`` specify the target
    phonetic representation (not yet applied at runtime).
    """

    string_to_match: str = Field(min_length=1)
    rule_type: RuleType
    alias: str | None = None
    phoneme: str | None = None
    alphabet: str | None = None

    @model_validator(mode="after")
    def _validate_rule_fields(self) -> PronunciationRule:
        if self.rule_type == RuleType.ALIAS and not self.alias:
            msg = "alias is required for alias rules"
            raise ValueError(msg)
        if self.rule_type == RuleType.PHONEME and not self.phoneme:
            msg = "phoneme is required for phoneme rules"
            raise ValueError(msg)
        return self


class PronunciationDictionary(BaseModel):
    """A named collection of pronunciation rules."""

    dictionary_id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    name: str = Field(min_length=1, max_length=200)
    description: str = ""
    rules: list[PronunciationRule] = Field(default_factory=list)
    version_id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    created_at: str = Field(default_factory=lambda: datetime.now(UTC).isoformat())


class PronDictLocator(BaseModel):
    """Reference to a pronunciation dictionary for use in TTS requests."""

    pronunciation_dictionary_id: str
    version_id: str | None = None
