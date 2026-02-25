"""Compiled regex patterns for entity detection organized by category.

Each pattern maps to (entity_type, category). Patterns are compiled once
at module import for performance.
"""

from __future__ import annotations

import re
from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class EntityPattern:
    """A compiled regex pattern with entity metadata."""

    pattern: re.Pattern[str]
    entity_type: str
    category: str


# --- PII patterns ---

_EMAIL_RE = re.compile(
    r"[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}",
)

_PHONE_US_RE = re.compile(
    r"(?:\+1[-.\s]?)?"  # optional +1 prefix
    r"(?:\(\d{3}\)[-.\s]?\d{3}[-.\s]?\d{4}"  # (xxx) xxx-xxxx
    r"|\d{3}[-.\s]\d{3}[-.\s]\d{4})",  # xxx-xxx-xxxx
)

_SSN_RE = re.compile(
    r"\b\d{3}-\d{2}-\d{4}\b",
)

# --- PHI patterns ---

_MRN_RE = re.compile(
    r"\bMRN\s*[:#]?\s*\d{4,12}\b",
    re.IGNORECASE,
)

# --- PCI patterns ---

_CREDIT_CARD_RE = re.compile(
    r"\b(?:\d[ -]?){12,18}\d\b",
)

_IPV4_RE = re.compile(
    r"\b(?:(?:25[0-5]|2[0-4]\d|[01]?\d\d?)\.){3}(?:25[0-5]|2[0-4]\d|[01]?\d\d?)\b",
)

# --- All patterns grouped ---

ENTITY_PATTERNS: tuple[EntityPattern, ...] = (
    EntityPattern(pattern=_EMAIL_RE, entity_type="email_address", category="pii"),
    EntityPattern(pattern=_PHONE_US_RE, entity_type="phone_number", category="pii"),
    EntityPattern(pattern=_SSN_RE, entity_type="ssn", category="pii"),
    EntityPattern(pattern=_MRN_RE, entity_type="medical_record_number", category="phi"),
    EntityPattern(pattern=_CREDIT_CARD_RE, entity_type="credit_card", category="pci"),
    EntityPattern(pattern=_IPV4_RE, entity_type="ip_address", category="pci"),
)

ALL_CATEGORIES: frozenset[str] = frozenset({"pii", "phi", "pci"})
