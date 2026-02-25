"""Regex-based entity detector implementation."""

from __future__ import annotations

from macaw.postprocessing.entity_detection.interface import DetectedEntity, EntityDetector
from macaw.postprocessing.entity_detection.patterns import ALL_CATEGORIES, ENTITY_PATTERNS


class RegexEntityDetector(EntityDetector):
    """Detect entities using compiled regex patterns.

    Scans text against a set of category-organized patterns and returns
    all matches sorted by character offset.
    """

    def detect(self, text: str, categories: list[str] | None = None) -> list[DetectedEntity]:
        """Detect entities in text, optionally filtered by category.

        Args:
            text: Input text to scan.
            categories: If None or contains "all", detect all categories.
                Otherwise filter to only the requested categories.

        Returns:
            List of detected entities sorted by start_char.
        """
        if not text:
            return []

        active_categories = self._resolve_categories(categories)

        entities: list[DetectedEntity] = []
        for ep in ENTITY_PATTERNS:
            if ep.category not in active_categories:
                continue
            for match in ep.pattern.finditer(text):
                entities.append(
                    DetectedEntity(
                        text=match.group(),
                        entity_type=ep.entity_type,
                        category=ep.category,
                        start_char=match.start(),
                        end_char=match.end(),
                    )
                )

        entities.sort(key=lambda e: e.start_char)
        return entities

    @staticmethod
    def _resolve_categories(categories: list[str] | None) -> frozenset[str]:
        """Resolve category filter to a concrete set of categories."""
        if categories is None or "all" in categories:
            return ALL_CATEGORIES
        return frozenset(categories)
