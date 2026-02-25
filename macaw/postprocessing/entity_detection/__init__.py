"""Entity detection module for PII/PHI/PCI in transcription text."""

from macaw.postprocessing.entity_detection.interface import DetectedEntity, EntityDetector
from macaw.postprocessing.entity_detection.regex_detector import RegexEntityDetector


def create_entity_detector() -> EntityDetector:
    """Create the default entity detector (regex-based)."""
    return RegexEntityDetector()


__all__ = [
    "DetectedEntity",
    "EntityDetector",
    "RegexEntityDetector",
    "create_entity_detector",
]
