"""SSML directive dataclasses — engine-neutral representations.

Each supported SSML tag is parsed into a directive that the TTS engine
can interpret independently of XML structure.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal


@dataclass(frozen=True, slots=True)
class BreakDirective:
    """Silence insertion: ``<break time="500ms"/>``.

    Attributes:
        time_ms: Silence duration in milliseconds.
        position: Character offset in the plain text where the break occurs.
    """

    time_ms: int
    position: int


@dataclass(frozen=True, slots=True)
class ProsodyDirective:
    """Speech rate/pitch control: ``<prosody rate="slow" pitch="high">``.

    Attributes:
        text: The text content inside the prosody tag.
        rate: Speech rate keyword or percentage (e.g., "slow", "120%").
        pitch: Pitch keyword or semitone offset (e.g., "high", "+2st").
        position: Character offset in the plain text.
    """

    text: str
    rate: str | None = None
    pitch: str | None = None
    position: int = 0


@dataclass(frozen=True, slots=True)
class EmphasisDirective:
    """Emphasis control: ``<emphasis level="strong">``.

    Attributes:
        text: The text content inside the emphasis tag.
        level: Emphasis level (strong, moderate, reduced, none).
        position: Character offset in the plain text.
    """

    text: str
    level: Literal["strong", "moderate", "reduced", "none"] = "moderate"
    position: int = 0


@dataclass(frozen=True, slots=True)
class SayAsDirective:
    """Pronunciation hint: ``<say-as interpret-as="cardinal">``.

    Attributes:
        text: The text content inside the say-as tag.
        interpret_as: How to interpret the text (e.g., "cardinal",
            "ordinal", "characters", "date", "telephone").
        format_str: Optional format (e.g., "mdy" for dates).
        position: Character offset in the plain text.
    """

    text: str
    interpret_as: str
    format_str: str | None = None
    position: int = 0


@dataclass(frozen=True, slots=True)
class PhonemeDirective:
    """IPA pronunciation: ``<phoneme alphabet="ipa" ph="...">``.

    Attributes:
        text: The visible text (what appears in output).
        alphabet: Phoneme alphabet ("ipa" or "x-sampa").
        ph: The phoneme string.
        position: Character offset in the plain text.
    """

    text: str
    alphabet: str
    ph: str
    position: int = 0


# Union type for all supported SSML directives.
SSMLDirective = (
    BreakDirective | ProsodyDirective | EmphasisDirective | SayAsDirective | PhonemeDirective
)
