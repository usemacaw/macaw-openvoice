"""SSML directive-to-engine parameter mapping.

Converts engine-neutral SSML directives into engine-specific synthesis
parameters (speed, silence insertions, text modifications).
"""

from __future__ import annotations

from macaw._audio_constants import TTS_DEFAULT_SAMPLE_RATE
from macaw.logging import get_logger
from macaw.postprocessing.ssml.directives import (
    BreakDirective,
    EmphasisDirective,
    PhonemeDirective,
    ProsodyDirective,
    SayAsDirective,
    SSMLDirective,
)

logger = get_logger("workers.tts.ssml_mapper")

# Prosody rate keyword → speed multiplier
_PROSODY_RATE_MAP: dict[str, float] = {
    "x-slow": 0.5,
    "slow": 0.75,
    "medium": 1.0,
    "fast": 1.25,
    "x-fast": 1.5,
}

# Emphasis level → speed multiplier (slower = more emphasis)
_EMPHASIS_SPEED_MAP: dict[str, float] = {
    "strong": 0.8,
    "moderate": 0.9,
    "reduced": 1.1,
    "none": 1.0,
}


def map_ssml_directives(
    text: str,
    directives: list[SSMLDirective],
    *,
    sample_rate: int = TTS_DEFAULT_SAMPLE_RATE,
    engine_name: str = "unknown",
) -> SSMLMappingResult:
    """Map SSML directives to engine-neutral synthesis parameters.

    Returns an SSMLMappingResult with the processed text, silence
    insertions, and speed override.

    Args:
        text: Plain text extracted from SSML parser.
        directives: List of SSMLDirective objects.
        sample_rate: Audio sample rate for silence generation.
        engine_name: Engine name for logging unsupported directives.

    Returns:
        SSMLMappingResult with synthesis parameters.
    """
    silence_insertions: list[tuple[int, bytes]] = []
    speed_override: float | None = None

    for directive in directives:
        if isinstance(directive, BreakDirective):
            num_samples = int(sample_rate * directive.time_ms / 1000)
            silence_bytes = b"\x00\x00" * num_samples
            silence_insertions.append((directive.position, silence_bytes))

        elif isinstance(directive, ProsodyDirective):
            if directive.rate is not None:
                speed = _parse_prosody_rate(directive.rate)
                if speed is not None:
                    speed_override = speed
                else:
                    logger.warning(
                        "ssml_unsupported_prosody_rate",
                        rate=directive.rate,
                        engine=engine_name,
                    )

        elif isinstance(directive, EmphasisDirective):
            speed = _EMPHASIS_SPEED_MAP.get(directive.level)
            if speed is not None and speed != 1.0:
                speed_override = speed

        elif isinstance(directive, SayAsDirective):
            # SayAs is a text-level hint — engines that support it
            # can use interpret_as to adjust pronunciation.
            # Default: log and pass through (text already extracted by parser).
            logger.debug(
                "ssml_say_as_directive",
                interpret_as=directive.interpret_as,
                text=directive.text,
                engine=engine_name,
            )

        elif isinstance(directive, PhonemeDirective):
            # Phoneme is engine-specific — log for now.
            logger.debug(
                "ssml_phoneme_directive",
                alphabet=directive.alphabet,
                ph=directive.ph,
                text=directive.text,
                engine=engine_name,
            )

        else:
            logger.warning(
                "ssml_unknown_directive_type",
                directive_type=type(directive).__name__,
                engine=engine_name,
            )

    return SSMLMappingResult(
        text=text,
        silence_insertions=silence_insertions,
        speed_override=speed_override,
    )


def _parse_prosody_rate(rate: str) -> float | None:
    """Parse a prosody rate value to a speed multiplier.

    Supports keywords ("slow", "fast", etc.) and percentage ("120%").
    """
    # Keyword lookup
    keyword_speed = _PROSODY_RATE_MAP.get(rate.lower())
    if keyword_speed is not None:
        return keyword_speed

    # Percentage (e.g., "120%", "80%")
    stripped = rate.strip()
    if stripped.endswith("%"):
        try:
            pct = float(stripped[:-1])
            return pct / 100.0
        except ValueError:
            return None

    return None


class SSMLMappingResult:
    """Result of mapping SSML directives to synthesis parameters.

    Attributes:
        text: The processed text (may be modified by say-as/phoneme).
        silence_insertions: List of (position, pcm_bytes) for breaks.
        speed_override: Speed multiplier from prosody/emphasis (or None).
    """

    __slots__ = ("silence_insertions", "speed_override", "text")

    def __init__(
        self,
        text: str,
        silence_insertions: list[tuple[int, bytes]],
        speed_override: float | None = None,
    ) -> None:
        self.text = text
        self.silence_insertions = silence_insertions
        self.speed_override = speed_override
