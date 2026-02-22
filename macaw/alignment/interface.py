"""Abstract interface for forced alignment backends.

Forced alignment maps synthesized audio back to the original text with
per-word or per-character timing data. This is used as a fallback when
the TTS engine does not support native alignment (e.g., Qwen3-TTS).
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from macaw._types import TTSAlignmentItem


class Aligner(ABC):
    """Forced alignment interface.

    Implementations produce timing data by aligning synthesized audio
    with the original text. The alignment model is lazy-loaded on first
    use and cached for subsequent calls.
    """

    @abstractmethod
    async def align(
        self,
        audio: bytes,
        text: str,
        sample_rate: int,
        language: str = "en",
        granularity: str = "word",
    ) -> tuple[TTSAlignmentItem, ...]:
        """Align text to audio, returning timing data.

        Args:
            audio: 16-bit PCM audio bytes (mono).
            text: Original text that was synthesized.
            sample_rate: Audio sample rate in Hz.
            language: Language code for alignment model selection.
            granularity: ``"word"`` or ``"character"``.

        Returns:
            Tuple of alignment items with timing data.
        """
        ...
