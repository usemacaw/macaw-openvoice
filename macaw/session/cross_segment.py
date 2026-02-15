"""CrossSegmentContext — cross-segment context for continuity.

Stores the last N tokens (words) from the most recent transcript.final
as initial_prompt for the next segment. Improves continuity for words
cut at segment boundaries.

Applicable only to engines that support conditioning (Whisper).
For CTC/streaming-native, the context is ignored by the worker.

Reference: PRD section RF-12 (Cross-Segment Context).
"""

from __future__ import annotations


class CrossSegmentContext:
    """Cross-segment context for conditioning the next segment.

    Stores the last ``max_tokens`` words from the most recent transcript.final.
    Used as ``initial_prompt`` in the next worker inference, improving continuity
    for phrases cut at segment boundaries.

    The default value of 224 tokens corresponds to half of Whisper's context
    window (448 tokens), as specified in the PRD.

    Typical usage:
        context = CrossSegmentContext(max_tokens=224)
        context.update("hello how can I help")
        prompt = context.get_prompt()  # "hello how can I help"
    """

    __slots__ = ("_max_tokens", "_text")

    def __init__(self, max_tokens: int = 224) -> None:
        self._max_tokens = max_tokens
        self._text: str | None = None

    def update(self, text: str) -> None:
        """Update context with transcript.final text.

        Stores the last ``max_tokens`` words from the provided text.
        If the text has more words than max_tokens, truncate from the start
        (keeping the most recent words).

        Args:
            text: transcript.final text emitted to the client.
        """
        stripped = text.strip()
        if not stripped:
            self._text = None
            return

        words = stripped.split()
        if len(words) > self._max_tokens:
            words = words[-self._max_tokens :]

        self._text = " ".join(words)

    def get_prompt(self) -> str | None:
        """Return stored context or None if empty.

        Returns:
            Text of the last max_tokens words from the previous transcript.final,
            or None if no context was recorded.
        """
        return self._text

    def reset(self) -> None:
        """Clear stored context."""
        self._text = None
