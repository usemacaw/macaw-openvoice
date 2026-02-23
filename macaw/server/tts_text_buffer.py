"""Accumulates incremental text for TTS synthesis.

Supports token-by-token streaming from LLMs: clients send text chunks
via ``tts.append``, and the buffer splits on sentence/paragraph boundaries
(or manual ``tts.flush``) to trigger synthesis of complete segments.

This class lives on ``SessionContext`` (WS handler level), NOT on Session
Manager. The session FSM is frozen and must not be touched. TTS is
stateless from the runtime perspective.
"""

from __future__ import annotations

import re

from macaw.logging import get_logger

logger = get_logger("server.tts_text_buffer")

# Regex patterns for split strategies.
# Sentence: split after sentence-ending punctuation followed by whitespace or end of string.
_SENTENCE_SPLIT_RE = re.compile(r"(?<=[.!?])\s+")
# Paragraph: split on double newline.
_PARAGRAPH_SPLIT_RE = re.compile(r"\n\n+")


class TTSTextBuffer:
    """Accumulates incremental text for TTS synthesis.

    Lives on SessionContext, NOT on Session Manager.

    Split strategies:
        - ``"sentence"`` (default): split on sentence-ending punctuation
          (``.``, ``!``, ``?``) followed by whitespace or end of string.
        - ``"paragraph"``: split on double newline ``\\n\\n``.
        - ``"none"``: never auto-split; only manual ``flush()`` triggers synthesis.
    """

    def __init__(
        self,
        split_strategy: str = "sentence",
        flush_timeout_ms: int = 5000,
    ) -> None:
        self._buffer: str = ""
        self._request_id: str | None = None
        self._split_strategy = split_strategy
        self._flush_timeout_ms = flush_timeout_ms

    def append(self, text: str, request_id: str) -> list[str]:
        """Append text to buffer. Returns list of splittable segments (may be empty).

        If ``request_id`` changes from the current one, the previous buffer
        content is auto-flushed and returned as the first element before
        the new text is processed.

        Args:
            text: Text chunk to append.
            request_id: Identifier for the current synthesis request.

        Returns:
            List of text segments ready for synthesis. May be empty if no
            complete segment boundary was found yet.
        """
        segments: list[str] = []

        # Auto-flush on request_id change.
        if self._request_id is not None and self._request_id != request_id:
            previous = self._flush_internal()
            if previous is not None:
                segments.append(previous)

        self._request_id = request_id
        self._buffer += text

        # Try to split according to strategy.
        split_segments = self._try_split()
        segments.extend(split_segments)

        return segments

    def flush(self) -> str | None:
        """Flush remaining buffer. Returns text or None if empty."""
        return self._flush_internal()

    def clear(self) -> None:
        """Clear buffer without flushing."""
        self._buffer = ""
        self._request_id = None

    @property
    def request_id(self) -> str | None:
        """Current request ID associated with the buffer content."""
        return self._request_id

    @property
    def pending_text(self) -> str:
        """Text currently accumulated in the buffer."""
        return self._buffer

    @property
    def split_strategy(self) -> str:
        """Current split strategy."""
        return self._split_strategy

    @split_strategy.setter
    def split_strategy(self, value: str) -> None:
        self._split_strategy = value

    @property
    def flush_timeout_ms(self) -> int:
        """Current flush timeout in milliseconds."""
        return self._flush_timeout_ms

    @flush_timeout_ms.setter
    def flush_timeout_ms(self, value: int) -> None:
        self._flush_timeout_ms = value

    def _flush_internal(self) -> str | None:
        """Flush and return buffer content, or None if empty."""
        text = self._buffer.strip()
        self._buffer = ""
        if not text:
            return None
        return text

    def _try_split(self) -> list[str]:
        """Try to split the buffer based on the current strategy.

        Complete segments are returned and removed from the buffer.
        The remainder (incomplete segment) stays in the buffer.
        """
        if self._split_strategy == "none":
            return []

        pattern = _SENTENCE_SPLIT_RE if self._split_strategy == "sentence" else _PARAGRAPH_SPLIT_RE

        parts = pattern.split(self._buffer)
        if len(parts) <= 1:
            # No split point found; everything stays in buffer.
            return []

        # All parts except the last are complete segments.
        complete = parts[:-1]
        self._buffer = parts[-1]

        # Filter out empty/whitespace-only segments.
        return [seg.strip() for seg in complete if seg.strip()]
