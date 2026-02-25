"""Multi-context TTS state management for WebSocket connections.

Tracks multiple independent TTS contexts on a single WebSocket.
Each context has its own text buffer, cancellation event, and lifecycle state.
See ADR-008 for design rationale.
"""

from __future__ import annotations

import asyncio
import time
from dataclasses import dataclass, field
from enum import StrEnum

from macaw.logging import get_logger
from macaw.server.tts_text_buffer import TTSTextBuffer

logger = get_logger("server.ws_context_manager")


class ContextState(StrEnum):
    """Lifecycle states for a TTS context."""

    ACTIVE = "active"
    FLUSHING = "flushing"
    CLOSED = "closed"


@dataclass
class TTSContext:
    """State for a single TTS context within a multi-context connection."""

    context_id: str
    text_buffer: TTSTextBuffer = field(default_factory=TTSTextBuffer)
    cancel_event: asyncio.Event = field(default_factory=asyncio.Event)
    state: ContextState = ContextState.ACTIVE
    tts_task: asyncio.Task[None] | None = None
    created_at: float = field(default_factory=time.monotonic)
    last_activity: float = field(default_factory=time.monotonic)

    def touch(self) -> None:
        """Update last activity timestamp."""
        self.last_activity = time.monotonic()

    @property
    def is_active(self) -> bool:
        """Whether the context is in ACTIVE or FLUSHING state."""
        return self.state != ContextState.CLOSED


class WSContextManager:
    """Manages multiple TTS contexts for a single WebSocket connection.

    Each context is identified by a client-supplied string ``context_id``.
    The manager enforces a maximum number of concurrent contexts and
    provides inactivity timeout detection.
    """

    def __init__(
        self,
        max_contexts: int = 10,
        inactivity_timeout_s: float = 20.0,
    ) -> None:
        self._contexts: dict[str, TTSContext] = {}
        self._max_contexts = max_contexts
        self._inactivity_timeout_s = inactivity_timeout_s

    @property
    def max_contexts(self) -> int:
        """Maximum number of concurrent contexts."""
        return self._max_contexts

    @property
    def inactivity_timeout_s(self) -> float:
        """Inactivity timeout per context in seconds."""
        return self._inactivity_timeout_s

    @property
    def active_count(self) -> int:
        """Number of non-closed contexts."""
        return sum(1 for ctx in self._contexts.values() if ctx.is_active)

    @property
    def context_ids(self) -> list[str]:
        """List of all context IDs (including closed)."""
        return list(self._contexts.keys())

    def create_context(self, context_id: str) -> TTSContext:
        """Create a new TTS context.

        Args:
            context_id: Client-supplied identifier for the context.

        Returns:
            The newly created TTSContext.

        Raises:
            ValueError: If ``context_id`` already exists or max contexts reached.
        """
        if context_id in self._contexts:
            existing = self._contexts[context_id]
            if existing.is_active:
                raise ValueError(f"Context '{context_id}' already exists")
            # Re-create over a closed context
            del self._contexts[context_id]

        if self.active_count >= self._max_contexts:
            raise ValueError(f"Maximum contexts ({self._max_contexts}) reached")

        ctx = TTSContext(context_id=context_id)
        self._contexts[context_id] = ctx
        logger.info(
            "context_created",
            context_id=context_id,
            active_count=self.active_count,
        )
        return ctx

    def get_context(self, context_id: str) -> TTSContext | None:
        """Get a context by ID, or None if not found."""
        return self._contexts.get(context_id)

    def get_active_context(self, context_id: str) -> TTSContext | None:
        """Get a context by ID only if it is active (not closed)."""
        ctx = self._contexts.get(context_id)
        if ctx is not None and ctx.is_active:
            return ctx
        return None

    def close_context(self, context_id: str) -> bool:
        """Close a context and cancel any active TTS task.

        Returns True if the context was closed, False if not found.
        """
        ctx = self._contexts.get(context_id)
        if ctx is None:
            return False

        if ctx.state == ContextState.CLOSED:
            return False

        ctx.state = ContextState.CLOSED
        ctx.cancel_event.set()

        if ctx.tts_task is not None and not ctx.tts_task.done():
            ctx.tts_task.cancel()

        logger.info(
            "context_closed",
            context_id=context_id,
            active_count=self.active_count,
        )
        return True

    def flush_context(self, context_id: str) -> str | None:
        """Flush remaining text from a context's buffer.

        Transitions the context to FLUSHING state. Returns the flushed
        text, or None if the buffer was empty or context not found.
        """
        ctx = self.get_active_context(context_id)
        if ctx is None:
            return None

        ctx.state = ContextState.FLUSHING
        ctx.touch()
        text = ctx.text_buffer.flush()
        logger.debug(
            "context_flushed",
            context_id=context_id,
            text_length=len(text) if text else 0,
        )
        return text

    def close_all(self) -> list[str]:
        """Close all active contexts. Returns list of closed context IDs."""
        closed_ids: list[str] = []
        for context_id in list(self._contexts.keys()):
            if self.close_context(context_id):
                closed_ids.append(context_id)
        return closed_ids

    def get_inactive_context_ids(self) -> list[str]:
        """Return IDs of active contexts that have exceeded the inactivity timeout."""
        now = time.monotonic()
        inactive: list[str] = []
        for ctx in self._contexts.values():
            if ctx.is_active and (now - ctx.last_activity) > self._inactivity_timeout_s:
                inactive.append(ctx.context_id)
        return inactive

    def touch(self, context_id: str) -> None:
        """Update the activity timestamp for a context."""
        ctx = self._contexts.get(context_id)
        if ctx is not None and ctx.is_active:
            ctx.touch()
