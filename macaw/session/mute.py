"""MuteController — coordinates mute-on-speak for Full-Duplex STT+TTS.

When TTS is active (bot speaking), STT should be muted to avoid
transcribing the bot's own audio. The MuteController manages a reference
counter and coordinates with StreamingSession.

Reference counting allows multiple concurrent TTS contexts to independently
mute/unmute without interfering. STT remains muted as long as at least one
foreground TTS context is active (mute_depth > 0).

Typical lifecycle (single context — backward-compatible):
    1. TTS worker starts producing audio
    2. Handler calls mute_controller.mute() -> depth becomes 1
    3. TTS ends (or is cancelled, or worker crashes)
    4. Handler calls mute_controller.unmute() -> depth becomes 0

Multi-context lifecycle:
    1. Context A starts TTS -> mute() -> depth=1
    2. Context B starts TTS -> mute() -> depth=2
    3. Context A finishes  -> unmute() -> depth=1, still muted
    4. Context B finishes  -> unmute() -> depth=0, unmuted

Guarantee: unmute MUST happen in try/finally to prevent permanent mute
in case of errors in the TTS flow.
"""

from __future__ import annotations

from macaw.logging import get_logger

logger = get_logger("session.mute")


class MuteController:
    """Controls mute-on-speak for a Full-Duplex session.

    Uses reference counting: each mute() increments the counter, each
    unmute() decrements it (floored at 0). STT is muted when counter > 0.

    Thread-safety is not needed: everything runs on the same asyncio event loop.

    Args:
        session_id: Session ID for logging.
    """

    def __init__(self, session_id: str = "") -> None:
        self._mute_count: int = 0
        self._session_id = session_id

    @property
    def is_muted(self) -> bool:
        """True if STT is muted (at least one TTS context active)."""
        return self._mute_count > 0

    @property
    def mute_depth(self) -> int:
        """Current mute reference count (for debugging)."""
        return self._mute_count

    def mute(self) -> None:
        """Increment mute reference count. STT is muted when count > 0."""
        self._mute_count += 1
        logger.debug(
            "stt_muted",
            session_id=self._session_id,
            mute_depth=self._mute_count,
        )

    def unmute(self) -> None:
        """Decrement mute reference count. Floored at 0 to guard underflow."""
        self._mute_count = max(0, self._mute_count - 1)
        logger.debug(
            "stt_unmuted",
            session_id=self._session_id,
            mute_depth=self._mute_count,
        )
