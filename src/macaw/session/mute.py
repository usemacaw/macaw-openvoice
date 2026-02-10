"""MuteController — coordinates mute-on-speak for Full-Duplex STT+TTS.

When TTS is active (bot speaking), STT should be muted to avoid
transcribing the bot's own audio. The MuteController manages this flag
and coordinates with StreamingSession.

Operations are idempotent: mute() when already muted is a no-op, and vice-versa.

Typical lifecycle:
    1. TTS worker starts producing audio
    2. Handler calls mute_controller.mute() -> StreamingSession mutes
    3. TTS ends (or is cancelled, or worker crashes)
    4. Handler calls mute_controller.unmute() -> StreamingSession resumes

Guarantee: unmute MUST happen in try/finally to prevent permanent mute
in case of errors in the TTS flow.
"""

from __future__ import annotations

from macaw.logging import get_logger

logger = get_logger("session.mute")


class MuteController:
    """Controls mute-on-speak for a Full-Duplex session.

    Thread-safety is not needed: everything runs on the same asyncio event loop.
    Idempotent: duplicate mute/unmute calls are no-ops.

    Args:
        session_id: Session ID for logging.
    """

    def __init__(self, session_id: str = "") -> None:
        self._muted = False
        self._session_id = session_id

    @property
    def is_muted(self) -> bool:
        """True if STT is muted (TTS active)."""
        return self._muted

    def mute(self) -> None:
        """Mute STT. Idempotent: no-op if already muted."""
        if self._muted:
            return
        self._muted = True
        logger.debug(
            "stt_muted",
            session_id=self._session_id,
        )

    def unmute(self) -> None:
        """Unmute STT. Idempotent: no-op if already unmuted."""
        if not self._muted:
            return
        self._muted = False
        logger.debug(
            "stt_unmuted",
            session_id=self._session_id,
        )
