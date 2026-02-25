"""Forced alignment module for TTS audio.

Provides a factory that returns an ``Aligner`` instance when torchaudio
is available, or ``None`` for graceful degradation.
"""

from __future__ import annotations

from macaw.alignment.interface import Aligner
from macaw.logging import get_logger

__all__ = ["Aligner", "create_aligner"]

logger = get_logger("alignment")


def create_aligner(device: str = "cpu") -> Aligner | None:
    """Create a forced alignment backend if torchaudio is available.

    Returns ``None`` when torchaudio is not installed or incompatible with
    the current environment (e.g. torchaudio version mismatch). The caller
    continues without alignment — the specific error is logged at WARNING
    level so operators can diagnose and fix the dependency issue.

    Args:
        device: Torch device for the CTC model (``"cpu"`` or ``"cuda"``).

    Returns:
        A :class:`CTCAligner` instance or ``None``.
    """
    try:
        import torchaudio  # type: ignore[import-untyped]

        # Verify forced_align is available (torchaudio >= 2.1).
        if not hasattr(torchaudio.functional, "forced_align"):
            logger.warning(
                "torchaudio_too_old",
                required=">=2.1",
                installed=torchaudio.__version__,
            )
            return None

    except ImportError:
        return None
    except Exception as exc:
        logger.warning(
            "aligner_init_failed",
            error=str(exc),
            error_type=type(exc).__name__,
            hint="torchaudio is installed but failed to initialize — check version compatibility",
        )
        return None

    from macaw.alignment.ctc_aligner import CTCAligner

    return CTCAligner(device=device)
