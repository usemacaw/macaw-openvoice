"""Forced alignment module for TTS audio.

Provides a factory that returns an ``Aligner`` instance when torchaudio
is available, or ``None`` for graceful degradation.
"""

from __future__ import annotations

from macaw.alignment.interface import Aligner

__all__ = ["Aligner", "create_aligner"]


def create_aligner(device: str = "cpu") -> Aligner | None:
    """Create a forced alignment backend if torchaudio is available.

    Returns ``None`` when torchaudio is not installed, allowing the
    caller to skip alignment without crashing.

    Args:
        device: Torch device for the CTC model (``"cpu"`` or ``"cuda"``).

    Returns:
        A :class:`CTCAligner` instance or ``None``.
    """
    try:
        import torchaudio  # type: ignore[import-untyped]

        # Verify forced_align is available (torchaudio >= 2.1).
        if not hasattr(torchaudio.functional, "forced_align"):
            from macaw.logging import get_logger

            get_logger("alignment").warning(
                "torchaudio_too_old",
                required=">=2.1",
                installed=torchaudio.__version__,
            )
            return None

    except ImportError:
        return None

    from macaw.alignment.ctc_aligner import CTCAligner

    return CTCAligner(device=device)
