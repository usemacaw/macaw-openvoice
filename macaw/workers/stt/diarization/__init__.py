"""Speaker diarization backends for Macaw OpenVoice.

Diarization runs as a POST-PROCESSING step after STT transcription (batch only).
It assigns speaker IDs to transcribed segments by analyzing audio patterns.
"""

from macaw.logging import get_logger
from macaw.workers.stt.diarization.interface import DiarizationBackend

logger = get_logger("worker.stt.diarization")


def create_diarizer() -> DiarizationBackend | None:
    """Create a diarization backend if pyannote.audio is available.

    Returns None if pyannote.audio is not installed or incompatible with the
    current environment (e.g. torchaudio version mismatch). The STT worker
    continues operating without diarization — the specific error is logged
    at WARNING level so operators can diagnose and fix the dependency issue.
    """
    try:
        from macaw.workers.stt.diarization.pyannote_backend import PyAnnoteDiarizer

        return PyAnnoteDiarizer()
    except ImportError:
        return None
    except Exception as exc:
        logger.warning(
            "diarizer_init_failed",
            error=str(exc),
            error_type=type(exc).__name__,
            hint="pyannote.audio is installed but failed to initialize — "
            "check version compatibility with torchaudio",
        )
        return None


__all__ = ["DiarizationBackend", "create_diarizer"]
