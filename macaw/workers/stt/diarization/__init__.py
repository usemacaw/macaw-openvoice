"""Speaker diarization backends for Macaw OpenVoice.

Diarization runs as a POST-PROCESSING step after STT transcription (batch only).
It assigns speaker IDs to transcribed segments by analyzing audio patterns.
"""

from macaw.workers.stt.diarization.interface import DiarizationBackend


def create_diarizer() -> DiarizationBackend | None:
    """Create a diarization backend if pyannote.audio is available.

    Returns None if pyannote.audio is not installed (graceful degradation).
    """
    try:
        from macaw.workers.stt.diarization.pyannote_backend import PyAnnoteDiarizer

        return PyAnnoteDiarizer()
    except ImportError:
        return None


__all__ = ["DiarizationBackend", "create_diarizer"]
