"""Audio decoding and encoding functions.

Converts between bytes (file formats) and numpy float32 arrays.
"""

from __future__ import annotations

import io
import wave

import numpy as np
import soundfile as sf

from macaw._audio_constants import PCM_INT16_SCALE, PCM_UINT8_SCALE
from macaw.exceptions import AudioFormatError
from macaw.logging import get_logger

logger = get_logger("preprocessing.audio_io")


def decode_audio(audio_bytes: bytes) -> tuple[np.ndarray, int]:
    """Decode audio bytes to numpy float32 mono array.

    Supports WAV, FLAC, OGG, and other formats via libsndfile.
    Automatically converts to mono if the audio is multi-channel.

    Args:
        audio_bytes: Audio file bytes.

    Returns:
        Tuple (float32 mono array, sample rate in Hz).

    Raises:
        AudioFormatError: If the format is unsupported or bytes are invalid.
    """
    if not audio_bytes:
        raise AudioFormatError("Empty audio (0 bytes)")

    try:
        data, sample_rate = sf.read(io.BytesIO(audio_bytes), dtype="float32")
    except Exception:
        # Fallback to wave stdlib (plain WAV PCM without complex headers)
        try:
            data, sample_rate = _decode_wav_stdlib(audio_bytes)
        except Exception as wav_err:
            raise AudioFormatError(f"Could not decode audio: {wav_err}") from wav_err

    # Convert to mono if multi-channel
    if data.ndim > 1:
        data = np.mean(data, axis=1)

    # Ensure float32
    data = data.astype(np.float32)

    logger.debug(
        "audio_decoded",
        samples=len(data),
        sample_rate=sample_rate,
        duration_s=round(len(data) / sample_rate, 3),
    )

    return data, int(sample_rate)


def _decode_wav_stdlib(audio_bytes: bytes) -> tuple[np.ndarray, int]:
    """Decode WAV PCM using wave stdlib as fallback.

    Args:
        audio_bytes: WAV file bytes.

    Returns:
        Tuple (float32 array, sample rate).

    Raises:
        AudioFormatError: If the WAV is invalid or uses non-PCM format.
    """
    try:
        with wave.open(io.BytesIO(audio_bytes), "rb") as wf:
            n_channels = wf.getnchannels()
            sampwidth = wf.getsampwidth()
            sample_rate = wf.getframerate()
            n_frames = wf.getnframes()

            if n_frames == 0:
                raise AudioFormatError("WAV file has no audio frames")

            raw_data = wf.readframes(n_frames)
    except wave.Error as err:
        raise AudioFormatError(f"Invalid WAV file: {err}") from err

    if sampwidth == 2:
        # PCM 16-bit -- zero-copy via np.frombuffer + single float32 cast
        data = np.frombuffer(raw_data, dtype=np.int16).astype(np.float32) / PCM_INT16_SCALE
    elif sampwidth == 1:
        # PCM 8-bit (unsigned) -- zero-copy via np.frombuffer
        data = np.frombuffer(raw_data, dtype=np.uint8).astype(np.float32) / PCM_UINT8_SCALE - 1.0
    else:
        raise AudioFormatError(f"Sample width {sampwidth} bytes not supported (expected 1 or 2)")

    # Convert multi-channel to mono
    if n_channels > 1:
        data = data.reshape(-1, n_channels)
        data = np.mean(data, axis=1)

    return data, sample_rate


def encode_pcm16(audio: np.ndarray, sample_rate: int) -> bytes:
    """Encode numpy float32 array to WAV PCM 16-bit bytes.

    Args:
        audio: Numpy float32 array with audio samples (mono).
        sample_rate: Sample rate in Hz.

    Returns:
        Complete WAV file bytes (with header).
    """
    # Clamp to avoid overflow
    audio_clamped = np.clip(audio, -1.0, 1.0)

    # Convert float32 to int16
    pcm_data = (audio_clamped * (PCM_INT16_SCALE - 1)).astype(np.int16)

    # Write WAV
    buffer = io.BytesIO()
    with wave.open(buffer, "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(sample_rate)
        wf.writeframes(pcm_data.tobytes())

    return buffer.getvalue()
