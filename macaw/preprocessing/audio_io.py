"""Funcoes de decodificacao e codificacao de audio.

Converte entre bytes (formatos de arquivo) e arrays numpy float32.
"""

from __future__ import annotations

import io
import wave

import numpy as np
import soundfile as sf

from macaw.exceptions import AudioFormatError
from macaw.logging import get_logger

# Maximum absolute value for PCM 16-bit signed integers.
PCM_INT16_MAX = 32768.0

logger = get_logger("preprocessing.audio_io")


def decode_audio(audio_bytes: bytes) -> tuple[np.ndarray, int]:
    """Decodifica bytes de audio para array numpy float32 mono.

    Suporta WAV, FLAC, OGG e outros formatos via libsndfile.
    Converte automaticamente para mono se o audio for multi-canal.

    Args:
        audio_bytes: Bytes do arquivo de audio.

    Returns:
        Tupla (array float32 mono, sample rate em Hz).

    Raises:
        AudioFormatError: Se o formato nao e suportado ou os bytes sao invalidos.
    """
    if not audio_bytes:
        raise AudioFormatError("Audio vazio (0 bytes)")

    try:
        data, sample_rate = sf.read(io.BytesIO(audio_bytes), dtype="float32")
    except Exception:
        # Fallback para wave stdlib (WAV PCM puro sem headers complexos)
        try:
            data, sample_rate = _decode_wav_stdlib(audio_bytes)
        except Exception as wav_err:
            raise AudioFormatError(f"Nao foi possivel decodificar o audio: {wav_err}") from wav_err

    # Converter para mono se multi-canal
    if data.ndim > 1:
        data = np.mean(data, axis=1)

    # Garantir float32
    data = data.astype(np.float32)

    logger.debug(
        "audio_decoded",
        samples=len(data),
        sample_rate=sample_rate,
        duration_s=round(len(data) / sample_rate, 3),
    )

    return data, int(sample_rate)


def _decode_wav_stdlib(audio_bytes: bytes) -> tuple[np.ndarray, int]:
    """Decodifica WAV PCM usando wave stdlib como fallback.

    Args:
        audio_bytes: Bytes do arquivo WAV.

    Returns:
        Tupla (array float32, sample rate).

    Raises:
        AudioFormatError: Se o WAV e invalido ou usa formato nao-PCM.
    """
    try:
        with wave.open(io.BytesIO(audio_bytes), "rb") as wf:
            n_channels = wf.getnchannels()
            sampwidth = wf.getsampwidth()
            sample_rate = wf.getframerate()
            n_frames = wf.getnframes()

            if n_frames == 0:
                raise AudioFormatError("Arquivo WAV sem frames de audio")

            raw_data = wf.readframes(n_frames)
    except wave.Error as err:
        raise AudioFormatError(f"Arquivo WAV invalido: {err}") from err

    if sampwidth == 2:
        # PCM 16-bit — zero-copy via np.frombuffer + single float32 cast
        data = np.frombuffer(raw_data, dtype=np.int16).astype(np.float32) / PCM_INT16_MAX
    elif sampwidth == 1:
        # PCM 8-bit (unsigned) — zero-copy via np.frombuffer
        data = np.frombuffer(raw_data, dtype=np.uint8).astype(np.float32) / 128.0 - 1.0
    else:
        raise AudioFormatError(f"Sample width {sampwidth} bytes nao suportado (esperado 1 ou 2)")

    # Converter multi-canal para mono
    if n_channels > 1:
        data = data.reshape(-1, n_channels)
        data = np.mean(data, axis=1)

    return data, sample_rate


def encode_pcm16(audio: np.ndarray, sample_rate: int) -> bytes:
    """Codifica array numpy float32 para bytes WAV PCM 16-bit.

    Args:
        audio: Array numpy float32 com amostras de audio (mono).
        sample_rate: Sample rate em Hz.

    Returns:
        Bytes do arquivo WAV completo (com header).
    """
    # Clamp para evitar overflow
    audio_clamped = np.clip(audio, -1.0, 1.0)

    # Converter float32 para int16
    pcm_data = (audio_clamped * (PCM_INT16_MAX - 1)).astype(np.int16)

    # Escrever WAV
    buffer = io.BytesIO()
    with wave.open(buffer, "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(sample_rate)
        wf.writeframes(pcm_data.tobytes())

    return buffer.getvalue()
