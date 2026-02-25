"""G.711 mu-law and A-law codec encoders (ITU-T G.711).

Pure numpy implementation — no external dependencies required.
G.711 outputs 8-bit audio at 8000 Hz, so internal resampling from the
TTS native sample rate (typically 24000 Hz) is performed automatically.

References:
    ITU-T Recommendation G.711 (11/88)
"""

from __future__ import annotations

import numpy as np

from macaw._audio_constants import PCM_INT16_SCALE
from macaw._dsp import resample_output
from macaw.codec.interface import CodecEncoder
from macaw.logging import get_logger

logger = get_logger("codec.g711")

G711_SAMPLE_RATE: int = 8000

# mu-law compression parameter (ITU-T G.711)
_MU: int = 255
_LOG_MU_PLUS_1: float = float(np.log1p(_MU))

# A-law compression parameter (ITU-T G.711)
_A: float = 87.6
_LOG_A: float = float(np.log(_A))


def _pcm16_bytes_to_float32(pcm_data: bytes) -> np.ndarray:
    """Convert 16-bit signed PCM bytes to float32 in [-1.0, 1.0]."""
    samples = np.frombuffer(pcm_data, dtype=np.int16)
    return samples.astype(np.float32) / PCM_INT16_SCALE


def _mu_law_compress(audio: np.ndarray) -> np.ndarray:
    """Apply mu-law compression to float32 audio in [-1.0, 1.0].

    Formula: sign(x) * ln(1 + mu * |x|) / ln(1 + mu)
    Output is quantized to uint8 [0, 255].
    """
    compressed = np.sign(audio) * np.log1p(_MU * np.abs(audio)) / _LOG_MU_PLUS_1
    # Map from [-1, 1] to [0, 255]
    result: np.ndarray = np.clip(((compressed + 1.0) / 2.0 * 255.0), 0, 255).astype(np.uint8)
    return result


def _a_law_compress(audio: np.ndarray) -> np.ndarray:
    """Apply A-law compression to float32 audio in [-1.0, 1.0].

    Formula:
        |x| < 1/A: A*|x| / (1 + ln(A))
        |x| >= 1/A: (1 + ln(A*|x|)) / (1 + ln(A))
    Output is quantized to uint8 [0, 255].
    """
    abs_audio = np.abs(audio)
    threshold = 1.0 / _A

    compressed = np.where(
        abs_audio < threshold,
        _A * abs_audio / (1.0 + _LOG_A),
        (1.0 + np.log(np.maximum(abs_audio * _A, 1e-10))) / (1.0 + _LOG_A),
    )
    compressed = np.sign(audio) * compressed
    # Map from [-1, 1] to [0, 255]
    result: np.ndarray = np.clip(((compressed + 1.0) / 2.0 * 255.0), 0, 255).astype(np.uint8)
    return result


class MuLawEncoder(CodecEncoder):
    """ITU-T G.711 mu-law codec encoder.

    Accepts 16-bit PCM at any sample rate, resamples internally to 8kHz,
    and applies mu-law compression to produce 8-bit output.

    Args:
        sample_rate: Input PCM sample rate (e.g., 24000 for TTS).
    """

    def __init__(self, sample_rate: int = 24000) -> None:
        self._sample_rate = sample_rate
        logger.info("mulaw_encoder_ready", sample_rate=sample_rate)

    @property
    def codec_name(self) -> str:
        return "mulaw"

    def encode(self, pcm_data: bytes) -> bytes:
        """Encode 16-bit PCM to mu-law 8-bit at 8kHz."""
        if not pcm_data:
            return b""
        audio = _pcm16_bytes_to_float32(pcm_data)
        if self._sample_rate != G711_SAMPLE_RATE:
            audio = resample_output(audio, self._sample_rate, G711_SAMPLE_RATE)
        compressed = _mu_law_compress(audio)
        return compressed.tobytes()

    def flush(self) -> bytes:
        """G.711 has no internal buffer — flush is always empty."""
        return b""


class ALawEncoder(CodecEncoder):
    """ITU-T G.711 A-law codec encoder.

    Accepts 16-bit PCM at any sample rate, resamples internally to 8kHz,
    and applies A-law compression to produce 8-bit output.

    Args:
        sample_rate: Input PCM sample rate (e.g., 24000 for TTS).
    """

    def __init__(self, sample_rate: int = 24000) -> None:
        self._sample_rate = sample_rate
        logger.info("alaw_encoder_ready", sample_rate=sample_rate)

    @property
    def codec_name(self) -> str:
        return "alaw"

    def encode(self, pcm_data: bytes) -> bytes:
        """Encode 16-bit PCM to A-law 8-bit at 8kHz."""
        if not pcm_data:
            return b""
        audio = _pcm16_bytes_to_float32(pcm_data)
        if self._sample_rate != G711_SAMPLE_RATE:
            audio = resample_output(audio, self._sample_rate, G711_SAMPLE_RATE)
        compressed = _a_law_compress(audio)
        return compressed.tobytes()

    def flush(self) -> bytes:
        """G.711 has no internal buffer — flush is always empty."""
        return b""
