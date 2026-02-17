"""Centralized audio format constants for the Macaw runtime.

Single source of truth for PCM format parameters, standard sample rates,
and default values shared across preprocessing, VAD, session, scheduler,
workers, and CLI.
"""

from __future__ import annotations

# --- Preprocessing defaults (single source of truth for settings + config) ---
DEFAULT_DC_CUTOFF_HZ: int = 20
DEFAULT_TARGET_DBFS: float = -3.0
DEFAULT_ITN_LANGUAGE: str = "pt"

# --- PCM 16-bit format ---
# Signed 16-bit integer range: [-32768, 32767]
PCM_INT16_MAX: int = 32767
PCM_INT16_MIN: int = -32768
# Scale factor for float32 <-> int16 conversion.
# Using 32768.0 (not 32767.0) is the industry convention for symmetric
# normalization: int16 / 32768.0 maps to [-1.0, ~0.99997].
PCM_INT16_SCALE: float = 32768.0

# Signed 32-bit integer scale (used by some WAV decoders).
PCM_INT32_SCALE: float = 2147483648.0

# Bytes per sample for 16-bit PCM.
BYTES_PER_SAMPLE_INT16: int = 2

# PCM 8-bit (unsigned): center value and scale for [-1.0, ~1.0] normalization.
# uint8 range is [0, 255]; midpoint 128 maps to 0.0.
PCM_UINT8_SCALE: float = 128.0

# --- Standard sample rates ---
# STT pipeline: Whisper, Silero VAD, and all preprocessing stages expect 16kHz.
STT_SAMPLE_RATE: int = 16000

# TTS pipeline: Kokoro, Qwen3-TTS, and other modern TTS engines output 24kHz.
TTS_DEFAULT_SAMPLE_RATE: int = 24000

# --- VAD constants ---
# Silero VAD maximum input chunk size (samples at 16kHz).
# Silero expects exactly 512 samples (32ms). Longer frames are split.
SILERO_VAD_CHUNK_SIZE: int = 512

# Spectral flatness threshold for energy pre-filter.
# Values above this indicate flat spectrum (white noise/silence).
# Tonal speech typically has flatness ~0.1-0.5.
DEFAULT_SPECTRAL_FLATNESS_THRESHOLD: float = 0.8
