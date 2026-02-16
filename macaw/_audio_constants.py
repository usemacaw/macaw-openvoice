"""Centralized audio format constants for the Macaw runtime.

Single source of truth for PCM format parameters and standard sample rates
used across preprocessing, VAD, session, scheduler, workers, and CLI.
"""

from __future__ import annotations

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

# --- Standard sample rates ---
# STT pipeline: Whisper, Silero VAD, and all preprocessing stages expect 16kHz.
STT_SAMPLE_RATE: int = 16000

# TTS pipeline: Kokoro, Qwen3-TTS, and other modern TTS engines output 24kHz.
TTS_DEFAULT_SAMPLE_RATE: int = 24000
