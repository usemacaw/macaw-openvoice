"""Effect chain compositor — applies a sequence of AudioEffects."""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

from macaw._audio_constants import PCM_INT16_MAX, PCM_INT16_MIN, PCM_INT16_SCALE

if TYPE_CHECKING:
    from macaw.audio_effects.interface import AudioEffect


class AudioEffectChain:
    """Applies a sequence of AudioEffect instances to audio data.

    Provides both float32 ndarray processing and PCM16 bytes
    convenience methods for integration with the TTS pipeline.
    """

    def __init__(self, effects: list[AudioEffect]) -> None:
        self._effects = effects

    def process(self, audio: np.ndarray, sample_rate: int) -> np.ndarray:
        """Apply all effects sequentially to float32 audio."""
        result = audio
        for effect in self._effects:
            result = effect.process(result, sample_rate)
        return result

    def process_bytes(self, pcm_data: bytes, sample_rate: int) -> bytes:
        """Convert PCM16 bytes -> float32, apply chain, convert back to PCM16 bytes."""
        audio = pcm16_bytes_to_float32(pcm_data)
        processed = self.process(audio, sample_rate)
        return _float32_to_pcm16_bytes(processed)

    def reset(self) -> None:
        """Reset all effects in the chain."""
        for effect in self._effects:
            effect.reset()


def pcm16_bytes_to_float32(pcm_data: bytes) -> np.ndarray:
    """Convert 16-bit PCM bytes to float32 array normalized to [-1, 1]."""
    return np.frombuffer(pcm_data, dtype=np.int16).astype(np.float32) / PCM_INT16_SCALE


def _float32_to_pcm16_bytes(audio: np.ndarray) -> bytes:
    """Convert float32 [-1, 1] array to 16-bit PCM bytes."""
    int16_data = (audio * PCM_INT16_SCALE).clip(PCM_INT16_MIN, PCM_INT16_MAX).astype(np.int16)
    return int16_data.tobytes()
