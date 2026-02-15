"""ResampleStage — converte audio para sample rate alvo.

Usa scipy.signal.resample_poly para resampling de alta qualidade.
Pipeline contract guarantees mono input; no multi-channel handling here.
"""

from __future__ import annotations

from math import gcd

import numpy as np
from scipy.signal import resample_poly

from macaw.preprocessing.stages import AudioStage


class ResampleStage(AudioStage):
    """Stage de resampling do pipeline de preprocessamento.

    Converte audio de qualquer sample rate para o sample rate alvo (default 16kHz).
    Expects mono input (pipeline contract).

    Args:
        target_sample_rate: Sample rate alvo em Hz (default: 16000).
    """

    def __init__(self, target_sample_rate: int = 16000) -> None:
        self._target_sample_rate = target_sample_rate

    @property
    def name(self) -> str:
        """Nome identificador do stage."""
        return "resample"

    def process(self, audio: np.ndarray, sample_rate: int) -> tuple[np.ndarray, int]:
        """Converte audio para sample rate alvo.

        Se o audio ja esta no sample rate alvo, retorna sem modificacao.
        Se o audio esta vazio, retorna sem modificacao.

        Args:
            audio: Array numpy com amostras de audio.
            sample_rate: Sample rate atual do audio em Hz.

        Returns:
            Tupla (audio resampleado float32, sample rate alvo).
        """
        if audio.size == 0:
            return audio, sample_rate

        # Skip se ja esta no sample rate alvo
        if sample_rate == self._target_sample_rate:
            return audio, sample_rate

        # Calcular fatores up/down simplificados pelo GCD
        divisor = gcd(self._target_sample_rate, sample_rate)
        up = self._target_sample_rate // divisor
        down = sample_rate // divisor

        resampled = resample_poly(audio, up, down).astype(np.float32)

        return resampled, self._target_sample_rate
