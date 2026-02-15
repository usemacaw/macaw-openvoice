"""Gain Normalize stage do Audio Preprocessing Pipeline.

Normaliza amplitude do audio para um nivel de pico alvo em dBFS.
Garante input consistente para VAD e engines de inferencia.
"""

from __future__ import annotations

import numpy as np

from macaw.preprocessing.stages import AudioStage

# Limiar abaixo do qual o audio e considerado silencio.
# Evita divisao por zero e amplificacao de ruido de fundo.
_SILENCE_THRESHOLD = 1e-10


class GainNormalizeStage(AudioStage):
    """Normaliza amplitude do audio para nivel de pico alvo.

    Calcula o pico do sinal e aplica ganho para atingir o nivel
    target_dbfs. Inclui protecao contra clipping.

    Args:
        target_dbfs: Nivel de pico alvo em dBFS. Default: -3.0.
    """

    def __init__(self, target_dbfs: float = -3.0) -> None:
        self._target_dbfs = target_dbfs
        self._target_linear = 10 ** (target_dbfs / 20)

    @property
    def name(self) -> str:
        """Nome identificador do stage."""
        return "gain_normalize"

    def process(self, audio: np.ndarray, sample_rate: int) -> tuple[np.ndarray, int]:
        """Normaliza amplitude do audio para nivel de pico alvo.

        Args:
            audio: Array numpy float32 com amostras de audio (mono).
            sample_rate: Sample rate atual do audio em Hz.

        Returns:
            Tupla (audio normalizado float32, sample rate inalterado).
        """
        if len(audio) == 0:
            return audio, sample_rate

        peak = np.max(np.abs(audio))

        if peak < _SILENCE_THRESHOLD:
            return audio, sample_rate

        gain = self._target_linear / peak
        normalized = audio * gain
        normalized = np.clip(normalized, -1.0, 1.0)

        return normalized.astype(np.float32), sample_rate
