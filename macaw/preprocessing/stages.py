"""Interface base para stages do Audio Preprocessing Pipeline."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    import numpy as np


class AudioStage(ABC):
    """Stage individual do pipeline de preprocessamento de audio.

    Cada stage recebe um array numpy float32 e sample rate,
    processa o audio e retorna o resultado com o novo sample rate.
    """

    @property
    @abstractmethod
    def name(self) -> str:
        """Nome identificador do stage (ex: 'resample', 'dc_remove')."""
        ...

    @abstractmethod
    def process(self, audio: np.ndarray, sample_rate: int) -> tuple[np.ndarray, int]:
        """Processa frame de audio.

        Args:
            audio: Array numpy float32 com amostras de audio (mono).
            sample_rate: Sample rate atual do audio em Hz.

        Returns:
            Tupla (audio processado, novo sample rate).
        """
        ...

    def reset(self) -> None:  # noqa: B027
        """Reset internal state for a new streaming session.

        Override in stateful stages (e.g. DCRemoveStage) to clear
        filter state. Default implementation is a no-op.
        """
