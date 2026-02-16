"""Base interface for Audio Preprocessing Pipeline stages."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    import numpy as np


class AudioStage(ABC):
    """Individual audio preprocessing pipeline stage.

    Each stage receives a numpy float32 array and sample rate,
    processes the audio, and returns the result with the new sample rate.
    """

    @property
    @abstractmethod
    def name(self) -> str:
        """Identifier name for the stage (e.g. 'resample', 'dc_remove')."""
        ...

    @abstractmethod
    def process(self, audio: np.ndarray, sample_rate: int) -> tuple[np.ndarray, int]:
        """Process audio frame.

        Args:
            audio: Numpy float32 array with audio samples (mono).
            sample_rate: Current audio sample rate in Hz.

        Returns:
            Tuple (processed audio, new sample rate).
        """
        ...

    def reset(self) -> None:  # noqa: B027
        """Reset internal state for a new streaming session.

        Override in stateful stages (e.g. DCRemoveStage) to clear
        filter state. Default implementation is a no-op.
        """
