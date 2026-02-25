"""Base interface for post-synthesis audio effects."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    import numpy as np


class AudioEffect(ABC):
    """Individual audio effect applied after TTS synthesis.

    Each effect receives a numpy float32 array and sample rate,
    processes the audio, and returns the transformed result.
    Unlike AudioStage, effects do not change sample rate.
    """

    @property
    @abstractmethod
    def name(self) -> str:
        """Identifier name for the effect (e.g. 'pitch_shift', 'reverb')."""
        ...

    @abstractmethod
    def process(self, audio: np.ndarray, sample_rate: int) -> np.ndarray:
        """Apply the effect to audio.

        Args:
            audio: Numpy float32 array with audio samples (mono).
            sample_rate: Audio sample rate in Hz.

        Returns:
            Processed audio as float32 array.
        """
        ...

    def reset(self) -> None:  # noqa: B027
        """Reset internal state for a new audio segment.

        Override in stateful effects (e.g. ReverbEffect) to clear
        delay line buffers. Default implementation is a no-op.
        """
