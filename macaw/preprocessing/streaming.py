"""Streaming Preprocessor — frame-by-frame adapter for audio preprocessing.

Receives raw PCM bytes (int16) from the WebSocket, converts to numpy float32,
applies Audio Preprocessing Pipeline stages in sequence, and returns
float32 16kHz mono.

Difference from AudioPreprocessingPipeline (batch):
- Batch: receives complete file (WAV/FLAC), decodes via soundfile
- Streaming: receives raw PCM 16-bit frames, converts directly
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

from macaw.logging import get_logger
from macaw.workers.audio_utils import pcm_bytes_to_float32

if TYPE_CHECKING:
    from macaw.preprocessing.stages import AudioStage

logger = get_logger("preprocessing.streaming")


class StreamingPreprocessor:
    """Preprocessing adapter for frame-by-frame streaming.

    Receives raw PCM bytes (int16) from the WebSocket, converts to numpy float32,
    applies M4 stages in sequence, and returns float32 16kHz mono.

    Args:
        stages: List of AudioStage to apply in sequence.
        input_sample_rate: Input audio sample rate (default 16000).
                           Can be changed via set_input_sample_rate().
    """

    def __init__(
        self,
        stages: list[AudioStage],
        input_sample_rate: int = 16000,
    ) -> None:
        self._stages = stages
        self._input_sample_rate = input_sample_rate

    @property
    def input_sample_rate(self) -> int:
        """Current input sample rate."""
        return self._input_sample_rate

    def set_input_sample_rate(self, sample_rate: int) -> None:
        """Update input sample rate (e.g. via session.configure).

        Args:
            sample_rate: New sample rate in Hz.
        """
        self._input_sample_rate = sample_rate

    def process_frame(self, raw_bytes: bytes) -> np.ndarray:
        """Process a raw audio frame.

        Converts PCM int16 bytes to normalized numpy float32 [-1.0, 1.0],
        applies all stages in sequence, and returns the result.

        Args:
            raw_bytes: PCM 16-bit little-endian bytes (mono).

        Returns:
            Numpy float32 16kHz mono array, processed by all stages.

        Raises:
            AudioFormatError: If bytes have odd length
                (PCM 16-bit = 2 bytes/sample).
        """
        if len(raw_bytes) == 0:
            return np.array([], dtype=np.float32)

        audio = pcm_bytes_to_float32(raw_bytes)

        sample_rate = self._input_sample_rate

        for stage in self._stages:
            audio, sample_rate = stage.process(audio, sample_rate)

        return audio
