"""Audio Preprocessing Pipeline.

Normalizes audio from any source before VAD and the engine.
Pipeline: Ingestion -> [Resample] -> [DC Remove] -> [Gain Normalize] -> Output PCM 16-bit 16kHz mono.
"""

from __future__ import annotations

from macaw.preprocessing.pipeline import AudioPreprocessingPipeline
from macaw.preprocessing.stages import AudioStage

__all__ = ["AudioPreprocessingPipeline", "AudioStage"]
