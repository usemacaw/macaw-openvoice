"""Pipeline de pos-processamento de texto (ITN, entity formatting, hot words)."""

from __future__ import annotations

from macaw.postprocessing.pipeline import PostProcessingPipeline
from macaw.postprocessing.stages import TextStage

__all__ = ["PostProcessingPipeline", "TextStage"]
