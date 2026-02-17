"""Text post-processing pipeline.

Orchestrates stages (ITN, entity formatting, hot word correction) in sequence.
Applied only to transcript.final, never to transcript.partial.
"""

from __future__ import annotations

from dataclasses import replace
from typing import TYPE_CHECKING

from macaw.logging import get_logger

if TYPE_CHECKING:
    from macaw._types import BatchResult
    from macaw.config.postprocessing import PostProcessingConfig
    from macaw.postprocessing.stages import TextStage

logger = get_logger("postprocessing.pipeline")


class PostProcessingPipeline:
    """Pipeline that executes post-processing stages in sequence.

    Receives raw text from the engine and produces formatted text (e.g. "two thousand" -> "2000").
    Each stage is independent and can be enabled/disabled via config.
    """

    def __init__(
        self,
        config: PostProcessingConfig,
        stages: list[TextStage] | None = None,
    ) -> None:
        self._stages: list[TextStage] = stages if stages is not None else []

    @property
    def stages(self) -> list[TextStage]:
        """Stages configured in the pipeline."""
        return list(self._stages)

    def process(self, text: str, *, language: str | None = None) -> str:
        """Process text through all stages in sequence.

        Args:
            text: Raw text from the engine.
            language: ISO 639-1 language code (optional). Forwarded to
                language-aware stages (e.g. ITN) for per-language
                normalization. Language-agnostic stages ignore it.

        Returns:
            Text processed by all stages.
        """
        for stage in self._stages:
            logger.debug("stage_start", stage=stage.name, text_length=len(text))
            text = stage.process(text, language=language)
            logger.debug("stage_complete", stage=stage.name, text_length=len(text))
        return text

    def process_result(self, result: BatchResult) -> BatchResult:
        """Process a complete BatchResult (main text + segments).

        Creates new BatchResult and SegmentDetail instances with processed
        texts, preserving all other fields. BatchResult and SegmentDetail
        are frozen dataclasses, so new instances are created via
        dataclasses.replace().

        Uses ``result.language`` (detected by the STT engine) as the
        language for post-processing stages. This ensures ITN uses the
        correct normalizer for the detected language.

        Args:
            result: Original BatchResult with raw text.

        Returns:
            New BatchResult with processed texts.
        """
        language = result.language or None
        processed_text = self.process(result.text, language=language)

        processed_segments = tuple(
            replace(segment, text=self.process(segment.text, language=language))
            for segment in result.segments
        )

        return replace(
            result,
            text=processed_text,
            segments=processed_segments,
        )
