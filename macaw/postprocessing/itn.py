"""Inverse Text Normalization (ITN) stage for the post-processing pipeline.

Converts written-out numeric text to numeric format:
"two thousand twenty five" -> "2025", "ten percent" -> "10%".

Uses nemo_text_processing as an optional dependency. If not installed,
the stage operates in fail-open mode: returns the original text unmodified.
"""

from __future__ import annotations

from typing import Any

from macaw.logging import get_logger
from macaw.postprocessing.stages import TextStage

logger = get_logger("postprocessing.itn")


class ITNStage(TextStage):
    """Inverse Text Normalization stage using NeMo.

    Operates in fail-open mode: if nemo_text_processing is not installed,
    fails to initialize, or fails during processing, returns the original
    text unchanged. Transcription works -- just without numeric formatting.
    """

    @property
    def name(self) -> str:
        """Identifier name for the stage."""
        return "itn"

    def __init__(self, language: str = "pt") -> None:
        self._language = language
        self._normalizer: Any | None = None
        self._available: bool | None = None

    def _ensure_loaded(self) -> bool:
        """Lazily load the NeMo normalizer.

        Returns:
            True if the normalizer is available, False otherwise.
        """
        if self._available is not None:
            return self._available

        try:
            from nemo_text_processing.inverse_text_normalization import (
                InverseNormalize,
            )

            self._normalizer = InverseNormalize(lang=self._language)
            self._available = True
        except ImportError:
            logger.warning(
                "nemo_text_processing_not_available",
                language=self._language,
                msg="nemo_text_processing not installed, ITN disabled",
            )
            self._available = False
        except Exception:
            logger.warning(
                "itn_init_failed",
                language=self._language,
                exc_info=True,
            )
            self._available = False

        return self._available

    def process(self, text: str) -> str:
        """Apply ITN to text.

        If the text is empty, NeMo is not available, or any error occurs,
        returns the original text unchanged (fail-open).

        Args:
            text: Raw text from the engine.

        Returns:
            Text with formatted numbers, or original text on failure.
        """
        if not text.strip():
            return text

        if not self._ensure_loaded():
            return text

        try:
            result: str = self._normalizer.inverse_normalize(text, verbose=False)  # type: ignore[union-attr]
            return result
        except Exception:
            logger.warning(
                "itn_process_failed",
                text_length=len(text),
                exc_info=True,
            )
            return text
