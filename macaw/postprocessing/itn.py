"""Inverse Text Normalization (ITN) stage for the post-processing pipeline.

Converts written-out numeric text to numeric format:
"two thousand twenty five" -> "2025", "ten percent" -> "10%".

Uses nemo_text_processing as an optional dependency. If not installed,
the stage operates in fail-open mode: returns the original text unmodified.

Supports multiple languages via a per-language normalizer cache. Each
language is lazily loaded on first use and cached for subsequent calls.
"""

from __future__ import annotations

from typing import Any

from macaw.logging import get_logger
from macaw.postprocessing.stages import TextStage

logger = get_logger("postprocessing.itn")


class ITNStage(TextStage):
    """Inverse Text Normalization stage using NeMo.

    Maintains a per-language cache of ``InverseNormalize`` instances.
    When ``process()`` is called with a ``language`` parameter, that
    language's normalizer is used (lazy-loaded on first call). When
    ``language`` is None, the ``default_language`` is used.

    Operates in fail-open mode: if nemo_text_processing is not installed,
    fails to initialize for a language, or fails during processing,
    returns the original text unchanged.
    """

    @property
    def name(self) -> str:
        """Identifier name for the stage."""
        return "itn"

    def __init__(self, default_language: str = "pt") -> None:
        self._default_language = default_language
        self._normalizers: dict[str, Any] = {}
        self._unavailable_languages: set[str] = set()

    def _get_normalizer(self, language: str) -> Any | None:
        """Get or create normalizer for language (lazy, cached).

        Returns:
            The cached normalizer, or None if the language is unavailable.
        """
        if language in self._unavailable_languages:
            return None

        if language in self._normalizers:
            return self._normalizers[language]

        try:
            from nemo_text_processing.inverse_text_normalization import (
                InverseNormalize,
            )

            normalizer = InverseNormalize(lang=language)
            self._normalizers[language] = normalizer
            return normalizer
        except ImportError:
            logger.warning(
                "nemo_text_processing_not_available",
                language=language,
                msg="nemo_text_processing not installed, ITN disabled",
            )
            self._unavailable_languages.add(language)
            return None
        except Exception:
            logger.warning(
                "itn_init_failed",
                language=language,
                exc_info=True,
            )
            self._unavailable_languages.add(language)
            return None

    def process(self, text: str, *, language: str | None = None) -> str:
        """Apply ITN to text using the appropriate language normalizer.

        If the text is empty, NeMo is not available for the requested
        language, or any error occurs, returns the original text
        unchanged (fail-open).

        Args:
            text: Raw text from the engine.
            language: ISO 639-1 language code. Falls back to
                ``default_language`` when None.

        Returns:
            Text with formatted numbers, or original text on failure.
        """
        if not text.strip():
            return text

        effective_language = language or self._default_language
        normalizer = self._get_normalizer(effective_language)
        if normalizer is None:
            return text

        try:
            result: str = normalizer.inverse_normalize(text, verbose=False)
            return result
        except Exception:
            logger.warning(
                "itn_process_failed",
                text_length=len(text),
                language=effective_language,
                exc_info=True,
            )
            return text
