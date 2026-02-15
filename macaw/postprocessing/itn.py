"""Inverse Text Normalization (ITN) stage para o pipeline de pos-processamento.

Converte texto numerico por extenso para formato numerico:
"dois mil e vinte e cinco" -> "2025", "dez por cento" -> "10%".

Usa nemo_text_processing como dependencia opcional. Se nao estiver instalado,
o stage opera em modo fail-open: retorna o texto original sem modificacao.
"""

from __future__ import annotations

from typing import Any

from macaw.logging import get_logger
from macaw.postprocessing.stages import TextStage

logger = get_logger("postprocessing.itn")


class ITNStage(TextStage):
    """Stage de Inverse Text Normalization usando NeMo.

    Opera em modo fail-open: se nemo_text_processing nao estiver instalado,
    falhar na inicializacao ou falhar no processamento, retorna o texto
    original inalterado. A transcricao funciona -- apenas sem formatacao
    numerica.
    """

    @property
    def name(self) -> str:
        """Nome identificador do stage."""
        return "itn"

    def __init__(self, language: str = "pt") -> None:
        self._language = language
        self._normalizer: Any | None = None
        self._available: bool | None = None

    def _ensure_loaded(self) -> bool:
        """Carrega o normalizador NeMo lazily.

        Returns:
            True se o normalizador esta disponivel, False caso contrario.
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
                msg="nemo_text_processing nao instalado, ITN desabilitado",
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
        """Aplica ITN ao texto.

        Se o texto estiver vazio, o NeMo nao estiver disponivel ou ocorrer
        qualquer erro, retorna o texto original inalterado (fail-open).

        Args:
            text: Texto cru da engine.

        Returns:
            Texto com numeros formatados, ou texto original em caso de falha.
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
