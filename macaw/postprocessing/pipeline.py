"""Pipeline de pos-processamento de texto.

Orquestra stages (ITN, entity formatting, hot word correction) em sequencia.
Aplicado apenas em transcript.final, nunca em transcript.partial.
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
    """Pipeline que executa stages de pos-processamento em sequencia.

    Recebe texto cru da engine e produz texto formatado (ex: "dois mil" -> "2000").
    Cada stage e independente e pode ser habilitado/desabilitado via config.
    """

    def __init__(
        self,
        config: PostProcessingConfig,
        stages: list[TextStage] | None = None,
    ) -> None:
        self._config = config
        self._stages: list[TextStage] = stages if stages is not None else []

    @property
    def stages(self) -> list[TextStage]:
        """Stages configurados no pipeline."""
        return list(self._stages)

    def process(self, text: str) -> str:
        """Processa texto atraves de todos os stages em sequencia.

        Args:
            text: Texto cru da engine.

        Returns:
            Texto processado por todos os stages.
        """
        for stage in self._stages:
            logger.debug("stage_start", stage=stage.name, text_length=len(text))
            text = stage.process(text)
            logger.debug("stage_complete", stage=stage.name, text_length=len(text))
        return text

    def process_result(self, result: BatchResult) -> BatchResult:
        """Processa um BatchResult completo (texto principal + segmentos).

        Cria novas instancias de BatchResult e SegmentDetail com textos
        processados, preservando todos os outros campos. BatchResult e
        SegmentDetail sao frozen dataclasses, entao novas instancias
        sao criadas via dataclasses.replace().

        Args:
            result: BatchResult original com texto cru.

        Returns:
            Novo BatchResult com textos processados.
        """
        processed_text = self.process(result.text)

        processed_segments = tuple(
            replace(segment, text=self.process(segment.text)) for segment in result.segments
        )

        return replace(
            result,
            text=processed_text,
            segments=processed_segments,
        )
