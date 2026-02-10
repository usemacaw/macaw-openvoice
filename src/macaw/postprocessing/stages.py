"""Interface base para stages do pipeline de pos-processamento de texto."""

from __future__ import annotations

from abc import ABC, abstractmethod


class TextStage(ABC):
    """Stage do pipeline de pos-processamento de texto.

    Cada stage recebe texto cru (ou parcialmente processado) e retorna
    texto transformado. Stages sao compostos em sequencia pelo
    PostProcessingPipeline.
    """

    @property
    @abstractmethod
    def name(self) -> str:
        """Nome identificador do stage (ex: 'itn', 'entity_formatting')."""
        ...

    @abstractmethod
    def process(self, text: str) -> str:
        """Processa texto e retorna texto transformado.

        Args:
            text: Texto de entrada (pode ser vazio).

        Returns:
            Texto processado.
        """
        ...
