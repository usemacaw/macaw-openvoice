"""Audio Preprocessing Pipeline.

Orquestra stages de preprocessamento de audio em sequencia.
Cada stage e toggleavel via PreprocessingConfig.
"""

from __future__ import annotations

import copy
from typing import TYPE_CHECKING

from macaw.logging import get_logger
from macaw.preprocessing.audio_io import decode_audio, encode_pcm16

if TYPE_CHECKING:
    from macaw.config.preprocessing import PreprocessingConfig
    from macaw.preprocessing.stages import AudioStage

logger = get_logger("preprocessing.pipeline")


class AudioPreprocessingPipeline:
    """Pipeline de preprocessamento de audio.

    Recebe bytes de audio em qualquer formato, aplica stages de
    processamento em sequencia e retorna PCM 16-bit WAV.

    Args:
        config: Configuracao do pipeline (stages habilitados, parametros).
        stages: Lista de stages a executar. Se None, usa lista vazia
                (stages concretos serao adicionados em E1-T2..T4).
    """

    def __init__(
        self,
        config: PreprocessingConfig,
        stages: list[AudioStage] | None = None,
    ) -> None:
        self._config = config
        self._stages = stages if stages is not None else []

    @property
    def config(self) -> PreprocessingConfig:
        """Configuracao do pipeline."""
        return self._config

    @property
    def stages(self) -> list[AudioStage]:
        """Lista de stages do pipeline."""
        return list(self._stages)

    def create_stages(self) -> list[AudioStage]:
        """Create fresh, independent stage instances for a new session.

        Returns deep copies of the template stages so each session gets
        its own filter state (e.g. DCRemoveStage zi). Resets each copy
        to ensure clean initial conditions.
        """
        fresh = [copy.deepcopy(stage) for stage in self._stages]
        for stage in fresh:
            stage.reset()
        return fresh

    def process(self, audio_bytes: bytes) -> bytes:
        """Processa audio atraves de todos os stages do pipeline.

        Args:
            audio_bytes: Bytes do audio de entrada (qualquer formato suportado).

        Returns:
            Bytes do audio processado em formato WAV PCM 16-bit, mono.

        Raises:
            AudioFormatError: Se o audio de entrada nao pode ser decodificado.
        """
        audio, sample_rate = decode_audio(audio_bytes)

        for stage in self._stages:
            logger.debug("stage_start", stage=stage.name, sample_rate=sample_rate)
            audio, sample_rate = stage.process(audio, sample_rate)
            logger.debug(
                "stage_complete",
                stage=stage.name,
                sample_rate=sample_rate,
                samples=len(audio),
            )

        return encode_pcm16(audio, sample_rate)
