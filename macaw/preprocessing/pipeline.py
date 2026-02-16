"""Audio Preprocessing Pipeline.

Orchestrates audio preprocessing stages in sequence.
Each stage is toggleable via PreprocessingConfig.
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
    """Audio preprocessing pipeline.

    Receives audio bytes in any format, applies processing stages
    in sequence, and returns PCM 16-bit WAV.

    Args:
        config: Pipeline configuration (enabled stages, parameters).
        stages: List of stages to execute. If None, uses empty list
                (concrete stages will be added in E1-T2..T4).
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
        """Pipeline configuration."""
        return self._config

    @property
    def stages(self) -> list[AudioStage]:
        """List of pipeline stages."""
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
        """Process audio through all pipeline stages.

        Args:
            audio_bytes: Input audio bytes (any supported format).

        Returns:
            Processed audio bytes in WAV PCM 16-bit, mono format.

        Raises:
            AudioFormatError: If the input audio cannot be decoded.
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
