"""Streaming Preprocessor — adapter frame-by-frame para preprocessing de audio.

Recebe bytes PCM raw (int16) do WebSocket, converte para numpy float32,
aplica stages do Audio Preprocessing Pipeline em sequencia, e retorna
float32 16kHz mono.

Diferenca do AudioPreprocessingPipeline (batch):
- Batch: recebe arquivo completo (WAV/FLAC), decodifica via soundfile
- Streaming: recebe frames PCM 16-bit crus, converte diretamente
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

from macaw.exceptions import AudioFormatError
from macaw.logging import get_logger
from macaw.workers.audio_utils import pcm_bytes_to_float32

if TYPE_CHECKING:
    from macaw.preprocessing.stages import AudioStage

logger = get_logger("preprocessing.streaming")


class StreamingPreprocessor:
    """Adapter de preprocessing para streaming frame-by-frame.

    Recebe bytes PCM raw (int16) do WebSocket, converte para numpy float32,
    aplica stages de M4 em sequencia, e retorna float32 16kHz mono.

    Args:
        stages: Lista de AudioStage a aplicar em sequencia.
        input_sample_rate: Sample rate do audio de entrada (default 16000).
                           Pode ser alterado via set_input_sample_rate().
    """

    def __init__(
        self,
        stages: list[AudioStage],
        input_sample_rate: int = 16000,
    ) -> None:
        self._stages = stages
        self._input_sample_rate = input_sample_rate

    @property
    def input_sample_rate(self) -> int:
        """Sample rate atual de entrada."""
        return self._input_sample_rate

    def set_input_sample_rate(self, sample_rate: int) -> None:
        """Atualiza o sample rate de entrada (ex: via session.configure).

        Args:
            sample_rate: Novo sample rate em Hz.
        """
        self._input_sample_rate = sample_rate

    def process_frame(self, raw_bytes: bytes) -> np.ndarray:
        """Processa um frame de audio cru.

        Converte PCM int16 bytes para numpy float32 normalizado [-1.0, 1.0],
        aplica todos os stages em sequencia, e retorna o resultado.

        Args:
            raw_bytes: Bytes PCM 16-bit little-endian (mono).

        Returns:
            Array numpy float32 16kHz mono, processado por todos os stages.

        Raises:
            AudioFormatError: Se os bytes nao tem tamanho par
                (PCM 16-bit = 2 bytes/sample).
        """
        if len(raw_bytes) % 2 != 0:
            raise AudioFormatError("Audio PCM 16-bit deve ter numero par de bytes")

        if len(raw_bytes) == 0:
            return np.array([], dtype=np.float32)

        audio = pcm_bytes_to_float32(raw_bytes)

        sample_rate = self._input_sample_rate

        for stage in self._stages:
            audio, sample_rate = stage.process(audio, sample_rate)

        return audio
