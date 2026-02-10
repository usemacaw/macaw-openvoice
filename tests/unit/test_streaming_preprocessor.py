"""Testes unitarios para StreamingPreprocessor.

Valida o adapter de preprocessing frame-by-frame para streaming.
Usa stages reais (ResampleStage, DCRemoveStage, GainNormalizeStage).
"""

from __future__ import annotations

import numpy as np
import pytest

from macaw.exceptions import AudioFormatError
from macaw.preprocessing.dc_remove import DCRemoveStage
from macaw.preprocessing.gain_normalize import GainNormalizeStage
from macaw.preprocessing.resample import ResampleStage
from macaw.preprocessing.streaming import StreamingPreprocessor


def _make_pcm16_frame(freq_hz: float, duration_ms: float, sample_rate: int) -> bytes:
    """Gera frame PCM 16-bit com sine wave.

    Args:
        freq_hz: Frequencia da onda senoidal em Hz.
        duration_ms: Duracao do frame em milissegundos.
        sample_rate: Sample rate em Hz.

    Returns:
        Bytes PCM 16-bit little-endian (mono).
    """
    num_samples = int(sample_rate * duration_ms / 1000)
    t = np.arange(num_samples) / sample_rate
    audio = np.sin(2 * np.pi * freq_hz * t) * 0.5  # amplitude 0.5
    int16 = (audio * 32767).astype(np.int16)
    return int16.tobytes()


class TestStreamingPreprocessor:
    """Testes do StreamingPreprocessor."""

    def test_process_frame_pcm16_44khz_to_float32_16khz(self) -> None:
        """Frame PCM 44.1kHz e convertido para float32 16kHz mono."""
        stages = [
            ResampleStage(target_sample_rate=16000),
            DCRemoveStage(cutoff_hz=20),
            GainNormalizeStage(target_dbfs=-3.0),
        ]
        preprocessor = StreamingPreprocessor(stages=stages, input_sample_rate=44100)

        frame = _make_pcm16_frame(freq_hz=440.0, duration_ms=40.0, sample_rate=44100)

        result = preprocessor.process_frame(frame)

        assert result.dtype == np.float32
        # 40ms a 44100Hz = 1764 samples, resampleado para 16kHz ~= 640 samples
        expected_samples = int(16000 * 40 / 1000)
        # Tolerancia de +-2 samples por arredondamento do resample
        assert abs(len(result) - expected_samples) <= 2
        # Deve ter conteudo (nao silencio)
        assert np.max(np.abs(result)) > 0.01

    def test_process_frame_pcm16_16khz_no_resample(self) -> None:
        """Frame PCM 16kHz retorna sem resample, mantendo ~mesmo numero de samples."""
        stages = [
            ResampleStage(target_sample_rate=16000),
            DCRemoveStage(cutoff_hz=20),
            GainNormalizeStage(target_dbfs=-3.0),
        ]
        preprocessor = StreamingPreprocessor(stages=stages, input_sample_rate=16000)

        duration_ms = 30.0
        frame = _make_pcm16_frame(freq_hz=440.0, duration_ms=duration_ms, sample_rate=16000)

        result = preprocessor.process_frame(frame)

        assert result.dtype == np.float32
        expected_samples = int(16000 * duration_ms / 1000)
        assert len(result) == expected_samples

    def test_process_frame_removes_dc_offset(self) -> None:
        """DC offset e removido pelo DCRemoveStage."""
        stages = [DCRemoveStage(cutoff_hz=20)]
        preprocessor = StreamingPreprocessor(stages=stages, input_sample_rate=16000)

        # Gerar sinal com DC offset: sine 440Hz + offset de 0.3
        num_samples = 16000  # 1 segundo para o filtro ser efetivo
        t = np.arange(num_samples) / 16000
        audio_with_dc = np.sin(2 * np.pi * 440 * t) * 0.3 + 0.3
        int16 = (audio_with_dc * 32767).astype(np.int16)
        frame = int16.tobytes()

        result = preprocessor.process_frame(frame)

        # DC offset (media) deve ser significativamente menor apos filtragem
        assert abs(np.mean(result)) < 0.05

    def test_process_frame_normalizes_gain(self) -> None:
        """Gain e normalizado para -3dBFS pelo GainNormalizeStage."""
        stages = [GainNormalizeStage(target_dbfs=-3.0)]
        preprocessor = StreamingPreprocessor(stages=stages, input_sample_rate=16000)

        # Sinal com amplitude baixa (0.1 = ~-20dBFS)
        frame = _make_pcm16_frame(freq_hz=440.0, duration_ms=30.0, sample_rate=16000)
        # Reduzir amplitude: recriar com amplitude menor
        num_samples = int(16000 * 30 / 1000)
        t = np.arange(num_samples) / 16000
        audio_quiet = np.sin(2 * np.pi * 440 * t) * 0.1
        int16 = (audio_quiet * 32767).astype(np.int16)
        frame = int16.tobytes()

        result = preprocessor.process_frame(frame)

        # Pico deve estar proximo de -3dBFS (~0.708)
        peak = np.max(np.abs(result))
        target_linear = 10 ** (-3.0 / 20)  # ~0.708
        assert abs(peak - target_linear) < 0.05

    def test_process_frame_empty_bytes_returns_empty(self) -> None:
        """Bytes vazios retornam array vazio."""
        stages = [
            ResampleStage(target_sample_rate=16000),
            DCRemoveStage(cutoff_hz=20),
            GainNormalizeStage(target_dbfs=-3.0),
        ]
        preprocessor = StreamingPreprocessor(stages=stages, input_sample_rate=16000)

        result = preprocessor.process_frame(b"")

        assert result.dtype == np.float32
        assert len(result) == 0

    def test_process_frame_odd_bytes_raises_error(self) -> None:
        """Bytes impares levantam AudioFormatError."""
        stages = [ResampleStage(target_sample_rate=16000)]
        preprocessor = StreamingPreprocessor(stages=stages, input_sample_rate=16000)

        with pytest.raises(AudioFormatError, match="numero par de bytes"):
            preprocessor.process_frame(b"\x00\x01\x02")

    def test_set_input_sample_rate(self) -> None:
        """set_input_sample_rate altera o sample rate de entrada."""
        stages = [ResampleStage(target_sample_rate=16000)]
        preprocessor = StreamingPreprocessor(stages=stages, input_sample_rate=16000)

        assert preprocessor.input_sample_rate == 16000

        preprocessor.set_input_sample_rate(8000)

        assert preprocessor.input_sample_rate == 8000

        # Frame 8kHz deve ser resampleado para 16kHz
        frame = _make_pcm16_frame(freq_hz=440.0, duration_ms=40.0, sample_rate=8000)
        result = preprocessor.process_frame(frame)

        # 40ms a 8kHz = 320 samples, resampleado para 16kHz = 640 samples
        expected_samples = int(16000 * 40 / 1000)
        assert abs(len(result) - expected_samples) <= 2
