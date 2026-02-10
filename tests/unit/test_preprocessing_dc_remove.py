"""Testes do DCRemoveStage.

Valida que o filtro Butterworth HPF remove DC offset sem degradar
sinais na banda de fala. Testa caching de coeficientes e edge cases.
"""

from __future__ import annotations

import numpy as np

from macaw.preprocessing.dc_remove import DCRemoveStage


def _make_sine(
    frequency: float = 440.0,
    sample_rate: int = 16000,
    duration: float = 0.5,
    amplitude: float = 0.5,
    dc_offset: float = 0.0,
) -> np.ndarray:
    """Gera sinal senoidal float32 com DC offset opcional."""
    t = np.arange(int(sample_rate * duration)) / sample_rate
    signal = amplitude * np.sin(2 * np.pi * frequency * t) + dc_offset
    return signal.astype(np.float32)


class TestDCRemoveStage:
    def test_dc_remove_offset_reduced(self) -> None:
        """Sinal com DC offset 0.1: apos filtragem, media < 0.01."""
        # Arrange
        stage = DCRemoveStage(cutoff_hz=20)
        audio = _make_sine(frequency=440.0, dc_offset=0.1, duration=1.0)
        assert abs(np.mean(audio)) > 0.09

        # Act
        result, sr = stage.process(audio, 16000)

        # Assert
        assert abs(np.mean(result)) < 0.01
        assert sr == 16000

    def test_dc_remove_preserves_signal(self) -> None:
        """Senoide pura sem DC: sinal nao e significativamente degradado."""
        # Arrange
        stage = DCRemoveStage(cutoff_hz=20)
        audio = _make_sine(frequency=440.0, dc_offset=0.0, duration=1.0)

        # Act
        result, _ = stage.process(audio, 16000)

        # Assert -- correlacao alta indica que o sinal foi preservado
        # Descarta primeiras amostras (transiente do filtro)
        skip = 1600  # 100ms a 16kHz
        correlation = np.corrcoef(audio[skip:], result[skip:])[0, 1]
        assert correlation > 0.95

    def test_dc_remove_large_offset(self) -> None:
        """DC offset de 0.5: removido efetivamente."""
        # Arrange
        stage = DCRemoveStage(cutoff_hz=20)
        audio = _make_sine(frequency=440.0, dc_offset=0.5, duration=1.0)
        assert abs(np.mean(audio)) > 0.45

        # Act
        result, _ = stage.process(audio, 16000)

        # Assert
        assert abs(np.mean(result)) < 0.01

    def test_dc_remove_coefficients_cached(self) -> None:
        """Mesmo sample rate: coeficientes nao sao recomputados."""
        # Arrange
        stage = DCRemoveStage(cutoff_hz=20)
        audio = _make_sine(duration=0.1)

        # Act -- processar duas vezes com mesmo sample rate
        stage.process(audio, 16000)
        sos_after_first = stage._cached_sos

        stage.process(audio, 16000)
        sos_after_second = stage._cached_sos

        # Assert -- mesmo objeto em memoria (nao recalculado)
        assert sos_after_first is sos_after_second
        assert stage._cached_sample_rate == 16000

    def test_dc_remove_coefficients_recalculated(self) -> None:
        """Sample rate diferente: coeficientes sao recalculados."""
        # Arrange
        stage = DCRemoveStage(cutoff_hz=20)
        audio_16k = _make_sine(sample_rate=16000, duration=0.1)
        audio_8k = _make_sine(sample_rate=8000, duration=0.1)

        # Act
        stage.process(audio_16k, 16000)
        sos_16k = stage._cached_sos

        stage.process(audio_8k, 8000)
        sos_8k = stage._cached_sos

        # Assert -- objetos diferentes (recalculados)
        assert sos_16k is not sos_8k
        assert stage._cached_sample_rate == 8000

    def test_dc_remove_preserves_float32(self) -> None:
        """Output e float32."""
        # Arrange
        stage = DCRemoveStage(cutoff_hz=20)
        audio = _make_sine(duration=0.1)

        # Act
        result, _ = stage.process(audio, 16000)

        # Assert
        assert result.dtype == np.float32

    def test_dc_remove_preserves_sample_rate(self) -> None:
        """Sample rate nao e alterado pelo stage."""
        # Arrange
        stage = DCRemoveStage(cutoff_hz=20)
        audio = _make_sine(duration=0.1)

        # Act
        _, sr = stage.process(audio, 44100)

        # Assert
        assert sr == 44100

    def test_dc_remove_empty_audio(self) -> None:
        """Array vazio retornado sem modificacao."""
        # Arrange
        stage = DCRemoveStage(cutoff_hz=20)
        empty = np.array([], dtype=np.float32)

        # Act
        result, sr = stage.process(empty, 16000)

        # Assert
        assert len(result) == 0
        assert sr == 16000

    def test_dc_remove_name_property(self) -> None:
        """Propriedade name retorna 'dc_remove'."""
        # Arrange
        stage = DCRemoveStage()

        # Act & Assert
        assert stage.name == "dc_remove"
