"""Testes do GainNormalizeStage.

Valida normalizacao de amplitude, protecao contra clipping,
tratamento de silencio e audio vazio.
"""

from __future__ import annotations

import numpy as np
import pytest

from macaw.preprocessing.gain_normalize import GainNormalizeStage


def peak_dbfs(audio: np.ndarray) -> float:
    """Calcula nivel de pico em dBFS."""
    peak = np.max(np.abs(audio))
    if peak < 1e-10:
        return -float("inf")
    return 20 * np.log10(peak)


def make_sine(
    frequency: float = 440.0,
    sample_rate: int = 16000,
    duration: float = 0.1,
    amplitude: float = 0.5,
) -> np.ndarray:
    """Cria onda senoidal float32 com amplitude especificada."""
    t = np.arange(int(sample_rate * duration), dtype=np.float32) / sample_rate
    return (amplitude * np.sin(2 * np.pi * frequency * t)).astype(np.float32)


class TestGainNormalizeStage:
    def test_gain_normalize_low_amplitude(self) -> None:
        """Audio com pico em -20dBFS normalizado para -3dBFS."""
        # Arrange
        amplitude_linear = 10 ** (-20.0 / 20)  # ~0.1
        audio = make_sine(amplitude=amplitude_linear)
        stage = GainNormalizeStage(target_dbfs=-3.0)

        # Act
        result, _sr = stage.process(audio, 16000)

        # Assert
        result_dbfs = peak_dbfs(result)
        assert result_dbfs == pytest.approx(-3.0, abs=0.5)

    def test_gain_normalize_high_amplitude(self) -> None:
        """Audio com pico em -1dBFS normalizado para -3dBFS."""
        # Arrange
        amplitude_linear = 10 ** (-1.0 / 20)  # ~0.891
        audio = make_sine(amplitude=amplitude_linear)
        stage = GainNormalizeStage(target_dbfs=-3.0)

        # Act
        result, _sr = stage.process(audio, 16000)

        # Assert
        result_dbfs = peak_dbfs(result)
        assert result_dbfs == pytest.approx(-3.0, abs=0.5)

    def test_gain_normalize_silence(self) -> None:
        """Audio todo zeros retorna inalterado, sem divisao por zero."""
        # Arrange
        audio = np.zeros(1600, dtype=np.float32)
        stage = GainNormalizeStage(target_dbfs=-3.0)

        # Act
        result, sr = stage.process(audio, 16000)

        # Assert
        np.testing.assert_array_equal(result, audio)
        assert sr == 16000

    def test_gain_normalize_near_zero(self) -> None:
        """Audio com pico abaixo de 1e-10 retorna inalterado."""
        # Arrange
        audio = np.full(1600, 1e-12, dtype=np.float32)
        stage = GainNormalizeStage(target_dbfs=-3.0)

        # Act
        result, sr = stage.process(audio, 16000)

        # Assert
        np.testing.assert_array_equal(result, audio)
        assert sr == 16000

    def test_gain_normalize_clipping_protection(self) -> None:
        """Audio que excederia 0dBFS apos ganho e clipado em [-1.0, 1.0]."""
        # Arrange: audio com pico em -40dBFS, target em -0.1dBFS
        # Ganho necessario seria enorme, resultando em clipping
        amplitude_linear = 10 ** (-40.0 / 20)  # ~0.01
        audio = make_sine(amplitude=amplitude_linear)
        # Target muito alto forca ganho grande
        stage = GainNormalizeStage(target_dbfs=-0.1)

        # Act
        result, _sr = stage.process(audio, 16000)

        # Assert: nenhum sample excede [-1.0, 1.0]
        assert np.all(result >= -1.0)
        assert np.all(result <= 1.0)

    def test_gain_normalize_preserves_float32(self) -> None:
        """Output e sempre float32."""
        # Arrange
        audio = make_sine(amplitude=0.5)
        stage = GainNormalizeStage(target_dbfs=-3.0)

        # Act
        result, _sr = stage.process(audio, 16000)

        # Assert
        assert result.dtype == np.float32

    def test_gain_normalize_preserves_sample_rate(self) -> None:
        """Sample rate nao e alterado pelo stage."""
        # Arrange
        audio = make_sine(amplitude=0.5)
        stage = GainNormalizeStage(target_dbfs=-3.0)

        # Act
        for sr in [8000, 16000, 44100, 48000]:
            _result, result_sr = stage.process(audio, sr)

            # Assert
            assert result_sr == sr

    def test_gain_normalize_empty_audio(self) -> None:
        """Array vazio retorna inalterado."""
        # Arrange
        audio = np.array([], dtype=np.float32)
        stage = GainNormalizeStage(target_dbfs=-3.0)

        # Act
        result, sr = stage.process(audio, 16000)

        # Assert
        assert len(result) == 0
        assert sr == 16000

    def test_gain_normalize_name_property(self) -> None:
        """Property name retorna 'gain_normalize'."""
        # Arrange & Act
        stage = GainNormalizeStage()

        # Assert
        assert stage.name == "gain_normalize"

    def test_gain_normalize_custom_target(self) -> None:
        """Target customizado (-6.0 dBFS) funciona corretamente."""
        # Arrange
        amplitude_linear = 10 ** (-20.0 / 20)  # ~0.1
        audio = make_sine(amplitude=amplitude_linear)
        stage = GainNormalizeStage(target_dbfs=-6.0)

        # Act
        result, _sr = stage.process(audio, 16000)

        # Assert
        result_dbfs = peak_dbfs(result)
        assert result_dbfs == pytest.approx(-6.0, abs=0.5)
