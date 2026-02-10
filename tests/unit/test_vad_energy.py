"""Testes do EnergyPreFilter.

Valida que o pre-filtro de energia classifica corretamente frames como
silencio ou nao-silencio baseado em RMS (dBFS) e spectral flatness.
Testa mapeamento de sensibilidade e edge cases.
"""

from __future__ import annotations

import numpy as np

from macaw._types import VADSensitivity
from macaw.vad.energy import EnergyPreFilter


def _make_sine(
    frequency: float = 440.0,
    sample_rate: int = 16000,
    duration: float = 0.064,
    amplitude: float = 0.5,
) -> np.ndarray:
    """Gera sinal senoidal float32 (simula fala tonal)."""
    t = np.arange(int(sample_rate * duration)) / sample_rate
    signal = amplitude * np.sin(2 * np.pi * frequency * t)
    return signal.astype(np.float32)


def _make_white_noise(
    sample_rate: int = 16000,
    duration: float = 0.064,
    amplitude: float = 0.001,
) -> np.ndarray:
    """Gera ruido branco float32 com amplitude controlada."""
    rng = np.random.default_rng(seed=42)
    n_samples = int(sample_rate * duration)
    noise = amplitude * rng.standard_normal(n_samples)
    return noise.astype(np.float32)


class TestEnergyPreFilter:
    def test_silence_frame_classified_as_silence(self) -> None:
        """Frame de zeros e classificado como silencio."""
        # Arrange
        pre_filter = EnergyPreFilter(sensitivity=VADSensitivity.NORMAL)
        frame = np.zeros(1024, dtype=np.float32)

        # Act
        result = pre_filter.is_silence(frame)

        # Assert
        assert result is True

    def test_speech_frame_classified_as_non_silence(self) -> None:
        """Sine wave 440Hz com amplitude alta nao e silencio."""
        # Arrange
        pre_filter = EnergyPreFilter(sensitivity=VADSensitivity.NORMAL)
        frame = _make_sine(frequency=440.0, amplitude=0.5)

        # Act
        result = pre_filter.is_silence(frame)

        # Assert
        assert result is False

    def test_low_energy_noise_classified_as_silence(self) -> None:
        """Ruido branco com amplitude muito baixa e silencio."""
        # Arrange
        pre_filter = EnergyPreFilter(sensitivity=VADSensitivity.NORMAL)
        frame = _make_white_noise(amplitude=0.0001)

        # Act
        result = pre_filter.is_silence(frame)

        # Assert
        assert result is True

    def test_high_energy_noise_classified_as_non_silence(self) -> None:
        """Ruido branco com amplitude alta nao e silencio (RMS alto)."""
        # Arrange
        pre_filter = EnergyPreFilter(sensitivity=VADSensitivity.NORMAL)
        frame = _make_white_noise(amplitude=0.5)

        # Act
        result = pre_filter.is_silence(frame)

        # Assert
        assert result is False

    def test_sensitivity_high_detects_quiet_speech(self) -> None:
        """Frame com amplitude baixa (sussurro) nao e silencio em HIGH."""
        # Arrange -- amplitude baixa que fica entre -50dBFS e -40dBFS
        # amplitude=0.005 -> RMS ~0.0035 -> ~-49dBFS (acima de -50)
        pre_filter = EnergyPreFilter(sensitivity=VADSensitivity.HIGH)
        frame = _make_sine(frequency=440.0, amplitude=0.005)

        # Act
        result = pre_filter.is_silence(frame)

        # Assert -- HIGH threshold (-50dBFS) nao classifica como silencio
        assert result is False

    def test_tonal_signal_not_classified_as_silence_despite_low_energy(self) -> None:
        """Sine wave com energia baixa nao e silencio -- spectral flatness e baixa (tonal).

        O pre-filter exige AMBOS os criterios: energia baixa E flatness alta.
        Sinal tonal (sine wave) tem flatness ~0, entao mesmo com RMS abaixo
        do threshold, nao e classificado como silencio. Isso evita false positives
        em fala sussurrada.
        """
        # Arrange -- sine wave com energia baixa (RMS ~-49dBFS, abaixo de -30dBFS)
        pre_filter = EnergyPreFilter(sensitivity=VADSensitivity.LOW)
        frame = _make_sine(frequency=440.0, amplitude=0.005)

        # Act
        result = pre_filter.is_silence(frame)

        # Assert -- nao e silencio porque flatness de sine wave e ~0 (tonal)
        assert result is False

    def test_sensitivity_low_classifies_weak_noise_as_silence(self) -> None:
        """Ruido branco com amplitude entre -40 e -30 dBFS: LOW diz silencio, NORMAL nao."""
        # Arrange -- amplitude que gera RMS ~-35dBFS
        # amplitude=0.01 -> RMS ~0.01 -> 20*log10(0.01) = -40dBFS
        # amplitude=0.02 -> RMS ~0.02 -> 20*log10(0.02) = -34dBFS
        frame = _make_white_noise(amplitude=0.015)

        pre_filter_low = EnergyPreFilter(sensitivity=VADSensitivity.LOW)
        pre_filter_normal = EnergyPreFilter(sensitivity=VADSensitivity.NORMAL)

        # Act
        result_low = pre_filter_low.is_silence(frame)
        result_normal = pre_filter_normal.is_silence(frame)

        # Assert -- LOW (-30dBFS) classifica como silencio, NORMAL (-40dBFS) nao
        assert result_low is True
        assert result_normal is False

    def test_empty_frame_is_silence(self) -> None:
        """Array vazio e classificado como silencio."""
        # Arrange
        pre_filter = EnergyPreFilter(sensitivity=VADSensitivity.NORMAL)
        frame = np.array([], dtype=np.float32)

        # Act
        result = pre_filter.is_silence(frame)

        # Assert
        assert result is True

    def test_single_sample_frame_is_silence(self) -> None:
        """Frame com 1 sample e classificado como silencio."""
        # Arrange
        pre_filter = EnergyPreFilter(sensitivity=VADSensitivity.NORMAL)
        frame = np.array([0.5], dtype=np.float32)

        # Act
        result = pre_filter.is_silence(frame)

        # Assert
        assert result is True

    def test_energy_threshold_property(self) -> None:
        """Propriedade energy_threshold_dbfs retorna valor correto por sensibilidade."""
        # Arrange & Act & Assert
        assert EnergyPreFilter(VADSensitivity.HIGH).energy_threshold_dbfs == -50.0
        assert EnergyPreFilter(VADSensitivity.NORMAL).energy_threshold_dbfs == -40.0
        assert EnergyPreFilter(VADSensitivity.LOW).energy_threshold_dbfs == -30.0
