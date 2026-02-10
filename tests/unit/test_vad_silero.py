"""Testes do SileroVADClassifier.

Valida threshold mapping por sensibilidade, lazy loading, classificacao
de frames e reset de estado. Todos os testes usam mock do modelo Silero
(sem dependencia de torch/onnxruntime).
"""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from macaw._types import VADSensitivity
from macaw.vad.silero import SileroVADClassifier


def _make_mock_model(return_prob: float) -> MagicMock:
    """Cria mock do modelo Silero que retorna probabilidade fixa."""
    model = MagicMock()
    result = MagicMock()
    result.item.return_value = return_prob
    model.return_value = result
    return model


def _make_classifier_with_mock(
    sensitivity: VADSensitivity = VADSensitivity.NORMAL,
    return_prob: float = 0.5,
) -> SileroVADClassifier:
    """Cria classifier com modelo mock pre-carregado (bypassa lazy loading)."""
    classifier = SileroVADClassifier(sensitivity=sensitivity)
    classifier._model = _make_mock_model(return_prob)
    classifier._model_loaded = True
    return classifier


class TestSileroVADClassifier:
    def test_speech_frame_detected_above_threshold(self) -> None:
        """Modelo retorna prob 0.8, threshold 0.5 -> is_speech=True."""
        # Arrange
        classifier = _make_classifier_with_mock(
            sensitivity=VADSensitivity.NORMAL,
            return_prob=0.8,
        )
        frame = np.zeros(512, dtype=np.float32)

        # Act
        result = classifier.is_speech(frame)

        # Assert
        assert result is True

    def test_silence_frame_detected_below_threshold(self) -> None:
        """Modelo retorna prob 0.2, threshold 0.5 -> is_speech=False."""
        # Arrange
        classifier = _make_classifier_with_mock(
            sensitivity=VADSensitivity.NORMAL,
            return_prob=0.2,
        )
        frame = np.zeros(512, dtype=np.float32)

        # Act
        result = classifier.is_speech(frame)

        # Assert
        assert result is False

    def test_sensitivity_high_uses_threshold_03(self) -> None:
        """VADSensitivity.HIGH -> threshold=0.3."""
        # Arrange & Act
        classifier = SileroVADClassifier(sensitivity=VADSensitivity.HIGH)

        # Assert
        assert classifier.threshold == pytest.approx(0.3)

    def test_sensitivity_normal_uses_threshold_05(self) -> None:
        """VADSensitivity.NORMAL -> threshold=0.5."""
        # Arrange & Act
        classifier = SileroVADClassifier(sensitivity=VADSensitivity.NORMAL)

        # Assert
        assert classifier.threshold == pytest.approx(0.5)

    def test_sensitivity_low_uses_threshold_07(self) -> None:
        """VADSensitivity.LOW -> threshold=0.7."""
        # Arrange & Act
        classifier = SileroVADClassifier(sensitivity=VADSensitivity.LOW)

        # Assert
        assert classifier.threshold == pytest.approx(0.7)

    def test_set_sensitivity_updates_threshold(self) -> None:
        """set_sensitivity muda threshold e sensitivity."""
        # Arrange
        classifier = SileroVADClassifier(sensitivity=VADSensitivity.NORMAL)
        assert classifier.threshold == pytest.approx(0.5)

        # Act
        classifier.set_sensitivity(VADSensitivity.HIGH)

        # Assert
        assert classifier.sensitivity == VADSensitivity.HIGH
        assert classifier.threshold == pytest.approx(0.3)

    def test_lazy_loading_called_on_first_use(self) -> None:
        """_ensure_model_loaded e chamado na primeira get_speech_probability."""
        # Arrange
        classifier = SileroVADClassifier(sensitivity=VADSensitivity.NORMAL)
        frame = np.zeros(512, dtype=np.float32)

        mock_model = _make_mock_model(return_prob=0.5)

        # Act -- patch _ensure_model_loaded para injetar modelo mock
        with patch.object(classifier, "_ensure_model_loaded") as mock_ensure:

            def side_effect() -> None:
                classifier._model = mock_model
                classifier._model_loaded = True

            mock_ensure.side_effect = side_effect
            classifier.get_speech_probability(frame)

        # Assert
        mock_ensure.assert_called_once()

    def test_get_speech_probability_returns_float(self) -> None:
        """get_speech_probability retorna float entre 0 e 1."""
        # Arrange
        classifier = _make_classifier_with_mock(return_prob=0.73)
        frame = np.zeros(512, dtype=np.float32)

        # Act
        prob = classifier.get_speech_probability(frame)

        # Assert
        assert isinstance(prob, float)
        assert prob == pytest.approx(0.73)

    def test_reset_calls_model_reset_states(self) -> None:
        """reset() chama reset_states do modelo quando disponivel."""
        # Arrange
        classifier = _make_classifier_with_mock(return_prob=0.5)
        assert classifier._model is not None

        # Act
        classifier.reset()

        # Assert
        classifier._model.reset_states.assert_called_once()  # type: ignore[union-attr]

    def test_reset_without_loaded_model_is_noop(self) -> None:
        """reset() sem modelo carregado nao levanta erro."""
        # Arrange
        classifier = SileroVADClassifier(sensitivity=VADSensitivity.NORMAL)
        assert classifier._model is None

        # Act & Assert -- nenhuma excecao
        classifier.reset()

    def test_invalid_sample_rate_raises_value_error(self) -> None:
        """Sample rate diferente de 16000 levanta ValueError."""
        # Act & Assert
        with pytest.raises(ValueError, match="requer sample rate 16000Hz"):
            SileroVADClassifier(sample_rate=8000)

    def test_boundary_probability_at_threshold(self) -> None:
        """Probabilidade exatamente no threshold nao e classificada como fala."""
        # Arrange -- prob == threshold (0.5), is_speech usa >, nao >=
        classifier = _make_classifier_with_mock(
            sensitivity=VADSensitivity.NORMAL,
            return_prob=0.5,
        )
        frame = np.zeros(512, dtype=np.float32)

        # Act
        result = classifier.is_speech(frame)

        # Assert -- exatamente no threshold nao e fala (> e estrito)
        assert result is False
