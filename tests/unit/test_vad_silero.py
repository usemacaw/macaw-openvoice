"""Tests for SileroVADClassifier.

Validates threshold mapping by sensitivity, lazy loading, frame classification
and state reset. All tests use a mock Silero model
(no torch/onnxruntime dependency).
"""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from macaw._types import VADSensitivity
from macaw.vad.silero import SileroVADClassifier


def _make_mock_model(return_prob: float) -> MagicMock:
    """Create mock Silero model that returns a fixed probability."""
    model = MagicMock()
    result = MagicMock()
    result.item.return_value = return_prob
    model.return_value = result
    return model


def _make_classifier_with_mock(
    sensitivity: VADSensitivity = VADSensitivity.NORMAL,
    return_prob: float = 0.5,
) -> SileroVADClassifier:
    """Create classifier with pre-loaded mock model (bypasses lazy loading)."""
    classifier = SileroVADClassifier(sensitivity=sensitivity)
    classifier._model = _make_mock_model(return_prob)
    classifier._model_loaded = True
    return classifier


class TestSileroVADClassifier:
    def test_speech_frame_detected_above_threshold(self) -> None:
        """Model returns prob 0.8, threshold 0.5 -> is_speech=True."""
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
        """Model returns prob 0.2, threshold 0.5 -> is_speech=False."""
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
        """set_sensitivity changes threshold and sensitivity."""
        # Arrange
        classifier = SileroVADClassifier(sensitivity=VADSensitivity.NORMAL)
        assert classifier.threshold == pytest.approx(0.5)

        # Act
        classifier.set_sensitivity(VADSensitivity.HIGH)

        # Assert
        assert classifier.sensitivity == VADSensitivity.HIGH
        assert classifier.threshold == pytest.approx(0.3)

    def test_lazy_loading_called_on_first_use(self) -> None:
        """_ensure_model_loaded is called on first get_speech_probability."""
        # Arrange
        classifier = SileroVADClassifier(sensitivity=VADSensitivity.NORMAL)
        frame = np.zeros(512, dtype=np.float32)

        mock_model = _make_mock_model(return_prob=0.5)

        # Act -- patch _ensure_model_loaded to inject mock model
        with patch.object(classifier, "_ensure_model_loaded") as mock_ensure:

            def side_effect() -> None:
                classifier._model = mock_model
                classifier._model_loaded = True

            mock_ensure.side_effect = side_effect
            classifier.get_speech_probability(frame)

        # Assert
        mock_ensure.assert_called_once()

    def test_get_speech_probability_returns_float(self) -> None:
        """get_speech_probability returns float between 0 and 1."""
        # Arrange
        classifier = _make_classifier_with_mock(return_prob=0.73)
        frame = np.zeros(512, dtype=np.float32)

        # Act
        prob = classifier.get_speech_probability(frame)

        # Assert
        assert isinstance(prob, float)
        assert prob == pytest.approx(0.73)

    def test_reset_calls_model_reset_states(self) -> None:
        """reset() calls reset_states on the model when available."""
        # Arrange
        classifier = _make_classifier_with_mock(return_prob=0.5)
        assert classifier._model is not None

        # Act
        classifier.reset()

        # Assert
        classifier._model.reset_states.assert_called_once()  # type: ignore[union-attr]

    def test_reset_without_loaded_model_is_noop(self) -> None:
        """reset() without loaded model does not raise error."""
        # Arrange
        classifier = SileroVADClassifier(sensitivity=VADSensitivity.NORMAL)
        assert classifier._model is None

        # Act & Assert -- no exception
        classifier.reset()

    def test_invalid_sample_rate_raises_value_error(self) -> None:
        """Sample rate different from 16000 raises ValueError."""
        # Act & Assert
        with pytest.raises(ValueError, match="requires sample rate 16000Hz"):
            SileroVADClassifier(sample_rate=8000)

    def test_boundary_probability_at_threshold(self) -> None:
        """Probability exactly at threshold is not classified as speech."""
        # Arrange -- prob == threshold (0.5), is_speech uses >, not >=
        classifier = _make_classifier_with_mock(
            sensitivity=VADSensitivity.NORMAL,
            return_prob=0.5,
        )
        frame = np.zeros(512, dtype=np.float32)

        # Act
        result = classifier.is_speech(frame)

        # Assert -- exactly at threshold is not speech (> is strict)
        assert result is False


class TestSileroThresholdOverride:
    """Tests for the threshold_override keyword argument."""

    def test_override_bypasses_sensitivity_preset(self) -> None:
        """threshold_override=0.4 ignores NORMAL preset (0.5)."""
        # Arrange & Act
        classifier = SileroVADClassifier(
            sensitivity=VADSensitivity.NORMAL,
            threshold_override=0.4,
        )

        # Assert
        assert classifier.threshold == pytest.approx(0.4)

    def test_override_none_uses_preset(self) -> None:
        """threshold_override=None falls back to sensitivity preset."""
        # Arrange & Act
        classifier = SileroVADClassifier(
            sensitivity=VADSensitivity.HIGH,
            threshold_override=None,
        )

        # Assert
        assert classifier.threshold == pytest.approx(0.3)

    def test_override_affects_is_speech(self) -> None:
        """Override threshold changes speech detection behavior."""
        # Arrange — prob 0.35, override threshold 0.3 => speech
        classifier = SileroVADClassifier(
            sensitivity=VADSensitivity.NORMAL,
            threshold_override=0.3,
        )
        classifier._model = _make_mock_model(return_prob=0.35)
        classifier._model_loaded = True
        frame = np.zeros(512, dtype=np.float32)

        # Act
        result = classifier.is_speech(frame)

        # Assert — 0.35 > 0.3 threshold
        assert result is True


class TestSileroSubFramePadding:
    """Tests for trailing sub-frame zero-padding (M-33 fix)."""

    def test_trailing_subframe_is_padded_not_dropped(self) -> None:
        """Frame with trailing sub-frame < 512 is zero-padded and processed.

        Before the fix, a 700-sample frame would only process the first
        512 samples and silently drop the last 188. Now it pads them to 512.
        """
        # Arrange
        classifier = _make_classifier_with_mock(
            sensitivity=VADSensitivity.NORMAL,
            return_prob=0.8,
        )
        # 700 samples = 512 (full chunk) + 188 (trailing, needs padding)
        frame = np.zeros(700, dtype=np.float32)

        # Act
        prob = classifier.get_speech_probability(frame)

        # Assert -- model should be called twice (512 + 188 padded to 512)
        assert classifier._model.call_count == 2  # type: ignore[union-attr]
        assert prob == pytest.approx(0.8)

    def test_exact_chunk_multiple_no_padding(self) -> None:
        """Frame that is exact multiple of 512 does not trigger padding."""
        # Arrange
        classifier = _make_classifier_with_mock(
            sensitivity=VADSensitivity.NORMAL,
            return_prob=0.6,
        )
        # 1024 samples = 2 * 512, no trailing sub-frame
        frame = np.zeros(1024, dtype=np.float32)

        # Act
        prob = classifier.get_speech_probability(frame)

        # Assert -- model called exactly twice
        assert classifier._model.call_count == 2  # type: ignore[union-attr]
        assert prob == pytest.approx(0.6)

    def test_padded_subframe_uses_zeros(self) -> None:
        """Trailing sub-frame is padded with zeros (not garbage)."""
        # Arrange
        classifier = _make_classifier_with_mock(
            sensitivity=VADSensitivity.NORMAL,
            return_prob=0.5,
        )
        classifier._to_tensor = lambda x: x  # Pass numpy arrays through

        # 600 samples: first 512 processed, last 88 padded to 512
        frame = np.ones(600, dtype=np.float32) * 0.3

        # Act
        classifier.get_speech_probability(frame)

        # Assert -- second call should have a 512-sample tensor
        calls = classifier._model.call_args_list  # type: ignore[union-attr]
        assert len(calls) == 2
        second_tensor = calls[1][0][0]
        assert len(second_tensor) == 512
        # First 88 samples should be 0.3, rest should be 0.0
        np.testing.assert_allclose(second_tensor[:88], 0.3, atol=1e-6)
        np.testing.assert_allclose(second_tensor[88:], 0.0, atol=1e-6)
