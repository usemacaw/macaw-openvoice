"""Tests for VADDetector.

Validates that the detector orchestrates EnergyPreFilter and SileroVADClassifier
correctly, managing debounce (min speech/silence duration),
max speech duration, and VAD event emission.

All tests use mocks -- no dependency on real torch/Silero.
"""

from __future__ import annotations

from unittest.mock import Mock

import numpy as np

from macaw.vad.detector import VADDetector, VADEvent, VADEventType
from tests.helpers import FRAME_SIZE, SAMPLE_RATE

_FRAME_DURATION_MS = FRAME_SIZE * 1000 // SAMPLE_RATE  # 64ms


def _make_frames(n: int, frame_size: int = FRAME_SIZE) -> list[np.ndarray]:
    """Generate N frames of float32 zeros."""
    return [np.zeros(frame_size, dtype=np.float32) for _ in range(n)]


def _make_energy_mock(*, is_silence: bool = False) -> Mock:
    """Create EnergyPreFilter mock with fixed return value."""
    mock = Mock()
    mock.is_silence.return_value = is_silence
    return mock


def _make_silero_mock(*, is_speech: bool = False) -> Mock:
    """Create SileroVADClassifier mock with fixed return value."""
    mock = Mock()
    mock.is_speech.return_value = is_speech
    return mock


def _make_detector(
    energy_is_silence: bool = False,
    silero_is_speech: bool = False,
    min_speech_duration_ms: int = 250,
    min_silence_duration_ms: int = 300,
    max_speech_duration_ms: int = 30_000,
) -> tuple[VADDetector, Mock, Mock]:
    """Create VADDetector with configured mocks. Returns (detector, energy_mock, silero_mock)."""
    energy_mock = _make_energy_mock(is_silence=energy_is_silence)
    silero_mock = _make_silero_mock(is_speech=silero_is_speech)

    detector = VADDetector(
        energy_pre_filter=energy_mock,
        silero_classifier=silero_mock,
        sample_rate=SAMPLE_RATE,
        min_speech_duration_ms=min_speech_duration_ms,
        min_silence_duration_ms=min_silence_duration_ms,
        max_speech_duration_ms=max_speech_duration_ms,
    )
    return detector, energy_mock, silero_mock


def _process_n_frames(
    detector: VADDetector,
    n: int,
    frame_size: int = FRAME_SIZE,
) -> list[VADEvent]:
    """Process N frames and return list of emitted events (excluding None)."""
    events = []
    for frame in _make_frames(n, frame_size):
        event = detector.process_frame(frame)
        if event is not None:
            events.append(event)
    return events


class TestSilenceFrames:
    """Frames classified as silence do not emit events."""

    def test_all_silence_no_events(self) -> None:
        """Sequence of silence frames does not emit any event."""
        # Arrange
        detector, _, _ = _make_detector(energy_is_silence=True)

        # Act
        events = _process_n_frames(detector, n=20)

        # Assert
        assert events == []

    def test_silence_via_energy_prefilter_skips_silero(self) -> None:
        """When EnergyPreFilter says silence, Silero is NOT called."""
        # Arrange
        detector, energy_mock, silero_mock = _make_detector(energy_is_silence=True)
        frames = _make_frames(5)

        # Act
        for frame in frames:
            detector.process_frame(frame)

        # Assert
        assert energy_mock.is_silence.call_count == 5
        silero_mock.is_speech.assert_not_called()


class TestSpeechStart:
    """SPEECH_START emission after speech debounce."""

    def test_speech_start_after_min_duration(self) -> None:
        """SPEECH_START emitted after min_speech_duration_ms of consecutive speech."""
        # Arrange -- 250ms min speech, frames de 64ms -> ceil(250/64) = 4 frames
        detector, _, _ = _make_detector(
            energy_is_silence=False,
            silero_is_speech=True,
            min_speech_duration_ms=250,
        )

        # Act
        events = _process_n_frames(detector, n=4)

        # Assert
        assert len(events) == 1
        assert events[0].type == VADEventType.SPEECH_START
        assert events[0].timestamp_ms == 4 * _FRAME_DURATION_MS  # 256ms

    def test_short_speech_burst_no_event(self) -> None:
        """Speech burst shorter than min_speech_duration does not emit event."""
        # Arrange -- 250ms min, but only 3 frames (192ms)
        detector, _, _ = _make_detector(
            energy_is_silence=False,
            silero_is_speech=True,
            min_speech_duration_ms=250,
        )

        # Act -- 3 speech frames, then silence
        events_speech: list[VADEvent] = []
        for frame in _make_frames(3):
            event = detector.process_frame(frame)
            if event is not None:
                events_speech.append(event)

        # Assert -- no SPEECH_START (did not reach min_speech_duration)
        assert events_speech == []

    def test_is_speaking_true_after_speech_start(self) -> None:
        """is_speaking returns True after SPEECH_START is emitted."""
        # Arrange
        detector, _, _ = _make_detector(
            energy_is_silence=False,
            silero_is_speech=True,
            min_speech_duration_ms=250,
        )
        assert detector.is_speaking is False

        # Act -- 4 speech frames = 256ms > 250ms
        _process_n_frames(detector, n=4)

        # Assert
        assert detector.is_speaking is True


class TestSpeechEnd:
    """SPEECH_END emission after silence debounce."""

    def test_speech_end_after_min_silence_duration(self) -> None:
        """SPEECH_START followed by SPEECH_END after min_silence_duration_ms."""
        # Arrange
        energy_mock = _make_energy_mock(is_silence=False)
        silero_mock = _make_silero_mock(is_speech=True)
        detector = VADDetector(
            energy_pre_filter=energy_mock,
            silero_classifier=silero_mock,
            sample_rate=SAMPLE_RATE,
            min_speech_duration_ms=250,
            min_silence_duration_ms=300,
        )

        # Act -- Phase 1: speech (4 frames = 256ms -> SPEECH_START)
        events_phase1 = _process_n_frames(detector, n=4)
        assert len(events_phase1) == 1
        assert events_phase1[0].type == VADEventType.SPEECH_START

        # Phase 2: silence (5 frames = 320ms -> SPEECH_END)
        silero_mock.is_speech.return_value = False
        events_phase2 = _process_n_frames(detector, n=5)

        # Assert
        assert len(events_phase2) == 1
        assert events_phase2[0].type == VADEventType.SPEECH_END

    def test_short_silence_during_speech_no_event(self) -> None:
        """Silence shorter than min_silence_duration does not emit SPEECH_END."""
        # Arrange
        energy_mock = _make_energy_mock(is_silence=False)
        silero_mock = _make_silero_mock(is_speech=True)
        detector = VADDetector(
            energy_pre_filter=energy_mock,
            silero_classifier=silero_mock,
            sample_rate=SAMPLE_RATE,
            min_speech_duration_ms=250,
            min_silence_duration_ms=300,
        )

        # Act -- Phase 1: speech (4 frames -> SPEECH_START)
        _process_n_frames(detector, n=4)
        assert detector.is_speaking is True

        # Phase 2: short silence (3 frames = 192ms < 300ms) -- should not emit SPEECH_END
        silero_mock.is_speech.return_value = False
        events_silence = _process_n_frames(detector, n=3)

        # Assert
        assert events_silence == []
        assert detector.is_speaking is True

    def test_is_speaking_false_after_speech_end(self) -> None:
        """is_speaking returns False after SPEECH_END is emitted."""
        # Arrange
        energy_mock = _make_energy_mock(is_silence=False)
        silero_mock = _make_silero_mock(is_speech=True)
        detector = VADDetector(
            energy_pre_filter=energy_mock,
            silero_classifier=silero_mock,
            sample_rate=SAMPLE_RATE,
            min_speech_duration_ms=250,
            min_silence_duration_ms=300,
        )

        # Act -- speech then silence
        _process_n_frames(detector, n=4)  # SPEECH_START
        silero_mock.is_speech.return_value = False
        _process_n_frames(detector, n=5)  # SPEECH_END

        # Assert
        assert detector.is_speaking is False


class TestEnergyPreFilterIntegration:
    """Verifies that energy pre-filter controls when Silero is called."""

    def test_energy_silence_skips_silero(self) -> None:
        """When energy pre-filter returns silence, Silero is not invoked."""
        # Arrange
        energy_mock = _make_energy_mock(is_silence=True)
        silero_mock = _make_silero_mock(is_speech=True)  # Silero would say "speech" if called
        detector = VADDetector(
            energy_pre_filter=energy_mock,
            silero_classifier=silero_mock,
            sample_rate=SAMPLE_RATE,
        )

        # Act
        _process_n_frames(detector, n=10)

        # Assert -- Silero never called
        silero_mock.is_speech.assert_not_called()

    def test_energy_non_silence_calls_silero(self) -> None:
        """When energy pre-filter returns non-silence, Silero is called."""
        # Arrange
        energy_mock = _make_energy_mock(is_silence=False)
        silero_mock = _make_silero_mock(is_speech=False)
        detector = VADDetector(
            energy_pre_filter=energy_mock,
            silero_classifier=silero_mock,
            sample_rate=SAMPLE_RATE,
        )

        # Act
        _process_n_frames(detector, n=3)

        # Assert -- Silero called for each frame
        assert silero_mock.is_speech.call_count == 3


class TestMaxSpeechDuration:
    """Force SPEECH_END after max_speech_duration_ms of continuous speech."""

    def test_force_speech_end_after_max_duration(self) -> None:
        """SPEECH_END forced after max_speech_duration_ms even without silence."""
        # Arrange -- max 640ms (= 10 frames of 64ms) for easier testing
        detector, _, _ = _make_detector(
            energy_is_silence=False,
            silero_is_speech=True,
            min_speech_duration_ms=64,  # 1 frame
            max_speech_duration_ms=640,
        )

        # Act
        all_events: list[VADEvent] = []
        # Frame 1: SPEECH_START (min_speech_duration_ms=64ms = 1 frame)
        # Frame 11: force SPEECH_END (640ms = 10 frames after speech start)
        for frame in _make_frames(15):
            event = detector.process_frame(frame)
            if event is not None:
                all_events.append(event)

        # Assert
        speech_starts = [e for e in all_events if e.type == VADEventType.SPEECH_START]
        speech_ends = [e for e in all_events if e.type == VADEventType.SPEECH_END]
        assert len(speech_starts) >= 1
        assert len(speech_ends) >= 1

    def test_max_duration_30s_default(self) -> None:
        """With default parameters, max speech duration is 30s."""
        # Arrange -- use larger frames to avoid processing 469 frames
        # 30000ms / 1000ms_per_frame = 30 frames of 16000 samples (1s each)
        large_frame_size = 16000  # 1 segundo de audio
        detector, _, _ = _make_detector(
            energy_is_silence=False,
            silero_is_speech=True,
            min_speech_duration_ms=250,
            max_speech_duration_ms=30_000,
        )

        # Act -- 4 frames of 1s to reach SPEECH_START (>250ms)
        # then more frames until reaching 30s total
        all_events: list[VADEvent] = []
        for _i in range(35):
            frame = np.zeros(large_frame_size, dtype=np.float32)
            event = detector.process_frame(frame)
            if event is not None:
                all_events.append(event)

        # Assert -- should have SPEECH_START and SPEECH_END (force)
        speech_starts = [e for e in all_events if e.type == VADEventType.SPEECH_START]
        speech_ends = [e for e in all_events if e.type == VADEventType.SPEECH_END]
        assert len(speech_starts) >= 1
        assert len(speech_ends) >= 1
        # SPEECH_END deve ter timestamp >= 30000ms
        assert speech_ends[0].timestamp_ms >= 30_000


class TestReset:
    """Verifies that reset() clears all state."""

    def test_reset_clears_all_state(self) -> None:
        """reset() clears samples_processed, is_speaking, counters."""
        # Arrange -- detector in speaking state
        energy_mock = _make_energy_mock(is_silence=False)
        silero_mock = _make_silero_mock(is_speech=True)
        detector = VADDetector(
            energy_pre_filter=energy_mock,
            silero_classifier=silero_mock,
            sample_rate=SAMPLE_RATE,
            min_speech_duration_ms=250,
        )

        # Process until SPEECH_START
        _process_n_frames(detector, n=4)
        assert detector.is_speaking is True

        # Act
        detector.reset()

        # Assert
        assert detector.is_speaking is False
        assert detector._samples_processed == 0
        assert detector._consecutive_speech_samples == 0
        assert detector._consecutive_silence_samples == 0
        silero_mock.reset.assert_called_once()

    def test_reset_allows_new_session(self) -> None:
        """After reset, detector can detect a new speech sequence."""
        # Arrange
        energy_mock = _make_energy_mock(is_silence=False)
        silero_mock = _make_silero_mock(is_speech=True)
        detector = VADDetector(
            energy_pre_filter=energy_mock,
            silero_classifier=silero_mock,
            sample_rate=SAMPLE_RATE,
            min_speech_duration_ms=250,
        )

        # First session
        events1 = _process_n_frames(detector, n=4)
        assert len(events1) == 1
        assert events1[0].type == VADEventType.SPEECH_START

        # Reset
        detector.reset()

        # Second session -- timestamps should start from zero
        events2 = _process_n_frames(detector, n=4)
        assert len(events2) == 1
        assert events2[0].type == VADEventType.SPEECH_START
        assert events2[0].timestamp_ms == events1[0].timestamp_ms  # mesmo offset relativo


class TestTimestamps:
    """Verifies correct timestamp calculation."""

    def test_timestamp_ms_calculated_from_samples(self) -> None:
        """timestamp_ms is calculated from processed samples."""
        # Arrange
        detector, _, _ = _make_detector(
            energy_is_silence=False,
            silero_is_speech=True,
            min_speech_duration_ms=250,
        )

        # Act -- 4 frames de 1024 samples = 4096 samples
        events = _process_n_frames(detector, n=4)

        # Assert -- 4096 / 16000 * 1000 = 256ms
        assert len(events) == 1
        assert events[0].timestamp_ms == 256

    def test_speech_end_timestamp_after_speech_start(self) -> None:
        """SPEECH_END timestamp is greater than SPEECH_START timestamp."""
        # Arrange
        energy_mock = _make_energy_mock(is_silence=False)
        silero_mock = _make_silero_mock(is_speech=True)
        detector = VADDetector(
            energy_pre_filter=energy_mock,
            silero_classifier=silero_mock,
            sample_rate=SAMPLE_RATE,
            min_speech_duration_ms=250,
            min_silence_duration_ms=300,
        )

        # Act -- speech (4 frames)
        events_start = _process_n_frames(detector, n=4)
        assert len(events_start) == 1

        # Then silence (5 frames)
        silero_mock.is_speech.return_value = False
        events_end = _process_n_frames(detector, n=5)
        assert len(events_end) == 1

        # Assert
        assert events_end[0].timestamp_ms > events_start[0].timestamp_ms

    def test_timestamps_accumulate_across_frames(self) -> None:
        """Timestamps accumulate correctly across many frames."""
        # Arrange
        detector, _, _ = _make_detector(
            energy_is_silence=True,  # silence -- no events
        )

        # Act -- 100 silence frames
        _process_n_frames(detector, n=100)

        # Assert -- samples processados = 100 * 1024 = 102400
        expected_ms = 100 * FRAME_SIZE * 1000 // SAMPLE_RATE  # 6400ms
        assert detector._samples_processed == 100 * FRAME_SIZE
        assert detector._samples_to_ms(detector._samples_processed) == expected_ms


class TestSpeechSilenceCycles:
    """Tests complete speech-silence-speech cycles."""

    def test_multiple_speech_cycles(self) -> None:
        """Multiple speech->silence cycles generate correct event pairs."""
        # Arrange
        energy_mock = _make_energy_mock(is_silence=False)
        silero_mock = _make_silero_mock(is_speech=True)
        detector = VADDetector(
            energy_pre_filter=energy_mock,
            silero_classifier=silero_mock,
            sample_rate=SAMPLE_RATE,
            min_speech_duration_ms=250,
            min_silence_duration_ms=300,
        )

        all_events: list[VADEvent] = []

        # Ciclo 1: speech (4 frames) + silence (5 frames)
        silero_mock.is_speech.return_value = True
        all_events.extend(_process_n_frames(detector, n=4))  # SPEECH_START
        silero_mock.is_speech.return_value = False
        all_events.extend(_process_n_frames(detector, n=5))  # SPEECH_END

        # Ciclo 2: speech (4 frames) + silence (5 frames)
        silero_mock.is_speech.return_value = True
        all_events.extend(_process_n_frames(detector, n=4))  # SPEECH_START
        silero_mock.is_speech.return_value = False
        all_events.extend(_process_n_frames(detector, n=5))  # SPEECH_END

        # Assert
        assert len(all_events) == 4
        assert all_events[0].type == VADEventType.SPEECH_START
        assert all_events[1].type == VADEventType.SPEECH_END
        assert all_events[2].type == VADEventType.SPEECH_START
        assert all_events[3].type == VADEventType.SPEECH_END

        # Timestamps are monotonically increasing
        for i in range(1, len(all_events)):
            assert all_events[i].timestamp_ms > all_events[i - 1].timestamp_ms

    def test_silence_resets_speech_counter(self) -> None:
        """Silence frame resets consecutive speech counter."""
        # Arrange -- min speech = 250ms (4 frames de 64ms)
        energy_mock = _make_energy_mock(is_silence=False)
        silero_mock = _make_silero_mock(is_speech=True)
        detector = VADDetector(
            energy_pre_filter=energy_mock,
            silero_classifier=silero_mock,
            sample_rate=SAMPLE_RATE,
            min_speech_duration_ms=250,
        )

        # Act -- 3 frames speech, 1 frame silence, 3 frames speech
        # Should not reach 4 consecutive frames
        all_events: list[VADEvent] = []

        # 3 speech frames
        for frame in _make_frames(3):
            event = detector.process_frame(frame)
            if event is not None:
                all_events.append(event)

        # 1 silence frame (resets counter)
        silero_mock.is_speech.return_value = False
        event = detector.process_frame(np.zeros(FRAME_SIZE, dtype=np.float32))
        if event is not None:
            all_events.append(event)

        # 3 speech frames (does not reach 4 consecutive)
        silero_mock.is_speech.return_value = True
        for frame in _make_frames(3):
            event = detector.process_frame(frame)
            if event is not None:
                all_events.append(event)

        # Assert -- no SPEECH_START
        assert all_events == []


class TestVADDetectorValidation:
    """Tests for VADDetector input validation."""

    def test_invalid_sample_rate_raises_value_error(self) -> None:
        """VADDetector rejects sample rates other than 16000."""
        import pytest

        energy_mock = _make_energy_mock(is_silence=False)
        silero_mock = _make_silero_mock(is_speech=False)

        with pytest.raises(ValueError, match="Only 16kHz supported"):
            VADDetector(
                energy_pre_filter=energy_mock,
                silero_classifier=silero_mock,
                sample_rate=8000,
            )

    def test_valid_sample_rate_16000_accepted(self) -> None:
        """VADDetector accepts sample_rate=16000 without error."""
        energy_mock = _make_energy_mock(is_silence=False)
        silero_mock = _make_silero_mock(is_speech=False)

        detector = VADDetector(
            energy_pre_filter=energy_mock,
            silero_classifier=silero_mock,
            sample_rate=16000,
        )
        assert detector._sample_rate == 16000


class TestVADEventDataclass:
    """Tests properties of the VADEvent dataclass."""

    def test_vad_event_is_frozen(self) -> None:
        """VADEvent is immutable (frozen=True)."""
        # Arrange
        event = VADEvent(type=VADEventType.SPEECH_START, timestamp_ms=100)

        # Act & Assert
        try:
            event.timestamp_ms = 200  # type: ignore[misc]
            raise AssertionError("Should raise FrozenInstanceError")
        except AttributeError:
            pass  # Expected -- frozen dataclass

    def test_vad_event_type_enum_values(self) -> None:
        """VADEventType has correct values."""
        assert VADEventType.SPEECH_START.value == "speech_start"
        assert VADEventType.SPEECH_END.value == "speech_end"
