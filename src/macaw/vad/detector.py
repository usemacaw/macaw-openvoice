"""VAD Detector that orchestrates EnergyPreFilter + SileroVADClassifier.

Coordinates the voice activity detection pipeline in two stages:
1. EnergyPreFilter (fast, ~0.1ms) discards obvious silence frames
2. SileroVADClassifier (~2ms) classifies remaining frames as speech/silence

Manages debounce (min speech/silence duration) and max speech duration,
emitting VADEvent events (SPEECH_START, SPEECH_END) when confirmed
state transitions occur.

Total cost: ~0.1ms/frame when the energy pre-filter discards, ~2ms/frame when Silero is invoked.
"""

from __future__ import annotations

import enum
from dataclasses import dataclass
from typing import TYPE_CHECKING

from macaw._types import VADSensitivity
from macaw.logging import get_logger

if TYPE_CHECKING:
    import numpy as np

    from macaw.vad.energy import EnergyPreFilter
    from macaw.vad.silero import SileroVADClassifier

logger = get_logger("vad.detector")


class VADEventType(enum.Enum):
    """VAD event type emitted by the detector."""

    SPEECH_START = "speech_start"
    SPEECH_END = "speech_end"


@dataclass(frozen=True, slots=True)
class VADEvent:
    """VAD state transition event.

    Emitted when a confirmed transition between speech and silence occurs
    (after debounce). Includes a timestamp in milliseconds computed
    from the total processed samples.
    """

    type: VADEventType
    timestamp_ms: int


class VADDetector:
    """Voice activity detector with debounce and max speech duration.

    Orchestrates EnergyPreFilter (fast pre-filter) and SileroVADClassifier
    (neural classifier) to detect speech/silence transitions with configurable
    debounce.

    Behavior:
        - EnergyPreFilter classifies as silence -> Silero is NOT called
        - EnergyPreFilter classifies as non-silence -> Silero is called
        - After min_speech_duration_ms of consecutive speech -> SPEECH_START
        - After min_silence_duration_ms of consecutive silence (during speech) -> SPEECH_END
        - After max_speech_duration_ms of continuous speech -> force SPEECH_END
    """

    def __init__(
        self,
        energy_pre_filter: EnergyPreFilter,
        silero_classifier: SileroVADClassifier,
        sensitivity: VADSensitivity = VADSensitivity.NORMAL,
        sample_rate: int = 16000,
        min_speech_duration_ms: int = 250,
        min_silence_duration_ms: int = 300,
        max_speech_duration_ms: int = 30_000,
    ) -> None:
        """Initialize the detector.

        Args:
            energy_pre_filter: Energy pre-filter (drops obvious silence).
            silero_classifier: Silero VAD classifier.
            sensitivity: Sensitivity level (informational; thresholds are
                         from injected components).
            sample_rate: Sample rate in Hz (must be 16000).
            min_speech_duration_ms: Minimum consecutive speech duration before
                                    emitting SPEECH_START (default: 250ms).
            min_silence_duration_ms: Minimum consecutive silence duration before
                                     emitting SPEECH_END (default: 300ms).
            max_speech_duration_ms: Maximum continuous speech duration before
                                    forcing SPEECH_END (default: 30000ms).
        """
        self._energy_pre_filter = energy_pre_filter
        self._silero_classifier = silero_classifier
        self._sensitivity = sensitivity
        self._sample_rate = sample_rate
        self._min_speech_duration_ms = min_speech_duration_ms
        self._min_silence_duration_ms = min_silence_duration_ms
        self._max_speech_duration_ms = max_speech_duration_ms

        # Internal state
        self._samples_processed: int = 0
        self._is_speaking: bool = False
        self._consecutive_speech_samples: int = 0
        self._consecutive_silence_samples: int = 0
        self._speech_start_sample: int = 0  # sample where speech started

    def process_frame(self, frame: np.ndarray) -> VADEvent | None:
        """Process an audio frame and return an event if a transition occurred.

        Args:
            frame: Float32 mono numpy array at 16kHz. Typical size: 1024 samples (64ms).

        Returns:
            VADEvent if a state transition occurred (SPEECH_START or SPEECH_END),
            None if no transition occurred.
        """
        frame_samples = len(frame)

        # Classify frame: silence or speech
        is_speech = self._classify_frame(frame)

        # Update debounce counters (by total samples, not frame count)
        if is_speech:
            self._consecutive_speech_samples += frame_samples
            self._consecutive_silence_samples = 0
        else:
            self._consecutive_silence_samples += frame_samples
            self._consecutive_speech_samples = 0

        event = self._check_transitions(frame_samples)

        self._samples_processed += frame_samples

        return event

    def _classify_frame(self, frame: np.ndarray) -> bool:
        """Classify frame as speech (True) or silence (False).

        Uses the energy pre-filter first. If pre-filter says silence,
        Silero is NOT called (performance optimization).
        """
        if self._energy_pre_filter.is_silence(frame):
            return False

        return self._silero_classifier.is_speech(frame)

    def _check_transitions(self, frame_samples: int) -> VADEvent | None:
        """Check whether debounce or max duration triggers a transition."""
        # Timestamp of the CURRENT moment (after processing this frame)
        current_timestamp_ms = self._compute_timestamp_ms(self._samples_processed + frame_samples)

        if not self._is_speaking:
            return self._check_speech_start(current_timestamp_ms, frame_samples)

        return self._check_speech_end(current_timestamp_ms, frame_samples)

    def _check_speech_start(self, timestamp_ms: int, frame_samples: int) -> VADEvent | None:
        """Check if speech debounce was reached to emit SPEECH_START."""
        consecutive_speech_ms = self._samples_to_ms(self._consecutive_speech_samples)

        if consecutive_speech_ms >= self._min_speech_duration_ms:
            self._is_speaking = True
            self._speech_start_sample = self._samples_processed + frame_samples
            self._consecutive_speech_samples = 0
            logger.debug(
                "VAD speech start",
                timestamp_ms=timestamp_ms,
                consecutive_ms=consecutive_speech_ms,
            )
            return VADEvent(type=VADEventType.SPEECH_START, timestamp_ms=timestamp_ms)

        return None

    def _check_speech_end(self, timestamp_ms: int, frame_samples: int) -> VADEvent | None:
        """Check if silence debounce or max duration were reached."""
        # Check max speech duration first
        speech_duration_ms = self._compute_timestamp_ms(
            self._samples_processed + frame_samples - self._speech_start_sample
        )

        if speech_duration_ms >= self._max_speech_duration_ms:
            self._is_speaking = False
            self._consecutive_speech_samples = 0
            self._consecutive_silence_samples = 0
            logger.info(
                "VAD force speech end (max duration)",
                timestamp_ms=timestamp_ms,
                speech_duration_ms=speech_duration_ms,
            )
            return VADEvent(type=VADEventType.SPEECH_END, timestamp_ms=timestamp_ms)

        # Check silence debounce
        consecutive_silence_ms = self._samples_to_ms(self._consecutive_silence_samples)

        if consecutive_silence_ms >= self._min_silence_duration_ms:
            self._is_speaking = False
            self._consecutive_silence_samples = 0
            logger.debug(
                "VAD speech end",
                timestamp_ms=timestamp_ms,
                consecutive_silence_ms=consecutive_silence_ms,
            )
            return VADEvent(type=VADEventType.SPEECH_END, timestamp_ms=timestamp_ms)

        return None

    def _compute_timestamp_ms(self, total_samples: int) -> int:
        """Convert total processed samples to milliseconds."""
        return int(total_samples * 1000 / self._sample_rate)

    def _samples_to_ms(self, total_samples: int) -> int:
        """Convert total accumulated samples to milliseconds."""
        return int(total_samples * 1000 / self._sample_rate)

    def reset(self) -> None:
        """Reset state for a new session.

        Clears debounce counters, speaking state, and processed samples.
        Also resets Silero internal state (temporal context).
        """
        self._samples_processed = 0
        self._is_speaking = False
        self._consecutive_speech_samples = 0
        self._consecutive_silence_samples = 0
        self._speech_start_sample = 0
        self._silero_classifier.reset()

    @property
    def is_speaking(self) -> bool:
        """Whether the detector is currently in the speaking state."""
        return self._is_speaking
