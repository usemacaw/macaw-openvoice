"""Voice Activity Detection classifier using Silero VAD.

Wraps the Silero VAD model with lazy-loading and configurable sensitivity levels.
Returns speech probability per frame.

Debounce (min speech/silence duration) is NOT the responsibility of this class
-- it is handled by VADDetector (layer above).

Cost: ~2ms/frame on CPU.
"""

from __future__ import annotations

import asyncio
import threading
from typing import TYPE_CHECKING, Any

from macaw._types import VADSensitivity
from macaw.logging import get_logger

if TYPE_CHECKING:
    from collections.abc import Callable

    import numpy as np

logger = get_logger("vad.silero")

# Sensitivity mapping to speech probability threshold.
# HIGH = detects soft speech (whisper), LOW = requires clear speech.
_THRESHOLDS: dict[VADSensitivity, float] = {
    VADSensitivity.HIGH: 0.3,
    VADSensitivity.NORMAL: 0.5,
    VADSensitivity.LOW: 0.7,
}

# Sample rate expected by Silero VAD.
_EXPECTED_SAMPLE_RATE = 16000


class SileroVADClassifier:
    """Voice Activity Detection classifier using Silero VAD.

    Lazy-loads the model on the first call to get_speech_probability().
    Supports PyTorch (torch.jit) as the inference backend.

    The classifier returns speech probability for each frame.
    Debounce (min speech/silence duration) is NOT the responsibility of this class
    -- it is handled by VADDetector.
    """

    def __init__(
        self,
        sensitivity: VADSensitivity = VADSensitivity.NORMAL,
        sample_rate: int = _EXPECTED_SAMPLE_RATE,
    ) -> None:
        """Initialize the classifier.

        Args:
            sensitivity: Sensitivity level (adjusts threshold).
            sample_rate: Expected sample rate (must be 16000).

        Raises:
            ValueError: If sample_rate is not 16000.
        """
        if sample_rate != _EXPECTED_SAMPLE_RATE:
            msg = (
                f"Silero VAD requer sample rate {_EXPECTED_SAMPLE_RATE}Hz, "
                f"recebido {sample_rate}Hz. Faca o resample antes."
            )
            raise ValueError(msg)

        self._sensitivity = sensitivity
        self._sample_rate = sample_rate
        self._threshold = _THRESHOLDS[sensitivity]
        self._model: object | None = None
        self._model_loaded = False
        self._to_tensor: Callable[[Any], Any] | None = None
        self._load_lock = threading.Lock()

    @property
    def threshold(self) -> float:
        """Current speech probability threshold."""
        return self._threshold

    @property
    def sensitivity(self) -> VADSensitivity:
        """Current sensitivity level."""
        return self._sensitivity

    def set_sensitivity(self, sensitivity: VADSensitivity) -> None:
        """Update sensitivity (and corresponding threshold).

        Args:
            sensitivity: New sensitivity level.
        """
        self._sensitivity = sensitivity
        self._threshold = _THRESHOLDS[sensitivity]

    async def preload(self) -> None:
        """Preload the Silero VAD model in a separate thread.

        Avoids blocking the asyncio event loop during download/load on first use.
        Call before starting streaming.

        Raises:
            ImportError: If torch is not installed.
        """
        loop = asyncio.get_running_loop()
        await loop.run_in_executor(None, self._ensure_model_loaded)

    def _ensure_model_loaded(self) -> None:
        """Lazy-load the Silero VAD model (thread-safe).

        Attempts to load via torch.hub. If torch is not available,
        raises ImportError with installation instructions.

        Uses threading.Lock to ensure the model is loaded only once
        even with concurrent calls.

        Raises:
            ImportError: If torch is not installed.
        """
        if self._model_loaded:
            return

        with self._load_lock:
            # Double-check after acquiring lock
            if self._model_loaded:
                return

            try:
                import torch
            except ImportError:
                msg = "Silero VAD requires torch. Install with: pip install torch"
                raise ImportError(msg) from None

            try:
                model, _utils = torch.hub.load(
                    repo_or_dir="snakers4/silero-vad",
                    model="silero_vad",
                    force_reload=False,
                )
            except ImportError as exc:
                msg = f"Silero VAD failed to load: {exc}. Try: pip install torchaudio"
                raise ImportError(msg) from exc

            self._model = model
            self._to_tensor = torch.from_numpy
            self._model_loaded = True
            logger.info("Silero VAD loaded via PyTorch")

    def get_speech_probability(self, frame: np.ndarray) -> float:
        """Compute speech probability for an audio frame.

        If the frame is larger than 512 samples, it is split into 512-sample
        sub-frames and processed sequentially (preserving Silero internal state).
        Returns the maximum probability among sub-frames.

        Args:
            frame: Float32 mono numpy array, 16kHz.

        Returns:
            Speech probability between 0.0 and 1.0.
        """
        self._ensure_model_loaded()

        chunk_size = 512
        if len(frame) <= chunk_size:
            tensor = self._to_tensor(frame) if self._to_tensor is not None else frame
            prob = self._model(tensor, self._sample_rate)  # type: ignore[misc, operator]
            return float(prob.item())

        max_prob = 0.0
        for offset in range(0, len(frame), chunk_size):
            sub_frame = frame[offset : offset + chunk_size]
            if len(sub_frame) < chunk_size:
                break
            tensor = self._to_tensor(sub_frame) if self._to_tensor is not None else sub_frame
            prob = self._model(tensor, self._sample_rate)  # type: ignore[misc, operator]
            max_prob = max(max_prob, float(prob.item()))
        return max_prob

    def is_speech(self, frame: np.ndarray) -> bool:
        """Check whether a frame contains speech (prob > threshold).

        Args:
            frame: Float32 mono numpy array, 16kHz.

        Returns:
            True if speech probability > threshold.
        """
        return self.get_speech_probability(frame) > self._threshold

    def reset(self) -> None:
        """Reset internal model state (for start of a new session).

        Silero VAD keeps state between calls (temporal context).
        Calling reset() at the start of each session ensures previous
        state does not interfere.
        """
        if self._model is not None and hasattr(self._model, "reset_states"):
            self._model.reset_states()
