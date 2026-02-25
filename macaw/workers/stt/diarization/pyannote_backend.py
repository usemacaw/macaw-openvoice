"""PyAnnote-based speaker diarization backend.

Uses pyannote.audio Pipeline for speaker diarization. pyannote.audio is an
OPTIONAL dependency -- if not installed, ``create_diarizer()`` returns None.

Inference is CPU-bound and runs in an executor thread to avoid blocking
the asyncio event loop.
"""

from __future__ import annotations

import asyncio
import struct
from typing import Any

from macaw._types import SpeakerSegment
from macaw.logging import get_logger
from macaw.workers.stt.diarization.interface import DiarizationBackend

logger = get_logger("worker.stt.diarization.pyannote")

_PCM_SAMPLE_WIDTH = 2  # 16-bit = 2 bytes per sample


class PyAnnoteDiarizer(DiarizationBackend):
    """Speaker diarization using pyannote.audio Pipeline.

    The model is loaded lazily on the first ``diarize()`` call to avoid
    startup overhead when diarization is not requested. All heavy inference
    runs in an executor thread.
    """

    def __init__(self) -> None:
        # Validate that pyannote.audio is importable at construction time.
        # This allows ``create_diarizer()`` to catch ImportError early.
        import pyannote.audio  # type: ignore[import-not-found]  # noqa: F401

        self._pipeline: Any = None
        self._loaded: bool = False

    async def load(self) -> None:
        """Load the pyannote speaker-diarization pipeline."""
        if self._loaded:
            return
        loop = asyncio.get_running_loop()
        self._pipeline = await loop.run_in_executor(None, self._load_pipeline)
        self._loaded = True
        logger.info("pyannote_pipeline_loaded")

    def _load_pipeline(self) -> Any:
        """Synchronous model loading (runs in executor)."""
        from pyannote.audio import Pipeline

        return Pipeline.from_pretrained(
            "pyannote/speaker-diarization-3.1",
        )

    async def diarize(
        self,
        audio: bytes,
        sample_rate: int,
        *,
        max_speakers: int | None = None,
    ) -> tuple[SpeakerSegment, ...]:
        """Run speaker diarization on PCM audio.

        Converts raw PCM bytes to a torch tensor, runs the pyannote
        pipeline in an executor thread, and returns SpeakerSegment tuples.
        """
        if not audio:
            return ()

        if not self._loaded:
            await self.load()

        loop = asyncio.get_running_loop()
        segments = await loop.run_in_executor(
            None,
            self._run_pipeline,
            audio,
            sample_rate,
            max_speakers,
        )
        return segments

    def _run_pipeline(
        self,
        audio: bytes,
        sample_rate: int,
        max_speakers: int | None,
    ) -> tuple[SpeakerSegment, ...]:
        """Synchronous pipeline execution (runs in executor)."""
        import torch

        # Convert 16-bit PCM bytes to float32 torch tensor
        num_samples = len(audio) // _PCM_SAMPLE_WIDTH
        samples = struct.unpack(f"<{num_samples}h", audio[: num_samples * _PCM_SAMPLE_WIDTH])
        waveform = torch.tensor(samples, dtype=torch.float32).unsqueeze(0) / 32768.0

        # pyannote expects {"waveform": tensor, "sample_rate": int}
        audio_input: dict[str, Any] = {
            "waveform": waveform,
            "sample_rate": sample_rate,
        }

        kwargs: dict[str, Any] = {}
        if max_speakers is not None and max_speakers > 0:
            kwargs["max_speakers"] = max_speakers

        diarization = self._pipeline(audio_input, **kwargs)

        # Convert pyannote Annotation to SpeakerSegment tuples
        result: list[SpeakerSegment] = []
        for turn, _, speaker_label in diarization.itertracks(yield_label=True):
            result.append(
                SpeakerSegment(
                    speaker_id=str(speaker_label),
                    start=turn.start,
                    end=turn.end,
                    text="",  # Diarization provides timing, not text
                )
            )

        logger.info(
            "diarization_complete",
            num_speakers=len({s.speaker_id for s in result}),
            num_segments=len(result),
        )
        return tuple(result)

    async def health(self) -> dict[str, str]:
        """Return diarization backend health status."""
        if self._loaded:
            return {"status": "ok", "backend": "pyannote"}
        return {"status": "not_loaded", "backend": "pyannote"}
