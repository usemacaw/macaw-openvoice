"""STT backend for Qwen3-ASR.

Implements STTBackend using Qwen3-ASR as the inference library.
qwen-asr is an optional dependency — the import is guarded.

Qwen3-ASR API (qwen_asr):
  model = Qwen3ASRModel.from_pretrained("Qwen/Qwen3-ASR-0.6B",
      dtype=torch.bfloat16, device_map="cuda:0")
  result = model.transcribe(audio=(np_array, sr), language=None)
  result.text, result.language, result.time_stamps
"""

from __future__ import annotations

import asyncio
from typing import TYPE_CHECKING

import numpy as np

from macaw._audio_constants import STT_SAMPLE_RATE
from macaw._types import (
    BatchResult,
    EngineCapabilities,
    SegmentDetail,
    STTArchitecture,
    TranscriptSegment,
    WordTimestamp,
)
from macaw.exceptions import AudioFormatError, ModelLoadError
from macaw.logging import get_logger
from macaw.workers.audio_utils import pcm_bytes_to_float32
from macaw.workers.stt.interface import STTBackend
from macaw.workers.torch_utils import release_gpu_memory, resolve_device

if TYPE_CHECKING:
    from collections.abc import AsyncIterator

try:
    from qwen_asr import Qwen3ASRModel as _Qwen3ASRModel
except ImportError:
    _Qwen3ASRModel = None

logger = get_logger("worker.stt.qwen3_asr")

_DEFAULT_ACCUMULATION_THRESHOLD_S = 5.0


def _get_torch_dtype(dtype_str: str) -> object:
    """Convert dtype string to torch dtype object."""
    import torch

    dtype_map: dict[str, object] = {
        "bfloat16": torch.bfloat16,
        "float16": torch.float16,
        "float32": torch.float32,
    }
    return dtype_map.get(dtype_str, torch.bfloat16)


class Qwen3ASRBackend(STTBackend):
    """STT backend using Qwen3-ASR (encoder-decoder ASR).

    Architecture: encoder-decoder. Streaming via LocalAgreement (runtime).
    Supports 52 languages with auto-detection and word timestamps.
    """

    def __init__(self) -> None:
        self._model: object | None = None
        self._model_path: str = ""
        self._accumulation_threshold_seconds: float = _DEFAULT_ACCUMULATION_THRESHOLD_S

    @property
    def architecture(self) -> STTArchitecture:
        return STTArchitecture.ENCODER_DECODER

    async def load(self, model_path: str, config: dict[str, object]) -> None:
        if _Qwen3ASRModel is None:
            msg = "qwen-asr is not installed. Install with: pip install macaw-openvoice[qwen3-asr]"
            raise ModelLoadError(model_path, msg)

        device_str = str(config.get("device", "auto"))
        dtype_str = str(config.get("dtype", "bfloat16"))
        device = resolve_device(device_str)

        # Three-tier resolution: engine_config (manifest) > env var (settings) > constant.
        accumulation_default = _DEFAULT_ACCUMULATION_THRESHOLD_S
        try:
            from macaw.config.settings import get_settings

            accumulation_default = get_settings().worker.stt_accumulation_threshold_s
        except Exception:
            logger.debug("settings unavailable, using default accumulation threshold")
        self._accumulation_threshold_seconds = float(
            config.get("accumulation_threshold_seconds", accumulation_default)  # type: ignore[arg-type]
        )

        loop = asyncio.get_running_loop()
        try:
            self._model = await loop.run_in_executor(
                None,
                lambda: _load_qwen3_asr_model(model_path, device, dtype_str),
            )
        except ModelLoadError:
            raise
        except Exception as exc:
            msg = str(exc)
            raise ModelLoadError(model_path, msg) from exc

        self._model_path = model_path
        logger.info(
            "model_loaded",
            model_path=model_path,
            device=device,
            dtype=dtype_str,
        )

    async def capabilities(self) -> EngineCapabilities:
        return EngineCapabilities(
            supports_hot_words=False,
            supports_initial_prompt=False,
            supports_batch=True,
            supports_word_timestamps=True,
            max_concurrent_sessions=1,
        )

    async def transcribe_file(
        self,
        audio_data: bytes,
        language: str | None = None,
        initial_prompt: str | None = None,
        hot_words: list[str] | None = None,
        temperature: float = 0.0,
        word_timestamps: bool = False,
    ) -> BatchResult:
        if self._model is None:
            msg = "Model not loaded. Call load() first."
            raise ModelLoadError("unknown", msg)

        if not audio_data:
            msg = "Empty audio"
            raise AudioFormatError(msg)

        audio_array = pcm_bytes_to_float32(audio_data)

        # Qwen3-ASR uses None for auto-detection
        asr_language: str | None = language
        if asr_language in ("auto", "mixed"):
            asr_language = None

        loop = asyncio.get_running_loop()
        result = await loop.run_in_executor(
            None,
            lambda: _transcribe_with_model(
                self._model,
                audio_array,
                asr_language,
            ),
        )

        text = str(result.text).strip()  # type: ignore[attr-defined]
        detected_language = str(result.language)  # type: ignore[attr-defined]
        duration = float(len(audio_array)) / STT_SAMPLE_RATE

        # Build segments
        segments = (
            SegmentDetail(
                id=0,
                start=0.0,
                end=duration,
                text=text,
            ),
        )

        # Extract word timestamps if available and requested
        words = _extract_word_timestamps(result, word_timestamps)

        return BatchResult(
            text=text,
            language=detected_language,
            duration=duration,
            segments=segments,
            words=words,
        )

    # AsyncGenerator is a subtype of AsyncIterator but mypy doesn't recognize
    # yield-based overrides. See docs/ADDING_ENGINE.md.
    async def transcribe_stream(  # type: ignore[override, misc]
        self,
        audio_chunks: AsyncIterator[bytes],
        language: str | None = None,
        initial_prompt: str | None = None,
        hot_words: list[str] | None = None,
    ) -> AsyncIterator[TranscriptSegment]:
        """Transcribe streaming audio via accumulation with a threshold.

        Encoder-decoder architecture: accumulates PCM chunks until the
        threshold is reached, then runs batch inference and yields a
        final TranscriptSegment. Empty chunk signals end of stream.
        """
        if self._model is None:
            msg = "Model not loaded. Call load() first."
            raise ModelLoadError("unknown", msg)

        threshold_samples = int(self._accumulation_threshold_seconds * STT_SAMPLE_RATE)

        buffer_chunks: list[np.ndarray] = []
        buffer_samples = 0
        segment_id = 0

        asr_language: str | None = language
        if asr_language in ("auto", "mixed"):
            asr_language = None

        async for chunk in audio_chunks:
            if not chunk:
                break

            audio_array = pcm_bytes_to_float32(chunk)
            buffer_chunks.append(audio_array)
            buffer_samples += len(audio_array)

            if buffer_samples >= threshold_samples:
                accumulated = np.concatenate(buffer_chunks)
                segment = await self._transcribe_accumulated(accumulated, asr_language, segment_id)
                yield segment
                segment_id += 1
                buffer_chunks = []
                buffer_samples = 0

        if buffer_chunks:
            accumulated = np.concatenate(buffer_chunks)
            if len(accumulated) > 0:
                segment = await self._transcribe_accumulated(accumulated, asr_language, segment_id)
                yield segment

    async def _transcribe_accumulated(
        self,
        audio: np.ndarray,
        language: str | None,
        segment_id: int,
    ) -> TranscriptSegment:
        """Transcribe accumulated audio and return a TranscriptSegment."""
        loop = asyncio.get_running_loop()
        result = await loop.run_in_executor(
            None,
            lambda: _transcribe_with_model(self._model, audio, language),
        )

        text = str(result.text).strip()  # type: ignore[attr-defined]
        detected_language = str(result.language)  # type: ignore[attr-defined]
        duration_s = float(len(audio)) / STT_SAMPLE_RATE

        return TranscriptSegment(
            text=text,
            is_final=True,
            segment_id=segment_id,
            start_ms=0,
            end_ms=int(duration_s * 1000),
            language=detected_language,
        )

    async def unload(self) -> None:
        if self._model is not None:
            del self._model
            self._model = None
            release_gpu_memory()
        self._model_path = ""
        logger.info("model_unloaded")

    async def health(self) -> dict[str, str]:
        if self._model is not None:
            return {"status": "ok"}
        return {"status": "not_loaded"}


# --- Pure helper functions ---


def _load_qwen3_asr_model(
    model_path: str,
    device: str,
    dtype_str: str,
) -> object:
    """Load Qwen3-ASR model (blocking, runs in executor).

    Returns:
        Loaded model instance.
    """
    dtype = _get_torch_dtype(dtype_str)
    return _Qwen3ASRModel.from_pretrained(
        model_path,
        device_map=device,
        dtype=dtype,
    )


def _transcribe_with_model(
    model: object,
    audio_array: np.ndarray,
    language: str | None,
) -> object:
    """Run transcription on the model (blocking, runs in executor).

    Args:
        model: Loaded Qwen3ASRModel instance.
        audio_array: Float32 audio at 16kHz.
        language: Language code or None for auto-detect.

    Returns:
        Transcription result with .text, .language, .time_stamps attributes.
    """
    from macaw.workers.torch_utils import get_inference_context

    with get_inference_context():
        return model.transcribe(  # type: ignore[attr-defined]
            audio=(audio_array, STT_SAMPLE_RATE),
            language=language,
        )


def _extract_word_timestamps(
    result: object,
    word_timestamps: bool,
) -> tuple[WordTimestamp, ...] | None:
    """Extract word-level timestamps from Qwen3-ASR result.

    Qwen3-ASR provides .time_stamps as a list of (word, start, end) tuples.
    """
    if not word_timestamps:
        return None

    time_stamps = getattr(result, "time_stamps", None)
    if not time_stamps:
        return None

    words: list[WordTimestamp] = []
    for ts in time_stamps:
        # time_stamps entries are (word, start_seconds, end_seconds)
        if len(ts) >= 3:
            words.append(
                WordTimestamp(
                    word=str(ts[0]).strip(),
                    start=float(ts[1]),
                    end=float(ts[2]),
                )
            )

    return tuple(words) if words else None
