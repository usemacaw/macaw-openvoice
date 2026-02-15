"""STT backend for Faster-Whisper (CTranslate2).

Implements STTBackend using faster-whisper as the inference library.
Faster-whisper is an optional dependency — the import is guarded.
"""

from __future__ import annotations

import asyncio
from typing import TYPE_CHECKING

import numpy as np

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

if TYPE_CHECKING:
    from collections.abc import AsyncIterator

try:
    from faster_whisper import WhisperModel
except ImportError:
    WhisperModel = None

logger = get_logger("worker.stt.faster_whisper")


class FasterWhisperBackend(STTBackend):
    """STT backend using Faster-Whisper (CTranslate2).

    Architecture: encoder-decoder. Streaming via LocalAgreement (M5).
    """

    def __init__(self) -> None:
        self._model: object | None = None
        self._beam_size: int = 5
        self._accumulation_threshold_seconds: float = 5.0

    @property
    def architecture(self) -> STTArchitecture:
        return STTArchitecture.ENCODER_DECODER

    async def load(self, model_path: str, config: dict[str, object]) -> None:
        if WhisperModel is None:
            msg = "faster-whisper nao esta instalado. Instale com: pip install macaw-openvoice[faster-whisper]"
            raise ModelLoadError(model_path, msg)

        model_size = str(config.get("model_size", model_path))
        compute_type = str(config.get("compute_type", "auto"))
        device = str(config.get("device", "auto"))
        self._beam_size = int(config.get("beam_size", 5))  # type: ignore[call-overload]
        self._accumulation_threshold_seconds = float(
            config.get("accumulation_threshold_seconds", 5.0)  # type: ignore[arg-type]
        )

        loop = asyncio.get_running_loop()
        try:
            self._model = await loop.run_in_executor(
                None,
                lambda: WhisperModel(
                    model_size,
                    device=device,
                    compute_type=compute_type,
                ),
            )
        except Exception as exc:
            msg = str(exc)
            raise ModelLoadError(model_path, msg) from exc

        logger.info(
            "model_loaded",
            model_size=model_size,
            compute_type=compute_type,
            device=device,
            accumulation_threshold_seconds=self._accumulation_threshold_seconds,
        )

    async def capabilities(self) -> EngineCapabilities:
        return EngineCapabilities(
            supports_hot_words=False,
            supports_initial_prompt=True,
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
            msg = "Modelo nao carregado. Chame load() primeiro."
            raise ModelLoadError("unknown", msg)

        if not audio_data:
            msg = "Audio vazio"
            raise AudioFormatError(msg)

        audio_array = pcm_bytes_to_float32(audio_data)

        fw_language: str | None = language
        if fw_language in ("auto", "mixed"):
            fw_language = None

        # Build initial_prompt with hot words if provided
        effective_prompt = initial_prompt
        if hot_words:
            hot_words_text = f"Terms: {', '.join(hot_words)}."
            effective_prompt = (
                f"{effective_prompt} {hot_words_text}" if effective_prompt else hot_words_text
            )

        loop = asyncio.get_running_loop()
        segments_iter, info = await loop.run_in_executor(
            None,
            lambda: self._model.transcribe(  # type: ignore[union-attr]
                audio_array,
                language=fw_language,
                initial_prompt=effective_prompt,
                temperature=temperature,
                word_timestamps=word_timestamps,
                beam_size=self._beam_size,
                vad_filter=False,
            ),
        )

        # Materialize segments (generator -> list)
        fw_segments = await loop.run_in_executor(None, list, segments_iter)

        segment_details = tuple(
            _fw_segment_to_detail(seg, idx) for idx, seg in enumerate(fw_segments)
        )

        full_text = " ".join(seg.text for seg in segment_details).strip()
        words = _extract_words(fw_segments, word_timestamps)

        return BatchResult(
            text=full_text,
            language=info.language,
            duration=info.duration,
            segments=segment_details,
            words=words,
        )

    async def transcribe_stream(  # type: ignore[override, misc]
        self,
        audio_chunks: AsyncIterator[bytes],
        language: str | None = None,
        initial_prompt: str | None = None,
        hot_words: list[str] | None = None,
    ) -> AsyncIterator[TranscriptSegment]:
        """Transcribe streaming audio via accumulation with a threshold.

        Accumulates 16-bit PCM 16kHz mono chunks in an internal buffer.
        When the buffer reaches the threshold (configurable via
        accumulation_threshold_seconds, default 5s), it runs inference on the
        accumulated buffer and yields TranscriptSegment(is_final=True).

        An empty chunk (b"") signals end of stream -- the remaining buffer is
        transcribed and emitted as the final segment.

        No LocalAgreement in this implementation -- each accumulated segment
        is emitted as final. LocalAgreement will be added in a future milestone.

        Args:
            audio_chunks: Async iterator of 16-bit PCM 16kHz mono chunks.
            language: ISO 639-1 code, "auto", or "mixed".
            initial_prompt: Context to guide transcription.
            hot_words: Words for keyword boosting via initial_prompt.

        Yields:
            TranscriptSegment with is_final=True for each transcribed buffer.
        """
        if self._model is None:
            msg = "Modelo nao carregado. Chame load() primeiro."
            raise ModelLoadError("unknown", msg)

        sample_rate = 16000
        threshold_samples = int(self._accumulation_threshold_seconds * sample_rate)

        buffer_chunks: list[np.ndarray] = []
        buffer_samples = 0
        segment_id = 0

        effective_prompt = initial_prompt
        if hot_words:
            hot_words_text = f"Terms: {', '.join(hot_words)}."
            effective_prompt = (
                f"{effective_prompt} {hot_words_text}" if effective_prompt else hot_words_text
            )

        fw_language: str | None = language
        if fw_language in ("auto", "mixed"):
            fw_language = None

        async for chunk in audio_chunks:
            if not chunk:
                break

            audio_array = pcm_bytes_to_float32(chunk)
            buffer_chunks.append(audio_array)
            buffer_samples += len(audio_array)

            if buffer_samples >= threshold_samples:
                accumulated = np.concatenate(buffer_chunks)
                segment = await self._transcribe_accumulated(
                    accumulated, fw_language, effective_prompt, segment_id
                )
                yield segment
                segment_id += 1
                buffer_chunks = []
                buffer_samples = 0

        if buffer_chunks:
            accumulated = np.concatenate(buffer_chunks)
            if len(accumulated) > 0:
                segment = await self._transcribe_accumulated(
                    accumulated, fw_language, effective_prompt, segment_id
                )
                yield segment

    async def _transcribe_accumulated(
        self,
        audio: np.ndarray,
        language: str | None,
        initial_prompt: str | None,
        segment_id: int,
    ) -> TranscriptSegment:
        """Transcribe accumulated audio and return a TranscriptSegment."""
        loop = asyncio.get_running_loop()
        segments_iter, info = await loop.run_in_executor(
            None,
            lambda: self._model.transcribe(  # type: ignore[union-attr]
                audio,
                language=language,
                initial_prompt=initial_prompt,
                temperature=0.0,
                beam_size=self._beam_size,
                vad_filter=False,
            ),
        )

        fw_segments = await loop.run_in_executor(None, list, segments_iter)

        full_text = " ".join(seg.text.strip() for seg in fw_segments).strip()

        start_ms: int | None = None
        end_ms: int | None = None
        avg_confidence: float | None = None

        if fw_segments:
            start_ms = int(fw_segments[0].start * 1000)
            end_ms = int(fw_segments[-1].end * 1000)
            logprobs = [seg.avg_logprob for seg in fw_segments]
            avg_confidence = sum(logprobs) / len(logprobs)

        return TranscriptSegment(
            text=full_text,
            is_final=True,
            segment_id=segment_id,
            start_ms=start_ms,
            end_ms=end_ms,
            language=info.language,
            confidence=avg_confidence,
        )

    async def unload(self) -> None:
        self._model = None
        logger.info("model_unloaded")

    async def health(self) -> dict[str, str]:
        if self._model is not None:
            return {"status": "ok"}
        return {"status": "not_loaded"}


def _fw_segment_to_detail(segment: object, index: int) -> SegmentDetail:
    """Convert a faster-whisper segment to Macaw SegmentDetail."""
    return SegmentDetail(
        id=index,
        start=segment.start,  # type: ignore[attr-defined]
        end=segment.end,  # type: ignore[attr-defined]
        text=segment.text.strip(),  # type: ignore[attr-defined]
        avg_logprob=segment.avg_logprob,  # type: ignore[attr-defined]
        no_speech_prob=segment.no_speech_prob,  # type: ignore[attr-defined]
        compression_ratio=segment.compression_ratio,  # type: ignore[attr-defined]
    )


def _fw_word_to_timestamp(word: object) -> WordTimestamp:
    """Convert a faster-whisper word to Macaw WordTimestamp."""
    return WordTimestamp(
        word=word.word.strip(),  # type: ignore[attr-defined]
        start=word.start,  # type: ignore[attr-defined]
        end=word.end,  # type: ignore[attr-defined]
        probability=word.probability,  # type: ignore[attr-defined]
    )


def _extract_words(
    fw_segments: list[object],
    word_timestamps: bool,
) -> tuple[WordTimestamp, ...] | None:
    """Extract all words from all faster-whisper segments."""
    if not word_timestamps:
        return None

    all_words: list[WordTimestamp] = []
    for seg in fw_segments:
        seg_words = getattr(seg, "words", None)
        if seg_words:
            all_words.extend(_fw_word_to_timestamp(w) for w in seg_words)

    return tuple(all_words) if all_words else None
