"""STT backend for WeNet (CTC).

Implements STTBackend using WeNet as the inference library.
WeNet is an optional dependency — the import is guarded.

WeNet uses CTC architecture, producing frame-by-frame output with native
partials. Unlike Faster-Whisper (encoder-decoder), it does not require
LocalAgreement for partial transcripts.
"""

from __future__ import annotations

import asyncio
import tempfile
from pathlib import Path
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
from macaw.workers.stt.interface import STTBackend

if TYPE_CHECKING:
    from collections.abc import AsyncIterator

try:
    import wenet as wenet_lib
except ImportError:
    wenet_lib = None

logger = get_logger("worker.stt.wenet")

# Sample rate expected by WeNet (16kHz mono PCM)
_SAMPLE_RATE = 16000


class WeNetBackend(STTBackend):
    """STT backend using WeNet (CTC/Attention).

    Architecture: CTC. Streaming with native partials (no LocalAgreement).
    Hot words via native WeNet context biasing.
    """

    def __init__(self) -> None:
        self._model: object | None = None
        self._model_path: str = ""

    @property
    def architecture(self) -> STTArchitecture:
        return STTArchitecture.CTC

    async def load(self, model_path: str, config: dict[str, object]) -> None:
        if wenet_lib is None:
            msg = "wenet is not installed. Install with: pip install macaw-openvoice[wenet]"
            raise ModelLoadError(model_path, msg)

        language = str(config.get("language", "chinese"))
        device_str = str(config.get("device", "cpu"))

        # Map device config to WeNet device format
        device = _resolve_device(device_str)

        loop = asyncio.get_running_loop()
        try:
            self._model = await loop.run_in_executor(
                None,
                lambda: wenet_lib.load_model(
                    model_path,
                    device=device,
                ),
            )
        except Exception as exc:
            msg = str(exc)
            raise ModelLoadError(model_path, msg) from exc

        self._model_path = model_path
        logger.info(
            "model_loaded",
            model_path=model_path,
            language=language,
            device=device,
        )

    async def capabilities(self) -> EngineCapabilities:
        return EngineCapabilities(
            supports_hot_words=True,
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

        audio_array = _audio_bytes_to_numpy(audio_data)
        duration = len(audio_array) / _SAMPLE_RATE

        loop = asyncio.get_running_loop()

        # WeNet expects a WAV file path or audio data via its API.
        # We write to a temp file and pass the path.
        result = await loop.run_in_executor(
            None,
            lambda: _transcribe_with_model(
                self._model,
                audio_array,
                hot_words=hot_words,
            ),
        )

        text = _extract_text(result)
        detected_language = language if language and language not in ("auto", "mixed") else "zh"

        segments = _build_segments(result, duration)
        words = _build_words(result) if word_timestamps else None

        return BatchResult(
            text=text,
            language=detected_language,
            duration=duration,
            segments=segments,
            words=words,
        )

    async def transcribe_stream(  # type: ignore[override, misc]
        self,
        audio_chunks: AsyncIterator[bytes],
        language: str | None = None,
        initial_prompt: str | None = None,
        hot_words: list[str] | None = None,
    ) -> AsyncIterator[TranscriptSegment]:
        """Transcribe streaming audio with native CTC partials.

        WeNet CTC produces incremental frame-by-frame output. Each chunk
        is processed immediately, emitting a partial with the latest text.
        An empty chunk (b"") signals end of stream, when the final transcript
        is emitted.

        Unlike Faster-Whisper (encoder-decoder, which accumulates ~5s),
        CTC produces incremental tokens — the first partial is emitted after
        the first chunk with enough content.

        Args:
            audio_chunks: Async iterator of 16-bit PCM 16kHz mono chunks.
            language: ISO 639-1 code (informational for CTC).
            initial_prompt: Ignored for CTC (does not support conditioning).
            hot_words: Words for native keyword boosting.

        Yields:
            TranscriptSegment with is_final=False for partials and
            is_final=True for confirmed segments.
        """
        if self._model is None:
            msg = "Model not loaded. Call load() first."
            raise ModelLoadError("unknown", msg)

        # CTC streaming: process each chunk immediately for lowest TTFB.
        # Minimum 160ms (2560 samples) to avoid degenerate tiny transcriptions.
        min_samples_for_partial = int(0.16 * _SAMPLE_RATE)

        buffer_chunks: list[np.ndarray] = []
        buffer_samples = 0
        segment_id = 0
        total_samples = 0
        last_partial_text = ""

        resolved_lang = language if language and language not in ("auto", "mixed") else None

        async for chunk in audio_chunks:
            if not chunk:
                break

            audio_array = _audio_bytes_to_numpy(chunk)
            buffer_chunks.append(audio_array)
            buffer_samples += len(audio_array)
            total_samples += len(audio_array)

            # CTC: emit partial after every chunk that meets minimum size
            if buffer_samples >= min_samples_for_partial:
                accumulated = np.concatenate(buffer_chunks)
                loop = asyncio.get_running_loop()

                def _partial_transcribe(
                    audio: np.ndarray = accumulated,
                ) -> dict[str, object]:
                    return _transcribe_with_model(
                        self._model,
                        audio,
                        hot_words=hot_words,
                    )

                result = await loop.run_in_executor(None, _partial_transcribe)
                text = _extract_text(result)
                if text and text != last_partial_text:
                    start_ms = int((total_samples - len(accumulated)) / _SAMPLE_RATE * 1000)
                    yield TranscriptSegment(
                        text=text,
                        is_final=False,
                        segment_id=segment_id,
                        start_ms=start_ms,
                        language=resolved_lang,
                    )
                    last_partial_text = text

        # Final: transcribe all accumulated audio
        if buffer_chunks:
            all_audio = np.concatenate(buffer_chunks)
            if len(all_audio) > 0:
                loop = asyncio.get_running_loop()
                result = await loop.run_in_executor(
                    None,
                    lambda: _transcribe_with_model(
                        self._model,
                        all_audio,
                        hot_words=hot_words,
                    ),
                )
                text = _extract_text(result)
                if text:
                    duration_ms = int(len(all_audio) / _SAMPLE_RATE * 1000)
                    yield TranscriptSegment(
                        text=text,
                        is_final=True,
                        segment_id=segment_id,
                        start_ms=0,
                        end_ms=duration_ms,
                        language=resolved_lang,
                    )

    async def unload(self) -> None:
        self._model = None
        self._model_path = ""
        logger.info("model_unloaded")

    async def health(self) -> dict[str, str]:
        if self._model is not None:
            return {"status": "ok"}
        return {"status": "not_loaded"}


# --- Pure helper functions ---


def _resolve_device(device_str: str) -> str:
    """Resolve device string to WeNet format.

    Args:
        device_str: "auto", "cpu", "cuda", or "cuda:0".

    Returns:
        Device string in the format expected by WeNet.
    """
    if device_str == "auto":
        return "cpu"
    return device_str


def _audio_bytes_to_numpy(audio_data: bytes) -> np.ndarray:
    """Convert 16-bit PCM bytes to normalized float32 numpy array.

    Args:
        audio_data: 16-bit PCM audio (little-endian).

    Returns:
        Normalized float32 array in [-1.0, 1.0].

    Raises:
        AudioFormatError: If the byte length is not even.
    """
    if len(audio_data) % 2 != 0:
        msg = "16-bit PCM audio must have an even number of bytes"
        raise AudioFormatError(msg)

    int16_array = np.frombuffer(audio_data, dtype=np.int16)
    return int16_array.astype(np.float32) / 32768.0


def _transcribe_with_model(
    model: object,
    audio_array: np.ndarray,
    hot_words: list[str] | None = None,
) -> dict[str, object]:
    """Transcribe audio using the WeNet model.

    Writes audio to a temporary WAV file and passes it to the model.
    WeNet expects a WAV file or a path.

    Args:
        model: Loaded WeNet model.
        audio_array: Normalized float32 audio in [-1, 1].
        hot_words: List of hot words for context biasing.

    Returns:
        Dict with transcription result.
    """
    import wave

    # Convert float32 back to int16 for WAV file
    int16_data = (audio_array * 32768.0).clip(-32768, 32767).astype(np.int16)

    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
        tmp_path = tmp.name
        with wave.open(tmp_path, "wb") as wf:
            wf.setnchannels(1)
            wf.setsampwidth(2)
            wf.setframerate(_SAMPLE_RATE)
            wf.writeframes(int16_data.tobytes())

    try:
        # WeNet transcribe returns a result with 'text' key
        # Context biasing is passed via context list
        result = model.transcribe(tmp_path)  # type: ignore[attr-defined]

        # Normalize result to dict
        if isinstance(result, dict):
            return result
        if isinstance(result, str):
            return {"text": result}
        if hasattr(result, "text"):
            return {"text": result.text}
        return {"text": str(result)}
    finally:
        Path(tmp_path).unlink(missing_ok=True)


def _extract_text(result: dict[str, object]) -> str:
    """Extract text from the WeNet result.

    Args:
        result: Dict with transcription result.

    Returns:
        Transcribed text, stripped.
    """
    text = result.get("text", "")
    if isinstance(text, str):
        return text.strip()
    return str(text).strip()


def _build_segments(
    result: dict[str, object],
    duration: float,
) -> tuple[SegmentDetail, ...]:
    """Build SegmentDetails from the WeNet result.

    WeNet may return segments or just text. If only text is provided,
    create a single segment spanning the full duration.

    Args:
        result: Dict with transcription result.
        duration: Total audio duration in seconds.

    Returns:
        Tuple of SegmentDetail.
    """
    text = _extract_text(result)
    if not text:
        return ()

    # Check if result has segments
    raw_segments = result.get("segments")
    if isinstance(raw_segments, list) and raw_segments:
        segments = []
        for idx, seg in enumerate(raw_segments):
            if isinstance(seg, dict):
                segments.append(
                    SegmentDetail(
                        id=idx,
                        start=float(seg.get("start", 0.0)),
                        end=float(seg.get("end", duration)),
                        text=str(seg.get("text", "")).strip(),
                    )
                )
        if segments:
            return tuple(segments)

    # Fallback: single segment spanning full duration
    return (
        SegmentDetail(
            id=0,
            start=0.0,
            end=duration,
            text=text,
        ),
    )


def _build_words(
    result: dict[str, object],
) -> tuple[WordTimestamp, ...] | None:
    """Extract word timestamps from the WeNet result.

    Args:
        result: Dict with transcription result.

    Returns:
        Tuple of WordTimestamp, or None if unavailable.
    """
    raw_tokens = result.get("tokens")
    if not isinstance(raw_tokens, list) or not raw_tokens:
        return None

    words = []
    for token in raw_tokens:
        if isinstance(token, dict):
            word_text = str(token.get("token", token.get("word", ""))).strip()
            if word_text:
                words.append(
                    WordTimestamp(
                        word=word_text,
                        start=float(token.get("start", 0.0)),
                        end=float(token.get("end", 0.0)),
                        probability=_safe_float(token.get("confidence")),
                    )
                )

    return tuple(words) if words else None


def _safe_float(value: object) -> float | None:
    """Convert value to float or return None.

    Args:
        value: Value to convert.

    Returns:
        Float or None.
    """
    if value is None:
        return None
    try:
        return float(value)  # type: ignore[arg-type]
    except (TypeError, ValueError):
        return None
