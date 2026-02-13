"""STT backend for WeNet (CTC).

Implements STTBackend using WeNet as the inference library.
WeNet is an optional dependency — the import is guarded.

WeNet uses CTC architecture, producing frame-by-frame output with native
partials. Unlike Faster-Whisper (encoder-decoder), it does not require
LocalAgreement for partial transcripts.
"""

from __future__ import annotations

import asyncio
import contextlib
import os
import struct
import threading
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
            msg = "wenet nao esta instalado. Instale com: pip install macaw-openvoice[wenet]"
            raise ModelLoadError(model_path, msg)

        language = str(config.get("language", "chinese"))
        device_str = str(config.get("device", "cpu"))
        compute_type = str(config.get("compute_type", "auto"))

        # Map device config to WeNet device format
        device = _resolve_device(device_str)

        # Resolve compute dtype based on device
        resolved_dtype = _resolve_compute_dtype(compute_type, device)

        loop = asyncio.get_running_loop()
        try:
            self._model = await loop.run_in_executor(
                None,
                lambda: _load_and_configure_model(model_path, device, resolved_dtype),
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
            compute_type=resolved_dtype,
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
            msg = "Modelo nao carregado. Chame load() primeiro."
            raise ModelLoadError("unknown", msg)

        if not audio_data:
            msg = "Audio vazio"
            raise AudioFormatError(msg)

        audio_array = pcm_bytes_to_float32(audio_data)
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
            msg = "Modelo nao carregado. Chame load() primeiro."
            raise ModelLoadError("unknown", msg)

        # CTC streaming: process each chunk immediately for lowest TTFB.
        # Minimum 160ms (2560 samples) to avoid degenerate tiny transcriptions.
        min_samples_for_partial = int(0.16 * _SAMPLE_RATE)

        # Segment boundary prevents O(n²) re-transcription growth.
        # Without it, buffer_chunks grows unbounded and np.concatenate +
        # transcription cost increases linearly per chunk (total = O(n²)).
        segment_threshold_samples = int(5.0 * _SAMPLE_RATE)

        buffer_chunks: list[np.ndarray] = []
        buffer_samples = 0
        segment_id = 0
        total_samples = 0
        segment_offset_samples = 0
        last_partial_text = ""

        resolved_lang = language if language and language not in ("auto", "mixed") else None

        async for chunk in audio_chunks:
            if not chunk:
                break

            audio_array = pcm_bytes_to_float32(chunk)
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

                # Segment boundary: emit final, clear buffer to bound work
                if buffer_samples >= segment_threshold_samples:
                    if text:
                        start_ms = int(segment_offset_samples / _SAMPLE_RATE * 1000)
                        end_ms = int(total_samples / _SAMPLE_RATE * 1000)
                        yield TranscriptSegment(
                            text=text,
                            is_final=True,
                            segment_id=segment_id,
                            start_ms=start_ms,
                            end_ms=end_ms,
                            language=resolved_lang,
                        )
                    buffer_chunks = []
                    buffer_samples = 0
                    segment_id += 1
                    segment_offset_samples = total_samples
                    last_partial_text = ""
                elif text and text != last_partial_text:
                    start_ms = int(segment_offset_samples / _SAMPLE_RATE * 1000)
                    yield TranscriptSegment(
                        text=text,
                        is_final=False,
                        segment_id=segment_id,
                        start_ms=start_ms,
                        language=resolved_lang,
                    )
                    last_partial_text = text

        # Final: transcribe remaining audio in buffer
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
                    start_ms = int(segment_offset_samples / _SAMPLE_RATE * 1000)
                    end_ms = int(total_samples / _SAMPLE_RATE * 1000)
                    yield TranscriptSegment(
                        text=text,
                        is_final=True,
                        segment_id=segment_id,
                        start_ms=start_ms,
                        end_ms=end_ms,
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


def _resolve_compute_dtype(compute_type: str, device: str) -> str:
    """Resolve compute dtype string based on device.

    Args:
        compute_type: "auto", "float16", "bfloat16", or "float32".
        device: Resolved device string ("cpu", "cuda", "cuda:0").

    Returns:
        Resolved dtype string: "float16", "bfloat16", or "float32".
    """
    is_cuda = device.startswith("cuda")

    if compute_type == "auto":
        return "float16" if is_cuda else "float32"

    # float16 on CPU is impractical (extremely slow), fall back to float32
    if compute_type == "float16" and not is_cuda:
        logger.warning(
            "compute_type_fallback",
            requested="float16",
            resolved="float32",
            reason="float16 not supported on CPU, falling back to float32",
        )
        return "float32"

    if compute_type in ("float16", "bfloat16", "float32"):
        return compute_type

    logger.warning(
        "compute_type_unknown",
        requested=compute_type,
        resolved="float32",
    )
    return "float32"


def _load_and_configure_model(
    model_path: str,
    device: str,
    compute_dtype: str,
) -> object:
    """Load WeNet model and apply compute dtype configuration.

    Args:
        model_path: Path to the model files.
        device: Device string ("cpu", "cuda", "cuda:0").
        compute_dtype: Resolved dtype ("float16", "bfloat16", "float32").

    Returns:
        Loaded and configured WeNet model.
    """
    model = wenet_lib.load_model(model_path, device=device)  # type: ignore[union-attr]

    if compute_dtype == "float16":
        model.half()  # type: ignore[union-attr]
    elif compute_dtype == "bfloat16":
        try:
            import torch

            model.to(torch.bfloat16)  # type: ignore[union-attr]
        except ImportError:
            logger.warning(
                "bfloat16_fallback",
                reason="torch not available for bfloat16 conversion",
            )

    return model


# Base directory for reusable WAV scratch files. /dev/shm is tmpfs
# (RAM-backed) on Linux, avoiding disk I/O entirely.  Falls back to
# the OS default temp directory when /dev/shm is unavailable.
_SHM_DIR = "/dev/shm" if os.path.isdir("/dev/shm") else None

# WAV header constants (mono 16-bit PCM at _SAMPLE_RATE)
_WAV_HEADER_SIZE = 44
_NUM_CHANNELS = 1
_BYTES_PER_SAMPLE = 2


def _build_wav_header(num_samples: int) -> bytes:
    """Build a 44-byte WAV header for mono 16-bit PCM at _SAMPLE_RATE.

    Uses struct.pack for a single-shot header construction — avoids the
    overhead of Python's wave module (multiple seeks and writes).

    Args:
        num_samples: Number of audio samples.

    Returns:
        44-byte WAV file header.
    """
    data_size = num_samples * _BYTES_PER_SAMPLE
    file_size = _WAV_HEADER_SIZE - 8 + data_size
    byte_rate = _SAMPLE_RATE * _NUM_CHANNELS * _BYTES_PER_SAMPLE
    block_align = _NUM_CHANNELS * _BYTES_PER_SAMPLE

    return struct.pack(
        "<4sI4s4sIHHIIHH4sI",
        b"RIFF",
        file_size,
        b"WAVE",
        b"fmt ",
        16,  # PCM format chunk size
        1,  # PCM format tag
        _NUM_CHANNELS,
        _SAMPLE_RATE,
        byte_rate,
        block_align,
        _BYTES_PER_SAMPLE * 8,  # bits per sample
        b"data",
        data_size,
    )


def _audio_to_wav_bytes(audio_array: np.ndarray) -> bytes:
    """Convert float32 audio array to WAV bytes.

    Uses struct.pack for the header instead of the wave module,
    avoiding multiple seeks and writes.

    Args:
        audio_array: Normalized float32 audio in [-1, 1].

    Returns:
        Complete WAV file content as bytes.
    """
    int16_data = (audio_array * 32768.0).clip(-32768, 32767).astype(np.int16)
    header = _build_wav_header(len(int16_data))
    return header + int16_data.tobytes()


def _get_scratch_path() -> str:
    """Return a per-thread reusable scratch file path on /dev/shm.

    Using a fixed path per thread avoids the overhead of creating and
    deleting NamedTemporaryFile on every transcription call (~5 syscalls
    saved per call).  Thread safety is guaranteed because each thread
    gets its own file path.

    Returns:
        Absolute path to the scratch WAV file.
    """
    tid = threading.get_ident()
    base_dir = _SHM_DIR or os.path.join(os.environ.get("TMPDIR", "/tmp"))
    return os.path.join(base_dir, f"macaw_wenet_{os.getpid()}_{tid}.wav")


def _transcribe_with_model(
    model: object,
    audio_array: np.ndarray,
    hot_words: list[str] | None = None,
) -> dict[str, object]:
    """Transcribe audio using the WeNet model.

    Writes a minimal WAV file to a reusable per-thread scratch path on
    /dev/shm (RAM-backed tmpfs) to satisfy WeNet's file-path API with
    near-zero I/O latency.  Runs under torch.inference_mode() to
    eliminate autograd overhead.

    Args:
        model: Loaded WeNet model.
        audio_array: Normalized float32 audio in [-1, 1].
        hot_words: List of hot words for context biasing.

    Returns:
        Dict with transcription result.
    """
    try:
        import torch

        inference_ctx = torch.inference_mode()
    except ImportError:
        from contextlib import nullcontext

        inference_ctx = nullcontext()  # type: ignore[assignment]

    with inference_ctx:
        result = _write_and_transcribe(model, audio_array)

    # Normalize result to dict
    if isinstance(result, dict):
        return result
    if isinstance(result, str):
        return {"text": result}
    if hasattr(result, "text"):
        return {"text": result.text}
    return {"text": str(result)}


def _write_and_transcribe(model: object, audio_array: np.ndarray) -> object:
    """Write WAV to a reusable scratch file and call model.transcribe().

    Uses a per-thread fixed path on /dev/shm to avoid NamedTemporaryFile
    creation/deletion overhead.  The file is overwritten on each call
    (2 syscalls: open + write) instead of created and unlinked each time
    (5+ syscalls).

    Args:
        model: Loaded WeNet model.
        audio_array: Normalized float32 audio in [-1, 1].

    Returns:
        Raw transcription result from the model.
    """
    wav_bytes = _audio_to_wav_bytes(audio_array)
    scratch_path = _get_scratch_path()

    with open(scratch_path, "wb") as f:
        f.write(wav_bytes)

    try:
        return model.transcribe(scratch_path)  # type: ignore[attr-defined]
    finally:
        # Best-effort cleanup — file will be overwritten on next call anyway
        with contextlib.suppress(OSError):
            os.unlink(scratch_path)


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
