"""Tests for FasterWhisperBackend.transcribe_stream().

Validates chunk accumulation, inference threshold, flush at end
of stream, and hot words handling. Uses mocks for WhisperModel.
"""

from __future__ import annotations

from types import SimpleNamespace
from typing import TYPE_CHECKING
from unittest.mock import MagicMock

import numpy as np
import pytest

from macaw.exceptions import ModelLoadError
from macaw.workers.stt.faster_whisper import FasterWhisperBackend

if TYPE_CHECKING:
    from collections.abc import AsyncIterator

    from macaw._types import TranscriptSegment


def _make_fw_segment(
    text: str = "hello world",
    start: float = 0.0,
    end: float = 2.0,
    avg_logprob: float = -0.25,
    no_speech_prob: float = 0.01,
    compression_ratio: float = 1.1,
) -> SimpleNamespace:
    """Create fake segment in faster-whisper format."""
    return SimpleNamespace(
        text=f" {text}",
        start=start,
        end=end,
        avg_logprob=avg_logprob,
        no_speech_prob=no_speech_prob,
        compression_ratio=compression_ratio,
        words=None,
    )


def _make_fw_info(language: str = "en", duration: float = 2.0) -> SimpleNamespace:
    """Create fake TranscriptionInfo."""
    return SimpleNamespace(language=language, duration=duration)


def _make_pcm16_silence(duration_seconds: float, sample_rate: int = 16000) -> bytes:
    """Generate PCM 16-bit silence."""
    num_samples = int(duration_seconds * sample_rate)
    return np.zeros(num_samples, dtype=np.int16).tobytes()


async def _make_chunk_iterator(chunks: list[bytes]) -> AsyncIterator[bytes]:
    """Create AsyncIterator of chunks for tests."""
    for chunk in chunks:
        yield chunk


def _make_loaded_backend(
    text: str = "hello world",
    language: str = "en",
    duration: float = 2.0,
    avg_logprob: float = -0.25,
) -> FasterWhisperBackend:
    """Create backend with loaded mock model."""
    backend = FasterWhisperBackend()
    mock_model = MagicMock()
    segment = _make_fw_segment(text=text, end=duration, avg_logprob=avg_logprob)
    info = _make_fw_info(language=language, duration=duration)
    mock_model.transcribe.return_value = (iter([segment]), info)
    backend._model = mock_model  # type: ignore[assignment]
    return backend


class TestTranscribeStreamAccumulation:
    """Tests chunk accumulation and inference threshold."""

    async def test_accumulates_and_transcribes_on_threshold(self) -> None:
        """Sends 5s of audio (threshold), verifies that TranscriptSegment is yielded."""
        backend = _make_loaded_backend()

        # 5s de audio em chunks de 1s (atinge threshold de 5s)
        chunks = [_make_pcm16_silence(1.0) for _ in range(5)]
        # Chunk vazio para sinalizar fim
        chunks.append(b"")

        segments: list[TranscriptSegment] = []
        async for seg in backend.transcribe_stream(_make_chunk_iterator(chunks)):
            segments.append(seg)

        assert len(segments) == 1
        assert segments[0].text == "hello world"
        assert segments[0].is_final is True
        assert segments[0].segment_id == 0
        assert segments[0].language == "en"

    async def test_flushes_remaining_buffer_on_empty_chunk(self) -> None:
        """Sends short audio + empty chunk, verifies buffer flush."""
        backend = _make_loaded_backend()

        # 2s de audio (abaixo do threshold de 5s) + fim
        chunks = [_make_pcm16_silence(2.0), b""]

        segments: list[TranscriptSegment] = []
        async for seg in backend.transcribe_stream(_make_chunk_iterator(chunks)):
            segments.append(seg)

        assert len(segments) == 1
        assert segments[0].text == "hello world"
        assert segments[0].is_final is True
        assert segments[0].segment_id == 0

    async def test_short_audio_transcribed_on_stream_end(self) -> None:
        """Audio de 0.5s (curto) transcrito normalmente no fim do stream."""
        backend = _make_loaded_backend()

        # 500ms de audio + fim
        chunks = [_make_pcm16_silence(0.5), b""]

        segments: list[TranscriptSegment] = []
        async for seg in backend.transcribe_stream(_make_chunk_iterator(chunks)):
            segments.append(seg)

        assert len(segments) == 1
        assert segments[0].text == "hello world"


class TestTranscribeStreamMultipleSegments:
    """Tests multiple segments and segment_id increment."""

    async def test_multiple_segments_increment_id(self) -> None:
        """2x threshold gera 2 segmentos com ids 0 e 1."""
        backend = FasterWhisperBackend()
        mock_model = MagicMock()

        call_count = 0

        def transcribe_side_effect(*args: object, **kwargs: object) -> tuple[object, object]:
            nonlocal call_count
            text = f"segment {call_count}"
            seg = _make_fw_segment(text=text)
            info = _make_fw_info()
            call_count += 1
            return iter([seg]), info

        mock_model.transcribe.side_effect = transcribe_side_effect
        backend._model = mock_model  # type: ignore[assignment]

        # 10s de audio em chunks de 1s (2x threshold de 5s) + fim
        chunks = [_make_pcm16_silence(1.0) for _ in range(10)]
        chunks.append(b"")

        segments: list[TranscriptSegment] = []
        async for seg in backend.transcribe_stream(_make_chunk_iterator(chunks)):
            segments.append(seg)

        assert len(segments) == 2
        assert segments[0].segment_id == 0
        assert segments[0].text == "segment 0"
        assert segments[1].segment_id == 1
        assert segments[1].text == "segment 1"

    async def test_threshold_plus_remainder(self) -> None:
        """7s de audio: 1 segmento no threshold (5s) + 1 flush (2s)."""
        backend = FasterWhisperBackend()
        mock_model = MagicMock()

        call_count = 0

        def transcribe_side_effect(*args: object, **kwargs: object) -> tuple[object, object]:
            nonlocal call_count
            seg = _make_fw_segment(text=f"part {call_count}")
            info = _make_fw_info()
            call_count += 1
            return iter([seg]), info

        mock_model.transcribe.side_effect = transcribe_side_effect
        backend._model = mock_model  # type: ignore[assignment]

        # 7s de audio em chunks de 1s + fim
        chunks = [_make_pcm16_silence(1.0) for _ in range(7)]
        chunks.append(b"")

        segments: list[TranscriptSegment] = []
        async for seg in backend.transcribe_stream(_make_chunk_iterator(chunks)):
            segments.append(seg)

        assert len(segments) == 2
        assert segments[0].segment_id == 0
        assert segments[1].segment_id == 1


class TestTranscribeStreamPrompt:
    """Tests hot words and initial_prompt in streaming."""

    async def test_hot_words_in_prompt(self) -> None:
        """Hot words are injected into the transcription initial_prompt."""
        backend = _make_loaded_backend()

        chunks = [_make_pcm16_silence(2.0), b""]

        segments: list[TranscriptSegment] = []
        async for seg in backend.transcribe_stream(
            _make_chunk_iterator(chunks),
            hot_words=["PIX", "TED"],
            initial_prompt="Contexto bancario",
        ):
            segments.append(seg)

        assert len(segments) == 1
        call_kwargs = backend._model.transcribe.call_args  # type: ignore[union-attr]
        prompt = call_kwargs.kwargs.get("initial_prompt") or call_kwargs[1].get("initial_prompt")
        assert "PIX" in prompt
        assert "TED" in prompt
        assert "Contexto bancario" in prompt

    async def test_auto_language_passed_as_none(self) -> None:
        """Language 'auto' and 'mixed' are converted to None."""
        backend = _make_loaded_backend()

        chunks = [_make_pcm16_silence(2.0), b""]

        async for _ in backend.transcribe_stream(
            _make_chunk_iterator(chunks),
            language="auto",
        ):
            pass

        call_kwargs = backend._model.transcribe.call_args  # type: ignore[union-attr]
        assert call_kwargs.kwargs.get("language") is None


class TestTranscribeStreamErrors:
    """Tests error scenarios in streaming."""

    async def test_model_not_loaded_raises_error(self) -> None:
        """Without loaded model raises ModelLoadError."""
        backend = FasterWhisperBackend()

        with pytest.raises(ModelLoadError, match="not loaded"):
            async for _ in backend.transcribe_stream(
                _make_chunk_iterator([_make_pcm16_silence(1.0)]),
            ):
                pass

    async def test_empty_stream_yields_nothing(self) -> None:
        """Empty stream (only empty chunk) yields no segments."""
        backend = _make_loaded_backend()

        segments: list[TranscriptSegment] = []
        async for seg in backend.transcribe_stream(_make_chunk_iterator([b""])):
            segments.append(seg)

        assert len(segments) == 0


class TestAccumulationThresholdConfig:
    """Configurable accumulation_threshold_seconds in FasterWhisperBackend."""

    def test_default_threshold_is_five_seconds(self) -> None:
        backend = FasterWhisperBackend()
        assert backend._accumulation_threshold_seconds == 5.0

    async def test_custom_threshold_from_config(self) -> None:
        """Backend reads accumulation_threshold_seconds from engine config."""
        backend = FasterWhisperBackend()
        mock_model = MagicMock()
        segment = _make_fw_segment(text="test")
        info = _make_fw_info()
        mock_model.return_value = mock_model
        mock_model.transcribe.return_value = (iter([segment]), info)

        # Mock WhisperModel constructor
        import macaw.workers.stt.faster_whisper as fw_mod

        original = fw_mod.WhisperModel
        fw_mod.WhisperModel = MagicMock(return_value=mock_model)
        try:
            await backend.load(
                "/models/test",
                {"model_size": "tiny", "accumulation_threshold_seconds": 3.0},
            )
            assert backend._accumulation_threshold_seconds == 3.0
        finally:
            fw_mod.WhisperModel = original

    async def test_short_threshold_triggers_earlier(self) -> None:
        """With 2s threshold, 2s of audio triggers inference immediately."""
        backend = FasterWhisperBackend()
        backend._accumulation_threshold_seconds = 2.0
        mock_model = MagicMock()

        call_count = 0

        def transcribe_side_effect(*args: object, **kwargs: object) -> tuple[object, object]:
            nonlocal call_count
            seg = _make_fw_segment(text=f"seg {call_count}")
            info = _make_fw_info()
            call_count += 1
            return iter([seg]), info

        mock_model.transcribe.side_effect = transcribe_side_effect
        backend._model = mock_model  # type: ignore[assignment]

        # 4s of audio in 1s chunks with 2s threshold -> 2 segments + possibly remainder
        chunks = [_make_pcm16_silence(1.0) for _ in range(4)]
        chunks.append(b"")

        segments: list[TranscriptSegment] = []
        async for seg in backend.transcribe_stream(_make_chunk_iterator(chunks)):
            segments.append(seg)

        assert len(segments) == 2
        assert segments[0].segment_id == 0
        assert segments[1].segment_id == 1

    async def test_large_threshold_accumulates_more(self) -> None:
        """With 10s threshold, 6s of audio is flushed only at stream end."""
        backend = FasterWhisperBackend()
        backend._accumulation_threshold_seconds = 10.0
        backend._model = MagicMock()  # type: ignore[assignment]

        segment = _make_fw_segment(text="all at once")
        info = _make_fw_info()
        backend._model.transcribe.return_value = (iter([segment]), info)  # type: ignore[union-attr]

        # 6s of audio: does NOT reach 10s threshold, only flushed at end
        chunks = [_make_pcm16_silence(2.0) for _ in range(3)]
        chunks.append(b"")

        segments: list[TranscriptSegment] = []
        async for seg in backend.transcribe_stream(_make_chunk_iterator(chunks)):
            segments.append(seg)

        # Only 1 segment (flush at end), not 2
        assert len(segments) == 1
        assert segments[0].text == "all at once"


class TestTranscribeStreamTimestamps:
    """Tests timestamps and confidence in segments."""

    async def test_segment_has_timestamps(self) -> None:
        """Returned segment has computed start_ms and end_ms."""
        backend = _make_loaded_backend(duration=3.5)

        chunks = [_make_pcm16_silence(2.0), b""]

        segments: list[TranscriptSegment] = []
        async for seg in backend.transcribe_stream(_make_chunk_iterator(chunks)):
            segments.append(seg)

        assert len(segments) == 1
        assert segments[0].start_ms == 0
        assert segments[0].end_ms == 3500

    async def test_segment_has_confidence(self) -> None:
        """Returned segment has confidence (avg_logprob)."""
        backend = _make_loaded_backend(avg_logprob=-0.3)

        chunks = [_make_pcm16_silence(2.0), b""]

        segments: list[TranscriptSegment] = []
        async for seg in backend.transcribe_stream(_make_chunk_iterator(chunks)):
            segments.append(seg)

        assert len(segments) == 1
        assert segments[0].confidence is not None
        assert abs(segments[0].confidence - (-0.3)) < 0.01
