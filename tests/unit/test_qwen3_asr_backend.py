"""Tests for Qwen3ASRBackend.

Uses mocks for qwen_asr — does not require qwen-asr installed.
Follows the same pattern as FasterWhisperBackend tests.
"""

from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest

from macaw._types import STTArchitecture
from macaw.exceptions import AudioFormatError, ModelLoadError


def _make_pcm_silence(duration_s: float = 1.0, sample_rate: int = 16000) -> bytes:
    """Create silent 16-bit PCM bytes."""
    num_samples = int(duration_s * sample_rate)
    return b"\x00\x00" * num_samples


def _make_transcription_result(
    text: str = "hello world",
    language: str = "en",
    time_stamps: list[tuple[str, float, float]] | None = None,
) -> SimpleNamespace:
    """Create a fake Qwen3-ASR transcription result."""
    return SimpleNamespace(
        text=text,
        language=language,
        time_stamps=time_stamps,
    )


class TestArchitecture:
    def test_is_encoder_decoder(self) -> None:
        from macaw.workers.stt.qwen3_asr import Qwen3ASRBackend

        backend = Qwen3ASRBackend()
        assert backend.architecture == STTArchitecture.ENCODER_DECODER


class TestLoad:
    async def test_load_raises_when_package_missing(self) -> None:
        import macaw.workers.stt.qwen3_asr as mod

        original = mod._Qwen3ASRModel
        mod._Qwen3ASRModel = None  # type: ignore[assignment]
        try:
            from macaw.workers.stt.qwen3_asr import Qwen3ASRBackend

            backend = Qwen3ASRBackend()
            with pytest.raises(ModelLoadError, match="qwen-asr is not installed"):
                await backend.load("/models/test", {})
        finally:
            mod._Qwen3ASRModel = original  # type: ignore[assignment]

    async def test_load_configures_device_and_dtype(self) -> None:
        import macaw.workers.stt.qwen3_asr as mod

        mock_model = MagicMock()
        mock_cls = MagicMock()
        mock_cls.from_pretrained = MagicMock(return_value=mock_model)

        original = mod._Qwen3ASRModel
        mod._Qwen3ASRModel = mock_cls  # type: ignore[assignment]
        try:
            from macaw.workers.stt.qwen3_asr import Qwen3ASRBackend

            backend = Qwen3ASRBackend()
            await backend.load(
                "/models/qwen3-asr",
                {"device": "cpu", "dtype": "float32"},
            )
            assert backend._model is not None
            mock_cls.from_pretrained.assert_called_once()
            call_kwargs = mock_cls.from_pretrained.call_args
            assert call_kwargs[0][0] == "/models/qwen3-asr"
            assert call_kwargs[1]["device_map"] == "cpu"
        finally:
            mod._Qwen3ASRModel = original  # type: ignore[assignment]

    async def test_load_failure_raises_model_load_error(self) -> None:
        import macaw.workers.stt.qwen3_asr as mod

        mock_cls = MagicMock()
        mock_cls.from_pretrained = MagicMock(side_effect=RuntimeError("CUDA OOM"))

        original = mod._Qwen3ASRModel
        mod._Qwen3ASRModel = mock_cls  # type: ignore[assignment]
        try:
            from macaw.workers.stt.qwen3_asr import Qwen3ASRBackend

            backend = Qwen3ASRBackend()
            with pytest.raises(ModelLoadError, match="CUDA OOM"):
                await backend.load("/models/test", {})
        finally:
            mod._Qwen3ASRModel = original  # type: ignore[assignment]


class TestCapabilities:
    async def test_reports_batch_and_word_timestamps(self) -> None:
        from macaw.workers.stt.qwen3_asr import Qwen3ASRBackend

        backend = Qwen3ASRBackend()
        caps = await backend.capabilities()
        assert caps.supports_batch is True
        assert caps.supports_word_timestamps is True
        assert caps.supports_hot_words is False
        assert caps.supports_initial_prompt is False


class TestTranscribeFile:
    async def test_returns_batch_result(self) -> None:
        from macaw.workers.stt.qwen3_asr import Qwen3ASRBackend

        result = _make_transcription_result(text="hello world", language="en")
        mock_model = MagicMock()
        mock_model.transcribe = MagicMock(return_value=result)

        backend = Qwen3ASRBackend()
        backend._model = mock_model

        audio = _make_pcm_silence(1.0)
        batch = await backend.transcribe_file(audio)

        assert batch.text == "hello world"
        assert batch.language == "en"
        assert batch.duration == pytest.approx(1.0)
        assert len(batch.segments) == 1

    async def test_with_word_timestamps(self) -> None:
        from macaw.workers.stt.qwen3_asr import Qwen3ASRBackend

        timestamps = [
            ("hello", 0.0, 0.5),
            ("world", 0.5, 1.0),
        ]
        result = _make_transcription_result(
            text="hello world",
            language="en",
            time_stamps=timestamps,
        )
        mock_model = MagicMock()
        mock_model.transcribe = MagicMock(return_value=result)

        backend = Qwen3ASRBackend()
        backend._model = mock_model

        audio = _make_pcm_silence(1.0)
        batch = await backend.transcribe_file(audio, word_timestamps=True)

        assert batch.words is not None
        assert len(batch.words) == 2
        assert batch.words[0].word == "hello"
        assert batch.words[0].start == 0.0
        assert batch.words[0].end == 0.5
        assert batch.words[1].word == "world"

    async def test_with_language_auto_detect(self) -> None:
        from macaw.workers.stt.qwen3_asr import Qwen3ASRBackend

        result = _make_transcription_result(text="bonjour", language="fr")
        mock_model = MagicMock()
        mock_model.transcribe = MagicMock(return_value=result)

        backend = Qwen3ASRBackend()
        backend._model = mock_model

        audio = _make_pcm_silence(1.0)
        batch = await backend.transcribe_file(audio, language="auto")

        assert batch.language == "fr"
        # Verify that None was passed to the model for auto-detection
        call_kwargs = mock_model.transcribe.call_args[1]
        assert call_kwargs["language"] is None

    async def test_empty_audio_raises_error(self) -> None:
        from macaw.workers.stt.qwen3_asr import Qwen3ASRBackend

        backend = Qwen3ASRBackend()
        backend._model = MagicMock()

        with pytest.raises(AudioFormatError, match="Empty audio"):
            await backend.transcribe_file(b"")

    async def test_model_not_loaded_raises_error(self) -> None:
        from macaw.workers.stt.qwen3_asr import Qwen3ASRBackend

        backend = Qwen3ASRBackend()

        with pytest.raises(ModelLoadError, match="not loaded"):
            await backend.transcribe_file(_make_pcm_silence())

    async def test_word_timestamps_not_requested_returns_none(self) -> None:
        from macaw.workers.stt.qwen3_asr import Qwen3ASRBackend

        timestamps = [("hello", 0.0, 0.5)]
        result = _make_transcription_result(
            text="hello",
            time_stamps=timestamps,
        )
        mock_model = MagicMock()
        mock_model.transcribe = MagicMock(return_value=result)

        backend = Qwen3ASRBackend()
        backend._model = mock_model

        audio = _make_pcm_silence(1.0)
        batch = await backend.transcribe_file(audio, word_timestamps=False)

        assert batch.words is None


class TestTranscribeStream:
    async def test_accumulates_and_yields(self) -> None:
        from macaw.workers.stt.qwen3_asr import Qwen3ASRBackend

        result = _make_transcription_result(text="streaming test", language="en")
        mock_model = MagicMock()
        mock_model.transcribe = MagicMock(return_value=result)

        backend = Qwen3ASRBackend()
        backend._model = mock_model
        # Set low threshold so we can trigger with small audio
        backend._accumulation_threshold_seconds = 0.1

        async def audio_gen() -> __import__("collections.abc").AsyncIterator[bytes]:
            # Yield enough audio to trigger threshold (0.1s = 1600 samples = 3200 bytes)
            yield _make_pcm_silence(0.15)
            yield b""  # End of stream

        segments = []
        async for seg in backend.transcribe_stream(audio_gen()):
            segments.append(seg)

        assert len(segments) == 1
        assert segments[0].text == "streaming test"
        assert segments[0].is_final is True

    async def test_flushes_remaining_on_empty_chunk(self) -> None:
        from macaw.workers.stt.qwen3_asr import Qwen3ASRBackend

        result = _make_transcription_result(text="final", language="en")
        mock_model = MagicMock()
        mock_model.transcribe = MagicMock(return_value=result)

        backend = Qwen3ASRBackend()
        backend._model = mock_model
        backend._accumulation_threshold_seconds = 10.0  # High threshold

        async def audio_gen() -> __import__("collections.abc").AsyncIterator[bytes]:
            yield _make_pcm_silence(0.5)  # Below threshold
            yield b""  # End of stream

        segments = []
        async for seg in backend.transcribe_stream(audio_gen()):
            segments.append(seg)

        # Should flush remaining buffer on end-of-stream
        assert len(segments) == 1
        assert segments[0].text == "final"

    async def test_model_not_loaded_raises_error(self) -> None:
        from macaw.workers.stt.qwen3_asr import Qwen3ASRBackend

        backend = Qwen3ASRBackend()

        async def audio_gen() -> __import__("collections.abc").AsyncIterator[bytes]:
            yield b""

        with pytest.raises(ModelLoadError, match="not loaded"):
            async for _ in backend.transcribe_stream(audio_gen()):
                pass


class TestUnload:
    async def test_releases_gpu_memory(self) -> None:
        from macaw.workers.stt.qwen3_asr import Qwen3ASRBackend

        backend = Qwen3ASRBackend()
        backend._model = MagicMock()

        await backend.unload()

        assert backend._model is None


class TestHealth:
    async def test_ok_when_loaded(self) -> None:
        from macaw.workers.stt.qwen3_asr import Qwen3ASRBackend

        backend = Qwen3ASRBackend()
        backend._model = MagicMock()

        health = await backend.health()
        assert health["status"] == "ok"

    async def test_not_loaded_when_model_none(self) -> None:
        from macaw.workers.stt.qwen3_asr import Qwen3ASRBackend

        backend = Qwen3ASRBackend()

        health = await backend.health()
        assert health["status"] == "not_loaded"


class TestExtractWordTimestamps:
    def test_extracts_from_result(self) -> None:
        from macaw.workers.stt.qwen3_asr import _extract_word_timestamps

        result = _make_transcription_result(
            time_stamps=[("hello", 0.0, 0.5), ("world", 0.5, 1.0)],
        )
        words = _extract_word_timestamps(result, word_timestamps=True)

        assert words is not None
        assert len(words) == 2
        assert words[0].word == "hello"
        assert words[1].word == "world"

    def test_returns_none_when_disabled(self) -> None:
        from macaw.workers.stt.qwen3_asr import _extract_word_timestamps

        result = _make_transcription_result(
            time_stamps=[("hello", 0.0, 0.5)],
        )
        words = _extract_word_timestamps(result, word_timestamps=False)
        assert words is None

    def test_returns_none_when_no_timestamps(self) -> None:
        from macaw.workers.stt.qwen3_asr import _extract_word_timestamps

        result = _make_transcription_result(time_stamps=None)
        words = _extract_word_timestamps(result, word_timestamps=True)
        assert words is None

    def test_returns_none_when_empty_timestamps(self) -> None:
        from macaw.workers.stt.qwen3_asr import _extract_word_timestamps

        result = _make_transcription_result(time_stamps=[])
        words = _extract_word_timestamps(result, word_timestamps=True)
        assert words is None
