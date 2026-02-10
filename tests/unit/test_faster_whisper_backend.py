"""Testes para FasterWhisperBackend.

Usa mocks para WhisperModel — nao requer faster-whisper instalado.
"""

from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from macaw._types import STTArchitecture
from macaw.exceptions import AudioFormatError, ModelLoadError
from macaw.workers.stt.faster_whisper import (
    FasterWhisperBackend,
    _audio_bytes_to_numpy,
)


def _make_fw_segment(
    text: str = "hello",
    start: float = 0.0,
    end: float = 1.0,
    avg_logprob: float = -0.3,
    no_speech_prob: float = 0.01,
    compression_ratio: float = 1.1,
    words: list[object] | None = None,
) -> SimpleNamespace:
    """Cria segmento fake no formato faster-whisper."""
    return SimpleNamespace(
        text=text,
        start=start,
        end=end,
        avg_logprob=avg_logprob,
        no_speech_prob=no_speech_prob,
        compression_ratio=compression_ratio,
        words=words,
    )


def _make_fw_word(
    word: str = "hello",
    start: float = 0.0,
    end: float = 0.5,
    probability: float = 0.9,
) -> SimpleNamespace:
    """Cria word fake no formato faster-whisper."""
    return SimpleNamespace(word=word, start=start, end=end, probability=probability)


def _make_fw_info(language: str = "en", duration: float = 1.0) -> SimpleNamespace:
    """Cria TranscriptionInfo fake."""
    return SimpleNamespace(language=language, duration=duration)


class TestArchitecture:
    def test_is_encoder_decoder(self) -> None:
        backend = FasterWhisperBackend()
        assert backend.architecture == STTArchitecture.ENCODER_DECODER


class TestLoad:
    async def test_load_succeeds_with_mock(self) -> None:
        mock_model = MagicMock()
        with patch.dict(
            "macaw.workers.stt.faster_whisper.__dict__",
            {"WhisperModel": MagicMock(return_value=mock_model)},
        ):
            # Re-import to get the patched version
            import macaw.workers.stt.faster_whisper as fw_mod

            original = fw_mod.WhisperModel
            fw_mod.WhisperModel = MagicMock(return_value=mock_model)  # type: ignore[assignment]
            try:
                backend = FasterWhisperBackend()
                await backend.load(
                    "/models/test", {"model_size": "tiny", "compute_type": "int8", "device": "cpu"}
                )
                assert backend._model is not None
            finally:
                fw_mod.WhisperModel = original  # type: ignore[assignment]

    async def test_load_failure_raises_model_load_error(self) -> None:
        import macaw.workers.stt.faster_whisper as fw_mod

        original = fw_mod.WhisperModel
        fw_mod.WhisperModel = MagicMock(side_effect=RuntimeError("CUDA not available"))  # type: ignore[assignment]
        try:
            backend = FasterWhisperBackend()
            with pytest.raises(ModelLoadError, match="CUDA not available"):
                await backend.load("/models/test", {"model_size": "large-v3"})
        finally:
            fw_mod.WhisperModel = original  # type: ignore[assignment]

    async def test_load_without_library_raises_model_load_error(self) -> None:
        import macaw.workers.stt.faster_whisper as fw_mod

        original = fw_mod.WhisperModel
        fw_mod.WhisperModel = None  # type: ignore[assignment]
        try:
            backend = FasterWhisperBackend()
            with pytest.raises(ModelLoadError, match="nao esta instalado"):
                await backend.load("/models/test", {})
        finally:
            fw_mod.WhisperModel = original  # type: ignore[assignment]


class TestTranscribeFile:
    async def _make_loaded_backend(self) -> FasterWhisperBackend:
        """Cria backend com modelo mock carregado."""
        backend = FasterWhisperBackend()
        mock_model = MagicMock()
        segments = [_make_fw_segment(text=" hello world ")]
        info = _make_fw_info()
        mock_model.transcribe.return_value = (iter(segments), info)
        backend._model = mock_model  # type: ignore[assignment]
        return backend

    async def test_returns_text(self) -> None:
        backend = await self._make_loaded_backend()
        result = await backend.transcribe_file(b"\x00\x00" * 8000)
        assert result.text == "hello world"

    async def test_returns_language(self) -> None:
        backend = await self._make_loaded_backend()
        result = await backend.transcribe_file(b"\x00\x00" * 8000)
        assert result.language == "en"

    async def test_returns_duration(self) -> None:
        backend = await self._make_loaded_backend()
        result = await backend.transcribe_file(b"\x00\x00" * 8000)
        assert abs(result.duration - 1.0) < 0.01

    async def test_segments_mapped(self) -> None:
        backend = await self._make_loaded_backend()
        result = await backend.transcribe_file(b"\x00\x00" * 8000)
        assert len(result.segments) == 1
        assert result.segments[0].text == "hello world"

    async def test_words_returned_when_requested(self) -> None:
        backend = FasterWhisperBackend()
        mock_model = MagicMock()
        words = [_make_fw_word("hello", 0.0, 0.5, 0.9), _make_fw_word("world", 0.5, 1.0, 0.8)]
        segments = [_make_fw_segment(text=" hello world ", words=words)]
        info = _make_fw_info()
        mock_model.transcribe.return_value = (iter(segments), info)
        backend._model = mock_model  # type: ignore[assignment]

        result = await backend.transcribe_file(b"\x00\x00" * 8000, word_timestamps=True)
        assert result.words is not None
        assert len(result.words) == 2
        assert result.words[0].word == "hello"
        assert abs(result.words[0].probability - 0.9) < 0.01  # type: ignore[operator]

    async def test_empty_audio_raises_error(self) -> None:
        backend = FasterWhisperBackend()
        backend._model = MagicMock()  # type: ignore[assignment]
        with pytest.raises(AudioFormatError, match="Audio vazio"):
            await backend.transcribe_file(b"")

    async def test_model_not_loaded_raises_error(self) -> None:
        backend = FasterWhisperBackend()
        with pytest.raises(ModelLoadError, match="nao carregado"):
            await backend.transcribe_file(b"\x00\x00" * 100)

    async def test_auto_language_passed_as_none(self) -> None:
        backend = await self._make_loaded_backend()
        await backend.transcribe_file(b"\x00\x00" * 8000, language="auto")
        call_kwargs = backend._model.transcribe.call_args  # type: ignore[union-attr]
        assert call_kwargs.kwargs.get("language") is None

    async def test_hot_words_added_to_prompt(self) -> None:
        backend = await self._make_loaded_backend()
        await backend.transcribe_file(
            b"\x00\x00" * 8000,
            hot_words=["PIX", "TED"],
            initial_prompt="Contexto",
        )
        call_kwargs = backend._model.transcribe.call_args  # type: ignore[union-attr]
        prompt = call_kwargs.kwargs.get("initial_prompt")
        assert "PIX" in prompt
        assert "TED" in prompt
        assert "Contexto" in prompt


class TestTranscribeStream:
    async def test_model_not_loaded_raises_error(self) -> None:
        backend = FasterWhisperBackend()
        with pytest.raises(ModelLoadError, match="nao carregado"):

            async def _empty_gen() -> None:
                return
                yield  # type: ignore[misc] # pragma: no cover

            async for _ in backend.transcribe_stream(_empty_gen()):  # type: ignore[arg-type]
                pass


class TestUnload:
    async def test_clears_model(self) -> None:
        backend = FasterWhisperBackend()
        backend._model = MagicMock()  # type: ignore[assignment]
        await backend.unload()
        assert backend._model is None

    async def test_unload_when_already_none(self) -> None:
        backend = FasterWhisperBackend()
        await backend.unload()
        assert backend._model is None


class TestHealth:
    async def test_ok_when_model_loaded(self) -> None:
        backend = FasterWhisperBackend()
        backend._model = MagicMock()  # type: ignore[assignment]
        health = await backend.health()
        assert health["status"] == "ok"

    async def test_not_loaded_when_model_none(self) -> None:
        backend = FasterWhisperBackend()
        health = await backend.health()
        assert health["status"] == "not_loaded"


class TestAudioBytesToNumpy:
    def test_valid_pcm_16bit(self) -> None:
        # 4 bytes = 2 samples of int16
        audio = b"\x00\x00\xff\x7f"  # 0 and 32767
        result = _audio_bytes_to_numpy(audio)
        assert result.dtype == np.float32
        assert len(result) == 2
        assert abs(result[0]) < 0.01
        assert abs(result[1] - 1.0) < 0.01

    def test_odd_bytes_raises_error(self) -> None:
        with pytest.raises(AudioFormatError, match="numero par"):
            _audio_bytes_to_numpy(b"\x00\x01\x02")

    def test_empty_audio_returns_empty_array(self) -> None:
        result = _audio_bytes_to_numpy(b"")
        assert len(result) == 0
