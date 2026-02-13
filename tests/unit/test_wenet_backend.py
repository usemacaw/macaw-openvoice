"""Testes para WeNetBackend.

Usa mocks para o modulo wenet — nao requer wenet instalado.
Segue o mesmo padrao de test_faster_whisper_backend.py.
"""

from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import MagicMock

import numpy as np
import pytest

from macaw._types import STTArchitecture
from macaw.exceptions import AudioFormatError, ModelLoadError
from macaw.workers.audio_utils import pcm_bytes_to_float32
from macaw.workers.stt.wenet import (
    WeNetBackend,
    _build_segments,
    _build_words,
    _extract_text,
    _load_and_configure_model,
    _resolve_compute_dtype,
    _resolve_device,
    _safe_float,
)


def _has_torch() -> bool:
    try:
        import torch  # noqa: F401

        return True
    except ImportError:
        return False


def _make_wenet_result(
    text: str = "hello world",
    segments: list[dict[str, object]] | None = None,
    tokens: list[dict[str, object]] | None = None,
) -> dict[str, object]:
    """Cria resultado fake no formato WeNet."""
    result: dict[str, object] = {"text": text}
    if segments is not None:
        result["segments"] = segments
    if tokens is not None:
        result["tokens"] = tokens
    return result


class TestArchitecture:
    def test_is_ctc(self) -> None:
        backend = WeNetBackend()
        assert backend.architecture == STTArchitecture.CTC


class TestCapabilities:
    async def test_supports_hot_words(self) -> None:
        backend = WeNetBackend()
        caps = await backend.capabilities()
        assert caps.supports_hot_words is True

    async def test_does_not_support_initial_prompt(self) -> None:
        backend = WeNetBackend()
        caps = await backend.capabilities()
        assert caps.supports_initial_prompt is False

    async def test_supports_batch(self) -> None:
        backend = WeNetBackend()
        caps = await backend.capabilities()
        assert caps.supports_batch is True

    async def test_supports_word_timestamps(self) -> None:
        backend = WeNetBackend()
        caps = await backend.capabilities()
        assert caps.supports_word_timestamps is True


class TestLoad:
    async def test_load_succeeds_with_mock(self) -> None:
        mock_model = MagicMock()
        import macaw.workers.stt.wenet as wenet_mod

        original = wenet_mod.wenet_lib
        mock_wenet = MagicMock()
        mock_wenet.load_model.return_value = mock_model
        wenet_mod.wenet_lib = mock_wenet  # type: ignore[assignment]
        try:
            backend = WeNetBackend()
            await backend.load("/models/wenet-ctc", {"language": "chinese", "device": "cpu"})
            assert backend._model is not None
            mock_wenet.load_model.assert_called_once()
        finally:
            wenet_mod.wenet_lib = original  # type: ignore[assignment]

    async def test_load_failure_raises_model_load_error(self) -> None:
        import macaw.workers.stt.wenet as wenet_mod

        original = wenet_mod.wenet_lib
        mock_wenet = MagicMock()
        mock_wenet.load_model.side_effect = RuntimeError("Model file not found")
        wenet_mod.wenet_lib = mock_wenet  # type: ignore[assignment]
        try:
            backend = WeNetBackend()
            with pytest.raises(ModelLoadError, match="Model file not found"):
                await backend.load("/models/wenet-ctc", {})
        finally:
            wenet_mod.wenet_lib = original  # type: ignore[assignment]

    async def test_load_without_library_raises_model_load_error(self) -> None:
        import macaw.workers.stt.wenet as wenet_mod

        original = wenet_mod.wenet_lib
        wenet_mod.wenet_lib = None  # type: ignore[assignment]
        try:
            backend = WeNetBackend()
            with pytest.raises(ModelLoadError, match="nao esta instalado"):
                await backend.load("/models/wenet-ctc", {})
        finally:
            wenet_mod.wenet_lib = original  # type: ignore[assignment]

    async def test_load_stores_model_path(self) -> None:
        import macaw.workers.stt.wenet as wenet_mod

        original = wenet_mod.wenet_lib
        mock_wenet = MagicMock()
        mock_wenet.load_model.return_value = MagicMock()
        wenet_mod.wenet_lib = mock_wenet  # type: ignore[assignment]
        try:
            backend = WeNetBackend()
            await backend.load("/models/wenet-ctc", {})
            assert backend._model_path == "/models/wenet-ctc"
        finally:
            wenet_mod.wenet_lib = original  # type: ignore[assignment]


class TestTranscribeFile:
    def _make_loaded_backend(self) -> WeNetBackend:
        """Cria backend com modelo mock carregado."""
        backend = WeNetBackend()
        mock_model = MagicMock()
        mock_model.transcribe.return_value = {"text": "hello world"}
        backend._model = mock_model  # type: ignore[assignment]
        return backend

    async def test_returns_text(self) -> None:
        backend = self._make_loaded_backend()
        result = await backend.transcribe_file(b"\x00\x00" * 8000)
        assert result.text == "hello world"

    async def test_returns_duration(self) -> None:
        backend = self._make_loaded_backend()
        # 8000 samples at 16kHz = 0.5s
        result = await backend.transcribe_file(b"\x00\x00" * 8000)
        assert abs(result.duration - 0.5) < 0.01

    async def test_returns_language(self) -> None:
        backend = self._make_loaded_backend()
        result = await backend.transcribe_file(b"\x00\x00" * 8000, language="pt")
        assert result.language == "pt"

    async def test_auto_language_defaults_to_zh(self) -> None:
        backend = self._make_loaded_backend()
        result = await backend.transcribe_file(b"\x00\x00" * 8000, language="auto")
        assert result.language == "zh"

    async def test_mixed_language_defaults_to_zh(self) -> None:
        backend = self._make_loaded_backend()
        result = await backend.transcribe_file(b"\x00\x00" * 8000, language="mixed")
        assert result.language == "zh"

    async def test_segments_created(self) -> None:
        backend = self._make_loaded_backend()
        result = await backend.transcribe_file(b"\x00\x00" * 8000)
        assert len(result.segments) == 1
        assert result.segments[0].text == "hello world"

    async def test_segments_with_detailed_result(self) -> None:
        backend = WeNetBackend()
        mock_model = MagicMock()
        mock_model.transcribe.return_value = {
            "text": "hello world",
            "segments": [
                {"text": "hello", "start": 0.0, "end": 0.3},
                {"text": "world", "start": 0.3, "end": 0.5},
            ],
        }
        backend._model = mock_model  # type: ignore[assignment]

        result = await backend.transcribe_file(b"\x00\x00" * 8000)
        assert len(result.segments) == 2
        assert result.segments[0].text == "hello"
        assert result.segments[1].text == "world"

    async def test_words_returned_when_requested(self) -> None:
        backend = WeNetBackend()
        mock_model = MagicMock()
        mock_model.transcribe.return_value = {
            "text": "hello world",
            "tokens": [
                {"token": "hello", "start": 0.0, "end": 0.3, "confidence": 0.95},
                {"token": "world", "start": 0.3, "end": 0.5, "confidence": 0.9},
            ],
        }
        backend._model = mock_model  # type: ignore[assignment]

        result = await backend.transcribe_file(b"\x00\x00" * 8000, word_timestamps=True)
        assert result.words is not None
        assert len(result.words) == 2
        assert result.words[0].word == "hello"
        assert abs(result.words[0].probability - 0.95) < 0.01  # type: ignore[operator]

    async def test_words_none_when_not_requested(self) -> None:
        backend = self._make_loaded_backend()
        result = await backend.transcribe_file(b"\x00\x00" * 8000, word_timestamps=False)
        assert result.words is None

    async def test_empty_audio_raises_error(self) -> None:
        backend = WeNetBackend()
        backend._model = MagicMock()  # type: ignore[assignment]
        with pytest.raises(AudioFormatError, match="Audio vazio"):
            await backend.transcribe_file(b"")

    async def test_model_not_loaded_raises_error(self) -> None:
        backend = WeNetBackend()
        with pytest.raises(ModelLoadError, match="nao carregado"):
            await backend.transcribe_file(b"\x00\x00" * 100)

    async def test_result_as_string(self) -> None:
        """WeNet pode retornar string diretamente."""
        backend = WeNetBackend()
        mock_model = MagicMock()
        mock_model.transcribe.return_value = "hello world"
        backend._model = mock_model  # type: ignore[assignment]

        result = await backend.transcribe_file(b"\x00\x00" * 8000)
        assert result.text == "hello world"

    async def test_result_as_namespace(self) -> None:
        """WeNet pode retornar objeto com atributo text."""
        backend = WeNetBackend()
        mock_model = MagicMock()
        mock_model.transcribe.return_value = SimpleNamespace(text="hello world")
        backend._model = mock_model  # type: ignore[assignment]

        result = await backend.transcribe_file(b"\x00\x00" * 8000)
        assert result.text == "hello world"


class TestTranscribeStream:
    async def test_model_not_loaded_raises_error(self) -> None:
        backend = WeNetBackend()

        async def _empty_gen() -> None:
            return
            yield  # type: ignore[misc] # pragma: no cover

        with pytest.raises(ModelLoadError, match="nao carregado"):
            async for _ in backend.transcribe_stream(_empty_gen()):  # type: ignore[arg-type]
                pass

    async def test_empty_stream_yields_nothing(self) -> None:
        backend = WeNetBackend()
        mock_model = MagicMock()
        backend._model = mock_model  # type: ignore[assignment]

        async def _empty_gen():  # type: ignore[no-untyped-def]
            yield b""

        segments = [s async for s in backend.transcribe_stream(_empty_gen())]
        assert len(segments) == 0

    async def test_single_chunk_yields_final(self) -> None:
        backend = WeNetBackend()
        mock_model = MagicMock()
        mock_model.transcribe.return_value = {"text": "hello world"}
        backend._model = mock_model  # type: ignore[assignment]

        # 16000 samples = 1 second of audio at 16kHz (2 bytes per sample)
        chunk = b"\x00\x01" * 16000

        async def _gen():  # type: ignore[no-untyped-def]
            yield chunk
            yield b""

        segments = [s async for s in backend.transcribe_stream(_gen())]

        # Should have partials + final
        finals = [s for s in segments if s.is_final]
        assert len(finals) >= 1
        assert finals[-1].text == "hello world"

    async def test_partials_emitted_before_final(self) -> None:
        backend = WeNetBackend()
        mock_model = MagicMock()
        # Each chunk triggers a partial transcribe, plus one final transcribe
        # 2 chunks >= threshold = 2 partial calls + 1 final call = 3 total
        mock_model.transcribe.side_effect = [
            {"text": "hello"},
            {"text": "hello world"},
            {"text": "hello world"},
        ]
        backend._model = mock_model  # type: ignore[assignment]

        # Create enough audio to trigger partial (>0.5s) plus final
        chunk_half_sec = b"\x00\x01" * 8000  # 0.5s

        async def _gen():  # type: ignore[no-untyped-def]
            yield chunk_half_sec
            yield chunk_half_sec
            yield b""

        segments = [s async for s in backend.transcribe_stream(_gen())]

        partials = [s for s in segments if not s.is_final]
        finals = [s for s in segments if s.is_final]

        assert len(partials) >= 1
        assert len(finals) == 1
        assert finals[0].text == "hello world"

    async def test_stream_segment_ids(self) -> None:
        backend = WeNetBackend()
        mock_model = MagicMock()
        mock_model.transcribe.return_value = {"text": "hello"}
        backend._model = mock_model  # type: ignore[assignment]

        chunk = b"\x00\x01" * 16000

        async def _gen():  # type: ignore[no-untyped-def]
            yield chunk
            yield b""

        segments = [s async for s in backend.transcribe_stream(_gen())]
        for seg in segments:
            assert seg.segment_id == 0

    async def test_stream_empty_result_not_emitted(self) -> None:
        backend = WeNetBackend()
        mock_model = MagicMock()
        mock_model.transcribe.return_value = {"text": ""}
        backend._model = mock_model  # type: ignore[assignment]

        chunk = b"\x00\x01" * 16000

        async def _gen():  # type: ignore[no-untyped-def]
            yield chunk
            yield b""

        segments = [s async for s in backend.transcribe_stream(_gen())]
        assert len(segments) == 0

    async def test_stream_with_language(self) -> None:
        backend = WeNetBackend()
        mock_model = MagicMock()
        mock_model.transcribe.return_value = {"text": "ola"}
        backend._model = mock_model  # type: ignore[assignment]

        chunk = b"\x00\x01" * 16000

        async def _gen():  # type: ignore[no-untyped-def]
            yield chunk
            yield b""

        segments = [s async for s in backend.transcribe_stream(_gen(), language="pt")]
        for seg in segments:
            assert seg.language == "pt"


class TestCTCStreamingBehavior:
    """Tests specific to CTC streaming characteristics."""

    async def test_partial_emitted_after_first_chunk(self) -> None:
        """CTC should emit partial after the very first chunk (>160ms)."""
        backend = WeNetBackend()
        mock_model = MagicMock()
        mock_model.transcribe.side_effect = [
            {"text": "hi"},
            {"text": "hi"},
        ]
        backend._model = mock_model  # type: ignore[assignment]

        # 3200 samples = 200ms (exceeds 160ms min), should trigger partial
        chunk_200ms = b"\x00\x01" * 3200

        async def _gen():  # type: ignore[no-untyped-def]
            yield chunk_200ms
            yield b""

        segments = [s async for s in backend.transcribe_stream(_gen())]
        partials = [s for s in segments if not s.is_final]
        assert len(partials) >= 1, "CTC should emit partial after first chunk"

    async def test_no_partial_for_tiny_chunk(self) -> None:
        """Chunks smaller than 160ms should not trigger partial."""
        backend = WeNetBackend()
        mock_model = MagicMock()
        # Only 1 call: the final transcription
        mock_model.transcribe.return_value = {"text": "hello"}
        backend._model = mock_model  # type: ignore[assignment]

        # 1600 samples = 100ms (below 160ms threshold)
        tiny_chunk = b"\x00\x01" * 1600

        async def _gen():  # type: ignore[no-untyped-def]
            yield tiny_chunk
            yield b""

        segments = [s async for s in backend.transcribe_stream(_gen())]
        # Should only have the final (tiny_chunk is below partial threshold
        # but still accumulated for final)
        partials = [s for s in segments if not s.is_final]
        assert len(partials) == 0

    async def test_each_chunk_triggers_partial(self) -> None:
        """For CTC, each chunk that exceeds min should trigger a new partial."""
        backend = WeNetBackend()
        mock_model = MagicMock()
        mock_model.transcribe.side_effect = [
            {"text": "hello"},
            {"text": "hello world"},
            {"text": "hello world today"},
            {"text": "hello world today"},
        ]
        backend._model = mock_model  # type: ignore[assignment]

        chunk = b"\x00\x01" * 8000  # 0.5s each

        async def _gen():  # type: ignore[no-untyped-def]
            yield chunk
            yield chunk
            yield chunk
            yield b""

        segments = [s async for s in backend.transcribe_stream(_gen())]
        partials = [s for s in segments if not s.is_final]
        finals = [s for s in segments if s.is_final]

        # 3 chunks => 3 partials (each with new text) + 1 final
        assert len(partials) == 3
        assert len(finals) == 1
        assert partials[0].text == "hello"
        assert partials[1].text == "hello world"
        assert partials[2].text == "hello world today"

    async def test_duplicate_text_not_emitted(self) -> None:
        """If CTC returns same text, no duplicate partial is emitted."""
        backend = WeNetBackend()
        mock_model = MagicMock()
        mock_model.transcribe.side_effect = [
            {"text": "hello"},
            {"text": "hello"},  # Same text — should be suppressed
            {"text": "hello"},
        ]
        backend._model = mock_model  # type: ignore[assignment]

        chunk = b"\x00\x01" * 8000

        async def _gen():  # type: ignore[no-untyped-def]
            yield chunk
            yield chunk
            yield b""

        segments = [s async for s in backend.transcribe_stream(_gen())]
        partials = [s for s in segments if not s.is_final]
        assert len(partials) == 1  # Only first unique partial

    async def test_confidence_from_result(self) -> None:
        """Final segment should carry confidence if available."""
        backend = WeNetBackend()
        mock_model = MagicMock()
        mock_model.transcribe.return_value = {"text": "test", "confidence": 0.95}
        backend._model = mock_model  # type: ignore[assignment]

        chunk = b"\x00\x01" * 8000

        async def _gen():  # type: ignore[no-untyped-def]
            yield chunk
            yield b""

        segments = [s async for s in backend.transcribe_stream(_gen())]
        finals = [s for s in segments if s.is_final]
        assert len(finals) == 1

    async def test_stream_auto_language_resolved_to_none(self) -> None:
        """Auto/mixed language should resolve to None in segments."""
        backend = WeNetBackend()
        mock_model = MagicMock()
        mock_model.transcribe.side_effect = [
            {"text": "hi"},
            {"text": "hi"},
        ]
        backend._model = mock_model  # type: ignore[assignment]

        chunk = b"\x00\x01" * 8000

        async def _gen():  # type: ignore[no-untyped-def]
            yield chunk
            yield b""

        segments = [s async for s in backend.transcribe_stream(_gen(), language="auto")]
        for seg in segments:
            assert seg.language is None

    async def test_stream_final_has_end_ms(self) -> None:
        """Final segment should have end_ms calculated from total audio."""
        backend = WeNetBackend()
        mock_model = MagicMock()
        mock_model.transcribe.side_effect = [
            {"text": "hello"},
            {"text": "hello"},
        ]
        backend._model = mock_model  # type: ignore[assignment]

        chunk = b"\x00\x01" * 16000  # 1 second

        async def _gen():  # type: ignore[no-untyped-def]
            yield chunk
            yield b""

        segments = [s async for s in backend.transcribe_stream(_gen())]
        finals = [s for s in segments if s.is_final]
        assert len(finals) == 1
        assert finals[0].end_ms == 1000
        assert finals[0].start_ms == 0

    async def test_stream_error_during_transcription(self) -> None:
        """Error during streaming transcription should propagate."""
        backend = WeNetBackend()
        mock_model = MagicMock()
        mock_model.transcribe.side_effect = RuntimeError("inference failed")
        backend._model = mock_model  # type: ignore[assignment]

        chunk = b"\x00\x01" * 8000

        async def _gen():  # type: ignore[no-untyped-def]
            yield chunk
            yield b""

        with pytest.raises(RuntimeError, match="inference failed"):
            async for _ in backend.transcribe_stream(_gen()):
                pass


class TestUnload:
    async def test_clears_model(self) -> None:
        backend = WeNetBackend()
        backend._model = MagicMock()  # type: ignore[assignment]
        backend._model_path = "/models/test"
        await backend.unload()
        assert backend._model is None
        assert backend._model_path == ""

    async def test_unload_when_already_none(self) -> None:
        backend = WeNetBackend()
        await backend.unload()
        assert backend._model is None


class TestHealth:
    async def test_ok_when_model_loaded(self) -> None:
        backend = WeNetBackend()
        backend._model = MagicMock()  # type: ignore[assignment]
        health = await backend.health()
        assert health["status"] == "ok"

    async def test_not_loaded_when_model_none(self) -> None:
        backend = WeNetBackend()
        health = await backend.health()
        assert health["status"] == "not_loaded"


class TestResolveDevice:
    def test_auto_defaults_to_cpu(self) -> None:
        assert _resolve_device("auto") == "cpu"

    def test_cpu_passthrough(self) -> None:
        assert _resolve_device("cpu") == "cpu"

    def test_cuda_passthrough(self) -> None:
        assert _resolve_device("cuda") == "cuda"

    def test_cuda_with_id(self) -> None:
        assert _resolve_device("cuda:0") == "cuda:0"


class TestAudioBytesToNumpy:
    def test_valid_pcm_16bit(self) -> None:
        audio = b"\x00\x00\xff\x7f"
        result = pcm_bytes_to_float32(audio)
        assert result.dtype == np.float32
        assert len(result) == 2
        assert abs(result[0]) < 0.01
        assert abs(result[1] - 1.0) < 0.01

    def test_odd_bytes_raises_error(self) -> None:
        with pytest.raises(AudioFormatError, match="numero par"):
            pcm_bytes_to_float32(b"\x00\x01\x02")

    def test_empty_audio_returns_empty_array(self) -> None:
        result = pcm_bytes_to_float32(b"")
        assert len(result) == 0


class TestExtractText:
    def test_extracts_from_dict(self) -> None:
        assert _extract_text({"text": "hello"}) == "hello"

    def test_strips_whitespace(self) -> None:
        assert _extract_text({"text": "  hello  "}) == "hello"

    def test_missing_text_key_returns_empty(self) -> None:
        assert _extract_text({}) == ""

    def test_non_string_text_converted(self) -> None:
        assert _extract_text({"text": 123}) == "123"


class TestBuildSegments:
    def test_single_segment_from_text(self) -> None:
        result = _build_segments({"text": "hello"}, duration=1.0)
        assert len(result) == 1
        assert result[0].id == 0
        assert result[0].start == 0.0
        assert result[0].end == 1.0
        assert result[0].text == "hello"

    def test_empty_text_returns_empty(self) -> None:
        result = _build_segments({"text": ""}, duration=1.0)
        assert len(result) == 0

    def test_detailed_segments(self) -> None:
        result_dict: dict[str, object] = {
            "text": "hello world",
            "segments": [
                {"text": "hello", "start": 0.0, "end": 0.5},
                {"text": "world", "start": 0.5, "end": 1.0},
            ],
        }
        result = _build_segments(result_dict, duration=1.0)
        assert len(result) == 2
        assert result[0].text == "hello"
        assert result[1].text == "world"

    def test_empty_segments_list_falls_back(self) -> None:
        result = _build_segments({"text": "hello", "segments": []}, duration=1.0)
        assert len(result) == 1
        assert result[0].text == "hello"


class TestBuildWords:
    def test_builds_from_tokens(self) -> None:
        result_dict: dict[str, object] = {
            "tokens": [
                {"token": "hello", "start": 0.0, "end": 0.3, "confidence": 0.95},
                {"token": "world", "start": 0.3, "end": 0.5, "confidence": 0.9},
            ]
        }
        words = _build_words(result_dict)
        assert words is not None
        assert len(words) == 2
        assert words[0].word == "hello"
        assert abs(words[0].probability - 0.95) < 0.01  # type: ignore[operator]

    def test_returns_none_when_no_tokens(self) -> None:
        assert _build_words({"text": "hello"}) is None

    def test_returns_none_for_empty_tokens(self) -> None:
        assert _build_words({"tokens": []}) is None

    def test_skips_empty_token_text(self) -> None:
        result_dict: dict[str, object] = {
            "tokens": [
                {"token": "", "start": 0.0, "end": 0.1},
                {"token": "hello", "start": 0.1, "end": 0.3},
            ]
        }
        words = _build_words(result_dict)
        assert words is not None
        assert len(words) == 1
        assert words[0].word == "hello"

    def test_uses_word_key_fallback(self) -> None:
        """Some WeNet versions use 'word' instead of 'token'."""
        result_dict: dict[str, object] = {
            "tokens": [
                {"word": "hello", "start": 0.0, "end": 0.3},
            ]
        }
        words = _build_words(result_dict)
        assert words is not None
        assert words[0].word == "hello"


class TestSafeFloat:
    def test_converts_float(self) -> None:
        assert _safe_float(0.95) == 0.95

    def test_converts_int(self) -> None:
        assert _safe_float(1) == 1.0

    def test_converts_string_number(self) -> None:
        assert _safe_float("0.5") == 0.5

    def test_returns_none_for_none(self) -> None:
        assert _safe_float(None) is None

    def test_returns_none_for_invalid(self) -> None:
        assert _safe_float("not_a_number") is None


class TestResolveComputeDtype:
    def test_auto_on_cuda_returns_float16(self) -> None:
        assert _resolve_compute_dtype("auto", "cuda") == "float16"

    def test_auto_on_cuda_with_id_returns_float16(self) -> None:
        assert _resolve_compute_dtype("auto", "cuda:0") == "float16"

    def test_auto_on_cpu_returns_float32(self) -> None:
        assert _resolve_compute_dtype("auto", "cpu") == "float32"

    def test_float16_on_cuda_passthrough(self) -> None:
        assert _resolve_compute_dtype("float16", "cuda") == "float16"

    def test_float16_on_cpu_falls_back_to_float32(self) -> None:
        assert _resolve_compute_dtype("float16", "cpu") == "float32"

    def test_bfloat16_on_cuda_passthrough(self) -> None:
        assert _resolve_compute_dtype("bfloat16", "cuda") == "bfloat16"

    def test_bfloat16_on_cpu_passthrough(self) -> None:
        assert _resolve_compute_dtype("bfloat16", "cpu") == "bfloat16"

    def test_float32_on_any_device_passthrough(self) -> None:
        assert _resolve_compute_dtype("float32", "cpu") == "float32"
        assert _resolve_compute_dtype("float32", "cuda") == "float32"

    def test_unknown_type_falls_back_to_float32(self) -> None:
        assert _resolve_compute_dtype("int8", "cuda") == "float32"


class TestLoadAndConfigureModel:
    def test_float32_does_not_call_half(self) -> None:
        import macaw.workers.stt.wenet as wenet_mod

        original = wenet_mod.wenet_lib
        mock_wenet = MagicMock()
        mock_model = MagicMock()
        mock_wenet.load_model.return_value = mock_model
        wenet_mod.wenet_lib = mock_wenet  # type: ignore[assignment]
        try:
            result = _load_and_configure_model("/models/test", "cpu", "float32")
            assert result is mock_model
            mock_model.half.assert_not_called()
        finally:
            wenet_mod.wenet_lib = original  # type: ignore[assignment]

    def test_float16_calls_half(self) -> None:
        import macaw.workers.stt.wenet as wenet_mod

        original = wenet_mod.wenet_lib
        mock_wenet = MagicMock()
        mock_model = MagicMock()
        mock_wenet.load_model.return_value = mock_model
        wenet_mod.wenet_lib = mock_wenet  # type: ignore[assignment]
        try:
            result = _load_and_configure_model("/models/test", "cuda", "float16")
            assert result is mock_model
            mock_model.half.assert_called_once()
        finally:
            wenet_mod.wenet_lib = original  # type: ignore[assignment]

    @pytest.mark.skipif(
        not _has_torch(),
        reason="torch not installed",
    )
    def test_bfloat16_calls_model_to(self) -> None:
        import torch

        import macaw.workers.stt.wenet as wenet_mod

        original = wenet_mod.wenet_lib
        mock_wenet = MagicMock()
        mock_model = MagicMock()
        mock_wenet.load_model.return_value = mock_model
        wenet_mod.wenet_lib = mock_wenet  # type: ignore[assignment]
        try:
            result = _load_and_configure_model("/models/test", "cuda", "bfloat16")
            assert result is mock_model
            mock_model.to.assert_called_once_with(torch.bfloat16)
        finally:
            wenet_mod.wenet_lib = original  # type: ignore[assignment]


class TestLoadWithComputeType:
    async def test_load_with_float16_on_cuda(self) -> None:
        import macaw.workers.stt.wenet as wenet_mod

        original = wenet_mod.wenet_lib
        mock_wenet = MagicMock()
        mock_model = MagicMock()
        mock_wenet.load_model.return_value = mock_model
        wenet_mod.wenet_lib = mock_wenet  # type: ignore[assignment]
        try:
            backend = WeNetBackend()
            await backend.load(
                "/models/wenet-ctc",
                {"device": "cuda", "compute_type": "float16"},
            )
            assert backend._model is not None
            mock_model.half.assert_called_once()
        finally:
            wenet_mod.wenet_lib = original  # type: ignore[assignment]

    async def test_load_with_auto_on_cpu_stays_float32(self) -> None:
        import macaw.workers.stt.wenet as wenet_mod

        original = wenet_mod.wenet_lib
        mock_wenet = MagicMock()
        mock_model = MagicMock()
        mock_wenet.load_model.return_value = mock_model
        wenet_mod.wenet_lib = mock_wenet  # type: ignore[assignment]
        try:
            backend = WeNetBackend()
            await backend.load(
                "/models/wenet-ctc",
                {"device": "cpu", "compute_type": "auto"},
            )
            assert backend._model is not None
            mock_model.half.assert_not_called()
            mock_model.to.assert_not_called()
        finally:
            wenet_mod.wenet_lib = original  # type: ignore[assignment]

    async def test_load_defaults_compute_type_to_auto(self) -> None:
        import macaw.workers.stt.wenet as wenet_mod

        original = wenet_mod.wenet_lib
        mock_wenet = MagicMock()
        mock_model = MagicMock()
        mock_wenet.load_model.return_value = mock_model
        wenet_mod.wenet_lib = mock_wenet  # type: ignore[assignment]
        try:
            backend = WeNetBackend()
            # No compute_type in config — should default to "auto"
            await backend.load("/models/wenet-ctc", {"device": "cpu"})
            assert backend._model is not None
            # auto + cpu = float32, no dtype conversion
            mock_model.half.assert_not_called()
            mock_model.to.assert_not_called()
        finally:
            wenet_mod.wenet_lib = original  # type: ignore[assignment]

    async def test_load_with_float16_on_cpu_falls_back_to_float32(self) -> None:
        import macaw.workers.stt.wenet as wenet_mod

        original = wenet_mod.wenet_lib
        mock_wenet = MagicMock()
        mock_model = MagicMock()
        mock_wenet.load_model.return_value = mock_model
        wenet_mod.wenet_lib = mock_wenet  # type: ignore[assignment]
        try:
            backend = WeNetBackend()
            # float16 on CPU should fall back to float32
            await backend.load(
                "/models/wenet-ctc",
                {"device": "cpu", "compute_type": "float16"},
            )
            assert backend._model is not None
            mock_model.half.assert_not_called()
        finally:
            wenet_mod.wenet_lib = original  # type: ignore[assignment]
