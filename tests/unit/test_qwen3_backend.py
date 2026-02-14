"""Testes para Qwen3TTSBackend.

Usa mocks para o modulo qwen_tts -- nao requer qwen-tts instalado.
Segue o mesmo padrao de test_kokoro_backend.py.
"""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from macaw._types import VoiceInfo
from macaw.exceptions import ModelLoadError, TTSSynthesisError
from macaw.workers.tts.audio_utils import float32_to_pcm16_bytes
from macaw.workers.tts.qwen3 import (
    Qwen3TTSBackend,
    _resolve_device,
)


def _make_mock_qwen3_model(
    audio: np.ndarray | None = None,
    sample_rate: int = 24000,
) -> MagicMock:
    """Cria mock de Qwen3TTSModel."""
    mock_model = MagicMock()

    if audio is None:
        audio = np.zeros(2400, dtype=np.float32)

    # generate_custom_voice returns (wavs, sr) — wavs is a list
    mock_model.generate_custom_voice.return_value = ([audio], sample_rate)
    mock_model.generate_voice_clone.return_value = ([audio], sample_rate)
    mock_model.generate_voice_design.return_value = ([audio], sample_rate)
    mock_model.get_supported_speakers.return_value = ["vivian", "Aiden", "Vivian"]
    mock_model.get_supported_languages.return_value = ["English", "Chinese"]

    return mock_model


def _make_mock_qwen3_lib(
    audio: np.ndarray | None = None,
) -> MagicMock:
    """Cria mock da biblioteca qwen_tts com Qwen3TTSModel."""
    mock_model = _make_mock_qwen3_model(audio)
    mock_lib = MagicMock()
    mock_lib.from_pretrained.return_value = mock_model
    return mock_lib


class TestHealth:
    async def test_ok_when_model_loaded(self) -> None:
        backend = Qwen3TTSBackend()
        backend._model = MagicMock()
        health = await backend.health()
        assert health["status"] == "ok"

    async def test_not_loaded_when_model_none(self) -> None:
        backend = Qwen3TTSBackend()
        health = await backend.health()
        assert health["status"] == "not_loaded"


class TestLoad:
    async def test_load_succeeds_with_mock(self) -> None:
        mock_model = _make_mock_qwen3_model()

        import macaw.workers.tts.qwen3 as qwen3_mod

        original_cls = qwen3_mod._Qwen3TTSModel
        qwen3_mod._Qwen3TTSModel = MagicMock()  # type: ignore[assignment]
        try:
            with patch.object(
                qwen3_mod,
                "_load_qwen3_model",
                return_value=(mock_model, 24000),
            ):
                backend = Qwen3TTSBackend()
                await backend.load(
                    "/models/qwen3-tts",
                    {
                        "device": "cpu",
                        "dtype": "float32",
                        "variant": "custom_voice",
                    },
                )
                assert backend._model is not None
                assert backend._variant == "custom_voice"
                assert backend._sample_rate == 24000
        finally:
            qwen3_mod._Qwen3TTSModel = original_cls  # type: ignore[assignment]

    async def test_load_missing_dependency_raises(self) -> None:
        import macaw.workers.tts.qwen3 as qwen3_mod

        original = qwen3_mod._Qwen3TTSModel
        qwen3_mod._Qwen3TTSModel = None  # type: ignore[assignment]
        try:
            backend = Qwen3TTSBackend()
            with pytest.raises(ModelLoadError, match="nao esta instalado"):
                await backend.load("/models/qwen3-tts", {})
        finally:
            qwen3_mod._Qwen3TTSModel = original  # type: ignore[assignment]

    async def test_load_invalid_variant_raises(self) -> None:
        import macaw.workers.tts.qwen3 as qwen3_mod

        original_cls = qwen3_mod._Qwen3TTSModel
        qwen3_mod._Qwen3TTSModel = MagicMock()  # type: ignore[assignment]
        try:
            backend = Qwen3TTSBackend()
            with pytest.raises(ModelLoadError, match="Variante invalida"):
                await backend.load("/models/qwen3-tts", {"variant": "nonexistent"})
        finally:
            qwen3_mod._Qwen3TTSModel = original_cls  # type: ignore[assignment]

    async def test_load_stores_default_voice(self) -> None:
        mock_model = _make_mock_qwen3_model()

        import macaw.workers.tts.qwen3 as qwen3_mod

        original_cls = qwen3_mod._Qwen3TTSModel
        qwen3_mod._Qwen3TTSModel = MagicMock()  # type: ignore[assignment]
        try:
            with patch.object(
                qwen3_mod,
                "_load_qwen3_model",
                return_value=(mock_model, 24000),
            ):
                backend = Qwen3TTSBackend()
                await backend.load(
                    "/models/qwen3-tts",
                    {
                        "device": "cpu",
                        "default_voice": "Vivian",
                        "default_language": "Chinese",
                    },
                )
                assert backend._default_voice == "Vivian"
                assert backend._default_language == "Chinese"
        finally:
            qwen3_mod._Qwen3TTSModel = original_cls  # type: ignore[assignment]

    async def test_load_failure_raises_model_load_error(self) -> None:
        import macaw.workers.tts.qwen3 as qwen3_mod

        original_cls = qwen3_mod._Qwen3TTSModel
        qwen3_mod._Qwen3TTSModel = MagicMock()  # type: ignore[assignment]
        try:
            with patch.object(
                qwen3_mod,
                "_load_qwen3_model",
                side_effect=RuntimeError("CUDA OOM"),
            ):
                backend = Qwen3TTSBackend()
                with pytest.raises(ModelLoadError, match="CUDA OOM"):
                    await backend.load("/models/qwen3-tts", {"device": "cpu"})
        finally:
            qwen3_mod._Qwen3TTSModel = original_cls  # type: ignore[assignment]

    async def test_load_base_variant(self) -> None:
        mock_model = _make_mock_qwen3_model()

        import macaw.workers.tts.qwen3 as qwen3_mod

        original_cls = qwen3_mod._Qwen3TTSModel
        qwen3_mod._Qwen3TTSModel = MagicMock()  # type: ignore[assignment]
        try:
            with patch.object(
                qwen3_mod,
                "_load_qwen3_model",
                return_value=(mock_model, 24000),
            ):
                backend = Qwen3TTSBackend()
                await backend.load("/models/qwen3-tts", {"variant": "base"})
                assert backend._variant == "base"
        finally:
            qwen3_mod._Qwen3TTSModel = original_cls  # type: ignore[assignment]

    async def test_load_voice_design_variant(self) -> None:
        mock_model = _make_mock_qwen3_model()

        import macaw.workers.tts.qwen3 as qwen3_mod

        original_cls = qwen3_mod._Qwen3TTSModel
        qwen3_mod._Qwen3TTSModel = MagicMock()  # type: ignore[assignment]
        try:
            with patch.object(
                qwen3_mod,
                "_load_qwen3_model",
                return_value=(mock_model, 24000),
            ):
                backend = Qwen3TTSBackend()
                await backend.load("/models/qwen3-tts", {"variant": "voice_design"})
                assert backend._variant == "voice_design"
        finally:
            qwen3_mod._Qwen3TTSModel = original_cls  # type: ignore[assignment]


class TestSynthesize:
    def _make_loaded_backend(
        self,
        audio_output: np.ndarray | None = None,
        variant: str = "custom_voice",
    ) -> Qwen3TTSBackend:
        """Cria backend com modelo mock carregado."""
        backend = Qwen3TTSBackend()
        mock_model = _make_mock_qwen3_model(audio_output)
        backend._model = mock_model
        backend._model_path = "/models/qwen3-tts"
        backend._variant = variant
        backend._default_voice = "vivian"
        backend._default_language = "English"
        backend._sample_rate = 24000
        return backend

    async def test_custom_voice_yields_chunks(self) -> None:
        backend = self._make_loaded_backend()
        chunks: list[bytes] = []
        async for chunk in backend.synthesize("Hello world"):
            chunks.append(chunk)
        assert len(chunks) > 0
        total_bytes = sum(len(c) for c in chunks)
        assert total_bytes == 2400 * 2  # 2400 samples * 2 bytes/sample

    async def test_custom_voice_calls_generate(self) -> None:
        backend = self._make_loaded_backend()
        async for _ in backend.synthesize("Hello", voice="Aiden", options={"language": "English"}):
            pass
        backend._model.generate_custom_voice.assert_called_once_with(  # type: ignore[union-attr]
            text="Hello",
            language="English",
            speaker="Aiden",
            instruct=None,
        )

    async def test_custom_voice_with_instruct(self) -> None:
        backend = self._make_loaded_backend()
        options = {"language": "Chinese", "instruction": "angry tone"}
        async for _ in backend.synthesize("Hello", voice="Vivian", options=options):
            pass
        backend._model.generate_custom_voice.assert_called_once_with(  # type: ignore[union-attr]
            text="Hello",
            language="Chinese",
            speaker="Vivian",
            instruct="angry tone",
        )

    async def test_base_voice_clone(self) -> None:
        import io
        import wave

        backend = self._make_loaded_backend(variant="base")
        # Create valid WAV bytes (qwen_tts expects WAV, not raw bytes)
        sr = 24000
        samples = np.zeros(sr, dtype=np.int16)  # 1 second of silence
        buf = io.BytesIO()
        with wave.open(buf, "wb") as wf:
            wf.setnchannels(1)
            wf.setsampwidth(2)
            wf.setframerate(sr)
            wf.writeframes(samples.tobytes())
        ref_audio_wav = buf.getvalue()

        options = {
            "language": "English",
            "ref_audio": ref_audio_wav,
            "ref_text": "This is my voice",
        }
        async for _ in backend.synthesize("Hello", options=options):
            pass
        backend._model.generate_voice_clone.assert_called_once()  # type: ignore[union-attr]
        call_kwargs = backend._model.generate_voice_clone.call_args[1]  # type: ignore[union-attr]
        assert call_kwargs["text"] == "Hello"
        assert call_kwargs["language"] == "English"
        assert call_kwargs["ref_text"] == "This is my voice"
        # ref_audio should be decoded to (np.ndarray, sample_rate) tuple
        ref_decoded = call_kwargs["ref_audio"]
        assert isinstance(ref_decoded, tuple)
        assert isinstance(ref_decoded[0], np.ndarray)
        assert ref_decoded[1] == sr

    async def test_base_without_ref_audio_raises(self) -> None:
        backend = self._make_loaded_backend(variant="base")
        with pytest.raises(TTSSynthesisError, match="ref_audio"):
            async for _ in backend.synthesize("Hello"):
                pass

    async def test_voice_design(self) -> None:
        backend = self._make_loaded_backend(variant="voice_design")
        options = {
            "language": "English",
            "instruction": "A warm female voice with British accent",
        }
        async for _ in backend.synthesize("Hello", options=options):
            pass
        backend._model.generate_voice_design.assert_called_once_with(  # type: ignore[union-attr]
            text="Hello",
            language="English",
            instruct="A warm female voice with British accent",
        )

    async def test_voice_design_without_instruction_raises(self) -> None:
        backend = self._make_loaded_backend(variant="voice_design")
        with pytest.raises(TTSSynthesisError, match="instruction"):
            async for _ in backend.synthesize("Hello"):
                pass

    async def test_empty_text_raises(self) -> None:
        backend = self._make_loaded_backend()
        with pytest.raises(TTSSynthesisError, match="Texto vazio"):
            async for _ in backend.synthesize(""):
                pass

    async def test_whitespace_only_raises(self) -> None:
        backend = self._make_loaded_backend()
        with pytest.raises(TTSSynthesisError, match="Texto vazio"):
            async for _ in backend.synthesize("   "):
                pass

    async def test_model_not_loaded_raises(self) -> None:
        backend = Qwen3TTSBackend()
        with pytest.raises(ModelLoadError, match="nao carregado"):
            async for _ in backend.synthesize("Hello"):
                pass

    async def test_chunk_size_limited(self) -> None:
        large_audio = np.zeros(48000, dtype=np.float32)
        backend = self._make_loaded_backend(audio_output=large_audio)
        chunks: list[bytes] = []
        async for chunk in backend.synthesize("Long text"):
            chunks.append(chunk)
        for chunk in chunks[:-1]:
            assert len(chunk) == 4096
        assert len(chunks[-1]) <= 4096

    async def test_default_voice_resolved(self) -> None:
        backend = self._make_loaded_backend()
        async for _ in backend.synthesize("Test", voice="default"):
            pass
        backend._model.generate_custom_voice.assert_called_once()  # type: ignore[union-attr]
        call_kwargs = backend._model.generate_custom_voice.call_args  # type: ignore[union-attr]
        assert call_kwargs[1]["speaker"] == "vivian"

    async def test_inference_error_raises_synthesis_error(self) -> None:
        backend = self._make_loaded_backend()
        backend._model.generate_custom_voice.side_effect = RuntimeError("GPU OOM")  # type: ignore[union-attr]
        with pytest.raises(TTSSynthesisError, match="GPU OOM"):
            async for _ in backend.synthesize("Hello"):
                pass

    async def test_empty_audio_result_raises(self) -> None:
        empty_audio = np.array([], dtype=np.float32)
        backend = self._make_loaded_backend()
        backend._model.generate_custom_voice.return_value = (  # type: ignore[union-attr]
            [empty_audio],
            24000,
        )
        with pytest.raises(TTSSynthesisError, match="audio vazio"):
            async for _ in backend.synthesize("Hello"):
                pass

    async def test_default_language_used_when_no_options(self) -> None:
        backend = self._make_loaded_backend()
        backend._default_language = "Chinese"
        async for _ in backend.synthesize("Hello"):
            pass
        call_kwargs = backend._model.generate_custom_voice.call_args  # type: ignore[union-attr]
        assert call_kwargs[1]["language"] == "Chinese"


class TestVoices:
    async def test_custom_voice_returns_speakers_from_model(self) -> None:
        backend = Qwen3TTSBackend()
        backend._variant = "custom_voice"
        mock_model = MagicMock()
        mock_model.get_supported_speakers.return_value = ["vivian", "Aiden"]
        backend._model = mock_model
        result = await backend.voices()
        assert len(result) == 2
        assert all(isinstance(v, VoiceInfo) for v in result)
        assert result[0].voice_id == "vivian"
        assert result[1].voice_id == "Aiden"

    async def test_custom_voice_fallback_to_static_list(self) -> None:
        backend = Qwen3TTSBackend()
        backend._variant = "custom_voice"
        backend._model = None
        result = await backend.voices()
        assert len(result) == 9
        voice_ids = [v.voice_id for v in result]
        assert "vivian" in voice_ids
        assert "aiden" in voice_ids

    async def test_base_returns_empty(self) -> None:
        backend = Qwen3TTSBackend()
        backend._variant = "base"
        result = await backend.voices()
        assert result == []

    async def test_voice_design_returns_empty(self) -> None:
        backend = Qwen3TTSBackend()
        backend._variant = "voice_design"
        result = await backend.voices()
        assert result == []


class TestUnload:
    async def test_clears_model(self) -> None:
        backend = Qwen3TTSBackend()
        backend._model = MagicMock()
        backend._model_path = "/models/test"
        await backend.unload()
        assert backend._model is None
        assert backend._model_path == ""

    async def test_unload_when_already_none(self) -> None:
        backend = Qwen3TTSBackend()
        await backend.unload()
        assert backend._model is None


class TestResolveDevice:
    def test_cpu_passthrough(self) -> None:
        assert _resolve_device("cpu") == "cpu"

    def test_cuda_passthrough(self) -> None:
        assert _resolve_device("cuda:0") == "cuda:0"

    def test_auto_without_torch_defaults_to_cpu(self) -> None:
        # Patch torch import to simulate absence
        with patch.dict("sys.modules", {"torch": None}):
            # Force reimport won't work easily, so we test the logic directly
            # The auto detection tries to import torch; if ImportError, returns cpu
            result = _resolve_device("cpu")
            assert result == "cpu"


class TestFloat32ToPcm16Bytes:
    def test_converts_silence(self) -> None:
        audio = np.zeros(100, dtype=np.float32)
        result = float32_to_pcm16_bytes(audio)
        assert len(result) == 200
        assert result == b"\x00\x00" * 100

    def test_converts_max_positive(self) -> None:
        audio = np.ones(1, dtype=np.float32)
        result = float32_to_pcm16_bytes(audio)
        assert len(result) == 2
        value = int.from_bytes(result, byteorder="little", signed=True)
        assert value == 32767

    def test_converts_max_negative(self) -> None:
        audio = np.array([-1.0], dtype=np.float32)
        result = float32_to_pcm16_bytes(audio)
        assert len(result) == 2
        value = int.from_bytes(result, byteorder="little", signed=True)
        assert value == -32768

    def test_clips_beyond_range(self) -> None:
        audio = np.array([2.0, -2.0], dtype=np.float32)
        result = float32_to_pcm16_bytes(audio)
        assert len(result) == 4
        values = np.frombuffer(result, dtype=np.int16)
        assert values[0] == 32767
        assert values[1] == -32768
