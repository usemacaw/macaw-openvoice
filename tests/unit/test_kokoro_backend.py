"""Testes para KokoroBackend.

Usa mocks para o modulo kokoro -- nao requer kokoro instalado.
Segue o mesmo padrao de test_wenet_backend.py.
"""

from __future__ import annotations

import os
from unittest.mock import MagicMock

import numpy as np
import pytest

from macaw._types import VoiceInfo
from macaw.exceptions import ModelLoadError, TTSSynthesisError
from macaw.workers.tts.kokoro import (
    KokoroBackend,
    _float32_to_pcm16_bytes,
    _resolve_device,
    _resolve_voice_path,
    _scan_voices_dir,
    _voice_id_to_gender,
    _voice_id_to_language,
)


def _make_mock_kokoro_lib(
    audio: np.ndarray | None = None,
) -> MagicMock:
    """Cria mock da biblioteca kokoro com KModel+KPipeline."""
    mock_lib = MagicMock()

    # KModel mock: model.to(device).eval() chainable
    mock_model = MagicMock()
    mock_model.to.return_value = mock_model
    mock_model.eval.return_value = mock_model
    mock_lib.KModel.return_value = mock_model

    # KPipeline mock: pipeline(text, voice=..., speed=...) -> generator
    mock_pipeline = MagicMock()
    if audio is None:
        audio = np.zeros(2400, dtype=np.float32)
    mock_pipeline.return_value = [(None, None, audio)]
    mock_lib.KPipeline.return_value = mock_pipeline

    return mock_lib


def _make_model_dir(tmp_path: object) -> str:
    """Cria diretorio de modelo fake com config.json e weights."""
    from pathlib import Path

    model_dir = Path(str(tmp_path)) / "kokoro-v1"
    model_dir.mkdir()
    (model_dir / "config.json").write_text("{}")
    (model_dir / "kokoro-v1_0.pth").write_bytes(b"fake_weights")

    voices_dir = model_dir / "voices"
    voices_dir.mkdir()
    (voices_dir / "af_heart.pt").write_bytes(b"voice_data")
    (voices_dir / "am_adam.pt").write_bytes(b"voice_data")
    (voices_dir / "pf_dora.pt").write_bytes(b"voice_data")

    return str(model_dir)


class TestHealth:
    async def test_ok_when_model_loaded(self) -> None:
        backend = KokoroBackend()
        backend._model = MagicMock()  # type: ignore[assignment]
        health = await backend.health()
        assert health["status"] == "ok"

    async def test_not_loaded_when_model_none(self) -> None:
        backend = KokoroBackend()
        health = await backend.health()
        assert health["status"] == "not_loaded"


class TestLoad:
    async def test_load_succeeds_with_mock(self, tmp_path: object) -> None:
        mock_kokoro = _make_mock_kokoro_lib()
        model_dir = _make_model_dir(tmp_path)

        import macaw.workers.tts.kokoro as kokoro_mod

        original = kokoro_mod.kokoro_lib
        kokoro_mod.kokoro_lib = mock_kokoro  # type: ignore[assignment]
        try:
            backend = KokoroBackend()
            await backend.load(model_dir, {"device": "cpu", "lang_code": "a"})
            assert backend._model is not None
            assert backend._pipeline is not None
            mock_kokoro.KModel.assert_called_once()
            mock_kokoro.KPipeline.assert_called_once()
        finally:
            kokoro_mod.kokoro_lib = original  # type: ignore[assignment]

    async def test_load_stores_model_path(self, tmp_path: object) -> None:
        mock_kokoro = _make_mock_kokoro_lib()
        model_dir = _make_model_dir(tmp_path)

        import macaw.workers.tts.kokoro as kokoro_mod

        original = kokoro_mod.kokoro_lib
        kokoro_mod.kokoro_lib = mock_kokoro  # type: ignore[assignment]
        try:
            backend = KokoroBackend()
            await backend.load(model_dir, {})
            assert backend._model_path == model_dir
        finally:
            kokoro_mod.kokoro_lib = original  # type: ignore[assignment]

    async def test_load_stores_voices_dir(self, tmp_path: object) -> None:
        mock_kokoro = _make_mock_kokoro_lib()
        model_dir = _make_model_dir(tmp_path)

        import macaw.workers.tts.kokoro as kokoro_mod

        original = kokoro_mod.kokoro_lib
        kokoro_mod.kokoro_lib = mock_kokoro  # type: ignore[assignment]
        try:
            backend = KokoroBackend()
            await backend.load(model_dir, {})
            assert backend._voices_dir == os.path.join(model_dir, "voices")
        finally:
            kokoro_mod.kokoro_lib = original  # type: ignore[assignment]

    async def test_load_stores_default_voice(self, tmp_path: object) -> None:
        mock_kokoro = _make_mock_kokoro_lib()
        model_dir = _make_model_dir(tmp_path)

        import macaw.workers.tts.kokoro as kokoro_mod

        original = kokoro_mod.kokoro_lib
        kokoro_mod.kokoro_lib = mock_kokoro  # type: ignore[assignment]
        try:
            backend = KokoroBackend()
            await backend.load(model_dir, {"default_voice": "am_adam"})
            assert backend._default_voice == "am_adam"
        finally:
            kokoro_mod.kokoro_lib = original  # type: ignore[assignment]

    async def test_load_without_library_raises_model_load_error(self) -> None:
        import macaw.workers.tts.kokoro as kokoro_mod

        original = kokoro_mod.kokoro_lib
        kokoro_mod.kokoro_lib = None  # type: ignore[assignment]
        try:
            backend = KokoroBackend()
            with pytest.raises(ModelLoadError, match="nao esta instalado"):
                await backend.load("/models/kokoro-v1", {})
        finally:
            kokoro_mod.kokoro_lib = original  # type: ignore[assignment]

    async def test_load_missing_config_json_raises(self, tmp_path: object) -> None:
        from pathlib import Path

        mock_kokoro = _make_mock_kokoro_lib()
        model_dir = Path(str(tmp_path)) / "bad-model"
        model_dir.mkdir()
        (model_dir / "model.pth").write_bytes(b"fake")
        # No config.json

        import macaw.workers.tts.kokoro as kokoro_mod

        original = kokoro_mod.kokoro_lib
        kokoro_mod.kokoro_lib = mock_kokoro  # type: ignore[assignment]
        try:
            backend = KokoroBackend()
            with pytest.raises(ModelLoadError, match=r"config\.json"):
                await backend.load(str(model_dir), {})
        finally:
            kokoro_mod.kokoro_lib = original  # type: ignore[assignment]

    async def test_load_missing_weights_raises(self, tmp_path: object) -> None:
        from pathlib import Path

        mock_kokoro = _make_mock_kokoro_lib()
        model_dir = Path(str(tmp_path)) / "bad-model"
        model_dir.mkdir()
        (model_dir / "config.json").write_text("{}")
        # No .pth file

        import macaw.workers.tts.kokoro as kokoro_mod

        original = kokoro_mod.kokoro_lib
        kokoro_mod.kokoro_lib = mock_kokoro  # type: ignore[assignment]
        try:
            backend = KokoroBackend()
            with pytest.raises(ModelLoadError, match=r"\.pth"):
                await backend.load(str(model_dir), {})
        finally:
            kokoro_mod.kokoro_lib = original  # type: ignore[assignment]

    async def test_load_failure_raises_model_load_error(self, tmp_path: object) -> None:
        model_dir = _make_model_dir(tmp_path)

        import macaw.workers.tts.kokoro as kokoro_mod

        original = kokoro_mod.kokoro_lib
        mock_kokoro = MagicMock()
        mock_kokoro.KModel.side_effect = RuntimeError("Model file corrupted")
        kokoro_mod.kokoro_lib = mock_kokoro  # type: ignore[assignment]
        try:
            backend = KokoroBackend()
            with pytest.raises(ModelLoadError, match="Model file corrupted"):
                await backend.load(model_dir, {})
        finally:
            kokoro_mod.kokoro_lib = original  # type: ignore[assignment]

    async def test_load_with_auto_device(self, tmp_path: object) -> None:
        mock_kokoro = _make_mock_kokoro_lib()
        model_dir = _make_model_dir(tmp_path)

        import macaw.workers.tts.kokoro as kokoro_mod

        original = kokoro_mod.kokoro_lib
        kokoro_mod.kokoro_lib = mock_kokoro  # type: ignore[assignment]
        try:
            backend = KokoroBackend()
            await backend.load(model_dir, {"device": "auto"})
            # "auto" resolved to "cpu" — KPipeline receives device="cpu"
            call_kwargs = mock_kokoro.KPipeline.call_args
            assert call_kwargs[1]["device"] == "cpu"
        finally:
            kokoro_mod.kokoro_lib = original  # type: ignore[assignment]

    async def test_load_passes_lang_code(self, tmp_path: object) -> None:
        mock_kokoro = _make_mock_kokoro_lib()
        model_dir = _make_model_dir(tmp_path)

        import macaw.workers.tts.kokoro as kokoro_mod

        original = kokoro_mod.kokoro_lib
        kokoro_mod.kokoro_lib = mock_kokoro  # type: ignore[assignment]
        try:
            backend = KokoroBackend()
            await backend.load(model_dir, {"lang_code": "p"})
            call_kwargs = mock_kokoro.KPipeline.call_args
            assert call_kwargs[1]["lang_code"] == "p"
        finally:
            kokoro_mod.kokoro_lib = original  # type: ignore[assignment]


class TestSynthesize:
    def _make_loaded_backend(
        self,
        audio_output: np.ndarray | None = None,
        voices_dir: str = "",
    ) -> KokoroBackend:
        """Cria backend com pipeline mock carregado."""
        backend = KokoroBackend()
        mock_model = MagicMock()
        mock_pipeline = MagicMock()

        if audio_output is None:
            audio_output = np.zeros(2400, dtype=np.float32)

        mock_pipeline.return_value = [(None, None, audio_output)]
        backend._model = mock_model  # type: ignore[assignment]
        backend._pipeline = mock_pipeline  # type: ignore[assignment]
        backend._model_path = "/models/kokoro-v1"
        backend._voices_dir = voices_dir
        backend._default_voice = "af_heart"
        return backend

    async def test_returns_audio_chunks(self) -> None:
        backend = self._make_loaded_backend()
        chunks: list[bytes] = []
        async for chunk in backend.synthesize("Hello world"):
            chunks.append(chunk)
        assert len(chunks) > 0
        # Total bytes should equal 2400 samples * 2 bytes/sample
        total_bytes = sum(len(c) for c in chunks)
        assert total_bytes == 2400 * 2

    async def test_chunk_size_limited(self) -> None:
        large_audio = np.zeros(48000, dtype=np.float32)  # 2 seconds at 24kHz
        backend = self._make_loaded_backend(audio_output=large_audio)
        chunks: list[bytes] = []
        async for chunk in backend.synthesize("Long text"):
            chunks.append(chunk)

        # Every chunk except possibly the last should be exactly 4096 bytes
        for chunk in chunks[:-1]:
            assert len(chunk) == 4096

        # Last chunk may be smaller
        assert len(chunks[-1]) <= 4096

    async def test_empty_text_raises_synthesis_error(self) -> None:
        backend = self._make_loaded_backend()
        with pytest.raises(TTSSynthesisError, match="Texto vazio"):
            async for _ in backend.synthesize(""):
                pass

    async def test_whitespace_only_raises_synthesis_error(self) -> None:
        backend = self._make_loaded_backend()
        with pytest.raises(TTSSynthesisError, match="Texto vazio"):
            async for _ in backend.synthesize("   "):
                pass

    async def test_model_not_loaded_raises_error(self) -> None:
        backend = KokoroBackend()
        with pytest.raises(ModelLoadError, match="nao carregado"):
            async for _ in backend.synthesize("Hello"):
                pass

    async def test_passes_voice_to_pipeline(self, tmp_path: object) -> None:
        from pathlib import Path

        # Create voices dir with the voice file
        voices_dir = Path(str(tmp_path)) / "voices"
        voices_dir.mkdir()
        (voices_dir / "af_heart.pt").write_bytes(b"data")

        backend = self._make_loaded_backend(voices_dir=str(voices_dir))
        async for _ in backend.synthesize("Test", voice="af_heart"):
            pass
        mock_pipeline = backend._pipeline
        assert mock_pipeline is not None
        call_kwargs = mock_pipeline.call_args  # type: ignore[union-attr]
        assert call_kwargs[1]["voice"] == str(voices_dir / "af_heart.pt")

    async def test_passes_speed_parameter(self) -> None:
        backend = self._make_loaded_backend()
        async for _ in backend.synthesize("Test", speed=1.5):
            pass
        mock_pipeline = backend._pipeline
        assert mock_pipeline is not None
        call_kwargs = mock_pipeline.call_args  # type: ignore[union-attr]
        assert call_kwargs[1]["speed"] == 1.5

    async def test_default_voice_resolved(self) -> None:
        backend = self._make_loaded_backend()
        async for _ in backend.synthesize("Test", voice="default"):
            pass
        mock_pipeline = backend._pipeline
        assert mock_pipeline is not None
        call_args = mock_pipeline.call_args  # type: ignore[union-attr]
        # "default" should resolve to "af_heart" (the default_voice)
        assert call_args[1]["voice"] == "af_heart"

    async def test_inference_error_raises_synthesis_error(self) -> None:
        backend = KokoroBackend()
        mock_model = MagicMock()
        mock_pipeline = MagicMock()
        mock_pipeline.side_effect = RuntimeError("GPU OOM")
        backend._model = mock_model  # type: ignore[assignment]
        backend._pipeline = mock_pipeline  # type: ignore[assignment]
        backend._model_path = "/models/kokoro-v1"

        with pytest.raises(TTSSynthesisError, match="GPU OOM"):
            async for _ in backend.synthesize("Hello"):
                pass

    async def test_empty_audio_result_raises_synthesis_error(self) -> None:
        backend = KokoroBackend()
        mock_model = MagicMock()
        mock_pipeline = MagicMock()
        # Pipeline returns generator with no audio
        mock_pipeline.return_value = [(None, None, np.array([], dtype=np.float32))]
        backend._model = mock_model  # type: ignore[assignment]
        backend._pipeline = mock_pipeline  # type: ignore[assignment]
        backend._model_path = "/models/kokoro-v1"

        with pytest.raises(TTSSynthesisError, match="audio vazio"):
            async for _ in backend.synthesize("Hello"):
                pass

    async def test_multi_chunk_pipeline_result(self) -> None:
        """Pipeline pode retornar multiplas tuplas (frases longas)."""
        backend = KokoroBackend()
        mock_model = MagicMock()
        mock_pipeline = MagicMock()
        audio1 = np.zeros(1200, dtype=np.float32)
        audio2 = np.zeros(1200, dtype=np.float32)
        mock_pipeline.return_value = [
            (None, None, audio1),
            (None, None, audio2),
        ]
        backend._model = mock_model  # type: ignore[assignment]
        backend._pipeline = mock_pipeline  # type: ignore[assignment]
        backend._model_path = "/models/kokoro-v1"

        chunks: list[bytes] = []
        async for chunk in backend.synthesize("Long text with multiple sentences"):
            chunks.append(chunk)

        total_bytes = sum(len(c) for c in chunks)
        assert total_bytes == (1200 + 1200) * 2


class TestVoices:
    async def test_returns_voice_info_list_with_dir(self, tmp_path: object) -> None:
        from pathlib import Path

        voices_dir = Path(str(tmp_path)) / "voices"
        voices_dir.mkdir()
        (voices_dir / "af_heart.pt").write_bytes(b"v")
        (voices_dir / "am_adam.pt").write_bytes(b"v")

        backend = KokoroBackend()
        backend._voices_dir = str(voices_dir)
        result = await backend.voices()
        assert isinstance(result, list)
        assert len(result) == 2

    async def test_all_items_are_voice_info(self, tmp_path: object) -> None:
        from pathlib import Path

        voices_dir = Path(str(tmp_path)) / "voices"
        voices_dir.mkdir()
        (voices_dir / "af_heart.pt").write_bytes(b"v")

        backend = KokoroBackend()
        backend._voices_dir = str(voices_dir)
        result = await backend.voices()
        for voice in result:
            assert isinstance(voice, VoiceInfo)
            assert voice.voice_id
            assert voice.name
            assert voice.language

    async def test_fallback_when_no_voices_dir(self) -> None:
        backend = KokoroBackend()
        backend._default_voice = "af_heart"
        result = await backend.voices()
        assert len(result) == 1
        assert result[0].voice_id == "af_heart"

    async def test_voice_language_detection(self, tmp_path: object) -> None:
        from pathlib import Path

        voices_dir = Path(str(tmp_path)) / "voices"
        voices_dir.mkdir()
        (voices_dir / "af_heart.pt").write_bytes(b"v")  # en
        (voices_dir / "pf_dora.pt").write_bytes(b"v")  # pt
        (voices_dir / "jf_alpha.pt").write_bytes(b"v")  # ja

        backend = KokoroBackend()
        backend._voices_dir = str(voices_dir)
        result = await backend.voices()

        voice_map = {v.voice_id: v for v in result}
        assert voice_map["af_heart"].language == "en"
        assert voice_map["pf_dora"].language == "pt"
        assert voice_map["jf_alpha"].language == "ja"

    async def test_voice_gender_detection(self, tmp_path: object) -> None:
        from pathlib import Path

        voices_dir = Path(str(tmp_path)) / "voices"
        voices_dir.mkdir()
        (voices_dir / "af_heart.pt").write_bytes(b"v")
        (voices_dir / "am_adam.pt").write_bytes(b"v")

        backend = KokoroBackend()
        backend._voices_dir = str(voices_dir)
        result = await backend.voices()

        voice_map = {v.voice_id: v for v in result}
        assert voice_map["af_heart"].gender == "female"
        assert voice_map["am_adam"].gender == "male"

    async def test_ignores_non_pt_files(self, tmp_path: object) -> None:
        from pathlib import Path

        voices_dir = Path(str(tmp_path)) / "voices"
        voices_dir.mkdir()
        (voices_dir / "af_heart.pt").write_bytes(b"v")
        (voices_dir / "readme.txt").write_text("not a voice")

        backend = KokoroBackend()
        backend._voices_dir = str(voices_dir)
        result = await backend.voices()
        assert len(result) == 1


class TestUnload:
    async def test_clears_model_and_pipeline(self) -> None:
        backend = KokoroBackend()
        backend._model = MagicMock()  # type: ignore[assignment]
        backend._pipeline = MagicMock()  # type: ignore[assignment]
        backend._model_path = "/models/test"
        backend._voices_dir = "/models/test/voices"
        await backend.unload()
        assert backend._model is None
        assert backend._pipeline is None
        assert backend._model_path == ""
        assert backend._voices_dir == ""

    async def test_unload_when_already_none(self) -> None:
        backend = KokoroBackend()
        await backend.unload()
        assert backend._model is None
        assert backend._pipeline is None


class TestResolveDevice:
    def test_auto_defaults_to_cpu(self) -> None:
        assert _resolve_device("auto") == "cpu"

    def test_cpu_passthrough(self) -> None:
        assert _resolve_device("cpu") == "cpu"

    def test_cuda_passthrough(self) -> None:
        assert _resolve_device("cuda") == "cuda"

    def test_cuda_with_id(self) -> None:
        assert _resolve_device("cuda:0") == "cuda:0"


class TestResolveVoicePath:
    def test_default_resolves_to_default_voice(self) -> None:
        result = _resolve_voice_path("default", "", "af_heart")
        assert result == "af_heart"

    def test_resolves_name_to_path(self, tmp_path: object) -> None:
        from pathlib import Path

        voices_dir = Path(str(tmp_path)) / "voices"
        voices_dir.mkdir()
        (voices_dir / "af_heart.pt").write_bytes(b"v")

        result = _resolve_voice_path("af_heart", str(voices_dir), "af_heart")
        assert result == str(voices_dir / "af_heart.pt")

    def test_absolute_path_passthrough(self) -> None:
        result = _resolve_voice_path("/abs/path/voice.pt", "/voices", "af_heart")
        assert result == "/abs/path/voice.pt"

    def test_pt_extension_passthrough(self) -> None:
        result = _resolve_voice_path("voice.pt", "/voices", "af_heart")
        assert result == "voice.pt"

    def test_fallback_to_name_when_file_missing(self) -> None:
        result = _resolve_voice_path("nonexistent", "/nonexistent/dir", "af_heart")
        assert result == "nonexistent"

    def test_no_voices_dir_returns_name(self) -> None:
        result = _resolve_voice_path("af_heart", "", "af_heart")
        assert result == "af_heart"


class TestVoiceIdToLanguage:
    def test_american_english(self) -> None:
        assert _voice_id_to_language("af_heart") == "en"

    def test_british_english(self) -> None:
        assert _voice_id_to_language("bf_alice") == "en"

    def test_portuguese(self) -> None:
        assert _voice_id_to_language("pf_dora") == "pt"

    def test_japanese(self) -> None:
        assert _voice_id_to_language("jf_alpha") == "ja"

    def test_chinese(self) -> None:
        assert _voice_id_to_language("zf_xiaobei") == "zh"

    def test_unknown_prefix_defaults_en(self) -> None:
        assert _voice_id_to_language("xf_unknown") == "en"

    def test_empty_defaults_en(self) -> None:
        assert _voice_id_to_language("") == "en"


class TestVoiceIdToGender:
    def test_female(self) -> None:
        assert _voice_id_to_gender("af_heart") == "female"

    def test_male(self) -> None:
        assert _voice_id_to_gender("am_adam") == "male"

    def test_short_id_returns_none(self) -> None:
        assert _voice_id_to_gender("a") is None

    def test_unknown_gender_returns_none(self) -> None:
        assert _voice_id_to_gender("ax_unknown") is None


class TestScanVoicesDir:
    def test_scans_pt_files(self, tmp_path: object) -> None:
        from pathlib import Path

        voices_dir = Path(str(tmp_path)) / "voices"
        voices_dir.mkdir()
        (voices_dir / "af_heart.pt").write_bytes(b"v")
        (voices_dir / "am_adam.pt").write_bytes(b"v")
        (voices_dir / "pf_dora.pt").write_bytes(b"v")

        result = _scan_voices_dir(str(voices_dir))
        assert len(result) == 3

        voice_ids = [v.voice_id for v in result]
        assert voice_ids == ["af_heart", "am_adam", "pf_dora"]  # sorted

    def test_ignores_non_pt(self, tmp_path: object) -> None:
        from pathlib import Path

        voices_dir = Path(str(tmp_path)) / "voices"
        voices_dir.mkdir()
        (voices_dir / "af_heart.pt").write_bytes(b"v")
        (voices_dir / "readme.txt").write_text("ignore")
        (voices_dir / "model.bin").write_bytes(b"ignore")

        result = _scan_voices_dir(str(voices_dir))
        assert len(result) == 1


class TestFloat32ToPcm16Bytes:
    def test_converts_silence(self) -> None:
        audio = np.zeros(100, dtype=np.float32)
        result = _float32_to_pcm16_bytes(audio)
        assert len(result) == 200  # 100 samples * 2 bytes/sample
        assert result == b"\x00\x00" * 100

    def test_converts_max_positive(self) -> None:
        audio = np.ones(1, dtype=np.float32)
        result = _float32_to_pcm16_bytes(audio)
        assert len(result) == 2
        value = int.from_bytes(result, byteorder="little", signed=True)
        assert value == 32767

    def test_converts_max_negative(self) -> None:
        audio = np.array([-1.0], dtype=np.float32)
        result = _float32_to_pcm16_bytes(audio)
        assert len(result) == 2
        value = int.from_bytes(result, byteorder="little", signed=True)
        assert value == -32768

    def test_clips_beyond_range(self) -> None:
        audio = np.array([2.0, -2.0], dtype=np.float32)
        result = _float32_to_pcm16_bytes(audio)
        assert len(result) == 4
        values = np.frombuffer(result, dtype=np.int16)
        assert values[0] == 32767
        assert values[1] == -32768
