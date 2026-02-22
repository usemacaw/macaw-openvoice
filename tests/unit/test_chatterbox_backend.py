"""Tests for ChatterboxTurboBackend.

Uses mocks for chatterbox — does not require chatterbox-tts installed.
Follows the same pattern as KokoroBackend / Qwen3TTSBackend tests.
"""

from __future__ import annotations

from unittest.mock import MagicMock

import numpy as np
import pytest

from macaw.exceptions import ModelLoadError, TTSEngineError, TTSSynthesisError


def _make_mock_wav(duration_s: float = 0.1, sample_rate: int = 24000) -> MagicMock:
    """Create a mock torch.Tensor representing audio output."""
    num_samples = int(duration_s * sample_rate)
    audio_array = np.random.default_rng(42).uniform(-1.0, 1.0, num_samples).astype(np.float32)

    mock_tensor = MagicMock()
    mock_tensor.cpu.return_value = mock_tensor
    mock_tensor.numpy.return_value = audio_array
    return mock_tensor


class TestLoad:
    async def test_load_raises_when_package_missing(self) -> None:
        import macaw.workers.tts.chatterbox as mod

        original = mod._ChatterboxTurboTTS
        mod._ChatterboxTurboTTS = None  # type: ignore[assignment]
        try:
            from macaw.workers.tts.chatterbox import ChatterboxTurboBackend

            backend = ChatterboxTurboBackend()
            with pytest.raises(ModelLoadError, match="chatterbox-tts is not installed"):
                await backend.load("/models/test", {})
        finally:
            mod._ChatterboxTurboTTS = original  # type: ignore[assignment]

    async def test_load_configures_device(self) -> None:
        import macaw.workers.tts.chatterbox as mod

        mock_model = MagicMock()
        mock_model.sr = 24000
        mock_cls = MagicMock()
        mock_cls.from_pretrained = MagicMock(return_value=mock_model)

        original = mod._ChatterboxTurboTTS
        mod._ChatterboxTurboTTS = mock_cls  # type: ignore[assignment]
        try:
            from macaw.workers.tts.chatterbox import ChatterboxTurboBackend

            backend = ChatterboxTurboBackend()
            await backend.load("/models/chatterbox", {"device": "cpu"})

            assert backend._model is not None
            assert backend._sample_rate == 24000
            mock_cls.from_pretrained.assert_called_once_with(device="cpu")
        finally:
            mod._ChatterboxTurboTTS = original  # type: ignore[assignment]

    async def test_load_failure_raises_model_load_error(self) -> None:
        import macaw.workers.tts.chatterbox as mod

        mock_cls = MagicMock()
        mock_cls.from_pretrained = MagicMock(side_effect=RuntimeError("GPU not found"))

        original = mod._ChatterboxTurboTTS
        mod._ChatterboxTurboTTS = mock_cls  # type: ignore[assignment]
        try:
            from macaw.workers.tts.chatterbox import ChatterboxTurboBackend

            backend = ChatterboxTurboBackend()
            with pytest.raises(ModelLoadError, match="GPU not found"):
                await backend.load("/models/test", {})
        finally:
            mod._ChatterboxTurboTTS = original  # type: ignore[assignment]


class TestCapabilities:
    async def test_reports_voice_cloning(self) -> None:
        from macaw.workers.tts.chatterbox import ChatterboxTurboBackend

        backend = ChatterboxTurboBackend()
        caps = await backend.capabilities()
        assert caps.supports_voice_cloning is True

    async def test_no_streaming(self) -> None:
        from macaw.workers.tts.chatterbox import ChatterboxTurboBackend

        backend = ChatterboxTurboBackend()
        caps = await backend.capabilities()
        assert caps.supports_streaming is False


class TestSynthesize:
    async def test_yields_pcm_chunks(self) -> None:
        from macaw.workers.tts.chatterbox import ChatterboxTurboBackend

        mock_wav = _make_mock_wav(0.5)
        mock_model = MagicMock()
        mock_model.generate = MagicMock(return_value=mock_wav)

        backend = ChatterboxTurboBackend()
        backend._model = mock_model
        backend._model_path = "/models/test"
        backend._sample_rate = 24000

        chunks: list[bytes] = []
        async for chunk in backend.synthesize("Hello world"):
            chunks.append(chunk)

        assert len(chunks) > 0
        total_bytes = sum(len(c) for c in chunks)
        assert total_bytes > 0
        # 0.5s at 24kHz, 16-bit = 24000 samples * 2 bytes = 24000 bytes
        expected_bytes = int(0.5 * 24000) * 2
        assert total_bytes == expected_bytes

    async def test_with_voice_cloning_ref_audio(self) -> None:
        from macaw.workers.tts.chatterbox import ChatterboxTurboBackend

        mock_wav = _make_mock_wav(0.1)
        mock_model = MagicMock()
        mock_model.generate = MagicMock(return_value=mock_wav)

        backend = ChatterboxTurboBackend()
        backend._model = mock_model
        backend._model_path = "/models/test"
        backend._sample_rate = 24000

        # Create fake WAV ref audio
        ref_audio_bytes = b"RIFF" + b"\x00" * 100  # Fake WAV bytes

        chunks = []
        async for chunk in backend.synthesize(
            "Hello",
            options={"ref_audio": ref_audio_bytes},
        ):
            chunks.append(chunk)

        assert len(chunks) > 0
        # Verify generate was called with audio_prompt_path (temp file)
        mock_model.generate.assert_called_once()
        call_kwargs = mock_model.generate.call_args[1]
        assert call_kwargs["audio_prompt_path"] is not None

    async def test_with_sampling_params(self) -> None:
        from macaw.workers.tts.chatterbox import ChatterboxTurboBackend

        mock_wav = _make_mock_wav(0.1)
        mock_model = MagicMock()
        mock_model.generate = MagicMock(return_value=mock_wav)

        backend = ChatterboxTurboBackend()
        backend._model = mock_model
        backend._model_path = "/models/test"
        backend._sample_rate = 24000

        chunks = []
        async for chunk in backend.synthesize(
            "Test",
            options={
                "temperature": 0.5,
                "top_p": 0.9,
                "top_k": 30,
            },
        ):
            chunks.append(chunk)

        assert len(chunks) > 0
        call_kwargs = mock_model.generate.call_args[1]
        assert call_kwargs["temperature"] == 0.5
        assert call_kwargs["top_p"] == 0.9
        assert call_kwargs["top_k"] == 30

    async def test_handles_speed_adjustment(self) -> None:
        from macaw.workers.tts.chatterbox import ChatterboxTurboBackend

        mock_wav = _make_mock_wav(0.5)
        mock_model = MagicMock()
        mock_model.generate = MagicMock(return_value=mock_wav)

        backend = ChatterboxTurboBackend()
        backend._model = mock_model
        backend._model_path = "/models/test"
        backend._sample_rate = 24000

        normal_chunks: list[bytes] = []
        async for chunk in backend.synthesize("Hello", speed=1.0):
            normal_chunks.append(chunk)
        normal_total = sum(len(c) for c in normal_chunks)

        fast_chunks: list[bytes] = []
        async for chunk in backend.synthesize("Hello", speed=2.0):
            fast_chunks.append(chunk)
        fast_total = sum(len(c) for c in fast_chunks)

        # 2x speed should produce ~half the audio
        assert fast_total < normal_total
        assert fast_total == pytest.approx(normal_total / 2, rel=0.1)

    async def test_empty_text_raises_error(self) -> None:
        from macaw.workers.tts.chatterbox import ChatterboxTurboBackend

        backend = ChatterboxTurboBackend()
        backend._model = MagicMock()
        backend._model_path = "/models/test"

        with pytest.raises(TTSSynthesisError, match="Empty text"):
            async for _ in backend.synthesize("   "):
                pass

    async def test_model_not_loaded_raises_error(self) -> None:
        from macaw.workers.tts.chatterbox import ChatterboxTurboBackend

        backend = ChatterboxTurboBackend()

        with pytest.raises(ModelLoadError, match="not loaded"):
            async for _ in backend.synthesize("Hello"):
                pass

    async def test_engine_error_wraps_exception(self) -> None:
        from macaw.workers.tts.chatterbox import ChatterboxTurboBackend

        mock_model = MagicMock()
        mock_model.generate = MagicMock(side_effect=RuntimeError("OOM"))

        backend = ChatterboxTurboBackend()
        backend._model = mock_model
        backend._model_path = "/models/test"
        backend._sample_rate = 24000

        with pytest.raises(TTSEngineError, match="OOM"):
            async for _ in backend.synthesize("Hello"):
                pass

    async def test_empty_audio_raises_engine_error(self) -> None:
        from macaw.workers.tts.chatterbox import ChatterboxTurboBackend

        # Return empty audio
        mock_tensor = MagicMock()
        mock_tensor.cpu.return_value = mock_tensor
        mock_tensor.numpy.return_value = np.array([], dtype=np.float32)

        mock_model = MagicMock()
        mock_model.generate = MagicMock(return_value=mock_tensor)

        backend = ChatterboxTurboBackend()
        backend._model = mock_model
        backend._model_path = "/models/test"
        backend._sample_rate = 24000

        with pytest.raises(TTSEngineError, match="empty audio"):
            async for _ in backend.synthesize("Hello"):
                pass


class TestVoices:
    async def test_returns_default(self) -> None:
        from macaw.workers.tts.chatterbox import ChatterboxTurboBackend

        backend = ChatterboxTurboBackend()
        voices = await backend.voices()

        assert len(voices) == 1
        assert voices[0].voice_id == "default"
        assert voices[0].language == "en"


class TestUnload:
    async def test_releases_gpu_memory(self) -> None:
        from macaw.workers.tts.chatterbox import ChatterboxTurboBackend

        backend = ChatterboxTurboBackend()
        backend._model = MagicMock()

        await backend.unload()

        assert backend._model is None
        assert backend._model_path == ""


class TestHealth:
    async def test_ok_when_loaded(self) -> None:
        from macaw.workers.tts.chatterbox import ChatterboxTurboBackend

        backend = ChatterboxTurboBackend()
        backend._model = MagicMock()

        health = await backend.health()
        assert health["status"] == "ok"

    async def test_not_loaded_when_model_none(self) -> None:
        from macaw.workers.tts.chatterbox import ChatterboxTurboBackend

        backend = ChatterboxTurboBackend()

        health = await backend.health()
        assert health["status"] == "not_loaded"


class TestApplySpeed:
    def test_speed_2x_halves_audio_length(self) -> None:
        from macaw.workers.tts.chatterbox import _apply_speed

        # 1 second of silence at 24kHz
        num_samples = 24000
        pcm_bytes = np.zeros(num_samples, dtype=np.int16).tobytes()

        result = _apply_speed(pcm_bytes, 2.0, 24000)

        # 2x speed should produce half the samples
        result_samples = len(result) // 2  # 2 bytes per int16 sample
        assert result_samples == pytest.approx(num_samples / 2, abs=1)

    def test_speed_1x_preserves_audio(self) -> None:
        from macaw.workers.tts.chatterbox import _apply_speed

        pcm_bytes = np.zeros(1000, dtype=np.int16).tobytes()

        result = _apply_speed(pcm_bytes, 1.0, 24000)

        assert len(result) == len(pcm_bytes)
