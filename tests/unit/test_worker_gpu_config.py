"""Tests for GPU configuration in worker startup.

Validates that CUDA allocator and inference optimizations are set
correctly at module level, before any torch import occurs.
Also validates torch inference configuration (grad_enabled, matmul precision).
"""

from __future__ import annotations

import importlib
import os
from unittest.mock import MagicMock, patch


class TestCudaExpandableSegments:
    """PYTORCH_CUDA_ALLOC_CONF is set with expandable_segments before torch import."""

    def test_stt_worker_sets_expandable_segments_when_empty(self) -> None:
        """When env var is empty, STT worker sets expandable_segments:True."""
        with patch.dict(os.environ, {"PYTORCH_CUDA_ALLOC_CONF": ""}, clear=False):
            # Force re-evaluation of module-level code
            import macaw.workers.stt.main as stt_main

            importlib.reload(stt_main)
            assert "expandable_segments:True" in os.environ["PYTORCH_CUDA_ALLOC_CONF"]

    def test_stt_worker_appends_when_existing_conf(self) -> None:
        """When env var has existing config, expandable_segments is appended."""
        with patch.dict(
            os.environ, {"PYTORCH_CUDA_ALLOC_CONF": "max_split_size_mb:512"}, clear=False
        ):
            import macaw.workers.stt.main as stt_main

            importlib.reload(stt_main)
            val = os.environ["PYTORCH_CUDA_ALLOC_CONF"]
            assert "max_split_size_mb:512" in val
            assert "expandable_segments:True" in val

    def test_stt_worker_does_not_duplicate_when_already_set(self) -> None:
        """When expandable_segments is already present, do not duplicate."""
        with patch.dict(
            os.environ,
            {"PYTORCH_CUDA_ALLOC_CONF": "expandable_segments:True"},
            clear=False,
        ):
            import macaw.workers.stt.main as stt_main

            importlib.reload(stt_main)
            val = os.environ["PYTORCH_CUDA_ALLOC_CONF"]
            assert val.count("expandable_segments") == 1

    def test_tts_worker_sets_expandable_segments_when_empty(self) -> None:
        """When env var is empty, TTS worker sets expandable_segments:True."""
        with patch.dict(os.environ, {"PYTORCH_CUDA_ALLOC_CONF": ""}, clear=False):
            import macaw.workers.tts.main as tts_main

            importlib.reload(tts_main)
            assert "expandable_segments:True" in os.environ["PYTORCH_CUDA_ALLOC_CONF"]

    def test_tts_worker_appends_when_existing_conf(self) -> None:
        """When env var has existing config, expandable_segments is appended."""
        with patch.dict(
            os.environ, {"PYTORCH_CUDA_ALLOC_CONF": "max_split_size_mb:512"}, clear=False
        ):
            import macaw.workers.tts.main as tts_main

            importlib.reload(tts_main)
            val = os.environ["PYTORCH_CUDA_ALLOC_CONF"]
            assert "max_split_size_mb:512" in val
            assert "expandable_segments:True" in val

    def test_tts_worker_does_not_duplicate_when_already_set(self) -> None:
        """When expandable_segments is already present, do not duplicate."""
        with patch.dict(
            os.environ,
            {"PYTORCH_CUDA_ALLOC_CONF": "expandable_segments:True"},
            clear=False,
        ):
            import macaw.workers.tts.main as tts_main

            importlib.reload(tts_main)
            val = os.environ["PYTORCH_CUDA_ALLOC_CONF"]
            assert val.count("expandable_segments") == 1

    def test_stt_worker_sets_when_env_not_defined(self) -> None:
        """When PYTORCH_CUDA_ALLOC_CONF is not defined at all."""
        env = os.environ.copy()
        env.pop("PYTORCH_CUDA_ALLOC_CONF", None)
        with patch.dict(os.environ, env, clear=True):
            import macaw.workers.stt.main as stt_main

            importlib.reload(stt_main)
            assert os.environ.get("PYTORCH_CUDA_ALLOC_CONF") == "expandable_segments:True"

    def test_tts_worker_sets_when_env_not_defined(self) -> None:
        """When PYTORCH_CUDA_ALLOC_CONF is not defined at all."""
        env = os.environ.copy()
        env.pop("PYTORCH_CUDA_ALLOC_CONF", None)
        with patch.dict(os.environ, env, clear=True):
            import macaw.workers.tts.main as tts_main

            importlib.reload(tts_main)
            assert os.environ.get("PYTORCH_CUDA_ALLOC_CONF") == "expandable_segments:True"


class TestConfigureTorchInference:
    """_configure_torch_inference() sets grad_enabled=False and matmul precision."""

    def test_stt_configure_calls_torch_settings(self) -> None:
        """STT _configure_torch_inference calls set_grad_enabled and set_float32_matmul_precision."""
        mock_torch = MagicMock()
        with patch.dict("sys.modules", {"torch": mock_torch}):
            from macaw.workers.stt.main import _configure_torch_inference

            _configure_torch_inference()
            mock_torch.set_grad_enabled.assert_called_once_with(False)
            mock_torch.set_float32_matmul_precision.assert_called_once_with("high")

    def test_tts_configure_calls_torch_settings(self) -> None:
        """TTS _configure_torch_inference calls set_grad_enabled and set_float32_matmul_precision."""
        mock_torch = MagicMock()
        with patch.dict("sys.modules", {"torch": mock_torch}):
            from macaw.workers.tts.main import _configure_torch_inference

            _configure_torch_inference()
            mock_torch.set_grad_enabled.assert_called_once_with(False)
            mock_torch.set_float32_matmul_precision.assert_called_once_with("high")

    def test_stt_configure_survives_missing_torch(self) -> None:
        """_configure_torch_inference does not raise when torch is absent."""
        import builtins

        original_import = builtins.__import__

        def _import_no_torch(name: str, *args: object, **kwargs: object) -> object:
            if name == "torch":
                raise ImportError("no torch")
            return original_import(name, *args, **kwargs)

        with patch("builtins.__import__", side_effect=_import_no_torch):
            from macaw.workers.stt.main import _configure_torch_inference

            # Should not raise
            _configure_torch_inference()


class TestSTTWarmup:
    """STT _warmup_backend runs configurable warmup passes with RTFx measurement."""

    async def test_warmup_runs_correct_number_of_steps(self) -> None:
        from macaw.workers.stt.main import _warmup_backend

        mock_backend = MagicMock()
        mock_backend.transcribe_file = MagicMock(return_value=MagicMock())

        # Make transcribe_file a coroutine
        async def _fake_transcribe(audio: bytes, **kwargs: object) -> object:
            return MagicMock()

        mock_backend.transcribe_file = _fake_transcribe

        call_count = 0
        original = mock_backend.transcribe_file

        async def _counting_transcribe(audio: bytes, **kwargs: object) -> object:
            nonlocal call_count
            call_count += 1
            return await original(audio, **kwargs)

        mock_backend.transcribe_file = _counting_transcribe

        await _warmup_backend(mock_backend, warmup_steps=3)
        assert call_count == 3

    async def test_warmup_zero_steps_skips(self) -> None:
        from macaw.workers.stt.main import _warmup_backend

        mock_backend = MagicMock()
        call_count = 0

        async def _counting_transcribe(audio: bytes, **kwargs: object) -> object:
            nonlocal call_count
            call_count += 1
            return MagicMock()

        mock_backend.transcribe_file = _counting_transcribe

        await _warmup_backend(mock_backend, warmup_steps=0)
        assert call_count == 0

    async def test_warmup_uses_varied_audio_lengths(self) -> None:
        from macaw.workers.stt.main import _warmup_backend

        audio_lengths: list[int] = []

        async def _tracking_transcribe(audio: bytes, **kwargs: object) -> object:
            audio_lengths.append(len(audio))
            return MagicMock()

        mock_backend = MagicMock()
        mock_backend.transcribe_file = _tracking_transcribe

        await _warmup_backend(mock_backend, warmup_steps=3)

        # Expected: 1s (32000 bytes), 3s (96000 bytes), 5s (160000 bytes)
        assert audio_lengths == [32000, 96000, 160000]

    async def test_warmup_stops_on_error(self) -> None:
        from macaw.workers.stt.main import _warmup_backend

        call_count = 0

        async def _failing_transcribe(audio: bytes, **kwargs: object) -> object:
            nonlocal call_count
            call_count += 1
            if call_count == 2:
                msg = "GPU OOM"
                raise RuntimeError(msg)
            return MagicMock()

        mock_backend = MagicMock()
        mock_backend.transcribe_file = _failing_transcribe

        await _warmup_backend(mock_backend, warmup_steps=3)
        # Should stop at step 2 (the failing one)
        assert call_count == 2

    async def test_warmup_single_step(self) -> None:
        from macaw.workers.stt.main import _warmup_backend

        call_count = 0

        async def _counting_transcribe(audio: bytes, **kwargs: object) -> object:
            nonlocal call_count
            call_count += 1
            return MagicMock()

        mock_backend = MagicMock()
        mock_backend.transcribe_file = _counting_transcribe

        await _warmup_backend(mock_backend, warmup_steps=1)
        assert call_count == 1


class TestTTSWarmup:
    """TTS _warmup_backend runs configurable warmup passes with RTFx measurement."""

    async def test_warmup_runs_correct_number_of_steps(self) -> None:
        from macaw.workers.tts.main import _warmup_backend

        call_count = 0

        async def _fake_synthesize(text: str, **kwargs: object):  # type: ignore[no-untyped-def]
            nonlocal call_count
            call_count += 1
            yield b"\x00" * 4800  # Some audio bytes

        mock_backend = MagicMock()
        mock_backend.synthesize = _fake_synthesize

        await _warmup_backend(mock_backend, warmup_steps=3)
        assert call_count == 3

    async def test_warmup_zero_steps_skips(self) -> None:
        from macaw.workers.tts.main import _warmup_backend

        call_count = 0

        async def _fake_synthesize(text: str, **kwargs: object):  # type: ignore[no-untyped-def]
            nonlocal call_count
            call_count += 1
            yield b"\x00" * 100

        mock_backend = MagicMock()
        mock_backend.synthesize = _fake_synthesize

        await _warmup_backend(mock_backend, warmup_steps=0)
        assert call_count == 0

    async def test_warmup_uses_varied_texts(self) -> None:
        from macaw.workers.tts.main import _warmup_backend

        texts: list[str] = []

        async def _tracking_synthesize(text: str, **kwargs: object):  # type: ignore[no-untyped-def]
            texts.append(text)
            yield b"\x00" * 100

        mock_backend = MagicMock()
        mock_backend.synthesize = _tracking_synthesize

        await _warmup_backend(mock_backend, warmup_steps=3)

        assert len(texts) == 3
        # Texts should be different lengths
        assert len(texts[0]) < len(texts[1]) < len(texts[2])

    async def test_warmup_stops_on_error(self) -> None:
        from macaw.workers.tts.main import _warmup_backend

        call_count = 0

        async def _failing_synthesize(text: str, **kwargs: object):  # type: ignore[no-untyped-def]
            nonlocal call_count
            call_count += 1
            if call_count == 2:
                msg = "Synthesis failed"
                raise RuntimeError(msg)
            yield b"\x00" * 100

        mock_backend = MagicMock()
        mock_backend.synthesize = _failing_synthesize

        await _warmup_backend(mock_backend, warmup_steps=3)
        assert call_count == 2
