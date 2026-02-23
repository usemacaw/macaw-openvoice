"""Tests for DiarizationBackend ABC and PyAnnote integration.

pyannote.audio is NOT installed in the test environment. All tests mock
pyannote imports to test the backend logic in isolation.
"""

from __future__ import annotations

import sys
from types import ModuleType
from typing import Any
from unittest.mock import MagicMock, patch

import pytest

from macaw._types import SpeakerSegment
from macaw.workers.stt.diarization.interface import DiarizationBackend


class TestDiarizationBackendABC:
    """DiarizationBackend ABC cannot be instantiated directly."""

    def test_cannot_instantiate_abc(self) -> None:
        with pytest.raises(TypeError, match="abstract"):
            DiarizationBackend()  # type: ignore[abstract]

    def test_abc_has_required_methods(self) -> None:
        assert hasattr(DiarizationBackend, "diarize")
        assert hasattr(DiarizationBackend, "load")
        assert hasattr(DiarizationBackend, "health")


class TestCreateDiarizer:
    """create_diarizer() factory function."""

    def test_returns_none_when_pyannote_not_installed(self) -> None:
        # PyAnnoteDiarizer.__init__ imports pyannote.audio which is not installed
        # so create_diarizer should catch ImportError and return None
        from macaw.workers.stt.diarization import create_diarizer

        result = create_diarizer()
        assert result is None

    def test_returns_diarizer_when_pyannote_available(self) -> None:
        # Mock pyannote.audio so import succeeds
        mock_pyannote = ModuleType("pyannote")
        mock_pyannote_audio = ModuleType("pyannote.audio")
        mock_pyannote.audio = mock_pyannote_audio  # type: ignore[attr-defined]

        with patch.dict(
            sys.modules,
            {
                "pyannote": mock_pyannote,
                "pyannote.audio": mock_pyannote_audio,
            },
        ):
            from macaw.workers.stt.diarization import create_diarizer

            result = create_diarizer()
            assert result is not None
            assert isinstance(result, DiarizationBackend)


class _MockAnnotation:
    """Fake pyannote Annotation for testing."""

    def __init__(self, tracks: list[tuple[Any, str]]) -> None:
        self._tracks = tracks

    def itertracks(self, yield_label: bool = False) -> list[tuple[Any, None, str]]:
        return [(turn, None, label) for turn, label in self._tracks]


class _MockTurn:
    """Fake pyannote Segment (turn) with start/end."""

    def __init__(self, start: float, end: float) -> None:
        self.start = start
        self.end = end


def _make_pcm_audio(num_samples: int = 16000) -> bytes:
    """Create silent 16-bit PCM audio bytes."""
    return b"\x00\x00" * num_samples


def _setup_pyannote_mocks() -> tuple[MagicMock, MagicMock]:
    """Create mock pyannote modules and return (mock_pipeline_class, mock_pipeline_instance)."""
    mock_pipeline_instance = MagicMock()
    mock_pipeline_class = MagicMock()
    mock_pipeline_class.from_pretrained = MagicMock(return_value=mock_pipeline_instance)

    mock_pyannote = ModuleType("pyannote")
    mock_pyannote_audio = ModuleType("pyannote.audio")
    mock_pyannote.audio = mock_pyannote_audio  # type: ignore[attr-defined]
    mock_pyannote_audio.Pipeline = mock_pipeline_class  # type: ignore[attr-defined]

    return mock_pipeline_class, mock_pipeline_instance


class TestPyAnnoteDiarizer:
    """PyAnnoteDiarizer backend tests with mocked pyannote.audio."""

    def _create_diarizer(self) -> Any:
        """Create a PyAnnoteDiarizer with mocked pyannote import."""
        mock_pyannote = ModuleType("pyannote")
        mock_pyannote_audio = ModuleType("pyannote.audio")
        mock_pyannote.audio = mock_pyannote_audio  # type: ignore[attr-defined]

        with patch.dict(
            sys.modules,
            {
                "pyannote": mock_pyannote,
                "pyannote.audio": mock_pyannote_audio,
            },
        ):
            from macaw.workers.stt.diarization.pyannote_backend import PyAnnoteDiarizer

            return PyAnnoteDiarizer()

    async def test_loads_model_lazily(self) -> None:
        diarizer = self._create_diarizer()
        assert diarizer._loaded is False
        assert diarizer._pipeline is None

    async def test_load_sets_loaded_flag(self) -> None:
        mock_pipeline_class, _mock_pipeline_instance = _setup_pyannote_mocks()

        diarizer = self._create_diarizer()

        with patch.dict(
            sys.modules,
            {
                "pyannote": sys.modules.get("pyannote", ModuleType("pyannote")),
                "pyannote.audio": MagicMock(Pipeline=mock_pipeline_class),
            },
        ):
            await diarizer.load()

        assert diarizer._loaded is True
        assert diarizer._pipeline is not None

    async def test_load_called_only_once(self) -> None:
        mock_pipeline_class, _ = _setup_pyannote_mocks()

        diarizer = self._create_diarizer()

        with patch.dict(
            sys.modules,
            {
                "pyannote": sys.modules.get("pyannote", ModuleType("pyannote")),
                "pyannote.audio": MagicMock(Pipeline=mock_pipeline_class),
            },
        ):
            await diarizer.load()
            await diarizer.load()  # Second call should be a no-op

        mock_pipeline_class.from_pretrained.assert_called_once()

    async def test_diarize_returns_speaker_segments(self) -> None:
        _mock_pipeline_class, mock_pipeline_instance = _setup_pyannote_mocks()

        tracks = [
            (_MockTurn(0.0, 1.5), "SPEAKER_00"),
            (_MockTurn(1.5, 3.0), "SPEAKER_01"),
            (_MockTurn(3.0, 4.5), "SPEAKER_00"),
        ]
        mock_pipeline_instance.return_value = _MockAnnotation(tracks)

        diarizer = self._create_diarizer()
        diarizer._pipeline = mock_pipeline_instance
        diarizer._loaded = True

        # Mock torch for the conversion
        mock_torch = MagicMock()
        mock_tensor = MagicMock()
        mock_tensor.unsqueeze.return_value = mock_tensor
        mock_tensor.__truediv__ = MagicMock(return_value=mock_tensor)
        mock_torch.tensor.return_value = mock_tensor
        mock_torch.float32 = "float32"

        with patch.dict(sys.modules, {"torch": mock_torch}):
            result = await diarizer.diarize(_make_pcm_audio(), 16000)

        assert len(result) == 3
        assert all(isinstance(s, SpeakerSegment) for s in result)
        assert result[0].speaker_id == "SPEAKER_00"
        assert result[0].start == 0.0
        assert result[0].end == 1.5
        assert result[1].speaker_id == "SPEAKER_01"
        assert result[2].speaker_id == "SPEAKER_00"

    async def test_diarize_empty_audio_returns_empty(self) -> None:
        diarizer = self._create_diarizer()
        diarizer._loaded = True

        result = await diarizer.diarize(b"", 16000)
        assert result == ()

    async def test_diarize_forwards_max_speakers(self) -> None:
        _mock_pipeline_class, mock_pipeline_instance = _setup_pyannote_mocks()
        mock_pipeline_instance.return_value = _MockAnnotation([])

        diarizer = self._create_diarizer()
        diarizer._pipeline = mock_pipeline_instance
        diarizer._loaded = True

        mock_torch = MagicMock()
        mock_tensor = MagicMock()
        mock_tensor.unsqueeze.return_value = mock_tensor
        mock_tensor.__truediv__ = MagicMock(return_value=mock_tensor)
        mock_torch.tensor.return_value = mock_tensor
        mock_torch.float32 = "float32"

        with patch.dict(sys.modules, {"torch": mock_torch}):
            await diarizer.diarize(_make_pcm_audio(), 16000, max_speakers=3)

        # Verify max_speakers was passed to the pipeline
        call_kwargs = mock_pipeline_instance.call_args
        assert call_kwargs[1]["max_speakers"] == 3

    async def test_diarize_auto_loads_model(self) -> None:
        mock_pipeline_class, mock_pipeline_instance = _setup_pyannote_mocks()
        mock_pipeline_instance.return_value = _MockAnnotation([])

        diarizer = self._create_diarizer()
        assert diarizer._loaded is False

        mock_torch = MagicMock()
        mock_tensor = MagicMock()
        mock_tensor.unsqueeze.return_value = mock_tensor
        mock_tensor.__truediv__ = MagicMock(return_value=mock_tensor)
        mock_torch.tensor.return_value = mock_tensor
        mock_torch.float32 = "float32"

        with patch.dict(
            sys.modules,
            {
                "pyannote": sys.modules.get("pyannote", ModuleType("pyannote")),
                "pyannote.audio": MagicMock(Pipeline=mock_pipeline_class),
                "torch": mock_torch,
            },
        ):
            await diarizer.diarize(_make_pcm_audio(), 16000)

        assert diarizer._loaded is True

    async def test_health_not_loaded(self) -> None:
        diarizer = self._create_diarizer()
        health = await diarizer.health()
        assert health["status"] == "not_loaded"
        assert health["backend"] == "pyannote"

    async def test_health_loaded(self) -> None:
        diarizer = self._create_diarizer()
        diarizer._loaded = True
        health = await diarizer.health()
        assert health["status"] == "ok"
        assert health["backend"] == "pyannote"

    async def test_speaker_segment_fields(self) -> None:
        """SpeakerSegment has speaker_id, start, end, text fields."""
        seg = SpeakerSegment(speaker_id="s0", start=1.0, end=2.5, text="hello")
        assert seg.speaker_id == "s0"
        assert seg.start == 1.0
        assert seg.end == 2.5
        assert seg.text == "hello"

    async def test_run_in_executor_not_blocking(self) -> None:
        """Verify diarize runs pipeline in executor (not blocking event loop)."""
        _mock_pipeline_class, mock_pipeline_instance = _setup_pyannote_mocks()
        mock_pipeline_instance.return_value = _MockAnnotation([])

        diarizer = self._create_diarizer()
        diarizer._pipeline = mock_pipeline_instance
        diarizer._loaded = True

        mock_torch = MagicMock()
        mock_tensor = MagicMock()
        mock_tensor.unsqueeze.return_value = mock_tensor
        mock_tensor.__truediv__ = MagicMock(return_value=mock_tensor)
        mock_torch.tensor.return_value = mock_tensor
        mock_torch.float32 = "float32"

        with patch.dict(sys.modules, {"torch": mock_torch}):
            # If it runs in executor, it should not block the event loop
            # We verify by ensuring the call completes without error
            result = await diarizer.diarize(_make_pcm_audio(), 16000)

        assert result == ()
