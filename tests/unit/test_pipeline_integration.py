"""Integration tests for preprocessing and post-processing pipelines in the HTTP flow."""

from __future__ import annotations

from pathlib import Path
from unittest.mock import AsyncMock, MagicMock

import httpx

from macaw._types import BatchResult, SegmentDetail
from macaw.config.postprocessing import PostProcessingConfig
from macaw.config.preprocessing import PreprocessingConfig
from macaw.postprocessing.pipeline import PostProcessingPipeline
from macaw.postprocessing.stages import TextStage
from macaw.preprocessing.pipeline import AudioPreprocessingPipeline
from macaw.preprocessing.stages import AudioStage
from macaw.server.app import create_app

AUDIO_FIXTURE = Path(__file__).parent.parent / "fixtures" / "audio" / "sample_44khz.wav"


def _make_mock_registry() -> MagicMock:
    registry = MagicMock()
    registry.has_model.return_value = True
    registry.get_manifest.return_value = MagicMock()
    return registry


def _make_mock_scheduler(text: str = "Ola mundo") -> MagicMock:
    scheduler = MagicMock()
    scheduler.transcribe = AsyncMock(
        return_value=BatchResult(
            text=text,
            language="pt",
            duration=1.5,
            segments=(SegmentDetail(id=0, start=0.0, end=1.5, text=text),),
        )
    )
    return scheduler


class _UpperCaseStage(TextStage):
    """Test stage that converts text to uppercase."""

    @property
    def name(self) -> str:
        return "uppercase"

    def process(self, text: str, *, language: str | None = None) -> str:
        return text.upper()


class _SuffixStage(TextStage):
    """Test stage that appends a suffix to text."""

    @property
    def name(self) -> str:
        return "suffix"

    def __init__(self, suffix: str = " [processado]") -> None:
        self._suffix = suffix

    def process(self, text: str, *, language: str | None = None) -> str:
        return text + self._suffix


# --- Backwards compatibility tests ---


async def test_no_pipelines_backwards_compatible() -> None:
    """App created without pipelines continues working normally."""
    scheduler = _make_mock_scheduler()
    app = create_app(registry=_make_mock_registry(), scheduler=scheduler)

    async with httpx.AsyncClient(
        transport=httpx.ASGITransport(app=app),
        base_url="http://test",
    ) as client:
        response = await client.post(
            "/v1/audio/transcriptions",
            files={"file": ("audio.wav", b"fake-audio-data", "audio/wav")},
            data={"model": "faster-whisper-tiny"},
        )

    assert response.status_code == 200
    assert response.json() == {"text": "Ola mundo"}
    scheduler.transcribe.assert_awaited_once()


# --- Preprocessing tests ---


async def test_preprocessing_applied_to_audio() -> None:
    """Preprocessing transforms audio before sending to scheduler."""
    scheduler = _make_mock_scheduler()
    audio_bytes = AUDIO_FIXTURE.read_bytes()

    config = PreprocessingConfig(target_sample_rate=16000)
    pipeline = AudioPreprocessingPipeline(config, stages=[])

    # Create mock stage that marks audio as processed
    mock_stage = MagicMock(spec=AudioStage)
    mock_stage.name = "test_marker"

    import numpy as np

    def mark_process(audio: np.ndarray, sample_rate: int) -> tuple[np.ndarray, int]:
        # Returns audio with all values zeroed (marker for verification)
        return np.zeros_like(audio), sample_rate

    mock_stage.process.side_effect = mark_process
    pipeline._stages = [mock_stage]

    app = create_app(
        registry=_make_mock_registry(),
        scheduler=scheduler,
        preprocessing_pipeline=pipeline,
    )

    async with httpx.AsyncClient(
        transport=httpx.ASGITransport(app=app),
        base_url="http://test",
    ) as client:
        response = await client.post(
            "/v1/audio/transcriptions",
            files={"file": ("audio.wav", audio_bytes, "audio/wav")},
            data={"model": "faster-whisper-tiny"},
        )

    assert response.status_code == 200

    # Verify the stage was called
    mock_stage.process.assert_called_once()

    # Verify the scheduler received audio different from the original
    call_args = scheduler.transcribe.call_args[0][0]
    assert call_args.audio_data != audio_bytes


async def test_preprocessing_only_without_postprocessing() -> None:
    """Only preprocessing configured, without post-processing."""
    scheduler = _make_mock_scheduler(text="texto original")
    audio_bytes = AUDIO_FIXTURE.read_bytes()

    config = PreprocessingConfig()
    pipeline = AudioPreprocessingPipeline(config, stages=[])

    app = create_app(
        registry=_make_mock_registry(),
        scheduler=scheduler,
        preprocessing_pipeline=pipeline,
    )

    async with httpx.AsyncClient(
        transport=httpx.ASGITransport(app=app),
        base_url="http://test",
    ) as client:
        response = await client.post(
            "/v1/audio/transcriptions",
            files={"file": ("audio.wav", audio_bytes, "audio/wav")},
            data={"model": "faster-whisper-tiny"},
        )

    assert response.status_code == 200
    # Text should not be modified (no post-processing)
    assert response.json() == {"text": "texto original"}


# --- Post-processing tests ---


async def test_postprocessing_applied_to_result() -> None:
    """Post-processing transforms result text before returning."""
    scheduler = _make_mock_scheduler(text="dois mil e vinte e cinco")

    config = PostProcessingConfig()
    pipeline = PostProcessingPipeline(config, stages=[_UpperCaseStage()])

    app = create_app(
        registry=_make_mock_registry(),
        scheduler=scheduler,
        postprocessing_pipeline=pipeline,
    )

    async with httpx.AsyncClient(
        transport=httpx.ASGITransport(app=app),
        base_url="http://test",
    ) as client:
        response = await client.post(
            "/v1/audio/transcriptions",
            files={"file": ("audio.wav", b"fake-audio-data", "audio/wav")},
            data={"model": "faster-whisper-tiny"},
        )

    assert response.status_code == 200
    assert response.json() == {"text": "DOIS MIL E VINTE E CINCO"}


async def test_postprocessing_transforms_segments_in_verbose_json() -> None:
    """Post-processing transforms segment text in verbose_json."""
    scheduler = _make_mock_scheduler(text="hello world")

    config = PostProcessingConfig()
    pipeline = PostProcessingPipeline(config, stages=[_UpperCaseStage()])

    app = create_app(
        registry=_make_mock_registry(),
        scheduler=scheduler,
        postprocessing_pipeline=pipeline,
    )

    async with httpx.AsyncClient(
        transport=httpx.ASGITransport(app=app),
        base_url="http://test",
    ) as client:
        response = await client.post(
            "/v1/audio/transcriptions",
            files={"file": ("audio.wav", b"fake-audio-data", "audio/wav")},
            data={"model": "faster-whisper-tiny", "response_format": "verbose_json"},
        )

    assert response.status_code == 200
    body = response.json()
    assert body["text"] == "HELLO WORLD"
    assert body["segments"][0]["text"] == "HELLO WORLD"


async def test_postprocessing_only_without_preprocessing() -> None:
    """Only post-processing configured, without preprocessing."""
    scheduler = _make_mock_scheduler(text="texto cru")

    config = PostProcessingConfig()
    pipeline = PostProcessingPipeline(config, stages=[_SuffixStage()])

    app = create_app(
        registry=_make_mock_registry(),
        scheduler=scheduler,
        postprocessing_pipeline=pipeline,
    )

    async with httpx.AsyncClient(
        transport=httpx.ASGITransport(app=app),
        base_url="http://test",
    ) as client:
        response = await client.post(
            "/v1/audio/transcriptions",
            files={"file": ("audio.wav", b"fake-audio-data", "audio/wav")},
            data={"model": "faster-whisper-tiny"},
        )

    assert response.status_code == 200
    assert response.json() == {"text": "texto cru [processado]"}


# --- Tests with both pipelines ---


async def test_both_pipelines_applied() -> None:
    """Both pipelines applied in the same request."""
    scheduler = _make_mock_scheduler(text="resultado do stt")
    audio_bytes = AUDIO_FIXTURE.read_bytes()

    pre_config = PreprocessingConfig()
    pre_pipeline = AudioPreprocessingPipeline(pre_config, stages=[])

    post_config = PostProcessingConfig()
    post_pipeline = PostProcessingPipeline(post_config, stages=[_UpperCaseStage()])

    app = create_app(
        registry=_make_mock_registry(),
        scheduler=scheduler,
        preprocessing_pipeline=pre_pipeline,
        postprocessing_pipeline=post_pipeline,
    )

    async with httpx.AsyncClient(
        transport=httpx.ASGITransport(app=app),
        base_url="http://test",
    ) as client:
        response = await client.post(
            "/v1/audio/transcriptions",
            files={"file": ("audio.wav", audio_bytes, "audio/wav")},
            data={"model": "faster-whisper-tiny"},
        )

    assert response.status_code == 200
    assert response.json() == {"text": "RESULTADO DO STT"}
    scheduler.transcribe.assert_awaited_once()


# --- Translation tests with pipelines ---


async def test_postprocessing_applied_to_translation() -> None:
    """Post-processing works on the translation route."""
    scheduler = _make_mock_scheduler(text="hello world")

    config = PostProcessingConfig()
    pipeline = PostProcessingPipeline(config, stages=[_UpperCaseStage()])

    app = create_app(
        registry=_make_mock_registry(),
        scheduler=scheduler,
        postprocessing_pipeline=pipeline,
    )

    async with httpx.AsyncClient(
        transport=httpx.ASGITransport(app=app),
        base_url="http://test",
    ) as client:
        response = await client.post(
            "/v1/audio/translations",
            files={"file": ("audio.wav", b"fake-audio-data", "audio/wav")},
            data={"model": "faster-whisper-tiny"},
        )

    assert response.status_code == 200
    assert response.json() == {"text": "HELLO WORLD"}


# --- Pipeline tests with multiple stages ---


async def test_postprocessing_multiple_stages_applied_in_order() -> None:
    """Multiple post-processing stages applied in sequence."""
    scheduler = _make_mock_scheduler(text="hello")

    config = PostProcessingConfig()
    pipeline = PostProcessingPipeline(config, stages=[_UpperCaseStage(), _SuffixStage(" [DONE]")])

    app = create_app(
        registry=_make_mock_registry(),
        scheduler=scheduler,
        postprocessing_pipeline=pipeline,
    )

    async with httpx.AsyncClient(
        transport=httpx.ASGITransport(app=app),
        base_url="http://test",
    ) as client:
        response = await client.post(
            "/v1/audio/transcriptions",
            files={"file": ("audio.wav", b"fake-audio-data", "audio/wav")},
            data={"model": "faster-whisper-tiny"},
        )

    assert response.status_code == 200
    # UpperCase primeiro, depois Suffix
    assert response.json() == {"text": "HELLO [DONE]"}
