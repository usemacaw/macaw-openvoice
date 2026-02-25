"""Tests for ITN control via itn parameter in the API and --no-itn in the CLI."""

from __future__ import annotations

from typing import TYPE_CHECKING
from unittest.mock import AsyncMock, MagicMock

import httpx
from click.testing import CliRunner

from macaw._types import BatchResult, SegmentDetail
from macaw.config.postprocessing import PostProcessingConfig
from macaw.postprocessing.pipeline import PostProcessingPipeline
from macaw.postprocessing.stages import TextStage
from macaw.server.app import create_app

if TYPE_CHECKING:
    from pathlib import Path


class UppercaseStage(TextStage):
    """Test stage that converts text to uppercase."""

    @property
    def name(self) -> str:
        return "uppercase"

    def process(self, text: str, *, language: str | None = None) -> str:
        return text.upper()


def _make_mock_scheduler(text: str = "dois mil e vinte e cinco") -> MagicMock:
    scheduler = MagicMock()
    scheduler.transcribe = AsyncMock(
        return_value=BatchResult(
            text=text,
            language="pt",
            duration=2.5,
            segments=(SegmentDetail(id=0, start=0.0, end=2.5, text=text),),
        )
    )
    return scheduler


def _make_mock_registry() -> MagicMock:
    registry = MagicMock()
    registry.has_model.return_value = True
    registry.get_manifest.return_value = MagicMock()
    return registry


def _make_postprocessing_pipeline() -> PostProcessingPipeline:
    return PostProcessingPipeline(
        config=PostProcessingConfig(),
        stages=[UppercaseStage()],
    )


# --- API: itn parameter ---


class TestITNDefaultBehavior:
    """Verifies that the default behavior (itn=True) applies post-processing."""

    async def test_itn_default_true_applies_postprocessing(self) -> None:
        """Request without itn field: post-processing applied (default True)."""
        scheduler = _make_mock_scheduler()
        pipeline = _make_postprocessing_pipeline()
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
        assert response.json()["text"] == "DOIS MIL E VINTE E CINCO"


class TestITNFalseSkipsPostprocessing:
    """Verifies that itn=false disables post-processing."""

    async def test_itn_false_skips_postprocessing(self) -> None:
        """Request with itn=false: post-processing NOT applied."""
        scheduler = _make_mock_scheduler()
        pipeline = _make_postprocessing_pipeline()
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
                data={"model": "faster-whisper-tiny", "itn": "false"},
            )

        assert response.status_code == 200
        assert response.json()["text"] == "dois mil e vinte e cinco"


class TestITNTrueAppliesPostprocessing:
    """Verifies that itn=true applies post-processing explicitly."""

    async def test_itn_true_applies_postprocessing(self) -> None:
        """Request with itn=true: post-processing applied."""
        scheduler = _make_mock_scheduler()
        pipeline = _make_postprocessing_pipeline()
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
                data={"model": "faster-whisper-tiny", "itn": "true"},
            )

        assert response.status_code == 200
        assert response.json()["text"] == "DOIS MIL E VINTE E CINCO"


class TestITNFalseTranscriptionsEndpoint:
    """Verifies itn=false on the /v1/audio/transcriptions endpoint."""

    async def test_itn_false_transcriptions_endpoint(self) -> None:
        scheduler = _make_mock_scheduler(text="ola mundo")
        pipeline = _make_postprocessing_pipeline()
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
                data={"model": "faster-whisper-tiny", "itn": "false"},
            )

        assert response.status_code == 200
        assert response.json()["text"] == "ola mundo"


class TestITNFalseTranslationsEndpoint:
    """Verifies itn=false on the /v1/audio/translations endpoint."""

    async def test_itn_false_translations_endpoint(self) -> None:
        scheduler = _make_mock_scheduler(text="hello world")
        pipeline = _make_postprocessing_pipeline()
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
                data={"model": "faster-whisper-tiny", "itn": "false"},
            )

        assert response.status_code == 200
        assert response.json()["text"] == "hello world"


class TestITNWithoutPipelineConfigured:
    """Verifies that itn=true without a configured pipeline does not cause an error."""

    async def test_itn_true_without_pipeline_configured(self) -> None:
        """Without configured pipeline + itn=true: works without error."""
        scheduler = _make_mock_scheduler(text="ola mundo")
        app = create_app(
            registry=_make_mock_registry(),
            scheduler=scheduler,
        )

        async with httpx.AsyncClient(
            transport=httpx.ASGITransport(app=app),
            base_url="http://test",
        ) as client:
            response = await client.post(
                "/v1/audio/transcriptions",
                files={"file": ("audio.wav", b"fake-audio-data", "audio/wav")},
                data={"model": "faster-whisper-tiny", "itn": "true"},
            )

        assert response.status_code == 200
        assert response.json()["text"] == "ola mundo"

    async def test_itn_false_without_pipeline_configured(self) -> None:
        """Without configured pipeline + itn=false: works without error."""
        scheduler = _make_mock_scheduler(text="ola mundo")
        app = create_app(
            registry=_make_mock_registry(),
            scheduler=scheduler,
        )

        async with httpx.AsyncClient(
            transport=httpx.ASGITransport(app=app),
            base_url="http://test",
        ) as client:
            response = await client.post(
                "/v1/audio/transcriptions",
                files={"file": ("audio.wav", b"fake-audio-data", "audio/wav")},
                data={"model": "faster-whisper-tiny", "itn": "false"},
            )

        assert response.status_code == 200
        assert response.json()["text"] == "ola mundo"


class TestITNTranslationsDefault:
    """Verifies that the translation endpoint also applies post-processing by default."""

    async def test_translations_default_applies_postprocessing(self) -> None:
        scheduler = _make_mock_scheduler(text="hello world")
        pipeline = _make_postprocessing_pipeline()
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
        assert response.json()["text"] == "HELLO WORLD"


# --- CLI: --no-itn flag ---


class TestCLINoITNFlag:
    """Tests for the --no-itn flag in CLI commands."""

    def test_transcribe_no_itn_sends_itn_false(self, tmp_path: Path) -> None:
        """--no-itn sends itn=false in form data."""
        from unittest.mock import patch

        from macaw.cli.transcribe import transcribe

        audio_file = tmp_path / "test.wav"
        audio_file.write_bytes(b"fake-audio-data")

        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.headers = {"content-type": "application/json"}
        mock_response.json.return_value = {"text": "hello"}

        with patch("httpx.post", return_value=mock_response) as mock_post:
            runner = CliRunner()
            result = runner.invoke(
                transcribe,
                [str(audio_file), "--model", "test-model", "--no-itn"],
            )

        assert result.exit_code == 0
        call_kwargs = mock_post.call_args
        sent_data = call_kwargs.kwargs.get("data") or call_kwargs[1].get("data")
        assert sent_data["itn"] == "false"

    def test_transcribe_without_no_itn_omits_itn_field(self, tmp_path: Path) -> None:
        """Without --no-itn, itn field is not sent (server uses default True)."""
        from unittest.mock import patch

        from macaw.cli.transcribe import transcribe

        audio_file = tmp_path / "test.wav"
        audio_file.write_bytes(b"fake-audio-data")

        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.headers = {"content-type": "application/json"}
        mock_response.json.return_value = {"text": "hello"}

        with patch("httpx.post", return_value=mock_response) as mock_post:
            runner = CliRunner()
            result = runner.invoke(
                transcribe,
                [str(audio_file), "--model", "test-model"],
            )

        assert result.exit_code == 0
        call_kwargs = mock_post.call_args
        sent_data = call_kwargs.kwargs.get("data") or call_kwargs[1].get("data")
        assert "itn" not in sent_data

    def test_translate_no_itn_sends_itn_false(self, tmp_path: Path) -> None:
        """--no-itn in translate sends itn=false in form data."""
        from unittest.mock import patch

        from macaw.cli.transcribe import translate

        audio_file = tmp_path / "test.wav"
        audio_file.write_bytes(b"fake-audio-data")

        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.headers = {"content-type": "application/json"}
        mock_response.json.return_value = {"text": "hello"}

        with patch("httpx.post", return_value=mock_response) as mock_post:
            runner = CliRunner()
            result = runner.invoke(
                translate,
                [str(audio_file), "--model", "test-model", "--no-itn"],
            )

        assert result.exit_code == 0
        call_kwargs = mock_post.call_args
        sent_data = call_kwargs.kwargs.get("data") or call_kwargs[1].get("data")
        assert sent_data["itn"] == "false"

    def test_translate_without_no_itn_omits_itn_field(self, tmp_path: Path) -> None:
        """Without --no-itn in translate, itn field is not sent."""
        from unittest.mock import patch

        from macaw.cli.transcribe import translate

        audio_file = tmp_path / "test.wav"
        audio_file.write_bytes(b"fake-audio-data")

        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.headers = {"content-type": "application/json"}
        mock_response.json.return_value = {"text": "hello"}

        with patch("httpx.post", return_value=mock_response) as mock_post:
            runner = CliRunner()
            result = runner.invoke(
                translate,
                [str(audio_file), "--model", "test-model"],
            )

        assert result.exit_code == 0
        call_kwargs = mock_post.call_args
        sent_data = call_kwargs.kwargs.get("data") or call_kwargs[1].get("data")
        assert "itn" not in sent_data
