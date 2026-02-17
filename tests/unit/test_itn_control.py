"""Testes do controle de ITN via parametro itn na API e --no-itn no CLI."""

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
    """Stage de teste que converte texto para maiusculas."""

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


# --- API: parametro itn ---


class TestITNDefaultBehavior:
    """Verifica que o comportamento padrao (itn=True) aplica post-processing."""

    async def test_itn_default_true_applies_postprocessing(self) -> None:
        """Request sem campo itn: post-processing aplicado (default True)."""
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
    """Verifica que itn=false desabilita post-processing."""

    async def test_itn_false_skips_postprocessing(self) -> None:
        """Request com itn=false: post-processing NAO aplicado."""
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
    """Verifica que itn=true aplica post-processing explicitamente."""

    async def test_itn_true_applies_postprocessing(self) -> None:
        """Request com itn=true: post-processing aplicado."""
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
    """Verifica itn=false no endpoint /v1/audio/transcriptions."""

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
    """Verifica itn=false no endpoint /v1/audio/translations."""

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
    """Verifica que itn=true sem pipeline configurado nao causa erro."""

    async def test_itn_true_without_pipeline_configured(self) -> None:
        """Sem pipeline configurado + itn=true: funciona sem erro."""
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
        """Sem pipeline configurado + itn=false: funciona sem erro."""
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
    """Verifica que o endpoint de traducao tambem aplica post-processing por default."""

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


# --- CLI: flag --no-itn ---


class TestCLINoITNFlag:
    """Testes da flag --no-itn nos comandos CLI."""

    def test_transcribe_no_itn_sends_itn_false(self, tmp_path: Path) -> None:
        """--no-itn envia itn=false no form data."""
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
        """Sem --no-itn, campo itn nao e enviado (server usa default True)."""
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
        """--no-itn no translate envia itn=false no form data."""
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
        """Sem --no-itn no translate, campo itn nao e enviado."""
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
