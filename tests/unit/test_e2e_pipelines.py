"""Testes end-to-end do pipeline completo com preprocessing e post-processing.

Valida o fluxo: audio em qualquer sample rate -> preprocessing -> worker (mock)
-> post-processing -> resposta formatada.

Usa stages REAIS de preprocessing (ResampleStage, DCRemoveStage, GainNormalizeStage)
e mock do Scheduler (sem worker gRPC real) e mock do ITN (nemo_text_processing
nao necessario).
"""

from __future__ import annotations

import io
import math
import struct
import wave
from typing import TYPE_CHECKING
from unittest.mock import AsyncMock, MagicMock

import httpx

from macaw._types import BatchResult, SegmentDetail, WordTimestamp
from macaw.config.postprocessing import PostProcessingConfig
from macaw.config.preprocessing import PreprocessingConfig
from macaw.postprocessing.pipeline import PostProcessingPipeline
from macaw.postprocessing.stages import TextStage
from macaw.preprocessing.dc_remove import DCRemoveStage
from macaw.preprocessing.gain_normalize import GainNormalizeStage
from macaw.preprocessing.pipeline import AudioPreprocessingPipeline
from macaw.preprocessing.resample import ResampleStage
from macaw.server.app import create_app

if TYPE_CHECKING:
    from macaw.server.models.requests import TranscribeRequest


# --- Helpers ---


def _generate_wav(sample_rate: int, duration: float = 1.0, frequency: float = 440.0) -> bytes:
    """Gera arquivo WAV em memoria com tom senoidal."""
    n_samples = int(sample_rate * duration)
    samples = []
    for i in range(n_samples):
        t = i / sample_rate
        value = int(32767 * 0.5 * math.sin(2 * math.pi * frequency * t))
        samples.append(value)

    buffer = io.BytesIO()
    with wave.open(buffer, "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(sample_rate)
        wf.writeframes(struct.pack(f"<{n_samples}h", *samples))
    return buffer.getvalue()


class MockITNStage(TextStage):
    """Mock ITN que aplica transformacoes conhecidas para testes."""

    @property
    def name(self) -> str:
        return "mock_itn"

    def process(self, text: str) -> str:
        replacements = {
            "dois mil e quinhentos reais": "R$2.500",
            "dez por cento": "10%",
        }
        for k, v in replacements.items():
            text = text.replace(k, v)
        return text


class MockScheduler:
    """Scheduler mock que captura o request e retorna BatchResult configuravel."""

    def __init__(self, result: BatchResult) -> None:
        self._result = result
        self.last_request: TranscribeRequest | None = None

    async def transcribe(self, request: TranscribeRequest) -> BatchResult:
        self.last_request = request
        return self._result


def _make_registry() -> MagicMock:
    registry = MagicMock()
    registry.has_model.return_value = True
    registry.get_manifest.return_value = MagicMock()
    registry.list_manifests.return_value = []
    return registry


def _make_itn_batch_result() -> BatchResult:
    """BatchResult com texto que contem numeros por extenso para ITN."""
    return BatchResult(
        text="O valor e dois mil e quinhentos reais com dez por cento de juros",
        language="pt",
        duration=3.0,
        segments=(
            SegmentDetail(
                id=0,
                start=0.0,
                end=1.5,
                text="O valor e dois mil e quinhentos reais",
                avg_logprob=-0.20,
                no_speech_prob=0.01,
                compression_ratio=1.1,
            ),
            SegmentDetail(
                id=1,
                start=1.5,
                end=3.0,
                text="com dez por cento de juros",
                avg_logprob=-0.25,
                no_speech_prob=0.02,
                compression_ratio=1.0,
            ),
        ),
        words=(
            WordTimestamp(word="O", start=0.0, end=0.1),
            WordTimestamp(word="valor", start=0.1, end=0.3),
        ),
    )


def _make_raw_batch_result() -> BatchResult:
    """BatchResult com texto simples sem numeros."""
    return BatchResult(
        text="Ola, como posso ajudar?",
        language="pt",
        duration=2.0,
        segments=(
            SegmentDetail(
                id=0,
                start=0.0,
                end=2.0,
                text="Ola, como posso ajudar?",
                avg_logprob=-0.20,
                no_speech_prob=0.01,
                compression_ratio=1.1,
            ),
        ),
    )


def _create_test_app(
    scheduler: MockScheduler,
    *,
    with_preprocessing: bool = True,
    with_postprocessing: bool = True,
    itn_stage: TextStage | None = None,
) -> object:
    """Cria app FastAPI com pipelines reais de preprocessing e mock de ITN."""
    pre_pipeline = None
    if with_preprocessing:
        config = PreprocessingConfig()
        stages = [
            ResampleStage(config.target_sample_rate),
            DCRemoveStage(config.dc_remove_cutoff_hz),
            GainNormalizeStage(config.target_dbfs),
        ]
        pre_pipeline = AudioPreprocessingPipeline(config, stages)

    post_pipeline = None
    if with_postprocessing and itn_stage is not None:
        post_config = PostProcessingConfig()
        post_pipeline = PostProcessingPipeline(post_config, stages=[itn_stage])

    return create_app(
        registry=_make_registry(),
        scheduler=scheduler,
        preprocessing_pipeline=pre_pipeline,
        postprocessing_pipeline=post_pipeline,
    )


# --- Testes E2E: Preprocessing + Postprocessing ---


class TestFullPipeline44kHz:
    """Audio 44.1kHz -> preprocessing converte para 16kHz -> ITN transforma texto."""

    async def test_full_pipeline_44khz_with_itn(self) -> None:
        wav_bytes = _generate_wav(sample_rate=44100, duration=0.5)
        result = _make_itn_batch_result()
        scheduler = MockScheduler(result)
        app = _create_test_app(scheduler, itn_stage=MockITNStage())

        async with httpx.AsyncClient(
            transport=httpx.ASGITransport(app=app), base_url="http://test"
        ) as client:
            response = await client.post(
                "/v1/audio/transcriptions",
                files={"file": ("test.wav", wav_bytes, "audio/wav")},
                data={"model": "test-model"},
            )

        assert response.status_code == 200
        body = response.json()
        # ITN deve ter transformado o texto
        assert "R$2.500" in body["text"]
        assert "10%" in body["text"]
        # Audio recebido pelo scheduler deve ser menor que o original (44.1kHz -> 16kHz)
        assert scheduler.last_request is not None
        assert len(scheduler.last_request.audio_data) < len(wav_bytes)


class TestFullPipeline8kHz:
    """Audio 8kHz -> preprocessing faz upsample para 16kHz -> ITN transforma texto."""

    async def test_full_pipeline_8khz_with_itn(self) -> None:
        wav_bytes = _generate_wav(sample_rate=8000, duration=0.5)
        result = _make_itn_batch_result()
        scheduler = MockScheduler(result)
        app = _create_test_app(scheduler, itn_stage=MockITNStage())

        async with httpx.AsyncClient(
            transport=httpx.ASGITransport(app=app), base_url="http://test"
        ) as client:
            response = await client.post(
                "/v1/audio/transcriptions",
                files={"file": ("test.wav", wav_bytes, "audio/wav")},
                data={"model": "test-model"},
            )

        assert response.status_code == 200
        body = response.json()
        assert "R$2.500" in body["text"]
        assert "10%" in body["text"]
        # Audio recebido pelo scheduler deve ser maior que o original (8kHz -> 16kHz)
        assert scheduler.last_request is not None
        assert len(scheduler.last_request.audio_data) > len(wav_bytes)


class TestFullPipeline16kHz:
    """Audio ja em 16kHz -> sem resample necessario -> funciona corretamente."""

    async def test_full_pipeline_16khz_no_resample(self) -> None:
        wav_bytes = _generate_wav(sample_rate=16000, duration=0.5)
        result = _make_itn_batch_result()
        scheduler = MockScheduler(result)
        app = _create_test_app(scheduler, itn_stage=MockITNStage())

        async with httpx.AsyncClient(
            transport=httpx.ASGITransport(app=app), base_url="http://test"
        ) as client:
            response = await client.post(
                "/v1/audio/transcriptions",
                files={"file": ("test.wav", wav_bytes, "audio/wav")},
                data={"model": "test-model"},
            )

        assert response.status_code == 200
        body = response.json()
        assert "R$2.500" in body["text"]
        assert "10%" in body["text"]
        # Audio recebido pelo scheduler: tamanho similar (16kHz -> 16kHz, sem resample)
        assert scheduler.last_request is not None


class TestFullPipelineITNDisabled:
    """itn=false -> texto NAO e transformado (raw do scheduler)."""

    async def test_full_pipeline_itn_disabled(self) -> None:
        wav_bytes = _generate_wav(sample_rate=16000, duration=0.5)
        result = _make_itn_batch_result()
        scheduler = MockScheduler(result)
        app = _create_test_app(scheduler, itn_stage=MockITNStage())

        async with httpx.AsyncClient(
            transport=httpx.ASGITransport(app=app), base_url="http://test"
        ) as client:
            response = await client.post(
                "/v1/audio/transcriptions",
                files={"file": ("test.wav", wav_bytes, "audio/wav")},
                data={"model": "test-model", "itn": "false"},
            )

        assert response.status_code == 200
        body = response.json()
        # Texto deve estar cru, sem transformacao ITN
        assert "dois mil e quinhentos reais" in body["text"]
        assert "dez por cento" in body["text"]
        assert "R$2.500" not in body["text"]


class TestFullPipelineVerboseJson:
    """verbose_json -> texto principal E textos de segmentos sao ITN-transformados."""

    async def test_full_pipeline_verbose_json_segments_transformed(self) -> None:
        wav_bytes = _generate_wav(sample_rate=16000, duration=0.5)
        result = _make_itn_batch_result()
        scheduler = MockScheduler(result)
        app = _create_test_app(scheduler, itn_stage=MockITNStage())

        async with httpx.AsyncClient(
            transport=httpx.ASGITransport(app=app), base_url="http://test"
        ) as client:
            response = await client.post(
                "/v1/audio/transcriptions",
                files={"file": ("test.wav", wav_bytes, "audio/wav")},
                data={"model": "test-model", "response_format": "verbose_json"},
            )

        assert response.status_code == 200
        body = response.json()
        # Texto principal transformado
        assert "R$2.500" in body["text"]
        assert "10%" in body["text"]
        # Segmentos tambem transformados
        assert "R$2.500" in body["segments"][0]["text"]
        assert "10%" in body["segments"][1]["text"]


class TestFullPipelineSrtFormat:
    """SRT format -> legendas contem texto ITN-transformado."""

    async def test_full_pipeline_srt_format_transformed(self) -> None:
        wav_bytes = _generate_wav(sample_rate=16000, duration=0.5)
        result = _make_itn_batch_result()
        scheduler = MockScheduler(result)
        app = _create_test_app(scheduler, itn_stage=MockITNStage())

        async with httpx.AsyncClient(
            transport=httpx.ASGITransport(app=app), base_url="http://test"
        ) as client:
            response = await client.post(
                "/v1/audio/transcriptions",
                files={"file": ("test.wav", wav_bytes, "audio/wav")},
                data={"model": "test-model", "response_format": "srt"},
            )

        assert response.status_code == 200
        srt_text = response.text
        # SRT deve conter texto transformado
        assert "R$2.500" in srt_text
        assert "10%" in srt_text
        # E ter formato SRT com timestamps
        assert "00:00:00,000 --> " in srt_text


class TestFullPipelineVttFormat:
    """VTT format -> legendas contem texto ITN-transformado."""

    async def test_full_pipeline_vtt_format_transformed(self) -> None:
        wav_bytes = _generate_wav(sample_rate=16000, duration=0.5)
        result = _make_itn_batch_result()
        scheduler = MockScheduler(result)
        app = _create_test_app(scheduler, itn_stage=MockITNStage())

        async with httpx.AsyncClient(
            transport=httpx.ASGITransport(app=app), base_url="http://test"
        ) as client:
            response = await client.post(
                "/v1/audio/transcriptions",
                files={"file": ("test.wav", wav_bytes, "audio/wav")},
                data={"model": "test-model", "response_format": "vtt"},
            )

        assert response.status_code == 200
        vtt_text = response.text
        assert vtt_text.startswith("WEBVTT\n")
        assert "R$2.500" in vtt_text
        assert "10%" in vtt_text


class TestFullPipelineTextFormat:
    """text format -> texto puro com ITN-transformado."""

    async def test_full_pipeline_text_format(self) -> None:
        wav_bytes = _generate_wav(sample_rate=16000, duration=0.5)
        result = _make_itn_batch_result()
        scheduler = MockScheduler(result)
        app = _create_test_app(scheduler, itn_stage=MockITNStage())

        async with httpx.AsyncClient(
            transport=httpx.ASGITransport(app=app), base_url="http://test"
        ) as client:
            response = await client.post(
                "/v1/audio/transcriptions",
                files={"file": ("test.wav", wav_bytes, "audio/wav")},
                data={"model": "test-model", "response_format": "text"},
            )

        assert response.status_code == 200
        assert "R$2.500" in response.text
        assert "10%" in response.text


class TestFullPipelinePreprocessingOnly:
    """Apenas preprocessing configurado (sem post-processing) -> texto raw."""

    async def test_full_pipeline_preprocessing_only(self) -> None:
        wav_bytes = _generate_wav(sample_rate=44100, duration=0.5)
        result = _make_itn_batch_result()
        scheduler = MockScheduler(result)
        app = _create_test_app(scheduler, with_postprocessing=False)

        async with httpx.AsyncClient(
            transport=httpx.ASGITransport(app=app), base_url="http://test"
        ) as client:
            response = await client.post(
                "/v1/audio/transcriptions",
                files={"file": ("test.wav", wav_bytes, "audio/wav")},
                data={"model": "test-model"},
            )

        assert response.status_code == 200
        body = response.json()
        # Texto deve estar cru (sem post-processing)
        assert "dois mil e quinhentos reais" in body["text"]
        assert "R$2.500" not in body["text"]
        # Mas preprocessing deve ter rodado (audio menor por causa do downsample)
        assert scheduler.last_request is not None
        assert len(scheduler.last_request.audio_data) < len(wav_bytes)


class TestFullPipelineNoPipelines:
    """Nenhum pipeline configurado -> funciona como antes (backwards compatible)."""

    async def test_full_pipeline_no_pipelines(self) -> None:
        scheduler_mock = MagicMock()
        scheduler_mock.transcribe = AsyncMock(return_value=_make_raw_batch_result())
        app = create_app(
            registry=_make_registry(),
            scheduler=scheduler_mock,
        )

        async with httpx.AsyncClient(
            transport=httpx.ASGITransport(app=app), base_url="http://test"
        ) as client:
            response = await client.post(
                "/v1/audio/transcriptions",
                files={"file": ("audio.wav", b"fake-audio", "audio/wav")},
                data={"model": "test-model"},
            )

        assert response.status_code == 200
        body = response.json()
        assert body["text"] == "Ola, como posso ajudar?"
