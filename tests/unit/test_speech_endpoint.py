"""Testes do endpoint POST /v1/audio/speech e tts_converters.

Valida:
- Pydantic model SpeechRequest (validacao, defaults)
- build_tts_proto_request e tts_proto_chunks_to_result (conversores)
- Rota POST /v1/audio/speech (sucesso, erros, formatos)
- _pcm_to_wav (conversao PCM -> WAV)
"""

from __future__ import annotations

import struct
from unittest.mock import AsyncMock, MagicMock, patch

import httpx
import pytest

from macaw._types import ModelType, TTSSpeechResult
from macaw.server.app import create_app
from macaw.server.models.speech import SpeechRequest

# ─── Helpers ───


def _make_mock_registry(*, model_type: ModelType = ModelType.TTS) -> MagicMock:
    registry = MagicMock()
    manifest = MagicMock()
    manifest.model_type = model_type
    registry.get_manifest.return_value = manifest
    return registry


def _make_mock_worker_manager(*, has_worker: bool = True) -> MagicMock:
    manager = MagicMock()
    if has_worker:
        worker = MagicMock()
        worker.port = 50051
        worker.worker_id = "tts-worker-1"
        manager.get_ready_worker.return_value = worker
    else:
        manager.get_ready_worker.return_value = None
    return manager


def _make_tts_result(
    audio_data: bytes = b"\x00\x01" * 100,
    sample_rate: int = 24000,
    duration: float = 0.5,
    voice: str = "default",
) -> TTSSpeechResult:
    return TTSSpeechResult(
        audio_data=audio_data,
        sample_rate=sample_rate,
        duration=duration,
        voice=voice,
    )


# ─── SpeechRequest Model ───


class TestSpeechRequest:
    def test_required_fields(self) -> None:
        req = SpeechRequest(model="kokoro-v1", input="Hello")
        assert req.model == "kokoro-v1"
        assert req.input == "Hello"

    def test_defaults(self) -> None:
        req = SpeechRequest(model="kokoro-v1", input="Hello")
        assert req.voice == "default"
        assert req.response_format == "wav"
        assert req.speed == 1.0

    def test_custom_values(self) -> None:
        req = SpeechRequest(
            model="kokoro-v1",
            input="Hello",
            voice="alloy",
            response_format="pcm",
            speed=1.5,
        )
        assert req.voice == "alloy"
        assert req.response_format == "pcm"
        assert req.speed == 1.5

    def test_speed_min_validation(self) -> None:
        with pytest.raises(Exception):  # noqa: B017
            SpeechRequest(model="m", input="t", speed=0.1)

    def test_speed_max_validation(self) -> None:
        with pytest.raises(Exception):  # noqa: B017
            SpeechRequest(model="m", input="t", speed=5.0)

    def test_speed_boundary_min(self) -> None:
        req = SpeechRequest(model="m", input="t", speed=0.25)
        assert req.speed == 0.25

    def test_speed_boundary_max(self) -> None:
        req = SpeechRequest(model="m", input="t", speed=4.0)
        assert req.speed == 4.0


# ─── TTS Converters ───


class TestTTSConverters:
    def test_build_tts_proto_request(self) -> None:
        from macaw.scheduler.tts_converters import build_tts_proto_request

        proto = build_tts_proto_request(
            request_id="req-123",
            text="Ola mundo",
            voice="default",
            sample_rate=24000,
            speed=1.0,
        )
        assert proto.request_id == "req-123"
        assert proto.text == "Ola mundo"
        assert proto.voice == "default"
        assert proto.sample_rate == 24000
        assert proto.speed == 1.0

    def test_build_tts_proto_request_custom_speed(self) -> None:
        from macaw.scheduler.tts_converters import build_tts_proto_request

        proto = build_tts_proto_request(
            request_id="req-456",
            text="Rapido",
            voice="alloy",
            sample_rate=22050,
            speed=2.0,
        )
        assert proto.speed == 2.0
        assert proto.voice == "alloy"
        assert proto.sample_rate == 22050

    def test_tts_proto_chunks_to_result_single_chunk(self) -> None:
        from macaw.scheduler.tts_converters import tts_proto_chunks_to_result

        audio = b"\x00\x01" * 50
        result = tts_proto_chunks_to_result(
            [audio],
            sample_rate=24000,
            voice="default",
            total_duration=0.5,
        )
        assert result.audio_data == audio
        assert result.sample_rate == 24000
        assert result.voice == "default"
        assert result.duration == 0.5

    def test_tts_proto_chunks_to_result_multiple_chunks(self) -> None:
        from macaw.scheduler.tts_converters import tts_proto_chunks_to_result

        chunk1 = b"\x00\x01" * 50
        chunk2 = b"\x02\x03" * 50
        result = tts_proto_chunks_to_result(
            [chunk1, chunk2],
            sample_rate=24000,
            voice="alloy",
            total_duration=1.0,
        )
        assert result.audio_data == chunk1 + chunk2
        assert result.duration == 1.0

    def test_tts_proto_chunks_to_result_empty(self) -> None:
        from macaw.scheduler.tts_converters import tts_proto_chunks_to_result

        result = tts_proto_chunks_to_result(
            [],
            sample_rate=24000,
            voice="default",
            total_duration=0.0,
        )
        assert result.audio_data == b""
        assert result.duration == 0.0


# ─── PCM to WAV ───


class TestPcmToWav:
    def test_wav_header_structure(self) -> None:
        from macaw.server.routes.speech import _pcm_to_wav

        pcm = b"\x00\x01" * 100  # 200 bytes of PCM data
        wav = _pcm_to_wav(pcm, sample_rate=24000)

        # RIFF header
        assert wav[:4] == b"RIFF"
        riff_size = struct.unpack("<I", wav[4:8])[0]
        assert riff_size == 36 + len(pcm)
        assert wav[8:12] == b"WAVE"

        # fmt subchunk
        assert wav[12:16] == b"fmt "
        fmt_size = struct.unpack("<I", wav[16:20])[0]
        assert fmt_size == 16  # PCM
        audio_format = struct.unpack("<H", wav[20:22])[0]
        assert audio_format == 1  # PCM
        num_channels = struct.unpack("<H", wav[22:24])[0]
        assert num_channels == 1  # mono
        sample_rate = struct.unpack("<I", wav[24:28])[0]
        assert sample_rate == 24000
        byte_rate = struct.unpack("<I", wav[28:32])[0]
        assert byte_rate == 24000 * 1 * 2  # sample_rate * channels * bytes_per_sample
        block_align = struct.unpack("<H", wav[32:34])[0]
        assert block_align == 2  # channels * bytes_per_sample
        bits_per_sample = struct.unpack("<H", wav[34:36])[0]
        assert bits_per_sample == 16

        # data subchunk
        assert wav[36:40] == b"data"
        data_size = struct.unpack("<I", wav[40:44])[0]
        assert data_size == len(pcm)
        assert wav[44:] == pcm

    def test_wav_total_size(self) -> None:
        from macaw.server.routes.speech import _pcm_to_wav

        pcm = b"\x00" * 480  # 480 bytes
        wav = _pcm_to_wav(pcm, sample_rate=16000)
        # WAV = 44 byte header + PCM data
        assert len(wav) == 44 + 480

    def test_wav_empty_pcm(self) -> None:
        from macaw.server.routes.speech import _pcm_to_wav

        wav = _pcm_to_wav(b"", sample_rate=24000)
        assert len(wav) == 44  # header only
        assert wav[36:40] == b"data"
        data_size = struct.unpack("<I", wav[40:44])[0]
        assert data_size == 0


# ─── POST /v1/audio/speech Route ───


class TestSpeechRoute:
    async def test_speech_returns_wav(self) -> None:
        registry = _make_mock_registry()
        manager = _make_mock_worker_manager()
        tts_result = _make_tts_result()

        app = create_app(registry=registry, worker_manager=manager)

        with patch(
            "macaw.server.routes.speech._synthesize_via_grpc",
            new_callable=AsyncMock,
            return_value=tts_result,
        ):
            async with httpx.AsyncClient(
                transport=httpx.ASGITransport(app=app),
                base_url="http://test",
            ) as client:
                response = await client.post(
                    "/v1/audio/speech",
                    json={"model": "kokoro-v1", "input": "Ola mundo"},
                )

        assert response.status_code == 200
        assert response.headers["content-type"] == "audio/wav"
        # WAV starts with RIFF header
        assert response.content[:4] == b"RIFF"

    async def test_speech_returns_pcm(self) -> None:
        registry = _make_mock_registry()
        manager = _make_mock_worker_manager()
        audio_data = b"\x00\x01" * 100
        tts_result = _make_tts_result(audio_data=audio_data)

        app = create_app(registry=registry, worker_manager=manager)

        with patch(
            "macaw.server.routes.speech._synthesize_via_grpc",
            new_callable=AsyncMock,
            return_value=tts_result,
        ):
            async with httpx.AsyncClient(
                transport=httpx.ASGITransport(app=app),
                base_url="http://test",
            ) as client:
                response = await client.post(
                    "/v1/audio/speech",
                    json={
                        "model": "kokoro-v1",
                        "input": "Ola",
                        "response_format": "pcm",
                    },
                )

        assert response.status_code == 200
        assert response.headers["content-type"] == "audio/pcm"
        assert response.content == audio_data

    async def test_speech_empty_input_returns_400(self) -> None:
        registry = _make_mock_registry()
        manager = _make_mock_worker_manager()

        app = create_app(
            registry=registry,
            worker_manager=manager,
        )

        async with httpx.AsyncClient(
            transport=httpx.ASGITransport(app=app, raise_app_exceptions=False),
            base_url="http://test",
        ) as client:
            response = await client.post(
                "/v1/audio/speech",
                json={"model": "kokoro-v1", "input": "   "},
            )

        assert response.status_code == 400
        assert "vazio" in response.json()["error"]["message"]

    async def test_speech_invalid_format_returns_400(self) -> None:
        registry = _make_mock_registry()
        manager = _make_mock_worker_manager()

        app = create_app(
            registry=registry,
            worker_manager=manager,
        )

        async with httpx.AsyncClient(
            transport=httpx.ASGITransport(app=app, raise_app_exceptions=False),
            base_url="http://test",
        ) as client:
            response = await client.post(
                "/v1/audio/speech",
                json={
                    "model": "kokoro-v1",
                    "input": "Hello",
                    "response_format": "mp3",
                },
            )

        assert response.status_code == 400
        assert "response_format" in response.json()["error"]["message"]

    async def test_speech_model_not_found_returns_404(self) -> None:
        from macaw.exceptions import ModelNotFoundError

        registry = MagicMock()
        registry.get_manifest.side_effect = ModelNotFoundError("unknown-model")
        manager = _make_mock_worker_manager()

        app = create_app(
            registry=registry,
            worker_manager=manager,
        )

        async with httpx.AsyncClient(
            transport=httpx.ASGITransport(app=app, raise_app_exceptions=False),
            base_url="http://test",
        ) as client:
            response = await client.post(
                "/v1/audio/speech",
                json={"model": "unknown-model", "input": "Hello"},
            )

        assert response.status_code == 404

    async def test_speech_stt_model_returns_404(self) -> None:
        registry = _make_mock_registry(model_type=ModelType.STT)
        manager = _make_mock_worker_manager()

        app = create_app(
            registry=registry,
            worker_manager=manager,
        )

        async with httpx.AsyncClient(
            transport=httpx.ASGITransport(app=app, raise_app_exceptions=False),
            base_url="http://test",
        ) as client:
            response = await client.post(
                "/v1/audio/speech",
                json={"model": "faster-whisper-tiny", "input": "Hello"},
            )

        assert response.status_code == 404

    async def test_speech_no_worker_returns_503(self) -> None:
        registry = _make_mock_registry()
        manager = _make_mock_worker_manager(has_worker=False)

        app = create_app(
            registry=registry,
            worker_manager=manager,
        )

        async with httpx.AsyncClient(
            transport=httpx.ASGITransport(app=app, raise_app_exceptions=False),
            base_url="http://test",
        ) as client:
            response = await client.post(
                "/v1/audio/speech",
                json={"model": "kokoro-v1", "input": "Hello"},
            )

        assert response.status_code == 503

    async def test_speech_passes_voice_parameter(self) -> None:
        registry = _make_mock_registry()
        manager = _make_mock_worker_manager()
        tts_result = _make_tts_result()

        app = create_app(registry=registry, worker_manager=manager)

        with patch(
            "macaw.server.routes.speech._synthesize_via_grpc",
            new_callable=AsyncMock,
            return_value=tts_result,
        ) as mock_grpc:
            async with httpx.AsyncClient(
                transport=httpx.ASGITransport(app=app),
                base_url="http://test",
            ) as client:
                await client.post(
                    "/v1/audio/speech",
                    json={
                        "model": "kokoro-v1",
                        "input": "Hello",
                        "voice": "alloy",
                    },
                )

        mock_grpc.assert_awaited_once()
        call_kwargs = mock_grpc.call_args[1]
        assert call_kwargs["voice"] == "alloy"

    async def test_speech_passes_speed_parameter(self) -> None:
        registry = _make_mock_registry()
        manager = _make_mock_worker_manager()
        tts_result = _make_tts_result()

        app = create_app(registry=registry, worker_manager=manager)

        with patch(
            "macaw.server.routes.speech._synthesize_via_grpc",
            new_callable=AsyncMock,
            return_value=tts_result,
        ) as mock_grpc:
            async with httpx.AsyncClient(
                transport=httpx.ASGITransport(app=app),
                base_url="http://test",
            ) as client:
                await client.post(
                    "/v1/audio/speech",
                    json={
                        "model": "kokoro-v1",
                        "input": "Hello",
                        "speed": 1.5,
                    },
                )

        # Verify the proto request was built with the right speed
        mock_grpc.assert_awaited_once()
        call_kwargs = mock_grpc.call_args[1]
        assert "proto_request" in call_kwargs
        assert call_kwargs["proto_request"].speed == 1.5

    async def test_speech_missing_model_returns_422(self) -> None:
        registry = _make_mock_registry()
        manager = _make_mock_worker_manager()

        app = create_app(registry=registry, worker_manager=manager)

        async with httpx.AsyncClient(
            transport=httpx.ASGITransport(app=app),
            base_url="http://test",
        ) as client:
            response = await client.post(
                "/v1/audio/speech",
                json={"input": "Hello"},
            )

        assert response.status_code == 422

    async def test_speech_missing_input_returns_422(self) -> None:
        registry = _make_mock_registry()
        manager = _make_mock_worker_manager()

        app = create_app(registry=registry, worker_manager=manager)

        async with httpx.AsyncClient(
            transport=httpx.ASGITransport(app=app),
            base_url="http://test",
        ) as client:
            response = await client.post(
                "/v1/audio/speech",
                json={"model": "kokoro-v1"},
            )

        assert response.status_code == 422

    async def test_speech_worker_manager_not_configured(self) -> None:
        registry = _make_mock_registry()
        app = create_app(registry=registry)  # No worker_manager

        async with httpx.AsyncClient(
            transport=httpx.ASGITransport(app=app, raise_app_exceptions=False),
            base_url="http://test",
        ) as client:
            response = await client.post(
                "/v1/audio/speech",
                json={"model": "kokoro-v1", "input": "Hello"},
            )

        assert response.status_code == 500


# ─── get_worker_manager dependency ───


class TestGetWorkerManagerDependency:
    def test_returns_manager_from_state(self) -> None:
        from macaw.server.dependencies import get_worker_manager

        mock_request = MagicMock()
        mock_manager = MagicMock()
        mock_request.app.state.worker_manager = mock_manager

        result = get_worker_manager(mock_request)
        assert result is mock_manager

    def test_raises_if_not_configured(self) -> None:
        from macaw.server.dependencies import get_worker_manager

        mock_request = MagicMock()
        mock_request.app.state.worker_manager = None

        with pytest.raises(RuntimeError, match="WorkerManager"):
            get_worker_manager(mock_request)
