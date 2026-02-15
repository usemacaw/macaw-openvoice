"""Testes do endpoint POST /v1/audio/speech e tts_converters.

Valida:
- Pydantic model SpeechRequest (validacao, defaults)
- build_tts_proto_request e tts_proto_chunks_to_result (conversores)
- Rota POST /v1/audio/speech (sucesso, erros, formatos)
"""

from __future__ import annotations

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


def _make_open_tts_stream_mock(
    audio_data: bytes = b"\x00\x01" * 100,
) -> AsyncMock:
    """Create an AsyncMock for _open_tts_stream that returns
    an empty async iterator and the audio data as first chunk."""

    # Empty async iterator (no more chunks after first_audio_chunk)
    async def _empty_iter():
        return
        yield  # make it a generator

    mock = AsyncMock(
        return_value=(_empty_iter(), audio_data),
    )
    return mock


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

    def test_input_max_length_accepted(self) -> None:
        """Text at the max_length limit is accepted."""
        text = "a" * 4096
        req = SpeechRequest(model="m", input=text)
        assert len(req.input) == 4096

    def test_input_exceeds_max_length_rejected(self) -> None:
        """Text exceeding max_length (4096) is rejected by Pydantic."""
        text = "a" * 4097
        with pytest.raises(Exception):  # noqa: B017
            SpeechRequest(model="m", input=text)


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


# ─── POST /v1/audio/speech Route ───


class TestSpeechRoute:
    async def test_speech_returns_wav(self) -> None:
        registry = _make_mock_registry()
        manager = _make_mock_worker_manager()
        audio_data = b"\x00\x01" * 100

        app = create_app(registry=registry, worker_manager=manager)

        with patch(
            "macaw.server.routes.speech._open_tts_stream",
            _make_open_tts_stream_mock(audio_data=audio_data),
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
        # WAV starts with RIFF header (44-byte header + audio data)
        assert response.content[:4] == b"RIFF"
        assert response.content[44:] == audio_data

    async def test_speech_returns_pcm(self) -> None:
        registry = _make_mock_registry()
        manager = _make_mock_worker_manager()
        audio_data = b"\x00\x01" * 100

        app = create_app(registry=registry, worker_manager=manager)

        with patch(
            "macaw.server.routes.speech._open_tts_stream",
            _make_open_tts_stream_mock(audio_data=audio_data),
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
        assert "cannot be empty" in response.json()["error"]["message"]

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

        app = create_app(registry=registry, worker_manager=manager)

        mock = _make_open_tts_stream_mock()
        with patch("macaw.server.routes.speech._open_tts_stream", mock):
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

        mock.assert_awaited_once()
        call_kwargs = mock.call_args[1]
        # Proto request is built upstream; voice is in the proto
        assert call_kwargs["proto_request"].voice == "alloy"
        # Channel is now pooled and passed from app.state.tts_channels
        assert "channel" in call_kwargs

    async def test_speech_passes_speed_parameter(self) -> None:
        registry = _make_mock_registry()
        manager = _make_mock_worker_manager()

        app = create_app(registry=registry, worker_manager=manager)

        mock = _make_open_tts_stream_mock()
        with patch("macaw.server.routes.speech._open_tts_stream", mock):
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
        mock.assert_awaited_once()
        call_kwargs = mock.call_args[1]
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


# ─── TTS gRPC Channel Pooling ───


class TestTTSChannelPooling:
    def test_get_or_create_creates_channel_on_first_call(self) -> None:
        from macaw.server.grpc_channels import get_or_create_tts_channel

        pool: dict[str, object] = {}
        with patch("macaw.server.grpc_channels.grpc.aio.insecure_channel") as mock_create:
            mock_channel = MagicMock()
            mock_create.return_value = mock_channel

            result = get_or_create_tts_channel(pool, "localhost:50052")

        assert result is mock_channel
        assert pool["localhost:50052"] is mock_channel
        mock_create.assert_called_once()

    def test_get_or_create_reuses_existing_channel(self) -> None:
        from macaw.server.grpc_channels import get_or_create_tts_channel

        existing_channel = MagicMock()
        pool: dict[str, object] = {"localhost:50052": existing_channel}

        with patch("macaw.server.grpc_channels.grpc.aio.insecure_channel") as mock_create:
            result = get_or_create_tts_channel(pool, "localhost:50052")

        assert result is existing_channel
        mock_create.assert_not_called()

    def test_get_or_create_different_addresses_get_different_channels(self) -> None:
        from macaw.server.grpc_channels import get_or_create_tts_channel

        pool: dict[str, object] = {}
        with patch("macaw.server.grpc_channels.grpc.aio.insecure_channel") as mock_create:
            ch1 = MagicMock()
            ch2 = MagicMock()
            mock_create.side_effect = [ch1, ch2]

            result1 = get_or_create_tts_channel(pool, "localhost:50052")
            result2 = get_or_create_tts_channel(pool, "localhost:50053")

        assert result1 is ch1
        assert result2 is ch2
        assert len(pool) == 2

    async def test_close_tts_channels_closes_all(self) -> None:
        from macaw.server.grpc_channels import close_tts_channels

        ch1 = AsyncMock()
        ch2 = AsyncMock()
        pool = {"localhost:50052": ch1, "localhost:50053": ch2}

        await close_tts_channels(pool)

        ch1.close.assert_awaited_once()
        ch2.close.assert_awaited_once()
        assert len(pool) == 0

    async def test_close_tts_channels_handles_close_error(self) -> None:
        from macaw.server.grpc_channels import close_tts_channels

        ch1 = AsyncMock()
        ch1.close.side_effect = RuntimeError("close failed")
        ch2 = AsyncMock()
        pool = {"localhost:50052": ch1, "localhost:50053": ch2}

        await close_tts_channels(pool)

        # Both channels attempted close, pool cleared despite error on ch1
        ch1.close.assert_awaited_once()
        ch2.close.assert_awaited_once()
        assert len(pool) == 0

    async def test_channel_reused_across_requests(self) -> None:
        """Verify the same pooled channel is passed to _open_tts_stream for
        consecutive requests to the same worker address."""
        registry = _make_mock_registry()
        manager = _make_mock_worker_manager()
        app = create_app(registry=registry, worker_manager=manager)

        mock = _make_open_tts_stream_mock()
        with patch("macaw.server.routes.speech._open_tts_stream", mock):
            async with httpx.AsyncClient(
                transport=httpx.ASGITransport(app=app),
                base_url="http://test",
            ) as client:
                await client.post(
                    "/v1/audio/speech",
                    json={"model": "kokoro-v1", "input": "First"},
                )
                # Reset mock to capture second call independently
                first_channel = mock.call_args[1]["channel"]

                mock.reset_mock()
                mock.return_value = _make_open_tts_stream_mock().return_value

                await client.post(
                    "/v1/audio/speech",
                    json={"model": "kokoro-v1", "input": "Second"},
                )
                second_channel = mock.call_args[1]["channel"]

        # Same channel object reused for both requests
        assert first_channel is second_channel

    async def test_app_state_has_tts_channels(self) -> None:
        app = create_app()
        assert hasattr(app.state, "tts_channels")
        assert app.state.tts_channels == {}


# ─── Saved Voice Resolution ───


class TestSpeechWithSavedVoice:
    """Speech route resolves voice_ prefix from VoiceStore."""

    async def test_speech_with_saved_voice_resolves_params(self, tmp_path: object) -> None:
        from macaw.server.voice_store import FileSystemVoiceStore

        voice_store = FileSystemVoiceStore(str(tmp_path))
        await voice_store.save(
            voice_id="abc123",
            name="My Clone",
            voice_type="cloned",
            ref_audio=b"\x00\x01" * 50,
            ref_text="Hello reference",
            instruction="Speak clearly",
            language="en",
        )

        registry = _make_mock_registry()
        manager = _make_mock_worker_manager()

        app = create_app(
            registry=registry,
            worker_manager=manager,
            voice_store=voice_store,
        )

        mock = _make_open_tts_stream_mock()
        with patch("macaw.server.routes.speech._open_tts_stream", mock):
            async with httpx.AsyncClient(
                transport=httpx.ASGITransport(app=app),
                base_url="http://test",
            ) as client:
                await client.post(
                    "/v1/audio/speech",
                    json={
                        "model": "kokoro-v1",
                        "input": "Generate speech",
                        "voice": "voice_abc123",
                    },
                )

        mock.assert_awaited_once()
        proto = mock.call_args[1]["proto_request"]
        # Voice is resolved to "default" (engine-agnostic)
        assert proto.voice == "default"
        # Saved voice params are injected into the proto
        assert proto.ref_text == "Hello reference"
        assert proto.instruction == "Speak clearly"
        assert proto.language == "en"
        assert len(proto.ref_audio) > 0

    async def test_speech_with_nonexistent_saved_voice_returns_404(self, tmp_path: object) -> None:
        from macaw.server.voice_store import FileSystemVoiceStore

        voice_store = FileSystemVoiceStore(str(tmp_path))
        registry = _make_mock_registry()
        manager = _make_mock_worker_manager()

        app = create_app(
            registry=registry,
            worker_manager=manager,
            voice_store=voice_store,
        )

        async with httpx.AsyncClient(
            transport=httpx.ASGITransport(app=app, raise_app_exceptions=False),
            base_url="http://test",
        ) as client:
            resp = await client.post(
                "/v1/audio/speech",
                json={
                    "model": "kokoro-v1",
                    "input": "Hello",
                    "voice": "voice_nonexistent",
                },
            )

        assert resp.status_code == 404

    async def test_speech_saved_voice_plus_inline_ref_audio_returns_400(
        self, tmp_path: object
    ) -> None:
        """Cannot provide both inline ref_audio and a saved cloned voice."""
        import base64

        from macaw.server.voice_store import FileSystemVoiceStore

        voice_store = FileSystemVoiceStore(str(tmp_path))
        await voice_store.save(
            voice_id="conflict-id",
            name="Clone",
            voice_type="cloned",
            ref_audio=b"\x00" * 100,
        )

        registry = _make_mock_registry()
        manager = _make_mock_worker_manager()

        app = create_app(
            registry=registry,
            worker_manager=manager,
            voice_store=voice_store,
        )

        async with httpx.AsyncClient(
            transport=httpx.ASGITransport(app=app, raise_app_exceptions=False),
            base_url="http://test",
        ) as client:
            resp = await client.post(
                "/v1/audio/speech",
                json={
                    "model": "kokoro-v1",
                    "input": "Hello",
                    "voice": "voice_conflict-id",
                    "ref_audio": base64.b64encode(b"\xff" * 50).decode(),
                },
            )

        assert resp.status_code == 400
        assert "inline ref_audio" in resp.json()["error"]["message"]
