"""Tests for the TTSWorkerServicer gRPC and TTS worker components."""

from __future__ import annotations

from typing import TYPE_CHECKING
from unittest.mock import AsyncMock, MagicMock

import grpc
import pytest

from macaw.proto.tts_worker_pb2 import (
    HealthRequest,
    HealthResponse,
    SynthesizeChunk,
    SynthesizeRequest,
)
from macaw.workers.tts.converters import (
    audio_chunk_to_proto,
    health_dict_to_proto_response,
    proto_request_to_synthesize_params,
)
from macaw.workers.tts.servicer import TTSWorkerServicer

if TYPE_CHECKING:
    from collections.abc import AsyncIterator

    from macaw._types import VoiceInfo


class MockTTSBackend:
    """Mock backend for TTS servicer tests.

    Implements the same interface as TTSBackend without inheriting from ABC,
    to avoid importing heavy dependencies in unit tests.
    """

    def __init__(self, *, chunks: list[bytes] | None = None) -> None:
        self._chunks = chunks or [b"\x00\x01" * 100, b"\x02\x03" * 100]
        self._loaded = True
        self._health_status = "ok"

    async def load(self, model_path: str, config: dict[str, object]) -> None:
        self._loaded = True

    async def synthesize(  # type: ignore[override, misc]
        self,
        text: str,
        voice: str = "default",
        *,
        sample_rate: int = 24000,
        speed: float = 1.0,
        options: dict[str, object] | None = None,
    ) -> AsyncIterator[bytes]:
        for chunk in self._chunks:
            yield chunk

    async def voices(self) -> list[VoiceInfo]:
        return []

    async def unload(self) -> None:
        self._loaded = False

    async def health(self) -> dict[str, str]:
        return {"status": self._health_status}


def _make_context() -> MagicMock:
    """Create mock of grpc.aio.ServicerContext."""
    ctx = MagicMock()
    ctx.abort = AsyncMock()
    ctx.cancelled = MagicMock(return_value=False)
    return ctx


# ===================================================================
# TTSWorkerServicer Tests
# ===================================================================


class TestSynthesizeHappyPath:
    """Tests normal synthesis flow."""

    @pytest.fixture()
    def servicer(self) -> TTSWorkerServicer:
        return TTSWorkerServicer(
            backend=MockTTSBackend(),  # type: ignore[arg-type]
            model_name="kokoro-v1",
            engine="kokoro",
        )

    async def test_returns_audio_chunks(self, servicer: TTSWorkerServicer) -> None:
        """Synthesize returns audio chunks from the backend."""
        request = SynthesizeRequest(
            request_id="req-1",
            text="Ola mundo",
            voice="default",
            sample_rate=24000,
            speed=1.0,
        )
        ctx = _make_context()

        chunks: list[SynthesizeChunk] = []
        async for chunk in servicer.Synthesize(request, ctx):
            chunks.append(chunk)

        # 2 chunks de audio + 1 chunk final vazio (is_last=True)
        assert len(chunks) == 3
        assert len(chunks[0].audio_data) > 0
        assert len(chunks[1].audio_data) > 0
        assert chunks[2].is_last is True
        assert chunks[2].audio_data == b""

    async def test_accumulated_duration_increases(self, servicer: TTSWorkerServicer) -> None:
        """Accumulated duration increases with each chunk."""
        request = SynthesizeRequest(
            request_id="req-2",
            text="Ola mundo",
            voice="default",
            sample_rate=24000,
            speed=1.0,
        )
        ctx = _make_context()

        durations: list[float] = []
        async for chunk in servicer.Synthesize(request, ctx):
            durations.append(chunk.duration)

        # Each audio chunk should have duration > 0
        assert durations[0] > 0.0
        # Duration should be monotonically increasing for audio chunks
        assert durations[1] >= durations[0]

    async def test_last_chunk_has_is_last_flag(self, servicer: TTSWorkerServicer) -> None:
        """Last chunk has is_last=True."""
        request = SynthesizeRequest(
            request_id="req-3",
            text="Teste",
            voice="default",
            sample_rate=24000,
            speed=1.0,
        )
        ctx = _make_context()

        last_chunk: SynthesizeChunk | None = None
        async for chunk in servicer.Synthesize(request, ctx):
            last_chunk = chunk

        assert last_chunk is not None
        assert last_chunk.is_last is True

    async def test_non_last_chunks_have_is_last_false(self, servicer: TTSWorkerServicer) -> None:
        """Intermediate chunks have is_last=False."""
        request = SynthesizeRequest(
            request_id="req-4",
            text="Teste",
            voice="default",
            sample_rate=24000,
            speed=1.0,
        )
        ctx = _make_context()

        chunks: list[SynthesizeChunk] = []
        async for chunk in servicer.Synthesize(request, ctx):
            chunks.append(chunk)

        # All except the last should have is_last=False
        for ch in chunks[:-1]:
            assert ch.is_last is False


class TestSynthesizeErrors:
    """Tests error scenarios in synthesis."""

    async def test_empty_text_aborts(self) -> None:
        """Empty text causes abort with INVALID_ARGUMENT."""
        servicer = TTSWorkerServicer(
            backend=MockTTSBackend(),  # type: ignore[arg-type]
            model_name="kokoro-v1",
            engine="kokoro",
        )
        request = SynthesizeRequest(
            request_id="req-err-1",
            text="",
            voice="default",
            sample_rate=24000,
            speed=1.0,
        )
        ctx = _make_context()
        ctx.abort = AsyncMock(
            side_effect=grpc.aio.AbortError(  # type: ignore[attr-defined]
                grpc.StatusCode.INVALID_ARGUMENT, "Text must not be empty"
            )
        )

        with pytest.raises(grpc.aio.AbortError):  # type: ignore[attr-defined]
            async for _chunk in servicer.Synthesize(request, ctx):
                pass  # pragma: no cover

        ctx.abort.assert_called_once_with(
            grpc.StatusCode.INVALID_ARGUMENT,
            "Text must not be empty",
        )

    async def test_whitespace_only_text_aborts(self) -> None:
        """Whitespace-only text causes abort with INVALID_ARGUMENT."""
        servicer = TTSWorkerServicer(
            backend=MockTTSBackend(),  # type: ignore[arg-type]
            model_name="kokoro-v1",
            engine="kokoro",
        )
        request = SynthesizeRequest(
            request_id="req-err-2",
            text="   \t\n  ",
            voice="default",
            sample_rate=24000,
            speed=1.0,
        )
        ctx = _make_context()
        ctx.abort = AsyncMock(
            side_effect=grpc.aio.AbortError(  # type: ignore[attr-defined]
                grpc.StatusCode.INVALID_ARGUMENT, "Text must not be empty"
            )
        )

        with pytest.raises(grpc.aio.AbortError):  # type: ignore[attr-defined]
            async for _chunk in servicer.Synthesize(request, ctx):
                pass  # pragma: no cover

    async def test_backend_error_aborts(self) -> None:
        """Backend error causes abort with INTERNAL."""
        backend = MockTTSBackend()

        async def _failing_synthesize(  # type: ignore[misc]
            text: str,
            voice: str = "default",
            *,
            sample_rate: int = 24000,
            speed: float = 1.0,
            options: dict[str, object] | None = None,
        ) -> AsyncIterator[bytes]:
            raise RuntimeError("GPU OOM")
            yield b""  # pragma: no cover

        backend.synthesize = _failing_synthesize  # type: ignore[assignment,method-assign]

        servicer = TTSWorkerServicer(
            backend=backend,  # type: ignore[arg-type]
            model_name="kokoro-v1",
            engine="kokoro",
        )
        request = SynthesizeRequest(
            request_id="req-err-3",
            text="Ola mundo",
            voice="default",
            sample_rate=24000,
            speed=1.0,
        )
        ctx = _make_context()
        ctx.abort = AsyncMock(
            side_effect=grpc.aio.AbortError(  # type: ignore[attr-defined]
                grpc.StatusCode.INTERNAL, "GPU OOM"
            )
        )

        with pytest.raises(grpc.aio.AbortError):  # type: ignore[attr-defined]
            async for _chunk in servicer.Synthesize(request, ctx):
                pass  # pragma: no cover

        ctx.abort.assert_called_once_with(grpc.StatusCode.INTERNAL, "GPU OOM")

    async def test_cancelled_context_stops_streaming(self) -> None:
        """If context is cancelled, stops sending chunks."""
        # Backend that produces many chunks
        many_chunks = [b"\x00\x01" * 50 for _ in range(100)]
        backend = MockTTSBackend(chunks=many_chunks)

        servicer = TTSWorkerServicer(
            backend=backend,  # type: ignore[arg-type]
            model_name="kokoro-v1",
            engine="kokoro",
        )
        request = SynthesizeRequest(
            request_id="req-cancel-1",
            text="Texto longo",
            voice="default",
            sample_rate=24000,
            speed=1.0,
        )
        ctx = _make_context()
        # Cancel after the second chunk
        call_count = 0

        def _cancelled_after_two() -> bool:
            nonlocal call_count
            call_count += 1
            return call_count > 2

        ctx.cancelled = _cancelled_after_two

        chunks: list[SynthesizeChunk] = []
        async for chunk in servicer.Synthesize(request, ctx):
            chunks.append(chunk)

        # Should have stopped before all 100 chunks
        assert len(chunks) < 100


# ===================================================================
# Health Tests
# ===================================================================


class TestHealth:
    async def test_returns_ok(self) -> None:
        """Health returns ok status when backend is loaded."""
        servicer = TTSWorkerServicer(
            backend=MockTTSBackend(),  # type: ignore[arg-type]
            model_name="kokoro-v1",
            engine="kokoro",
        )
        ctx = _make_context()
        response = await servicer.Health(HealthRequest(), ctx)
        assert response.status == "ok"

    async def test_returns_not_loaded(self) -> None:
        """Health returns not_loaded when backend is not loaded."""
        backend = MockTTSBackend()
        backend._health_status = "not_loaded"
        servicer = TTSWorkerServicer(
            backend=backend,  # type: ignore[arg-type]
            model_name="kokoro-v1",
            engine="kokoro",
        )
        ctx = _make_context()
        response = await servicer.Health(HealthRequest(), ctx)
        assert response.status == "not_loaded"

    async def test_returns_model_name_and_engine(self) -> None:
        """Health returns model name and engine."""
        servicer = TTSWorkerServicer(
            backend=MockTTSBackend(),  # type: ignore[arg-type]
            model_name="kokoro-v1",
            engine="kokoro",
        )
        ctx = _make_context()
        response = await servicer.Health(HealthRequest(), ctx)
        assert response.model_name == "kokoro-v1"
        assert response.engine == "kokoro"


# ===================================================================
# Converter Tests
# ===================================================================


class TestProtoRequestToSynthesizeParams:
    def test_converts_all_fields(self) -> None:
        """Converts all fields from SynthesizeRequest."""
        request = SynthesizeRequest(
            request_id="req-1",
            text="Ola mundo",
            voice="pt-br-female",
            sample_rate=22050,
            speed=1.5,
        )
        params = proto_request_to_synthesize_params(request)
        assert params.text == "Ola mundo"
        assert params.voice == "pt-br-female"
        assert params.sample_rate == 22050
        assert params.speed == 1.5

    def test_defaults_for_empty_voice(self) -> None:
        """Uses default when voice is empty string."""
        request = SynthesizeRequest(
            request_id="req-2",
            text="Teste",
            voice="",
            sample_rate=0,
            speed=0.0,
        )
        params = proto_request_to_synthesize_params(request)
        assert params.voice == "default"
        assert params.sample_rate == 24000
        assert params.speed == 1.0


class TestAudioChunkToProto:
    def test_creates_chunk(self) -> None:
        """Creates correct SynthesizeChunk."""
        chunk = audio_chunk_to_proto(
            audio_data=b"\x00\x01" * 50,
            is_last=False,
            duration=0.5,
        )
        assert isinstance(chunk, SynthesizeChunk)
        assert len(chunk.audio_data) == 100
        assert chunk.is_last is False
        assert chunk.duration == pytest.approx(0.5)

    def test_creates_last_chunk(self) -> None:
        """Creates final chunk with is_last=True."""
        chunk = audio_chunk_to_proto(
            audio_data=b"",
            is_last=True,
            duration=2.5,
        )
        assert chunk.is_last is True
        assert chunk.audio_data == b""
        assert chunk.duration == pytest.approx(2.5)


class TestHealthDictToProtoResponse:
    def test_converts_health(self) -> None:
        """Converts health dict to proto."""
        response = health_dict_to_proto_response(
            {"status": "ok"},
            model_name="kokoro-v1",
            engine="kokoro",
        )
        assert isinstance(response, HealthResponse)
        assert response.status == "ok"
        assert response.model_name == "kokoro-v1"
        assert response.engine == "kokoro"

    def test_unknown_status_default(self) -> None:
        """Uses 'unknown' when status is not in dict."""
        response = health_dict_to_proto_response(
            {},
            model_name="test",
            engine="test",
        )
        assert response.status == "unknown"


# ===================================================================
# Proto Serialization Roundtrip Tests
# ===================================================================


class TestProtoRoundtrip:
    def test_synthesize_request_roundtrip(self) -> None:
        """SynthesizeRequest serializes and deserializes correctly."""
        original = SynthesizeRequest(
            request_id="req-rt-1",
            text="Ola mundo",
            voice="pt-br-female",
            sample_rate=22050,
            speed=1.5,
        )
        serialized = original.SerializeToString()
        restored = SynthesizeRequest()
        restored.ParseFromString(serialized)

        assert restored.request_id == "req-rt-1"
        assert restored.text == "Ola mundo"
        assert restored.voice == "pt-br-female"
        assert restored.sample_rate == 22050
        assert restored.speed == pytest.approx(1.5)

    def test_synthesize_chunk_roundtrip(self) -> None:
        """SynthesizeChunk serializes and deserializes correctly."""
        original = SynthesizeChunk(
            audio_data=b"\x00\x01\x02\x03",
            is_last=False,
            duration=1.234,
        )
        serialized = original.SerializeToString()
        restored = SynthesizeChunk()
        restored.ParseFromString(serialized)

        assert restored.audio_data == b"\x00\x01\x02\x03"
        assert restored.is_last is False
        assert restored.duration == pytest.approx(1.234, abs=1e-3)

    def test_synthesize_chunk_last_roundtrip(self) -> None:
        """SynthesizeChunk with is_last=True serializes correctly."""
        original = SynthesizeChunk(
            audio_data=b"",
            is_last=True,
            duration=5.0,
        )
        serialized = original.SerializeToString()
        restored = SynthesizeChunk()
        restored.ParseFromString(serialized)

        assert restored.is_last is True
        assert restored.audio_data == b""

    def test_health_response_roundtrip(self) -> None:
        """HealthResponse serializes and deserializes correctly."""
        original = HealthResponse(
            status="ok",
            model_name="kokoro-v1",
            engine="kokoro",
        )
        serialized = original.SerializeToString()
        restored = HealthResponse()
        restored.ParseFromString(serialized)

        assert restored.status == "ok"
        assert restored.model_name == "kokoro-v1"
        assert restored.engine == "kokoro"


# ===================================================================
# Factory _create_backend Tests
# ===================================================================


class TestCreateBackend:
    def test_unknown_engine_raises_value_error(self) -> None:
        """Unknown engine raises ValueError."""
        from macaw.workers.tts.main import _create_backend

        with pytest.raises(ValueError, match="Unsupported TTS engine: nonexistent"):
            _create_backend("nonexistent")

    def test_kokoro_returns_backend_instance(self) -> None:
        """Kokoro engine returns KokoroBackend instance."""
        from macaw.workers.tts.kokoro import KokoroBackend
        from macaw.workers.tts.main import _create_backend

        backend = _create_backend("kokoro")
        assert isinstance(backend, KokoroBackend)

    def test_qwen3_tts_returns_backend_instance(self) -> None:
        """Qwen3-TTS engine returns Qwen3TTSBackend instance."""
        from macaw.workers.tts.main import _create_backend
        from macaw.workers.tts.qwen3 import Qwen3TTSBackend

        backend = _create_backend("qwen3-tts")
        assert isinstance(backend, Qwen3TTSBackend)


# ===================================================================
# parse_args Tests
# ===================================================================


class TestParseArgs:
    def test_default_values(self) -> None:
        """Default arguments are correct."""
        from macaw.workers.tts.main import parse_args

        args = parse_args(["--model-path", "/models/kokoro"])
        assert args.port == 50052
        assert args.engine == "kokoro"
        assert args.model_path == "/models/kokoro"
        assert args.engine_config == "{}"

    def test_custom_values(self) -> None:
        """Custom arguments are parsed correctly."""
        import json

        from macaw.workers.tts.main import parse_args

        config = {"device": "cuda", "model_name": "piper-v1", "variant": "custom_voice"}
        args = parse_args(
            [
                "--port",
                "60000",
                "--engine",
                "piper",
                "--model-path",
                "/models/piper",
                "--engine-config",
                json.dumps(config),
            ]
        )
        assert args.port == 60000
        assert args.engine == "piper"
        assert args.model_path == "/models/piper"
        parsed = json.loads(args.engine_config)
        assert parsed["device"] == "cuda"
        assert parsed["model_name"] == "piper-v1"
        assert parsed["variant"] == "custom_voice"
