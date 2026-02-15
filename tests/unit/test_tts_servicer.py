"""Testes para o TTSWorkerServicer gRPC e componentes TTS worker."""

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
    """Backend mock para testes do servicer TTS.

    Implementa a mesma interface que TTSBackend sem herdar da ABC,
    para evitar importar dependencias pesadas em testes unitarios.
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
    """Cria mock de grpc.aio.ServicerContext."""
    ctx = MagicMock()
    ctx.abort = AsyncMock()
    ctx.cancelled = MagicMock(return_value=False)
    return ctx


# ===================================================================
# Testes do TTSWorkerServicer
# ===================================================================


class TestSynthesizeHappyPath:
    """Testa fluxo normal de sintese."""

    @pytest.fixture()
    def servicer(self) -> TTSWorkerServicer:
        return TTSWorkerServicer(
            backend=MockTTSBackend(),  # type: ignore[arg-type]
            model_name="kokoro-v1",
            engine="kokoro",
        )

    async def test_returns_audio_chunks(self, servicer: TTSWorkerServicer) -> None:
        """Synthesize retorna chunks de audio do backend."""
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
        """Duracao acumulada aumenta a cada chunk."""
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

        # Cada chunk de audio deve ter duracao > 0
        assert durations[0] > 0.0
        # Duracao deve ser monotonicamente crescente para chunks de audio
        assert durations[1] >= durations[0]

    async def test_last_chunk_has_is_last_flag(self, servicer: TTSWorkerServicer) -> None:
        """Ultimo chunk tem is_last=True."""
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
        """Chunks intermediarios tem is_last=False."""
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

        # Todos exceto o ultimo devem ter is_last=False
        for ch in chunks[:-1]:
            assert ch.is_last is False


class TestSynthesizeErrors:
    """Testa cenarios de erro na sintese."""

    async def test_empty_text_aborts(self) -> None:
        """Texto vazio causa abort com INVALID_ARGUMENT."""
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
        """Texto com apenas espacos causa abort com INVALID_ARGUMENT."""
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
        """Erro no backend causa abort com INTERNAL."""
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
        """Se context cancelado, para de enviar chunks."""
        # Backend que produz muitos chunks
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
        # Cancelar apos o segundo chunk
        call_count = 0

        def _cancelled_after_two() -> bool:
            nonlocal call_count
            call_count += 1
            return call_count > 2

        ctx.cancelled = _cancelled_after_two

        chunks: list[SynthesizeChunk] = []
        async for chunk in servicer.Synthesize(request, ctx):
            chunks.append(chunk)

        # Deve ter parado antes de todos os 100 chunks
        assert len(chunks) < 100


# ===================================================================
# Testes do Health
# ===================================================================


class TestHealth:
    async def test_returns_ok(self) -> None:
        """Health retorna status ok quando backend carregado."""
        servicer = TTSWorkerServicer(
            backend=MockTTSBackend(),  # type: ignore[arg-type]
            model_name="kokoro-v1",
            engine="kokoro",
        )
        ctx = _make_context()
        response = await servicer.Health(HealthRequest(), ctx)
        assert response.status == "ok"

    async def test_returns_not_loaded(self) -> None:
        """Health retorna not_loaded quando backend nao carregado."""
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
        """Health retorna nome do modelo e engine."""
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
# Testes dos Converters
# ===================================================================


class TestProtoRequestToSynthesizeParams:
    def test_converts_all_fields(self) -> None:
        """Converte todos os campos do SynthesizeRequest."""
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
        """Usa default quando voice e string vazia."""
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
        """Cria SynthesizeChunk correto."""
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
        """Cria chunk final com is_last=True."""
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
        """Converte dict de health para proto."""
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
        """Usa 'unknown' quando status nao esta no dict."""
        response = health_dict_to_proto_response(
            {},
            model_name="test",
            engine="test",
        )
        assert response.status == "unknown"


# ===================================================================
# Testes do Proto Serialization Roundtrip
# ===================================================================


class TestProtoRoundtrip:
    def test_synthesize_request_roundtrip(self) -> None:
        """SynthesizeRequest serializa e desserializa corretamente."""
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
        """SynthesizeChunk serializa e desserializa corretamente."""
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
        """SynthesizeChunk com is_last=True serializa corretamente."""
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
        """HealthResponse serializa e desserializa corretamente."""
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
# Testes do Factory _create_backend
# ===================================================================


class TestCreateBackend:
    def test_unknown_engine_raises_value_error(self) -> None:
        """Engine desconhecida levanta ValueError."""
        from macaw.workers.tts.main import _create_backend

        with pytest.raises(ValueError, match="Unsupported TTS engine: nonexistent"):
            _create_backend("nonexistent")

    def test_kokoro_returns_backend_instance(self) -> None:
        """Kokoro engine retorna instancia de KokoroBackend."""
        from macaw.workers.tts.kokoro import KokoroBackend
        from macaw.workers.tts.main import _create_backend

        backend = _create_backend("kokoro")
        assert isinstance(backend, KokoroBackend)

    def test_qwen3_tts_returns_backend_instance(self) -> None:
        """Qwen3-TTS engine retorna instancia de Qwen3TTSBackend."""
        from macaw.workers.tts.main import _create_backend
        from macaw.workers.tts.qwen3 import Qwen3TTSBackend

        backend = _create_backend("qwen3-tts")
        assert isinstance(backend, Qwen3TTSBackend)


# ===================================================================
# Testes do parse_args
# ===================================================================


class TestParseArgs:
    def test_default_values(self) -> None:
        """Argumentos default sao corretos."""
        from macaw.workers.tts.main import parse_args

        args = parse_args(["--model-path", "/models/kokoro"])
        assert args.port == 50052
        assert args.engine == "kokoro"
        assert args.model_path == "/models/kokoro"
        assert args.engine_config == "{}"

    def test_custom_values(self) -> None:
        """Argumentos customizados sao parseados corretamente."""
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
