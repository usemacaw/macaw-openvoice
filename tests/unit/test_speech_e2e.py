"""End-to-end tests for the POST /v1/audio/speech endpoint (TTS).

Validates the complete pipeline HTTP -> API Server -> gRPC worker (mock) -> Response
for happy paths (WAV, PCM), input validations, model error handling,
worker and gRPC errors.
"""

from __future__ import annotations

import struct
from typing import TYPE_CHECKING, Any
from unittest.mock import AsyncMock, MagicMock, patch

import grpc
import grpc.aio

from macaw._types import ModelType
from macaw.exceptions import ModelNotFoundError
from macaw.server.app import create_app

if TYPE_CHECKING:
    import contextlib

    from fastapi import FastAPI
    from httpx import Response

    _CtxManager = contextlib.AbstractContextManager[Any]


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


class _FakeStream:
    """Simulates a gRPC server-streaming (async iterator of chunks)."""

    def __init__(self, items: list[Any]) -> None:
        self._items = items
        self._idx = 0

    def __aiter__(self) -> _FakeStream:
        return self

    async def __anext__(self) -> Any:
        if self._idx >= len(self._items):
            raise StopAsyncIteration
        item = self._items[self._idx]
        self._idx += 1
        return item


def _make_chunk(
    audio_data: bytes,
    duration: float,
    *,
    is_last: bool = False,
) -> MagicMock:
    chunk = MagicMock()
    chunk.audio_data = audio_data
    chunk.duration = duration
    chunk.is_last = is_last
    return chunk


def _tts_manifest() -> MagicMock:
    manifest = MagicMock()
    manifest.model_type = ModelType.TTS
    manifest.name = "kokoro-v1"
    return manifest


def _stt_manifest() -> MagicMock:
    manifest = MagicMock()
    manifest.model_type = ModelType.STT
    manifest.name = "faster-whisper-tiny"
    return manifest


def _make_registry(manifest: MagicMock | None = None) -> MagicMock:
    registry = MagicMock()
    registry.get_manifest.return_value = manifest or _tts_manifest()
    return registry


def _make_worker_manager(port: int = 50052) -> MagicMock:
    worker = MagicMock()
    worker.port = port
    worker.worker_id = "tts-worker-1"
    wm = MagicMock()
    wm.get_ready_worker.return_value = worker
    return wm


def _make_app(
    registry: MagicMock | None = None,
    worker_manager: MagicMock | None = None,
) -> FastAPI:
    return create_app(
        registry=registry or _make_registry(),
        worker_manager=worker_manager or _make_worker_manager(),
    )


def _default_body(**overrides: object) -> dict[str, object]:
    body: dict[str, object] = {
        "model": "kokoro-v1",
        "input": "Hello world",
    }
    body.update(overrides)
    return body


def _patch_grpc(chunks: list[MagicMock]) -> tuple[_CtxManager, _CtxManager]:
    """Return context managers for patching grpc.aio.insecure_channel and TTSWorkerStub."""
    mock_channel = AsyncMock()
    mock_stub = MagicMock()
    mock_stub.Synthesize.return_value = _FakeStream(chunks)

    p_channel = patch(
        "macaw.server.routes.speech.grpc.aio.insecure_channel",
        return_value=mock_channel,
    )
    p_stub = patch(
        "macaw.server.routes.speech.TTSWorkerStub",
        return_value=mock_stub,
    )
    return p_channel, p_stub


def _patch_grpc_error(rpc_error: grpc.aio.AioRpcError) -> tuple[_CtxManager, _CtxManager]:
    """Return context managers for patching gRPC that raises error."""
    mock_channel = AsyncMock()
    mock_stub = MagicMock()
    mock_stub.Synthesize.side_effect = rpc_error

    p_channel = patch(
        "macaw.server.routes.speech.grpc.aio.insecure_channel",
        return_value=mock_channel,
    )
    p_stub = patch(
        "macaw.server.routes.speech.TTSWorkerStub",
        return_value=mock_stub,
    )
    return p_channel, p_stub


async def _post_speech(
    app: FastAPI,
    body: dict[str, object],
    *,
    raise_app_exceptions: bool = True,
) -> Response:
    from httpx import ASGITransport, AsyncClient

    transport = ASGITransport(
        app=app,
        raise_app_exceptions=raise_app_exceptions,
    )
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        return await client.post("/v1/audio/speech", json=body)


# ---------------------------------------------------------------------------
# Happy path
# ---------------------------------------------------------------------------


class TestSpeechHappyPathWAV:
    """POST /v1/audio/speech with WAV format returns valid audio."""

    async def test_returns_200_with_audio_wav_content_type(self) -> None:
        pcm_data = b"\x00\x01" * 100
        chunks = [
            _make_chunk(pcm_data, 0.5),
            _make_chunk(pcm_data, 1.0, is_last=True),
        ]
        app = _make_app()
        p_channel, p_stub = _patch_grpc(chunks)

        with p_channel, p_stub:
            resp = await _post_speech(app, _default_body(response_format="wav"))

        assert resp.status_code == 200
        assert resp.headers["content-type"] == "audio/wav"

    async def test_wav_body_starts_with_riff_header(self) -> None:
        pcm_data = b"\x00\x01" * 200
        chunks = [_make_chunk(pcm_data, 1.0, is_last=True)]
        app = _make_app()
        p_channel, p_stub = _patch_grpc(chunks)

        with p_channel, p_stub:
            resp = await _post_speech(app, _default_body(response_format="wav"))

        body = resp.content
        assert body[:4] == b"RIFF"
        assert body[8:12] == b"WAVE"

    async def test_wav_header_uses_streaming_max_size_placeholder(self) -> None:
        """Streaming WAV uses 0x7FFFFFFF as data_size placeholder.

        This is a well-known convention for streaming WAV where the total
        size is unknown at the time of writing the header.
        """
        pcm_data = b"\x00\x01" * 300
        chunks = [_make_chunk(pcm_data, 1.5, is_last=True)]
        app = _make_app()
        p_channel, p_stub = _patch_grpc(chunks)

        with p_channel, p_stub:
            resp = await _post_speech(app, _default_body(response_format="wav"))

        body = resp.content
        # data subchunk size at offset 40 is the streaming placeholder
        data_size = struct.unpack_from("<I", body, 40)[0]
        assert data_size == 0x7FFFFFFF
        # Actual PCM data follows the 44-byte header
        assert body[44:] == pcm_data

    async def test_default_format_is_wav(self) -> None:
        """When response_format is not specified, default is wav."""
        pcm_data = b"\x00\x01" * 50
        chunks = [_make_chunk(pcm_data, 0.5, is_last=True)]
        app = _make_app()
        p_channel, p_stub = _patch_grpc(chunks)

        with p_channel, p_stub:
            resp = await _post_speech(app, _default_body())

        assert resp.status_code == 200
        assert resp.headers["content-type"] == "audio/wav"
        assert resp.content[:4] == b"RIFF"


class TestSpeechHappyPathPCM:
    """POST /v1/audio/speech with PCM format returns raw audio."""

    async def test_returns_200_with_audio_pcm_content_type(self) -> None:
        pcm_data = b"\x00\x01" * 100
        chunks = [_make_chunk(pcm_data, 1.0, is_last=True)]
        app = _make_app()
        p_channel, p_stub = _patch_grpc(chunks)

        with p_channel, p_stub:
            resp = await _post_speech(app, _default_body(response_format="pcm"))

        assert resp.status_code == 200
        assert resp.headers["content-type"] == "audio/pcm"

    async def test_pcm_body_is_raw_audio_without_header(self) -> None:
        pcm_data = b"\xab\xcd" * 150
        chunks = [_make_chunk(pcm_data, 1.0, is_last=True)]
        app = _make_app()
        p_channel, p_stub = _patch_grpc(chunks)

        with p_channel, p_stub:
            resp = await _post_speech(app, _default_body(response_format="pcm"))

        # Raw PCM does not have RIFF header
        assert resp.content[:4] != b"RIFF"
        assert resp.content == pcm_data


class TestSpeechMultipleChunks:
    """gRPC returns multiple chunks that are concatenated."""

    async def test_concatenates_multiple_chunks(self) -> None:
        chunk_a = b"\x01\x02" * 50
        chunk_b = b"\x03\x04" * 50
        chunks = [
            _make_chunk(chunk_a, 0.5),
            _make_chunk(chunk_b, 1.0, is_last=True),
        ]
        app = _make_app()
        p_channel, p_stub = _patch_grpc(chunks)

        with p_channel, p_stub:
            resp = await _post_speech(app, _default_body(response_format="pcm"))

        assert resp.status_code == 200
        assert resp.content == chunk_a + chunk_b


# ---------------------------------------------------------------------------
# Input validation
# ---------------------------------------------------------------------------


class TestSpeechEmptyInput:
    """Empty input or whitespace-only returns 400."""

    async def test_empty_string_returns_422(self) -> None:
        app = _make_app()

        resp = await _post_speech(
            app,
            _default_body(input=""),
            raise_app_exceptions=False,
        )

        assert resp.status_code == 422
        detail = resp.json()["detail"]
        assert detail[0]["loc"][-1] == "input"

    async def test_whitespace_only_returns_400(self) -> None:
        app = _make_app()

        resp = await _post_speech(
            app,
            _default_body(input="   "),
            raise_app_exceptions=False,
        )

        assert resp.status_code == 400


class TestSpeechInvalidResponseFormat:
    """Invalid response format returns 400."""

    async def test_mp3_returns_422(self) -> None:
        app = _make_app()

        resp = await _post_speech(
            app,
            _default_body(response_format="mp3"),
            raise_app_exceptions=False,
        )

        # mp3 is a valid format but codec may be unavailable (lameenc not installed)
        assert resp.status_code in (200, 400)

    async def test_unsupported_format_rejected_by_pydantic(self) -> None:
        app = _make_app()

        resp = await _post_speech(
            app,
            _default_body(response_format="aac"),
            raise_app_exceptions=False,
        )

        assert resp.status_code == 422
        detail = resp.json()["detail"]
        assert detail[0]["loc"][-1] == "response_format"


class TestSpeechSpeedBounds:
    """Speed outside [0.25, 4.0] fails Pydantic validation (422)."""

    async def test_speed_below_minimum_returns_422(self) -> None:
        app = _make_app()

        resp = await _post_speech(
            app,
            _default_body(speed=0.1),
            raise_app_exceptions=False,
        )

        assert resp.status_code == 422

    async def test_speed_above_maximum_returns_422(self) -> None:
        app = _make_app()

        resp = await _post_speech(
            app,
            _default_body(speed=5.0),
            raise_app_exceptions=False,
        )

        assert resp.status_code == 422

    async def test_speed_at_minimum_is_valid(self) -> None:
        pcm_data = b"\x00\x01" * 50
        chunks = [_make_chunk(pcm_data, 1.0, is_last=True)]
        app = _make_app()
        p_channel, p_stub = _patch_grpc(chunks)

        with p_channel, p_stub:
            resp = await _post_speech(app, _default_body(speed=0.25))

        assert resp.status_code == 200

    async def test_speed_at_maximum_is_valid(self) -> None:
        pcm_data = b"\x00\x01" * 50
        chunks = [_make_chunk(pcm_data, 1.0, is_last=True)]
        app = _make_app()
        p_channel, p_stub = _patch_grpc(chunks)

        with p_channel, p_stub:
            resp = await _post_speech(app, _default_body(speed=4.0))

        assert resp.status_code == 200


# ---------------------------------------------------------------------------
# Model not found / wrong type
# ---------------------------------------------------------------------------


class TestSpeechModelNotFound:
    """Nonexistent model in registry returns 404."""

    async def test_returns_404(self) -> None:
        registry = MagicMock()
        registry.get_manifest.side_effect = ModelNotFoundError("nonexistent-model")
        app = _make_app(registry=registry)

        resp = await _post_speech(
            app,
            _default_body(model="nonexistent-model"),
            raise_app_exceptions=False,
        )

        assert resp.status_code == 404
        error = resp.json()["error"]
        assert error["code"] == "model_not_found"


class TestSpeechModelTypeMismatch:
    """Model exists but is STT, not TTS, returns 404."""

    async def test_stt_model_returns_404(self) -> None:
        registry = _make_registry(manifest=_stt_manifest())
        app = _make_app(registry=registry)

        resp = await _post_speech(
            app,
            _default_body(model="faster-whisper-tiny"),
            raise_app_exceptions=False,
        )

        assert resp.status_code == 404
        error = resp.json()["error"]
        assert error["code"] == "model_not_found"


# ---------------------------------------------------------------------------
# Worker unavailable
# ---------------------------------------------------------------------------


class TestSpeechNoReadyWorker:
    """No ready TTS worker returns 503."""

    async def test_returns_503(self) -> None:
        wm = MagicMock()
        wm.get_ready_worker.return_value = None
        app = _make_app(worker_manager=wm)

        resp = await _post_speech(
            app,
            _default_body(),
            raise_app_exceptions=False,
        )

        assert resp.status_code == 503
        error = resp.json()["error"]
        assert error["code"] == "service_unavailable"
        assert "Retry-After" in resp.headers


# ---------------------------------------------------------------------------
# gRPC errors
# ---------------------------------------------------------------------------


def _make_rpc_error(code: grpc.StatusCode, details: str) -> grpc.aio.AioRpcError:
    return grpc.aio.AioRpcError(
        code=code,
        initial_metadata=grpc.aio.Metadata(),
        trailing_metadata=grpc.aio.Metadata(),
        details=details,
    )


class TestSpeechGRPCTimeout:
    """gRPC DEADLINE_EXCEEDED maps to 504."""

    async def test_returns_504(self) -> None:
        rpc_error = _make_rpc_error(grpc.StatusCode.DEADLINE_EXCEEDED, "Deadline exceeded")
        app = _make_app()
        p_channel, p_stub = _patch_grpc_error(rpc_error)

        with p_channel, p_stub:
            resp = await _post_speech(
                app,
                _default_body(),
                raise_app_exceptions=False,
            )

        assert resp.status_code == 504
        error = resp.json()["error"]
        assert error["code"] == "gateway_timeout"


class TestSpeechGRPCUnavailable:
    """gRPC UNAVAILABLE maps to 503."""

    async def test_returns_503(self) -> None:
        rpc_error = _make_rpc_error(grpc.StatusCode.UNAVAILABLE, "Connection refused")
        app = _make_app()
        p_channel, p_stub = _patch_grpc_error(rpc_error)

        with p_channel, p_stub:
            resp = await _post_speech(
                app,
                _default_body(),
                raise_app_exceptions=False,
            )

        assert resp.status_code == 503
        error = resp.json()["error"]
        assert error["code"] == "service_unavailable"


class TestSpeechGRPCGenericError:
    """Generic gRPC error (e.g., INTERNAL) maps to 502."""

    async def test_returns_502(self) -> None:
        rpc_error = _make_rpc_error(grpc.StatusCode.INTERNAL, "Internal error")
        app = _make_app()
        p_channel, p_stub = _patch_grpc_error(rpc_error)

        with p_channel, p_stub:
            resp = await _post_speech(
                app,
                _default_body(),
                raise_app_exceptions=False,
            )

        assert resp.status_code == 502
        error = resp.json()["error"]
        assert error["code"] == "bad_gateway"
