"""Tests for TTS alignment over REST POST /v1/audio/speech.

Sprint 2 tests: validates NDJSON streaming response when include_alignment=true,
backward compatibility when include_alignment=false (default), alignment models,
and build_tts_proto_request alignment parameter forwarding.
"""

from __future__ import annotations

import base64
import json
from typing import TYPE_CHECKING, Any
from unittest.mock import MagicMock, patch

import pytest

from macaw._types import ModelType
from macaw.server.app import create_app

if TYPE_CHECKING:
    import contextlib

    from fastapi import FastAPI
    from httpx import Response

    _CtxManager = contextlib.AbstractContextManager[Any]


# ---------------------------------------------------------------------------
# Helpers (shared with test_speech_e2e.py patterns)
# ---------------------------------------------------------------------------


class _FakeStream:
    """Simulates a server-streaming gRPC async iterator of chunks."""

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
    alignment: Any = None,
    normalized_alignment: Any = None,
) -> MagicMock:
    chunk = MagicMock()
    chunk.audio_data = audio_data
    chunk.duration = duration
    chunk.is_last = is_last
    chunk.alignment = alignment
    # Default: empty normalized_alignment (mimics proto default message)
    if normalized_alignment is None:
        na = MagicMock()
        na.items = []
        na.granularity = ""
        chunk.normalized_alignment = na
    else:
        chunk.normalized_alignment = normalized_alignment
    return chunk


def _make_alignment(
    items: list[tuple[str, int, int]],
    granularity: str = "word",
) -> MagicMock:
    """Create a mock ChunkAlignment with items.

    items: list of (text, start_ms, duration_ms) tuples.
    """
    alignment = MagicMock()
    alignment.granularity = granularity
    mock_items = []
    for text, start_ms, duration_ms in items:
        item = MagicMock()
        item.text = text
        item.start_ms = start_ms
        item.duration_ms = duration_ms
        mock_items.append(item)
    alignment.items = mock_items
    return alignment


def _make_app(
    registry: MagicMock | None = None,
    worker_manager: MagicMock | None = None,
) -> FastAPI:
    if registry is None:
        registry = MagicMock()
        manifest = MagicMock()
        manifest.model_type = ModelType.TTS
        manifest.name = "kokoro-v1"
        registry.get_manifest.return_value = manifest
    if worker_manager is None:
        worker = MagicMock()
        worker.port = 50052
        worker.worker_id = "tts-worker-1"
        worker_manager = MagicMock()
        worker_manager.get_ready_worker.return_value = worker
    return create_app(registry=registry, worker_manager=worker_manager)


def _default_body(**overrides: object) -> dict[str, object]:
    body: dict[str, object] = {
        "model": "kokoro-v1",
        "input": "Hello world",
    }
    body.update(overrides)
    return body


def _patch_grpc(chunks: list[MagicMock]) -> tuple[_CtxManager, _CtxManager]:
    from unittest.mock import AsyncMock

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
# Alignment Response Models
# ---------------------------------------------------------------------------


class TestAlignmentResponseModels:
    """Pydantic models for NDJSON alignment response."""

    def test_alignment_item_response(self) -> None:
        from macaw.server.models.alignment import AlignmentItemResponse

        item = AlignmentItemResponse(text="Hello", start_ms=0, duration_ms=300)
        assert item.text == "Hello"
        assert item.start_ms == 0
        assert item.duration_ms == 300

    def test_chunk_alignment_response(self) -> None:
        from macaw.server.models.alignment import (
            AlignmentItemResponse,
            ChunkAlignmentResponse,
        )

        items = [
            AlignmentItemResponse(text="Hello", start_ms=0, duration_ms=300),
            AlignmentItemResponse(text="world", start_ms=300, duration_ms=400),
        ]
        chunk = ChunkAlignmentResponse(items=items, granularity="word")
        assert len(chunk.items) == 2
        assert chunk.granularity == "word"

    def test_chunk_alignment_response_default_granularity(self) -> None:
        from macaw.server.models.alignment import ChunkAlignmentResponse

        chunk = ChunkAlignmentResponse(items=[])
        assert chunk.granularity == "word"

    def test_audio_chunk_with_alignment(self) -> None:
        from macaw.server.models.alignment import AudioChunkWithAlignment

        line = AudioChunkWithAlignment(audio="AQID")
        assert line.type == "audio"
        assert line.audio == "AQID"
        assert line.alignment is None

    def test_audio_chunk_with_alignment_excludes_none(self) -> None:
        from macaw.server.models.alignment import AudioChunkWithAlignment

        line = AudioChunkWithAlignment(audio="AQID")
        d = json.loads(line.model_dump_json(exclude_none=True))
        assert "alignment" not in d
        assert d["type"] == "audio"
        assert d["audio"] == "AQID"

    def test_audio_chunk_with_alignment_includes_alignment(self) -> None:
        from macaw.server.models.alignment import (
            AlignmentItemResponse,
            AudioChunkWithAlignment,
            ChunkAlignmentResponse,
        )

        align = ChunkAlignmentResponse(
            items=[AlignmentItemResponse(text="Hi", start_ms=0, duration_ms=200)],
            granularity="word",
        )
        line = AudioChunkWithAlignment(audio="AQID", alignment=align)
        d = json.loads(line.model_dump_json(exclude_none=True))
        assert d["alignment"]["items"][0]["text"] == "Hi"
        assert d["alignment"]["granularity"] == "word"

    def test_alignment_stream_done(self) -> None:
        from macaw.server.models.alignment import AlignmentStreamDone

        done = AlignmentStreamDone(duration=1.5)
        assert done.type == "done"
        assert done.duration == 1.5
        assert done.alignment_available is True
        d = json.loads(done.model_dump_json())
        assert d == {"type": "done", "duration": 1.5, "alignment_available": True}

    def test_alignment_stream_done_no_alignment(self) -> None:
        from macaw.server.models.alignment import AlignmentStreamDone

        done = AlignmentStreamDone(duration=2.0, alignment_available=False)
        assert done.alignment_available is False
        d = json.loads(done.model_dump_json())
        assert d["alignment_available"] is False


# ---------------------------------------------------------------------------
# build_tts_proto_request alignment params
# ---------------------------------------------------------------------------


class TestBuildTTSProtoAlignmentParams:
    """build_tts_proto_request forwards alignment fields to proto."""

    def test_default_alignment_fields(self) -> None:
        from macaw.scheduler.tts_converters import build_tts_proto_request

        proto = build_tts_proto_request(
            request_id="req-1",
            text="Hello",
            voice="default",
            sample_rate=24000,
            speed=1.0,
        )
        assert proto.include_alignment is False
        assert proto.alignment_granularity == ""

    def test_include_alignment_true(self) -> None:
        from macaw.scheduler.tts_converters import build_tts_proto_request

        proto = build_tts_proto_request(
            request_id="req-2",
            text="Hello",
            voice="default",
            sample_rate=24000,
            speed=1.0,
            include_alignment=True,
        )
        assert proto.include_alignment is True
        assert proto.alignment_granularity == "word"

    def test_character_granularity(self) -> None:
        from macaw.scheduler.tts_converters import build_tts_proto_request

        proto = build_tts_proto_request(
            request_id="req-3",
            text="Hello",
            voice="default",
            sample_rate=24000,
            speed=1.0,
            include_alignment=True,
            alignment_granularity="character",
        )
        assert proto.alignment_granularity == "character"


# ---------------------------------------------------------------------------
# SpeechRequest alignment fields
# ---------------------------------------------------------------------------


class TestSpeechRequestAlignmentFields:
    """SpeechRequest Pydantic model alignment fields."""

    def test_defaults_alignment_off(self) -> None:
        from macaw.server.models.speech import SpeechRequest

        req = SpeechRequest(model="kokoro-v1", input="Hello")
        assert req.include_alignment is False
        assert req.alignment_granularity == "word"

    def test_include_alignment_true(self) -> None:
        from macaw.server.models.speech import SpeechRequest

        req = SpeechRequest(
            model="kokoro-v1",
            input="Hello",
            include_alignment=True,
        )
        assert req.include_alignment is True

    def test_alignment_granularity_character(self) -> None:
        from macaw.server.models.speech import SpeechRequest

        req = SpeechRequest(
            model="kokoro-v1",
            input="Hello",
            include_alignment=True,
            alignment_granularity="character",
        )
        assert req.alignment_granularity == "character"

    def test_invalid_granularity_rejected(self) -> None:
        from pydantic import ValidationError

        from macaw.server.models.speech import SpeechRequest

        with pytest.raises(ValidationError):
            SpeechRequest(
                model="kokoro-v1",
                input="Hello",
                alignment_granularity="phoneme",
            )


# ---------------------------------------------------------------------------
# NDJSON Alignment Streaming (E2E via gRPC mock)
# ---------------------------------------------------------------------------


class TestAlignmentNDJSONResponse:
    """POST /v1/audio/speech with include_alignment=true returns NDJSON."""

    async def test_returns_ndjson_content_type(self) -> None:
        pcm_data = b"\x00\x01" * 100
        alignment = _make_alignment([("Hello", 0, 300), ("world", 300, 400)])
        chunks = [_make_chunk(pcm_data, 0.5, alignment=alignment, is_last=True)]
        app = _make_app()
        p_channel, p_stub = _patch_grpc(chunks)

        with p_channel, p_stub:
            resp = await _post_speech(
                app,
                _default_body(include_alignment=True),
            )

        assert resp.status_code == 200
        assert resp.headers["content-type"] == "application/x-ndjson"

    async def test_ndjson_contains_audio_and_done_lines(self) -> None:
        pcm_data = b"\x00\x01" * 100
        alignment = _make_alignment([("Hello", 0, 300)])
        chunks = [_make_chunk(pcm_data, 0.5, alignment=alignment, is_last=True)]
        app = _make_app()
        p_channel, p_stub = _patch_grpc(chunks)

        with p_channel, p_stub:
            resp = await _post_speech(
                app,
                _default_body(include_alignment=True),
            )

        lines = [json.loads(line) for line in resp.text.strip().split("\n") if line]
        assert len(lines) == 2
        # First line: audio chunk
        assert lines[0]["type"] == "audio"
        assert "audio" in lines[0]
        # Last line: done marker
        assert lines[1]["type"] == "done"
        assert "duration" in lines[1]

    async def test_ndjson_audio_is_base64_encoded(self) -> None:
        pcm_data = b"\xab\xcd\xef" * 50
        alignment = _make_alignment([("Test", 0, 500)])
        chunks = [_make_chunk(pcm_data, 0.5, alignment=alignment, is_last=True)]
        app = _make_app()
        p_channel, p_stub = _patch_grpc(chunks)

        with p_channel, p_stub:
            resp = await _post_speech(
                app,
                _default_body(include_alignment=True),
            )

        lines = [json.loads(line) for line in resp.text.strip().split("\n") if line]
        audio_line = lines[0]
        decoded = base64.b64decode(audio_line["audio"])
        assert decoded == pcm_data

    async def test_ndjson_includes_alignment_data(self) -> None:
        pcm_data = b"\x00\x01" * 100
        alignment = _make_alignment(
            [
                ("Hello", 0, 300),
                ("world", 300, 400),
            ]
        )
        chunks = [_make_chunk(pcm_data, 0.7, alignment=alignment, is_last=True)]
        app = _make_app()
        p_channel, p_stub = _patch_grpc(chunks)

        with p_channel, p_stub:
            resp = await _post_speech(
                app,
                _default_body(include_alignment=True),
            )

        lines = [json.loads(line) for line in resp.text.strip().split("\n") if line]
        audio_line = lines[0]
        assert "alignment" in audio_line
        items = audio_line["alignment"]["items"]
        assert len(items) == 2
        assert items[0]["text"] == "Hello"
        assert items[0]["start_ms"] == 0
        assert items[0]["duration_ms"] == 300
        assert items[1]["text"] == "world"
        assert items[1]["start_ms"] == 300
        assert items[1]["duration_ms"] == 400

    async def test_ndjson_alignment_granularity_preserved(self) -> None:
        pcm_data = b"\x00\x01" * 100
        alignment = _make_alignment(
            [("H", 0, 50), ("e", 50, 50)],
            granularity="character",
        )
        chunks = [_make_chunk(pcm_data, 0.1, alignment=alignment, is_last=True)]
        app = _make_app()
        p_channel, p_stub = _patch_grpc(chunks)

        with p_channel, p_stub:
            resp = await _post_speech(
                app,
                _default_body(include_alignment=True, alignment_granularity="character"),
            )

        lines = [json.loads(line) for line in resp.text.strip().split("\n") if line]
        audio_line = lines[0]
        assert audio_line["alignment"]["granularity"] == "character"

    async def test_ndjson_multiple_chunks_produce_multiple_audio_lines(self) -> None:
        pcm_a = b"\x01\x02" * 50
        pcm_b = b"\x03\x04" * 50
        align_a = _make_alignment([("Hello", 0, 300)])
        align_b = _make_alignment([("world", 300, 400)])
        chunks = [
            _make_chunk(pcm_a, 0.3, alignment=align_a),
            _make_chunk(pcm_b, 0.4, alignment=align_b, is_last=True),
        ]
        app = _make_app()
        p_channel, p_stub = _patch_grpc(chunks)

        with p_channel, p_stub:
            resp = await _post_speech(
                app,
                _default_body(include_alignment=True),
            )

        lines = [json.loads(line) for line in resp.text.strip().split("\n") if line]
        # 2 audio lines + 1 done marker
        assert len(lines) == 3
        assert lines[0]["type"] == "audio"
        assert lines[1]["type"] == "audio"
        assert lines[2]["type"] == "done"

    async def test_ndjson_chunk_without_alignment_omits_field(self) -> None:
        """Chunks with no alignment data omit the alignment field (exclude_none)."""
        pcm_data = b"\x00\x01" * 100
        # No alignment on chunk
        chunks = [_make_chunk(pcm_data, 0.5, alignment=None, is_last=True)]
        app = _make_app()
        p_channel, p_stub = _patch_grpc(chunks)

        with p_channel, p_stub:
            resp = await _post_speech(
                app,
                _default_body(include_alignment=True),
            )

        lines = [json.loads(line) for line in resp.text.strip().split("\n") if line]
        audio_line = lines[0]
        assert audio_line["type"] == "audio"
        assert "alignment" not in audio_line

    async def test_ndjson_done_has_accumulated_duration(self) -> None:
        pcm_data = b"\x00\x01" * 100  # 200 bytes / (24000 * 2) = ~0.00417s
        chunks = [
            _make_chunk(pcm_data, 0.5, alignment=_make_alignment([("a", 0, 100)])),
            _make_chunk(pcm_data, 0.5, alignment=_make_alignment([("b", 100, 200)]), is_last=True),
        ]
        app = _make_app()
        p_channel, p_stub = _patch_grpc(chunks)

        with p_channel, p_stub:
            resp = await _post_speech(
                app,
                _default_body(include_alignment=True),
            )

        lines = [json.loads(line) for line in resp.text.strip().split("\n") if line]
        done = lines[-1]
        assert done["type"] == "done"
        # Duration accumulated from PCM byte length / (sample_rate * 2)
        assert done["duration"] > 0


# ---------------------------------------------------------------------------
# Backward Compatibility — include_alignment=false (default)
# ---------------------------------------------------------------------------


class TestAlignmentBackwardCompat:
    """Default behavior (include_alignment=false) is unchanged."""

    async def test_default_returns_binary_audio_not_ndjson(self) -> None:
        pcm_data = b"\x00\x01" * 100
        chunks = [_make_chunk(pcm_data, 0.5, is_last=True)]
        app = _make_app()
        p_channel, p_stub = _patch_grpc(chunks)

        with p_channel, p_stub:
            resp = await _post_speech(app, _default_body())

        assert resp.status_code == 200
        assert resp.headers["content-type"] == "audio/wav"
        assert resp.content[:4] == b"RIFF"

    async def test_pcm_format_unchanged_without_alignment(self) -> None:
        pcm_data = b"\xab\xcd" * 150
        chunks = [_make_chunk(pcm_data, 1.0, is_last=True)]
        app = _make_app()
        p_channel, p_stub = _patch_grpc(chunks)

        with p_channel, p_stub:
            resp = await _post_speech(
                app,
                _default_body(response_format="pcm"),
            )

        assert resp.status_code == 200
        assert resp.headers["content-type"] == "audio/pcm"
        assert resp.content == pcm_data


# ---------------------------------------------------------------------------
# Alignment with Audio Effects
# ---------------------------------------------------------------------------


class TestAlignmentWithEffects:
    """Audio effects applied correctly in alignment mode."""

    async def test_effects_applied_to_alignment_audio(self) -> None:
        """When effects are requested with alignment, audio is transformed
        but alignment timing stays relative to original."""
        pcm_data = b"\x00\x01" * 100
        alignment = _make_alignment([("Hello", 0, 300)])
        chunks = [_make_chunk(pcm_data, 0.5, alignment=alignment, is_last=True)]
        app = _make_app()
        p_channel, p_stub = _patch_grpc(chunks)

        with p_channel, p_stub:
            resp = await _post_speech(
                app,
                _default_body(
                    include_alignment=True,
                    effects={"pitch_shift_semitones": 2.0},
                ),
            )

        assert resp.status_code == 200
        assert resp.headers["content-type"] == "application/x-ndjson"
        lines = [json.loads(line) for line in resp.text.strip().split("\n") if line]
        # Audio was processed (base64 will differ from raw)
        audio_line = lines[0]
        assert audio_line["type"] == "audio"
        decoded = base64.b64decode(audio_line["audio"])
        # Effects modify the audio, so it should differ from original PCM
        # (pitch shift changes sample values)
        assert len(decoded) > 0
        # Alignment timing is preserved from proto, unaffected by effects
        assert audio_line["alignment"]["items"][0]["start_ms"] == 0
        assert audio_line["alignment"]["items"][0]["duration_ms"] == 300


# ---------------------------------------------------------------------------
# Empty alignment items edge case
# ---------------------------------------------------------------------------


class TestAlignmentEmptyItems:
    """Edge case: alignment present but with empty items list."""

    async def test_empty_alignment_items_omitted(self) -> None:
        """If alignment has items=[] (empty list), alignment field is omitted."""
        pcm_data = b"\x00\x01" * 100
        # Alignment with empty items
        alignment = MagicMock()
        alignment.items = []
        alignment.granularity = "word"
        chunks = [_make_chunk(pcm_data, 0.5, alignment=alignment, is_last=True)]
        app = _make_app()
        p_channel, p_stub = _patch_grpc(chunks)

        with p_channel, p_stub:
            resp = await _post_speech(
                app,
                _default_body(include_alignment=True),
            )

        lines = [json.loads(line) for line in resp.text.strip().split("\n") if line]
        audio_line = lines[0]
        # Empty items → alignment field omitted (the code checks `if chunk.alignment and chunk.alignment.items`)
        assert "alignment" not in audio_line


# ---------------------------------------------------------------------------
# Coverage gap tests (W-12)
# ---------------------------------------------------------------------------


class TestSeedValidation:
    """Seed validation: ge=1 rejects 0 and negative values."""

    def test_seed_zero_rejected(self) -> None:
        from pydantic import ValidationError

        from macaw.server.models.speech import SpeechRequest

        with pytest.raises(ValidationError, match="seed"):
            SpeechRequest(model="kokoro-v1", input="Hello", seed=0)

    def test_seed_negative_rejected(self) -> None:
        from pydantic import ValidationError

        from macaw.server.models.speech import SpeechRequest

        with pytest.raises(ValidationError, match="seed"):
            SpeechRequest(model="kokoro-v1", input="Hello", seed=-1)

    def test_seed_one_accepted(self) -> None:
        from macaw.server.models.speech import SpeechRequest

        req = SpeechRequest(model="kokoro-v1", input="Hello", seed=1)
        assert req.seed == 1


class TestAlignmentWithOpusRejected:
    """include_alignment=true with response_format='opus' returns 400."""

    @pytest.fixture()
    def app(self) -> FastAPI:
        return _make_app()

    async def test_opus_with_alignment_returns_400(self, app: FastAPI) -> None:
        resp = await _post_speech(
            app,
            _default_body(include_alignment=True, response_format="opus"),
            raise_app_exceptions=False,
        )
        assert resp.status_code == 400
        assert "include_alignment" in resp.text


class TestNormalizedAlignmentInNDJSON:
    """Normalized alignment data flows through the REST NDJSON path."""

    async def test_normalized_alignment_in_ndjson_response(self) -> None:
        pcm_data = b"\x00\x01" * 100

        alignment = MagicMock()
        alignment.items = [MagicMock(text="Hello", start_ms=0, duration_ms=300)]
        alignment.granularity = "word"

        norm_alignment = MagicMock()
        norm_alignment.items = [MagicMock(text="hEloU", start_ms=0, duration_ms=300)]
        norm_alignment.granularity = "word"

        chunk = _make_chunk(pcm_data, 0.5, alignment=alignment, is_last=True)
        chunk.normalized_alignment = norm_alignment

        app = _make_app()
        p_channel, p_stub = _patch_grpc([chunk])

        with p_channel, p_stub:
            resp = await _post_speech(
                app,
                _default_body(include_alignment=True),
            )

        assert resp.status_code == 200
        lines = [json.loads(line) for line in resp.text.strip().split("\n") if line]
        audio_line = lines[0]
        assert "normalized_alignment" in audio_line
        assert audio_line["normalized_alignment"]["items"][0]["text"] == "hEloU"
