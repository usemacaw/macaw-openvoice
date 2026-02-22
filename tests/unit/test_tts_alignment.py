"""Tests for TTS alignment feature (Sprint 1).

Tests cover:
- TTSAlignmentItem and TTSChunkResult dataclasses
- TTSEngineCapabilities alignment fields
- Kokoro alignment extraction from MToken timestamps
- Proto conversion with alignment data
- Servicer alignment-aware synthesis path
- TTSBackend default synthesize_with_alignment fallback
"""

from __future__ import annotations

from types import SimpleNamespace
from typing import TYPE_CHECKING
from unittest.mock import AsyncMock, MagicMock

import pytest

from macaw._types import TTSAlignmentItem, TTSChunkResult, TTSEngineCapabilities
from macaw.proto.tts_worker_pb2 import (
    AlignmentItem,
    ChunkAlignment,
    SynthesizeChunk,
    SynthesizeRequest,
)
from macaw.workers.tts.converters import (
    audio_chunk_to_proto,
    proto_request_to_synthesize_params,
)
from macaw.workers.tts.kokoro import _extract_alignment
from macaw.workers.tts.servicer import TTSWorkerServicer

if TYPE_CHECKING:
    from collections.abc import AsyncIterator

    from macaw._types import VoiceInfo


# ===================================================================
# Helpers
# ===================================================================


def _make_token(text: str, start_ts: float | None, end_ts: float | None) -> SimpleNamespace:
    """Create a mock MToken-like object with timing data."""
    return SimpleNamespace(text=text, start_ts=start_ts, end_ts=end_ts)


def _make_result(
    tokens: list[SimpleNamespace] | None = None,
) -> SimpleNamespace:
    """Create a mock KPipeline.Result-like object."""
    return SimpleNamespace(tokens=tokens)


def _make_context() -> MagicMock:
    """Create mock grpc.aio.ServicerContext."""
    ctx = MagicMock()
    ctx.abort = AsyncMock()
    ctx.cancelled = MagicMock(return_value=False)
    return ctx


class MockAlignmentBackend:
    """Backend mock that supports synthesize_with_alignment.

    Returns TTSChunkResult with alignment data for testing the
    servicer alignment path.
    """

    def __init__(
        self,
        *,
        chunks: list[tuple[bytes, tuple[TTSAlignmentItem, ...] | None]] | None = None,
    ) -> None:
        default_alignment = (
            TTSAlignmentItem(text="Hello", start_ms=0, duration_ms=200),
            TTSAlignmentItem(text="world", start_ms=200, duration_ms=300),
        )
        self._chunks = chunks or [
            (b"\x00\x01" * 100, default_alignment),
            (b"\x02\x03" * 100, None),
        ]

    async def capabilities(self) -> TTSEngineCapabilities:
        return TTSEngineCapabilities(supports_alignment=True)

    async def load(self, model_path: str, config: dict[str, object]) -> None:
        pass

    async def synthesize(  # type: ignore[misc]
        self,
        text: str,
        voice: str = "default",
        *,
        sample_rate: int = 24000,
        speed: float = 1.0,
        options: dict[str, object] | None = None,
    ) -> AsyncIterator[bytes]:
        for audio, _alignment in self._chunks:
            yield audio

    async def synthesize_with_alignment(  # type: ignore[misc]
        self,
        text: str,
        voice: str = "default",
        *,
        sample_rate: int = 24000,
        speed: float = 1.0,
        alignment_granularity: str = "word",
        options: dict[str, object] | None = None,
    ) -> AsyncIterator[TTSChunkResult]:
        for audio, alignment in self._chunks:
            yield TTSChunkResult(
                audio=audio,
                alignment=alignment,
                alignment_granularity=alignment_granularity,
            )

    async def voices(self) -> list[VoiceInfo]:
        return []

    async def unload(self) -> None:
        pass

    async def health(self) -> dict[str, str]:
        return {"status": "ok"}


# ===================================================================
# TTSAlignmentItem dataclass
# ===================================================================


class TestTTSAlignmentItem:
    def test_creation(self) -> None:
        item = TTSAlignmentItem(text="Hello", start_ms=0, duration_ms=200)
        assert item.text == "Hello"
        assert item.start_ms == 0
        assert item.duration_ms == 200

    def test_frozen(self) -> None:
        item = TTSAlignmentItem(text="Hello", start_ms=0, duration_ms=200)
        with pytest.raises(AttributeError):
            item.text = "world"  # type: ignore[misc]

    def test_equality(self) -> None:
        a = TTSAlignmentItem(text="Hello", start_ms=0, duration_ms=200)
        b = TTSAlignmentItem(text="Hello", start_ms=0, duration_ms=200)
        assert a == b

    def test_inequality(self) -> None:
        a = TTSAlignmentItem(text="Hello", start_ms=0, duration_ms=200)
        b = TTSAlignmentItem(text="World", start_ms=0, duration_ms=200)
        assert a != b


# ===================================================================
# TTSChunkResult dataclass
# ===================================================================


class TestTTSChunkResult:
    def test_creation_with_defaults(self) -> None:
        result = TTSChunkResult(audio=b"\x00\x01")
        assert result.audio == b"\x00\x01"
        assert result.alignment is None
        assert result.alignment_granularity == "word"

    def test_creation_with_alignment(self) -> None:
        alignment = (TTSAlignmentItem(text="Hi", start_ms=0, duration_ms=100),)
        result = TTSChunkResult(
            audio=b"\x00\x01",
            alignment=alignment,
            alignment_granularity="character",
        )
        assert result.alignment is not None
        assert len(result.alignment) == 1
        assert result.alignment[0].text == "Hi"
        assert result.alignment_granularity == "character"

    def test_frozen(self) -> None:
        result = TTSChunkResult(audio=b"\x00\x01")
        with pytest.raises(AttributeError):
            result.audio = b""  # type: ignore[misc]


# ===================================================================
# TTSEngineCapabilities alignment fields
# ===================================================================


class TestTTSEngineCapabilitiesAlignment:
    def test_default_no_alignment(self) -> None:
        caps = TTSEngineCapabilities()
        assert caps.supports_alignment is False
        assert caps.supports_character_alignment is False

    def test_with_alignment(self) -> None:
        caps = TTSEngineCapabilities(supports_alignment=True)
        assert caps.supports_alignment is True
        assert caps.supports_character_alignment is False

    def test_with_character_alignment(self) -> None:
        caps = TTSEngineCapabilities(
            supports_alignment=True,
            supports_character_alignment=True,
        )
        assert caps.supports_alignment is True
        assert caps.supports_character_alignment is True


# ===================================================================
# _extract_alignment (Kokoro)
# ===================================================================


class TestExtractAlignment:
    def test_extracts_word_timing(self) -> None:
        """Extracts alignment from tokens with valid timing."""
        result = _make_result(
            tokens=[
                _make_token("Hello", 0.0, 0.2),
                _make_token("world", 0.2, 0.5),
            ]
        )
        alignment = _extract_alignment(result)
        assert alignment is not None
        assert len(alignment) == 2
        assert alignment[0] == TTSAlignmentItem(text="Hello", start_ms=0, duration_ms=200)
        assert alignment[1] == TTSAlignmentItem(text="world", start_ms=200, duration_ms=300)

    def test_returns_none_when_no_tokens(self) -> None:
        """Returns None when result has no tokens attribute."""
        result = SimpleNamespace()
        alignment = _extract_alignment(result)
        assert alignment is None

    def test_returns_none_when_tokens_is_none(self) -> None:
        """Returns None when tokens is explicitly None."""
        result = _make_result(tokens=None)
        alignment = _extract_alignment(result)
        assert alignment is None

    def test_skips_tokens_without_timing(self) -> None:
        """Tokens with None start_ts or end_ts are skipped."""
        result = _make_result(
            tokens=[
                _make_token("Hello", 0.0, 0.2),
                _make_token("world", None, None),
                _make_token("!", 0.5, 0.6),
            ]
        )
        alignment = _extract_alignment(result)
        assert alignment is not None
        assert len(alignment) == 2
        assert alignment[0].text == "Hello"
        assert alignment[1].text == "!"

    def test_skips_tokens_with_zero_duration(self) -> None:
        """Tokens where start_ts == end_ts (zero duration) are skipped."""
        result = _make_result(
            tokens=[
                _make_token("Hello", 0.0, 0.2),
                _make_token(",", 0.2, 0.2),  # zero duration punctuation
            ]
        )
        alignment = _extract_alignment(result)
        assert alignment is not None
        assert len(alignment) == 1
        assert alignment[0].text == "Hello"

    def test_skips_empty_text_tokens(self) -> None:
        """Tokens with empty text are skipped."""
        result = _make_result(
            tokens=[
                _make_token("", 0.0, 0.2),
                _make_token("Hello", 0.2, 0.5),
            ]
        )
        alignment = _extract_alignment(result)
        assert alignment is not None
        assert len(alignment) == 1
        assert alignment[0].text == "Hello"

    def test_returns_none_when_all_tokens_filtered(self) -> None:
        """Returns None when all tokens are filtered out."""
        result = _make_result(
            tokens=[
                _make_token("", 0.0, 0.2),
                _make_token(",", 0.2, 0.2),
            ]
        )
        alignment = _extract_alignment(result)
        assert alignment is None

    def test_strips_whitespace_from_token_text(self) -> None:
        """Token text is stripped of leading/trailing whitespace."""
        result = _make_result(
            tokens=[
                _make_token("  Hello  ", 0.0, 0.2),
            ]
        )
        alignment = _extract_alignment(result)
        assert alignment is not None
        assert alignment[0].text == "Hello"

    def test_millisecond_conversion(self) -> None:
        """Timestamps in seconds are correctly converted to milliseconds."""
        result = _make_result(
            tokens=[
                _make_token("word", 1.234, 2.567),
            ]
        )
        alignment = _extract_alignment(result)
        assert alignment is not None
        assert alignment[0].start_ms == 1234
        assert alignment[0].duration_ms == 1333

    def test_empty_tokens_list(self) -> None:
        """Empty tokens list returns None."""
        result = _make_result(tokens=[])
        alignment = _extract_alignment(result)
        assert alignment is None

    def test_returns_tuple(self) -> None:
        """Result is an immutable tuple."""
        result = _make_result(tokens=[_make_token("Hi", 0.0, 0.1)])
        alignment = _extract_alignment(result)
        assert isinstance(alignment, tuple)


# ===================================================================
# Proto conversion with alignment
# ===================================================================


class TestAudioChunkToProtoAlignment:
    def test_without_alignment(self) -> None:
        """Chunk without alignment has no alignment field set."""
        chunk = audio_chunk_to_proto(
            audio_data=b"\x00\x01",
            is_last=False,
            duration=0.5,
        )
        assert not chunk.alignment.items
        assert chunk.alignment.granularity == ""

    def test_with_alignment(self) -> None:
        """Chunk with alignment includes AlignmentItem protos."""
        alignment = (
            TTSAlignmentItem(text="Hello", start_ms=0, duration_ms=200),
            TTSAlignmentItem(text="world", start_ms=200, duration_ms=300),
        )
        chunk = audio_chunk_to_proto(
            audio_data=b"\x00\x01" * 50,
            is_last=False,
            duration=0.5,
            alignment=alignment,
            alignment_granularity="word",
        )
        assert len(chunk.alignment.items) == 2
        assert chunk.alignment.granularity == "word"
        assert chunk.alignment.items[0].text == "Hello"
        assert chunk.alignment.items[0].start_ms == 0
        assert chunk.alignment.items[0].duration_ms == 200
        assert chunk.alignment.items[1].text == "world"
        assert chunk.alignment.items[1].start_ms == 200
        assert chunk.alignment.items[1].duration_ms == 300

    def test_with_character_granularity(self) -> None:
        """Granularity is correctly set to 'character'."""
        alignment = (TTSAlignmentItem(text="H", start_ms=0, duration_ms=50),)
        chunk = audio_chunk_to_proto(
            audio_data=b"\x00\x01",
            is_last=False,
            duration=0.1,
            alignment=alignment,
            alignment_granularity="character",
        )
        assert chunk.alignment.granularity == "character"

    def test_alignment_with_codec(self) -> None:
        """Alignment works alongside codec field."""
        alignment = (TTSAlignmentItem(text="Hi", start_ms=0, duration_ms=100),)
        chunk = audio_chunk_to_proto(
            audio_data=b"\x00\x01",
            is_last=False,
            duration=0.1,
            codec="opus",
            alignment=alignment,
        )
        assert chunk.codec == "opus"
        assert len(chunk.alignment.items) == 1


class TestProtoRequestAlignmentFields:
    def test_defaults_when_not_set(self) -> None:
        """include_alignment defaults to False, granularity to 'word'."""
        request = SynthesizeRequest(
            request_id="req-1",
            text="Hello",
        )
        params = proto_request_to_synthesize_params(request)
        assert params.include_alignment is False
        assert params.alignment_granularity == "word"

    def test_extracts_alignment_fields(self) -> None:
        """Alignment fields are correctly extracted from proto."""
        request = SynthesizeRequest(
            request_id="req-2",
            text="Hello",
            include_alignment=True,
            alignment_granularity="character",
        )
        params = proto_request_to_synthesize_params(request)
        assert params.include_alignment is True
        assert params.alignment_granularity == "character"

    def test_empty_granularity_defaults_to_word(self) -> None:
        """Empty alignment_granularity defaults to 'word'."""
        request = SynthesizeRequest(
            request_id="req-3",
            text="Hello",
            include_alignment=True,
            alignment_granularity="",
        )
        params = proto_request_to_synthesize_params(request)
        assert params.alignment_granularity == "word"


# ===================================================================
# Proto serialization roundtrip with alignment
# ===================================================================


class TestAlignmentProtoRoundtrip:
    def test_alignment_item_roundtrip(self) -> None:
        """AlignmentItem survives proto serialization roundtrip."""
        original = AlignmentItem(text="Hello", start_ms=100, duration_ms=250)
        serialized = original.SerializeToString()
        restored = AlignmentItem()
        restored.ParseFromString(serialized)
        assert restored.text == "Hello"
        assert restored.start_ms == 100
        assert restored.duration_ms == 250

    def test_chunk_alignment_roundtrip(self) -> None:
        """ChunkAlignment with items survives roundtrip."""
        original = ChunkAlignment(
            items=[
                AlignmentItem(text="Hello", start_ms=0, duration_ms=200),
                AlignmentItem(text="world", start_ms=200, duration_ms=300),
            ],
            granularity="word",
        )
        serialized = original.SerializeToString()
        restored = ChunkAlignment()
        restored.ParseFromString(serialized)
        assert len(restored.items) == 2
        assert restored.granularity == "word"
        assert restored.items[0].text == "Hello"
        assert restored.items[1].text == "world"

    def test_synthesize_chunk_with_alignment_roundtrip(self) -> None:
        """SynthesizeChunk with alignment survives roundtrip."""
        original = SynthesizeChunk(
            audio_data=b"\x00\x01\x02",
            is_last=False,
            duration=0.5,
            alignment=ChunkAlignment(
                items=[
                    AlignmentItem(text="Hi", start_ms=0, duration_ms=100),
                ],
                granularity="word",
            ),
        )
        serialized = original.SerializeToString()
        restored = SynthesizeChunk()
        restored.ParseFromString(serialized)
        assert restored.audio_data == b"\x00\x01\x02"
        assert len(restored.alignment.items) == 1
        assert restored.alignment.items[0].text == "Hi"
        assert restored.alignment.granularity == "word"

    def test_synthesize_request_alignment_fields_roundtrip(self) -> None:
        """SynthesizeRequest alignment fields survive roundtrip."""
        original = SynthesizeRequest(
            request_id="req-1",
            text="Hello world",
            include_alignment=True,
            alignment_granularity="character",
        )
        serialized = original.SerializeToString()
        restored = SynthesizeRequest()
        restored.ParseFromString(serialized)
        assert restored.include_alignment is True
        assert restored.alignment_granularity == "character"

    def test_backward_compat_old_chunk_no_alignment(self) -> None:
        """Old-format chunk (no alignment) deserializes with empty alignment."""
        # Simulate an old worker that doesn't know about alignment
        old_chunk = SynthesizeChunk(
            audio_data=b"\x00\x01",
            is_last=False,
            duration=0.5,
        )
        serialized = old_chunk.SerializeToString()
        restored = SynthesizeChunk()
        restored.ParseFromString(serialized)
        # alignment is empty (default proto value)
        assert len(restored.alignment.items) == 0
        assert restored.alignment.granularity == ""


# ===================================================================
# Servicer alignment path
# ===================================================================


class TestServicerAlignmentPath:
    @pytest.fixture()
    def servicer(self) -> TTSWorkerServicer:
        return TTSWorkerServicer(
            backend=MockAlignmentBackend(),  # type: ignore[arg-type]
            model_name="kokoro-v1",
            engine="kokoro",
        )

    async def test_alignment_chunks_returned(self, servicer: TTSWorkerServicer) -> None:
        """Synthesize with include_alignment returns alignment data."""
        request = SynthesizeRequest(
            request_id="req-align-1",
            text="Hello world",
            voice="default",
            sample_rate=24000,
            speed=1.0,
            include_alignment=True,
            alignment_granularity="word",
        )
        ctx = _make_context()

        chunks: list[SynthesizeChunk] = []
        async for chunk in servicer.Synthesize(request, ctx):
            chunks.append(chunk)

        # 2 audio chunks + 1 final chunk
        assert len(chunks) == 3

        # First chunk should have alignment
        assert len(chunks[0].alignment.items) == 2
        assert chunks[0].alignment.items[0].text == "Hello"
        assert chunks[0].alignment.items[0].start_ms == 0
        assert chunks[0].alignment.items[0].duration_ms == 200
        assert chunks[0].alignment.items[1].text == "world"
        assert chunks[0].alignment.granularity == "word"

        # Second chunk has no alignment (was None in mock)
        assert len(chunks[1].alignment.items) == 0

        # Final chunk
        assert chunks[2].is_last is True

    async def test_no_alignment_when_not_requested(self, servicer: TTSWorkerServicer) -> None:
        """Synthesize without include_alignment uses standard path."""
        request = SynthesizeRequest(
            request_id="req-no-align",
            text="Hello world",
            voice="default",
            sample_rate=24000,
            speed=1.0,
            include_alignment=False,
        )
        ctx = _make_context()

        chunks: list[SynthesizeChunk] = []
        async for chunk in servicer.Synthesize(request, ctx):
            chunks.append(chunk)

        # All non-final chunks should have no alignment
        for ch in chunks[:-1]:
            assert len(ch.alignment.items) == 0

    async def test_alignment_with_cancellation(self, servicer: TTSWorkerServicer) -> None:
        """Alignment path respects context cancellation."""
        request = SynthesizeRequest(
            request_id="req-align-cancel",
            text="Hello world",
            voice="default",
            sample_rate=24000,
            speed=1.0,
            include_alignment=True,
        )
        ctx = _make_context()
        # Cancel immediately
        ctx.cancelled = MagicMock(return_value=True)

        chunks: list[SynthesizeChunk] = []
        async for chunk in servicer.Synthesize(request, ctx):
            chunks.append(chunk)

        assert len(chunks) == 0


# ===================================================================
# TTSBackend default synthesize_with_alignment
# ===================================================================


class TestTTSBackendDefaultAlignment:
    async def test_default_wraps_synthesize(self) -> None:
        """Default synthesize_with_alignment wraps synthesize() in TTSChunkResult."""
        from macaw.workers.tts.interface import TTSBackend

        class MinimalBackend(TTSBackend):
            async def load(self, model_path: str, config: dict[str, object]) -> None:
                pass

            async def capabilities(self) -> TTSEngineCapabilities:
                return TTSEngineCapabilities()

            async def synthesize(  # type: ignore[override, misc]
                self,
                text: str,
                voice: str = "default",
                *,
                sample_rate: int = 24000,
                speed: float = 1.0,
                options: dict[str, object] | None = None,
            ) -> AsyncIterator[bytes]:
                yield b"\x00\x01" * 50
                yield b"\x02\x03" * 50

            async def voices(self) -> list[VoiceInfo]:
                return []

            async def unload(self) -> None:
                pass

            async def health(self) -> dict[str, str]:
                return {"status": "ok"}

        backend = MinimalBackend()
        results: list[TTSChunkResult] = []
        async for chunk_result in backend.synthesize_with_alignment("Hello"):
            results.append(chunk_result)

        assert len(results) == 2
        assert results[0].audio == b"\x00\x01" * 50
        assert results[0].alignment is None
        assert results[0].alignment_granularity == "word"
        assert results[1].audio == b"\x02\x03" * 50
        assert results[1].alignment is None
