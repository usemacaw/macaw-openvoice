"""Tests for character-level and normalized alignment (Sprint 6).

Tests cover:
- Character-level alignment via _extract_alignment(granularity="character")
- _distribute_timing_to_chars helper
- Normalized alignment via _extract_normalized_alignment (phoneme-based)
- Proto normalized_alignment field roundtrip
- Converters: audio_chunk_to_proto with normalized_alignment
- TTSChunkResult with normalized_alignment field
- Kokoro capabilities: supports_character_alignment=True
- REST NDJSON response with normalized_alignment
- WS TTSAlignmentEvent with normalized_items
- Servicer native path forwards normalized_alignment
"""

from __future__ import annotations

import json
from types import SimpleNamespace
from typing import TYPE_CHECKING, Any
from unittest.mock import AsyncMock, MagicMock

from macaw._types import TTSAlignmentItem, TTSChunkResult, TTSEngineCapabilities
from macaw.proto.tts_worker_pb2 import (
    AlignmentItem,
    ChunkAlignment,
    SynthesizeChunk,
)
from macaw.server.models.alignment import (
    AlignmentItemResponse,
    AudioChunkWithAlignment,
    ChunkAlignmentResponse,
)
from macaw.server.models.events import (
    TTSAlignmentEvent,
    TTSAlignmentItemEvent,
)
from macaw.workers.tts.converters import audio_chunk_to_proto
from macaw.workers.tts.kokoro import (
    _distribute_timing_to_chars,
    _extract_alignment,
    _extract_normalized_alignment,
)

if TYPE_CHECKING:
    from collections.abc import AsyncIterator

    from macaw._types import VoiceInfo


# ===================================================================
# Helpers
# ===================================================================


def _make_token(
    text: str,
    start_ts: float | None,
    end_ts: float | None,
    phonemes: str | None = None,
) -> SimpleNamespace:
    """Create a mock MToken-like object with timing and phoneme data."""
    return SimpleNamespace(text=text, start_ts=start_ts, end_ts=end_ts, phonemes=phonemes)


def _make_result(tokens: list[SimpleNamespace] | None = None) -> SimpleNamespace:
    """Create a mock KPipeline.Result-like object."""
    return SimpleNamespace(tokens=tokens)


# ===================================================================
# _distribute_timing_to_chars
# ===================================================================


class TestDistributeTimingToChars:
    def test_single_char(self) -> None:
        """Single character gets the full duration."""
        items = _distribute_timing_to_chars("A", 100, 200)
        assert len(items) == 1
        assert items[0] == TTSAlignmentItem(text="A", start_ms=100, duration_ms=200)

    def test_uniform_distribution(self) -> None:
        """Duration is evenly distributed across characters."""
        items = _distribute_timing_to_chars("Hi", 0, 100)
        assert len(items) == 2
        assert items[0] == TTSAlignmentItem(text="H", start_ms=0, duration_ms=50)
        assert items[1] == TTSAlignmentItem(text="i", start_ms=50, duration_ms=50)

    def test_remainder_on_last_char(self) -> None:
        """Rounding remainder is added to the last character."""
        items = _distribute_timing_to_chars("ABC", 0, 100)
        assert len(items) == 3
        # 100 // 3 = 33, remainder = 100 - 33*3 = 1
        assert items[0].duration_ms == 33
        assert items[1].duration_ms == 33
        assert items[2].duration_ms == 34  # 33 + 1 remainder

    def test_offsets_are_sequential(self) -> None:
        """Each character starts where the previous ends (uniform spacing)."""
        items = _distribute_timing_to_chars("HELLO", 10, 500)
        for i in range(1, len(items)):
            assert items[i].start_ms == items[i - 1].start_ms + (500 // 5)

    def test_minimum_duration_one_ms(self) -> None:
        """Characters get at least 1ms duration even with very small total."""
        items = _distribute_timing_to_chars("AB", 0, 1)
        assert items[0].duration_ms >= 1
        assert items[1].duration_ms >= 1

    def test_five_chars_exact_division(self) -> None:
        """When duration divides evenly, no remainder."""
        items = _distribute_timing_to_chars("Hello", 0, 500)
        assert all(item.duration_ms == 100 for item in items)


# ===================================================================
# _extract_alignment with granularity="character"
# ===================================================================


class TestExtractAlignmentCharacter:
    def test_character_granularity(self) -> None:
        """Character granularity distributes word timing to characters."""
        result = _make_result(
            tokens=[
                _make_token("Hi", 0.0, 0.1),
            ]
        )
        alignment = _extract_alignment(result, granularity="character")
        assert alignment is not None
        assert len(alignment) == 2
        assert alignment[0].text == "H"
        assert alignment[1].text == "i"

    def test_word_granularity_unchanged(self) -> None:
        """Word granularity still returns word-level items."""
        result = _make_result(
            tokens=[
                _make_token("Hello", 0.0, 0.5),
            ]
        )
        alignment = _extract_alignment(result, granularity="word")
        assert alignment is not None
        assert len(alignment) == 1
        assert alignment[0].text == "Hello"

    def test_multi_word_character_granularity(self) -> None:
        """Multiple words each get character-level distribution."""
        result = _make_result(
            tokens=[
                _make_token("Hi", 0.0, 0.1),
                _make_token("world", 0.1, 0.6),
            ]
        )
        alignment = _extract_alignment(result, granularity="character")
        assert alignment is not None
        # "Hi" = 2 chars + "world" = 5 chars = 7 total
        assert len(alignment) == 7
        assert alignment[0].text == "H"
        assert alignment[1].text == "i"
        assert alignment[2].text == "w"

    def test_character_timing_sums_to_word_duration(self) -> None:
        """Sum of character durations equals the original word duration."""
        result = _make_result(
            tokens=[
                _make_token("Hello", 0.0, 0.5),
            ]
        )
        alignment = _extract_alignment(result, granularity="character")
        assert alignment is not None
        total_dur = sum(item.duration_ms for item in alignment)
        assert total_dur == 500

    def test_skips_tokens_without_timing_in_character_mode(self) -> None:
        """Tokens without timing are still skipped in character mode."""
        result = _make_result(
            tokens=[
                _make_token("Hi", 0.0, 0.1),
                _make_token("world", None, None),
            ]
        )
        alignment = _extract_alignment(result, granularity="character")
        assert alignment is not None
        assert len(alignment) == 2  # only "Hi" chars


# ===================================================================
# _extract_normalized_alignment
# ===================================================================


class TestExtractNormalizedAlignment:
    def test_extracts_phoneme_text(self) -> None:
        """Extracts alignment using phonemes instead of graphemes."""
        result = _make_result(
            tokens=[
                _make_token("Hello", 0.0, 0.5, phonemes="hEHloU"),
            ]
        )
        norm = _extract_normalized_alignment(result)
        assert norm is not None
        assert len(norm) == 1
        assert norm[0].text == "hEHloU"
        assert norm[0].start_ms == 0
        assert norm[0].duration_ms == 500

    def test_returns_none_when_no_tokens(self) -> None:
        result = SimpleNamespace()
        assert _extract_normalized_alignment(result) is None

    def test_returns_none_when_no_phonemes(self) -> None:
        """Tokens without phonemes are skipped."""
        result = _make_result(
            tokens=[
                _make_token("Hello", 0.0, 0.5, phonemes=None),
            ]
        )
        assert _extract_normalized_alignment(result) is None

    def test_returns_none_when_empty_phonemes(self) -> None:
        """Tokens with empty phoneme string are skipped."""
        result = _make_result(
            tokens=[
                _make_token("Hello", 0.0, 0.5, phonemes=""),
            ]
        )
        assert _extract_normalized_alignment(result) is None

    def test_character_granularity_distributes_phoneme_chars(self) -> None:
        """Character granularity distributes across phoneme characters."""
        result = _make_result(
            tokens=[
                _make_token("Hi", 0.0, 0.1, phonemes="haI"),
            ]
        )
        norm = _extract_normalized_alignment(result, granularity="character")
        assert norm is not None
        assert len(norm) == 3  # h, a, I
        assert norm[0].text == "h"
        assert norm[1].text == "a"
        assert norm[2].text == "I"

    def test_word_granularity_returns_phoneme_words(self) -> None:
        """Word granularity returns whole phoneme strings."""
        result = _make_result(
            tokens=[
                _make_token("Hello", 0.0, 0.3, phonemes="hEHloU"),
                _make_token("world", 0.3, 0.6, phonemes="wURld"),
            ]
        )
        norm = _extract_normalized_alignment(result, granularity="word")
        assert norm is not None
        assert len(norm) == 2
        assert norm[0].text == "hEHloU"
        assert norm[1].text == "wURld"

    def test_skips_tokens_without_timing(self) -> None:
        """Tokens without timing are skipped even with phonemes."""
        result = _make_result(
            tokens=[
                _make_token("Hello", 0.0, 0.3, phonemes="hEHloU"),
                _make_token("world", None, None, phonemes="wURld"),
            ]
        )
        norm = _extract_normalized_alignment(result)
        assert norm is not None
        assert len(norm) == 1

    def test_mixed_with_and_without_phonemes(self) -> None:
        """Only tokens with phonemes appear in normalized alignment."""
        result = _make_result(
            tokens=[
                _make_token("Hello", 0.0, 0.3, phonemes="hEHloU"),
                _make_token(",", 0.3, 0.35, phonemes=None),
                _make_token("world", 0.35, 0.6, phonemes="wURld"),
            ]
        )
        norm = _extract_normalized_alignment(result)
        assert norm is not None
        assert len(norm) == 2
        assert norm[0].text == "hEHloU"
        assert norm[1].text == "wURld"


# ===================================================================
# Proto normalized_alignment field
# ===================================================================


class TestNormalizedAlignmentProto:
    def test_chunk_with_normalized_alignment(self) -> None:
        """SynthesizeChunk carries normalized_alignment through proto."""
        norm = ChunkAlignment(
            items=[
                AlignmentItem(text="hEHloU", start_ms=0, duration_ms=300),
            ],
            granularity="word",
        )
        chunk = SynthesizeChunk(
            audio_data=b"\x00",
            normalized_alignment=norm,
        )
        assert len(chunk.normalized_alignment.items) == 1
        assert chunk.normalized_alignment.items[0].text == "hEHloU"

    def test_roundtrip(self) -> None:
        """Normalized alignment survives proto serialization roundtrip."""
        original = SynthesizeChunk(
            audio_data=b"\x00\x01",
            normalized_alignment=ChunkAlignment(
                items=[AlignmentItem(text="haI", start_ms=0, duration_ms=100)],
                granularity="character",
            ),
        )
        serialized = original.SerializeToString()
        restored = SynthesizeChunk()
        restored.ParseFromString(serialized)
        assert len(restored.normalized_alignment.items) == 1
        assert restored.normalized_alignment.items[0].text == "haI"
        assert restored.normalized_alignment.granularity == "character"

    def test_empty_by_default(self) -> None:
        """Chunk without normalized_alignment has empty items list."""
        chunk = SynthesizeChunk(audio_data=b"\x00")
        assert not chunk.normalized_alignment.items


# ===================================================================
# Converter: audio_chunk_to_proto with normalized_alignment
# ===================================================================


class TestConverterNormalizedAlignment:
    def test_with_normalized_alignment(self) -> None:
        """Converter serializes normalized_alignment to proto."""
        alignment = (TTSAlignmentItem(text="Hello", start_ms=0, duration_ms=200),)
        norm = (TTSAlignmentItem(text="hEHloU", start_ms=0, duration_ms=200),)
        chunk = audio_chunk_to_proto(
            audio_data=b"\x00\x01",
            is_last=False,
            duration=0.2,
            alignment=alignment,
            normalized_alignment=norm,
        )
        assert len(chunk.alignment.items) == 1
        assert len(chunk.normalized_alignment.items) == 1
        assert chunk.normalized_alignment.items[0].text == "hEHloU"

    def test_without_normalized_alignment(self) -> None:
        """Converter works without normalized_alignment (backward compat)."""
        alignment = (TTSAlignmentItem(text="Hello", start_ms=0, duration_ms=200),)
        chunk = audio_chunk_to_proto(
            audio_data=b"\x00\x01",
            is_last=False,
            duration=0.2,
            alignment=alignment,
        )
        assert len(chunk.alignment.items) == 1
        assert not chunk.normalized_alignment.items

    def test_normalized_without_regular(self) -> None:
        """Normalized alignment can be set without regular alignment."""
        norm = (TTSAlignmentItem(text="haI", start_ms=0, duration_ms=100),)
        chunk = audio_chunk_to_proto(
            audio_data=b"\x00\x01",
            is_last=False,
            duration=0.1,
            normalized_alignment=norm,
        )
        assert not chunk.alignment.items
        assert len(chunk.normalized_alignment.items) == 1


# ===================================================================
# TTSChunkResult with normalized_alignment
# ===================================================================


class TestTTSChunkResultNormalized:
    def test_default_is_none(self) -> None:
        result = TTSChunkResult(audio=b"\x00")
        assert result.normalized_alignment is None

    def test_with_normalized_alignment(self) -> None:
        norm = (TTSAlignmentItem(text="haI", start_ms=0, duration_ms=100),)
        result = TTSChunkResult(audio=b"\x00", normalized_alignment=norm)
        assert result.normalized_alignment is not None
        assert len(result.normalized_alignment) == 1

    def test_both_alignments(self) -> None:
        align = (TTSAlignmentItem(text="Hi", start_ms=0, duration_ms=100),)
        norm = (TTSAlignmentItem(text="haI", start_ms=0, duration_ms=100),)
        result = TTSChunkResult(audio=b"\x00", alignment=align, normalized_alignment=norm)
        assert result.alignment is not None
        assert result.normalized_alignment is not None


# ===================================================================
# REST model: AudioChunkWithAlignment.normalized_alignment
# ===================================================================


class TestRestNormalizedAlignment:
    def test_model_includes_normalized_alignment(self) -> None:
        """AudioChunkWithAlignment can carry normalized_alignment."""
        chunk = AudioChunkWithAlignment(
            audio="AAAA",
            alignment=ChunkAlignmentResponse(
                items=[AlignmentItemResponse(text="Hi", start_ms=0, duration_ms=100)],
            ),
            normalized_alignment=ChunkAlignmentResponse(
                items=[AlignmentItemResponse(text="haI", start_ms=0, duration_ms=100)],
            ),
        )
        data = chunk.model_dump(exclude_none=True)
        assert "normalized_alignment" in data
        assert len(data["normalized_alignment"]["items"]) == 1

    def test_model_without_normalized_alignment(self) -> None:
        """normalized_alignment is excluded from JSON when None."""
        chunk = AudioChunkWithAlignment(audio="AAAA")
        data = chunk.model_dump(exclude_none=True)
        assert "normalized_alignment" not in data

    def test_json_serialization(self) -> None:
        """JSON output has normalized_alignment when present."""
        chunk = AudioChunkWithAlignment(
            audio="AA==",
            normalized_alignment=ChunkAlignmentResponse(
                items=[AlignmentItemResponse(text="haI", start_ms=0, duration_ms=100)],
            ),
        )
        json_str = chunk.model_dump_json(exclude_none=True)
        parsed = json.loads(json_str)
        assert "normalized_alignment" in parsed
        assert parsed["normalized_alignment"]["items"][0]["text"] == "haI"


# ===================================================================
# WS event: TTSAlignmentEvent.normalized_items
# ===================================================================


class TestWsNormalizedAlignment:
    def test_event_with_normalized_items(self) -> None:
        """TTSAlignmentEvent can carry normalized_items."""
        event = TTSAlignmentEvent(
            request_id="req-1",
            items=[TTSAlignmentItemEvent(text="Hi", start_ms=0, duration_ms=100)],
            normalized_items=[TTSAlignmentItemEvent(text="haI", start_ms=0, duration_ms=100)],
        )
        assert event.normalized_items is not None
        assert len(event.normalized_items) == 1
        assert event.normalized_items[0].text == "haI"

    def test_event_without_normalized_items(self) -> None:
        """normalized_items defaults to None."""
        event = TTSAlignmentEvent(
            request_id="req-1",
            items=[TTSAlignmentItemEvent(text="Hi", start_ms=0, duration_ms=100)],
        )
        assert event.normalized_items is None

    def test_json_excludes_none_normalized(self) -> None:
        """JSON output excludes normalized_items when None."""
        event = TTSAlignmentEvent(
            request_id="req-1",
            items=[TTSAlignmentItemEvent(text="Hi", start_ms=0, duration_ms=100)],
        )
        data = event.model_dump(exclude_none=True)
        assert "normalized_items" not in data

    def test_json_includes_normalized_when_present(self) -> None:
        """JSON output includes normalized_items when set."""
        event = TTSAlignmentEvent(
            request_id="req-1",
            items=[TTSAlignmentItemEvent(text="Hi", start_ms=0, duration_ms=100)],
            normalized_items=[TTSAlignmentItemEvent(text="haI", start_ms=0, duration_ms=100)],
        )
        data = event.model_dump(exclude_none=True)
        assert "normalized_items" in data


# ===================================================================
# Servicer: native path forwards normalized_alignment
# ===================================================================


class MockCharAlignmentBackend:
    """Backend that returns both alignment and normalized_alignment."""

    async def capabilities(self) -> TTSEngineCapabilities:
        return TTSEngineCapabilities(
            supports_alignment=True,
            supports_character_alignment=True,
        )

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
        yield b"\x00\x01" * 100

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
        yield TTSChunkResult(
            audio=b"\x00\x01" * 100,
            alignment=(TTSAlignmentItem(text="Hello", start_ms=0, duration_ms=200),),
            normalized_alignment=(TTSAlignmentItem(text="hEHloU", start_ms=0, duration_ms=200),),
            alignment_granularity=alignment_granularity,
        )

    async def voices(self) -> list[VoiceInfo]:
        return []

    async def unload(self) -> None:
        pass

    async def health(self) -> dict[str, str]:
        return {"status": "ok"}


class TestServicerNormalizedAlignment:
    async def test_native_path_forwards_normalized_alignment(self) -> None:
        """Servicer native alignment path includes normalized_alignment in proto."""
        from macaw.workers.tts.servicer import TTSWorkerServicer

        backend = MockCharAlignmentBackend()
        servicer = TTSWorkerServicer(backend=backend, model_name="test", engine="test")  # type: ignore[arg-type]

        from macaw.proto.tts_worker_pb2 import SynthesizeRequest

        request = SynthesizeRequest(
            request_id="req-test",
            text="Hello",
            include_alignment=True,
            alignment_granularity="word",
        )
        ctx = MagicMock()
        ctx.cancelled = MagicMock(return_value=False)
        ctx.abort = AsyncMock()

        chunks = []
        async for chunk in servicer.Synthesize(request, ctx):
            chunks.append(chunk)

        # At least one non-final chunk with audio
        audio_chunks = [c for c in chunks if c.audio_data and not c.is_last]
        assert len(audio_chunks) >= 1

        first = audio_chunks[0]
        assert len(first.alignment.items) == 1
        assert first.alignment.items[0].text == "Hello"
        assert len(first.normalized_alignment.items) == 1
        assert first.normalized_alignment.items[0].text == "hEHloU"

    async def test_forced_alignment_has_no_normalized(self) -> None:
        """Forced alignment fallback does not produce normalized_alignment."""
        from macaw.workers.tts.servicer import TTSWorkerServicer

        # Backend without native alignment support
        backend = MagicMock()
        caps = TTSEngineCapabilities(supports_alignment=False)
        backend.capabilities = AsyncMock(return_value=caps)

        async def fake_synth(**kwargs: Any) -> AsyncIterator[bytes]:
            yield b"\x00\x01" * 100

        backend.synthesize = fake_synth

        servicer = TTSWorkerServicer(backend=backend, model_name="test", engine="test")

        from macaw.proto.tts_worker_pb2 import SynthesizeRequest

        request = SynthesizeRequest(
            request_id="req-test",
            text="Hello",
            include_alignment=True,
        )
        ctx = MagicMock()
        ctx.cancelled = MagicMock(return_value=False)
        ctx.abort = AsyncMock()

        chunks = []
        async for chunk in servicer.Synthesize(request, ctx):
            chunks.append(chunk)

        # Forced path: no normalized_alignment
        for c in chunks:
            assert not c.normalized_alignment.items
