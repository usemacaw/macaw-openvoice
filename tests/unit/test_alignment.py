"""Tests for the forced alignment module (macaw.alignment).

Tests cover:
- Pure helper functions (text tokenization, path-to-items, word merging)
- CTCAligner class behavior (mocked torchaudio)
- Factory (create_aligner) with import guard
- Servicer forced alignment fallback path
"""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from macaw._types import TTSAlignmentItem, TTSChunkResult, TTSEngineCapabilities
from macaw.alignment.ctc_aligner import (
    _merge_chars_to_words,
    _path_to_char_items,
    _text_to_tokens,
)

# ─────────────────────────────────────────────────────────────────────
# _text_to_tokens
# ─────────────────────────────────────────────────────────────────────


# Simulated wav2vec2 label dictionary (subset).
_LABELS = ["-", "|", "E", "T", "A", "O", "N", "I", "H", "S", "R", "D", "L", "U"]
_LABEL_TO_INDEX = {c: i for i, c in enumerate(_LABELS)}


class TestTextToTokens:
    """Test _text_to_tokens conversion."""

    def test_simple_word(self) -> None:
        tokens = _text_to_tokens("HELLO", _LABEL_TO_INDEX)
        assert tokens == [
            _LABEL_TO_INDEX["H"],
            _LABEL_TO_INDEX["E"],
            _LABEL_TO_INDEX["L"],
            _LABEL_TO_INDEX["L"],
            _LABEL_TO_INDEX["O"],
        ]

    def test_two_words_with_separator(self) -> None:
        tokens = _text_to_tokens("HI THERE", _LABEL_TO_INDEX)
        assert _LABEL_TO_INDEX["|"] in tokens
        expected = [
            _LABEL_TO_INDEX["H"],
            _LABEL_TO_INDEX["I"],
            _LABEL_TO_INDEX["|"],
            _LABEL_TO_INDEX["T"],
            _LABEL_TO_INDEX["H"],
            _LABEL_TO_INDEX["E"],
            _LABEL_TO_INDEX["R"],
            _LABEL_TO_INDEX["E"],
        ]
        assert tokens == expected

    def test_lowercase_is_uppercased(self) -> None:
        tokens = _text_to_tokens("hello", _LABEL_TO_INDEX)
        assert tokens == _text_to_tokens("HELLO", _LABEL_TO_INDEX)

    def test_unknown_chars_are_skipped(self) -> None:
        tokens = _text_to_tokens("H3LL0!", _LABEL_TO_INDEX)
        # 3, 0, ! are not in the dictionary → skipped
        assert tokens == [
            _LABEL_TO_INDEX["H"],
            _LABEL_TO_INDEX["L"],
            _LABEL_TO_INDEX["L"],
        ]

    def test_empty_text_returns_empty(self) -> None:
        assert _text_to_tokens("", _LABEL_TO_INDEX) == []

    def test_only_unknown_chars(self) -> None:
        assert _text_to_tokens("123!@#", _LABEL_TO_INDEX) == []

    def test_consecutive_spaces_produce_single_separator(self) -> None:
        tokens = _text_to_tokens("HI   THERE", _LABEL_TO_INDEX)
        separator_count = tokens.count(_LABEL_TO_INDEX["|"])
        assert separator_count == 1

    def test_leading_trailing_spaces_stripped(self) -> None:
        tokens = _text_to_tokens(" HELLO ", _LABEL_TO_INDEX)
        assert tokens[0] != _LABEL_TO_INDEX["|"]
        assert tokens[-1] != _LABEL_TO_INDEX["|"]

    def test_no_separator_in_dict(self) -> None:
        """When | is not in the dictionary, spaces are ignored."""
        label_to_index = {"A": 0, "B": 1}
        tokens = _text_to_tokens("A B", label_to_index)
        assert tokens == [0, 1]


# ─────────────────────────────────────────────────────────────────────
# _path_to_char_items
# ─────────────────────────────────────────────────────────────────────


class TestPathToCharItems:
    """Test _path_to_char_items extraction from alignment path."""

    def test_simple_two_tokens(self) -> None:
        """Two tokens with blank separation."""
        path = [0, 1, 1, 0, 2, 2, 0]
        tokens = [0, 1]
        labels = ["A", "B", "C"]
        items = _path_to_char_items(path, tokens, labels, frame_ms=20.0)
        assert len(items) == 2
        assert items[0] == ("A", 20, 40)  # start=1*20, dur=2*20
        assert items[1] == ("B", 80, 40)  # start=4*20, dur=2*20

    def test_consecutive_different_tokens(self) -> None:
        """Tokens transition without blank between them."""
        path = [1, 1, 2, 2]
        tokens = [0, 1]
        labels = ["X", "Y", "Z"]
        items = _path_to_char_items(path, tokens, labels, frame_ms=10.0)
        assert len(items) == 2
        assert items[0] == ("X", 0, 20)
        assert items[1] == ("Y", 20, 20)

    def test_empty_path(self) -> None:
        items = _path_to_char_items([], [], ["A"], frame_ms=20.0)
        assert items == []

    def test_all_blanks(self) -> None:
        path = [0, 0, 0, 0]
        items = _path_to_char_items(path, [0], ["A"], frame_ms=20.0)
        assert items == []

    def test_min_duration_is_1ms(self) -> None:
        """Duration should never be 0, minimum is 1ms."""
        path = [1, 0]
        tokens = [0]
        labels = ["A", "B"]
        items = _path_to_char_items(path, tokens, labels, frame_ms=0.5)
        assert items[0][2] >= 1

    def test_last_token_finalized(self) -> None:
        """Last token in path (no trailing blank) is still emitted."""
        path = [0, 1, 1, 1]
        tokens = [0]
        labels = ["A", "B"]
        items = _path_to_char_items(path, tokens, labels, frame_ms=20.0)
        assert len(items) == 1
        assert items[0] == ("A", 20, 60)


# ─────────────────────────────────────────────────────────────────────
# _merge_chars_to_words
# ─────────────────────────────────────────────────────────────────────


class TestMergeCharsToWords:
    """Test _merge_chars_to_words grouping."""

    def test_single_word(self) -> None:
        chars = [("H", 0, 20), ("I", 20, 20)]
        words = _merge_chars_to_words(chars)
        assert len(words) == 1
        assert words[0] == ("HI", 0, 40)

    def test_two_words(self) -> None:
        chars = [
            ("H", 0, 20),
            ("I", 20, 20),
            ("|", 40, 10),
            ("T", 50, 20),
            ("O", 70, 20),
        ]
        words = _merge_chars_to_words(chars)
        assert len(words) == 2
        assert words[0] == ("HI", 0, 40)
        assert words[1] == ("TO", 50, 40)

    def test_empty_input(self) -> None:
        assert _merge_chars_to_words([]) == []

    def test_only_separator(self) -> None:
        words = _merge_chars_to_words([("|", 100, 10)])
        assert words == []

    def test_three_words(self) -> None:
        chars = [
            ("A", 0, 10),
            ("|", 10, 5),
            ("B", 15, 10),
            ("|", 25, 5),
            ("C", 30, 10),
        ]
        words = _merge_chars_to_words(chars)
        assert len(words) == 3
        assert words[0] == ("A", 0, 10)
        assert words[1] == ("B", 15, 10)
        assert words[2] == ("C", 30, 10)


# ─────────────────────────────────────────────────────────────────────
# CTCAligner
# ─────────────────────────────────────────────────────────────────────


class TestCTCAligner:
    """Test CTCAligner with mocked torchaudio."""

    async def test_empty_audio_returns_empty(self) -> None:
        from macaw.alignment.ctc_aligner import CTCAligner

        aligner = CTCAligner()
        result = await aligner.align(audio=b"", text="hello", sample_rate=24000)
        assert result == ()

    async def test_empty_text_returns_empty(self) -> None:
        from macaw.alignment.ctc_aligner import CTCAligner

        aligner = CTCAligner()
        result = await aligner.align(audio=b"\x00\x00", text="   ", sample_rate=24000)
        assert result == ()

    async def test_align_calls_forced_align(self) -> None:
        """Verify the full pipeline is invoked with correct structure."""
        import numpy as np
        import torch

        from macaw.alignment.ctc_aligner import CTCAligner

        aligner = CTCAligner()

        sr = 16000
        samples = np.zeros(int(sr * 1.0), dtype=np.int16)
        audio_bytes = samples.tobytes()

        # Set up model mock (already "loaded")
        mock_model = MagicMock()
        n_frames = 50
        n_labels = 6
        emissions = torch.randn(1, n_frames, n_labels)
        mock_model.return_value = (emissions, None)

        aligner._model = mock_model
        aligner._labels = ["-", "|", "H", "E", "L", "O"]
        aligner._label_to_index = {c: i for i, c in enumerate(aligner._labels)}

        # Build mock forced_align result
        path = torch.zeros(1, n_frames, dtype=torch.int32)
        path[0, 5:10] = 1  # H (token index 0 in targets → label[tokens[0]])
        path[0, 15:20] = 2  # E
        path[0, 25:30] = 3  # L
        path[0, 35:40] = 3  # L
        path[0, 42:47] = 4  # O
        scores = torch.zeros(1, n_frames)

        with (
            patch("torchaudio.functional.forced_align", return_value=(path, scores)),
            patch("torchaudio.functional.resample", side_effect=lambda wf, *_a: wf),
        ):
            result = await aligner.align(
                audio=audio_bytes,
                text="HELLO",
                sample_rate=sr,
                granularity="word",
            )

        assert len(result) >= 1
        for item in result:
            assert isinstance(item, TTSAlignmentItem)
            assert item.start_ms >= 0
            assert item.duration_ms >= 1

    async def test_align_character_granularity(self) -> None:
        """Test character-level alignment output."""
        import numpy as np
        import torch

        from macaw.alignment.ctc_aligner import CTCAligner

        aligner = CTCAligner()

        sr = 16000
        samples = np.zeros(sr, dtype=np.int16)
        audio_bytes = samples.tobytes()

        mock_model = MagicMock()
        n_frames = 50
        emissions = torch.randn(1, n_frames, 10)
        mock_model.return_value = (emissions, None)

        aligner._model = mock_model
        aligner._labels = ["-", "|", "H", "I"]
        aligner._label_to_index = {c: i for i, c in enumerate(aligner._labels)}

        path = torch.zeros(1, n_frames, dtype=torch.int32)
        path[0, 5:15] = 1  # H
        path[0, 20:30] = 2  # I
        scores = torch.zeros(1, n_frames)

        with (
            patch("torchaudio.functional.forced_align", return_value=(path, scores)),
            patch("torchaudio.functional.resample", side_effect=lambda wf, *_a: wf),
        ):
            result = await aligner.align(
                audio=audio_bytes,
                text="HI",
                sample_rate=sr,
                granularity="character",
            )

        assert len(result) == 2
        assert result[0].text == "H"
        assert result[1].text == "I"

    async def test_align_handles_forced_align_error(self) -> None:
        """If forced_align throws, return empty tuple."""
        import numpy as np
        import torch

        from macaw.alignment.ctc_aligner import CTCAligner

        aligner = CTCAligner()

        sr = 16000
        audio_bytes = np.zeros(sr, dtype=np.int16).tobytes()

        mock_model = MagicMock()
        emissions = torch.randn(1, 50, 10)
        mock_model.return_value = (emissions, None)

        aligner._model = mock_model
        aligner._labels = ["-", "|", "H", "E", "L", "O"]
        aligner._label_to_index = {c: i for i, c in enumerate(aligner._labels)}

        with (
            patch(
                "torchaudio.functional.forced_align",
                side_effect=RuntimeError("CTC error"),
            ),
            patch("torchaudio.functional.resample", side_effect=lambda wf, *_a: wf),
        ):
            result = await aligner.align(audio=audio_bytes, text="HELLO", sample_rate=sr)

        assert result == ()


# ─────────────────────────────────────────────────────────────────────
# create_aligner factory
# ─────────────────────────────────────────────────────────────────────


class TestCreateAligner:
    """Test create_aligner factory with import guard."""

    def test_returns_ctc_aligner_when_torchaudio_available(self) -> None:
        """When torchaudio is available with forced_align, returns CTCAligner."""
        from macaw.alignment import create_aligner
        from macaw.alignment.ctc_aligner import CTCAligner

        aligner = create_aligner()
        assert isinstance(aligner, CTCAligner)

    def test_returns_none_when_torchaudio_missing(self) -> None:
        """When torchaudio is not installed, returns None."""
        with patch.dict("sys.modules", {"torchaudio": None}):
            import importlib

            from macaw import alignment

            importlib.reload(alignment)
            result = alignment.create_aligner()
            assert result is None

            importlib.reload(alignment)

    def test_returns_none_when_forced_align_missing(self) -> None:
        """When torchaudio is too old (no forced_align), returns None."""
        mock_ta = MagicMock()
        mock_ta.__version__ = "2.0.0"
        del mock_ta.functional.forced_align

        with patch.dict("sys.modules", {"torchaudio": mock_ta}):
            import importlib

            from macaw import alignment

            importlib.reload(alignment)
            result = alignment.create_aligner()
            assert result is None

            importlib.reload(alignment)


# ─────────────────────────────────────────────────────────────────────
# Servicer forced alignment fallback
# ─────────────────────────────────────────────────────────────────────


def _make_grpc_context() -> MagicMock:
    """Build a mock gRPC servicer context.

    ``cancelled()`` is synchronous in gRPC so we use a plain MagicMock.
    ``abort()`` is async so we set it to an AsyncMock.
    """
    ctx = MagicMock()
    ctx.cancelled.return_value = False
    ctx.abort = AsyncMock()
    return ctx


def _make_synth_request(
    *,
    text: str = "hello",
    include_alignment: bool = True,
    language: str = "",
) -> MagicMock:
    """Build a mock SynthesizeRequest proto."""
    request = MagicMock()
    request.request_id = "test-req"
    request.codec = ""
    request.text = text
    request.voice = "default"
    request.sample_rate = 24000
    request.speed = 1.0
    request.include_alignment = include_alignment
    request.alignment_granularity = "word"
    request.language = language
    request.ref_audio = b""
    request.ref_text = ""
    request.instruction = ""
    request.seed = 0
    request.text_normalization = ""
    request.temperature = 0.0
    request.top_k = 0
    request.top_p = 0.0
    return request


class TestServicerForcedAlignmentFallback:
    """Test the servicer dispatches to forced alignment when engine lacks support."""

    def _make_servicer(
        self,
        *,
        supports_alignment: bool = False,
    ) -> tuple[object, AsyncMock]:
        """Create a TTSWorkerServicer with a mocked backend."""
        from macaw.workers.tts.servicer import TTSWorkerServicer

        backend = AsyncMock()
        backend.capabilities.return_value = TTSEngineCapabilities(
            supports_alignment=supports_alignment,
        )

        servicer = TTSWorkerServicer(
            backend=backend,
            model_name="test-model",
            engine="test-engine",
        )
        return servicer, backend

    async def test_native_alignment_used_when_supported(self) -> None:
        """When engine supports alignment, use synthesize_with_alignment."""
        servicer, backend = self._make_servicer(supports_alignment=True)

        chunk = TTSChunkResult(
            audio=b"\x00" * 100,
            alignment=(TTSAlignmentItem(text="hello", start_ms=0, duration_ms=500),),
        )

        async def mock_align_gen(*_a: object, **_kw: object):  # type: ignore[no-untyped-def]
            yield chunk

        backend.synthesize_with_alignment = mock_align_gen

        request = _make_synth_request()
        context = _make_grpc_context()

        chunks = []
        async for chunk_proto in servicer.Synthesize(request, context):
            chunks.append(chunk_proto)

        # Data chunk + final chunk
        assert len(chunks) >= 2
        # synthesize was NOT called (native alignment path used)
        backend.synthesize.assert_not_called()

    async def test_forced_alignment_fallback_when_not_supported(self) -> None:
        """When engine lacks alignment, accumulate audio and run forced aligner."""
        servicer, backend = self._make_servicer(supports_alignment=False)

        async def mock_synth(*_a: object, **_kw: object):  # type: ignore[no-untyped-def]
            yield b"\x00" * 200
            yield b"\x00" * 200

        backend.synthesize = mock_synth

        mock_aligner = AsyncMock()
        mock_aligner.align.return_value = (
            TTSAlignmentItem(text="HELLO", start_ms=0, duration_ms=500),
        )
        servicer._aligner = mock_aligner
        servicer._aligner_checked = True

        request = _make_synth_request()
        context = _make_grpc_context()

        chunks = []
        async for chunk_proto in servicer.Synthesize(request, context):
            chunks.append(chunk_proto)

        mock_aligner.align.assert_called_once()
        call_kwargs = mock_aligner.align.call_args.kwargs
        assert len(call_kwargs["audio"]) == 400

    async def test_no_aligner_yields_chunks_without_alignment(self) -> None:
        """When no aligner available, still yield audio without alignment."""
        servicer, backend = self._make_servicer(supports_alignment=False)

        async def mock_synth(*_a: object, **_kw: object):  # type: ignore[no-untyped-def]
            yield b"\x00" * 100

        backend.synthesize = mock_synth

        servicer._aligner = None
        servicer._aligner_checked = True

        request = _make_synth_request()
        context = _make_grpc_context()

        chunks = []
        async for chunk_proto in servicer.Synthesize(request, context):
            chunks.append(chunk_proto)

        # Audio chunk + final chunk
        assert len(chunks) >= 2

    async def test_forced_alignment_error_is_graceful(self) -> None:
        """If forced alignment fails, yield audio without alignment."""
        servicer, backend = self._make_servicer(supports_alignment=False)

        async def mock_synth(*_a: object, **_kw: object):  # type: ignore[no-untyped-def]
            yield b"\x00" * 100

        backend.synthesize = mock_synth

        mock_aligner = AsyncMock()
        mock_aligner.align.side_effect = RuntimeError("alignment failed")
        servicer._aligner = mock_aligner
        servicer._aligner_checked = True

        request = _make_synth_request()
        context = _make_grpc_context()

        chunks = []
        async for chunk_proto in servicer.Synthesize(request, context):
            chunks.append(chunk_proto)

        # Should not raise, should still yield audio
        assert len(chunks) >= 2

    async def test_language_extracted_from_options(self) -> None:
        """Language for forced alignment is extracted from request options."""
        servicer, backend = self._make_servicer(supports_alignment=False)

        async def mock_synth(*_a: object, **_kw: object):  # type: ignore[no-untyped-def]
            yield b"\x00" * 100

        backend.synthesize = mock_synth

        mock_aligner = AsyncMock()
        mock_aligner.align.return_value = ()
        servicer._aligner = mock_aligner
        servicer._aligner_checked = True

        request = _make_synth_request(text="bonjour", language="fr")
        context = _make_grpc_context()

        chunks = []
        async for chunk_proto in servicer.Synthesize(request, context):
            chunks.append(chunk_proto)

        mock_aligner.align.assert_called_once()
        call_kwargs = mock_aligner.align.call_args.kwargs
        assert call_kwargs["language"] == "fr"

    async def test_standard_path_when_no_alignment_requested(self) -> None:
        """When include_alignment=False, use standard synthesis path."""
        servicer, backend = self._make_servicer(supports_alignment=False)

        async def mock_synth(*_a: object, **_kw: object):  # type: ignore[no-untyped-def]
            yield b"\x00" * 100

        backend.synthesize = mock_synth

        request = _make_synth_request(include_alignment=False)
        context = _make_grpc_context()

        chunks = []
        async for chunk_proto in servicer.Synthesize(request, context):
            chunks.append(chunk_proto)

        # capabilities() should NOT have been called (no alignment requested)
        backend.capabilities.assert_not_called()
        assert len(chunks) >= 2


# ─────────────────────────────────────────────────────────────────────
# Aligner ABC
# ─────────────────────────────────────────────────────────────────────


class TestAlignerInterface:
    """Test Aligner ABC contract."""

    def test_cannot_instantiate_abc(self) -> None:
        from macaw.alignment.interface import Aligner

        with pytest.raises(TypeError, match="abstract"):
            Aligner()  # type: ignore[abstract]

    def test_concrete_subclass_works(self) -> None:
        from macaw.alignment.interface import Aligner

        class DummyAligner(Aligner):
            async def align(
                self,
                audio: bytes,
                text: str,
                sample_rate: int,
                language: str = "en",
                granularity: str = "word",
            ) -> tuple[TTSAlignmentItem, ...]:
                return ()

        aligner = DummyAligner()
        assert aligner is not None
