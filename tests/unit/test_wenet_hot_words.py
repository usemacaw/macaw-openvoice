"""Testes de hot words para WeNet e roteamento baseado em capabilities."""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock

from macaw.session.streaming import StreamingSession
from macaw.workers.stt.wenet import WeNetBackend


class TestWeNetHotWordsCapabilities:
    """WeNetBackend reporta supports_hot_words=True."""

    async def test_capabilities_supports_hot_words(self) -> None:
        backend = WeNetBackend()
        caps = await backend.capabilities()
        assert caps.supports_hot_words is True

    async def test_capabilities_no_initial_prompt(self) -> None:
        backend = WeNetBackend()
        caps = await backend.capabilities()
        assert caps.supports_initial_prompt is False


class TestWeNetHotWordsBatch:
    """Hot words em transcribe_file passados para a engine."""

    async def test_hot_words_passed_to_transcribe(self) -> None:
        backend = WeNetBackend()
        mock_model = MagicMock()
        mock_model.transcribe.return_value = {"text": "transferencia PIX"}
        backend._model = mock_model  # type: ignore[assignment]

        audio = b"\x00\x01" * 16000  # 1s
        result = await backend.transcribe_file(audio, language="pt", hot_words=["PIX", "TED"])
        assert result.text == "transferencia PIX"

    async def test_no_hot_words_works(self) -> None:
        backend = WeNetBackend()
        mock_model = MagicMock()
        mock_model.transcribe.return_value = {"text": "hello"}
        backend._model = mock_model  # type: ignore[assignment]

        audio = b"\x00\x01" * 16000
        result = await backend.transcribe_file(audio)
        assert result.text == "hello"


class TestWeNetHotWordsStreaming:
    """Hot words em transcribe_stream passados para a engine."""

    async def test_hot_words_passed_to_stream(self) -> None:
        backend = WeNetBackend()
        mock_model = MagicMock()
        mock_model.transcribe.side_effect = [
            {"text": "PIX transfer"},
            {"text": "PIX transfer"},
        ]
        backend._model = mock_model  # type: ignore[assignment]

        chunk = b"\x00\x01" * 8000

        async def _gen():  # type: ignore[no-untyped-def]
            yield chunk
            yield b""

        segments = [s async for s in backend.transcribe_stream(_gen(), hot_words=["PIX", "TED"])]
        assert len(segments) >= 1


class TestHotWordsRouting:
    """StreamingSession escolhe mecanismo baseado em capabilities."""

    def _make_session(
        self,
        *,
        engine_supports_hot_words: bool = False,
        hot_words: list[str] | None = None,
    ) -> StreamingSession:
        """Cria StreamingSession com mocks minimos."""
        preprocessor = MagicMock()
        vad = MagicMock()
        grpc_client = MagicMock()
        on_event = AsyncMock()

        return StreamingSession(
            session_id="test-sess",
            preprocessor=preprocessor,
            vad=vad,
            grpc_client=grpc_client,
            postprocessor=None,
            on_event=on_event,
            hot_words=hot_words,
            engine_supports_hot_words=engine_supports_hot_words,
        )

    def test_native_hot_words_not_in_prompt(self) -> None:
        """When engine supports hot words natively, they are NOT in initial_prompt."""
        session = self._make_session(
            engine_supports_hot_words=True,
            hot_words=["PIX", "TED", "Selic"],
        )
        prompt = session._build_initial_prompt()
        # Hot words should NOT be injected into prompt
        assert prompt is None

    def test_whisper_hot_words_in_prompt(self) -> None:
        """When engine does NOT support hot words, inject into initial_prompt."""
        session = self._make_session(
            engine_supports_hot_words=False,
            hot_words=["PIX", "TED", "Selic"],
        )
        prompt = session._build_initial_prompt()
        assert prompt is not None
        assert "PIX" in prompt
        assert "TED" in prompt
        assert "Selic" in prompt
        assert "Termos:" in prompt

    def test_no_hot_words_no_prompt(self) -> None:
        """Without hot words, prompt is None regardless of capability."""
        session = self._make_session(
            engine_supports_hot_words=True,
            hot_words=None,
        )
        prompt = session._build_initial_prompt()
        assert prompt is None

    def test_whisper_no_hot_words_no_prompt(self) -> None:
        """Without hot words, prompt is None for Whisper too."""
        session = self._make_session(
            engine_supports_hot_words=False,
            hot_words=None,
        )
        prompt = session._build_initial_prompt()
        assert prompt is None

    def test_decision_by_capability_not_architecture(self) -> None:
        """Routing is by supports_hot_words, NOT by architecture."""
        # Hypothetical encoder-decoder that supports hot words natively
        session = self._make_session(
            engine_supports_hot_words=True,
            hot_words=["PIX"],
        )
        prompt = session._build_initial_prompt()
        # Should NOT inject into prompt even though it could be encoder-decoder
        assert prompt is None

    def test_hot_words_field_always_sent(self) -> None:
        """Hot words are always sent via hot_words field regardless of capability."""
        session = self._make_session(
            engine_supports_hot_words=True,
            hot_words=["PIX", "TED"],
        )
        # The _hot_words field is always populated
        assert session._hot_words == ["PIX", "TED"]

    def test_default_engine_supports_hot_words_is_false(self) -> None:
        """Default is False (backward compatible with Whisper behavior)."""
        preprocessor = MagicMock()
        vad = MagicMock()
        grpc_client = MagicMock()
        on_event = AsyncMock()

        session = StreamingSession(
            session_id="test",
            preprocessor=preprocessor,
            vad=vad,
            grpc_client=grpc_client,
            postprocessor=None,
            on_event=on_event,
            hot_words=["PIX"],
        )
        # Default: inject into prompt (Whisper behavior)
        prompt = session._build_initial_prompt()
        assert prompt is not None
        assert "PIX" in prompt
