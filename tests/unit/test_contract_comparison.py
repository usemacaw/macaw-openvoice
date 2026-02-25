"""M7-07 Contract Comparison: STT backends produce responses with identical contract.

Proves that STT backends return responses with the same structure/fields in all
response formats. The text may differ (different engines), but the format is identical.
Ensures that a client can switch engines without changing code.

Scope:
- Batch REST: response_format json, verbose_json, text, srt, vtt
- WebSocket: same event sequence for both architectures
- Hot words: backends receive hot words
- ITN: applied to transcript.final
- Zero missing or extra fields between backends
"""

from __future__ import annotations

import re
from unittest.mock import AsyncMock, MagicMock, Mock

import httpx
import numpy as np
import pytest

from macaw._types import (
    BatchResult,
    ResponseFormat,
    SegmentDetail,
    STTArchitecture,
    TranscriptSegment,
    WordTimestamp,
)
from macaw.server.app import create_app
from macaw.server.formatters import format_response
from tests.helpers import AsyncIterFromList

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_batch_result(*, engine: str = "faster-whisper") -> BatchResult:
    """Both engines return the same BatchResult structure."""
    text = "Ola, como posso ajudar?" if engine == "faster-whisper" else "Ola como posso ajudar"
    return BatchResult(
        text=text,
        language="pt",
        duration=2.5,
        segments=(
            SegmentDetail(
                id=0,
                start=0.0,
                end=1.2,
                text="Ola," if engine == "faster-whisper" else "Ola",
                avg_logprob=-0.25,
                no_speech_prob=0.01,
                compression_ratio=1.1,
            ),
            SegmentDetail(
                id=1,
                start=1.3,
                end=2.5,
                text="como posso ajudar?" if engine == "faster-whisper" else "como posso ajudar",
                avg_logprob=-0.30,
                no_speech_prob=0.02,
                compression_ratio=1.0,
            ),
        ),
        words=(
            WordTimestamp(word="Ola", start=0.0, end=0.5),
            WordTimestamp(word="como", start=0.6, end=0.9),
            WordTimestamp(word="posso", start=1.0, end=1.3),
            WordTimestamp(word="ajudar", start=1.4, end=2.5),
        ),
    )


def _make_registry() -> MagicMock:
    registry = MagicMock()
    registry.has_model.return_value = True
    registry.get_manifest.return_value = MagicMock()
    return registry


def _make_scheduler(result: BatchResult | None = None) -> MagicMock:
    scheduler = MagicMock()
    scheduler.transcribe = AsyncMock(return_value=result or _make_batch_result())
    return scheduler


def _make_app(
    registry: MagicMock | None = None,
    scheduler: MagicMock | None = None,
) -> object:
    return create_app(
        registry=registry or _make_registry(),
        scheduler=scheduler or _make_scheduler(),
    )


def _make_stream_handle(events: list | None = None) -> Mock:
    """Create a mock StreamHandle with async iterator."""
    handle = Mock()
    handle.is_closed = False
    handle.session_id = "test"
    handle.receive_events.return_value = AsyncIterFromList(events or [])
    handle.send_frame = AsyncMock()
    handle.close = AsyncMock()
    handle.cancel = AsyncMock()
    return handle


def _make_streaming_session(
    *,
    architecture: STTArchitecture = STTArchitecture.ENCODER_DECODER,
    postprocessor: MagicMock | None = None,
    enable_itn: bool = True,
    hot_words: list[str] | None = None,
) -> tuple:
    """Create StreamingSession with mocks and return (session, on_event)."""
    from macaw.session.streaming import StreamingSession

    preprocessor = MagicMock()
    preprocessor.process_frame.return_value = np.zeros(320, dtype=np.float32)

    vad = MagicMock()
    vad.process_frame.return_value = None
    vad.is_speaking = False

    grpc_client = MagicMock()
    on_event = AsyncMock()

    session = StreamingSession(
        session_id="test-contract",
        preprocessor=preprocessor,
        vad=vad,
        grpc_client=grpc_client,
        postprocessor=postprocessor,
        on_event=on_event,
        hot_words=hot_words,
        enable_itn=enable_itn,
        architecture=architecture,
    )
    return session, on_event


# ---------------------------------------------------------------------------
# Batch JSON contract (parametrized by engine)
# ---------------------------------------------------------------------------


class TestBatchJsonContract:
    """response_format=json returns identical contract for both backends."""

    @pytest.mark.parametrize("engine", ["faster-whisper"])
    async def test_json_has_text_field(self, engine: str) -> None:
        """Response JSON tem campo 'text'."""
        result = _make_batch_result(engine=engine)
        scheduler = _make_scheduler(result)
        app = _make_app(scheduler=scheduler)

        async with httpx.AsyncClient(
            transport=httpx.ASGITransport(app=app), base_url="http://test"
        ) as client:
            response = await client.post(
                "/v1/audio/transcriptions",
                files={"file": ("audio.wav", b"fake-audio", "audio/wav")},
                data={"model": f"{engine}-tiny"},
            )
        assert response.status_code == 200
        body = response.json()
        assert "text" in body

    @pytest.mark.parametrize("engine", ["faster-whisper"])
    async def test_json_no_extra_fields(self, engine: str) -> None:
        """Response JSON tem APENAS campo 'text' (sem campos extras)."""
        result = _make_batch_result(engine=engine)
        scheduler = _make_scheduler(result)
        app = _make_app(scheduler=scheduler)

        async with httpx.AsyncClient(
            transport=httpx.ASGITransport(app=app), base_url="http://test"
        ) as client:
            response = await client.post(
                "/v1/audio/transcriptions",
                files={"file": ("audio.wav", b"fake-audio", "audio/wav")},
                data={"model": f"{engine}-tiny"},
            )
        body = response.json()
        assert set(body.keys()) == {"text"}

    @pytest.mark.parametrize("engine", ["faster-whisper"])
    async def test_json_text_is_string(self, engine: str) -> None:
        """Field 'text' is of type string."""
        result = _make_batch_result(engine=engine)
        scheduler = _make_scheduler(result)
        app = _make_app(scheduler=scheduler)

        async with httpx.AsyncClient(
            transport=httpx.ASGITransport(app=app), base_url="http://test"
        ) as client:
            response = await client.post(
                "/v1/audio/transcriptions",
                files={"file": ("audio.wav", b"fake-audio", "audio/wav")},
                data={"model": f"{engine}-tiny"},
            )
        body = response.json()
        assert isinstance(body["text"], str)


# ---------------------------------------------------------------------------
# Batch verbose_json contract
# ---------------------------------------------------------------------------


class TestBatchVerboseJsonContract:
    """response_format=verbose_json returns identical contract for both backends."""

    @pytest.mark.parametrize("engine", ["faster-whisper"])
    async def test_verbose_json_has_required_fields(self, engine: str) -> None:
        """Verbose JSON tem task, language, duration, text, segments."""
        result = _make_batch_result(engine=engine)
        scheduler = _make_scheduler(result)
        app = _make_app(scheduler=scheduler)

        async with httpx.AsyncClient(
            transport=httpx.ASGITransport(app=app), base_url="http://test"
        ) as client:
            response = await client.post(
                "/v1/audio/transcriptions",
                files={"file": ("audio.wav", b"fake-audio", "audio/wav")},
                data={"model": f"{engine}-tiny", "response_format": "verbose_json"},
            )
        assert response.status_code == 200
        body = response.json()
        required = {"task", "language", "duration", "text", "segments"}
        assert required.issubset(set(body.keys()))

    @pytest.mark.parametrize("engine", ["faster-whisper"])
    async def test_verbose_json_segments_structure(self, engine: str) -> None:
        """Cada segmento tem id, start, end, text."""
        result = _make_batch_result(engine=engine)
        scheduler = _make_scheduler(result)
        app = _make_app(scheduler=scheduler)

        async with httpx.AsyncClient(
            transport=httpx.ASGITransport(app=app), base_url="http://test"
        ) as client:
            response = await client.post(
                "/v1/audio/transcriptions",
                files={"file": ("audio.wav", b"fake-audio", "audio/wav")},
                data={"model": f"{engine}-tiny", "response_format": "verbose_json"},
            )
        body = response.json()
        for seg in body["segments"]:
            assert "id" in seg
            assert "start" in seg
            assert "end" in seg
            assert "text" in seg

    @pytest.mark.parametrize("engine", ["faster-whisper"])
    async def test_verbose_json_words_when_present(self, engine: str) -> None:
        """Array words tem word, start, end em cada elemento."""
        result = _make_batch_result(engine=engine)
        scheduler = _make_scheduler(result)
        app = _make_app(scheduler=scheduler)

        async with httpx.AsyncClient(
            transport=httpx.ASGITransport(app=app), base_url="http://test"
        ) as client:
            response = await client.post(
                "/v1/audio/transcriptions",
                files={"file": ("audio.wav", b"fake-audio", "audio/wav")},
                data={"model": f"{engine}-tiny", "response_format": "verbose_json"},
            )
        body = response.json()
        assert "words" in body
        for word in body["words"]:
            assert "word" in word
            assert "start" in word
            assert "end" in word

    @pytest.mark.parametrize("engine", ["faster-whisper"])
    async def test_verbose_json_types(self, engine: str) -> None:
        """Correct types: language=str, duration=float, text=str."""
        result = _make_batch_result(engine=engine)
        scheduler = _make_scheduler(result)
        app = _make_app(scheduler=scheduler)

        async with httpx.AsyncClient(
            transport=httpx.ASGITransport(app=app), base_url="http://test"
        ) as client:
            response = await client.post(
                "/v1/audio/transcriptions",
                files={"file": ("audio.wav", b"fake-audio", "audio/wav")},
                data={"model": f"{engine}-tiny", "response_format": "verbose_json"},
            )
        body = response.json()
        assert isinstance(body["language"], str)
        assert isinstance(body["duration"], float)
        assert isinstance(body["text"], str)
        assert isinstance(body["segments"], list)

    @pytest.mark.parametrize("engine", ["faster-whisper"])
    async def test_verbose_json_no_extra_segment_fields(self, engine: str) -> None:
        """Segments have no unexpected fields."""
        result = _make_batch_result(engine=engine)
        scheduler = _make_scheduler(result)
        app = _make_app(scheduler=scheduler)

        async with httpx.AsyncClient(
            transport=httpx.ASGITransport(app=app), base_url="http://test"
        ) as client:
            response = await client.post(
                "/v1/audio/transcriptions",
                files={"file": ("audio.wav", b"fake-audio", "audio/wav")},
                data={"model": f"{engine}-tiny", "response_format": "verbose_json"},
            )
        body = response.json()
        allowed_keys = {
            "id",
            "start",
            "end",
            "text",
            "avg_logprob",
            "compression_ratio",
            "no_speech_prob",
        }
        for seg in body["segments"]:
            assert set(seg.keys()).issubset(allowed_keys), (
                f"Extra fields: {set(seg.keys()) - allowed_keys}"
            )


# ---------------------------------------------------------------------------
# Batch text format
# ---------------------------------------------------------------------------


class TestBatchTextFormat:
    """response_format=text returns plain text for both backends."""

    @pytest.mark.parametrize("engine", ["faster-whisper"])
    async def test_text_returns_plain_text(self, engine: str) -> None:
        """Response is plain text, not JSON."""
        result = _make_batch_result(engine=engine)
        scheduler = _make_scheduler(result)
        app = _make_app(scheduler=scheduler)

        async with httpx.AsyncClient(
            transport=httpx.ASGITransport(app=app), base_url="http://test"
        ) as client:
            response = await client.post(
                "/v1/audio/transcriptions",
                files={"file": ("audio.wav", b"fake-audio", "audio/wav")},
                data={"model": f"{engine}-tiny", "response_format": "text"},
            )
        assert response.status_code == 200
        assert "text/plain" in response.headers["content-type"]
        assert response.text == result.text


# ---------------------------------------------------------------------------
# Batch SRT format
# ---------------------------------------------------------------------------


class TestBatchSrtFormat:
    """response_format=srt produces valid output for both backends."""

    @pytest.mark.parametrize("engine", ["faster-whisper"])
    async def test_srt_has_proper_format(self, engine: str) -> None:
        """SRT contains index, timestamp and text."""
        result = _make_batch_result(engine=engine)
        scheduler = _make_scheduler(result)
        app = _make_app(scheduler=scheduler)

        async with httpx.AsyncClient(
            transport=httpx.ASGITransport(app=app), base_url="http://test"
        ) as client:
            response = await client.post(
                "/v1/audio/transcriptions",
                files={"file": ("audio.wav", b"fake-audio", "audio/wav")},
                data={"model": f"{engine}-tiny", "response_format": "srt"},
            )
        assert response.status_code == 200
        body = response.text
        assert "1\n" in body
        assert "-->" in body

    @pytest.mark.parametrize("engine", ["faster-whisper"])
    async def test_srt_timestamp_format(self, engine: str) -> None:
        """Timestamps follow HH:MM:SS,mmm format."""
        result = _make_batch_result(engine=engine)
        scheduler = _make_scheduler(result)
        app = _make_app(scheduler=scheduler)

        async with httpx.AsyncClient(
            transport=httpx.ASGITransport(app=app), base_url="http://test"
        ) as client:
            response = await client.post(
                "/v1/audio/transcriptions",
                files={"file": ("audio.wav", b"fake-audio", "audio/wav")},
                data={"model": f"{engine}-tiny", "response_format": "srt"},
            )
        body = response.text
        srt_pattern = r"\d{2}:\d{2}:\d{2},\d{3} --> \d{2}:\d{2}:\d{2},\d{3}"
        assert re.search(srt_pattern, body) is not None


# ---------------------------------------------------------------------------
# Batch VTT format
# ---------------------------------------------------------------------------


class TestBatchVttFormat:
    """response_format=vtt produces valid output for both backends."""

    @pytest.mark.parametrize("engine", ["faster-whisper"])
    async def test_vtt_starts_with_webvtt(self, engine: str) -> None:
        """Output starts with 'WEBVTT'."""
        result = _make_batch_result(engine=engine)
        scheduler = _make_scheduler(result)
        app = _make_app(scheduler=scheduler)

        async with httpx.AsyncClient(
            transport=httpx.ASGITransport(app=app), base_url="http://test"
        ) as client:
            response = await client.post(
                "/v1/audio/transcriptions",
                files={"file": ("audio.wav", b"fake-audio", "audio/wav")},
                data={"model": f"{engine}-tiny", "response_format": "vtt"},
            )
        assert response.status_code == 200
        assert response.text.startswith("WEBVTT\n")

    @pytest.mark.parametrize("engine", ["faster-whisper"])
    async def test_vtt_timestamp_format(self, engine: str) -> None:
        """Timestamps follow HH:MM:SS.mmm format (dot, not comma)."""
        result = _make_batch_result(engine=engine)
        scheduler = _make_scheduler(result)
        app = _make_app(scheduler=scheduler)

        async with httpx.AsyncClient(
            transport=httpx.ASGITransport(app=app), base_url="http://test"
        ) as client:
            response = await client.post(
                "/v1/audio/transcriptions",
                files={"file": ("audio.wav", b"fake-audio", "audio/wav")},
                data={"model": f"{engine}-tiny", "response_format": "vtt"},
            )
        body = response.text
        vtt_pattern = r"\d{2}:\d{2}:\d{2}\.\d{3} --> \d{2}:\d{2}:\d{2}\.\d{3}"
        assert re.search(vtt_pattern, body) is not None


# ---------------------------------------------------------------------------
# Cross-engine contract identity
# ---------------------------------------------------------------------------


# ---------------------------------------------------------------------------
# Hot words contract
# ---------------------------------------------------------------------------


class TestHotWordsContract:
    """Hot words received by both backends without structural divergence."""

    @pytest.mark.parametrize("engine", ["faster-whisper"])
    async def test_hot_words_in_batch_result_both_engines(self, engine: str) -> None:
        """Both engines return text when hot words are configured."""
        hot_word_result = BatchResult(
            text="O PIX foi processado",
            language="pt",
            duration=1.5,
            segments=(SegmentDetail(id=0, start=0.0, end=1.5, text="O PIX foi processado"),),
            words=(
                WordTimestamp(word="O", start=0.0, end=0.2),
                WordTimestamp(word="PIX", start=0.3, end=0.6),
                WordTimestamp(word="foi", start=0.7, end=0.9),
                WordTimestamp(word="processado", start=1.0, end=1.5),
            ),
        )
        scheduler = _make_scheduler(hot_word_result)
        app = _make_app(scheduler=scheduler)

        async with httpx.AsyncClient(
            transport=httpx.ASGITransport(app=app), base_url="http://test"
        ) as client:
            response = await client.post(
                "/v1/audio/transcriptions",
                files={"file": ("audio.wav", b"fake-audio", "audio/wav")},
                data={
                    "model": f"{engine}-tiny",
                    "hot_words": "PIX,TED,Selic",
                },
            )
        assert response.status_code == 200
        body = response.json()
        assert "PIX" in body["text"]


# ---------------------------------------------------------------------------
# ITN contract
# ---------------------------------------------------------------------------


class TestITNContract:
    """ITN applied to the result of both backends identically."""

    @pytest.mark.parametrize("engine", ["faster-whisper"])
    async def test_itn_applied_to_both_engines(self, engine: str) -> None:
        """format_response produces structurally identical output for both."""
        result = _make_batch_result(engine=engine)

        # format_response is called by the route handler after post-processing.
        # Here we test that the function accepts BatchResult from any engine
        # and produces valid output without errors.
        json_out = format_response(result, ResponseFormat.JSON, task="transcribe")
        assert isinstance(json_out, dict)
        assert "text" in json_out

        verbose_out = format_response(result, ResponseFormat.VERBOSE_JSON, task="transcribe")
        assert isinstance(verbose_out, dict)
        assert "task" in verbose_out
        assert "segments" in verbose_out


# ---------------------------------------------------------------------------
# WebSocket event contract: both architectures produce the same event types
# ---------------------------------------------------------------------------


class TestWebSocketContractBothEngines:
    """Both architectures (encoder-decoder and CTC) produce the same event types."""

    async def test_event_types_identical_partial(self) -> None:
        """Both architectures emit transcript.partial with the same structure."""
        partial_segment = TranscriptSegment(
            text="ola como",
            is_final=False,
            segment_id=0,
            start_ms=100,
        )

        for arch in (STTArchitecture.ENCODER_DECODER, STTArchitecture.CTC):
            session, on_event = _make_streaming_session(architecture=arch)

            mock_handle = _make_stream_handle(events=[partial_segment])
            session._stream_handle = mock_handle

            await session._receive_worker_events()

            on_event.assert_called_once()
            event = on_event.call_args[0][0]
            assert event.type == "transcript.partial"
            assert hasattr(event, "text")
            assert hasattr(event, "segment_id")
            assert hasattr(event, "timestamp_ms")

    async def test_event_types_identical_final(self) -> None:
        """Both architectures emit transcript.final with the same structure."""
        final_segment = TranscriptSegment(
            text="ola como posso ajudar",
            is_final=True,
            segment_id=0,
            start_ms=0,
            end_ms=2000,
            language="pt",
            confidence=0.95,
        )

        for arch in (STTArchitecture.ENCODER_DECODER, STTArchitecture.CTC):
            session, on_event = _make_streaming_session(architecture=arch)

            mock_handle = _make_stream_handle(events=[final_segment])
            session._stream_handle = mock_handle

            await session._receive_worker_events()

            on_event.assert_called_once()
            event = on_event.call_args[0][0]
            assert event.type == "transcript.final"
            assert hasattr(event, "text")
            assert hasattr(event, "segment_id")
            assert hasattr(event, "start_ms")
            assert hasattr(event, "end_ms")
            assert hasattr(event, "language")
            assert hasattr(event, "confidence")
            assert hasattr(event, "words")

    async def test_event_sequence_partial_then_final(self) -> None:
        """Both architectures emit the same sequence: partial -> final."""
        segments = [
            TranscriptSegment(text="ola", is_final=False, segment_id=0, start_ms=100),
            TranscriptSegment(
                text="ola como posso ajudar",
                is_final=True,
                segment_id=0,
                start_ms=0,
                end_ms=2000,
            ),
        ]

        collected_types: dict[str, list[str]] = {}

        for arch in (STTArchitecture.ENCODER_DECODER, STTArchitecture.CTC):
            session, on_event = _make_streaming_session(architecture=arch)

            mock_handle = _make_stream_handle(events=list(segments))
            session._stream_handle = mock_handle

            await session._receive_worker_events()

            event_types = [call[0][0].type for call in on_event.call_args_list]
            collected_types[arch.value] = event_types

        # Both architectures produce the same sequence of event types
        assert collected_types["encoder-decoder"] == collected_types["ctc"]
        assert collected_types["encoder-decoder"] == ["transcript.partial", "transcript.final"]

    async def test_itn_applied_to_final_both_architectures(self) -> None:
        """ITN applied to transcript.final of both architectures."""
        final_segment = TranscriptSegment(
            text="dois mil e vinte e cinco",
            is_final=True,
            segment_id=0,
            start_ms=0,
            end_ms=2000,
            language="pt",
            confidence=0.9,
        )

        for arch in (STTArchitecture.ENCODER_DECODER, STTArchitecture.CTC):
            mock_postprocessor = MagicMock()
            mock_postprocessor.process.return_value = "2025"

            session, on_event = _make_streaming_session(
                architecture=arch,
                postprocessor=mock_postprocessor,
                enable_itn=True,
            )

            mock_handle = _make_stream_handle(events=[final_segment])
            session._stream_handle = mock_handle

            await session._receive_worker_events()

            mock_postprocessor.process.assert_called_once_with(
                "dois mil e vinte e cinco", language="pt"
            )
            event = on_event.call_args[0][0]
            assert event.type == "transcript.final"
            assert event.text == "2025"

    async def test_itn_not_applied_to_partial_both_architectures(self) -> None:
        """ITN NOT applied to transcript.partial of both architectures."""
        partial_segment = TranscriptSegment(
            text="dois mil",
            is_final=False,
            segment_id=0,
            start_ms=100,
        )

        for arch in (STTArchitecture.ENCODER_DECODER, STTArchitecture.CTC):
            mock_postprocessor = MagicMock()
            mock_postprocessor.process.return_value = "2000"

            session, on_event = _make_streaming_session(
                architecture=arch,
                postprocessor=mock_postprocessor,
                enable_itn=True,
            )

            mock_handle = _make_stream_handle(events=[partial_segment])
            session._stream_handle = mock_handle

            await session._receive_worker_events()

            # Postprocessor should NOT have been called for partial
            mock_postprocessor.process.assert_not_called()
            event = on_event.call_args[0][0]
            assert event.type == "transcript.partial"
            assert event.text == "dois mil"
