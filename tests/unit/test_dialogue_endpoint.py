"""Tests for dialogue (multi-speaker TTS) endpoint and models."""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from macaw.server.models.dialogue import (
    MAX_DIALOGUE_SEGMENTS,
    MAX_DIALOGUE_VOICES,
    DialogueInput,
    DialogueRequest,
    DialogueSegmentInfo,
)

# ---------------------------------------------------------------------------
# DialogueInput model
# ---------------------------------------------------------------------------


class TestDialogueInput:
    def test_defaults(self) -> None:
        inp = DialogueInput(text="Hello")
        assert inp.text == "Hello"
        assert inp.voice_id == "default"

    def test_custom_voice(self) -> None:
        inp = DialogueInput(text="Hello", voice_id="voice_custom")
        assert inp.voice_id == "voice_custom"

    def test_empty_text_rejected(self) -> None:
        from pydantic import ValidationError

        with pytest.raises(ValidationError, match="String should have at least 1 character"):
            DialogueInput(text="")


# ---------------------------------------------------------------------------
# DialogueRequest model
# ---------------------------------------------------------------------------


class TestDialogueRequest:
    def test_minimal_request(self) -> None:
        req = DialogueRequest(
            model="kokoro-v1",
            inputs=[DialogueInput(text="Hello")],
        )
        assert req.model == "kokoro-v1"
        assert len(req.inputs) == 1
        assert req.response_format == "wav"
        assert req.speed == 1.0
        assert req.language is None

    def test_multiple_segments(self) -> None:
        req = DialogueRequest(
            model="kokoro-v1",
            inputs=[
                DialogueInput(text="Hello", voice_id="alice"),
                DialogueInput(text="Hi there", voice_id="bob"),
                DialogueInput(text="Good morning", voice_id="alice"),
            ],
        )
        assert len(req.inputs) == 3
        assert req.inputs[1].voice_id == "bob"

    def test_empty_inputs_rejected(self) -> None:
        from pydantic import ValidationError

        with pytest.raises(ValidationError, match="at least 1"):
            DialogueRequest(model="kokoro-v1", inputs=[])

    def test_pcm_format(self) -> None:
        req = DialogueRequest(
            model="kokoro-v1",
            inputs=[DialogueInput(text="Hello")],
            response_format="pcm",
        )
        assert req.response_format == "pcm"

    def test_speed_range(self) -> None:
        req = DialogueRequest(
            model="kokoro-v1",
            inputs=[DialogueInput(text="Hello")],
            speed=2.5,
        )
        assert req.speed == 2.5

    def test_speed_too_high_rejected(self) -> None:
        from pydantic import ValidationError

        with pytest.raises(ValidationError, match="less than or equal to 4"):
            DialogueRequest(
                model="kokoro-v1",
                inputs=[DialogueInput(text="Hello")],
                speed=5.0,
            )

    def test_output_format_override(self) -> None:
        req = DialogueRequest(
            model="kokoro-v1",
            inputs=[DialogueInput(text="Hello")],
            output_format="pcm_16000",
        )
        assert req.output_format == "pcm_16000"


# ---------------------------------------------------------------------------
# DialogueSegmentInfo model
# ---------------------------------------------------------------------------


class TestDialogueSegmentInfo:
    def test_creation(self) -> None:
        info = DialogueSegmentInfo(
            index=0,
            voice_id="alice",
            text_length=10,
            audio_bytes=1024,
        )
        assert info.index == 0
        assert info.voice_id == "alice"
        assert info.text_length == 10
        assert info.audio_bytes == 1024


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------


class TestDialogueConstants:
    def test_max_voices(self) -> None:
        assert MAX_DIALOGUE_VOICES == 10

    def test_max_segments(self) -> None:
        assert MAX_DIALOGUE_SEGMENTS == 100


# ---------------------------------------------------------------------------
# Route — voice validation
# ---------------------------------------------------------------------------


class TestDialogueVoiceValidation:
    def _make_app(
        self,
        *,
        voice_store: object | None = None,
    ) -> MagicMock:
        """Create a minimal mock FastAPI app for dialogue tests."""
        from macaw._types import ModelType

        app = MagicMock()
        registry = MagicMock()
        manifest = MagicMock()
        manifest.model_type = ModelType.TTS
        manifest.name = "kokoro-v1"
        registry.get_manifest.return_value = manifest
        registry.has_model.return_value = True

        worker = MagicMock()
        worker.worker_id = "w-tts-1"
        worker.port = 50052

        worker_manager = MagicMock()
        worker_manager.get_ready_worker.return_value = worker

        app.state.registry = registry
        app.state.worker_manager = worker_manager
        app.state.voice_store = voice_store
        app.state.tts_channels = {}

        return app

    async def test_too_many_unique_voices_rejected(self) -> None:
        """More than MAX_DIALOGUE_VOICES unique voices returns 422."""
        from macaw.exceptions import InvalidRequestError
        from macaw.server.routes.dialogue import create_dialogue

        inputs = [
            DialogueInput(text=f"Text {i}", voice_id=f"voice_{i}")
            for i in range(MAX_DIALOGUE_VOICES + 1)
        ]
        body = DialogueRequest(model="kokoro-v1", inputs=inputs)

        request = MagicMock()
        request.app = self._make_app()

        with pytest.raises(InvalidRequestError, match="Too many unique voices"):
            await create_dialogue(
                body=body,
                request=request,
                registry=request.app.state.registry,
                worker_manager=request.app.state.worker_manager,
            )

    async def test_whitespace_only_text_rejected(self) -> None:
        """Segment with whitespace-only text returns 422."""
        from macaw.exceptions import InvalidRequestError
        from macaw.server.routes.dialogue import create_dialogue

        body = DialogueRequest(
            model="kokoro-v1",
            inputs=[DialogueInput(text="   ")],
        )

        request = MagicMock()
        request.app = self._make_app()

        with pytest.raises(InvalidRequestError, match="Segment 0"):
            await create_dialogue(
                body=body,
                request=request,
                registry=request.app.state.registry,
                worker_manager=request.app.state.worker_manager,
            )


# ---------------------------------------------------------------------------
# Route — saved voice resolution
# ---------------------------------------------------------------------------


class TestDialogueSavedVoiceResolution:
    async def test_saved_voice_resolved(self) -> None:
        """Saved voices (voice_ prefix) are resolved via VoiceStore."""
        from macaw.server.routes.dialogue import _resolve_all_voices

        saved = MagicMock()
        saved.ref_audio_path = None
        saved.ref_text = "ref text"
        saved.instruction = "warm voice"
        saved.language = "en"

        voice_store = AsyncMock()
        voice_store.get.return_value = saved

        request = MagicMock()
        request.app.state.voice_store = voice_store

        body = DialogueRequest(
            model="kokoro-v1",
            inputs=[DialogueInput(text="Hello", voice_id="voice_my_voice")],
        )

        result = await _resolve_all_voices(body, request)

        assert len(result) == 1
        assert result[0].voice == "default"
        assert result[0].ref_text == "ref text"
        assert result[0].instruction == "warm voice"
        assert result[0].language == "en"
        voice_store.get.assert_awaited_once_with("my_voice")

    async def test_saved_voice_not_found_raises(self) -> None:
        """Unknown saved voice raises VoiceNotFoundError."""
        from macaw.exceptions import VoiceNotFoundError
        from macaw.server.routes.dialogue import _resolve_all_voices

        voice_store = AsyncMock()
        voice_store.get.return_value = None

        request = MagicMock()
        request.app.state.voice_store = voice_store

        body = DialogueRequest(
            model="kokoro-v1",
            inputs=[DialogueInput(text="Hello", voice_id="voice_unknown")],
        )

        with pytest.raises(VoiceNotFoundError):
            await _resolve_all_voices(body, request)

    async def test_no_voice_store_raises(self) -> None:
        """Saved voice without VoiceStore raises InvalidRequestError."""
        from macaw.exceptions import InvalidRequestError
        from macaw.server.routes.dialogue import _resolve_all_voices

        request = MagicMock()
        request.app.state.voice_store = None

        body = DialogueRequest(
            model="kokoro-v1",
            inputs=[DialogueInput(text="Hello", voice_id="voice_test")],
        )

        with pytest.raises(InvalidRequestError, match="VoiceStore not configured"):
            await _resolve_all_voices(body, request)

    async def test_preset_voice_passthrough(self) -> None:
        """Non-saved voices pass through as-is."""
        from macaw.server.routes.dialogue import _resolve_all_voices

        request = MagicMock()
        request.app.state.voice_store = None

        body = DialogueRequest(
            model="kokoro-v1",
            inputs=[
                DialogueInput(text="Hello", voice_id="alice"),
                DialogueInput(text="Hi", voice_id="bob"),
            ],
        )

        result = await _resolve_all_voices(body, request)
        assert len(result) == 2
        assert result[0].voice == "alice"
        assert result[1].voice == "bob"

    async def test_saved_voice_with_ref_audio(self) -> None:
        """Saved voice with ref_audio_path loads audio bytes."""
        from macaw.server.routes.dialogue import _resolve_all_voices

        saved = MagicMock()
        saved.ref_audio_path = "/tmp/ref.wav"
        saved.ref_text = None
        saved.instruction = None
        saved.language = None

        voice_store = AsyncMock()
        voice_store.get.return_value = saved

        request = MagicMock()
        request.app.state.voice_store = voice_store

        body = DialogueRequest(
            model="kokoro-v1",
            inputs=[DialogueInput(text="Hello", voice_id="voice_cloned")],
        )

        with patch(
            "macaw.server.routes.dialogue._read_file_bytes",
            return_value=b"fake-audio-data",
        ):
            result = await _resolve_all_voices(body, request)

        assert len(result) == 1
        assert result[0].voice == "default"
        assert result[0].ref_audio == b"fake-audio-data"


# ---------------------------------------------------------------------------
# Route — streaming response
# ---------------------------------------------------------------------------


async def _async_iter(*items: object) -> object:
    """Helper: create a proper async iterator from items."""
    for item in items:
        yield item


class TestDialogueStreaming:
    async def test_stream_yields_wav_header_first(self) -> None:
        """Streaming dialogue in WAV format yields header before audio."""
        from macaw.codec.output_format import parse_output_format
        from macaw.server.routes.dialogue import _ResolvedVoice, _stream_dialogue

        body = DialogueRequest(
            model="kokoro-v1",
            inputs=[DialogueInput(text="Hello", voice_id="alice")],
        )
        resolved = [_ResolvedVoice(voice="alice")]
        output_fmt = parse_output_format("wav")

        mock_chunk = MagicMock()
        mock_chunk.audio_data = b"\x00\x00" * 100
        mock_chunk.is_last = True

        mock_channel = MagicMock()

        with patch(
            "macaw.server.routes.dialogue.TTSWorkerStub",
        ) as mock_stub_cls:
            stub_instance = MagicMock()
            stub_instance.Synthesize.return_value = _async_iter(mock_chunk)
            mock_stub_cls.return_value = stub_instance

            chunks: list[bytes] = []
            async for chunk in _stream_dialogue(
                channel=mock_channel,
                body=body,
                resolved_voices=resolved,
                output_fmt=output_fmt,
                worker_id="w1",
                request_id="req1",
            ):
                chunks.append(chunk)

        # First chunk should be WAV header (44 bytes)
        assert len(chunks) >= 2
        assert chunks[0][:4] == b"RIFF"
        assert len(chunks[0]) == 44

    async def test_stream_pcm_no_header(self) -> None:
        """PCM format streams raw audio without header."""
        from macaw.codec.output_format import parse_output_format
        from macaw.server.routes.dialogue import _ResolvedVoice, _stream_dialogue

        body = DialogueRequest(
            model="kokoro-v1",
            inputs=[DialogueInput(text="Hello", voice_id="alice")],
            response_format="pcm",
        )
        resolved = [_ResolvedVoice(voice="alice")]
        output_fmt = parse_output_format("pcm")

        mock_chunk = MagicMock()
        mock_chunk.audio_data = b"\x01\x02" * 50
        mock_chunk.is_last = True

        mock_channel = MagicMock()

        with patch(
            "macaw.server.routes.dialogue.TTSWorkerStub",
        ) as mock_stub_cls:
            stub_instance = MagicMock()
            stub_instance.Synthesize.return_value = _async_iter(mock_chunk)
            mock_stub_cls.return_value = stub_instance

            chunks: list[bytes] = []
            async for chunk in _stream_dialogue(
                channel=mock_channel,
                body=body,
                resolved_voices=resolved,
                output_fmt=output_fmt,
                worker_id="w1",
                request_id="req1",
            ):
                chunks.append(chunk)

        # No WAV header — first chunk is raw audio
        assert len(chunks) >= 1
        assert chunks[0][:4] != b"RIFF"

    async def test_multiple_segments_concatenated(self) -> None:
        """Multiple segments yield audio from each in order."""
        from macaw.codec.output_format import parse_output_format
        from macaw.server.routes.dialogue import _ResolvedVoice, _stream_dialogue

        body = DialogueRequest(
            model="kokoro-v1",
            inputs=[
                DialogueInput(text="Hello", voice_id="alice"),
                DialogueInput(text="Hi", voice_id="bob"),
            ],
            response_format="pcm",
        )
        resolved = [
            _ResolvedVoice(voice="alice"),
            _ResolvedVoice(voice="bob"),
        ]
        output_fmt = parse_output_format("pcm")

        chunk_alice = MagicMock()
        chunk_alice.audio_data = b"\xaa" * 100
        chunk_alice.is_last = True

        chunk_bob = MagicMock()
        chunk_bob.audio_data = b"\xbb" * 100
        chunk_bob.is_last = True

        call_count = 0

        def make_stream(*_args: object, **_kwargs: object) -> object:
            nonlocal call_count
            if call_count == 0:
                call_count += 1
                return _async_iter(chunk_alice)
            call_count += 1
            return _async_iter(chunk_bob)

        mock_channel = MagicMock()

        with patch(
            "macaw.server.routes.dialogue.TTSWorkerStub",
        ) as mock_stub_cls:
            stub_instance = MagicMock()
            stub_instance.Synthesize.side_effect = make_stream
            mock_stub_cls.return_value = stub_instance

            chunks: list[bytes] = []
            async for chunk in _stream_dialogue(
                channel=mock_channel,
                body=body,
                resolved_voices=resolved,
                output_fmt=output_fmt,
                worker_id="w1",
                request_id="req1",
            ):
                chunks.append(chunk)

        # Both segments' audio should be present
        all_audio = b"".join(chunks)
        assert b"\xaa" * 100 in all_audio
        assert b"\xbb" * 100 in all_audio

    async def test_segment_uses_correct_voice(self) -> None:
        """Each segment passes the correct voice to build_tts_proto_request."""
        from macaw.codec.output_format import parse_output_format
        from macaw.server.routes.dialogue import _ResolvedVoice, _stream_dialogue

        body = DialogueRequest(
            model="kokoro-v1",
            inputs=[
                DialogueInput(text="Hello", voice_id="alice"),
                DialogueInput(text="Hi", voice_id="bob"),
            ],
            response_format="pcm",
        )
        resolved = [
            _ResolvedVoice(voice="alice"),
            _ResolvedVoice(voice="bob"),
        ]
        output_fmt = parse_output_format("pcm")

        empty_chunk = MagicMock()
        empty_chunk.audio_data = b"\x00\x00"
        empty_chunk.is_last = True

        mock_channel = MagicMock()

        with (
            patch(
                "macaw.server.routes.dialogue.TTSWorkerStub",
            ) as mock_stub_cls,
            patch(
                "macaw.server.routes.dialogue.build_tts_proto_request",
            ) as mock_build,
        ):
            mock_build.return_value = MagicMock()
            stub_instance = MagicMock()
            stub_instance.Synthesize.return_value = _async_iter(empty_chunk)
            mock_stub_cls.return_value = stub_instance

            chunks: list[bytes] = []
            async for chunk in _stream_dialogue(
                channel=mock_channel,
                body=body,
                resolved_voices=resolved,
                output_fmt=output_fmt,
                worker_id="w1",
                request_id="req1",
            ):
                chunks.append(chunk)

        # Verify build_tts_proto_request was called with correct voices
        assert mock_build.call_count == 2
        first_call = mock_build.call_args_list[0]
        second_call = mock_build.call_args_list[1]
        assert first_call.kwargs["voice"] == "alice"
        assert second_call.kwargs["voice"] == "bob"


# ---------------------------------------------------------------------------
# Route — ResolvedVoice
# ---------------------------------------------------------------------------


class TestResolvedVoice:
    def test_defaults(self) -> None:
        from macaw.server.routes.dialogue import _ResolvedVoice

        rv = _ResolvedVoice(voice="alice")
        assert rv.voice == "alice"
        assert rv.ref_audio is None
        assert rv.ref_text is None
        assert rv.instruction is None
        assert rv.language is None

    def test_full_params(self) -> None:
        from macaw.server.routes.dialogue import _ResolvedVoice

        rv = _ResolvedVoice(
            voice="default",
            ref_audio=b"audio",
            ref_text="ref",
            instruction="warm",
            language="en",
        )
        assert rv.voice == "default"
        assert rv.ref_audio == b"audio"
        assert rv.ref_text == "ref"
        assert rv.instruction == "warm"
        assert rv.language == "en"


# ---------------------------------------------------------------------------
# WAV header
# ---------------------------------------------------------------------------


class TestDialogueWavHeader:
    def test_wav_header_structure(self) -> None:
        from macaw.server.routes.dialogue import _wav_streaming_header

        header = _wav_streaming_header(24000)
        assert len(header) == 44
        assert header[:4] == b"RIFF"
        assert header[8:12] == b"WAVE"
        assert header[12:16] == b"fmt "
        assert header[36:40] == b"data"

    def test_wav_header_sample_rate(self) -> None:
        import struct

        from macaw.server.routes.dialogue import _wav_streaming_header

        header = _wav_streaming_header(16000)
        sample_rate = struct.unpack_from("<I", header, 24)[0]
        assert sample_rate == 16000


# ---------------------------------------------------------------------------
# Output format resolution
# ---------------------------------------------------------------------------


class TestDialogueOutputFormat:
    def test_default_wav(self) -> None:
        from macaw.server.routes.dialogue import _resolve_output_format

        body = DialogueRequest(
            model="kokoro-v1",
            inputs=[DialogueInput(text="Hello")],
        )
        fmt = _resolve_output_format(body)
        assert fmt.codec == "wav"

    def test_output_format_takes_precedence(self) -> None:
        from macaw.server.routes.dialogue import _resolve_output_format

        body = DialogueRequest(
            model="kokoro-v1",
            inputs=[DialogueInput(text="Hello")],
            response_format="wav",
            output_format="pcm_16000",
        )
        fmt = _resolve_output_format(body)
        assert fmt.codec == "pcm"
        assert fmt.sample_rate == 16000

    def test_invalid_output_format_raises(self) -> None:
        from macaw.exceptions import InvalidRequestError
        from macaw.server.routes.dialogue import _resolve_output_format

        body = DialogueRequest(
            model="kokoro-v1",
            inputs=[DialogueInput(text="Hello")],
            output_format="invalid_format",
        )
        with pytest.raises(InvalidRequestError):
            _resolve_output_format(body)
