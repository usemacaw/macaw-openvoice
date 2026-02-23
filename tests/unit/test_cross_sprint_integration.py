"""Cross-sprint integration tests for Sprints A, B, and C.

Verifies that independently developed sprints compose correctly:
- Sprint A: GET /v1/models capabilities, POST /v1/audio/align, TTS param validation
- Sprint B: tts.append/flush/clear WS commands, TTSTextBuffer, tts.buffer_flushed
- Sprint C: hot_words_mode on EngineCapabilities, proto diarization fields, SpeakerSegment

These tests focus on contract verification: types, unions, protocol dispatch,
and field availability across sprint boundaries.
"""

from __future__ import annotations

import inspect
import typing

from macaw._types import (
    EngineCapabilities,
    SpeakerSegment,
    TTSAlignmentItem,
    TTSEngineCapabilities,
)
from macaw.server.models.events import (
    ClientCommand,
    ServerEvent,
    SessionConfigureCommand,
    TTSAppendCommand,
    TTSBufferFlushedEvent,
    TTSClearCommand,
    TTSFlushCommand,
    TTSSpeakCommand,
)
from macaw.server.models.models import ModelCapabilitiesResponse
from macaw.server.ws_protocol import (
    _COMMAND_TYPES,
    CommandResult,
    dispatch_message,
)
from macaw.workers.tts._validation import validate_params_against_capabilities
from macaw.workers.tts.converters import SynthesizeParams

# ---------------------------------------------------------------------------
# 1. Models endpoint combines all capability fields
# ---------------------------------------------------------------------------


def test_models_capabilities_includes_all_sprint_fields():
    """ModelCapabilitiesResponse has fields from Sprint A (TTS) and Sprint C (STT)."""
    # Arrange
    caps = ModelCapabilitiesResponse(
        streaming=True,
        voice_cloning=True,
        alignment=True,
        hot_words_mode="native",
        diarization=True,
    )

    # Assert -- Sprint A fields
    assert caps.voice_cloning is True
    assert caps.alignment is True
    # Assert -- Sprint C fields
    assert caps.hot_words_mode == "native"
    assert caps.diarization is True
    # Assert -- baseline fields unchanged
    assert caps.streaming is True


# ---------------------------------------------------------------------------
# 2. WS protocol handles all 9 command types
# ---------------------------------------------------------------------------


def test_ws_protocol_dispatches_all_nine_commands():
    """_COMMAND_TYPES dict has all 9 command types registered."""
    expected = {
        "session.configure",
        "session.cancel",
        "session.close",
        "input_audio_buffer.commit",
        "tts.speak",
        "tts.cancel",
        "tts.append",
        "tts.flush",
        "tts.clear",
    }

    assert set(_COMMAND_TYPES.keys()) == expected


# ---------------------------------------------------------------------------
# 3. ClientCommand union includes all command types
# ---------------------------------------------------------------------------


def test_client_command_union_complete():
    """ClientCommand union type covers all 9 command models."""
    args = typing.get_args(ClientCommand)

    assert len(args) == 9


# ---------------------------------------------------------------------------
# 4. ServerEvent union includes TTSBufferFlushedEvent
# ---------------------------------------------------------------------------


def test_server_event_union_includes_buffer_flushed():
    """ServerEvent union type includes TTSBufferFlushedEvent (Sprint B)."""
    args = typing.get_args(ServerEvent)

    assert TTSBufferFlushedEvent in args


# ---------------------------------------------------------------------------
# 5. SessionConfigureCommand has all fields from all sprints
# ---------------------------------------------------------------------------


def test_session_configure_has_all_fields():
    """session.configure includes fields from base + Sprint B (split_strategy, flush_timeout)."""
    # Arrange & Act
    cmd = SessionConfigureCommand(
        vad_sensitivity="normal",
        language="en",
        tts_split_strategy="sentence",
        tts_flush_timeout_ms=3000,
    )

    # Assert -- Sprint B fields
    assert cmd.tts_split_strategy == "sentence"
    assert cmd.tts_flush_timeout_ms == 3000
    # Assert -- baseline fields unchanged
    assert cmd.language == "en"


# ---------------------------------------------------------------------------
# 6. EngineCapabilities has both hot_words_mode and supports_diarization
# ---------------------------------------------------------------------------


def test_engine_capabilities_has_all_sprint_c_fields():
    """EngineCapabilities has hot_words_mode AND supports_diarization from Sprint C."""
    # Arrange & Act
    caps = EngineCapabilities(
        hot_words_mode="prompt_injection",
        supports_diarization=True,
        supports_hot_words=True,
    )

    # Assert
    assert caps.hot_words_mode == "prompt_injection"
    assert caps.supports_diarization is True
    assert caps.supports_hot_words is True


# ---------------------------------------------------------------------------
# 7. TTSEngineCapabilities has all Sprint A validation fields
# ---------------------------------------------------------------------------


def test_tts_engine_capabilities_has_all_sprint_a_fields():
    """TTSEngineCapabilities has supports_seed, supports_temperature, etc from Sprint A."""
    # Arrange & Act
    caps = TTSEngineCapabilities(
        supports_seed=True,
        supports_temperature=True,
        supports_top_k=True,
        supports_top_p=True,
        supports_text_normalization=True,
        supports_speed=True,
        supports_alignment=True,
    )

    # Assert
    assert caps.supports_seed is True
    assert caps.supports_temperature is True
    assert caps.supports_top_k is True
    assert caps.supports_top_p is True
    assert caps.supports_text_normalization is True
    assert caps.supports_speed is True
    assert caps.supports_alignment is True


# ---------------------------------------------------------------------------
# 8. TTS param validation works with full capabilities
# ---------------------------------------------------------------------------


def test_tts_validation_with_full_capabilities():
    """validate_params_against_capabilities accepts all params when engine supports everything."""
    # Arrange
    params = SynthesizeParams(
        text="hello",
        voice="default",
        sample_rate=24000,
        speed=1.5,
        include_alignment=False,
        alignment_granularity="word",
        options={
            "seed": 42,
            "temperature": 0.8,
            "top_k": 50,
            "top_p": 0.9,
            "text_normalization": "off",
        },
    )
    caps = TTSEngineCapabilities(
        supports_seed=True,
        supports_temperature=True,
        supports_top_k=True,
        supports_top_p=True,
        supports_text_normalization=True,
        supports_speed=True,
    )

    # Act
    unsupported = validate_params_against_capabilities(params, caps)

    # Assert
    assert unsupported == []


def test_tts_validation_flags_unsupported_params():
    """validate_params_against_capabilities rejects params the engine does not support."""
    # Arrange
    params = SynthesizeParams(
        text="hello",
        voice="default",
        sample_rate=24000,
        speed=2.0,
        options={"seed": 42, "temperature": 0.5},
    )
    caps = TTSEngineCapabilities(
        supports_seed=False,
        supports_temperature=False,
        supports_speed=False,
    )

    # Act
    unsupported = validate_params_against_capabilities(params, caps)

    # Assert
    assert "speed" in unsupported
    assert "seed" in unsupported
    assert "temperature" in unsupported


# ---------------------------------------------------------------------------
# 9. Proto backward compatibility
# ---------------------------------------------------------------------------


def test_proto_old_request_without_new_fields():
    """TranscribeFileRequest works without diarize/max_speakers (backward compat)."""
    from macaw.proto.stt_worker_pb2 import TranscribeFileRequest

    # Arrange & Act
    req = TranscribeFileRequest(
        request_id="test",
        audio_data=b"audio",
        language="en",
    )

    # Assert -- new fields should have defaults
    assert req.diarize is False
    assert req.max_speakers == 0


def test_proto_speaker_segment_roundtrip():
    """Proto SpeakerSegment can be created and fields are accessible."""
    from macaw.proto.stt_worker_pb2 import SpeakerSegment as ProtoSpeakerSegment

    # Arrange & Act
    seg = ProtoSpeakerSegment(
        speaker_id="speaker_0",
        start=0.0,
        end=1.5,
        text="hello world",
    )

    # Assert
    assert seg.speaker_id == "speaker_0"
    assert seg.start == 0.0
    assert seg.end == 1.5
    assert seg.text == "hello world"


# ---------------------------------------------------------------------------
# 10. tts.append and tts.speak coexist
# ---------------------------------------------------------------------------


def test_tts_append_and_speak_coexist_in_protocol():
    """Both tts.speak and tts.append are valid commands in the same protocol."""
    # Arrange & Act
    speak_result = dispatch_message({"text": '{"type":"tts.speak","text":"hello"}'})
    append_result = dispatch_message({"text": '{"type":"tts.append","text":"hello"}'})

    # Assert
    assert isinstance(speak_result, CommandResult)
    assert isinstance(append_result, CommandResult)
    assert isinstance(speak_result.command, TTSSpeakCommand)
    assert isinstance(append_result.command, TTSAppendCommand)


def test_tts_flush_and_clear_are_valid_commands():
    """tts.flush and tts.clear are valid commands in the WS protocol."""
    # Act
    flush_result = dispatch_message({"text": '{"type":"tts.flush"}'})
    clear_result = dispatch_message({"text": '{"type":"tts.clear"}'})

    # Assert
    assert isinstance(flush_result, CommandResult)
    assert isinstance(clear_result, CommandResult)
    assert isinstance(flush_result.command, TTSFlushCommand)
    assert isinstance(clear_result.command, TTSClearCommand)


# ---------------------------------------------------------------------------
# 11. TTSTextBuffer independent from Session Manager
# ---------------------------------------------------------------------------


def test_tts_text_buffer_no_session_dependency():
    """TTSTextBuffer has no imports from macaw.session -- respects FSM isolation."""
    import macaw.server.tts_text_buffer as mod

    # Arrange
    source = inspect.getsource(mod)

    # Assert -- no session imports
    assert "macaw.session" not in source
    assert "SessionStateMachine" not in source
    assert "_VALID_TRANSITIONS" not in source


def test_tts_text_buffer_basic_functionality():
    """TTSTextBuffer accumulates and splits text independently of session state."""
    from macaw.server.tts_text_buffer import TTSTextBuffer

    # Arrange
    buf = TTSTextBuffer(split_strategy="sentence")

    # Act -- append a complete sentence
    segments = buf.append("Hello world. ", "req-1")

    # Assert
    assert len(segments) == 1
    assert segments[0] == "Hello world."

    # Act -- flush remaining
    remaining = buf.flush()
    assert remaining is None or remaining == ""


# ---------------------------------------------------------------------------
# 12. ModelCapabilitiesResponse has fields from all sprints
# ---------------------------------------------------------------------------


def test_model_capabilities_response_all_fields():
    """ModelCapabilitiesResponse includes fields from Sprint A + Sprint C."""
    # Arrange & Act
    caps = ModelCapabilitiesResponse(
        streaming=True,
        voice_cloning=True,
        alignment=True,
        character_alignment=True,
        hot_words_mode="native",
        diarization=True,
        hot_words=True,
    )

    # Assert -- Sprint A
    assert caps.voice_cloning is True
    assert caps.alignment is True
    assert caps.character_alignment is True
    # Assert -- Sprint C
    assert caps.hot_words_mode == "native"
    assert caps.diarization is True
    assert caps.hot_words is True


def test_model_capabilities_response_defaults():
    """ModelCapabilitiesResponse defaults are sane for omitted fields."""
    # Arrange & Act
    caps = ModelCapabilitiesResponse()

    # Assert -- all booleans default to False, strings default to "none"
    assert caps.streaming is False
    assert caps.voice_cloning is False
    assert caps.alignment is False
    assert caps.diarization is False
    assert caps.hot_words_mode == "none"


# ---------------------------------------------------------------------------
# 13. SpeakerSegment and TTSAlignmentItem are separate types
# ---------------------------------------------------------------------------


def test_speaker_segment_and_alignment_item_distinct():
    """SpeakerSegment (Sprint C) and TTSAlignmentItem (Sprint 6) are separate types."""
    # Assert -- type identity
    assert SpeakerSegment is not TTSAlignmentItem

    # Arrange
    seg = SpeakerSegment(speaker_id="s0", start=0.0, end=1.0, text="hi")
    item = TTSAlignmentItem(text="hi", start_ms=0, duration_ms=1000)

    # Assert -- SpeakerSegment has speaker_id
    assert hasattr(seg, "speaker_id")
    assert seg.speaker_id == "s0"
    # Assert -- TTSAlignmentItem does NOT have speaker_id
    assert not hasattr(item, "speaker_id")
    # Assert -- TTSAlignmentItem has timing fields that SpeakerSegment does not
    assert hasattr(item, "start_ms")
    assert hasattr(item, "duration_ms")
    assert not hasattr(seg, "start_ms")
    assert not hasattr(seg, "duration_ms")


# ---------------------------------------------------------------------------
# 14. Sprint B commands carry request_id
# ---------------------------------------------------------------------------


def test_sprint_b_commands_carry_request_id():
    """tts.append, tts.flush, tts.clear all support optional request_id."""
    # Arrange & Act
    append = TTSAppendCommand(text="hello", request_id="req-42")
    flush = TTSFlushCommand(request_id="req-42")
    clear = TTSClearCommand(request_id="req-42")

    # Assert
    assert append.request_id == "req-42"
    assert flush.request_id == "req-42"
    assert clear.request_id == "req-42"


def test_sprint_b_commands_request_id_defaults_to_none():
    """Sprint B commands default request_id to None."""
    # Arrange & Act
    flush = TTSFlushCommand()
    clear = TTSClearCommand()

    # Assert
    assert flush.request_id is None
    assert clear.request_id is None


# ---------------------------------------------------------------------------
# 15. TTSBufferFlushedEvent carries expected fields
# ---------------------------------------------------------------------------


def test_tts_buffer_flushed_event_fields():
    """TTSBufferFlushedEvent has request_id, text, and trigger fields."""
    # Arrange & Act
    event = TTSBufferFlushedEvent(
        request_id="req-1",
        text="Hello world.",
        trigger="manual",
    )

    # Assert
    assert event.type == "tts.buffer_flushed"
    assert event.request_id == "req-1"
    assert event.text == "Hello world."
    assert event.trigger == "manual"


def test_tts_buffer_flushed_event_trigger_values():
    """TTSBufferFlushedEvent trigger accepts all 4 valid values."""
    for trigger in ("manual", "auto_split", "auto_timeout", "new_request_id"):
        event = TTSBufferFlushedEvent(request_id="req-1", text="test", trigger=trigger)
        assert event.trigger == trigger


# ---------------------------------------------------------------------------
# 16. EngineCapabilities frozen immutability
# ---------------------------------------------------------------------------


def test_engine_capabilities_is_frozen():
    """EngineCapabilities (Sprint C fields) is immutable (frozen dataclass)."""
    # Arrange
    caps = EngineCapabilities(
        hot_words_mode="native",
        supports_diarization=True,
    )

    # Act & Assert
    try:
        caps.hot_words_mode = "none"  # type: ignore[misc]
        raise AssertionError("Expected FrozenInstanceError")
    except AttributeError:
        pass  # Expected: frozen dataclass rejects mutation


def test_tts_engine_capabilities_is_frozen():
    """TTSEngineCapabilities (Sprint A fields) is immutable (frozen dataclass)."""
    # Arrange
    caps = TTSEngineCapabilities(supports_seed=True)

    # Act & Assert
    try:
        caps.supports_seed = False  # type: ignore[misc]
        raise AssertionError("Expected FrozenInstanceError")
    except AttributeError:
        pass  # Expected: frozen dataclass rejects mutation


# ---------------------------------------------------------------------------
# 17. Combined Sprint A + B: tts.speak with alignment + Sprint B buffer
# ---------------------------------------------------------------------------


def test_tts_speak_command_has_alignment_fields():
    """TTSSpeakCommand (used by Sprint A alignment) has include_alignment field."""
    # Arrange & Act
    cmd = TTSSpeakCommand(
        text="hello world",
        include_alignment=True,
        alignment_granularity="character",
    )

    # Assert
    assert cmd.include_alignment is True
    assert cmd.alignment_granularity == "character"


def test_tts_speak_and_append_have_disjoint_concerns():
    """tts.speak carries synthesis params; tts.append only carries text + request_id."""
    # Arrange
    speak_fields = set(TTSSpeakCommand.model_fields.keys())
    append_fields = set(TTSAppendCommand.model_fields.keys())

    # Assert -- tts.speak has richer param set
    assert "voice" in speak_fields
    assert "speed" in speak_fields
    assert "include_alignment" in speak_fields
    # Assert -- tts.append is minimal
    assert "voice" not in append_fields
    assert "speed" not in append_fields
    assert "include_alignment" not in append_fields
    # Assert -- both share text and type
    assert "text" in speak_fields
    assert "text" in append_fields
    assert "type" in speak_fields
    assert "type" in append_fields
