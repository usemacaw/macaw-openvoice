"""Testes dos modelos Pydantic para eventos WebSocket de streaming STT."""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from macaw._types import VADSensitivity
from macaw.server.models.events import (
    InputAudioBufferCommitCommand,
    PreprocessingOverrides,
    SessionCancelCommand,
    SessionCloseCommand,
    SessionClosedEvent,
    SessionConfig,
    SessionConfigureCommand,
    SessionCreatedEvent,
    SessionFramesDroppedEvent,
    SessionHoldEvent,
    SessionRateLimitEvent,
    StreamingErrorEvent,
    TranscriptFinalEvent,
    TranscriptPartialEvent,
    VADSpeechEndEvent,
    VADSpeechStartEvent,
    WordEvent,
)


class TestServerEvents:
    """Testes de serializacao dos eventos server -> client."""

    def test_session_created_event_serializes_with_type_field(self) -> None:
        event = SessionCreatedEvent(
            session_id="sess_abc123",
            model="faster-whisper-large-v3",
            config=SessionConfig(),
        )
        data = event.model_dump()
        assert data["type"] == "session.created"
        assert data["session_id"] == "sess_abc123"
        assert data["model"] == "faster-whisper-large-v3"
        assert data["config"]["vad_sensitivity"] == VADSensitivity.NORMAL

    def test_vad_speech_start_event_serializes(self) -> None:
        event = VADSpeechStartEvent(timestamp_ms=1500)
        data = event.model_dump()
        assert data["type"] == "vad.speech_start"
        assert data["timestamp_ms"] == 1500

    def test_vad_speech_end_event_serializes(self) -> None:
        event = VADSpeechEndEvent(timestamp_ms=4000)
        data = event.model_dump()
        assert data["type"] == "vad.speech_end"
        assert data["timestamp_ms"] == 4000

    def test_transcript_partial_event_serializes(self) -> None:
        event = TranscriptPartialEvent(
            text="Ola como",
            segment_id=0,
            timestamp_ms=2000,
        )
        data = event.model_dump()
        assert data["type"] == "transcript.partial"
        assert data["text"] == "Ola como"
        assert data["segment_id"] == 0

    def test_transcript_final_event_serializes_with_words(self) -> None:
        event = TranscriptFinalEvent(
            text="Ola, como posso ajudar?",
            segment_id=0,
            start_ms=1500,
            end_ms=4000,
            language="pt",
            confidence=0.95,
            words=[
                WordEvent(word="Ola", start=1.5, end=2.0),
                WordEvent(word="como", start=2.1, end=2.4),
            ],
        )
        data = event.model_dump()
        assert data["type"] == "transcript.final"
        assert data["confidence"] == 0.95
        assert len(data["words"]) == 2
        assert data["words"][0]["word"] == "Ola"

    def test_transcript_final_event_optional_fields_default_to_none(self) -> None:
        event = TranscriptFinalEvent(
            text="Hello",
            segment_id=1,
            start_ms=0,
            end_ms=1000,
        )
        assert event.language is None
        assert event.confidence is None
        assert event.words is None

    def test_session_hold_event_serializes(self) -> None:
        event = SessionHoldEvent(timestamp_ms=34000, hold_timeout_ms=300_000)
        data = event.model_dump()
        assert data["type"] == "session.hold"
        assert data["hold_timeout_ms"] == 300_000

    def test_session_rate_limit_event_serializes(self) -> None:
        event = SessionRateLimitEvent(
            delay_ms=100,
            message="Client sending faster than real-time, please throttle",
        )
        data = event.model_dump()
        assert data["type"] == "session.rate_limit"
        assert data["delay_ms"] == 100

    def test_session_frames_dropped_event_serializes(self) -> None:
        event = SessionFramesDroppedEvent(
            dropped_ms=500,
            message="Backlog exceeded 10s, frames dropped",
        )
        data = event.model_dump()
        assert data["type"] == "session.frames_dropped"
        assert data["dropped_ms"] == 500

    def test_streaming_error_event_recoverable(self) -> None:
        event = StreamingErrorEvent(
            code="worker_crash",
            message="Worker restarted, resuming from segment 5",
            recoverable=True,
            resume_segment_id=5,
        )
        data = event.model_dump()
        assert data["type"] == "error"
        assert data["recoverable"] is True
        assert data["resume_segment_id"] == 5

    def test_streaming_error_event_irrecoverable(self) -> None:
        event = StreamingErrorEvent(
            code="internal_error",
            message="Fatal error",
            recoverable=False,
        )
        assert event.resume_segment_id is None

    def test_session_closed_event_serializes(self) -> None:
        event = SessionClosedEvent(
            reason="client_request",
            total_duration_ms=45000,
            segments_transcribed=12,
        )
        data = event.model_dump()
        assert data["type"] == "session.closed"
        assert data["reason"] == "client_request"
        assert data["segments_transcribed"] == 12


class TestClientCommands:
    """Testes de desserializacao dos comandos client -> server."""

    def test_session_configure_command_deserializes(self) -> None:
        raw = {
            "type": "session.configure",
            "vad_sensitivity": "high",
            "silence_timeout_ms": 500,
            "language": "pt",
            "hot_words": ["PIX", "TED", "Selic"],
            "hot_word_boost": 2.0,
            "enable_itn": True,
        }
        cmd = SessionConfigureCommand.model_validate(raw)
        assert cmd.vad_sensitivity == VADSensitivity.HIGH
        assert cmd.silence_timeout_ms == 500
        assert cmd.hot_words == ["PIX", "TED", "Selic"]

    def test_session_configure_command_optional_fields_default_to_none(self) -> None:
        cmd = SessionConfigureCommand()
        assert cmd.vad_sensitivity is None
        assert cmd.silence_timeout_ms is None
        assert cmd.language is None
        assert cmd.hot_words is None
        assert cmd.preprocessing is None

    def test_session_cancel_command_serializes(self) -> None:
        cmd = SessionCancelCommand()
        assert cmd.model_dump() == {"type": "session.cancel"}

    def test_input_audio_buffer_commit_command_serializes(self) -> None:
        cmd = InputAudioBufferCommitCommand()
        assert cmd.model_dump() == {"type": "input_audio_buffer.commit"}

    def test_session_close_command_serializes(self) -> None:
        cmd = SessionCloseCommand()
        assert cmd.model_dump() == {"type": "session.close"}


class TestRoundTrip:
    """Testes de round-trip: model -> JSON -> model."""

    def test_session_created_event_roundtrip(self) -> None:
        original = SessionCreatedEvent(
            session_id="sess_xyz",
            model="faster-whisper-tiny",
            config=SessionConfig(
                vad_sensitivity=VADSensitivity.LOW,
                preprocessing=PreprocessingOverrides(denoise=True),
            ),
        )
        json_str = original.model_dump_json()
        restored = SessionCreatedEvent.model_validate_json(json_str)
        assert restored == original
        assert restored.config.vad_sensitivity == VADSensitivity.LOW
        assert restored.config.preprocessing.denoise is True

    def test_transcript_final_event_roundtrip(self) -> None:
        original = TranscriptFinalEvent(
            text="Ola, como posso ajudar?",
            segment_id=0,
            start_ms=1500,
            end_ms=4000,
            language="pt",
            confidence=0.95,
            words=[WordEvent(word="Ola", start=1.5, end=2.0)],
        )
        json_str = original.model_dump_json()
        restored = TranscriptFinalEvent.model_validate_json(json_str)
        assert restored == original

    def test_session_configure_command_roundtrip(self) -> None:
        original = SessionConfigureCommand(
            vad_sensitivity=VADSensitivity.HIGH,
            hot_words=["PIX", "TED"],
            hot_word_boost=2.0,
            enable_partial_transcripts=False,
        )
        json_str = original.model_dump_json()
        restored = SessionConfigureCommand.model_validate_json(json_str)
        assert restored == original


class TestValidation:
    """Testes de validacao e imutabilidade."""

    def test_invalid_vad_sensitivity_raises_validation_error(self) -> None:
        with pytest.raises(ValidationError):
            SessionConfigureCommand.model_validate(
                {"type": "session.configure", "vad_sensitivity": "ultra"}
            )

    def test_frozen_models_are_immutable(self) -> None:
        event = VADSpeechStartEvent(timestamp_ms=1000)
        with pytest.raises(ValidationError):
            event.timestamp_ms = 2000

    def test_session_config_defaults(self) -> None:
        config = SessionConfig()
        assert config.vad_sensitivity == VADSensitivity.NORMAL
        assert config.silence_timeout_ms == 30_000
        assert config.hold_timeout_ms == 300_000
        assert config.max_segment_duration_ms == 30_000
        assert config.enable_partial_transcripts is True
        assert config.enable_itn is True
        assert config.preprocessing.denoise is False

    def test_preprocessing_overrides_defaults(self) -> None:
        overrides = PreprocessingOverrides()
        assert overrides.denoise is False
        assert overrides.denoise_engine == "rnnoise"
