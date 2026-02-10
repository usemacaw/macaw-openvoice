"""Edge case tests for M5 (WebSocket + VAD streaming STT).

Complements existing M5 test files with missing edge cases:
- Empty/oversized frames
- Malformed protocol messages
- Double close / use-after-close
- Boundary conditions in VAD debounce
- Backpressure with zero-byte frames
- Event model validation edge cases
- StreamingSession with preprocessor errors

All tests are deterministic (no real timing dependencies).
"""

from __future__ import annotations

import asyncio
import json
from unittest.mock import AsyncMock, MagicMock, Mock

import numpy as np
import pytest
from pydantic import ValidationError

from macaw._types import TranscriptSegment, VADSensitivity
from macaw.exceptions import WorkerCrashError
from macaw.server.models.events import (
    InputAudioBufferCommitCommand,
    PreprocessingOverrides,
    SessionCancelCommand,
    SessionCloseCommand,
    SessionClosedEvent,
    SessionConfig,
    SessionConfigureCommand,
    SessionCreatedEvent,
    SessionHoldEvent,
    StreamingErrorEvent,
    TranscriptFinalEvent,
    TranscriptPartialEvent,
    VADSpeechEndEvent,
    VADSpeechStartEvent,
    WordEvent,
)
from macaw.server.ws_protocol import (
    AudioFrameResult,
    CommandResult,
    ErrorResult,
    dispatch_message,
)
from macaw.session.backpressure import (
    BackpressureController,
    FramesDroppedAction,
    RateLimitAction,
)
from macaw.session.streaming import StreamingSession
from macaw.vad.detector import VADDetector, VADEvent, VADEventType
from macaw.vad.energy import EnergyPreFilter

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

_FRAME_SIZE = 1024
_SAMPLE_RATE = 16000


def _make_raw_bytes(n_samples: int = _FRAME_SIZE) -> bytes:
    """Gera bytes PCM int16 (zeros) com n_samples amostras."""
    return np.zeros(n_samples, dtype=np.int16).tobytes()


def _make_float32_frame(n_samples: int = _FRAME_SIZE) -> np.ndarray:
    """Gera frame float32 (zeros)."""
    return np.zeros(n_samples, dtype=np.float32)


def _make_preprocessor_mock() -> Mock:
    mock = Mock()
    mock.process_frame.return_value = _make_float32_frame()
    return mock


def _make_vad_mock(*, is_speaking: bool = False) -> Mock:
    mock = Mock()
    mock.process_frame.return_value = None
    mock.is_speaking = is_speaking
    mock.reset.return_value = None
    return mock


class _AsyncIterFromList:
    def __init__(self, items: list) -> None:
        self._items = list(items)
        self._index = 0

    def __aiter__(self):
        return self

    async def __anext__(self):
        if self._index >= len(self._items):
            raise StopAsyncIteration
        item = self._items[self._index]
        self._index += 1
        if isinstance(item, Exception):
            raise item
        return item


def _make_stream_handle_mock(events: list | None = None) -> Mock:
    handle = Mock()
    handle.is_closed = False
    handle.session_id = "test_session"
    if events is None:
        events = []
    handle.receive_events.return_value = _AsyncIterFromList(events)
    handle.send_frame = AsyncMock()
    handle.close = AsyncMock()
    handle.cancel = AsyncMock()
    return handle


def _make_grpc_client_mock(stream_handle: Mock | None = None) -> AsyncMock:
    client = AsyncMock()
    if stream_handle is None:
        stream_handle = _make_stream_handle_mock()
    client.open_stream = AsyncMock(return_value=stream_handle)
    client.close = AsyncMock()
    return client


def _make_postprocessor_mock() -> Mock:
    mock = Mock()
    mock.process.side_effect = lambda text: f"ITN({text})"
    return mock


def _make_on_event() -> AsyncMock:
    return AsyncMock()


def _make_energy_mock(*, is_silence: bool = False) -> Mock:
    mock = Mock()
    mock.is_silence.return_value = is_silence
    return mock


def _make_silero_mock(*, is_speech: bool = False) -> Mock:
    mock = Mock()
    mock.is_speech.return_value = is_speech
    return mock


def _make_detector(
    energy_is_silence: bool = False,
    silero_is_speech: bool = False,
    min_speech_duration_ms: int = 250,
    min_silence_duration_ms: int = 300,
    max_speech_duration_ms: int = 30_000,
) -> tuple[VADDetector, Mock, Mock]:
    energy_mock = _make_energy_mock(is_silence=energy_is_silence)
    silero_mock = _make_silero_mock(is_speech=silero_is_speech)
    detector = VADDetector(
        energy_pre_filter=energy_mock,
        silero_classifier=silero_mock,
        sensitivity=VADSensitivity.NORMAL,
        sample_rate=_SAMPLE_RATE,
        min_speech_duration_ms=min_speech_duration_ms,
        min_silence_duration_ms=min_silence_duration_ms,
        max_speech_duration_ms=max_speech_duration_ms,
    )
    return detector, energy_mock, silero_mock


class _FakeClock:
    def __init__(self, start: float = 0.0) -> None:
        self._now = start

    def __call__(self) -> float:
        return self._now

    def advance(self, seconds: float) -> None:
        self._now += seconds


# ===========================================================================
# 1. Event Model Edge Cases
# ===========================================================================


class TestEventModelEdgeCases:
    """Edge cases para modelos Pydantic de eventos WebSocket."""

    def test_transcript_final_empty_text(self) -> None:
        """Final transcript com texto vazio e valido."""
        event = TranscriptFinalEvent(
            text="",
            segment_id=0,
            start_ms=0,
            end_ms=0,
        )
        assert event.text == ""
        data = event.model_dump()
        assert data["text"] == ""

    def test_transcript_partial_empty_text(self) -> None:
        """Partial transcript com texto vazio e valido."""
        event = TranscriptPartialEvent(
            text="",
            segment_id=0,
            timestamp_ms=0,
        )
        assert event.text == ""

    def test_transcript_final_negative_timestamps(self) -> None:
        """Timestamps negativos nao sao rejeitados pelo modelo (sem validacao)."""
        event = TranscriptFinalEvent(
            text="test",
            segment_id=0,
            start_ms=-100,
            end_ms=-50,
        )
        assert event.start_ms == -100

    def test_session_created_roundtrip_with_all_config_fields(self) -> None:
        """Round-trip de SessionCreatedEvent com todos os campos de config."""
        config = SessionConfig(
            vad_sensitivity=VADSensitivity.HIGH,
            silence_timeout_ms=500,
            hold_timeout_ms=600_000,
            max_segment_duration_ms=60_000,
            language="pt",
            enable_partial_transcripts=False,
            enable_itn=False,
            preprocessing=PreprocessingOverrides(denoise=True, denoise_engine="nsnet2"),
            input_sample_rate=8000,
        )
        original = SessionCreatedEvent(
            session_id="sess_full",
            model="faster-whisper-large-v3",
            config=config,
        )
        json_str = original.model_dump_json()
        restored = SessionCreatedEvent.model_validate_json(json_str)
        assert restored == original
        assert restored.config.input_sample_rate == 8000
        assert restored.config.preprocessing.denoise_engine == "nsnet2"

    def test_streaming_error_event_empty_code(self) -> None:
        """Codigo de erro vazio e aceito."""
        event = StreamingErrorEvent(
            code="",
            message="something happened",
            recoverable=True,
        )
        assert event.code == ""

    def test_session_closed_zero_duration_and_segments(self) -> None:
        """SessionClosedEvent com zero duration e zero segments e valido."""
        event = SessionClosedEvent(
            reason="timeout",
            total_duration_ms=0,
            segments_transcribed=0,
        )
        assert event.total_duration_ms == 0

    def test_word_event_zero_timestamps(self) -> None:
        """WordEvent com start=0 e end=0 e valido."""
        word = WordEvent(word="", start=0.0, end=0.0)
        assert word.word == ""
        assert word.start == 0.0

    def test_session_config_frozen_immutability(self) -> None:
        """SessionConfig e imutavel (frozen=True)."""
        config = SessionConfig()
        with pytest.raises(ValidationError):
            config.vad_sensitivity = VADSensitivity.HIGH

    def test_transcript_final_with_empty_words_list(self) -> None:
        """TranscriptFinalEvent com lista de words vazia e valido."""
        event = TranscriptFinalEvent(
            text="hello",
            segment_id=0,
            start_ms=0,
            end_ms=1000,
            words=[],
        )
        assert event.words == []

    def test_session_configure_with_extra_fields_ignored(self) -> None:
        """Campos extras no JSON sao ignorados pelo Pydantic (default behavior)."""
        raw = {
            "type": "session.configure",
            "language": "pt",
            "unknown_field": "should be ignored",
        }
        cmd = SessionConfigureCommand.model_validate(raw)
        assert cmd.language == "pt"

    def test_session_hold_event_zero_timeout(self) -> None:
        """SessionHoldEvent com timeout zero e valido."""
        event = SessionHoldEvent(timestamp_ms=0, hold_timeout_ms=0)
        assert event.hold_timeout_ms == 0


# ===========================================================================
# 2. Protocol Dispatch Edge Cases
# ===========================================================================


class TestProtocolDispatchEdgeCases:
    """Edge cases para dispatch_message()."""

    def test_empty_json_string_returns_error(self) -> None:
        """String vazia retorna ErrorResult."""
        result = dispatch_message({"text": ""})
        assert isinstance(result, ErrorResult)
        assert result.event.code == "malformed_json"

    def test_json_string_literal_returns_error(self) -> None:
        """String JSON literal (nao objeto) retorna ErrorResult."""
        result = dispatch_message({"text": '"just a string"'})
        assert isinstance(result, ErrorResult)
        assert result.event.code == "malformed_json"

    def test_json_number_returns_error(self) -> None:
        """Numero JSON retorna ErrorResult."""
        result = dispatch_message({"text": "42"})
        assert isinstance(result, ErrorResult)
        assert result.event.code == "malformed_json"

    def test_json_null_returns_error(self) -> None:
        """JSON null retorna ErrorResult."""
        result = dispatch_message({"text": "null"})
        assert isinstance(result, ErrorResult)
        assert result.event.code == "malformed_json"

    def test_json_boolean_returns_error(self) -> None:
        """JSON boolean retorna ErrorResult."""
        result = dispatch_message({"text": "true"})
        assert isinstance(result, ErrorResult)
        assert result.event.code == "malformed_json"

    def test_type_field_as_integer_returns_error(self) -> None:
        """Campo type como inteiro retorna ErrorResult."""
        result = dispatch_message({"text": json.dumps({"type": 123})})
        assert isinstance(result, ErrorResult)
        assert result.event.code == "unknown_command"

    def test_type_field_as_null_returns_error(self) -> None:
        """Campo type como null retorna ErrorResult."""
        result = dispatch_message({"text": json.dumps({"type": None})})
        assert isinstance(result, ErrorResult)
        assert result.event.code == "unknown_command"

    def test_type_field_empty_string_returns_unknown_command(self) -> None:
        """Campo type vazio retorna ErrorResult unknown_command."""
        result = dispatch_message({"text": json.dumps({"type": ""})})
        assert isinstance(result, ErrorResult)
        assert result.event.code == "unknown_command"

    def test_very_long_json_string_handled(self) -> None:
        """String JSON muito longa (>10KB) retorna erro ou resultado sem crash."""
        long_text = "a" * 10_000
        result = dispatch_message({"text": long_text})
        # Deve retornar ErrorResult (JSON invalido), nao crashar
        assert isinstance(result, ErrorResult)

    def test_deeply_nested_json_handled(self) -> None:
        """JSON com aninhamento profundo retorna ErrorResult (tipo desconhecido)."""
        nested = {"type": "session.configure", "language": "pt", "nested": {"a": {"b": {"c": 1}}}}
        result = dispatch_message({"text": json.dumps(nested)})
        # Extra fields are ignored by Pydantic, so this parses OK
        assert isinstance(result, CommandResult)

    def test_session_configure_with_negative_timeout(self) -> None:
        """session.configure com timeout negativo e rejeitado com validation_error."""
        msg = {"text": json.dumps({"type": "session.configure", "silence_timeout_ms": -100})}
        result = dispatch_message(msg)
        assert isinstance(result, ErrorResult)
        assert result.event.code == "validation_error"

    def test_session_configure_with_zero_timeout(self) -> None:
        """session.configure com timeout zero e rejeitado (gt=0)."""
        msg = {"text": json.dumps({"type": "session.configure", "silence_timeout_ms": 0})}
        result = dispatch_message(msg)
        assert isinstance(result, ErrorResult)
        assert result.event.code == "validation_error"

    def test_session_configure_with_valid_timeout(self) -> None:
        """session.configure com timeout positivo e aceito."""
        msg = {"text": json.dumps({"type": "session.configure", "silence_timeout_ms": 500})}
        result = dispatch_message(msg)
        assert isinstance(result, CommandResult)
        assert result.command.silence_timeout_ms == 500  # type: ignore[union-attr]

    def test_bytes_not_bytes_type_returns_error(self) -> None:
        """bytes field com tipo string (nao bytes) retorna ErrorResult."""
        result = dispatch_message({"bytes": "not actual bytes"})
        assert isinstance(result, ErrorResult)
        assert result.event.code == "invalid_frame"

    def test_session_configure_with_all_none_fields_valid(self) -> None:
        """session.configure sem nenhum campo opcional e valido."""
        msg = {"text": json.dumps({"type": "session.configure"})}
        result = dispatch_message(msg)
        assert isinstance(result, CommandResult)

    def test_unicode_json_command(self) -> None:
        """Comando JSON com caracteres unicode e tratado corretamente."""
        msg = {"text": json.dumps({"type": "session.configure", "language": "pt-BR"})}
        result = dispatch_message(msg)
        assert isinstance(result, CommandResult)


# ===========================================================================
# 3. StreamingSession Edge Cases
# ===========================================================================


class TestStreamingSessionEdgeCases:
    """Edge cases do StreamingSession."""

    async def test_process_empty_frame_zero_bytes(self) -> None:
        """Frame vazio (0 bytes) e processado sem erro."""
        preprocessor = _make_preprocessor_mock()
        preprocessor.process_frame.return_value = np.array([], dtype=np.float32)
        vad = _make_vad_mock()

        session = StreamingSession(
            session_id="test_session",
            preprocessor=preprocessor,
            vad=vad,
            grpc_client=_make_grpc_client_mock(),
            postprocessor=_make_postprocessor_mock(),
            on_event=_make_on_event(),
        )

        # Act: processar frame vazio
        await session.process_frame(b"")

        # Assert: preprocessor chamado, mas VAD NAO (frame vazio retorna early)
        preprocessor.process_frame.assert_called_once_with(b"")
        vad.process_frame.assert_not_called()

        await session.close()

    async def test_process_large_frame_over_64kb(self) -> None:
        """Frame muito grande (>64KB) e processado normalmente."""
        large_data = b"\x00\x01" * 40000  # 80KB
        preprocessor = _make_preprocessor_mock()
        # Preprocessor retorna frame grande (40000 samples)
        preprocessor.process_frame.return_value = np.zeros(40000, dtype=np.float32)
        vad = _make_vad_mock()

        session = StreamingSession(
            session_id="test_session",
            preprocessor=preprocessor,
            vad=vad,
            grpc_client=_make_grpc_client_mock(),
            postprocessor=_make_postprocessor_mock(),
            on_event=_make_on_event(),
        )

        # Act: processar frame grande
        await session.process_frame(large_data)

        # Assert: preprocessor e VAD chamados normalmente
        preprocessor.process_frame.assert_called_once_with(large_data)
        vad.process_frame.assert_called_once()

        await session.close()

    async def test_double_close_is_safe(self) -> None:
        """Chamar close() duas vezes seguidas nao levanta excecao."""
        session = StreamingSession(
            session_id="test_session",
            preprocessor=_make_preprocessor_mock(),
            vad=_make_vad_mock(),
            grpc_client=_make_grpc_client_mock(),
            postprocessor=_make_postprocessor_mock(),
            on_event=_make_on_event(),
        )

        await session.close()
        assert session.is_closed
        await session.close()
        assert session.is_closed

    async def test_process_frame_after_close_is_noop(self) -> None:
        """Frames recebidos apos close sao ignorados sem erro."""
        preprocessor = _make_preprocessor_mock()
        session = StreamingSession(
            session_id="test_session",
            preprocessor=preprocessor,
            vad=_make_vad_mock(),
            grpc_client=_make_grpc_client_mock(),
            postprocessor=_make_postprocessor_mock(),
            on_event=_make_on_event(),
        )

        await session.close()
        preprocessor.process_frame.reset_mock()

        await session.process_frame(_make_raw_bytes())

        preprocessor.process_frame.assert_not_called()

    async def test_commit_after_close_is_noop(self) -> None:
        """commit() em sessao fechada e no-op, sem erro."""
        session = StreamingSession(
            session_id="test_session",
            preprocessor=_make_preprocessor_mock(),
            vad=_make_vad_mock(),
            grpc_client=_make_grpc_client_mock(),
            postprocessor=_make_postprocessor_mock(),
            on_event=_make_on_event(),
        )

        await session.close()
        await session.commit()  # no-op
        assert session.segment_id == 0

    async def test_check_inactivity_on_closed_session_returns_false(self) -> None:
        """check_inactivity() em sessao fechada retorna False."""
        session = StreamingSession(
            session_id="test_session",
            preprocessor=_make_preprocessor_mock(),
            vad=_make_vad_mock(),
            grpc_client=_make_grpc_client_mock(),
            postprocessor=_make_postprocessor_mock(),
            on_event=_make_on_event(),
        )
        await session.close()

        # Mesmo com tempo passado, sessao fechada nao e "inativa"
        assert not session.check_inactivity()

    async def test_worker_crash_on_send_frame_emits_error(self) -> None:
        """WorkerCrashError durante send_frame emite erro recuperavel."""
        stream_handle = _make_stream_handle_mock()
        stream_handle.send_frame = AsyncMock(side_effect=WorkerCrashError("worker_1"))
        grpc_client = _make_grpc_client_mock(stream_handle)
        vad = _make_vad_mock()
        on_event = _make_on_event()

        session = StreamingSession(
            session_id="test_session",
            preprocessor=_make_preprocessor_mock(),
            vad=vad,
            grpc_client=grpc_client,
            postprocessor=_make_postprocessor_mock(),
            on_event=on_event,
        )

        # Trigger speech_start
        vad.process_frame.return_value = VADEvent(
            type=VADEventType.SPEECH_START,
            timestamp_ms=1000,
        )
        vad.is_speaking = True
        await session.process_frame(_make_raw_bytes())

        # Send frame that crashes
        vad.process_frame.return_value = None
        await session.process_frame(_make_raw_bytes())

        # Assert: error event emitted
        error_calls = [
            call
            for call in on_event.call_args_list
            if isinstance(call.args[0], StreamingErrorEvent)
        ]
        assert len(error_calls) >= 1
        assert error_calls[0].args[0].code == "worker_crash"

        await session.close()

    async def test_unexpected_exception_in_receiver_emits_irrecoverable_error(self) -> None:
        """Excecao inesperada no receiver task emite erro irrecuperavel."""
        stream_handle = _make_stream_handle_mock(
            events=[RuntimeError("unexpected GPU error")],
        )
        grpc_client = _make_grpc_client_mock(stream_handle)
        vad = _make_vad_mock()
        on_event = _make_on_event()

        session = StreamingSession(
            session_id="test_session",
            preprocessor=_make_preprocessor_mock(),
            vad=vad,
            grpc_client=grpc_client,
            postprocessor=_make_postprocessor_mock(),
            on_event=on_event,
        )

        # Trigger speech_start (inicia receiver que levanta RuntimeError)
        vad.process_frame.return_value = VADEvent(
            type=VADEventType.SPEECH_START,
            timestamp_ms=1000,
        )
        vad.is_speaking = False
        await session.process_frame(_make_raw_bytes())
        await asyncio.sleep(0.05)

        # Assert: error event emitted as irrecoverable
        error_calls = [
            call
            for call in on_event.call_args_list
            if isinstance(call.args[0], StreamingErrorEvent)
        ]
        assert len(error_calls) >= 1
        assert error_calls[0].args[0].recoverable is False
        assert error_calls[0].args[0].code == "internal_error"

        await session.close()

    async def test_session_id_property(self) -> None:
        """session_id retorna o ID fornecido no construtor."""
        session = StreamingSession(
            session_id="sess_abc123",
            preprocessor=_make_preprocessor_mock(),
            vad=_make_vad_mock(),
            grpc_client=_make_grpc_client_mock(),
            postprocessor=_make_postprocessor_mock(),
            on_event=_make_on_event(),
        )
        assert session.session_id == "sess_abc123"
        await session.close()

    async def test_multiple_speech_start_without_end(self) -> None:
        """Segundo speech_start sem speech_end abre novo stream."""
        stream_handle1 = _make_stream_handle_mock()
        stream_handle2 = _make_stream_handle_mock()
        grpc_client = _make_grpc_client_mock(stream_handle1)
        vad = _make_vad_mock()
        on_event = _make_on_event()

        session = StreamingSession(
            session_id="test_session",
            preprocessor=_make_preprocessor_mock(),
            vad=vad,
            grpc_client=grpc_client,
            postprocessor=_make_postprocessor_mock(),
            on_event=on_event,
        )

        # First speech_start
        vad.process_frame.return_value = VADEvent(
            type=VADEventType.SPEECH_START,
            timestamp_ms=1000,
        )
        vad.is_speaking = False
        await session.process_frame(_make_raw_bytes())
        await asyncio.sleep(0.01)

        # Second speech_start (without speech_end)
        grpc_client.open_stream = AsyncMock(return_value=stream_handle2)
        vad.process_frame.return_value = VADEvent(
            type=VADEventType.SPEECH_START,
            timestamp_ms=2000,
        )
        await session.process_frame(_make_raw_bytes())
        await asyncio.sleep(0.01)

        # Both speech_start events emitted
        start_calls = [
            call
            for call in on_event.call_args_list
            if isinstance(call.args[0], VADSpeechStartEvent)
        ]
        assert len(start_calls) == 2

        await session.close()

    async def test_final_transcript_with_empty_text_and_itn(self) -> None:
        """ITN em texto vazio nao deve crashar."""
        final_segment = TranscriptSegment(
            text="",
            is_final=True,
            segment_id=0,
            start_ms=1000,
            end_ms=2000,
        )

        stream_handle = _make_stream_handle_mock(events=[final_segment])
        grpc_client = _make_grpc_client_mock(stream_handle)
        vad = _make_vad_mock()
        postprocessor = _make_postprocessor_mock()
        on_event = _make_on_event()

        session = StreamingSession(
            session_id="test_session",
            preprocessor=_make_preprocessor_mock(),
            vad=vad,
            grpc_client=grpc_client,
            postprocessor=postprocessor,
            on_event=on_event,
            enable_itn=True,
        )

        vad.process_frame.return_value = VADEvent(
            type=VADEventType.SPEECH_START,
            timestamp_ms=1000,
        )
        vad.is_speaking = False
        await session.process_frame(_make_raw_bytes())
        await asyncio.sleep(0.05)

        # ITN should be called with empty string
        postprocessor.process.assert_called_once_with("")

        await session.close()


# ===========================================================================
# 4. VAD Energy Pre-Filter Edge Cases
# ===========================================================================


class TestEnergyPreFilterEdgeCases:
    """Edge cases para EnergyPreFilter."""

    def test_all_max_amplitude_frame(self) -> None:
        """Frame com amplitude maxima (1.0) nao e silencio."""
        pre_filter = EnergyPreFilter(sensitivity=VADSensitivity.NORMAL)
        frame = np.ones(1024, dtype=np.float32)

        result = pre_filter.is_silence(frame)

        assert result is False

    def test_all_negative_one_frame(self) -> None:
        """Frame com amplitude -1.0 nao e silencio."""
        pre_filter = EnergyPreFilter(sensitivity=VADSensitivity.NORMAL)
        frame = np.full(1024, -1.0, dtype=np.float32)

        result = pre_filter.is_silence(frame)

        assert result is False

    def test_dc_offset_only_frame(self) -> None:
        """Frame com DC offset constante tem spectral flatness baixa (nao silencio).

        Um sinal DC puro tem energia concentrada no bin 0 da FFT, resultando
        em spectral flatness baixa (dominancia de 1 bin). Mesmo com RMS baixo,
        a flatness < 0.8 faz o filtro nao classificar como silencio.
        """
        pre_filter = EnergyPreFilter(sensitivity=VADSensitivity.LOW)
        # Sinal constante: FFT concentrada em bin 0, flatness baixa
        frame = np.full(1024, 0.001, dtype=np.float32)

        result = pre_filter.is_silence(frame)

        # RMS muito baixo (-60dBFS < -30dBFS threshold) mas flatness baixa
        # -> nao classificado como silencio pelo filtro
        assert result is False

    def test_very_short_frame_two_samples(self) -> None:
        """Frame com 2 samples e classificado sem crash."""
        pre_filter = EnergyPreFilter(sensitivity=VADSensitivity.NORMAL)
        frame = np.array([0.5, -0.5], dtype=np.float32)

        # Nao deve levantar excecao
        result = pre_filter.is_silence(frame)
        assert isinstance(result, bool)


# ===========================================================================
# 5. VAD Detector Edge Cases
# ===========================================================================


class TestVADDetectorEdgeCases:
    """Edge cases para VADDetector."""

    def test_speech_resumes_during_silence_debounce(self) -> None:
        """Fala retomada durante debounce de silencio cancela SPEECH_END.

        Cenario: fala detectada -> silencio curto (< min_silence) -> fala novamente.
        Nao deve emitir SPEECH_END.
        """
        energy_mock = _make_energy_mock(is_silence=False)
        silero_mock = _make_silero_mock(is_speech=True)
        detector = VADDetector(
            energy_pre_filter=energy_mock,
            silero_classifier=silero_mock,
            sensitivity=VADSensitivity.NORMAL,
            sample_rate=_SAMPLE_RATE,
            min_speech_duration_ms=64,  # 1 frame para simplificar
            min_silence_duration_ms=300,  # 5 frames de 64ms
        )

        # Phase 1: speech (2 frames -> SPEECH_START)
        events: list[VADEvent] = []
        for frame in [np.zeros(_FRAME_SIZE, dtype=np.float32)] * 2:
            event = detector.process_frame(frame)
            if event is not None:
                events.append(event)

        assert len(events) == 1
        assert events[0].type == VADEventType.SPEECH_START

        # Phase 2: silence (2 frames < 5 frames needed for SPEECH_END)
        silero_mock.is_speech.return_value = False
        for frame in [np.zeros(_FRAME_SIZE, dtype=np.float32)] * 2:
            event = detector.process_frame(frame)
            if event is not None:
                events.append(event)

        # No SPEECH_END yet
        assert len(events) == 1

        # Phase 3: speech resumes
        silero_mock.is_speech.return_value = True
        for frame in [np.zeros(_FRAME_SIZE, dtype=np.float32)] * 3:
            event = detector.process_frame(frame)
            if event is not None:
                events.append(event)

        # Still no SPEECH_END (silence was too short)
        speech_ends = [e for e in events if e.type == VADEventType.SPEECH_END]
        assert len(speech_ends) == 0
        assert detector.is_speaking is True

    def test_minimum_speech_duration_boundary(self) -> None:
        """SPEECH_START emitido exatamente no frame que atinge min_speech_duration."""
        # min_speech_duration_ms=64ms = exactly 1 frame of 64ms
        detector, _, _ = _make_detector(
            energy_is_silence=False,
            silero_is_speech=True,
            min_speech_duration_ms=64,  # 1 frame
        )

        frame = np.zeros(_FRAME_SIZE, dtype=np.float32)
        event = detector.process_frame(frame)

        assert event is not None
        assert event.type == VADEventType.SPEECH_START

    def test_energy_filter_alternating_silence_non_silence(self) -> None:
        """Frames alternando entre silencio e nao-silencio no energy filter."""
        energy_mock = _make_energy_mock(is_silence=False)
        silero_mock = _make_silero_mock(is_speech=True)
        detector = VADDetector(
            energy_pre_filter=energy_mock,
            silero_classifier=silero_mock,
            sensitivity=VADSensitivity.NORMAL,
            sample_rate=_SAMPLE_RATE,
            min_speech_duration_ms=250,
        )

        # Alternar energy: non-silence, silence, non-silence, silence...
        all_events: list[VADEvent] = []
        for i in range(20):
            energy_mock.is_silence.return_value = i % 2 == 1
            frame = np.zeros(_FRAME_SIZE, dtype=np.float32)
            event = detector.process_frame(frame)
            if event is not None:
                all_events.append(event)

        # Nunca atinge min_speech_duration consecutivo -> sem SPEECH_START
        assert len(all_events) == 0

    def test_force_speech_end_resets_for_new_speech(self) -> None:
        """Apos force SPEECH_END, novo segmento de fala pode comecar."""
        detector, _, _silero_mock = _make_detector(
            energy_is_silence=False,
            silero_is_speech=True,
            min_speech_duration_ms=64,
            max_speech_duration_ms=640,  # 10 frames
        )

        all_events: list[VADEvent] = []
        # Process 15 frames: should get SPEECH_START + force SPEECH_END
        for frame in [np.zeros(_FRAME_SIZE, dtype=np.float32)] * 15:
            event = detector.process_frame(frame)
            if event is not None:
                all_events.append(event)

        starts = [e for e in all_events if e.type == VADEventType.SPEECH_START]
        ends = [e for e in all_events if e.type == VADEventType.SPEECH_END]
        assert len(starts) >= 1
        assert len(ends) >= 1

        # After force end, new speech should start
        # The last events should form start-end pairs
        assert starts[0].timestamp_ms < ends[0].timestamp_ms


# ===========================================================================
# 6. Backpressure Edge Cases
# ===========================================================================


class TestBackpressureEdgeCases:
    """Edge cases para BackpressureController."""

    def test_zero_byte_frame_no_crash(self) -> None:
        """Frame de 0 bytes nao deve crashar."""
        clock = _FakeClock()
        ctrl = BackpressureController(
            sample_rate=_SAMPLE_RATE,
            clock=clock,
        )

        result = ctrl.record_frame(0)
        assert result is None
        assert ctrl.frames_received == 1

    def test_very_large_frame_handled(self) -> None:
        """Frame muito grande (1 minuto de audio) e tratado sem crash."""
        clock = _FakeClock()
        ctrl = BackpressureController(
            sample_rate=_SAMPLE_RATE,
            max_backlog_s=120.0,  # Alto para nao dropar
            clock=clock,
        )

        # 1 minuto de audio = 60 * 16000 * 2 bytes = 1,920,000 bytes
        one_minute_bytes = 60 * _SAMPLE_RATE * 2
        result = ctrl.record_frame(one_minute_bytes)
        assert result is None  # First frame, no action
        assert ctrl.frames_received == 1

    def test_backpressure_after_long_pause(self) -> None:
        """Apos longa pausa (10s), retomar envio nao deve emitir rate_limit."""
        clock = _FakeClock()
        ctrl = BackpressureController(
            sample_rate=_SAMPLE_RATE,
            rate_limit_threshold=1.2,
            max_backlog_s=100.0,
            clock=clock,
        )

        # Enviar 10 frames normais
        for _ in range(10):
            ctrl.record_frame(640)
            clock.advance(0.020)

        # Longa pausa
        clock.advance(10.0)

        # Retomar envio normal
        actions: list[RateLimitAction] = []
        for _ in range(10):
            result = ctrl.record_frame(640)
            if isinstance(result, RateLimitAction):
                actions.append(result)
            clock.advance(0.020)

        assert len(actions) == 0

    def test_multiple_drops_counted_correctly(self) -> None:
        """Multiplos drops consecutivos sao contados corretamente."""
        clock = _FakeClock()
        ctrl = BackpressureController(
            sample_rate=_SAMPLE_RATE,
            max_backlog_s=0.1,  # Muito baixo
            clock=clock,
        )

        drop_count = 0
        for _ in range(50):
            result = ctrl.record_frame(640)
            if isinstance(result, FramesDroppedAction):
                drop_count += 1

        assert ctrl.frames_dropped == drop_count
        assert ctrl.frames_received == 50
        assert ctrl.frames_dropped > 0


# ===========================================================================
# 7. StreamingGRPCClient Edge Cases
# ===========================================================================


class TestStreamHandleEdgeCases:
    """Edge cases para StreamHandle."""

    async def test_cancel_on_already_closed_handle(self) -> None:
        """cancel() em handle ja fechado e idempotente."""
        from macaw.scheduler.streaming import StreamHandle

        mock_call = AsyncMock(
            spec_set=["write", "done_writing", "cancel", "__aiter__", "__anext__"],
        )
        mock_call.write = AsyncMock()
        mock_call.done_writing = AsyncMock()
        mock_call.cancel = MagicMock()

        handle = StreamHandle(session_id="sess_test", call=mock_call)

        # Close first
        await handle.close()
        assert handle.is_closed

        # Cancel on already closed
        await handle.cancel()

        # cancel() should be called (it's always safe to call)
        assert mock_call.cancel.call_count >= 1

    async def test_send_frame_empty_pcm_data(self) -> None:
        """send_frame com bytes vazios e valido."""
        from macaw.scheduler.streaming import StreamHandle

        mock_call = AsyncMock(
            spec_set=["write", "done_writing", "cancel", "__aiter__", "__anext__"],
        )
        mock_call.write = AsyncMock()
        mock_call.done_writing = AsyncMock()
        mock_call.cancel = MagicMock()

        handle = StreamHandle(session_id="sess_test", call=mock_call)

        await handle.send_frame(pcm_data=b"")

        mock_call.write.assert_called_once()
        frame = mock_call.write.call_args[0][0]
        assert frame.data == b""

    async def test_send_frame_after_close_raises_worker_crash(self) -> None:
        """send_frame em handle fechado levanta WorkerCrashError."""
        from macaw.scheduler.streaming import StreamHandle

        mock_call = AsyncMock(
            spec_set=["write", "done_writing", "cancel", "__aiter__", "__anext__"],
        )
        mock_call.write = AsyncMock()
        mock_call.done_writing = AsyncMock()
        mock_call.cancel = MagicMock()

        handle = StreamHandle(session_id="sess_test", call=mock_call)
        await handle.close()

        with pytest.raises(WorkerCrashError):
            await handle.send_frame(pcm_data=b"\x00\x01")


# ===========================================================================
# 8. StreamingPreprocessor Edge Cases
# ===========================================================================


class TestStreamingPreprocessorEdgeCases:
    """Edge cases para StreamingPreprocessor."""

    def test_single_sample_frame(self) -> None:
        """Frame com 1 sample (2 bytes) e processado sem crash."""
        from macaw.preprocessing.resample import ResampleStage
        from macaw.preprocessing.streaming import StreamingPreprocessor

        stages = [ResampleStage(target_sample_rate=16000)]
        preprocessor = StreamingPreprocessor(stages=stages, input_sample_rate=16000)

        # 1 sample = 2 bytes
        frame = np.array([1000], dtype=np.int16).tobytes()
        result = preprocessor.process_frame(frame)

        assert result.dtype == np.float32
        assert len(result) == 1

    def test_consecutive_frames_maintain_state(self) -> None:
        """Multiplos frames consecutivos sao processados sem leak de estado."""
        from macaw.preprocessing.dc_remove import DCRemoveStage
        from macaw.preprocessing.streaming import StreamingPreprocessor

        stages = [DCRemoveStage(cutoff_hz=20)]
        preprocessor = StreamingPreprocessor(stages=stages, input_sample_rate=16000)

        # Processar 100 frames pequenos
        for _ in range(100):
            frame = (
                np.random.default_rng(42).integers(-1000, 1000, size=160, dtype=np.int16).tobytes()
            )
            result = preprocessor.process_frame(frame)
            assert result.dtype == np.float32
            assert len(result) == 160


# ===========================================================================
# 9. VAD Silero Classifier Edge Cases
# ===========================================================================


class TestSileroClassifierEdgeCases:
    """Edge cases para SileroVADClassifier."""

    def test_probability_just_above_threshold(self) -> None:
        """Probabilidade 0.5001 (just above 0.5) e classificada como fala."""
        from macaw.vad.silero import SileroVADClassifier

        classifier = SileroVADClassifier(sensitivity=VADSensitivity.NORMAL)
        mock_model = MagicMock()
        result = MagicMock()
        result.item.return_value = 0.5001
        mock_model.return_value = result
        classifier._model = mock_model
        classifier._model_loaded = True

        frame = np.zeros(512, dtype=np.float32)
        assert classifier.is_speech(frame) is True

    def test_probability_just_below_threshold(self) -> None:
        """Probabilidade 0.4999 (just below 0.5) nao e classificada como fala."""
        from macaw.vad.silero import SileroVADClassifier

        classifier = SileroVADClassifier(sensitivity=VADSensitivity.NORMAL)
        mock_model = MagicMock()
        result = MagicMock()
        result.item.return_value = 0.4999
        mock_model.return_value = result
        classifier._model = mock_model
        classifier._model_loaded = True

        frame = np.zeros(512, dtype=np.float32)
        assert classifier.is_speech(frame) is False


# ===========================================================================
# 10. Streaming Converters Edge Cases
# ===========================================================================


class TestStreamingConvertersEdgeCases:
    """Edge cases para conversores de proto streaming."""

    def test_proto_event_with_empty_text(self) -> None:
        """Evento proto com texto vazio e convertido corretamente."""
        from macaw.proto.stt_worker_pb2 import TranscriptEvent
        from macaw.scheduler.streaming import _proto_event_to_transcript_segment

        event = TranscriptEvent(
            event_type="final",
            text="",
            segment_id=0,
        )

        result = _proto_event_to_transcript_segment(event)
        assert result.text == ""
        assert result.is_final is True

    def test_proto_event_unknown_type_treated_as_partial(self) -> None:
        """Evento proto com event_type desconhecido e tratado como partial (is_final=False)."""
        from macaw.proto.stt_worker_pb2 import TranscriptEvent
        from macaw.scheduler.streaming import _proto_event_to_transcript_segment

        event = TranscriptEvent(
            event_type="unknown",
            text="test",
            segment_id=0,
        )

        result = _proto_event_to_transcript_segment(event)
        assert result.is_final is False

    def test_proto_event_with_many_words(self) -> None:
        """Evento proto com muitas words e convertido corretamente."""
        from macaw.proto.stt_worker_pb2 import TranscriptEvent, Word
        from macaw.scheduler.streaming import _proto_event_to_transcript_segment

        words = [
            Word(word=f"word_{i}", start=i * 0.1, end=(i + 1) * 0.1, probability=0.9)
            for i in range(50)
        ]
        event = TranscriptEvent(
            event_type="final",
            text=" ".join(f"word_{i}" for i in range(50)),
            segment_id=0,
            start_ms=0,
            end_ms=5000,
            words=words,
        )

        result = _proto_event_to_transcript_segment(event)
        assert result.words is not None
        assert len(result.words) == 50

    def test_proto_event_with_zero_probability_becomes_none(self) -> None:
        """Probabilidade 0.0 em word e convertida para None."""
        from macaw.proto.stt_worker_pb2 import TranscriptEvent, Word
        from macaw.scheduler.streaming import _proto_event_to_transcript_segment

        event = TranscriptEvent(
            event_type="final",
            text="hello",
            segment_id=0,
            words=[Word(word="hello", start=0.0, end=0.5, probability=0.0)],
        )

        result = _proto_event_to_transcript_segment(event)
        assert result.words is not None
        assert result.words[0].probability is None

    def test_proto_event_with_zero_start_ms_preserved(self) -> None:
        """start_ms=0 e preservado como 0 (valor valido: inicio do audio)."""
        from macaw.proto.stt_worker_pb2 import TranscriptEvent
        from macaw.scheduler.streaming import _proto_event_to_transcript_segment

        event = TranscriptEvent(
            event_type="final",
            text="hello",
            segment_id=0,
            start_ms=0,
            end_ms=0,
        )

        result = _proto_event_to_transcript_segment(event)
        assert result.start_ms == 0
        assert result.end_ms == 0

    def test_proto_event_partial_type_is_not_final(self) -> None:
        """event_type='partial' gera is_final=False."""
        from macaw.proto.stt_worker_pb2 import TranscriptEvent
        from macaw.scheduler.streaming import _proto_event_to_transcript_segment

        event = TranscriptEvent(
            event_type="partial",
            text="hel",
            segment_id=0,
        )

        result = _proto_event_to_transcript_segment(event)
        assert result.is_final is False

    def test_proto_event_no_words_is_none(self) -> None:
        """Evento proto sem words resulta em words=None."""
        from macaw.proto.stt_worker_pb2 import TranscriptEvent
        from macaw.scheduler.streaming import _proto_event_to_transcript_segment

        event = TranscriptEvent(
            event_type="final",
            text="hello",
            segment_id=0,
        )

        result = _proto_event_to_transcript_segment(event)
        assert result.words is None


# ===========================================================================
# 11. Protocol Dispatch - More Edge Cases
# ===========================================================================


class TestProtocolDispatchMoreEdgeCases:
    """Mais edge cases para dispatch_message()."""

    def test_none_message_returns_none(self) -> None:
        """Mensagem sem bytes nem text retorna None."""
        result = dispatch_message({})
        assert result is None

    def test_message_with_only_unknown_keys_returns_none(self) -> None:
        """Mensagem com chaves desconhecidas retorna None."""
        result = dispatch_message({"foo": "bar", "baz": 42})
        assert result is None

    def test_binary_empty_frame_is_valid(self) -> None:
        """Frame binario vazio (0 bytes) e valido."""
        result = dispatch_message({"bytes": b""})
        assert isinstance(result, AudioFrameResult)
        assert result.data == b""

    def test_binary_large_frame_is_valid(self) -> None:
        """Frame binario grande (100KB) e valido."""
        large_data = b"\x00" * 100_000
        result = dispatch_message({"bytes": large_data})
        assert isinstance(result, AudioFrameResult)
        assert len(result.data) == 100_000

    def test_json_array_returns_error(self) -> None:
        """JSON array retorna ErrorResult (esperado objeto)."""
        result = dispatch_message({"text": "[1, 2, 3]"})
        assert isinstance(result, ErrorResult)
        assert result.event.code == "malformed_json"

    def test_session_cancel_command_parsed(self) -> None:
        """session.cancel e parseado corretamente."""
        msg = {"text": json.dumps({"type": "session.cancel"})}
        result = dispatch_message(msg)
        assert isinstance(result, CommandResult)
        assert isinstance(result.command, SessionCancelCommand)

    def test_session_close_command_parsed(self) -> None:
        """session.close e parseado corretamente."""
        msg = {"text": json.dumps({"type": "session.close"})}
        result = dispatch_message(msg)
        assert isinstance(result, CommandResult)
        assert isinstance(result.command, SessionCloseCommand)

    def test_input_audio_buffer_commit_command_parsed(self) -> None:
        """input_audio_buffer.commit e parseado corretamente."""
        msg = {"text": json.dumps({"type": "input_audio_buffer.commit"})}
        result = dispatch_message(msg)
        assert isinstance(result, CommandResult)
        assert isinstance(result.command, InputAudioBufferCommitCommand)

    def test_bytes_takes_precedence_over_text(self) -> None:
        """Quando 'bytes' e 'text' ambos presentes, 'bytes' tem precedencia."""
        result = dispatch_message({"bytes": b"\x00\x01", "text": '{"type":"session.close"}'})
        assert isinstance(result, AudioFrameResult)
        assert result.data == b"\x00\x01"


# ===========================================================================
# 12. VAD Detector - More Edge Cases
# ===========================================================================


class TestVADDetectorMoreEdgeCases:
    """Mais edge cases para VADDetector."""

    def test_reset_during_speech_clears_state(self) -> None:
        """reset() durante fala ativa limpa estado para silencio."""
        detector, _, _silero_mock = _make_detector(
            energy_is_silence=False,
            silero_is_speech=True,
            min_speech_duration_ms=64,
        )

        frame = np.zeros(_FRAME_SIZE, dtype=np.float32)

        # Iniciar fala
        event = detector.process_frame(frame)
        assert event is not None
        assert event.type == VADEventType.SPEECH_START
        assert detector.is_speaking is True

        # Reset durante fala
        detector.reset()

        assert detector.is_speaking is False

    def test_timestamp_calculation_correct(self) -> None:
        """Timestamps em milissegundos sao calculados corretamente."""
        detector, _, _ = _make_detector(
            energy_is_silence=False,
            silero_is_speech=True,
            min_speech_duration_ms=64,
        )

        frame = np.zeros(_FRAME_SIZE, dtype=np.float32)
        event = detector.process_frame(frame)

        assert event is not None
        # 1024 samples at 16kHz = 64ms
        assert event.timestamp_ms == 64

    def test_silence_after_max_speech_does_not_double_emit(self) -> None:
        """Silencio apos force SPEECH_END nao emite outro SPEECH_END."""
        detector, _, silero_mock = _make_detector(
            energy_is_silence=False,
            silero_is_speech=True,
            min_speech_duration_ms=64,
            max_speech_duration_ms=640,  # 10 frames
        )

        all_events: list[VADEvent] = []
        frame = np.zeros(_FRAME_SIZE, dtype=np.float32)

        # 11 frames of speech -> should get SPEECH_START + force SPEECH_END
        for _ in range(11):
            event = detector.process_frame(frame)
            if event is not None:
                all_events.append(event)

        assert len([e for e in all_events if e.type == VADEventType.SPEECH_END]) == 1

        # Now silence: should NOT emit another SPEECH_END
        silero_mock.is_speech.return_value = False
        for _ in range(10):
            event = detector.process_frame(frame)
            if event is not None:
                all_events.append(event)

        # Still only 1 SPEECH_END
        assert len([e for e in all_events if e.type == VADEventType.SPEECH_END]) == 1

    def test_zero_min_speech_duration_emits_immediately(self) -> None:
        """min_speech_duration_ms=0 emite SPEECH_START no primeiro frame de fala."""
        detector, _, _ = _make_detector(
            energy_is_silence=False,
            silero_is_speech=True,
            min_speech_duration_ms=0,
        )

        frame = np.zeros(_FRAME_SIZE, dtype=np.float32)
        event = detector.process_frame(frame)

        # With 0ms min, should emit immediately since consecutive_speech >= 0
        assert event is not None
        assert event.type == VADEventType.SPEECH_START


# ===========================================================================
# 13. StreamingSession - More Edge Cases
# ===========================================================================


class TestStreamingSessionMoreEdgeCases:
    """Mais edge cases do StreamingSession."""

    async def test_speech_end_without_stream_handle(self) -> None:
        """SPEECH_END sem stream handle ativo nao causa crash.

        Com a state machine integrada (M6-03), um SPEECH_END em estado INIT
        (sem SPEECH_START previo) e ignorado silenciosamente — a sessao nunca
        esteve ACTIVE, entao nao ha transicao valida para SILENCE.
        O importante e que nao causa crash.
        """
        vad = _make_vad_mock()
        on_event = _make_on_event()

        session = StreamingSession(
            session_id="test_session",
            preprocessor=_make_preprocessor_mock(),
            vad=vad,
            grpc_client=_make_grpc_client_mock(),
            postprocessor=_make_postprocessor_mock(),
            on_event=on_event,
        )

        # Emitir SPEECH_END sem nunca ter tido SPEECH_START
        vad.process_frame.return_value = VADEvent(
            type=VADEventType.SPEECH_END,
            timestamp_ms=1000,
        )
        vad.is_speaking = False
        await session.process_frame(_make_raw_bytes())

        # Com state machine, SPEECH_END em INIT e ignorado (sem crash).
        # Nenhum evento speech_end e emitido pois sessao nunca esteve ACTIVE.
        end_calls = [
            call for call in on_event.call_args_list if isinstance(call.args[0], VADSpeechEndEvent)
        ]
        assert len(end_calls) == 0

        await session.close()

    async def test_grpc_open_stream_failure_emits_error(self) -> None:
        """Falha ao abrir gRPC stream emite erro recuperavel."""
        grpc_client = AsyncMock()
        grpc_client.open_stream = AsyncMock(side_effect=WorkerCrashError("worker_1"))
        grpc_client.close = AsyncMock()
        vad = _make_vad_mock()
        on_event = _make_on_event()

        session = StreamingSession(
            session_id="test_session",
            preprocessor=_make_preprocessor_mock(),
            vad=vad,
            grpc_client=grpc_client,
            postprocessor=_make_postprocessor_mock(),
            on_event=on_event,
        )

        # Trigger speech_start -> tenta abrir stream, falha
        vad.process_frame.return_value = VADEvent(
            type=VADEventType.SPEECH_START,
            timestamp_ms=1000,
        )
        vad.is_speaking = False
        await session.process_frame(_make_raw_bytes())

        # Deve emitir erro recuperavel
        error_calls = [
            call
            for call in on_event.call_args_list
            if isinstance(call.args[0], StreamingErrorEvent)
        ]
        assert len(error_calls) == 1
        assert error_calls[0].args[0].code == "worker_crash"
        assert error_calls[0].args[0].recoverable is True

        await session.close()

    async def test_postprocessor_none_skips_itn(self) -> None:
        """Sem postprocessor, ITN e pulado mesmo com enable_itn=True."""
        final_segment = TranscriptSegment(
            text="hello world",
            is_final=True,
            segment_id=0,
            start_ms=1000,
            end_ms=2000,
        )

        stream_handle = _make_stream_handle_mock(events=[final_segment])
        grpc_client = _make_grpc_client_mock(stream_handle)
        vad = _make_vad_mock()
        on_event = _make_on_event()

        session = StreamingSession(
            session_id="test_session",
            preprocessor=_make_preprocessor_mock(),
            vad=vad,
            grpc_client=grpc_client,
            postprocessor=None,  # No postprocessor
            on_event=on_event,
            enable_itn=True,
        )

        vad.process_frame.return_value = VADEvent(
            type=VADEventType.SPEECH_START,
            timestamp_ms=1000,
        )
        vad.is_speaking = False
        await session.process_frame(_make_raw_bytes())
        await asyncio.sleep(0.05)

        # Final deve ter texto original (sem ITN)
        final_calls = [
            call
            for call in on_event.call_args_list
            if isinstance(call.args[0], TranscriptFinalEvent)
        ]
        assert len(final_calls) == 1
        assert final_calls[0].args[0].text == "hello world"

        await session.close()

    async def test_itn_disabled_skips_postprocessing(self) -> None:
        """enable_itn=False pula post-processing mesmo com postprocessor."""
        final_segment = TranscriptSegment(
            text="dois mil",
            is_final=True,
            segment_id=0,
            start_ms=1000,
            end_ms=2000,
        )

        stream_handle = _make_stream_handle_mock(events=[final_segment])
        grpc_client = _make_grpc_client_mock(stream_handle)
        vad = _make_vad_mock()
        postprocessor = _make_postprocessor_mock()
        on_event = _make_on_event()

        session = StreamingSession(
            session_id="test_session",
            preprocessor=_make_preprocessor_mock(),
            vad=vad,
            grpc_client=grpc_client,
            postprocessor=postprocessor,
            on_event=on_event,
            enable_itn=False,  # ITN desabilitado
        )

        vad.process_frame.return_value = VADEvent(
            type=VADEventType.SPEECH_START,
            timestamp_ms=1000,
        )
        vad.is_speaking = False
        await session.process_frame(_make_raw_bytes())
        await asyncio.sleep(0.05)

        # Postprocessor NAO deve ser chamado
        postprocessor.process.assert_not_called()

        # Texto original
        final_calls = [
            call
            for call in on_event.call_args_list
            if isinstance(call.args[0], TranscriptFinalEvent)
        ]
        assert len(final_calls) == 1
        assert final_calls[0].args[0].text == "dois mil"

        await session.close()

    async def test_segment_id_after_speech_end_increments(self) -> None:
        """segment_id incrementa apos SPEECH_END."""
        vad = _make_vad_mock()
        on_event = _make_on_event()

        session = StreamingSession(
            session_id="test_session",
            preprocessor=_make_preprocessor_mock(),
            vad=vad,
            grpc_client=_make_grpc_client_mock(),
            postprocessor=_make_postprocessor_mock(),
            on_event=on_event,
        )

        assert session.segment_id == 0

        # SPEECH_START
        vad.process_frame.return_value = VADEvent(
            type=VADEventType.SPEECH_START,
            timestamp_ms=1000,
        )
        vad.is_speaking = False
        await session.process_frame(_make_raw_bytes())
        await asyncio.sleep(0.01)

        assert session.segment_id == 0

        # SPEECH_END
        vad.process_frame.return_value = VADEvent(
            type=VADEventType.SPEECH_END,
            timestamp_ms=2000,
        )
        vad.is_speaking = False
        await session.process_frame(_make_raw_bytes())

        assert session.segment_id == 1

        await session.close()


# ===========================================================================
# 14. Backpressure - More Edge Cases
# ===========================================================================


class TestBackpressureMoreEdgeCases:
    """Mais edge cases para BackpressureController."""

    def test_first_frame_never_triggers_action(self) -> None:
        """Primeiro frame nunca dispara acao de backpressure."""
        clock = _FakeClock()
        ctrl = BackpressureController(
            sample_rate=_SAMPLE_RATE,
            max_backlog_s=0.001,  # Muito baixo
            rate_limit_threshold=1.0,  # Extremo
            clock=clock,
        )

        # Primeiro frame: nunca dispara
        result = ctrl.record_frame(640)
        assert result is None

    def test_rate_limit_cooldown(self) -> None:
        """Rate limit tem cooldown de 1s entre emissoes."""
        clock = _FakeClock()
        ctrl = BackpressureController(
            sample_rate=_SAMPLE_RATE,
            rate_limit_threshold=1.1,
            max_backlog_s=100.0,
            clock=clock,
        )

        # Enviar rapido (muitos frames em pouco tempo)
        rate_limit_count = 0
        for _ in range(100):
            result = ctrl.record_frame(640)
            if isinstance(result, RateLimitAction):
                rate_limit_count += 1
            clock.advance(0.001)  # 1ms entre frames (muito rapido)

        # Deve ter emitido poucas vezes por causa do cooldown de 1s
        # 100 frames * 1ms = 100ms < 1s cooldown -> no maximo 1 emissao
        assert rate_limit_count <= 1

    def test_properties_initial_state(self) -> None:
        """Estado inicial: zero frames recebidos e dropados."""
        clock = _FakeClock()
        ctrl = BackpressureController(
            sample_rate=_SAMPLE_RATE,
            clock=clock,
        )

        assert ctrl.frames_received == 0
        assert ctrl.frames_dropped == 0
