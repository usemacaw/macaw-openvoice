"""Testes do VADDetector.

Valida que o detector orquestra EnergyPreFilter e SileroVADClassifier
corretamente, gerenciando debounce (min speech/silence duration),
max speech duration e emissao de eventos VAD.

Todos os testes usam mocks -- sem dependencia de torch/Silero real.
"""

from __future__ import annotations

from unittest.mock import Mock

import numpy as np

from macaw._types import VADSensitivity
from macaw.vad.detector import VADDetector, VADEvent, VADEventType

# Frame size padrao: 1024 samples a 16kHz = 64ms
_FRAME_SIZE = 1024
_SAMPLE_RATE = 16000
_FRAME_DURATION_MS = _FRAME_SIZE * 1000 // _SAMPLE_RATE  # 64ms


def _make_frames(n: int, frame_size: int = _FRAME_SIZE) -> list[np.ndarray]:
    """Gera N frames de zeros float32."""
    return [np.zeros(frame_size, dtype=np.float32) for _ in range(n)]


def _make_energy_mock(*, is_silence: bool = False) -> Mock:
    """Cria mock de EnergyPreFilter com retorno fixo."""
    mock = Mock()
    mock.is_silence.return_value = is_silence
    return mock


def _make_silero_mock(*, is_speech: bool = False) -> Mock:
    """Cria mock de SileroVADClassifier com retorno fixo."""
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
    """Cria VADDetector com mocks configurados. Retorna (detector, energy_mock, silero_mock)."""
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


def _process_n_frames(
    detector: VADDetector,
    n: int,
    frame_size: int = _FRAME_SIZE,
) -> list[VADEvent]:
    """Processa N frames e retorna lista de eventos emitidos (sem None)."""
    events = []
    for frame in _make_frames(n, frame_size):
        event = detector.process_frame(frame)
        if event is not None:
            events.append(event)
    return events


class TestSilenceFrames:
    """Frames classificados como silencio nao emitem eventos."""

    def test_all_silence_no_events(self) -> None:
        """Sequencia de frames de silencio nao emite nenhum evento."""
        # Arrange
        detector, _, _ = _make_detector(energy_is_silence=True)

        # Act
        events = _process_n_frames(detector, n=20)

        # Assert
        assert events == []

    def test_silence_via_energy_prefilter_skips_silero(self) -> None:
        """Quando EnergyPreFilter diz silencio, Silero NAO e chamado."""
        # Arrange
        detector, energy_mock, silero_mock = _make_detector(energy_is_silence=True)
        frames = _make_frames(5)

        # Act
        for frame in frames:
            detector.process_frame(frame)

        # Assert
        assert energy_mock.is_silence.call_count == 5
        silero_mock.is_speech.assert_not_called()


class TestSpeechStart:
    """Emissao de SPEECH_START apos debounce de fala."""

    def test_speech_start_after_min_duration(self) -> None:
        """SPEECH_START emitido apos min_speech_duration_ms de fala consecutiva."""
        # Arrange -- 250ms min speech, frames de 64ms -> ceil(250/64) = 4 frames
        detector, _, _ = _make_detector(
            energy_is_silence=False,
            silero_is_speech=True,
            min_speech_duration_ms=250,
        )

        # Act
        events = _process_n_frames(detector, n=4)

        # Assert
        assert len(events) == 1
        assert events[0].type == VADEventType.SPEECH_START
        assert events[0].timestamp_ms == 4 * _FRAME_DURATION_MS  # 256ms

    def test_short_speech_burst_no_event(self) -> None:
        """Burst de fala mais curto que min_speech_duration nao emite evento."""
        # Arrange -- 250ms min, mas so 3 frames (192ms)
        detector, _, _ = _make_detector(
            energy_is_silence=False,
            silero_is_speech=True,
            min_speech_duration_ms=250,
        )

        # Act -- 3 frames de fala, depois silencio
        events_speech: list[VADEvent] = []
        for frame in _make_frames(3):
            event = detector.process_frame(frame)
            if event is not None:
                events_speech.append(event)

        # Assert -- sem SPEECH_START (nao atingiu min_speech_duration)
        assert events_speech == []

    def test_is_speaking_true_after_speech_start(self) -> None:
        """is_speaking retorna True apos SPEECH_START ser emitido."""
        # Arrange
        detector, _, _ = _make_detector(
            energy_is_silence=False,
            silero_is_speech=True,
            min_speech_duration_ms=250,
        )
        assert detector.is_speaking is False

        # Act -- 4 frames de fala = 256ms > 250ms
        _process_n_frames(detector, n=4)

        # Assert
        assert detector.is_speaking is True


class TestSpeechEnd:
    """Emissao de SPEECH_END apos debounce de silencio."""

    def test_speech_end_after_min_silence_duration(self) -> None:
        """SPEECH_START seguido de SPEECH_END apos min_silence_duration_ms."""
        # Arrange
        energy_mock = _make_energy_mock(is_silence=False)
        silero_mock = _make_silero_mock(is_speech=True)
        detector = VADDetector(
            energy_pre_filter=energy_mock,
            silero_classifier=silero_mock,
            sensitivity=VADSensitivity.NORMAL,
            sample_rate=_SAMPLE_RATE,
            min_speech_duration_ms=250,
            min_silence_duration_ms=300,
        )

        # Act -- Phase 1: speech (4 frames = 256ms -> SPEECH_START)
        events_phase1 = _process_n_frames(detector, n=4)
        assert len(events_phase1) == 1
        assert events_phase1[0].type == VADEventType.SPEECH_START

        # Phase 2: silence (5 frames = 320ms -> SPEECH_END)
        silero_mock.is_speech.return_value = False
        events_phase2 = _process_n_frames(detector, n=5)

        # Assert
        assert len(events_phase2) == 1
        assert events_phase2[0].type == VADEventType.SPEECH_END

    def test_short_silence_during_speech_no_event(self) -> None:
        """Silencio mais curto que min_silence_duration nao emite SPEECH_END."""
        # Arrange
        energy_mock = _make_energy_mock(is_silence=False)
        silero_mock = _make_silero_mock(is_speech=True)
        detector = VADDetector(
            energy_pre_filter=energy_mock,
            silero_classifier=silero_mock,
            sensitivity=VADSensitivity.NORMAL,
            sample_rate=_SAMPLE_RATE,
            min_speech_duration_ms=250,
            min_silence_duration_ms=300,
        )

        # Act -- Phase 1: speech (4 frames -> SPEECH_START)
        _process_n_frames(detector, n=4)
        assert detector.is_speaking is True

        # Phase 2: short silence (3 frames = 192ms < 300ms) -- nao deve emitir SPEECH_END
        silero_mock.is_speech.return_value = False
        events_silence = _process_n_frames(detector, n=3)

        # Assert
        assert events_silence == []
        assert detector.is_speaking is True

    def test_is_speaking_false_after_speech_end(self) -> None:
        """is_speaking retorna False apos SPEECH_END ser emitido."""
        # Arrange
        energy_mock = _make_energy_mock(is_silence=False)
        silero_mock = _make_silero_mock(is_speech=True)
        detector = VADDetector(
            energy_pre_filter=energy_mock,
            silero_classifier=silero_mock,
            sensitivity=VADSensitivity.NORMAL,
            sample_rate=_SAMPLE_RATE,
            min_speech_duration_ms=250,
            min_silence_duration_ms=300,
        )

        # Act -- speech then silence
        _process_n_frames(detector, n=4)  # SPEECH_START
        silero_mock.is_speech.return_value = False
        _process_n_frames(detector, n=5)  # SPEECH_END

        # Assert
        assert detector.is_speaking is False


class TestEnergyPreFilterIntegration:
    """Verifica que energy pre-filter controla quando Silero e chamado."""

    def test_energy_silence_skips_silero(self) -> None:
        """Quando energy pre-filter retorna silence, Silero nao e invocado."""
        # Arrange
        energy_mock = _make_energy_mock(is_silence=True)
        silero_mock = _make_silero_mock(is_speech=True)  # Silero diria "fala" se chamado
        detector = VADDetector(
            energy_pre_filter=energy_mock,
            silero_classifier=silero_mock,
            sensitivity=VADSensitivity.NORMAL,
            sample_rate=_SAMPLE_RATE,
        )

        # Act
        _process_n_frames(detector, n=10)

        # Assert -- Silero nunca chamado
        silero_mock.is_speech.assert_not_called()

    def test_energy_non_silence_calls_silero(self) -> None:
        """Quando energy pre-filter retorna nao-silencio, Silero e chamado."""
        # Arrange
        energy_mock = _make_energy_mock(is_silence=False)
        silero_mock = _make_silero_mock(is_speech=False)
        detector = VADDetector(
            energy_pre_filter=energy_mock,
            silero_classifier=silero_mock,
            sensitivity=VADSensitivity.NORMAL,
            sample_rate=_SAMPLE_RATE,
        )

        # Act
        _process_n_frames(detector, n=3)

        # Assert -- Silero chamado para cada frame
        assert silero_mock.is_speech.call_count == 3


class TestMaxSpeechDuration:
    """Force SPEECH_END apos max_speech_duration_ms de fala continua."""

    def test_force_speech_end_after_max_duration(self) -> None:
        """SPEECH_END forcado apos max_speech_duration_ms mesmo sem silencio."""
        # Arrange -- max 640ms (= 10 frames de 64ms) para facilitar o teste
        detector, _, _ = _make_detector(
            energy_is_silence=False,
            silero_is_speech=True,
            min_speech_duration_ms=64,  # 1 frame
            max_speech_duration_ms=640,
        )

        # Act
        all_events: list[VADEvent] = []
        # Frame 1: SPEECH_START (min_speech_duration_ms=64ms = 1 frame)
        # Frame 11: force SPEECH_END (640ms = 10 frames apos speech start)
        for frame in _make_frames(15):
            event = detector.process_frame(frame)
            if event is not None:
                all_events.append(event)

        # Assert
        speech_starts = [e for e in all_events if e.type == VADEventType.SPEECH_START]
        speech_ends = [e for e in all_events if e.type == VADEventType.SPEECH_END]
        assert len(speech_starts) >= 1
        assert len(speech_ends) >= 1

    def test_max_duration_30s_default(self) -> None:
        """Com parametros default, max speech duration e 30s."""
        # Arrange -- usar frames maiores para nao processar 469 frames
        # 30000ms / 1000ms_per_frame = 30 frames de 16000 samples (1s cada)
        large_frame_size = 16000  # 1 segundo de audio
        detector, _, _ = _make_detector(
            energy_is_silence=False,
            silero_is_speech=True,
            min_speech_duration_ms=250,
            max_speech_duration_ms=30_000,
        )

        # Act -- 4 frames de 1s para atingir SPEECH_START (>250ms)
        # depois mais frames ate atingir 30s total
        all_events: list[VADEvent] = []
        for _i in range(35):
            frame = np.zeros(large_frame_size, dtype=np.float32)
            event = detector.process_frame(frame)
            if event is not None:
                all_events.append(event)

        # Assert -- deve ter SPEECH_START e SPEECH_END (force)
        speech_starts = [e for e in all_events if e.type == VADEventType.SPEECH_START]
        speech_ends = [e for e in all_events if e.type == VADEventType.SPEECH_END]
        assert len(speech_starts) >= 1
        assert len(speech_ends) >= 1
        # SPEECH_END deve ter timestamp >= 30000ms
        assert speech_ends[0].timestamp_ms >= 30_000


class TestReset:
    """Verifica que reset() limpa todo estado."""

    def test_reset_clears_all_state(self) -> None:
        """reset() limpa samples_processed, is_speaking, contadores."""
        # Arrange -- detector em estado de fala
        energy_mock = _make_energy_mock(is_silence=False)
        silero_mock = _make_silero_mock(is_speech=True)
        detector = VADDetector(
            energy_pre_filter=energy_mock,
            silero_classifier=silero_mock,
            sensitivity=VADSensitivity.NORMAL,
            sample_rate=_SAMPLE_RATE,
            min_speech_duration_ms=250,
        )

        # Processar ate SPEECH_START
        _process_n_frames(detector, n=4)
        assert detector.is_speaking is True

        # Act
        detector.reset()

        # Assert
        assert detector.is_speaking is False
        assert detector._samples_processed == 0
        assert detector._consecutive_speech_samples == 0
        assert detector._consecutive_silence_samples == 0
        silero_mock.reset.assert_called_once()

    def test_reset_allows_new_session(self) -> None:
        """Apos reset, detector pode detectar nova sequencia de fala."""
        # Arrange
        energy_mock = _make_energy_mock(is_silence=False)
        silero_mock = _make_silero_mock(is_speech=True)
        detector = VADDetector(
            energy_pre_filter=energy_mock,
            silero_classifier=silero_mock,
            sensitivity=VADSensitivity.NORMAL,
            sample_rate=_SAMPLE_RATE,
            min_speech_duration_ms=250,
        )

        # Primeira sessao
        events1 = _process_n_frames(detector, n=4)
        assert len(events1) == 1
        assert events1[0].type == VADEventType.SPEECH_START

        # Reset
        detector.reset()

        # Segunda sessao -- timestamps devem comecar do zero
        events2 = _process_n_frames(detector, n=4)
        assert len(events2) == 1
        assert events2[0].type == VADEventType.SPEECH_START
        assert events2[0].timestamp_ms == events1[0].timestamp_ms  # mesmo offset relativo


class TestTimestamps:
    """Verifica calculo correto de timestamps."""

    def test_timestamp_ms_calculated_from_samples(self) -> None:
        """timestamp_ms e calculado a partir de samples processados."""
        # Arrange
        detector, _, _ = _make_detector(
            energy_is_silence=False,
            silero_is_speech=True,
            min_speech_duration_ms=250,
        )

        # Act -- 4 frames de 1024 samples = 4096 samples
        events = _process_n_frames(detector, n=4)

        # Assert -- 4096 / 16000 * 1000 = 256ms
        assert len(events) == 1
        assert events[0].timestamp_ms == 256

    def test_speech_end_timestamp_after_speech_start(self) -> None:
        """SPEECH_END timestamp e maior que SPEECH_START timestamp."""
        # Arrange
        energy_mock = _make_energy_mock(is_silence=False)
        silero_mock = _make_silero_mock(is_speech=True)
        detector = VADDetector(
            energy_pre_filter=energy_mock,
            silero_classifier=silero_mock,
            sensitivity=VADSensitivity.NORMAL,
            sample_rate=_SAMPLE_RATE,
            min_speech_duration_ms=250,
            min_silence_duration_ms=300,
        )

        # Act -- speech (4 frames)
        events_start = _process_n_frames(detector, n=4)
        assert len(events_start) == 1

        # Then silence (5 frames)
        silero_mock.is_speech.return_value = False
        events_end = _process_n_frames(detector, n=5)
        assert len(events_end) == 1

        # Assert
        assert events_end[0].timestamp_ms > events_start[0].timestamp_ms

    def test_timestamps_accumulate_across_frames(self) -> None:
        """Timestamps acumulam corretamente ao longo de muitos frames."""
        # Arrange
        detector, _, _ = _make_detector(
            energy_is_silence=True,  # silencio -- sem eventos
        )

        # Act -- 100 frames de silencio
        _process_n_frames(detector, n=100)

        # Assert -- samples processados = 100 * 1024 = 102400
        expected_ms = 100 * _FRAME_SIZE * 1000 // _SAMPLE_RATE  # 6400ms
        assert detector._samples_processed == 100 * _FRAME_SIZE
        assert detector._compute_timestamp_ms(detector._samples_processed) == expected_ms


class TestSpeechSilenceCycles:
    """Testa ciclos completos de fala-silencio-fala."""

    def test_multiple_speech_cycles(self) -> None:
        """Multiplos ciclos speech->silence geram pares corretos de eventos."""
        # Arrange
        energy_mock = _make_energy_mock(is_silence=False)
        silero_mock = _make_silero_mock(is_speech=True)
        detector = VADDetector(
            energy_pre_filter=energy_mock,
            silero_classifier=silero_mock,
            sensitivity=VADSensitivity.NORMAL,
            sample_rate=_SAMPLE_RATE,
            min_speech_duration_ms=250,
            min_silence_duration_ms=300,
        )

        all_events: list[VADEvent] = []

        # Ciclo 1: speech (4 frames) + silence (5 frames)
        silero_mock.is_speech.return_value = True
        all_events.extend(_process_n_frames(detector, n=4))  # SPEECH_START
        silero_mock.is_speech.return_value = False
        all_events.extend(_process_n_frames(detector, n=5))  # SPEECH_END

        # Ciclo 2: speech (4 frames) + silence (5 frames)
        silero_mock.is_speech.return_value = True
        all_events.extend(_process_n_frames(detector, n=4))  # SPEECH_START
        silero_mock.is_speech.return_value = False
        all_events.extend(_process_n_frames(detector, n=5))  # SPEECH_END

        # Assert
        assert len(all_events) == 4
        assert all_events[0].type == VADEventType.SPEECH_START
        assert all_events[1].type == VADEventType.SPEECH_END
        assert all_events[2].type == VADEventType.SPEECH_START
        assert all_events[3].type == VADEventType.SPEECH_END

        # Timestamps sao monotonicamente crescentes
        for i in range(1, len(all_events)):
            assert all_events[i].timestamp_ms > all_events[i - 1].timestamp_ms

    def test_silence_resets_speech_counter(self) -> None:
        """Frame de silencio reseta contador de speech consecutivo."""
        # Arrange -- min speech = 250ms (4 frames de 64ms)
        energy_mock = _make_energy_mock(is_silence=False)
        silero_mock = _make_silero_mock(is_speech=True)
        detector = VADDetector(
            energy_pre_filter=energy_mock,
            silero_classifier=silero_mock,
            sensitivity=VADSensitivity.NORMAL,
            sample_rate=_SAMPLE_RATE,
            min_speech_duration_ms=250,
        )

        # Act -- 3 frames speech, 1 frame silence, 3 frames speech
        # Nao deve atingir 4 frames consecutivos
        all_events: list[VADEvent] = []

        # 3 frames de fala
        for frame in _make_frames(3):
            event = detector.process_frame(frame)
            if event is not None:
                all_events.append(event)

        # 1 frame de silencio (reseta contador)
        silero_mock.is_speech.return_value = False
        event = detector.process_frame(np.zeros(_FRAME_SIZE, dtype=np.float32))
        if event is not None:
            all_events.append(event)

        # 3 frames de fala (nao atinge 4 consecutivos)
        silero_mock.is_speech.return_value = True
        for frame in _make_frames(3):
            event = detector.process_frame(frame)
            if event is not None:
                all_events.append(event)

        # Assert -- sem SPEECH_START
        assert all_events == []


class TestVADEventDataclass:
    """Testa propriedades do dataclass VADEvent."""

    def test_vad_event_is_frozen(self) -> None:
        """VADEvent e imutavel (frozen=True)."""
        # Arrange
        event = VADEvent(type=VADEventType.SPEECH_START, timestamp_ms=100)

        # Act & Assert
        try:
            event.timestamp_ms = 200  # type: ignore[misc]
            raise AssertionError("Deveria levantar FrozenInstanceError")
        except AttributeError:
            pass  # Expected -- frozen dataclass

    def test_vad_event_type_enum_values(self) -> None:
        """VADEventType tem valores corretos."""
        assert VADEventType.SPEECH_START.value == "speech_start"
        assert VADEventType.SPEECH_END.value == "speech_end"
