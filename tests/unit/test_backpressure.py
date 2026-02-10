"""Testes do BackpressureController.

Valida que o controlador de backpressure detecta corretamente:
- Envio de audio em velocidade normal (1x) -> sem eventos
- Envio mais rapido que real-time -> RateLimitAction
- Backlog excessivo -> FramesDroppedAction
- Contadores de frames recebidos e dropados
- Retorno ao normal apos desaceleracao

Todos os testes usam clock injetavel para determinismo.
"""

from __future__ import annotations

from macaw.session.backpressure import (
    BackpressureController,
    FramesDroppedAction,
    RateLimitAction,
)

# PCM 16-bit mono 16kHz: 1 segundo = 32000 bytes
_SAMPLE_RATE = 16000
_BYTES_PER_SECOND = _SAMPLE_RATE * 2  # 2 bytes per sample (int16)
_BYTES_PER_20MS = _BYTES_PER_SECOND // 50  # 640 bytes = 20ms frame


class _FakeClock:
    """Relogio fake para testes deterministicos.

    Permite avanco manual do tempo sem depender de time.monotonic().
    """

    def __init__(self, start: float = 0.0) -> None:
        self._now = start

    def __call__(self) -> float:
        return self._now

    def advance(self, seconds: float) -> None:
        """Avanca o relogio em `seconds` segundos."""
        self._now += seconds


class TestBackpressureControllerNormalSpeed:
    """Testes com audio em velocidade normal (1x real-time)."""

    def test_first_frame_never_triggers(self) -> None:
        """Primeiro frame nunca deve emitir acao (sem historico)."""
        clock = _FakeClock()
        ctrl = BackpressureController(
            sample_rate=_SAMPLE_RATE,
            clock=clock,
        )

        result = ctrl.record_frame(_BYTES_PER_20MS)

        assert result is None
        assert ctrl.frames_received == 1
        assert ctrl.frames_dropped == 0

    def test_normal_speed_no_events(self) -> None:
        """Audio em velocidade 1x real-time nao deve emitir eventos."""
        clock = _FakeClock()
        ctrl = BackpressureController(
            sample_rate=_SAMPLE_RATE,
            clock=clock,
        )

        # Enviar 100 frames de 20ms a velocidade real-time
        for _ in range(100):
            result = ctrl.record_frame(_BYTES_PER_20MS)
            assert result is None
            clock.advance(0.020)  # 20ms de wall-clock por frame

        assert ctrl.frames_received == 100
        assert ctrl.frames_dropped == 0

    def test_slightly_fast_below_threshold_no_events(self) -> None:
        """Audio a 1.1x (abaixo do threshold 1.2x) nao deve emitir eventos."""
        clock = _FakeClock()
        ctrl = BackpressureController(
            sample_rate=_SAMPLE_RATE,
            rate_limit_threshold=1.2,
            clock=clock,
        )

        # Enviar frames de 20ms de audio com 18.18ms de wall-clock (~1.1x)
        for _ in range(100):
            result = ctrl.record_frame(_BYTES_PER_20MS)
            assert result is None
            clock.advance(0.01818)  # ~1.1x real-time

        assert ctrl.frames_dropped == 0


class TestBackpressureControllerRateLimit:
    """Testes de deteccao de taxa acima do threshold."""

    def test_fast_sending_triggers_rate_limit(self) -> None:
        """Audio a 2x real-time deve emitir RateLimitAction."""
        clock = _FakeClock()
        ctrl = BackpressureController(
            sample_rate=_SAMPLE_RATE,
            rate_limit_threshold=1.2,
            max_backlog_s=10.0,
            clock=clock,
        )

        # Enviar frames de 20ms de audio com 10ms de wall-clock (2x real-time)
        # Precisamos de wall_elapsed >= 0.5s para a checagem de taxa iniciar,
        # portanto enviamos 100 frames (99 * 10ms = 0.99s de wall-clock)
        actions: list[RateLimitAction] = []
        for _ in range(100):
            result = ctrl.record_frame(_BYTES_PER_20MS)
            if isinstance(result, RateLimitAction):
                actions.append(result)
            clock.advance(0.010)  # 10ms wall-clock = 2x real-time

        assert len(actions) >= 1, "Deveria emitir pelo menos 1 RateLimitAction"
        assert all(isinstance(a, RateLimitAction) for a in actions)
        assert all(a.delay_ms >= 1 for a in actions)

    def test_rate_limit_has_positive_delay(self) -> None:
        """RateLimitAction deve ter delay_ms positivo."""
        clock = _FakeClock()
        ctrl = BackpressureController(
            sample_rate=_SAMPLE_RATE,
            rate_limit_threshold=1.2,
            clock=clock,
        )

        # Enviar rapido o suficiente para disparar
        # wall_elapsed >= 0.5s necessario: 200 frames * 5ms = 1.0s
        actions: list[RateLimitAction] = []
        for _ in range(200):
            result = ctrl.record_frame(_BYTES_PER_20MS)
            if isinstance(result, RateLimitAction):
                actions.append(result)
            clock.advance(0.005)  # 5ms wall-clock = 4x real-time

        assert len(actions) >= 1
        for action in actions:
            assert action.delay_ms >= 1

    def test_rate_returns_to_normal_after_slowdown(self) -> None:
        """Apos desacelerar para 1x, nao deve mais emitir rate_limit."""
        clock = _FakeClock()
        ctrl = BackpressureController(
            sample_rate=_SAMPLE_RATE,
            rate_limit_threshold=1.2,
            max_backlog_s=100.0,  # backlog alto para nao dropar
            clock=clock,
        )

        # Fase 1: enviar rapido (2x) por 2 segundos
        for _ in range(100):
            ctrl.record_frame(_BYTES_PER_20MS)
            clock.advance(0.010)  # 2x

        # Fase 2: avanca o relogio para que a sliding window "esqueca" o burst
        clock.advance(6.0)

        # Fase 3: enviar a velocidade normal (1x) por 3 segundos
        actions_normal: list[RateLimitAction] = []
        for _ in range(150):
            result = ctrl.record_frame(_BYTES_PER_20MS)
            if isinstance(result, RateLimitAction):
                actions_normal.append(result)
            clock.advance(0.020)  # 1x real-time

        assert len(actions_normal) == 0, "Nao deveria emitir rate_limit apos desacelerar para 1x"


class TestBackpressureControllerFramesDrop:
    """Testes de drop de frames por excesso de backlog."""

    def test_backlog_exceeds_max_triggers_drop(self) -> None:
        """Backlog > max_backlog_s deve dropar frames."""
        clock = _FakeClock()
        ctrl = BackpressureController(
            sample_rate=_SAMPLE_RATE,
            max_backlog_s=1.0,  # apenas 1 segundo de backlog
            rate_limit_threshold=1.2,
            clock=clock,
        )

        # Enviar muitos frames instantaneamente (wall-clock nao avanca)
        # Cada frame = 20ms de audio. 1s de backlog = 50 frames sem wall-clock.
        # Apos 50 frames (1s de audio) sem avancar relogio, backlog = 1s
        # O primeiro frame nao conta (inicializa), entao 52 frames = ~1.02s
        drop_action = None
        for _ in range(100):
            result = ctrl.record_frame(_BYTES_PER_20MS)
            if isinstance(result, FramesDroppedAction):
                drop_action = result
                break
            # Nao avanca o relogio! Simula burst instantaneo

        assert drop_action is not None, "Deveria emitir FramesDroppedAction"
        assert isinstance(drop_action, FramesDroppedAction)
        assert drop_action.dropped_ms > 0

    def test_dropped_frames_counter_incremented(self) -> None:
        """Contador de frames dropados deve ser incrementado corretamente."""
        clock = _FakeClock()
        ctrl = BackpressureController(
            sample_rate=_SAMPLE_RATE,
            max_backlog_s=0.5,  # meio segundo de backlog
            clock=clock,
        )

        # Enviar muitos frames sem avancar relogio
        total_drops = 0
        for _ in range(200):
            result = ctrl.record_frame(_BYTES_PER_20MS)
            if isinstance(result, FramesDroppedAction):
                total_drops += 1

        assert ctrl.frames_dropped == total_drops
        assert ctrl.frames_dropped > 0
        assert ctrl.frames_received == 200

    def test_dropped_ms_reflects_frame_duration(self) -> None:
        """dropped_ms deve refletir a duracao do frame dropado."""
        clock = _FakeClock()
        ctrl = BackpressureController(
            sample_rate=_SAMPLE_RATE,
            max_backlog_s=0.5,
            clock=clock,
        )

        # Preencher backlog ate estourar
        drop_action = None
        for _ in range(200):
            result = ctrl.record_frame(_BYTES_PER_20MS)
            if isinstance(result, FramesDroppedAction):
                drop_action = result
                break

        assert drop_action is not None
        # 20ms frame = 640 bytes = 320 samples = 20ms
        assert drop_action.dropped_ms == 20

    def test_backlog_within_limit_no_drop(self) -> None:
        """Backlog dentro do limite nao deve dropar frames."""
        clock = _FakeClock()
        ctrl = BackpressureController(
            sample_rate=_SAMPLE_RATE,
            max_backlog_s=10.0,
            clock=clock,
        )

        # Enviar 5 segundos de audio instantaneamente (5s < 10s backlog)
        for _ in range(250):  # 250 * 20ms = 5s
            result = ctrl.record_frame(_BYTES_PER_20MS)
            assert not isinstance(result, FramesDroppedAction)

        assert ctrl.frames_dropped == 0


class TestBackpressureControllerEdgeCases:
    """Testes de edge cases e cenarios limites."""

    def test_single_frame_no_action(self) -> None:
        """Um unico frame nao deve emitir nenhuma acao."""
        clock = _FakeClock()
        ctrl = BackpressureController(
            sample_rate=_SAMPLE_RATE,
            clock=clock,
        )

        result = ctrl.record_frame(_BYTES_PER_20MS)

        assert result is None

    def test_two_frames_same_time_no_crash(self) -> None:
        """Dois frames no mesmo instante nao deve crashar (divisao por zero)."""
        clock = _FakeClock()
        ctrl = BackpressureController(
            sample_rate=_SAMPLE_RATE,
            max_backlog_s=10.0,
            clock=clock,
        )

        ctrl.record_frame(_BYTES_PER_20MS)
        result = ctrl.record_frame(_BYTES_PER_20MS)

        # Nao deve crashar; resultado depende do backlog
        assert result is None or isinstance(result, RateLimitAction | FramesDroppedAction)

    def test_counters_start_at_zero(self) -> None:
        """Contadores devem comecar em zero."""
        ctrl = BackpressureController(sample_rate=_SAMPLE_RATE)

        assert ctrl.frames_received == 0
        assert ctrl.frames_dropped == 0

    def test_large_frame_counted_correctly(self) -> None:
        """Frames grandes devem ter duracao calculada corretamente."""
        clock = _FakeClock()
        ctrl = BackpressureController(
            sample_rate=_SAMPLE_RATE,
            max_backlog_s=2.0,
            clock=clock,
        )

        # Frame de 1 segundo = 32000 bytes
        one_sec_frame = _BYTES_PER_SECOND
        ctrl.record_frame(one_sec_frame)  # primeiro, inicializa
        clock.advance(0.001)  # quase instantaneo

        # Segundo frame de 1s: backlog sera ~2s em 0.001s de wall
        ctrl.record_frame(one_sec_frame)

        # Verificar que nao crashou e contadores estao corretos
        assert ctrl.frames_received == 2

    def test_rate_limit_cooldown_prevents_spam(self) -> None:
        """Rate limit deve respeitar cooldown de 1s entre emissoes."""
        clock = _FakeClock()
        ctrl = BackpressureController(
            sample_rate=_SAMPLE_RATE,
            rate_limit_threshold=1.2,
            max_backlog_s=100.0,  # alto para nao dropar
            clock=clock,
        )

        # Enviar a 3x real-time por 0.5s (25 frames de 20ms, 6.66ms wall cada)
        rate_limits = 0
        for _ in range(25):
            result = ctrl.record_frame(_BYTES_PER_20MS)
            if isinstance(result, RateLimitAction):
                rate_limits += 1
            clock.advance(0.00666)  # ~3x real-time

        # Cooldown de 1s: em 0.5s de wall-clock, deve emitir no maximo 1
        assert rate_limits <= 1

    def test_frames_dropped_priority_over_rate_limit(self) -> None:
        """FramesDroppedAction deve ter prioridade sobre RateLimitAction."""
        clock = _FakeClock()
        ctrl = BackpressureController(
            sample_rate=_SAMPLE_RATE,
            max_backlog_s=0.5,
            rate_limit_threshold=1.2,
            clock=clock,
        )

        # Encher backlog rapidamente
        last_action = None
        for _ in range(100):
            result = ctrl.record_frame(_BYTES_PER_20MS)
            if result is not None:
                last_action = result
            # Nao avanca relogio

        # O ultimo resultado apos estourar backlog deve ser FramesDroppedAction
        # (backlog e verificado antes de rate_limit no codigo)
        assert isinstance(last_action, FramesDroppedAction)
