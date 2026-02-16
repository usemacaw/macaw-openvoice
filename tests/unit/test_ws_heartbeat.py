"""Testes de heartbeat e inactivity timeout do WebSocket /v1/realtime."""

from __future__ import annotations

import time
from unittest.mock import MagicMock

from starlette.testclient import TestClient

from macaw.exceptions import ModelNotFoundError
from macaw.server.app import create_app


def _make_mock_registry(*, known_models: list[str] | None = None) -> MagicMock:
    """Cria mock do ModelRegistry que conhece os modelos em known_models."""
    if known_models is None:
        known_models = ["faster-whisper-tiny"]

    registry = MagicMock()

    def _get_manifest(model_name: str) -> MagicMock:
        if model_name in known_models:
            manifest = MagicMock()
            manifest.name = model_name
            return manifest
        raise ModelNotFoundError(model_name)

    registry.get_manifest = MagicMock(side_effect=_get_manifest)
    return registry


def _make_app_with_short_timeouts(
    *,
    inactivity_s: float = 0.3,
    heartbeat_s: float = 10.0,
    check_s: float = 0.1,
) -> TestClient:
    """Cria app FastAPI com timeouts curtos para testes rapidos."""
    app = create_app(registry=_make_mock_registry())
    app.state.ws_inactivity_timeout_s = inactivity_s
    app.state.ws_heartbeat_interval_s = heartbeat_s
    app.state.ws_check_interval_s = check_s
    return TestClient(app)


def test_inactivity_timeout_closes_session() -> None:
    """Sessao sem audio frames recebidos e fechada apos inactivity timeout."""
    client = _make_app_with_short_timeouts(inactivity_s=0.3, check_s=0.1)

    with client.websocket_connect("/v1/realtime?model=faster-whisper-tiny") as ws:
        created = ws.receive_json()
        assert created["type"] == "session.created"

        # Nao enviar nenhum audio -- aguardar timeout
        # O monitor vai detectar inatividade e emitir session.closed
        closed = ws.receive_json()
        assert closed["type"] == "session.closed"
        assert closed["reason"] == "inactivity_timeout"
        assert closed["total_duration_ms"] >= 0
        assert closed["segments_transcribed"] == 0


def test_audio_frame_resets_inactivity_timer() -> None:
    """Envio de audio frame reseta o timer de inatividade."""
    client = _make_app_with_short_timeouts(inactivity_s=0.4, check_s=0.1)

    with client.websocket_connect("/v1/realtime?model=faster-whisper-tiny") as ws:
        created = ws.receive_json()
        assert created["type"] == "session.created"

        # Enviar audio frames por um periodo maior que inactivity_timeout
        # para provar que o timer e resetado
        start = time.monotonic()
        for _ in range(3):
            ws.send_bytes(b"\x00\x01\x02\x03" * 100)
            time.sleep(0.15)

        elapsed = time.monotonic() - start
        assert elapsed > 0.4, "Deveria ter passado mais que inactivity_timeout"

        # Agora parar de enviar e aguardar timeout
        closed = ws.receive_json()
        assert closed["type"] == "session.closed"
        assert closed["reason"] == "inactivity_timeout"


def test_session_closed_event_has_correct_duration() -> None:
    """Evento session.closed por timeout tem total_duration_ms > 0."""
    client = _make_app_with_short_timeouts(inactivity_s=0.2, check_s=0.1)

    with client.websocket_connect("/v1/realtime?model=faster-whisper-tiny") as ws:
        created = ws.receive_json()
        assert created["type"] == "session.created"

        closed = ws.receive_json()
        assert closed["type"] == "session.closed"
        assert closed["reason"] == "inactivity_timeout"
        # A duracao deve ser pelo menos o tempo do timeout
        assert closed["total_duration_ms"] >= 200


def test_normal_close_does_not_emit_duplicate_session_closed() -> None:
    """Fechamento normal (session.close) nao emite session.closed duplicado."""
    client = _make_app_with_short_timeouts(inactivity_s=5.0, check_s=0.1)

    with client.websocket_connect("/v1/realtime?model=faster-whisper-tiny") as ws:
        created = ws.receive_json()
        assert created["type"] == "session.created"

        ws.send_json({"type": "session.close"})

        # Deve receber exatamente um session.closed com reason=client_request
        closed = ws.receive_json()
        assert closed["type"] == "session.closed"
        assert closed["reason"] == "client_request"


def test_cancel_does_not_emit_duplicate_session_closed() -> None:
    """Cancelamento (session.cancel) nao emite session.closed duplicado."""
    client = _make_app_with_short_timeouts(inactivity_s=5.0, check_s=0.1)

    with client.websocket_connect("/v1/realtime?model=faster-whisper-tiny") as ws:
        created = ws.receive_json()
        assert created["type"] == "session.created"

        ws.send_json({"type": "session.cancel"})

        closed = ws.receive_json()
        assert closed["type"] == "session.closed"
        assert closed["reason"] == "cancelled"


def test_default_timeouts_used_when_not_configured() -> None:
    """Default values are used when app.state has no configuration."""
    from macaw.server.constants import (
        WS_CHECK_INTERVAL_S,
        WS_HEARTBEAT_INTERVAL_S,
        WS_INACTIVITY_TIMEOUT_S,
    )

    assert WS_HEARTBEAT_INTERVAL_S == 10.0
    assert WS_INACTIVITY_TIMEOUT_S == 60.0
    assert WS_CHECK_INTERVAL_S == 5.0


def test_text_command_does_not_reset_inactivity_timer() -> None:
    """Comandos JSON nao resetam o timer de inatividade (apenas audio)."""
    client = _make_app_with_short_timeouts(inactivity_s=0.3, check_s=0.1)

    with client.websocket_connect("/v1/realtime?model=faster-whisper-tiny") as ws:
        created = ws.receive_json()
        assert created["type"] == "session.created"

        # Enviar comandos de configuracao (nao audio)
        ws.send_json({"type": "session.configure", "language": "pt"})
        time.sleep(0.15)
        ws.send_json({"type": "session.configure", "language": "en"})

        # Mesmo com comandos, o timeout de inatividade deve disparar
        closed = ws.receive_json()
        assert closed["type"] == "session.closed"
        assert closed["reason"] == "inactivity_timeout"
