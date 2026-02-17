"""Testes do endpoint WebSocket /v1/realtime."""

from __future__ import annotations

import time
from typing import Any
from unittest.mock import MagicMock

import pytest
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
    registry.has_model = MagicMock(side_effect=lambda name: name in known_models)
    return registry


def test_successful_handshake_emits_session_created() -> None:
    """Conexao com modelo valido emite session.created."""
    app = create_app(registry=_make_mock_registry())
    client = TestClient(app)

    with client.websocket_connect("/v1/realtime?model=faster-whisper-tiny") as ws:
        event = ws.receive_json()

    assert event["type"] == "session.created"
    assert event["model"] == "faster-whisper-tiny"
    assert event["session_id"].startswith("sess_")
    assert "config" in event


def test_session_created_includes_default_config() -> None:
    """session.created inclui config com defaults."""
    app = create_app(registry=_make_mock_registry())
    client = TestClient(app)

    with client.websocket_connect("/v1/realtime?model=faster-whisper-tiny") as ws:
        event = ws.receive_json()

    config = event["config"]
    assert config["vad_sensitivity"] == "normal"
    assert config["silence_timeout_ms"] == 30_000
    assert config["hold_timeout_ms"] == 300_000
    assert config["max_segment_duration_ms"] == 30_000
    assert config["enable_partial_transcripts"] is True
    assert config["enable_itn"] is True


def test_session_created_includes_language_when_provided() -> None:
    """session.created config inclui language quando especificado no handshake."""
    app = create_app(registry=_make_mock_registry())
    client = TestClient(app)

    with client.websocket_connect("/v1/realtime?model=faster-whisper-tiny&language=pt") as ws:
        event = ws.receive_json()

    assert event["config"]["language"] == "pt"


def test_missing_model_closes_with_error() -> None:
    """Conexao sem model param recebe erro e fecha com 1008."""
    app = create_app(registry=_make_mock_registry())
    client = TestClient(app)

    with client.websocket_connect("/v1/realtime") as ws:
        error_event = ws.receive_json()
        assert error_event["type"] == "error"
        assert error_event["code"] == "invalid_request"
        assert "model" in error_event["message"].lower()
        assert error_event["recoverable"] is False


def test_invalid_model_closes_with_error() -> None:
    """Conexao com modelo inexistente recebe erro e fecha com 1008."""
    app = create_app(registry=_make_mock_registry(known_models=["faster-whisper-tiny"]))
    client = TestClient(app)

    with client.websocket_connect("/v1/realtime?model=nonexistent-model") as ws:
        error_event = ws.receive_json()
        assert error_event["type"] == "error"
        assert error_event["code"] == "model_not_found"
        assert "nonexistent-model" in error_event["message"]
        assert error_event["recoverable"] is False


def test_no_registry_closes_with_error() -> None:
    """Conexao quando registry e None recebe erro e fecha com 1008."""
    app = create_app(registry=None)
    client = TestClient(app)

    with client.websocket_connect("/v1/realtime?model=faster-whisper-tiny") as ws:
        error_event = ws.receive_json()
        assert error_event["type"] == "error"
        assert error_event["code"] == "service_unavailable"
        assert "no models" in error_event["message"].lower()
        assert error_event["recoverable"] is False


def test_multiple_connections_have_unique_session_ids() -> None:
    """Cada conexao recebe session_id unico."""
    app = create_app(registry=_make_mock_registry())
    client = TestClient(app)

    session_ids = []
    for _ in range(3):
        with client.websocket_connect("/v1/realtime?model=faster-whisper-tiny") as ws:
            event = ws.receive_json()
            session_ids.append(event["session_id"])

    assert len(set(session_ids)) == 3, f"Session IDs nao sao unicos: {session_ids}"


def test_client_disconnect_emits_session_closed() -> None:
    """Desconexao do cliente emite session.closed."""
    app = create_app(registry=_make_mock_registry())
    client = TestClient(app)

    with client.websocket_connect("/v1/realtime?model=faster-whisper-tiny") as ws:
        created = ws.receive_json()
        assert created["type"] == "session.created"

        # Enviar um frame de texto para manter a sessao ativa
        ws.send_text('{"type": "session.close"}')

        # O proximo evento deve ser session.closed
        # (o handler fecha apos receber session.close no finally)


def test_binary_frame_accepted() -> None:
    """Frames binarios (audio) sao aceitos sem erro."""
    app = create_app(registry=_make_mock_registry())
    client = TestClient(app)

    with client.websocket_connect("/v1/realtime?model=faster-whisper-tiny") as ws:
        created = ws.receive_json()
        assert created["type"] == "session.created"

        # Enviar frame binario (audio fake)
        ws.send_bytes(b"\x00\x01\x02\x03" * 100)


def test_text_command_accepted() -> None:
    """Comandos JSON sao aceitos sem erro."""
    app = create_app(registry=_make_mock_registry())
    client = TestClient(app)

    with client.websocket_connect("/v1/realtime?model=faster-whisper-tiny") as ws:
        created = ws.receive_json()
        assert created["type"] == "session.created"

        # Enviar comando JSON
        ws.send_json({"type": "session.configure", "language": "pt"})


def test_session_id_format() -> None:
    """Session ID segue formato sess_ + 12 chars hex."""
    app = create_app(registry=_make_mock_registry())
    client = TestClient(app)

    with client.websocket_connect("/v1/realtime?model=faster-whisper-tiny") as ws:
        event = ws.receive_json()

    session_id = event["session_id"]
    assert session_id.startswith("sess_")
    hex_part = session_id[5:]
    assert len(hex_part) == 12
    # Verificar que e hexadecimal valido
    int(hex_part, 16)


# ---------------------------------------------------------------------------
# T5-03: Protocol dispatch integration tests
# ---------------------------------------------------------------------------


def test_session_close_command_closes_connection() -> None:
    """session.close emite session.closed com reason=client_request e fecha."""
    app = create_app(registry=_make_mock_registry())
    client = TestClient(app)

    with client.websocket_connect("/v1/realtime?model=faster-whisper-tiny") as ws:
        created = ws.receive_json()
        assert created["type"] == "session.created"

        ws.send_json({"type": "session.close"})

        # Deve receber session.closed com reason=client_request
        closed = ws.receive_json()
        assert closed["type"] == "session.closed"
        assert closed["reason"] == "client_request"


def test_session_cancel_command_closes_connection() -> None:
    """session.cancel emite session.closed com reason=cancelled e fecha."""
    app = create_app(registry=_make_mock_registry())
    client = TestClient(app)

    with client.websocket_connect("/v1/realtime?model=faster-whisper-tiny") as ws:
        created = ws.receive_json()
        assert created["type"] == "session.created"

        ws.send_json({"type": "session.cancel"})

        # Deve receber session.closed com reason=cancelled
        closed = ws.receive_json()
        assert closed["type"] == "session.closed"
        assert closed["reason"] == "cancelled"


def test_malformed_json_returns_error_and_keeps_connection() -> None:
    """JSON malformado retorna erro recuperavel sem fechar conexao."""
    app = create_app(registry=_make_mock_registry())
    client = TestClient(app)

    with client.websocket_connect("/v1/realtime?model=faster-whisper-tiny") as ws:
        created = ws.receive_json()
        assert created["type"] == "session.created"

        # Enviar JSON invalido
        ws.send_text("this is not json {{{")

        # Deve receber erro recuperavel
        error = ws.receive_json()
        assert error["type"] == "error"
        assert error["code"] == "malformed_json"
        assert error["recoverable"] is True

        # Conexao ainda esta ativa -- podemos enviar outro comando
        ws.send_json({"type": "session.close"})
        closed = ws.receive_json()
        assert closed["type"] == "session.closed"
        assert closed["reason"] == "client_request"


def test_unknown_command_returns_error_and_keeps_connection() -> None:
    """Tipo de comando desconhecido retorna erro recuperavel sem fechar conexao."""
    app = create_app(registry=_make_mock_registry())
    client = TestClient(app)

    with client.websocket_connect("/v1/realtime?model=faster-whisper-tiny") as ws:
        created = ws.receive_json()
        assert created["type"] == "session.created"

        ws.send_json({"type": "unknown.nonsense"})

        error = ws.receive_json()
        assert error["type"] == "error"
        assert error["code"] == "unknown_command"
        assert error["recoverable"] is True

        # Conexao ainda funciona
        ws.send_json({"type": "session.close"})
        closed = ws.receive_json()
        assert closed["type"] == "session.closed"


def test_session_configure_accepted_without_closing() -> None:
    """session.configure e aceito e nao fecha a conexao."""
    app = create_app(registry=_make_mock_registry())
    client = TestClient(app)

    with client.websocket_connect("/v1/realtime?model=faster-whisper-tiny") as ws:
        created = ws.receive_json()
        assert created["type"] == "session.created"

        ws.send_json(
            {
                "type": "session.configure",
                "language": "pt",
                "vad_sensitivity": "high",
            }
        )

        # session.configure nao emite resposta ainda (placeholder)
        # Mas a conexao deve estar ativa para continuar
        ws.send_json({"type": "session.close"})
        closed = ws.receive_json()
        assert closed["type"] == "session.closed"
        assert closed["reason"] == "client_request"


def test_input_audio_buffer_commit_accepted_without_closing() -> None:
    """input_audio_buffer.commit e aceito e nao fecha a conexao."""
    app = create_app(registry=_make_mock_registry())
    client = TestClient(app)

    with client.websocket_connect("/v1/realtime?model=faster-whisper-tiny") as ws:
        created = ws.receive_json()
        assert created["type"] == "session.created"

        ws.send_json({"type": "input_audio_buffer.commit"})

        # Nao emite resposta, conexao deve estar ativa
        ws.send_json({"type": "session.close"})
        closed = ws.receive_json()
        assert closed["type"] == "session.closed"
        assert closed["reason"] == "client_request"


# ---------------------------------------------------------------------------
# HTTP stub for Swagger documentation
# ---------------------------------------------------------------------------


def test_http_get_returns_426_upgrade_required() -> None:
    """GET /v1/realtime retorna 426 Upgrade Required com hint de WebSocket."""
    app = create_app(registry=_make_mock_registry())
    client = TestClient(app, raise_server_exceptions=False)

    response = client.get("/v1/realtime", params={"model": "faster-whisper-tiny"})

    assert response.status_code == 426
    body = response.json()
    assert "websocket" in body["error"].lower()
    assert "faster-whisper-tiny" in body["hint"]


def test_http_get_includes_upgrade_header() -> None:
    """GET /v1/realtime inclui header Upgrade: websocket na resposta."""
    app = create_app(registry=_make_mock_registry())
    client = TestClient(app, raise_server_exceptions=False)

    response = client.get("/v1/realtime", params={"model": "faster-whisper-tiny"})

    assert response.headers.get("upgrade") == "websocket"


# ---------------------------------------------------------------------------
# M-09: Frame size limit
# ---------------------------------------------------------------------------


def test_oversized_binary_frame_returns_error() -> None:
    """Binary frame exceeding MAX_WS_FRAME_SIZE returns frame_too_large error."""
    from macaw.server.routes.realtime import MAX_WS_FRAME_SIZE

    app = create_app(registry=_make_mock_registry())
    client = TestClient(app)

    with client.websocket_connect("/v1/realtime?model=faster-whisper-tiny") as ws:
        created = ws.receive_json()
        assert created["type"] == "session.created"

        # Send a frame that exceeds the limit
        oversized = b"\x00" * (MAX_WS_FRAME_SIZE + 1)
        ws.send_bytes(oversized)

        error = ws.receive_json()
        assert error["type"] == "error"
        assert error["code"] == "frame_too_large"
        assert error["recoverable"] is True

        # Connection should still be alive
        ws.send_json({"type": "session.close"})
        closed = ws.receive_json()
        assert closed["type"] == "session.closed"


def test_max_size_binary_frame_accepted() -> None:
    """Binary frame at exactly MAX_WS_FRAME_SIZE is accepted (no error)."""
    from macaw.server.routes.realtime import MAX_WS_FRAME_SIZE

    app = create_app(registry=_make_mock_registry())
    client = TestClient(app)

    with client.websocket_connect("/v1/realtime?model=faster-whisper-tiny") as ws:
        created = ws.receive_json()
        assert created["type"] == "session.created"

        # Send a frame at exactly the limit — should be accepted
        exact_frame = b"\x00" * MAX_WS_FRAME_SIZE
        ws.send_bytes(exact_frame)

        # No error expected; close cleanly
        ws.send_json({"type": "session.close"})
        closed = ws.receive_json()
        assert closed["type"] == "session.closed"


# ---------------------------------------------------------------------------
# F-12: _handle_configure_command wires timeout fields
# ---------------------------------------------------------------------------


def _make_session_context_with_mock_session() -> tuple[Any, MagicMock, Any]:
    """Create a minimal SessionContext with a mocked session for configure tests.

    Returns (ctx, mock_session, current_timeouts).
    """
    from macaw.server.routes.realtime import SessionContext
    from macaw.session.state_machine import SessionTimeouts

    current_timeouts = SessionTimeouts(
        init_timeout_s=30.0,
        silence_timeout_s=30.0,
        hold_timeout_s=300.0,
        closing_timeout_s=2.0,
    )

    mock_session = MagicMock()
    mock_session.current_timeouts = current_timeouts
    mock_session.update_session_timeouts = MagicMock()
    mock_session.update_hot_words = MagicMock()
    mock_session.update_itn = MagicMock()

    mock_websocket = MagicMock()
    ctx = SessionContext(
        session_id="sess_test123",
        session_start=time.monotonic(),
        websocket=mock_websocket,
        session=mock_session,
    )
    return ctx, mock_session, current_timeouts


class TestConfigureCommandTimeouts:
    """Tests for _handle_configure_command timeout wiring (F-12)."""

    @pytest.fixture()
    def _ctx_and_mocks(self) -> tuple[Any, MagicMock, Any]:
        return _make_session_context_with_mock_session()

    async def test_silence_timeout_only_updates_timeouts(
        self,
        _ctx_and_mocks: tuple[Any, MagicMock, Any],
    ) -> None:
        """When only silence_timeout_ms is provided, update_session_timeouts is called."""
        from macaw.server.models.events import SessionConfigureCommand
        from macaw.server.routes.realtime import _handle_configure_command

        ctx, mock_session, current_timeouts = _ctx_and_mocks
        cmd = SessionConfigureCommand(silence_timeout_ms=5000)

        result = await _handle_configure_command(ctx, cmd)

        assert result is False
        mock_session.update_session_timeouts.assert_called_once()
        new_timeouts = mock_session.update_session_timeouts.call_args[0][0]
        assert new_timeouts.silence_timeout_s == 5.0
        assert new_timeouts.hold_timeout_s == current_timeouts.hold_timeout_s

    async def test_hold_timeout_only_updates_timeouts(
        self,
        _ctx_and_mocks: tuple[Any, MagicMock, Any],
    ) -> None:
        """When only hold_timeout_ms is provided, update_session_timeouts is called."""
        from macaw.server.models.events import SessionConfigureCommand
        from macaw.server.routes.realtime import _handle_configure_command

        ctx, mock_session, current_timeouts = _ctx_and_mocks
        cmd = SessionConfigureCommand(hold_timeout_ms=600_000)

        result = await _handle_configure_command(ctx, cmd)

        assert result is False
        mock_session.update_session_timeouts.assert_called_once()
        new_timeouts = mock_session.update_session_timeouts.call_args[0][0]
        assert new_timeouts.hold_timeout_s == 600.0
        assert new_timeouts.silence_timeout_s == current_timeouts.silence_timeout_s

    async def test_both_timeouts_updates_together(
        self,
        _ctx_and_mocks: tuple[Any, MagicMock, Any],
    ) -> None:
        """When both silence_timeout_ms and hold_timeout_ms are provided, both update."""
        from macaw.server.models.events import SessionConfigureCommand
        from macaw.server.routes.realtime import _handle_configure_command

        ctx, mock_session, _current_timeouts = _ctx_and_mocks
        cmd = SessionConfigureCommand(silence_timeout_ms=2000, hold_timeout_ms=120_000)

        result = await _handle_configure_command(ctx, cmd)

        assert result is False
        mock_session.update_session_timeouts.assert_called_once()
        new_timeouts = mock_session.update_session_timeouts.call_args[0][0]
        assert new_timeouts.silence_timeout_s == 2.0
        assert new_timeouts.hold_timeout_s == 120.0

    async def test_no_timeouts_does_not_update(
        self,
        _ctx_and_mocks: tuple[Any, MagicMock, Any],
    ) -> None:
        """When neither timeout is provided, update_session_timeouts is NOT called."""
        from macaw.server.models.events import SessionConfigureCommand
        from macaw.server.routes.realtime import _handle_configure_command

        ctx, mock_session, _current_timeouts = _ctx_and_mocks
        cmd = SessionConfigureCommand(hot_words=["Macaw", "OpenVoice"])

        result = await _handle_configure_command(ctx, cmd)

        assert result is False
        mock_session.update_session_timeouts.assert_not_called()
        mock_session.update_hot_words.assert_called_once_with(["Macaw", "OpenVoice"])

    async def test_no_session_skips_all_updates(self) -> None:
        """When ctx.session is None, no session methods are called."""
        from macaw.server.models.events import SessionConfigureCommand
        from macaw.server.routes.realtime import SessionContext, _handle_configure_command

        mock_websocket = MagicMock()
        ctx = SessionContext(
            session_id="sess_nosession",
            session_start=time.monotonic(),
            websocket=mock_websocket,
            session=None,
        )
        cmd = SessionConfigureCommand(silence_timeout_ms=5000, hold_timeout_ms=120_000)

        result = await _handle_configure_command(ctx, cmd)

        assert result is False

    async def test_model_tts_updated_regardless_of_session(self) -> None:
        """model_tts is updated on ctx even when session is None."""
        from macaw.server.models.events import SessionConfigureCommand
        from macaw.server.routes.realtime import SessionContext, _handle_configure_command

        mock_websocket = MagicMock()
        ctx = SessionContext(
            session_id="sess_ttsonly",
            session_start=time.monotonic(),
            websocket=mock_websocket,
            session=None,
        )
        cmd = SessionConfigureCommand(model_tts="kokoro-v1")

        result = await _handle_configure_command(ctx, cmd)

        assert result is False
        assert ctx.model_tts == "kokoro-v1"
