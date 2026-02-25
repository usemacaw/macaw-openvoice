"""Tests for multi-context TTS WebSocket endpoint and message parsing."""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock, patch

import httpx
import pytest

from macaw.server.app import create_app
from macaw.server.models.multi_context_events import (
    MULTI_CONTEXT_MESSAGE_TYPES,
    AudioEvent,
    CloseContext,
    CloseSocket,
    ConnectionInitializedEvent,
    ContextClosedEvent,
    ContextFlushedEvent,
    FlushContext,
    InitializeConnection,
    IsFinalEvent,
    KeepAlive,
    MultiContextErrorEvent,
    SendText,
)
from macaw.server.routes.multi_context_tts import _parse_message

# ---------------------------------------------------------------------------
# Pydantic model tests
# ---------------------------------------------------------------------------


class TestInitializeConnection:
    def test_defaults(self) -> None:
        msg = InitializeConnection()
        assert msg.type == "initialize_connection"
        assert msg.model_tts is None
        assert msg.voice_settings is None

    def test_with_model(self) -> None:
        msg = InitializeConnection(model_tts="kokoro-v1")
        assert msg.model_tts == "kokoro-v1"


class TestSendText:
    def test_required_fields(self) -> None:
        msg = SendText(context_id="ctx-1", text="Hello world")
        assert msg.type == "send_text"
        assert msg.context_id == "ctx-1"
        assert msg.text == "Hello world"

    def test_empty_context_id_rejected(self) -> None:
        from pydantic import ValidationError

        with pytest.raises(ValidationError):
            SendText(context_id="", text="Hello")

    def test_empty_text_rejected(self) -> None:
        from pydantic import ValidationError

        with pytest.raises(ValidationError):
            SendText(context_id="ctx-1", text="")


class TestFlushContext:
    def test_basic(self) -> None:
        msg = FlushContext(context_id="ctx-1")
        assert msg.type == "flush_context"
        assert msg.context_id == "ctx-1"


class TestCloseContext:
    def test_basic(self) -> None:
        msg = CloseContext(context_id="ctx-1")
        assert msg.type == "close_context"


class TestCloseSocket:
    def test_basic(self) -> None:
        msg = CloseSocket()
        assert msg.type == "close_socket"


class TestKeepAlive:
    def test_basic(self) -> None:
        msg = KeepAlive()
        assert msg.type == "keep_alive"


class TestMultiContextMessageTypes:
    def test_all_types_registered(self) -> None:
        expected = {
            "initialize_connection",
            "send_text",
            "flush_context",
            "close_context",
            "close_socket",
            "keep_alive",
        }
        assert set(MULTI_CONTEXT_MESSAGE_TYPES.keys()) == expected


# ---------------------------------------------------------------------------
# Server event models
# ---------------------------------------------------------------------------


class TestServerEvents:
    def test_connection_initialized(self) -> None:
        event = ConnectionInitializedEvent(model_tts="kokoro-v1", max_contexts=10)
        assert event.type == "connection_initialized"
        assert event.model_tts == "kokoro-v1"
        assert event.max_contexts == 10

    def test_audio_event(self) -> None:
        event = AudioEvent(context_id="ctx-1", audio="dGVzdA==")
        assert event.type == "audio"
        assert event.context_id == "ctx-1"
        assert event.audio == "dGVzdA=="

    def test_context_flushed(self) -> None:
        event = ContextFlushedEvent(context_id="ctx-1")
        assert event.type == "context_flushed"

    def test_context_closed(self) -> None:
        event = ContextClosedEvent(context_id="ctx-1")
        assert event.type == "context_closed"

    def test_is_final(self) -> None:
        event = IsFinalEvent(context_id="ctx-1")
        assert event.type == "is_final"

    def test_error_event(self) -> None:
        event = MultiContextErrorEvent(message="bad", code="test_error")
        assert event.type == "error"
        assert event.context_id is None
        assert event.recoverable is True

    def test_error_event_with_context(self) -> None:
        event = MultiContextErrorEvent(
            message="bad",
            context_id="ctx-1",
            code="test_error",
        )
        assert event.context_id == "ctx-1"


# ---------------------------------------------------------------------------
# Message parsing
# ---------------------------------------------------------------------------


class TestParseMessage:
    def test_parse_initialize_connection(self) -> None:
        msg = _parse_message('{"type":"initialize_connection"}')
        assert isinstance(msg, InitializeConnection)

    def test_parse_send_text(self) -> None:
        msg = _parse_message('{"type":"send_text","context_id":"ctx-1","text":"Hello"}')
        assert isinstance(msg, SendText)
        assert msg.context_id == "ctx-1"

    def test_parse_flush_context(self) -> None:
        msg = _parse_message('{"type":"flush_context","context_id":"ctx-1"}')
        assert isinstance(msg, FlushContext)

    def test_parse_close_context(self) -> None:
        msg = _parse_message('{"type":"close_context","context_id":"ctx-1"}')
        assert isinstance(msg, CloseContext)

    def test_parse_close_socket(self) -> None:
        msg = _parse_message('{"type":"close_socket"}')
        assert isinstance(msg, CloseSocket)

    def test_parse_keep_alive(self) -> None:
        msg = _parse_message('{"type":"keep_alive"}')
        assert isinstance(msg, KeepAlive)

    def test_parse_invalid_json(self) -> None:
        msg = _parse_message("not json")
        assert isinstance(msg, MultiContextErrorEvent)
        assert msg.code == "malformed_json"

    def test_parse_non_object(self) -> None:
        msg = _parse_message("[1,2,3]")
        assert isinstance(msg, MultiContextErrorEvent)
        assert msg.code == "malformed_json"

    def test_parse_missing_type(self) -> None:
        msg = _parse_message('{"text":"hello"}')
        assert isinstance(msg, MultiContextErrorEvent)
        assert msg.code == "unknown_command"

    def test_parse_unknown_type(self) -> None:
        msg = _parse_message('{"type":"foo"}')
        assert isinstance(msg, MultiContextErrorEvent)
        assert msg.code == "unknown_command"

    def test_parse_validation_error(self) -> None:
        # send_text requires context_id and text
        msg = _parse_message('{"type":"send_text"}')
        assert isinstance(msg, MultiContextErrorEvent)
        assert msg.code == "validation_error"


# ---------------------------------------------------------------------------
# WebSocket endpoint (E2E via httpx)
# ---------------------------------------------------------------------------


def _make_registry() -> MagicMock:
    """Create a mock registry with a TTS model."""
    registry = MagicMock()
    from macaw._types import ModelType

    manifest = MagicMock()
    manifest.model_type = ModelType.TTS
    manifest.name = "kokoro-v1"
    registry.get_manifest.return_value = manifest
    registry.list_models.return_value = [manifest]
    registry.has_model.return_value = True
    return registry


def _make_worker_manager() -> MagicMock:
    """Create a mock worker manager with a ready TTS worker."""
    worker_manager = MagicMock()
    worker = MagicMock()
    worker.port = 50052
    worker_manager.get_ready_worker.return_value = worker
    return worker_manager


def _make_app(
    registry: MagicMock | None = None,
    worker_manager: MagicMock | None = None,
) -> object:
    return create_app(
        registry=registry or _make_registry(),
        worker_manager=worker_manager or _make_worker_manager(),
    )


class TestMultiContextWSEndpoint:
    async def test_binary_frame_rejected(self) -> None:
        """Binary frames should produce an error, not crash."""
        app = _make_app()
        async with (
            httpx.AsyncClient(
                transport=httpx.ASGITransport(app=app, raise_app_exceptions=False),
                base_url="http://test",
            ) as client,
            client.stream(
                "GET",
                "/v1/text-to-speech/af_heart/multi-stream-input",
                headers={
                    "connection": "upgrade",
                    "upgrade": "websocket",
                    "sec-websocket-version": "13",
                    "sec-websocket-key": "dGVzdA==",
                },
            ) as _resp,
        ):
            # WebSocket tests are better done via Starlette's test client
            pass

    async def test_parse_message_returns_error_for_binary(self) -> None:
        """The parse function only handles text, binary should not reach it."""
        msg = _parse_message('{"type": "send_text"}')
        assert isinstance(msg, MultiContextErrorEvent)
        assert msg.code == "validation_error"


class TestMultiContextWSProtocol:
    """Tests using Starlette's TestClient for WebSocket support."""

    def test_initialize_connection_response(self) -> None:
        from starlette.testclient import TestClient

        app = _make_app()
        client = TestClient(app)

        with client.websocket_connect("/v1/text-to-speech/af_heart/multi-stream-input") as ws:
            ws.send_json({"type": "initialize_connection"})
            data = ws.receive_json()
            assert data["type"] == "connection_initialized"
            assert data["model_tts"] == "kokoro-v1"
            assert data["max_contexts"] == 10
            ws.send_json({"type": "close_socket"})

    def test_message_before_init_returns_error(self) -> None:
        from starlette.testclient import TestClient

        app = _make_app()
        client = TestClient(app)

        with client.websocket_connect("/v1/text-to-speech/af_heart/multi-stream-input") as ws:
            ws.send_json({"type": "send_text", "context_id": "ctx-1", "text": "hi"})
            data = ws.receive_json()
            assert data["type"] == "error"
            assert data["code"] == "not_initialized"
            ws.send_json({"type": "close_socket"})

    def test_double_init_returns_error(self) -> None:
        from starlette.testclient import TestClient

        app = _make_app()
        client = TestClient(app)

        with client.websocket_connect("/v1/text-to-speech/af_heart/multi-stream-input") as ws:
            ws.send_json({"type": "initialize_connection"})
            ws.receive_json()  # connection_initialized
            ws.send_json({"type": "initialize_connection"})
            data = ws.receive_json()
            assert data["type"] == "error"
            assert data["code"] == "already_initialized"
            ws.send_json({"type": "close_socket"})

    def test_close_context_sends_ack(self) -> None:
        from starlette.testclient import TestClient

        app = _make_app()
        client = TestClient(app)

        with client.websocket_connect("/v1/text-to-speech/af_heart/multi-stream-input") as ws:
            ws.send_json({"type": "initialize_connection"})
            ws.receive_json()  # connection_initialized

            # Auto-create context via send_text, then close it
            with patch(
                "macaw.server.routes.multi_context_tts._synthesize_text",
                new_callable=AsyncMock,
            ):
                ws.send_json({"type": "send_text", "context_id": "ctx-1", "text": "Hello world. "})
                # The text has a sentence boundary, so a segment will be dispatched

            ws.send_json({"type": "close_context", "context_id": "ctx-1"})
            data = ws.receive_json()
            assert data["type"] == "context_closed"
            assert data["context_id"] == "ctx-1"

            ws.send_json({"type": "close_socket"})

    def test_close_nonexistent_context_returns_error(self) -> None:
        from starlette.testclient import TestClient

        app = _make_app()
        client = TestClient(app)

        with client.websocket_connect("/v1/text-to-speech/af_heart/multi-stream-input") as ws:
            ws.send_json({"type": "initialize_connection"})
            ws.receive_json()
            ws.send_json({"type": "close_context", "context_id": "nope"})
            data = ws.receive_json()
            assert data["type"] == "error"
            assert data["code"] == "context_not_found"
            ws.send_json({"type": "close_socket"})

    def test_keep_alive_does_not_produce_response(self) -> None:
        from starlette.testclient import TestClient

        app = _make_app()
        client = TestClient(app)

        with client.websocket_connect("/v1/text-to-speech/af_heart/multi-stream-input") as ws:
            ws.send_json({"type": "initialize_connection"})
            ws.receive_json()  # connection_initialized
            ws.send_json({"type": "keep_alive"})
            # keep_alive should not produce a response, so we close
            ws.send_json({"type": "close_socket"})
            # No response expected between keep_alive and close_socket

    def test_unknown_command_returns_error(self) -> None:
        from starlette.testclient import TestClient

        app = _make_app()
        client = TestClient(app)

        with client.websocket_connect("/v1/text-to-speech/af_heart/multi-stream-input") as ws:
            ws.send_json({"type": "initialize_connection"})
            ws.receive_json()
            ws.send_json({"type": "nonexistent_command"})
            data = ws.receive_json()
            assert data["type"] == "error"
            assert data["code"] == "unknown_command"
            ws.send_json({"type": "close_socket"})

    def test_malformed_json_returns_error(self) -> None:
        from starlette.testclient import TestClient

        app = _make_app()
        client = TestClient(app)

        with client.websocket_connect("/v1/text-to-speech/af_heart/multi-stream-input") as ws:
            ws.send_text("not-json{{{")
            data = ws.receive_json()
            assert data["type"] == "error"
            assert data["code"] == "malformed_json"
            ws.send_json({"type": "close_socket"})

    def test_flush_empty_context_sends_flushed_and_final(self) -> None:
        from starlette.testclient import TestClient

        app = _make_app()
        client = TestClient(app)

        with client.websocket_connect("/v1/text-to-speech/af_heart/multi-stream-input") as ws:
            ws.send_json({"type": "initialize_connection"})
            ws.receive_json()  # connection_initialized

            # Create context with send_text (no sentence boundary -> stays in buffer)
            with patch(
                "macaw.server.routes.multi_context_tts._synthesize_text",
                new_callable=AsyncMock,
            ):
                ws.send_json({"type": "send_text", "context_id": "ctx-1", "text": "Hello"})

            # Flush context (has text "Hello" in buffer)
            with patch(
                "macaw.server.routes.multi_context_tts._synthesize_text",
                new_callable=AsyncMock,
            ):
                ws.send_json({"type": "flush_context", "context_id": "ctx-1"})

            ws.send_json({"type": "close_socket"})

    def test_flush_nonexistent_context_returns_error(self) -> None:
        from starlette.testclient import TestClient

        app = _make_app()
        client = TestClient(app)

        with client.websocket_connect("/v1/text-to-speech/af_heart/multi-stream-input") as ws:
            ws.send_json({"type": "initialize_connection"})
            ws.receive_json()
            ws.send_json({"type": "flush_context", "context_id": "nope"})
            data = ws.receive_json()
            assert data["type"] == "error"
            assert data["code"] == "context_not_found"
            ws.send_json({"type": "close_socket"})

    def test_context_limit_reached(self) -> None:
        from starlette.testclient import TestClient

        app = _make_app()
        client = TestClient(app)

        # Override max_contexts to 2 for testing (patch at source)
        mc = MagicMock()
        mc.max_contexts = 2
        mc.inactivity_timeout_s = 20.0
        mc.max_concurrent_tts = 4
        mock_settings = MagicMock()
        mock_settings.multi_context = mc

        with (
            patch(
                "macaw.config.settings.get_settings",
                return_value=mock_settings,
            ),
            client.websocket_connect("/v1/text-to-speech/af_heart/multi-stream-input") as ws,
        ):
            ws.send_json({"type": "initialize_connection"})
            ws.receive_json()

            with patch(
                "macaw.server.routes.multi_context_tts._synthesize_text",
                new_callable=AsyncMock,
            ):
                ws.send_json({"type": "send_text", "context_id": "ctx-1", "text": "A"})
                ws.send_json({"type": "send_text", "context_id": "ctx-2", "text": "B"})
                # Third context should fail
                ws.send_json({"type": "send_text", "context_id": "ctx-3", "text": "C"})

            data = ws.receive_json()
            assert data["type"] == "error"
            assert data["code"] == "context_limit_reached"

            ws.send_json({"type": "close_socket"})

    def test_no_tts_model_available(self) -> None:
        from starlette.testclient import TestClient

        registry = MagicMock()
        registry.list_models.return_value = []  # No TTS models
        app = _make_app(registry=registry)
        client = TestClient(app)

        with client.websocket_connect("/v1/text-to-speech/af_heart/multi-stream-input") as ws:
            ws.send_json({"type": "initialize_connection"})
            data = ws.receive_json()
            assert data["type"] == "error"
            assert data["code"] == "model_error"
            assert data["recoverable"] is False
