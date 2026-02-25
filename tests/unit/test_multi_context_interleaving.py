"""Tests for multi-context TTS audio interleaving and concurrent synthesis."""

from __future__ import annotations

import asyncio
import contextlib
from unittest.mock import AsyncMock, MagicMock, patch

from macaw.server.app import create_app
from macaw.server.routes.multi_context_tts import (
    _dispatch_tts_for_context,
    _synthesize_text,
)
from macaw.server.ws_context_manager import ContextState, WSContextManager

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_registry() -> MagicMock:
    from macaw._types import ModelType

    registry = MagicMock()
    manifest = MagicMock()
    manifest.model_type = ModelType.TTS
    manifest.name = "kokoro-v1"
    registry.get_manifest.return_value = manifest
    registry.list_models.return_value = [manifest]
    registry.has_model.return_value = True
    return registry


def _make_worker_manager() -> MagicMock:
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


def _make_mock_ws() -> MagicMock:
    """Create a mock WebSocket that tracks sent messages."""
    from starlette.websockets import WebSocketState

    ws = AsyncMock()
    ws.client_state = WebSocketState.CONNECTED
    ws.send_json = AsyncMock()
    return ws


# ---------------------------------------------------------------------------
# _synthesize_text — isolation
# ---------------------------------------------------------------------------


class TestSynthesizeText:
    async def test_sends_audio_events(self) -> None:
        ws = _make_mock_ws()
        cancel_event = asyncio.Event()

        # Mock gRPC stream returning 2 chunks
        chunk1 = MagicMock()
        chunk1.audio_data = b"\x00\x01\x02\x03"
        chunk2 = MagicMock()
        chunk2.audio_data = b"\x04\x05\x06\x07"

        async def mock_synthesize(*_args: object, **_kwargs: object) -> AsyncMock:
            async def gen():  # type: ignore[return]
                yield chunk1
                yield chunk2

            return gen()

        with (
            patch(
                "macaw.server.routes.multi_context_tts.resolve_tts_resources",
            ) as mock_resolve,
            patch(
                "macaw.server.routes.multi_context_tts.get_or_create_tts_channel",
            ),
            patch(
                "macaw.server.routes.multi_context_tts.TTSWorkerStub",
            ) as mock_stub_cls,
        ):
            mock_resolve.return_value = (MagicMock(), MagicMock(), "localhost:50052")

            # Make the stub's Synthesize return an async iterable
            async def async_gen():  # type: ignore[return]
                yield chunk1
                yield chunk2

            mock_stub = MagicMock()
            mock_stub.Synthesize.return_value = async_gen()
            mock_stub_cls.return_value = mock_stub

            await _synthesize_text(
                ws,
                "ctx-1",
                "Hello world",
                model_tts="kokoro-v1",
                voice_id="af_heart",
                voice_settings=None,
                tts_channels={},
                registry=MagicMock(),
                worker_manager=MagicMock(),
                cancel_event=cancel_event,
            )

        # Should have sent 2 audio events
        calls = ws.send_json.call_args_list
        assert len(calls) == 2
        for call in calls:
            data = call[0][0]
            assert data["type"] == "audio"
            assert data["context_id"] == "ctx-1"
            assert "audio" in data

    async def test_cancel_stops_synthesis(self) -> None:
        ws = _make_mock_ws()
        cancel_event = asyncio.Event()
        cancel_event.set()  # Pre-cancel

        chunk = MagicMock()
        chunk.audio_data = b"\x00"

        with (
            patch(
                "macaw.server.routes.multi_context_tts.resolve_tts_resources",
            ) as mock_resolve,
            patch(
                "macaw.server.routes.multi_context_tts.get_or_create_tts_channel",
            ),
            patch(
                "macaw.server.routes.multi_context_tts.TTSWorkerStub",
            ) as mock_stub_cls,
        ):
            mock_resolve.return_value = (MagicMock(), MagicMock(), "localhost:50052")

            async def async_gen():  # type: ignore[return]
                yield chunk

            mock_stub = MagicMock()
            mock_stub.Synthesize.return_value = async_gen()
            mock_stub_cls.return_value = mock_stub

            await _synthesize_text(
                ws,
                "ctx-1",
                "Hello",
                model_tts="kokoro-v1",
                voice_id="af_heart",
                voice_settings=None,
                tts_channels={},
                registry=MagicMock(),
                worker_manager=MagicMock(),
                cancel_event=cancel_event,
            )

        # Cancelled before any chunk could be sent
        assert ws.send_json.call_count == 0

    async def test_model_not_found_sends_error(self) -> None:
        from macaw.exceptions import ModelNotFoundError

        ws = _make_mock_ws()
        cancel_event = asyncio.Event()

        with patch(
            "macaw.server.routes.multi_context_tts.resolve_tts_resources",
            side_effect=ModelNotFoundError("no-model"),
        ):
            await _synthesize_text(
                ws,
                "ctx-1",
                "Hello",
                model_tts="no-model",
                voice_id="af_heart",
                voice_settings=None,
                tts_channels={},
                registry=MagicMock(),
                worker_manager=MagicMock(),
                cancel_event=cancel_event,
            )

        data = ws.send_json.call_args[0][0]
        assert data["type"] == "error"
        assert data["context_id"] == "ctx-1"
        assert data["code"] == "model_error"

    async def test_grpc_error_sends_synthesis_error(self) -> None:
        ws = _make_mock_ws()
        cancel_event = asyncio.Event()

        with (
            patch(
                "macaw.server.routes.multi_context_tts.resolve_tts_resources",
            ) as mock_resolve,
            patch(
                "macaw.server.routes.multi_context_tts.get_or_create_tts_channel",
            ),
            patch(
                "macaw.server.routes.multi_context_tts.TTSWorkerStub",
            ) as mock_stub_cls,
        ):
            mock_resolve.return_value = (MagicMock(), MagicMock(), "localhost:50052")

            async def failing_gen():  # type: ignore[return]
                raise RuntimeError("GPU OOM")
                yield

            mock_stub = MagicMock()
            mock_stub.Synthesize.return_value = failing_gen()
            mock_stub_cls.return_value = mock_stub

            await _synthesize_text(
                ws,
                "ctx-1",
                "Hello",
                model_tts="kokoro-v1",
                voice_id="af_heart",
                voice_settings=None,
                tts_channels={},
                registry=MagicMock(),
                worker_manager=MagicMock(),
                cancel_event=cancel_event,
            )

        data = ws.send_json.call_args[0][0]
        assert data["type"] == "error"
        assert data["code"] == "synthesis_error"
        assert data["context_id"] == "ctx-1"


# ---------------------------------------------------------------------------
# _dispatch_tts_for_context — semaphore and is_final
# ---------------------------------------------------------------------------


class TestDispatchTTSForContext:
    async def test_sends_is_final_when_flushing(self) -> None:
        ws = _make_mock_ws()
        ctx_mgr = WSContextManager()
        ctx = ctx_mgr.create_context("ctx-1")
        ctx.state = ContextState.FLUSHING

        with patch(
            "macaw.server.routes.multi_context_tts._synthesize_text",
            new_callable=AsyncMock,
        ):
            await _dispatch_tts_for_context(
                ws,
                ctx_mgr,
                "ctx-1",
                "Hello",
                model_tts="kokoro-v1",
                voice_id="af_heart",
                voice_settings=None,
                tts_channels={},
                registry=MagicMock(),
                worker_manager=MagicMock(),
                tts_semaphore=asyncio.Semaphore(4),
            )

        # Check is_final was sent
        calls = ws.send_json.call_args_list
        is_final_calls = [c for c in calls if c[0][0].get("type") == "is_final"]
        assert len(is_final_calls) == 1
        assert is_final_calls[0][0][0]["context_id"] == "ctx-1"

    async def test_context_closed_after_is_final(self) -> None:
        ws = _make_mock_ws()
        ctx_mgr = WSContextManager()
        ctx = ctx_mgr.create_context("ctx-1")
        ctx.state = ContextState.FLUSHING

        with patch(
            "macaw.server.routes.multi_context_tts._synthesize_text",
            new_callable=AsyncMock,
        ):
            await _dispatch_tts_for_context(
                ws,
                ctx_mgr,
                "ctx-1",
                "Hello",
                model_tts="kokoro-v1",
                voice_id="af_heart",
                voice_settings=None,
                tts_channels={},
                registry=MagicMock(),
                worker_manager=MagicMock(),
                tts_semaphore=asyncio.Semaphore(4),
            )

        # Context should be closed after is_final
        assert ctx.state == ContextState.CLOSED

    async def test_skips_closed_context(self) -> None:
        ws = _make_mock_ws()
        ctx_mgr = WSContextManager()
        ctx_mgr.create_context("ctx-1")
        ctx_mgr.close_context("ctx-1")

        with patch(
            "macaw.server.routes.multi_context_tts._synthesize_text",
            new_callable=AsyncMock,
        ) as mock_synth:
            await _dispatch_tts_for_context(
                ws,
                ctx_mgr,
                "ctx-1",
                "Hello",
                model_tts="kokoro-v1",
                voice_id="af_heart",
                voice_settings=None,
                tts_channels={},
                registry=MagicMock(),
                worker_manager=MagicMock(),
                tts_semaphore=asyncio.Semaphore(4),
            )

        # Should not have called synthesize for a closed context
        mock_synth.assert_not_called()

    async def test_semaphore_bounds_concurrency(self) -> None:
        """Verify the semaphore is acquired during synthesis."""
        ws = _make_mock_ws()
        ctx_mgr = WSContextManager()
        ctx_mgr.create_context("ctx-1")

        semaphore = asyncio.Semaphore(1)

        with patch(
            "macaw.server.routes.multi_context_tts._synthesize_text",
            new_callable=AsyncMock,
        ):
            # First dispatch should acquire the semaphore
            await _dispatch_tts_for_context(
                ws,
                ctx_mgr,
                "ctx-1",
                "Hello",
                model_tts="kokoro-v1",
                voice_id="af_heart",
                voice_settings=None,
                tts_channels={},
                registry=MagicMock(),
                worker_manager=MagicMock(),
                tts_semaphore=semaphore,
            )

        # After completion, semaphore should be released
        assert not semaphore.locked()


# ---------------------------------------------------------------------------
# Context isolation — errors in one don't affect others
# ---------------------------------------------------------------------------


class TestContextIsolation:
    async def test_error_in_one_context_does_not_affect_others(self) -> None:
        """If synthesis fails for ctx-1, ctx-2 should still work."""
        ws = _make_mock_ws()
        ctx_mgr = WSContextManager()
        ctx_mgr.create_context("ctx-1")
        ctx_mgr.create_context("ctx-2")

        call_count = {"ctx-1": 0, "ctx-2": 0}

        async def mock_synthesize(
            _ws: object,
            context_id: str,
            _text: str,
            **_kwargs: object,
        ) -> None:
            call_count[context_id] += 1
            if context_id == "ctx-1":
                raise RuntimeError("GPU OOM")

        with patch(
            "macaw.server.routes.multi_context_tts._synthesize_text",
            side_effect=mock_synthesize,
        ):
            # Dispatch both contexts
            semaphore = asyncio.Semaphore(4)

            # ctx-1 will fail
            with contextlib.suppress(RuntimeError):
                await _dispatch_tts_for_context(
                    ws,
                    ctx_mgr,
                    "ctx-1",
                    "Hello",
                    model_tts="kokoro-v1",
                    voice_id="af_heart",
                    voice_settings=None,
                    tts_channels={},
                    registry=MagicMock(),
                    worker_manager=MagicMock(),
                    tts_semaphore=semaphore,
                )

            # ctx-2 should still work
            await _dispatch_tts_for_context(
                ws,
                ctx_mgr,
                "ctx-2",
                "World",
                model_tts="kokoro-v1",
                voice_id="af_heart",
                voice_settings=None,
                tts_channels={},
                registry=MagicMock(),
                worker_manager=MagicMock(),
                tts_semaphore=semaphore,
            )

        assert call_count["ctx-1"] == 1
        assert call_count["ctx-2"] == 1

        # ctx-2 should still be active
        assert ctx_mgr.get_active_context("ctx-2") is not None


# ---------------------------------------------------------------------------
# End-to-end multi-context flow via TestClient
# ---------------------------------------------------------------------------


class TestMultiContextE2E:
    def test_send_text_auto_creates_context(self) -> None:
        from starlette.testclient import TestClient

        app = _make_app()
        client = TestClient(app)

        with client.websocket_connect("/v1/text-to-speech/af_heart/multi-stream-input") as ws:
            ws.send_json({"type": "initialize_connection"})
            init_data = ws.receive_json()
            assert init_data["type"] == "connection_initialized"

            # send_text auto-creates context
            with patch(
                "macaw.server.routes.multi_context_tts._synthesize_text",
                new_callable=AsyncMock,
            ):
                ws.send_json({"type": "send_text", "context_id": "auto-ctx", "text": "Test."})

            # Close and verify
            ws.send_json({"type": "close_context", "context_id": "auto-ctx"})
            data = ws.receive_json()
            assert data["type"] == "context_closed"
            ws.send_json({"type": "close_socket"})

    def test_multiple_contexts_independent_close(self) -> None:
        from starlette.testclient import TestClient

        app = _make_app()
        client = TestClient(app)

        with client.websocket_connect("/v1/text-to-speech/af_heart/multi-stream-input") as ws:
            ws.send_json({"type": "initialize_connection"})
            ws.receive_json()

            with patch(
                "macaw.server.routes.multi_context_tts._synthesize_text",
                new_callable=AsyncMock,
            ):
                ws.send_json({"type": "send_text", "context_id": "ctx-A", "text": "A"})
                ws.send_json({"type": "send_text", "context_id": "ctx-B", "text": "B"})

            # Close only ctx-A
            ws.send_json({"type": "close_context", "context_id": "ctx-A"})
            data = ws.receive_json()
            assert data["type"] == "context_closed"
            assert data["context_id"] == "ctx-A"

            # ctx-B should still be closeable
            ws.send_json({"type": "close_context", "context_id": "ctx-B"})
            data = ws.receive_json()
            assert data["type"] == "context_closed"
            assert data["context_id"] == "ctx-B"

            ws.send_json({"type": "close_socket"})
