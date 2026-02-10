"""Testes para o modulo de logging estruturado."""

from __future__ import annotations

import json

import structlog

import macaw.logging as macaw_logging


def _reset_logging() -> None:
    """Reset do estado global de logging para isolamento entre testes."""
    macaw_logging._configured = False
    structlog.reset_defaults()


class TestGetLogger:
    def setup_method(self) -> None:
        _reset_logging()

    def teardown_method(self) -> None:
        _reset_logging()

    def test_get_logger_returns_bound_logger(self) -> None:
        logger = macaw_logging.get_logger("test")
        assert isinstance(logger, structlog.stdlib.BoundLogger)

    def test_get_logger_binds_component(self) -> None:
        logger = macaw_logging.get_logger("worker.stt")
        # Access the bound context via _context
        context = logger._context  # type: ignore[attr-defined]
        assert context.get("component") == "worker.stt"

    def test_bind_adds_context(self) -> None:
        logger = macaw_logging.get_logger("scheduler")
        bound = logger.bind(request_id="req-123")
        context = bound._context  # type: ignore[attr-defined]
        assert context.get("component") == "scheduler"
        assert context.get("request_id") == "req-123"


class TestConfigureLogging:
    def setup_method(self) -> None:
        _reset_logging()

    def teardown_method(self) -> None:
        _reset_logging()

    def test_configure_idempotent(self) -> None:
        macaw_logging.configure_logging(log_format="console", level="DEBUG")
        assert macaw_logging._configured is True
        # Second call should not raise
        macaw_logging.configure_logging(log_format="json", level="ERROR")
        # Still configured from first call
        assert macaw_logging._configured is True

    def test_configure_console_format_no_exception(self) -> None:
        macaw_logging.configure_logging(log_format="console", level="INFO")
        logger = macaw_logging.get_logger("test")
        # Should not raise
        logger.info("test message")

    def test_configure_json_format_has_required_fields(self, capsys: object) -> None:
        macaw_logging.configure_logging(log_format="json", level="DEBUG")
        logger = macaw_logging.get_logger("test_component")
        logger.info("test event")

        import sys

        # Flush stderr where logging writes
        sys.stderr.flush()
        # JSON renderer outputs to stderr via logging
        # Capture by reading the handler output directly
        import logging

        root = logging.getLogger()
        assert len(root.handlers) > 0

        # Verify JSON output structure by capturing via a custom handler
        captured_records: list[str] = []

        class CaptureHandler(logging.Handler):
            def emit(self, record: logging.LogRecord) -> None:
                captured_records.append(self.format(record))

        capture_handler = CaptureHandler()
        capture_handler.setFormatter(root.handlers[0].formatter)
        root.addHandler(capture_handler)

        logger.info("structured event", key="value")

        assert len(captured_records) > 0
        parsed = json.loads(captured_records[-1])
        assert "event" in parsed
        assert "level" in parsed
        assert "timestamp" in parsed
        assert parsed["component"] == "test_component"
        assert parsed["event"] == "structured event"
        assert parsed["key"] == "value"

        root.removeHandler(capture_handler)
