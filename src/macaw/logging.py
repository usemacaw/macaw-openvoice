"""Structured logging for Macaw OpenVoice.

Uses structlog with stdlib logging as the backend. Two formats:
- console: human-readable for development (default)
- json: structured for production
"""

from __future__ import annotations

import logging
import os

import structlog

_configured = False


def configure_logging(
    log_format: str | None = None,
    level: str | None = None,
) -> None:
    """Configure structured logging for the runtime.

    Idempotent — subsequent calls are ignored.

    Args:
        log_format: "json" or "console". Default via MACAW_LOG_FORMAT env or "console".
        level: Log level (DEBUG, INFO, WARNING, ERROR). Default via MACAW_LOG_LEVEL env or "INFO".
    """
    global _configured
    if _configured:
        return

    resolved_format = log_format or os.environ.get("MACAW_LOG_FORMAT", "console")
    resolved_level = level or os.environ.get("MACAW_LOG_LEVEL", "INFO")

    shared_processors: list[structlog.types.Processor] = [
        structlog.stdlib.add_log_level,
        structlog.processors.TimeStamper(fmt="iso"),
        structlog.processors.format_exc_info,
    ]

    if resolved_format == "json":
        renderer: structlog.types.Processor = structlog.processors.JSONRenderer()
    else:
        renderer = structlog.dev.ConsoleRenderer()

    structlog.configure(
        processors=[
            *shared_processors,
            structlog.stdlib.ProcessorFormatter.wrap_for_formatter,
        ],
        logger_factory=structlog.stdlib.LoggerFactory(),
        wrapper_class=structlog.stdlib.BoundLogger,
        cache_logger_on_first_use=True,
    )

    formatter = structlog.stdlib.ProcessorFormatter(
        processors=[
            structlog.stdlib.ProcessorFormatter.remove_processors_meta,
            renderer,
        ],
    )

    handler = logging.StreamHandler()
    handler.setFormatter(formatter)

    root_logger = logging.getLogger()
    root_logger.handlers.clear()
    root_logger.addHandler(handler)
    root_logger.setLevel(getattr(logging, resolved_level.upper(), logging.INFO))

    _configured = True


def get_logger(component: str) -> structlog.stdlib.BoundLogger:
    """Return a logger with component context.

    Args:
        component: Component name (e.g., "worker.stt", "session_manager").

    Returns:
        BoundLogger with the component field bound.
    """
    configure_logging()
    return structlog.get_logger().bind(component=component)  # type: ignore[no-any-return]
