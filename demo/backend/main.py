"""Entry point to run the demo backend."""

from __future__ import annotations

import uvicorn

from macaw.logging import configure_logging

from .config import DemoConfig


def run() -> None:
    settings = DemoConfig()
    configure_logging(log_format="console", level="INFO")
    uvicorn.run(
        "examples.demo.backend.app:app",
        host=settings.host,
        port=settings.port,
        reload=False,
        log_level="info",
    )


if __name__ == "__main__":
    run()
