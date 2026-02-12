"""Lightweight uvicorn shim used in test environments.

This module provides a minimal subset of the uvicorn API the test
suite expects (Config and Server). It intentionally avoids bringing in
the real uvicorn dependency so integration tests can run in CI
environments where uvicorn isn't installed.

The shim is simple: Server.serve() is an async coroutine that sets
`started = True` and waits until `should_exit` becomes truthy.
force build release
"""

from __future__ import annotations

import asyncio
from typing import Any


class Config:
    def __init__(self, app: Any, host: str = "127.0.0.1", port: int = 8000, **kwargs: Any) -> None:
        self.app = app
        self.host = host
        self.port = port
        # accept and ignore other kwargs (log_level, ws, etc.) so callers
        # using the real uvicorn API don't break when the shim is used.
        for k, v in kwargs.items():
            setattr(self, k, v)


class Server:
    def __init__(self, config: Config) -> None:
        self.config = config
        self.started = False
        self.should_exit = False

    async def serve(self) -> None:
        """Start a simple loop that runs until `should_exit` is set.

        Real network serving is out of scope for tests that only need a
        running server flag; this implementation mirrors the behaviour
        needed by the test-suite.
        """
        self.started = True
        try:
            # wait until someone sets should_exit = True
            while not self.should_exit:
                await asyncio.sleep(0.1)
        finally:
            self.started = False
