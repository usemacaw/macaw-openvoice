"""Shared test helpers for streaming session tests.

These helpers were extracted from 23+ test files where they were
duplicated identically. Import from here instead of copy-pasting.

Usage:
    from tests.helpers import (
        FRAME_SIZE,
        SAMPLE_RATE,
        AsyncIterFromList,
        make_float32_frame,
        make_preprocessor_mock,
        make_raw_bytes,
        make_vad_mock,
    )
"""

from __future__ import annotations

from unittest.mock import Mock

import numpy as np

# Standard frame size: 1024 samples at 16kHz = 64ms
FRAME_SIZE = 1024
SAMPLE_RATE = 16000


def make_raw_bytes(n_samples: int = FRAME_SIZE) -> bytes:
    """Generate PCM int16 bytes (zeros) with n_samples samples."""
    return np.zeros(n_samples, dtype=np.int16).tobytes()


def make_float32_frame(n_samples: int = FRAME_SIZE) -> np.ndarray:
    """Generate float32 frame (zeros) for preprocessor mock."""
    return np.zeros(n_samples, dtype=np.float32)


def make_preprocessor_mock() -> Mock:
    """Create a StreamingPreprocessor mock."""
    mock = Mock()
    mock.process_frame.return_value = make_float32_frame()
    return mock


def make_vad_mock(*, is_speaking: bool = False) -> Mock:
    """Create a VADDetector mock."""
    mock = Mock()
    mock.process_frame.return_value = None
    mock.is_speaking = is_speaking
    mock.reset.return_value = None
    return mock


class AsyncIterFromList:
    """Async iterator that yields items from a list.

    Needed because AsyncMock.return_value does not support async generators
    directly. This class implements __aiter__ and __anext__ for use
    with ``async for``.

    If an item is an Exception instance, it is raised instead of yielded.
    """

    def __init__(self, items: list) -> None:
        self._items = list(items)
        self._index = 0

    def __aiter__(self):
        return self

    async def __anext__(self):
        if self._index >= len(self._items):
            raise StopAsyncIteration
        item = self._items[self._index]
        self._index += 1
        if isinstance(item, Exception):
            raise item
        return item
