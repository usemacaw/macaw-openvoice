"""Tests for MP3 codec encoder.

Validates:
- Codec name property
- Encode calls lameenc (mocked)
- Flush calls lameenc flush (mocked)
- Lazy init (encoder created on first encode)
- CodecUnavailableError when lameenc is not installed
- Cached error on repeated calls
"""

from __future__ import annotations

import contextlib
import sys
from typing import TYPE_CHECKING
from unittest.mock import MagicMock, patch

if TYPE_CHECKING:
    from collections.abc import Generator

import pytest

from macaw.exceptions import CodecUnavailableError


class TestMp3EncoderProperties:
    def test_codec_name(self) -> None:
        with _mock_lameenc():
            from macaw.codec.mp3 import Mp3Encoder

            encoder = Mp3Encoder(sample_rate=24000)
            assert encoder.codec_name == "mp3"


class TestMp3EncoderLazyInit:
    def test_encoder_not_created_on_init(self) -> None:
        with _mock_lameenc():
            from macaw.codec.mp3 import Mp3Encoder

            encoder = Mp3Encoder(sample_rate=24000)
            assert encoder._encoder is None

    def test_encoder_created_on_first_encode(self) -> None:
        with _mock_lameenc() as mock_lameenc:
            from macaw.codec.mp3 import Mp3Encoder

            encoder = Mp3Encoder(sample_rate=24000)
            encoder.encode(b"\x00\x00" * 160)
            mock_lameenc.Encoder.assert_called_once()

    def test_raises_when_lameenc_unavailable(self) -> None:
        with _mock_lameenc_unavailable():
            from macaw.codec.mp3 import Mp3Encoder

            encoder = Mp3Encoder(sample_rate=24000)
            with pytest.raises(CodecUnavailableError, match="mp3"):
                encoder.encode(b"\x00\x00" * 160)

    def test_cached_error_on_repeated_calls(self) -> None:
        with _mock_lameenc_unavailable():
            from macaw.codec.mp3 import Mp3Encoder

            encoder = Mp3Encoder(sample_rate=24000)
            with pytest.raises(CodecUnavailableError):
                encoder.encode(b"\x00\x00" * 160)
            # Second call should raise immediately (cached error)
            with pytest.raises(CodecUnavailableError, match="mp3"):
                encoder.encode(b"\x00\x00" * 160)


class TestMp3EncoderEncode:
    def test_encode_returns_bytes(self) -> None:
        with _mock_lameenc() as mock_lameenc:
            mock_lameenc.Encoder.return_value.encode.return_value = b"\xff\xfb"
            from macaw.codec.mp3 import Mp3Encoder

            encoder = Mp3Encoder(sample_rate=24000)
            result = encoder.encode(b"\x00\x00" * 160)
            assert isinstance(result, bytes)

    def test_encode_passes_pcm_to_lame(self) -> None:
        with _mock_lameenc() as mock_lameenc:
            mock_encoder = mock_lameenc.Encoder.return_value
            mock_encoder.encode.return_value = b"\xff\xfb"
            from macaw.codec.mp3 import Mp3Encoder

            pcm = b"\x00\x01" * 160
            encoder = Mp3Encoder(sample_rate=24000)
            encoder.encode(pcm)
            mock_encoder.encode.assert_called_once_with(pcm)


class TestMp3EncoderFlush:
    def test_flush_without_encode_returns_empty(self) -> None:
        with _mock_lameenc():
            from macaw.codec.mp3 import Mp3Encoder

            encoder = Mp3Encoder(sample_rate=24000)
            # Never called encode, so encoder is None
            assert encoder.flush() == b""

    def test_flush_after_encode(self) -> None:
        with _mock_lameenc() as mock_lameenc:
            mock_encoder = mock_lameenc.Encoder.return_value
            mock_encoder.encode.return_value = b"\xff\xfb"
            mock_encoder.flush.return_value = b"\x00\x00"
            from macaw.codec.mp3 import Mp3Encoder

            encoder = Mp3Encoder(sample_rate=24000)
            encoder.encode(b"\x00\x00" * 160)
            result = encoder.flush()
            assert result == b"\x00\x00"
            mock_encoder.flush.assert_called_once()


class TestMp3EncoderConfig:
    def test_bitrate_passed_to_encoder(self) -> None:
        with _mock_lameenc() as mock_lameenc:
            mock_encoder = mock_lameenc.Encoder.return_value
            mock_encoder.encode.return_value = b""
            from macaw.codec.mp3 import Mp3Encoder

            encoder = Mp3Encoder(sample_rate=44100, bitrate=192)
            encoder.encode(b"\x00\x00" * 160)
            mock_encoder.set_in_sample_rate.assert_called_once_with(44100)
            mock_encoder.set_bit_rate.assert_called_once_with(192)

    def test_quality_passed_to_encoder(self) -> None:
        with _mock_lameenc() as mock_lameenc:
            mock_encoder = mock_lameenc.Encoder.return_value
            mock_encoder.encode.return_value = b""
            from macaw.codec.mp3 import Mp3Encoder

            encoder = Mp3Encoder(sample_rate=24000, quality=5)
            encoder.encode(b"\x00\x00" * 160)
            mock_encoder.set_quality.assert_called_once_with(5)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


@contextlib.contextmanager
def _mock_lameenc() -> Generator[MagicMock, None, None]:
    """Mock lameenc module for tests that don't need the real library."""
    mock_module = MagicMock()
    # Remove cached module to force re-import with mock
    _cleanup_keys = [k for k in sys.modules if k.startswith("macaw.codec.mp3")]
    for k in _cleanup_keys:
        del sys.modules[k]

    with patch.dict(sys.modules, {"lameenc": mock_module}):
        # Re-import so the import guard picks up the mock
        yield mock_module


@contextlib.contextmanager
def _mock_lameenc_unavailable() -> Generator[None, None, None]:
    """Simulate lameenc not installed."""
    _cleanup_keys = [k for k in sys.modules if k.startswith("macaw.codec.mp3")]
    for k in _cleanup_keys:
        del sys.modules[k]

    with patch.dict(sys.modules, {"lameenc": None}):
        yield
