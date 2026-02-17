"""Unit tests for macaw.codec — CodecEncoder ABC and OpusEncoder."""

from __future__ import annotations

import sys
from unittest.mock import MagicMock, patch

import pytest

from macaw.codec import CodecEncoder, create_encoder
from macaw.codec.opus import OpusEncoder


def _make_encoder_with_mock(
    sample_rate: int = 24000,
    channels: int = 1,
    encode_return: bytes | list[bytes] = b"encoded",
) -> tuple[OpusEncoder, MagicMock]:
    """Create an OpusEncoder with a pre-injected mock opuslib encoder.

    Bypasses lazy init by directly setting internal state, which is
    the correct approach when opuslib is not installed.
    """
    mock_enc = MagicMock()
    if isinstance(encode_return, list):
        mock_enc.encode.side_effect = encode_return
    else:
        mock_enc.encode.return_value = encode_return

    encoder = OpusEncoder(sample_rate=sample_rate, channels=channels)
    encoder._encoder = mock_enc
    encoder._available = True

    return encoder, mock_enc


# ---------------------------------------------------------------------------
# create_encoder factory
# ---------------------------------------------------------------------------


class TestCreateEncoder:
    def test_create_encoder_opus(self) -> None:
        """Factory returns OpusEncoder for 'opus'."""
        enc = create_encoder("opus")
        assert isinstance(enc, OpusEncoder)

    def test_create_encoder_unknown_returns_none(self) -> None:
        """Factory returns None for unrecognized codec name."""
        assert create_encoder("mp3") is None
        assert create_encoder("") is None

    def test_create_encoder_custom_params(self) -> None:
        """Factory passes sample_rate and bitrate to OpusEncoder."""
        enc = create_encoder("opus", sample_rate=48000, bitrate=128000)
        assert isinstance(enc, OpusEncoder)
        assert enc._sample_rate == 48000
        assert enc._bitrate == 128000


# ---------------------------------------------------------------------------
# OpusEncoder — codec_name
# ---------------------------------------------------------------------------


class TestOpusEncoderCodecName:
    def test_opus_encoder_codec_name(self) -> None:
        enc = OpusEncoder()
        assert enc.codec_name == "opus"


# ---------------------------------------------------------------------------
# OpusEncoder — frame size calculation
# ---------------------------------------------------------------------------


class TestOpusEncoderFrameSize:
    def test_frame_size_24khz_mono(self) -> None:
        """At 24kHz mono, 20ms = 480 samples = 960 bytes."""
        enc = OpusEncoder(sample_rate=24000, channels=1)
        assert enc._frame_samples == 480
        assert enc._frame_bytes == 960

    def test_frame_size_48khz_mono(self) -> None:
        """At 48kHz mono, 20ms = 960 samples = 1920 bytes."""
        enc = OpusEncoder(sample_rate=48000, channels=1)
        assert enc._frame_samples == 960
        assert enc._frame_bytes == 1920

    def test_frame_size_24khz_stereo(self) -> None:
        """At 24kHz stereo, 20ms = 480 samples = 1920 bytes."""
        enc = OpusEncoder(sample_rate=24000, channels=2)
        assert enc._frame_samples == 480
        assert enc._frame_bytes == 1920


# ---------------------------------------------------------------------------
# OpusEncoder — encode with mock encoder
# ---------------------------------------------------------------------------


class TestOpusEncoderEncode:
    def test_encode_buffers_small_input(self) -> None:
        """Input smaller than one frame is buffered; encode returns empty."""
        encoder, mock_enc = _make_encoder_with_mock()

        small_data = b"\x00" * 100  # < 960 bytes frame
        result = encoder.encode(small_data)

        assert result == b""
        mock_enc.encode.assert_not_called()
        assert len(encoder._buffer) == 100

    def test_encode_full_frame(self) -> None:
        """Exactly one frame produces encoded output."""
        encoder, mock_enc = _make_encoder_with_mock(encode_return=b"encoded_opus_data")

        frame_data = b"\x00" * 960  # Exactly one frame
        result = encoder.encode(frame_data)

        assert result == b"encoded_opus_data"
        mock_enc.encode.assert_called_once_with(frame_data, 480)
        assert len(encoder._buffer) == 0

    def test_encode_multiple_frames(self) -> None:
        """Input spanning multiple frames produces concatenated output."""
        encoder, mock_enc = _make_encoder_with_mock(
            encode_return=[b"frame1", b"frame2"],
        )

        # 2 full frames + 100 bytes partial
        data = b"\x00" * (960 * 2 + 100)
        result = encoder.encode(data)

        assert result == b"frame1frame2"
        assert mock_enc.encode.call_count == 2
        assert len(encoder._buffer) == 100

    def test_encode_accumulates_across_calls(self) -> None:
        """Partial frames accumulate across multiple encode() calls."""
        encoder, _mock_enc = _make_encoder_with_mock(encode_return=b"encoded")

        # First call: 500 bytes (< 960 frame)
        result1 = encoder.encode(b"\x00" * 500)
        assert result1 == b""

        # Second call: 500 bytes -> total 1000 = 1 frame + 40 remainder
        result2 = encoder.encode(b"\x00" * 500)
        assert result2 == b"encoded"
        assert len(encoder._buffer) == 40


# ---------------------------------------------------------------------------
# OpusEncoder — flush
# ---------------------------------------------------------------------------


class TestOpusEncoderFlush:
    def test_flush_partial_frame(self) -> None:
        """Flush zero-pads remaining buffer and encodes."""
        encoder, mock_enc = _make_encoder_with_mock(encode_return=b"flushed_opus")

        # Buffer some data
        encoder.encode(b"\x01" * 100)
        result = encoder.flush()

        assert result == b"flushed_opus"
        # The padded frame should be 960 bytes total
        call_args = mock_enc.encode.call_args
        padded_data = call_args[0][0]
        assert len(padded_data) == 960
        # First 100 bytes are our data, rest is zeros
        assert padded_data[:100] == b"\x01" * 100
        assert padded_data[100:] == b"\x00" * 860

    def test_flush_empty_buffer(self) -> None:
        """Flush with empty buffer returns empty bytes."""
        encoder, _ = _make_encoder_with_mock()

        result = encoder.flush()
        assert result == b""


# ---------------------------------------------------------------------------
# OpusEncoder — passthrough when unavailable
# ---------------------------------------------------------------------------


class TestOpusEncoderPassthrough:
    def test_passthrough_when_opuslib_missing(self) -> None:
        """Returns raw PCM when opuslib is not installed."""
        encoder = OpusEncoder()
        # Simulate import failure
        encoder._available = False

        pcm = b"\x00\x01" * 500
        result = encoder.encode(pcm)
        assert result == pcm

    def test_flush_passthrough_when_unavailable(self) -> None:
        """Flush returns raw buffer content when encoder is unavailable."""
        encoder = OpusEncoder()
        encoder._available = False

        # Manually put data in buffer
        encoder._buffer.extend(b"\xab" * 50)
        result = encoder.flush()
        assert result == b"\xab" * 50
        assert len(encoder._buffer) == 0


# ---------------------------------------------------------------------------
# OpusEncoder — lazy initialization
# ---------------------------------------------------------------------------


class TestOpusEncoderLazyInit:
    def test_ensure_encoder_marks_unavailable_when_import_fails(self) -> None:
        """When opuslib is not installed, _ensure_encoder sets _available=False."""
        encoder = OpusEncoder(sample_rate=24000)
        assert encoder._available is None

        # opuslib is not installed in test env — import will fail
        result = encoder._ensure_encoder()

        assert result is False
        assert encoder._available is False

    def test_ensure_encoder_caches_unavailable(self) -> None:
        """Second call to _ensure_encoder returns cached False without retrying."""
        encoder = OpusEncoder()
        encoder._ensure_encoder()
        assert encoder._available is False

        # Second call should return immediately
        result = encoder._ensure_encoder()
        assert result is False

    def test_ensure_encoder_caches_available(self) -> None:
        """When _available is True, _ensure_encoder returns True immediately."""
        encoder = OpusEncoder()
        encoder._available = True  # Pre-set

        result = encoder._ensure_encoder()
        assert result is True

    def test_ensure_encoder_with_mock_opuslib(self) -> None:
        """When opuslib is importable, _ensure_encoder creates the encoder."""
        mock_opuslib = MagicMock()
        mock_enc = MagicMock()
        mock_opuslib.Encoder.return_value = mock_enc

        with patch.dict(sys.modules, {"opuslib": mock_opuslib}):
            encoder = OpusEncoder(sample_rate=24000)
            result = encoder._ensure_encoder()

            assert result is True
            assert encoder._available is True
            assert encoder._encoder is mock_enc
            mock_opuslib.Encoder.assert_called_once()


# ---------------------------------------------------------------------------
# CodecEncoder ABC
# ---------------------------------------------------------------------------


class TestCodecEncoderABC:
    def test_cannot_instantiate_abc(self) -> None:
        """CodecEncoder cannot be instantiated directly."""
        with pytest.raises(TypeError):
            CodecEncoder()  # type: ignore[abstract]
