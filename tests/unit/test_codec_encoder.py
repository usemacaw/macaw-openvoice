"""Unit tests for macaw.codec — CodecEncoder ABC and OpusEncoder."""

from __future__ import annotations

import sys
from unittest.mock import MagicMock, patch

import pytest

from macaw.codec import CodecEncoder, create_encoder, is_codec_available
from macaw.codec.opus import OpusEncoder
from macaw.exceptions import CodecUnavailableError


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
    encoder._init_attempted = True

    return encoder, mock_enc


# ---------------------------------------------------------------------------
# create_encoder factory
# ---------------------------------------------------------------------------


class TestCreateEncoder:
    def test_create_encoder_opus(self) -> None:
        """Factory returns OpusEncoder for 'opus'."""
        enc = create_encoder("opus")
        assert isinstance(enc, OpusEncoder)

    def test_create_encoder_unknown_raises_codec_unavailable(self) -> None:
        """Factory raises CodecUnavailableError for unrecognized codec name."""
        with pytest.raises(CodecUnavailableError, match="unknown codec"):
            create_encoder("flac")

    def test_create_encoder_empty_name_raises_codec_unavailable(self) -> None:
        """Factory raises CodecUnavailableError for empty codec name."""
        with pytest.raises(CodecUnavailableError, match="unknown codec"):
            create_encoder("")

    def test_create_encoder_custom_params(self) -> None:
        """Factory passes sample_rate and bitrate to OpusEncoder."""
        enc = create_encoder("opus", sample_rate=48000, bitrate=128000)
        assert isinstance(enc, OpusEncoder)
        assert enc._sample_rate == 48000
        assert enc._bitrate == 128000

    def test_create_encoder_mulaw(self) -> None:
        """Factory returns MuLawEncoder for 'mulaw'."""
        from macaw.codec.g711 import MuLawEncoder

        enc = create_encoder("mulaw")
        assert isinstance(enc, MuLawEncoder)

    def test_create_encoder_alaw(self) -> None:
        """Factory returns ALawEncoder for 'alaw'."""
        from macaw.codec.g711 import ALawEncoder

        enc = create_encoder("alaw")
        assert isinstance(enc, ALawEncoder)


# ---------------------------------------------------------------------------
# is_codec_available
# ---------------------------------------------------------------------------


class TestIsCodecAvailable:
    def test_is_codec_available_opus_installed(self) -> None:
        """Returns True when opuslib is importable."""
        mock_opuslib = MagicMock()
        with patch.dict(sys.modules, {"opuslib": mock_opuslib}):
            assert is_codec_available("opus") is True

    def test_is_codec_available_opus_missing(self) -> None:
        """Returns False when opuslib import fails."""
        with patch.dict(sys.modules, {"opuslib": None}):
            assert is_codec_available("opus") is False

    def test_is_codec_available_unknown_codec(self) -> None:
        """Returns False for unknown codec names."""
        assert is_codec_available("flac") is False
        assert is_codec_available("") is False

    def test_is_codec_available_mulaw(self) -> None:
        """G.711 mu-law is always available (pure numpy)."""
        assert is_codec_available("mulaw") is True

    def test_is_codec_available_alaw(self) -> None:
        """G.711 A-law is always available (pure numpy)."""
        assert is_codec_available("alaw") is True


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

    def test_flush_empty_buffer_returns_empty_bytes(self) -> None:
        """Flush with empty buffer returns empty bytes without checking encoder."""
        encoder = OpusEncoder()
        # Do not set _init_attempted or _encoder — empty buffer should return
        # immediately without triggering _ensure_encoder.
        result = encoder.flush()
        assert result == b""
        # Verify encoder was NOT initialized (no attempt was made)
        assert encoder._init_attempted is False

    def test_flush_empty_buffer_with_mock(self) -> None:
        """Flush with empty buffer returns empty bytes (mock encoder)."""
        encoder, _ = _make_encoder_with_mock()

        result = encoder.flush()
        assert result == b""


# ---------------------------------------------------------------------------
# OpusEncoder — raises when unavailable
# ---------------------------------------------------------------------------


class TestOpusEncoderFailFast:
    def test_encode_raises_when_opuslib_missing(self) -> None:
        """Raises CodecUnavailableError when opuslib is not installed."""
        encoder = OpusEncoder()

        pcm = b"\x00\x01" * 500
        with pytest.raises(CodecUnavailableError, match="opuslib not installed"):
            encoder.encode(pcm)

    def test_flush_raises_when_unavailable_and_buffer_has_data(self) -> None:
        """Flush raises CodecUnavailableError when encoder is unavailable and buffer has data."""
        encoder = OpusEncoder()
        # Manually put data in buffer
        encoder._buffer.extend(b"\xab" * 50)

        with pytest.raises(CodecUnavailableError, match="opuslib not installed"):
            encoder.flush()


# ---------------------------------------------------------------------------
# OpusEncoder — lazy initialization
# ---------------------------------------------------------------------------


class TestOpusEncoderLazyInit:
    def test_ensure_encoder_raises_when_import_fails(self) -> None:
        """When opuslib is not installed, _ensure_encoder raises CodecUnavailableError."""
        encoder = OpusEncoder(sample_rate=24000)
        assert encoder._init_attempted is False

        # opuslib is not installed in test env — import will fail
        with pytest.raises(CodecUnavailableError, match="opuslib not installed"):
            encoder._ensure_encoder()

        assert encoder._init_error is not None

    def test_ensure_encoder_caches_failure(self) -> None:
        """Second call to _ensure_encoder raises immediately with cached error."""
        encoder = OpusEncoder()

        # First call triggers import failure
        with pytest.raises(CodecUnavailableError):
            encoder._ensure_encoder()

        assert encoder._init_error is not None
        cached_error = encoder._init_error

        # Second call should raise immediately with same error
        with pytest.raises(CodecUnavailableError, match=cached_error):
            encoder._ensure_encoder()

    def test_ensure_encoder_succeeds_with_mock_opuslib(self) -> None:
        """When opuslib is importable, _ensure_encoder creates the encoder."""
        mock_opuslib = MagicMock()
        mock_enc = MagicMock()
        mock_opuslib.Encoder.return_value = mock_enc

        with patch.dict(sys.modules, {"opuslib": mock_opuslib}):
            encoder = OpusEncoder(sample_rate=24000)
            encoder._ensure_encoder()

            assert encoder._encoder is mock_enc
            assert encoder._init_error is None
            mock_opuslib.Encoder.assert_called_once()

    def test_ensure_encoder_no_op_when_already_initialized(self) -> None:
        """When encoder is already set, _ensure_encoder returns immediately."""
        encoder, _ = _make_encoder_with_mock()

        # Should not raise, just return
        encoder._ensure_encoder()

    def test_ensure_encoder_caches_init_exception(self) -> None:
        """When opuslib.Encoder() raises, error is cached and re-raised."""
        mock_opuslib = MagicMock()
        mock_opuslib.Encoder.side_effect = RuntimeError("GPU init failed")

        with patch.dict(sys.modules, {"opuslib": mock_opuslib}):
            encoder = OpusEncoder(sample_rate=24000)

            with pytest.raises(CodecUnavailableError, match="GPU init failed"):
                encoder._ensure_encoder()

            assert encoder._init_error == "GPU init failed"

            # Second call raises immediately without calling Encoder again
            mock_opuslib.Encoder.reset_mock()
            with pytest.raises(CodecUnavailableError, match="GPU init failed"):
                encoder._ensure_encoder()
            mock_opuslib.Encoder.assert_not_called()


# ---------------------------------------------------------------------------
# CodecEncoder ABC
# ---------------------------------------------------------------------------


class TestCodecEncoderABC:
    def test_cannot_instantiate_abc(self) -> None:
        """CodecEncoder cannot be instantiated directly."""
        with pytest.raises(TypeError):
            CodecEncoder()  # type: ignore[abstract]
