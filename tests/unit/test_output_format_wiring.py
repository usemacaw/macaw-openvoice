"""Tests for output_format wiring across REST and WebSocket endpoints.

Sprint J: Output Format Granularity — verifies that output_format parameter
flows correctly through the API layers and integrates with codec creation.
"""

from __future__ import annotations

import pytest

from macaw.codec.output_format import OutputFormat, parse_output_format
from macaw.server.models.events import TTSSpeakCommand
from macaw.server.models.speech import SpeechRequest

# ---------------------------------------------------------------------------
# SpeechRequest model — output_format field
# ---------------------------------------------------------------------------


class TestSpeechRequestOutputFormat:
    """Tests for the output_format field on SpeechRequest."""

    def test_output_format_default_none(self) -> None:
        req = SpeechRequest(model="test", input="hello")
        assert req.output_format is None

    def test_output_format_set(self) -> None:
        req = SpeechRequest(model="test", input="hello", output_format="mp3_44100_128")
        assert req.output_format == "mp3_44100_128"

    def test_response_format_default_wav(self) -> None:
        req = SpeechRequest(model="test", input="hello")
        assert req.response_format == "wav"

    def test_output_format_coexists_with_response_format(self) -> None:
        req = SpeechRequest(
            model="test",
            input="hello",
            response_format="pcm",
            output_format="mp3_22050_64",
        )
        assert req.response_format == "pcm"
        assert req.output_format == "mp3_22050_64"


# ---------------------------------------------------------------------------
# TTSSpeakCommand model — output_format field
# ---------------------------------------------------------------------------


class TestTTSSpeakCommandOutputFormat:
    """Tests for the output_format field on TTSSpeakCommand."""

    def test_output_format_default_none(self) -> None:
        cmd = TTSSpeakCommand(text="hello")
        assert cmd.output_format is None

    def test_output_format_set(self) -> None:
        cmd = TTSSpeakCommand(text="hello", output_format="opus_48000_64")
        assert cmd.output_format == "opus_48000_64"

    def test_codec_still_works(self) -> None:
        """Backward compat: codec field still accepted."""
        cmd = TTSSpeakCommand(text="hello", codec="opus")
        assert cmd.codec == "opus"
        assert cmd.output_format is None

    def test_output_format_coexists_with_codec(self) -> None:
        cmd = TTSSpeakCommand(text="hello", codec="opus", output_format="mp3_44100_128")
        assert cmd.codec == "opus"
        assert cmd.output_format == "mp3_44100_128"


# ---------------------------------------------------------------------------
# _resolve_output_format (REST speech.py helper)
# ---------------------------------------------------------------------------


class TestResolveOutputFormat:
    """Tests for the _resolve_output_format logic used in speech.py."""

    def test_output_format_takes_precedence(self) -> None:
        """When output_format is set, it takes precedence over response_format."""
        body = SpeechRequest(
            model="test",
            input="hello",
            response_format="wav",
            output_format="mp3_44100_128",
        )
        # Simulate what _resolve_output_format does
        if body.output_format is not None:
            fmt = parse_output_format(body.output_format)
        else:
            fmt = parse_output_format(body.response_format)

        assert fmt.codec == "mp3"
        assert fmt.sample_rate == 44100
        assert fmt.bitrate == 128

    def test_falls_back_to_response_format(self) -> None:
        body = SpeechRequest(model="test", input="hello", response_format="opus")
        if body.output_format is not None:
            fmt = parse_output_format(body.output_format)
        else:
            fmt = parse_output_format(body.response_format)

        assert fmt.codec == "opus"
        assert fmt.sample_rate == 48000  # Default for opus

    def test_wav_default(self) -> None:
        body = SpeechRequest(model="test", input="hello")
        fmt = parse_output_format(body.response_format)
        assert fmt.codec == "wav"
        assert not fmt.needs_encoding


# ---------------------------------------------------------------------------
# _resolve_ws_output_format (WebSocket realtime.py helper)
# ---------------------------------------------------------------------------


class TestResolveWsOutputFormat:
    """Tests for the WS output format resolution logic."""

    def test_output_format_parses(self) -> None:
        cmd = TTSSpeakCommand(text="hello", output_format="mp3_44100_128")
        # Simulate _resolve_ws_output_format
        if cmd.output_format is not None:
            result = parse_output_format(cmd.output_format)
        elif cmd.codec is not None:
            result = parse_output_format(cmd.codec)
        else:
            result = None

        assert isinstance(result, OutputFormat)
        assert result.codec == "mp3"
        assert result.sample_rate == 44100

    def test_codec_backward_compat(self) -> None:
        cmd = TTSSpeakCommand(text="hello", codec="opus")
        if cmd.output_format is not None:
            result = parse_output_format(cmd.output_format)
        elif cmd.codec is not None:
            result = parse_output_format(cmd.codec)
        else:
            result = None

        assert isinstance(result, OutputFormat)
        assert result.codec == "opus"
        assert result.sample_rate == 48000

    def test_output_format_takes_precedence_over_codec(self) -> None:
        cmd = TTSSpeakCommand(text="hello", codec="opus", output_format="mp3_22050")
        if cmd.output_format is not None:
            result = parse_output_format(cmd.output_format)
        elif cmd.codec is not None:
            result = parse_output_format(cmd.codec)
        else:
            result = None

        assert isinstance(result, OutputFormat)
        assert result.codec == "mp3"
        assert result.sample_rate == 22050

    def test_no_format_returns_none(self) -> None:
        cmd = TTSSpeakCommand(text="hello")
        if cmd.output_format is not None:
            result: OutputFormat | None = parse_output_format(cmd.output_format)
        elif cmd.codec is not None:
            result = parse_output_format(cmd.codec)
        else:
            result = None

        assert result is None

    def test_invalid_output_format_raises(self) -> None:
        with pytest.raises(ValueError, match="Unknown codec"):
            parse_output_format("flac_44100")


# ---------------------------------------------------------------------------
# Integration: OutputFormat → codec encoder
# ---------------------------------------------------------------------------


class TestOutputFormatEncoderIntegration:
    """Tests for OutputFormat properties used by codec encoder creation."""

    def test_bitrate_bps_for_mp3(self) -> None:
        fmt = parse_output_format("mp3_44100_192")
        assert fmt.bitrate_bps == 192_000

    def test_bitrate_bps_default_for_mp3(self) -> None:
        fmt = parse_output_format("mp3_44100")
        assert fmt.bitrate_bps == 128_000  # Default 128kbps

    def test_bitrate_bps_for_opus(self) -> None:
        fmt = parse_output_format("opus_48000_96")
        assert fmt.bitrate_bps == 96_000

    def test_bitrate_bps_default_for_opus(self) -> None:
        fmt = parse_output_format("opus_48000")
        assert fmt.bitrate_bps == 64_000  # Default 64kbps

    def test_wav_no_encoder_needed(self) -> None:
        fmt = parse_output_format("wav_24000")
        assert not fmt.needs_encoding

    def test_pcm_no_encoder_needed(self) -> None:
        fmt = parse_output_format("pcm_16000")
        assert not fmt.needs_encoding

    def test_wav_content_type(self) -> None:
        fmt = parse_output_format("wav")
        assert fmt.content_type == "audio/wav"

    def test_mp3_content_type(self) -> None:
        fmt = parse_output_format("mp3")
        assert fmt.content_type == "audio/mpeg"

    def test_opus_content_type(self) -> None:
        fmt = parse_output_format("opus")
        assert fmt.content_type == "audio/opus"

    def test_mulaw_content_type(self) -> None:
        fmt = parse_output_format("mulaw")
        assert fmt.content_type == "audio/basic"


# ---------------------------------------------------------------------------
# WAV header with custom sample rate
# ---------------------------------------------------------------------------


class TestWavHeaderSampleRate:
    """Tests that WAV header uses the output_format sample rate."""

    def test_wav_16k_header(self) -> None:
        """WAV at 16kHz should produce valid header with byte_rate=32000."""
        import struct

        from macaw.server.routes.speech import _wav_streaming_header

        header = _wav_streaming_header(16000)
        assert header[:4] == b"RIFF"
        assert header[8:12] == b"WAVE"
        # byte_rate at offset 28: sample_rate * channels * bits_per_sample / 8
        byte_rate = struct.unpack_from("<I", header, 28)[0]
        assert byte_rate == 16000 * 1 * 16 // 8  # 32000

    def test_wav_44100_header(self) -> None:
        """WAV at 44.1kHz should have byte_rate=88200."""
        import struct

        from macaw.server.routes.speech import _wav_streaming_header

        header = _wav_streaming_header(44100)
        byte_rate = struct.unpack_from("<I", header, 28)[0]
        assert byte_rate == 44100 * 1 * 16 // 8
