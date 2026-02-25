"""Tests for macaw.codec.output_format — output format parser.

Sprint J: Output Format Granularity (ElevenLabs-style compound format strings).
"""

from __future__ import annotations

import pytest

from macaw._audio_constants import TTS_DEFAULT_SAMPLE_RATE
from macaw.codec.output_format import (
    _ALLOWED_BITRATE_RANGE,
    _ALLOWED_SAMPLE_RATES,
    _CONTENT_TYPES,
    _DEFAULT_BITRATE_KBPS,
    _DEFAULT_SAMPLE_RATES,
    SUPPORTED_CODECS,
    OutputFormat,
    parse_output_format,
)

# ---------------------------------------------------------------------------
# OutputFormat dataclass
# ---------------------------------------------------------------------------


class TestOutputFormat:
    """Tests for the OutputFormat frozen dataclass."""

    def test_content_type_known_codecs(self) -> None:
        for codec, expected_ct in _CONTENT_TYPES.items():
            fmt = OutputFormat(codec=codec, sample_rate=16000)
            assert fmt.content_type == expected_ct

    def test_content_type_unknown_codec(self) -> None:
        fmt = OutputFormat(codec="flac", sample_rate=44100)
        assert fmt.content_type == "application/octet-stream"

    def test_needs_encoding_false_for_wav(self) -> None:
        assert not OutputFormat(codec="wav", sample_rate=24000).needs_encoding

    def test_needs_encoding_false_for_pcm(self) -> None:
        assert not OutputFormat(codec="pcm", sample_rate=16000).needs_encoding

    def test_needs_encoding_true_for_mp3(self) -> None:
        assert OutputFormat(codec="mp3", sample_rate=44100).needs_encoding

    def test_needs_encoding_true_for_opus(self) -> None:
        assert OutputFormat(codec="opus", sample_rate=48000).needs_encoding

    def test_needs_encoding_true_for_mulaw(self) -> None:
        assert OutputFormat(codec="mulaw", sample_rate=8000).needs_encoding

    def test_needs_encoding_true_for_alaw(self) -> None:
        assert OutputFormat(codec="alaw", sample_rate=8000).needs_encoding

    def test_bitrate_bps_explicit(self) -> None:
        fmt = OutputFormat(codec="mp3", sample_rate=44100, bitrate=192)
        assert fmt.bitrate_bps == 192_000

    def test_bitrate_bps_default_mp3(self) -> None:
        fmt = OutputFormat(codec="mp3", sample_rate=44100)
        assert fmt.bitrate_bps == _DEFAULT_BITRATE_KBPS["mp3"] * 1000

    def test_bitrate_bps_default_opus(self) -> None:
        fmt = OutputFormat(codec="opus", sample_rate=48000)
        assert fmt.bitrate_bps == _DEFAULT_BITRATE_KBPS["opus"] * 1000

    def test_bitrate_bps_default_fallback(self) -> None:
        """Codecs without explicit default get 128kbps."""
        fmt = OutputFormat(codec="wav", sample_rate=24000)
        assert fmt.bitrate_bps == 128_000

    def test_frozen(self) -> None:
        fmt = OutputFormat(codec="wav", sample_rate=24000)
        with pytest.raises(AttributeError):
            fmt.codec = "mp3"  # type: ignore[misc]


# ---------------------------------------------------------------------------
# parse_output_format — simple names
# ---------------------------------------------------------------------------


class TestParseSimpleFormat:
    """Tests for simple (single-word) format strings."""

    def test_wav_defaults(self) -> None:
        fmt = parse_output_format("wav")
        assert fmt.codec == "wav"
        assert fmt.sample_rate == TTS_DEFAULT_SAMPLE_RATE
        assert fmt.bitrate is None

    def test_pcm_defaults(self) -> None:
        fmt = parse_output_format("pcm")
        assert fmt.codec == "pcm"
        assert fmt.sample_rate == TTS_DEFAULT_SAMPLE_RATE

    def test_mp3_defaults(self) -> None:
        fmt = parse_output_format("mp3")
        assert fmt.codec == "mp3"
        assert fmt.sample_rate == 44100
        assert fmt.bitrate is None

    def test_opus_defaults(self) -> None:
        fmt = parse_output_format("opus")
        assert fmt.codec == "opus"
        assert fmt.sample_rate == 48000

    def test_mulaw_defaults(self) -> None:
        fmt = parse_output_format("mulaw")
        assert fmt.codec == "mulaw"
        assert fmt.sample_rate == 8000

    def test_alaw_defaults(self) -> None:
        fmt = parse_output_format("alaw")
        assert fmt.codec == "alaw"
        assert fmt.sample_rate == 8000

    def test_case_insensitive(self) -> None:
        fmt = parse_output_format("MP3")
        assert fmt.codec == "mp3"

    def test_whitespace_stripped(self) -> None:
        fmt = parse_output_format("  wav  ")
        assert fmt.codec == "wav"

    def test_all_supported_codecs_parseable(self) -> None:
        for codec in SUPPORTED_CODECS:
            fmt = parse_output_format(codec)
            assert fmt.codec == codec
            assert fmt.sample_rate == _DEFAULT_SAMPLE_RATES[codec]


# ---------------------------------------------------------------------------
# parse_output_format — compound names
# ---------------------------------------------------------------------------


class TestParseCompoundFormat:
    """Tests for compound format strings (codec_samplerate[_bitrate])."""

    def test_pcm_16000(self) -> None:
        fmt = parse_output_format("pcm_16000")
        assert fmt.codec == "pcm"
        assert fmt.sample_rate == 16000
        assert fmt.bitrate is None

    def test_pcm_44100(self) -> None:
        fmt = parse_output_format("pcm_44100")
        assert fmt.codec == "pcm"
        assert fmt.sample_rate == 44100

    def test_wav_16000(self) -> None:
        fmt = parse_output_format("wav_16000")
        assert fmt.codec == "wav"
        assert fmt.sample_rate == 16000

    def test_mp3_44100_128(self) -> None:
        fmt = parse_output_format("mp3_44100_128")
        assert fmt.codec == "mp3"
        assert fmt.sample_rate == 44100
        assert fmt.bitrate == 128

    def test_mp3_22050_64(self) -> None:
        fmt = parse_output_format("mp3_22050_64")
        assert fmt.codec == "mp3"
        assert fmt.sample_rate == 22050
        assert fmt.bitrate == 64

    def test_opus_48000_64(self) -> None:
        fmt = parse_output_format("opus_48000_64")
        assert fmt.codec == "opus"
        assert fmt.sample_rate == 48000
        assert fmt.bitrate == 64

    def test_opus_24000(self) -> None:
        fmt = parse_output_format("opus_24000")
        assert fmt.codec == "opus"
        assert fmt.sample_rate == 24000

    def test_mp3_with_sample_rate_only(self) -> None:
        fmt = parse_output_format("mp3_24000")
        assert fmt.codec == "mp3"
        assert fmt.sample_rate == 24000
        assert fmt.bitrate is None

    def test_case_insensitive_compound(self) -> None:
        fmt = parse_output_format("MP3_44100_128")
        assert fmt.codec == "mp3"
        assert fmt.sample_rate == 44100
        assert fmt.bitrate == 128


# ---------------------------------------------------------------------------
# parse_output_format — validation errors
# ---------------------------------------------------------------------------


class TestParseFormatErrors:
    """Tests for parse_output_format error handling."""

    def test_empty_string(self) -> None:
        with pytest.raises(ValueError, match="cannot be empty"):
            parse_output_format("")

    def test_whitespace_only(self) -> None:
        with pytest.raises(ValueError, match="cannot be empty"):
            parse_output_format("   ")

    def test_unknown_codec(self) -> None:
        with pytest.raises(ValueError, match="Unknown codec"):
            parse_output_format("flac")

    def test_too_many_parts(self) -> None:
        with pytest.raises(ValueError, match="Invalid format"):
            parse_output_format("mp3_44100_128_extra")

    def test_invalid_sample_rate_not_int(self) -> None:
        with pytest.raises(ValueError, match="not a valid integer"):
            parse_output_format("mp3_notanum")

    def test_invalid_bitrate_not_int(self) -> None:
        with pytest.raises(ValueError, match="not a valid integer"):
            parse_output_format("mp3_44100_abc")

    def test_sample_rate_zero(self) -> None:
        with pytest.raises(ValueError, match="must be positive"):
            parse_output_format("mp3_0")

    def test_sample_rate_negative(self) -> None:
        with pytest.raises(ValueError, match="must be positive"):
            parse_output_format("mp3_-1")

    def test_mp3_unsupported_sample_rate(self) -> None:
        with pytest.raises(ValueError, match="not supported for codec"):
            parse_output_format("mp3_12345")

    def test_opus_unsupported_sample_rate(self) -> None:
        with pytest.raises(ValueError, match="not supported for codec"):
            parse_output_format("opus_44100")

    def test_mulaw_only_8000(self) -> None:
        with pytest.raises(ValueError, match="not supported for codec"):
            parse_output_format("mulaw_16000")

    def test_alaw_only_8000(self) -> None:
        with pytest.raises(ValueError, match="not supported for codec"):
            parse_output_format("alaw_16000")

    def test_mp3_bitrate_too_low(self) -> None:
        min_br = _ALLOWED_BITRATE_RANGE["mp3"][0]
        with pytest.raises(ValueError, match="out of range"):
            parse_output_format(f"mp3_44100_{min_br - 1}")

    def test_mp3_bitrate_too_high(self) -> None:
        max_br = _ALLOWED_BITRATE_RANGE["mp3"][1]
        with pytest.raises(ValueError, match="out of range"):
            parse_output_format(f"mp3_44100_{max_br + 1}")

    def test_opus_bitrate_too_low(self) -> None:
        min_br = _ALLOWED_BITRATE_RANGE["opus"][0]
        with pytest.raises(ValueError, match="out of range"):
            parse_output_format(f"opus_48000_{min_br - 1}")

    def test_opus_bitrate_too_high(self) -> None:
        max_br = _ALLOWED_BITRATE_RANGE["opus"][1]
        with pytest.raises(ValueError, match="out of range"):
            parse_output_format(f"opus_48000_{max_br + 1}")

    def test_wav_bitrate_not_supported(self) -> None:
        with pytest.raises(ValueError, match="does not support bitrate"):
            parse_output_format("wav_24000_128")

    def test_pcm_bitrate_not_supported(self) -> None:
        with pytest.raises(ValueError, match="does not support bitrate"):
            parse_output_format("pcm_16000_128")

    def test_mulaw_bitrate_not_supported(self) -> None:
        with pytest.raises(ValueError, match="does not support bitrate"):
            parse_output_format("mulaw_8000_64")

    def test_mp3_boundary_bitrate_min(self) -> None:
        """Min bitrate boundary should be accepted."""
        min_br = _ALLOWED_BITRATE_RANGE["mp3"][0]
        fmt = parse_output_format(f"mp3_44100_{min_br}")
        assert fmt.bitrate == min_br

    def test_mp3_boundary_bitrate_max(self) -> None:
        """Max bitrate boundary should be accepted."""
        max_br = _ALLOWED_BITRATE_RANGE["mp3"][1]
        fmt = parse_output_format(f"mp3_44100_{max_br}")
        assert fmt.bitrate == max_br

    def test_all_allowed_sample_rates_accepted(self) -> None:
        """Verify every allowed sample rate for each codec parses successfully."""
        for codec, rates in _ALLOWED_SAMPLE_RATES.items():
            for rate in rates:
                fmt = parse_output_format(f"{codec}_{rate}")
                assert fmt.sample_rate == rate
