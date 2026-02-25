"""Output format parser for ElevenLabs-style compound format strings.

Parses format strings like ``mp3_44100_128`` into structured
``OutputFormat`` objects with codec, sample_rate, and bitrate.
Supports both compound strings and simple format names (backward compat).

Examples::

    parse_output_format("mp3_44100_128")  # MP3 at 44.1kHz, 128kbps
    parse_output_format("pcm_16000")      # PCM at 16kHz
    parse_output_format("wav")            # WAV at default rate
    parse_output_format("opus_48000_64")  # Opus at 48kHz, 64kbps
"""

from __future__ import annotations

from dataclasses import dataclass

from macaw._audio_constants import TTS_DEFAULT_SAMPLE_RATE


@dataclass(frozen=True, slots=True)
class OutputFormat:
    """Parsed output format specification.

    Attributes:
        codec: Codec name (wav, pcm, mp3, opus, mulaw, alaw).
        sample_rate: Target output sample rate in Hz.
        bitrate: Target bitrate in kbps (None for lossless/uncompressed).
    """

    codec: str
    sample_rate: int
    bitrate: int | None = None

    @property
    def content_type(self) -> str:
        """HTTP Content-Type for this format."""
        return _CONTENT_TYPES.get(self.codec, "application/octet-stream")

    @property
    def needs_encoding(self) -> bool:
        """True if a codec encoder is needed (not raw PCM/WAV)."""
        return self.codec not in ("wav", "pcm")

    @property
    def bitrate_bps(self) -> int:
        """Bitrate in bits/s (for codec encoder). Defaults to 128kbps."""
        return (self.bitrate or _DEFAULT_BITRATE_KBPS.get(self.codec, 128)) * 1000


# --- Constants ---

#: Supported codec names.
SUPPORTED_CODECS: frozenset[str] = frozenset({"wav", "pcm", "mp3", "opus", "mulaw", "alaw"})

#: Default sample rates per codec (used when not specified in format string).
_DEFAULT_SAMPLE_RATES: dict[str, int] = {
    "wav": TTS_DEFAULT_SAMPLE_RATE,
    "pcm": TTS_DEFAULT_SAMPLE_RATE,
    "mp3": 44100,
    "opus": 48000,
    "mulaw": 8000,
    "alaw": 8000,
}

#: Default bitrates per codec in kbps (used when not specified).
_DEFAULT_BITRATE_KBPS: dict[str, int] = {
    "mp3": 128,
    "opus": 64,
}

#: Allowed sample rates per codec.
_ALLOWED_SAMPLE_RATES: dict[str, frozenset[int]] = {
    "mp3": frozenset({8000, 11025, 16000, 22050, 24000, 32000, 44100, 48000}),
    "opus": frozenset({8000, 12000, 16000, 24000, 48000}),
    "pcm": frozenset({8000, 16000, 22050, 24000, 44100, 48000}),
    "wav": frozenset({8000, 16000, 22050, 24000, 44100, 48000}),
    "mulaw": frozenset({8000}),
    "alaw": frozenset({8000}),
}

#: Allowed bitrate ranges per codec in kbps (min, max).
_ALLOWED_BITRATE_RANGE: dict[str, tuple[int, int]] = {
    "mp3": (32, 320),
    "opus": (6, 512),
}

#: Content-Type mapping.
_CONTENT_TYPES: dict[str, str] = {
    "wav": "audio/wav",
    "pcm": "audio/pcm",
    "mp3": "audio/mpeg",
    "opus": "audio/opus",
    "mulaw": "audio/basic",
    "alaw": "audio/basic",
}


def parse_output_format(format_str: str) -> OutputFormat:
    """Parse an output format string into an ``OutputFormat``.

    Accepts:
        - Simple names: ``wav``, ``pcm``, ``mp3``, ``opus``, ``mulaw``, ``alaw``
        - Compound: ``codec_samplerate`` (e.g., ``pcm_16000``)
        - Compound: ``codec_samplerate_bitrate`` (e.g., ``mp3_44100_128``)

    Args:
        format_str: Format string to parse.

    Returns:
        Parsed ``OutputFormat`` dataclass.

    Raises:
        ValueError: If the format string is invalid, the codec is unknown,
            or sample_rate/bitrate are out of range.
    """
    if not format_str or not format_str.strip():
        msg = "Output format cannot be empty."
        raise ValueError(msg)

    parts = format_str.strip().lower().split("_")
    codec = parts[0]

    if codec not in SUPPORTED_CODECS:
        msg = f"Unknown codec '{codec}'. Supported: {', '.join(sorted(SUPPORTED_CODECS))}."
        raise ValueError(msg)

    sample_rate: int | None = None
    bitrate: int | None = None

    if len(parts) == 1:
        # Simple format: use defaults
        sample_rate = _DEFAULT_SAMPLE_RATES[codec]
    elif len(parts) == 2:
        # codec_samplerate
        sample_rate = _parse_int(parts[1], "sample_rate")
    elif len(parts) == 3:
        # codec_samplerate_bitrate
        sample_rate = _parse_int(parts[1], "sample_rate")
        bitrate = _parse_int(parts[2], "bitrate")
    else:
        msg = (
            f"Invalid format '{format_str}'. "
            "Expected: codec, codec_samplerate, or codec_samplerate_bitrate."
        )
        raise ValueError(msg)

    # Validate sample rate against allowed values
    allowed_rates = _ALLOWED_SAMPLE_RATES.get(codec)
    if allowed_rates is not None and sample_rate not in allowed_rates:
        msg = (
            f"Sample rate {sample_rate}Hz is not supported for codec '{codec}'. "
            f"Allowed: {', '.join(str(r) for r in sorted(allowed_rates))}."
        )
        raise ValueError(msg)

    # Validate bitrate against allowed range
    if bitrate is not None:
        br_range = _ALLOWED_BITRATE_RANGE.get(codec)
        if br_range is None:
            msg = f"Codec '{codec}' does not support bitrate parameter."
            raise ValueError(msg)
        min_br, max_br = br_range
        if not (min_br <= bitrate <= max_br):
            msg = (
                f"Bitrate {bitrate}kbps is out of range for codec '{codec}'. "
                f"Allowed: {min_br}-{max_br}kbps."
            )
            raise ValueError(msg)

    # Codecs that don't support bitrate should not receive one
    if bitrate is not None and codec not in _ALLOWED_BITRATE_RANGE:
        msg = f"Codec '{codec}' does not support bitrate parameter."
        raise ValueError(msg)

    return OutputFormat(
        codec=codec,
        sample_rate=sample_rate,
        bitrate=bitrate,
    )


def _parse_int(value: str, field: str) -> int:
    """Parse a string to int with a clear error message."""
    try:
        result = int(value)
    except ValueError:
        msg = f"Invalid {field}: '{value}' is not a valid integer."
        raise ValueError(msg) from None
    if result <= 0:
        msg = f"Invalid {field}: {result} must be positive."
        raise ValueError(msg)
    return result
