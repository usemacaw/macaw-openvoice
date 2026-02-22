"""Runtime codec layer for audio encoding/decoding."""

from __future__ import annotations

from macaw.codec.interface import CodecEncoder
from macaw.exceptions import CodecUnavailableError

__all__ = ["CodecEncoder", "create_encoder", "is_codec_available"]


def create_encoder(
    codec_name: str,
    sample_rate: int = 24000,
    bitrate: int = 64000,
) -> CodecEncoder:
    """Create a codec encoder by name.

    Raises:
        CodecUnavailableError: If the codec name is not recognized.

    Args:
        codec_name: Codec identifier (e.g., "opus").
        sample_rate: Input PCM sample rate.
        bitrate: Target bitrate in bits/s (codec-specific).

    Returns:
        CodecEncoder instance.
    """
    if codec_name == "opus":
        from macaw.codec.opus import OpusEncoder

        return OpusEncoder(sample_rate=sample_rate, bitrate=bitrate)
    if codec_name == "mp3":
        from macaw.codec.mp3 import Mp3Encoder

        return Mp3Encoder(sample_rate=sample_rate, bitrate=bitrate // 1000)
    if codec_name in ("mulaw", "alaw"):
        from macaw.codec.g711 import ALawEncoder, MuLawEncoder

        if codec_name == "mulaw":
            return MuLawEncoder(sample_rate=sample_rate)
        return ALawEncoder(sample_rate=sample_rate)
    raise CodecUnavailableError(codec_name, "unknown codec")


def is_codec_available(codec_name: str) -> bool:
    """Check if a codec is available without creating an encoder.

    Returns True if the codec is recognized and its underlying library
    is importable. Does not create an encoder instance.
    """
    if codec_name == "opus":
        try:
            import opuslib  # noqa: F401

            return True
        except ImportError:
            return False
    if codec_name == "mp3":
        try:
            import lameenc  # noqa: F401

            return True
        except ImportError:
            return False
    # G.711 is pure numpy — always available
    return codec_name in ("mulaw", "alaw")
