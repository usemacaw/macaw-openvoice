"""Runtime codec layer for audio encoding/decoding."""

from __future__ import annotations

from macaw.codec.interface import CodecEncoder

__all__ = ["CodecEncoder", "create_encoder"]


def create_encoder(
    codec_name: str,
    sample_rate: int = 24000,
    bitrate: int = 64000,
) -> CodecEncoder | None:
    """Create a codec encoder by name.

    Returns None if the codec is not recognized.
    The encoder itself handles unavailability of the underlying library
    by falling back to pass-through (fail-open).

    Args:
        codec_name: Codec identifier (e.g., "opus").
        sample_rate: Input PCM sample rate.
        bitrate: Target bitrate in bits/s (codec-specific).

    Returns:
        CodecEncoder instance, or None if codec_name is unknown.
    """
    if codec_name == "opus":
        from macaw.codec.opus import OpusEncoder

        return OpusEncoder(sample_rate=sample_rate, bitrate=bitrate)
    return None
