"""Abstract interface for audio codec encoders."""

from __future__ import annotations

from abc import ABC, abstractmethod


class CodecEncoder(ABC):
    """Contract for audio codec encoders.

    Encoders accept raw PCM chunks and produce encoded audio packets.
    They maintain internal state for frame alignment (e.g., Opus requires
    fixed-size frames).
    """

    @property
    @abstractmethod
    def codec_name(self) -> str:
        """Codec identifier (e.g., 'opus', 'mp3')."""
        pass

    @abstractmethod
    def encode(self, pcm_data: bytes) -> bytes:
        """Encode PCM audio data.

        May buffer internally if the codec requires fixed frame sizes.
        Returns encoded bytes, or empty bytes if buffering.

        Args:
            pcm_data: Raw 16-bit PCM audio.

        Returns:
            Encoded audio bytes (may be empty if buffering).
        """
        pass

    @abstractmethod
    def flush(self) -> bytes:
        """Flush any buffered PCM data.

        Called at the end of a stream to encode remaining buffered audio.

        Returns:
            Final encoded audio bytes (may be empty).
        """
        pass
