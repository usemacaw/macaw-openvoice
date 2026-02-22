"""Opus codec encoder for TTS audio streams."""

from __future__ import annotations

from typing import Any

from macaw.codec.interface import CodecEncoder
from macaw.exceptions import CodecUnavailableError
from macaw.logging import get_logger

logger = get_logger("codec.opus")

# Opus frame size: 20ms at the encoder sample rate
OPUS_FRAME_DURATION_MS: int = 20
OPUS_APPLICATION_AUDIO: int = 2049  # OPUS_APPLICATION_AUDIO constant


class OpusEncoder(CodecEncoder):
    """Opus codec encoder with internal PCM buffering for frame alignment.

    Opus requires fixed-size input frames (typically 20ms). This encoder
    buffers incoming PCM chunks and encodes complete frames, holding
    partial frames until more data arrives or flush() is called.

    Args:
        sample_rate: Input PCM sample rate (e.g., 24000 for TTS).
        channels: Number of audio channels (default: 1 mono).
        bitrate: Target bitrate in bits/s (default: 64000).
    """

    def __init__(
        self,
        sample_rate: int = 24000,
        channels: int = 1,
        bitrate: int = 64000,
    ) -> None:
        self._sample_rate = sample_rate
        self._channels = channels
        self._bitrate = bitrate

        # Frame size in samples: 20ms at sample_rate
        self._frame_samples = (OPUS_FRAME_DURATION_MS * sample_rate) // 1000
        # Frame size in bytes: samples * channels * 2 (16-bit)
        self._frame_bytes = self._frame_samples * channels * 2

        # Internal PCM buffer for frame alignment
        self._buffer = bytearray()

        # Lazy-load opuslib encoder
        self._encoder: Any | None = None
        self._init_attempted: bool = False
        self._init_error: str | None = None

    def _ensure_encoder(self) -> None:
        """Lazily create the Opus encoder.

        Raises:
            CodecUnavailableError: If opuslib is not installed or encoder
                initialization fails. The error is cached so subsequent
                calls raise immediately without retrying.
        """
        if self._encoder is not None:
            return
        if self._init_error is not None:
            raise CodecUnavailableError("opus", self._init_error)
        if self._init_attempted:
            return
        self._init_attempted = True
        try:
            import opuslib

            self._encoder = opuslib.Encoder(
                self._sample_rate, self._channels, OPUS_APPLICATION_AUDIO
            )
            logger.info(
                "opus_encoder_ready",
                sample_rate=self._sample_rate,
                bitrate=self._bitrate,
            )
        except ImportError as exc:
            self._init_error = "opuslib not installed. Install with: pip install opuslib"
            raise CodecUnavailableError("opus", self._init_error) from exc
        except Exception as exc:
            self._init_error = str(exc)
            raise CodecUnavailableError("opus", self._init_error) from exc

    @property
    def codec_name(self) -> str:
        return "opus"

    def encode(self, pcm_data: bytes) -> bytes:
        """Encode PCM data, buffering for frame alignment.

        Raises:
            CodecUnavailableError: If opuslib is not installed.
        """
        self._ensure_encoder()
        assert self._encoder is not None  # Guaranteed by _ensure_encoder()
        self._buffer.extend(pcm_data)

        encoded_parts: list[bytes] = []
        while len(self._buffer) >= self._frame_bytes:
            frame = bytes(self._buffer[: self._frame_bytes])
            del self._buffer[: self._frame_bytes]
            encoded_parts.append(self._encoder.encode(frame, self._frame_samples))

        return b"".join(encoded_parts)

    def flush(self) -> bytes:
        """Encode remaining buffered data (zero-padded to frame boundary).

        Raises:
            CodecUnavailableError: If opuslib is not installed and buffer
                has data to flush.
        """
        if not self._buffer:
            return b""

        self._ensure_encoder()
        assert self._encoder is not None  # Guaranteed by _ensure_encoder()

        # Pad to frame boundary
        remaining = bytes(self._buffer)
        self._buffer.clear()

        pad_length = self._frame_bytes - len(remaining)
        padded = remaining + b"\x00" * pad_length
        return bytes(self._encoder.encode(padded, self._frame_samples))
