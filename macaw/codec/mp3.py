"""MP3 codec encoder via lameenc.

lameenc is an optional dependency — import is guarded.
The encoder raises ``CodecUnavailableError`` if lameenc is not installed.
"""

from __future__ import annotations

from typing import Any

from macaw.codec.interface import CodecEncoder
from macaw.exceptions import CodecUnavailableError
from macaw.logging import get_logger

logger = get_logger("codec.mp3")

# Import guard: checked at encode() time via _ensure_encoder()
try:
    import lameenc as _lameenc
except ImportError:
    _lameenc = None

# Default MP3 encoding quality — VBR quality 2 (high quality, ~190 kbps)
_DEFAULT_MP3_QUALITY = 2


class Mp3Encoder(CodecEncoder):
    """MP3 codec encoder using lameenc (LAME wrapper).

    Accepts raw 16-bit PCM input and produces MP3-encoded output.
    Unlike Opus, MP3 does not require fixed frame sizes — lameenc
    handles internal buffering.

    Args:
        sample_rate: Input PCM sample rate (e.g., 24000 for TTS).
        channels: Number of audio channels (default: 1 mono).
        bitrate: Target bitrate in kbps (default: 128).
        quality: VBR quality 0-9, lower is better (default: 2).
    """

    def __init__(
        self,
        sample_rate: int = 24000,
        channels: int = 1,
        bitrate: int = 128,
        quality: int = _DEFAULT_MP3_QUALITY,
    ) -> None:
        self._sample_rate = sample_rate
        self._channels = channels
        self._bitrate = bitrate
        self._quality = quality
        self._encoder: Any | None = None
        self._init_attempted: bool = False
        self._init_error: str | None = None

    def _ensure_encoder(self) -> None:
        """Lazily create the LAME encoder.

        Raises:
            CodecUnavailableError: If lameenc is not installed or encoder
                initialization fails.
        """
        if self._encoder is not None:
            return
        if self._init_error is not None:
            raise CodecUnavailableError("mp3", self._init_error)
        if self._init_attempted:
            return
        self._init_attempted = True
        try:
            if _lameenc is None:
                msg = "lameenc not installed. Install with: pip install macaw-openvoice[mp3]"
                raise ImportError(msg)

            encoder = _lameenc.Encoder()
            encoder.set_in_sample_rate(self._sample_rate)
            encoder.set_channels(self._channels)
            encoder.set_bit_rate(self._bitrate)
            encoder.set_quality(self._quality)
            self._encoder = encoder
            logger.info(
                "mp3_encoder_ready",
                sample_rate=self._sample_rate,
                bitrate=self._bitrate,
                quality=self._quality,
            )
        except ImportError as exc:
            self._init_error = str(exc)
            raise CodecUnavailableError("mp3", self._init_error) from exc
        except Exception as exc:
            self._init_error = str(exc)
            raise CodecUnavailableError("mp3", self._init_error) from exc

    @property
    def codec_name(self) -> str:
        return "mp3"

    def encode(self, pcm_data: bytes) -> bytes:
        """Encode PCM data to MP3.

        Raises:
            CodecUnavailableError: If lameenc is not installed.
        """
        self._ensure_encoder()
        assert self._encoder is not None
        return bytes(self._encoder.encode(pcm_data))

    def flush(self) -> bytes:
        """Flush the LAME encoder buffer.

        Raises:
            CodecUnavailableError: If lameenc is not installed and
                there is data to flush.
        """
        if self._encoder is None:
            # Never initialized — no data to flush
            return b""
        return bytes(self._encoder.flush())
