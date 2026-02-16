"""Shared gRPC channel pooling for TTS workers."""

from __future__ import annotations

import grpc.aio

from macaw._grpc_constants import GRPC_TTS_CHANNEL_OPTIONS
from macaw.logging import get_logger

logger = get_logger("server.grpc_channels")


def get_or_create_tts_channel(
    tts_channels: dict[str, grpc.aio.Channel],
    address: str,
) -> grpc.aio.Channel:
    """Return a pooled gRPC channel for the TTS worker address, creating if needed.

    Channels are reused across requests to avoid TCP+HTTP/2 handshake
    overhead (~5-20ms per request).
    """
    channel = tts_channels.get(address)
    if channel is None:
        channel = grpc.aio.insecure_channel(address, options=GRPC_TTS_CHANNEL_OPTIONS)
        tts_channels[address] = channel
    return channel


async def close_tts_channels(tts_channels: dict[str, grpc.aio.Channel]) -> None:
    """Close all pooled TTS gRPC channels. Called on server shutdown."""
    for address, channel in tts_channels.items():
        try:
            await channel.close()
        except Exception:
            logger.warning("tts_channel_close_error", address=address)
    tts_channels.clear()
