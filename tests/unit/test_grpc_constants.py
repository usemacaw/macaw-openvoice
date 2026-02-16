"""Tests for macaw._grpc_constants — function builders and PEP 562 compat."""

from __future__ import annotations

import pytest

from macaw._grpc_constants import (
    get_batch_channel_options,
    get_streaming_channel_options,
    get_tts_channel_options,
)
from macaw.config.settings import get_settings


@pytest.fixture(autouse=True)
def _clear_settings_cache() -> None:
    """Ensure each test sees fresh settings."""
    get_settings.cache_clear()


class TestGetBatchChannelOptions:
    def test_returns_list_of_tuples(self) -> None:
        opts = get_batch_channel_options()
        assert isinstance(opts, list)
        assert all(isinstance(t, tuple) and len(t) == 2 for t in opts)

    def test_default_message_size(self) -> None:
        opts = dict(get_batch_channel_options())
        expected = 30 * 1024 * 1024
        assert opts["grpc.max_send_message_length"] == expected
        assert opts["grpc.max_receive_message_length"] == expected

    def test_includes_keepalive(self) -> None:
        opts = dict(get_batch_channel_options())
        assert "grpc.keepalive_time_ms" in opts
        assert "grpc.keepalive_timeout_ms" in opts

    def test_env_override_reflected(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("MACAW_GRPC_MAX_BATCH_MESSAGE_MB", "50")
        get_settings.cache_clear()
        opts = dict(get_batch_channel_options())
        expected = 50 * 1024 * 1024
        assert opts["grpc.max_send_message_length"] == expected
        assert opts["grpc.max_receive_message_length"] == expected


class TestGetStreamingChannelOptions:
    def test_returns_list_of_tuples(self) -> None:
        opts = get_streaming_channel_options()
        assert isinstance(opts, list)
        assert all(isinstance(t, tuple) and len(t) == 2 for t in opts)

    def test_default_message_size(self) -> None:
        opts = dict(get_streaming_channel_options())
        expected = 10 * 1024 * 1024
        assert opts["grpc.max_send_message_length"] == expected
        assert opts["grpc.max_receive_message_length"] == expected

    def test_includes_aggressive_keepalive(self) -> None:
        opts = dict(get_streaming_channel_options())
        assert opts["grpc.keepalive_time_ms"] == 10_000
        assert opts["grpc.keepalive_timeout_ms"] == 5_000
        assert opts["grpc.http2.max_pings_without_data"] == 0

    def test_env_override_reflected(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("MACAW_GRPC_MAX_STREAMING_MESSAGE_MB", "20")
        get_settings.cache_clear()
        opts = dict(get_streaming_channel_options())
        expected = 20 * 1024 * 1024
        assert opts["grpc.max_send_message_length"] == expected


class TestGetTTSChannelOptions:
    def test_returns_list_of_tuples(self) -> None:
        opts = get_tts_channel_options()
        assert isinstance(opts, list)

    def test_default_message_size_matches_batch(self) -> None:
        opts = dict(get_tts_channel_options())
        expected = 30 * 1024 * 1024
        assert opts["grpc.max_send_message_length"] == expected

    def test_no_custom_keepalive(self) -> None:
        opts = dict(get_tts_channel_options())
        assert "grpc.keepalive_time_ms" not in opts


class TestPEP562BackwardCompat:
    """Old import names resolve via __getattr__ to function calls."""

    def test_batch_compat_import(self) -> None:
        import macaw._grpc_constants as mod

        opts = mod.GRPC_BATCH_CHANNEL_OPTIONS  # type: ignore[attr-defined]
        assert isinstance(opts, list)
        assert len(opts) > 0

    def test_streaming_compat_import(self) -> None:
        import macaw._grpc_constants as mod

        opts = mod.GRPC_STREAMING_CHANNEL_OPTIONS  # type: ignore[attr-defined]
        assert isinstance(opts, list)
        assert len(opts) > 0

    def test_tts_compat_import(self) -> None:
        import macaw._grpc_constants as mod

        opts = mod.GRPC_TTS_CHANNEL_OPTIONS  # type: ignore[attr-defined]
        assert isinstance(opts, list)
        assert len(opts) > 0

    def test_unknown_attr_raises(self) -> None:
        import macaw._grpc_constants as mod

        with pytest.raises(AttributeError, match="NONEXISTENT"):
            _ = mod.NONEXISTENT  # type: ignore[attr-defined]
