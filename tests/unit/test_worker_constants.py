"""Tests for macaw.workers._constants — PEP 562 lazy delegation to settings."""

from __future__ import annotations

import pytest

from macaw.config.settings import get_settings


@pytest.fixture(autouse=True)
def _clear_settings_cache() -> None:
    """Ensure each test sees fresh settings."""
    get_settings.cache_clear()


class TestDefaultValues:
    """Default values match WorkerLifecycleSettings defaults."""

    def test_stop_grace_period_default(self) -> None:
        import macaw.workers._constants as mod

        assert mod.STOP_GRACE_PERIOD == 5.0  # type: ignore[attr-defined]

    def test_default_warmup_steps_default(self) -> None:
        import macaw.workers._constants as mod

        assert mod.DEFAULT_WARMUP_STEPS == 3  # type: ignore[attr-defined]

    def test_grpc_worker_server_options_static(self) -> None:
        import macaw.workers._constants as mod

        assert isinstance(mod.GRPC_WORKER_SERVER_OPTIONS, list)
        assert len(mod.GRPC_WORKER_SERVER_OPTIONS) == 2


class TestEnvOverrides:
    """Env overrides flow through PEP 562 -> settings."""

    def test_stop_grace_period_from_env(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("MACAW_WORKER_STOP_GRACE_PERIOD_S", "10.0")
        get_settings.cache_clear()
        import macaw.workers._constants as mod

        assert mod.STOP_GRACE_PERIOD == 10.0  # type: ignore[attr-defined]

    def test_warmup_steps_from_env(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("MACAW_WORKER_WARMUP_STEPS", "0")
        get_settings.cache_clear()
        import macaw.workers._constants as mod

        assert mod.DEFAULT_WARMUP_STEPS == 0  # type: ignore[attr-defined]


class TestUnknownAttr:
    def test_unknown_attr_raises(self) -> None:
        import macaw.workers._constants as mod

        with pytest.raises(AttributeError, match="NONEXISTENT"):
            _ = mod.NONEXISTENT  # type: ignore[attr-defined]
