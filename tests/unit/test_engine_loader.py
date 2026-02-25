"""Tests for dynamic external engine loading."""

from __future__ import annotations

import types
from abc import ABC, abstractmethod

import pytest

from macaw.exceptions import ModelLoadError
from macaw.workers._engine_loader import load_external_backend

# --- Fixtures: fake ABCs and backends for testing ---


class _FakeBackend(ABC):
    """Minimal ABC to simulate STTBackend / TTSBackend."""

    @abstractmethod
    async def do_work(self) -> str: ...


class _ConcreteFakeBackend(_FakeBackend):
    async def do_work(self) -> str:
        return "ok"


class _AnotherConcreteFakeBackend(_FakeBackend):
    async def do_work(self) -> str:
        return "also ok"


class _AbstractMiddle(_FakeBackend, ABC):
    """Abstract intermediate class — should NOT be picked up."""

    @abstractmethod
    async def extra(self) -> None: ...


# --- Tests ---


class TestLoadExternalBackend:
    def test_loads_single_concrete_subclass(self) -> None:
        mod = types.ModuleType("fake_engine")
        mod.MyBackend = _ConcreteFakeBackend  # type: ignore[attr-defined]

        with pytest.MonkeyPatch.context() as mp:
            mp.setitem(__import__("sys").modules, "fake_engine", mod)
            backend = load_external_backend("fake_engine", _FakeBackend)

        assert isinstance(backend, _ConcreteFakeBackend)

    def test_import_error_raises_model_load_error(self) -> None:
        with pytest.raises(ModelLoadError, match="Cannot import"):
            load_external_backend("nonexistent.package.xyz", _FakeBackend)

    def test_no_subclass_raises_model_load_error(self) -> None:
        mod = types.ModuleType("empty_module")

        with pytest.MonkeyPatch.context() as mp:
            mp.setitem(__import__("sys").modules, "empty_module", mod)
            with pytest.raises(ModelLoadError, match="does not contain"):
                load_external_backend("empty_module", _FakeBackend)

    def test_multiple_subclasses_raises_model_load_error(self) -> None:
        mod = types.ModuleType("ambiguous_engine")
        mod.BackendA = _ConcreteFakeBackend  # type: ignore[attr-defined]
        mod.BackendB = _AnotherConcreteFakeBackend  # type: ignore[attr-defined]

        with pytest.MonkeyPatch.context() as mp:
            mp.setitem(__import__("sys").modules, "ambiguous_engine", mod)
            with pytest.raises(ModelLoadError, match="multiple concrete subclasses"):
                load_external_backend("ambiguous_engine", _FakeBackend)

    def test_ignores_the_abc_itself(self) -> None:
        """If the module re-exports the ABC, it should be ignored."""
        mod = types.ModuleType("reexport_engine")
        mod.FakeBackend = _FakeBackend  # type: ignore[attr-defined]
        mod.Impl = _ConcreteFakeBackend  # type: ignore[attr-defined]

        with pytest.MonkeyPatch.context() as mp:
            mp.setitem(__import__("sys").modules, "reexport_engine", mod)
            backend = load_external_backend("reexport_engine", _FakeBackend)

        assert isinstance(backend, _ConcreteFakeBackend)

    def test_ignores_abstract_intermediate_classes(self) -> None:
        """Abstract intermediate subclasses should not count as candidates."""
        mod = types.ModuleType("abstract_mid_engine")
        mod.Middle = _AbstractMiddle  # type: ignore[attr-defined]

        with pytest.MonkeyPatch.context() as mp:
            mp.setitem(__import__("sys").modules, "abstract_mid_engine", mod)
            with pytest.raises(ModelLoadError, match="does not contain"):
                load_external_backend("abstract_mid_engine", _FakeBackend)

    def test_returns_fresh_instance(self) -> None:
        """Each call should return a new instance."""
        mod = types.ModuleType("fresh_engine")
        mod.Backend = _ConcreteFakeBackend  # type: ignore[attr-defined]

        with pytest.MonkeyPatch.context() as mp:
            mp.setitem(__import__("sys").modules, "fresh_engine", mod)
            a = load_external_backend("fresh_engine", _FakeBackend)
            b = load_external_backend("fresh_engine", _FakeBackend)

        assert a is not b

    def test_error_message_includes_class_names_on_ambiguity(self) -> None:
        mod = types.ModuleType("multi_engine")
        mod.Alpha = _ConcreteFakeBackend  # type: ignore[attr-defined]
        mod.Beta = _AnotherConcreteFakeBackend  # type: ignore[attr-defined]

        with pytest.MonkeyPatch.context() as mp:
            mp.setitem(__import__("sys").modules, "multi_engine", mod)
            with pytest.raises(ModelLoadError) as exc_info:
                load_external_backend("multi_engine", _FakeBackend)

        msg = str(exc_info.value)
        assert "_ConcreteFakeBackend" in msg or "_AnotherConcreteFakeBackend" in msg
