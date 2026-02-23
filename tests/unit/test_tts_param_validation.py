"""Tests for TTS parameter validation against engine capabilities."""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock

import grpc
import pytest

from macaw._types import TTSEngineCapabilities
from macaw.workers.tts._validation import validate_params_against_capabilities
from macaw.workers.tts.converters import SynthesizeParams

# ===================================================================
# Helper factories
# ===================================================================


def _kokoro_caps() -> TTSEngineCapabilities:
    """Capabilities matching KokoroBackend."""
    return TTSEngineCapabilities(
        supports_streaming=True,
        supports_alignment=True,
        supports_character_alignment=True,
        supports_seed=False,
        supports_temperature=False,
        supports_top_k=False,
        supports_top_p=False,
        supports_text_normalization=False,
        supports_speed=True,
    )


def _qwen3_caps() -> TTSEngineCapabilities:
    """Capabilities matching Qwen3TTSBackend."""
    return TTSEngineCapabilities(
        supports_streaming=False,
        supports_voice_cloning=True,
        supports_instruct=True,
        supports_seed=True,
        supports_temperature=True,
        supports_top_k=True,
        supports_top_p=True,
        supports_text_normalization=True,
        supports_speed=True,
    )


def _chatterbox_caps() -> TTSEngineCapabilities:
    """Capabilities matching ChatterboxTurboBackend."""
    return TTSEngineCapabilities(
        supports_streaming=False,
        supports_voice_cloning=True,
        supports_seed=False,
        supports_temperature=True,
        supports_top_k=True,
        supports_top_p=True,
        supports_text_normalization=False,
        supports_speed=True,
    )


def _all_supported_caps() -> TTSEngineCapabilities:
    """Capabilities with all params supported."""
    return TTSEngineCapabilities(
        supports_seed=True,
        supports_temperature=True,
        supports_top_k=True,
        supports_top_p=True,
        supports_text_normalization=True,
        supports_speed=True,
    )


def _none_supported_caps() -> TTSEngineCapabilities:
    """Capabilities with no optional params supported."""
    return TTSEngineCapabilities(
        supports_seed=False,
        supports_temperature=False,
        supports_top_k=False,
        supports_top_p=False,
        supports_text_normalization=False,
        supports_speed=False,
    )


def _params(
    *,
    speed: float = 1.0,
    options: dict[str, object] | None = None,
) -> SynthesizeParams:
    return SynthesizeParams(
        text="Hello world",
        voice="default",
        sample_rate=24000,
        speed=speed,
        options=options,
    )


# ===================================================================
# Pure validation function tests
# ===================================================================


class TestValidateAllSupported:
    """When all params are supported, validation returns empty."""

    def test_all_supported_returns_empty(self) -> None:
        """All params supported by engine produces no errors."""
        params = _params(
            speed=1.5,
            options={
                "seed": 42,
                "temperature": 0.9,
                "top_k": 50,
                "top_p": 0.95,
                "text_normalization": "off",
            },
        )
        result = validate_params_against_capabilities(params, _all_supported_caps())
        assert result == []

    def test_options_none_returns_empty(self) -> None:
        """When options is None, no params to validate."""
        params = _params(options=None)
        result = validate_params_against_capabilities(params, _none_supported_caps())
        assert result == []


class TestValidateSingleParam:
    """Each unsupported param is flagged individually."""

    def test_seed_unsupported(self) -> None:
        """Seed in options + supports_seed=False -> ['seed']."""
        caps = TTSEngineCapabilities(supports_seed=False)
        params = _params(options={"seed": 42})
        result = validate_params_against_capabilities(params, caps)
        assert result == ["seed"]

    def test_temperature_unsupported(self) -> None:
        """Temperature in options + supports_temperature=False -> ['temperature']."""
        caps = TTSEngineCapabilities(supports_temperature=False)
        params = _params(options={"temperature": 0.8})
        result = validate_params_against_capabilities(params, caps)
        assert result == ["temperature"]

    def test_top_k_unsupported(self) -> None:
        """Top_k in options + supports_top_k=False -> ['top_k']."""
        caps = TTSEngineCapabilities(supports_top_k=False)
        params = _params(options={"top_k": 50})
        result = validate_params_against_capabilities(params, caps)
        assert result == ["top_k"]

    def test_top_p_unsupported(self) -> None:
        """Top_p in options + supports_top_p=False -> ['top_p']."""
        caps = TTSEngineCapabilities(supports_top_p=False)
        params = _params(options={"top_p": 0.95})
        result = validate_params_against_capabilities(params, caps)
        assert result == ["top_p"]

    def test_text_normalization_unsupported(self) -> None:
        """Text_normalization in options + supports_text_normalization=False."""
        caps = TTSEngineCapabilities(supports_text_normalization=False)
        params = _params(options={"text_normalization": "off"})
        result = validate_params_against_capabilities(params, caps)
        assert result == ["text_normalization"]

    def test_speed_unsupported(self) -> None:
        """Speed != 1.0 + supports_speed=False -> ['speed']."""
        caps = TTSEngineCapabilities(supports_speed=False)
        params = _params(speed=1.5)
        result = validate_params_against_capabilities(params, caps)
        assert result == ["speed"]


class TestValidateDefaultNotFlagged:
    """Default values are not flagged even when unsupported."""

    def test_speed_default_not_flagged(self) -> None:
        """Speed == 1.0 + supports_speed=False -> [] (default not flagged)."""
        caps = TTSEngineCapabilities(supports_speed=False)
        params = _params(speed=1.0)
        result = validate_params_against_capabilities(params, caps)
        assert result == []

    def test_param_not_in_options_not_flagged(self) -> None:
        """Seed not in options + supports_seed=False -> [] (not requested)."""
        caps = TTSEngineCapabilities(supports_seed=False)
        params = _params(options={"language": "en"})
        result = validate_params_against_capabilities(params, caps)
        assert result == []

    def test_empty_options_not_flagged(self) -> None:
        """Empty options dict with nothing supported -> []."""
        params = _params(options={})
        result = validate_params_against_capabilities(params, _none_supported_caps())
        assert result == []


class TestValidateMultipleUnsupported:
    """Multiple unsupported params are returned together."""

    def test_multiple_unsupported(self) -> None:
        """Seed + temperature both unsupported -> ['seed', 'temperature']."""
        caps = TTSEngineCapabilities(
            supports_seed=False,
            supports_temperature=False,
        )
        params = _params(options={"seed": 42, "temperature": 0.8})
        result = validate_params_against_capabilities(params, caps)
        assert "seed" in result
        assert "temperature" in result
        assert len(result) == 2

    def test_all_unsupported(self) -> None:
        """All params unsupported with all params set."""
        params = _params(
            speed=2.0,
            options={
                "seed": 42,
                "temperature": 0.8,
                "top_k": 50,
                "top_p": 0.95,
                "text_normalization": "off",
            },
        )
        result = validate_params_against_capabilities(params, _none_supported_caps())
        assert len(result) == 6
        assert set(result) == {
            "speed",
            "seed",
            "temperature",
            "top_k",
            "top_p",
            "text_normalization",
        }


# ===================================================================
# Engine-specific realistic tests
# ===================================================================


class TestKokoroValidation:
    """Validation against Kokoro capabilities."""

    def test_kokoro_rejects_temperature(self) -> None:
        """Kokoro does not support temperature."""
        params = _params(options={"temperature": 0.8})
        result = validate_params_against_capabilities(params, _kokoro_caps())
        assert result == ["temperature"]

    def test_kokoro_rejects_seed(self) -> None:
        """Kokoro does not support seed."""
        params = _params(options={"seed": 42})
        result = validate_params_against_capabilities(params, _kokoro_caps())
        assert result == ["seed"]

    def test_kokoro_rejects_all_sampling_params(self) -> None:
        """Kokoro rejects all sampling params at once."""
        params = _params(
            options={
                "seed": 42,
                "temperature": 0.8,
                "top_k": 50,
                "top_p": 0.95,
                "text_normalization": "off",
            },
        )
        result = validate_params_against_capabilities(params, _kokoro_caps())
        assert set(result) == {
            "seed",
            "temperature",
            "top_k",
            "top_p",
            "text_normalization",
        }

    def test_kokoro_accepts_speed(self) -> None:
        """Kokoro supports speed."""
        params = _params(speed=2.0)
        result = validate_params_against_capabilities(params, _kokoro_caps())
        assert result == []


class TestQwen3Validation:
    """Validation against Qwen3 capabilities."""

    def test_qwen3_accepts_all(self) -> None:
        """Qwen3 supports all params."""
        params = _params(
            speed=1.5,
            options={
                "seed": 42,
                "temperature": 0.9,
                "top_k": 50,
                "top_p": 0.95,
                "text_normalization": "off",
            },
        )
        result = validate_params_against_capabilities(params, _qwen3_caps())
        assert result == []


class TestChatterboxValidation:
    """Validation against Chatterbox capabilities."""

    def test_chatterbox_rejects_seed(self) -> None:
        """Chatterbox does not support seed."""
        params = _params(options={"seed": 42})
        result = validate_params_against_capabilities(params, _chatterbox_caps())
        assert result == ["seed"]

    def test_chatterbox_rejects_text_normalization(self) -> None:
        """Chatterbox does not support text_normalization."""
        params = _params(options={"text_normalization": "off"})
        result = validate_params_against_capabilities(params, _chatterbox_caps())
        assert result == ["text_normalization"]

    def test_chatterbox_accepts_temperature(self) -> None:
        """Chatterbox supports temperature."""
        params = _params(options={"temperature": 0.8})
        result = validate_params_against_capabilities(params, _chatterbox_caps())
        assert result == []

    def test_chatterbox_accepts_top_k_and_top_p(self) -> None:
        """Chatterbox supports top_k and top_p."""
        params = _params(options={"top_k": 50, "top_p": 0.95})
        result = validate_params_against_capabilities(params, _chatterbox_caps())
        assert result == []


# ===================================================================
# Servicer integration tests (validation wired into Synthesize)
# ===================================================================


class _MockBackendWithCaps:
    """Mock backend that exposes configurable capabilities."""

    def __init__(
        self,
        caps: TTSEngineCapabilities,
        chunks: list[bytes] | None = None,
    ) -> None:
        self._caps = caps
        self._chunks = chunks or [b"\x00\x01" * 100]

    async def capabilities(self) -> TTSEngineCapabilities:
        return self._caps

    async def load(self, model_path: str, config: dict[str, object]) -> None:
        pass

    async def synthesize(  # type: ignore[override, misc]
        self,
        text: str,
        voice: str = "default",
        *,
        sample_rate: int = 24000,
        speed: float = 1.0,
        options: dict[str, object] | None = None,
    ) -> None:
        for chunk in self._chunks:
            yield chunk

    async def voices(self) -> list[object]:
        return []

    async def unload(self) -> None:
        pass

    async def health(self) -> dict[str, str]:
        return {"status": "ok"}


def _make_context() -> MagicMock:
    """Create mock of grpc.aio.ServicerContext."""
    ctx = MagicMock()
    ctx.abort = AsyncMock()
    ctx.cancelled = MagicMock(return_value=False)
    return ctx


class TestServicerRejectsUnsupportedParams:
    """Servicer-level tests: Synthesize rejects unsupported params with INVALID_ARGUMENT."""

    async def test_synthesize_rejects_unsupported_temperature(self) -> None:
        """Synthesize aborts with INVALID_ARGUMENT when temperature is unsupported."""
        from macaw.proto.tts_worker_pb2 import SynthesizeRequest
        from macaw.workers.tts.servicer import TTSWorkerServicer

        caps = TTSEngineCapabilities(supports_temperature=False)
        backend = _MockBackendWithCaps(caps)
        servicer = TTSWorkerServicer(
            backend=backend,  # type: ignore[arg-type]
            model_name="test-model",
            engine="test-engine",
        )

        request = SynthesizeRequest(
            request_id="req-val-1",
            text="Hello world",
            voice="default",
            sample_rate=24000,
            speed=1.0,
            temperature=0.8,
        )
        ctx = _make_context()
        ctx.abort = AsyncMock(
            side_effect=grpc.aio.AbortError(  # type: ignore[attr-defined]
                grpc.StatusCode.INVALID_ARGUMENT,
                "Engine 'test-engine' does not support: temperature",
            )
        )

        with pytest.raises(grpc.aio.AbortError):  # type: ignore[attr-defined]
            async for _chunk in servicer.Synthesize(request, ctx):
                pass  # pragma: no cover

        ctx.abort.assert_called_once_with(
            grpc.StatusCode.INVALID_ARGUMENT,
            "Engine 'test-engine' does not support: temperature",
        )

    async def test_synthesize_rejects_multiple_unsupported(self) -> None:
        """Synthesize aborts listing all unsupported params."""
        from macaw.proto.tts_worker_pb2 import SynthesizeRequest
        from macaw.workers.tts.servicer import TTSWorkerServicer

        caps = TTSEngineCapabilities(
            supports_seed=False,
            supports_temperature=False,
        )
        backend = _MockBackendWithCaps(caps)
        servicer = TTSWorkerServicer(
            backend=backend,  # type: ignore[arg-type]
            model_name="test-model",
            engine="test-engine",
        )

        request = SynthesizeRequest(
            request_id="req-val-2",
            text="Hello world",
            voice="default",
            sample_rate=24000,
            speed=1.0,
            seed=42,
            temperature=0.8,
        )
        ctx = _make_context()
        ctx.abort = AsyncMock(
            side_effect=grpc.aio.AbortError(  # type: ignore[attr-defined]
                grpc.StatusCode.INVALID_ARGUMENT, "unsupported"
            )
        )

        with pytest.raises(grpc.aio.AbortError):  # type: ignore[attr-defined]
            async for _chunk in servicer.Synthesize(request, ctx):
                pass  # pragma: no cover

        ctx.abort.assert_called_once()
        call_args = ctx.abort.call_args
        assert call_args[0][0] == grpc.StatusCode.INVALID_ARGUMENT
        assert "seed" in call_args[0][1]
        assert "temperature" in call_args[0][1]

    async def test_synthesize_passes_when_all_supported(self) -> None:
        """Synthesize proceeds normally when all params are supported."""
        from macaw.proto.tts_worker_pb2 import SynthesizeChunk, SynthesizeRequest
        from macaw.workers.tts.servicer import TTSWorkerServicer

        caps = TTSEngineCapabilities(
            supports_seed=True,
            supports_temperature=True,
        )
        backend = _MockBackendWithCaps(caps)
        servicer = TTSWorkerServicer(
            backend=backend,  # type: ignore[arg-type]
            model_name="test-model",
            engine="test-engine",
        )

        request = SynthesizeRequest(
            request_id="req-val-3",
            text="Hello world",
            voice="default",
            sample_rate=24000,
            speed=1.0,
            seed=42,
            temperature=0.8,
        )
        ctx = _make_context()

        chunks: list[SynthesizeChunk] = []
        async for chunk in servicer.Synthesize(request, ctx):
            chunks.append(chunk)

        # Should have audio chunk(s) + final chunk
        assert len(chunks) >= 2
        assert chunks[-1].is_last is True
        # abort should NOT have been called
        ctx.abort.assert_not_called()

    async def test_synthesize_passes_when_no_params_set(self) -> None:
        """Synthesize proceeds when no optional params are set, even if unsupported."""
        from macaw.proto.tts_worker_pb2 import SynthesizeChunk, SynthesizeRequest
        from macaw.workers.tts.servicer import TTSWorkerServicer

        caps = TTSEngineCapabilities(
            supports_seed=False,
            supports_temperature=False,
        )
        backend = _MockBackendWithCaps(caps)
        servicer = TTSWorkerServicer(
            backend=backend,  # type: ignore[arg-type]
            model_name="test-model",
            engine="test-engine",
        )

        request = SynthesizeRequest(
            request_id="req-val-4",
            text="Hello world",
            voice="default",
            sample_rate=24000,
            speed=1.0,
        )
        ctx = _make_context()

        chunks: list[SynthesizeChunk] = []
        async for chunk in servicer.Synthesize(request, ctx):
            chunks.append(chunk)

        assert len(chunks) >= 2
        ctx.abort.assert_not_called()
