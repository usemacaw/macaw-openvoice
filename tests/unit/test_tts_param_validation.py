"""Tests for TTS parameter validation against engine capabilities."""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock

import grpc

from macaw._types import TTSEngineCapabilities
from macaw.workers.tts._validation import (
    CapabilityValidator,
    ChatterboxValidator,
    KokoroValidator,
    ParamValidator,
    Qwen3TTSValidator,
    SamplingBoundsValidator,
    SpeedValidator,
    _is_numeric,
    get_validators_for_engine,
    register_engine_validators,
    validate_params,
    validate_params_against_capabilities,
)
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
        supports_text_normalization=True,
        supports_speed=True,
    )


def _chatterbox_caps() -> TTSEngineCapabilities:
    """Capabilities matching ChatterboxTurboBackend."""
    return TTSEngineCapabilities(
        supports_streaming=False,
        supports_voice_cloning=True,
        supports_seed=False,
        supports_text_normalization=False,
        supports_speed=True,
    )


def _all_supported_caps() -> TTSEngineCapabilities:
    """Capabilities with all params supported."""
    return TTSEngineCapabilities(
        supports_seed=True,
        supports_text_normalization=True,
        supports_speed=True,
    )


def _none_supported_caps() -> TTSEngineCapabilities:
    """Capabilities with no optional params supported."""
    return TTSEngineCapabilities(
        supports_seed=False,
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

    def test_seed_always_accepted(self) -> None:
        """Seed is never rejected — deterministic engines satisfy reproducibility."""
        caps = TTSEngineCapabilities(supports_seed=False)
        params = _params(options={"seed": 42})
        result = validate_params_against_capabilities(params, caps)
        assert result == []

    def test_temperature_always_accepted(self) -> None:
        """Temperature is never rejected by default validators."""
        caps = TTSEngineCapabilities()
        params = _params(options={"temperature": 0.8})
        result = validate_params_against_capabilities(params, caps)
        assert result == []

    def test_top_k_always_accepted(self) -> None:
        """Top_k within bounds is never rejected by default validators."""
        caps = TTSEngineCapabilities()
        params = _params(options={"top_k": 50})
        result = validate_params_against_capabilities(params, caps)
        assert result == []

    def test_top_p_always_accepted(self) -> None:
        """Top_p within bounds is never rejected by default validators."""
        caps = TTSEngineCapabilities()
        params = _params(options={"top_p": 0.95})
        result = validate_params_against_capabilities(params, caps)
        assert result == []

    def test_text_normalization_always_accepted(self) -> None:
        """Text normalization is never rejected — backends handle gracefully."""
        caps = TTSEngineCapabilities(supports_text_normalization=False)
        params = _params(options={"text_normalization": "off"})
        result = validate_params_against_capabilities(params, caps)
        assert result == []

    def test_speed_unsupported(self) -> None:
        """Speed != 1.0 + supports_speed=False -> error mentioning 'speed'."""
        caps = TTSEngineCapabilities(supports_speed=False)
        params = _params(speed=1.5)
        result = validate_params_against_capabilities(params, caps)
        assert len(result) == 1
        assert "speed" in result[0]


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

    def test_multiple_options_all_accepted(self) -> None:
        """Seed and temperature are both accepted by default validators."""
        caps = TTSEngineCapabilities(supports_seed=False)
        params = _params(options={"seed": 42, "temperature": 0.8})
        result = validate_params_against_capabilities(params, caps)
        assert result == []

    def test_only_speed_flagged_when_all_unsupported(self) -> None:
        """All params unsupported — only speed capability is flagged."""
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
        assert len(result) == 1
        assert "speed" in result[0]


# ===================================================================
# Engine-specific realistic tests
# ===================================================================


class TestKokoroValidation:
    """Validation against Kokoro capabilities."""

    def test_kokoro_accepts_temperature(self) -> None:
        """Kokoro accepts temperature — deterministic engine ignores it."""
        params = _params(options={"temperature": 0.8})
        result = validate_params_against_capabilities(params, _kokoro_caps())
        assert result == []

    def test_kokoro_accepts_seed(self) -> None:
        """Kokoro accepts seed — deterministic engine satisfies reproducibility."""
        params = _params(options={"seed": 42})
        result = validate_params_against_capabilities(params, _kokoro_caps())
        assert result == []

    def test_kokoro_accepts_all_sampling_params(self) -> None:
        """Kokoro accepts all sampling params — deterministic engine ignores them."""
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
        assert result == []

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

    def test_chatterbox_accepts_seed(self) -> None:
        """Chatterbox accepts seed — deterministic engines satisfy reproducibility."""
        params = _params(options={"seed": 42})
        result = validate_params_against_capabilities(params, _chatterbox_caps())
        assert result == []

    def test_chatterbox_accepts_text_normalization(self) -> None:
        """Chatterbox accepts text_normalization — backend handles gracefully."""
        params = _params(options={"text_normalization": "off"})
        result = validate_params_against_capabilities(params, _chatterbox_caps())
        assert result == []

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

    async def test_synthesize_accepts_temperature_on_default_caps(self) -> None:
        """Synthesize proceeds when temperature is sent with default capabilities."""
        from macaw.proto.tts_worker_pb2 import SynthesizeChunk, SynthesizeRequest
        from macaw.workers.tts.servicer import TTSWorkerServicer

        caps = TTSEngineCapabilities()
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

        chunks: list[SynthesizeChunk] = []
        async for chunk in servicer.Synthesize(request, ctx):
            chunks.append(chunk)

        assert len(chunks) >= 2
        assert chunks[-1].is_last is True
        ctx.abort.assert_not_called()

    async def test_synthesize_accepts_temperature_and_seed_on_default_caps(self) -> None:
        """Synthesize proceeds with seed + temperature on default capabilities."""
        from macaw.proto.tts_worker_pb2 import SynthesizeChunk, SynthesizeRequest
        from macaw.workers.tts.servicer import TTSWorkerServicer

        caps = TTSEngineCapabilities(supports_seed=False)
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

        chunks: list[SynthesizeChunk] = []
        async for chunk in servicer.Synthesize(request, ctx):
            chunks.append(chunk)

        assert len(chunks) >= 2
        ctx.abort.assert_not_called()

    async def test_synthesize_passes_when_all_supported(self) -> None:
        """Synthesize proceeds normally when all params are supported."""
        from macaw.proto.tts_worker_pb2 import SynthesizeChunk, SynthesizeRequest
        from macaw.workers.tts.servicer import TTSWorkerServicer

        caps = TTSEngineCapabilities(supports_seed=True)
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

        caps = TTSEngineCapabilities(supports_seed=False)
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

    async def test_synthesize_rejects_top_k_above_bounds(self) -> None:
        """Synthesize aborts when top_k exceeds safety bounds."""
        from macaw.proto.tts_worker_pb2 import SynthesizeRequest
        from macaw.workers.tts.servicer import TTSWorkerServicer

        caps = TTSEngineCapabilities()
        backend = _MockBackendWithCaps(caps)
        servicer = TTSWorkerServicer(
            backend=backend,  # type: ignore[arg-type]
            model_name="test-model",
            engine="test-engine",
        )

        request = SynthesizeRequest(
            request_id="req-bounds-1",
            text="Hello world",
            voice="default",
            sample_rate=24000,
            speed=1.0,
            top_k=5000,
        )
        ctx = _make_context()

        chunks = []
        async for chunk in servicer.Synthesize(request, ctx):
            chunks.append(chunk)

        ctx.abort.assert_called_once()
        call_args = ctx.abort.call_args
        assert call_args[0][0] == grpc.StatusCode.INVALID_ARGUMENT
        assert "top_k" in call_args[0][1]

    async def test_synthesize_rejects_speed_out_of_bounds(self) -> None:
        """Synthesize aborts when speed is outside 0.25-4.0 range."""
        from macaw.proto.tts_worker_pb2 import SynthesizeRequest
        from macaw.workers.tts.servicer import TTSWorkerServicer

        caps = TTSEngineCapabilities(supports_speed=True)
        backend = _MockBackendWithCaps(caps)
        servicer = TTSWorkerServicer(
            backend=backend,  # type: ignore[arg-type]
            model_name="test-model",
            engine="test-engine",
        )

        request = SynthesizeRequest(
            request_id="req-bounds-2",
            text="Hello world",
            voice="default",
            sample_rate=24000,
            speed=10.0,
        )
        ctx = _make_context()

        chunks = []
        async for chunk in servicer.Synthesize(request, ctx):
            chunks.append(chunk)

        ctx.abort.assert_called_once()
        call_args = ctx.abort.call_args
        assert call_args[0][0] == grpc.StatusCode.INVALID_ARGUMENT
        assert "speed" in call_args[0][1]


# ===================================================================
# ParamValidator Protocol tests
# ===================================================================


class TestParamValidatorProtocol:
    """Verify the runtime_checkable Protocol works correctly."""

    def test_speed_validator_is_param_validator(self) -> None:
        """SpeedValidator satisfies ParamValidator Protocol."""
        assert isinstance(SpeedValidator(), ParamValidator)

    def test_capability_validator_is_param_validator(self) -> None:
        """CapabilityValidator satisfies ParamValidator Protocol."""
        assert isinstance(CapabilityValidator(), ParamValidator)

    def test_sampling_bounds_validator_is_param_validator(self) -> None:
        """SamplingBoundsValidator satisfies ParamValidator Protocol."""
        assert isinstance(SamplingBoundsValidator(), ParamValidator)

    def test_kokoro_validator_is_param_validator(self) -> None:
        """KokoroValidator satisfies ParamValidator Protocol."""
        assert isinstance(KokoroValidator(), ParamValidator)

    def test_chatterbox_validator_is_param_validator(self) -> None:
        """ChatterboxValidator satisfies ParamValidator Protocol."""
        assert isinstance(ChatterboxValidator(), ParamValidator)

    def test_qwen3_validator_is_param_validator(self) -> None:
        """Qwen3TTSValidator satisfies ParamValidator Protocol."""
        assert isinstance(Qwen3TTSValidator(), ParamValidator)

    def test_arbitrary_object_not_param_validator(self) -> None:
        """An object without validate() is not a ParamValidator."""
        assert not isinstance("not a validator", ParamValidator)


# ===================================================================
# SpeedValidator (isolated) tests
# ===================================================================


class TestSpeedValidator:
    """Test SpeedValidator in isolation."""

    def test_default_speed_always_passes(self) -> None:
        """speed=1.0 is never flagged, even when unsupported."""
        v = SpeedValidator()
        result = v.validate(_params(speed=1.0), _none_supported_caps())
        assert result == []

    def test_non_default_speed_supported(self) -> None:
        """Non-default speed with supports_speed=True passes."""
        v = SpeedValidator()
        caps = TTSEngineCapabilities(supports_speed=True)
        result = v.validate(_params(speed=2.0), caps)
        assert result == []

    def test_non_default_speed_unsupported(self) -> None:
        """Non-default speed with supports_speed=False returns error."""
        v = SpeedValidator()
        caps = TTSEngineCapabilities(supports_speed=False)
        result = v.validate(_params(speed=1.5), caps)
        assert len(result) == 1
        assert "speed" in result[0]

    def test_speed_at_boundary_supported(self) -> None:
        """Speed=0.25 (minimum) with supports_speed=True passes."""
        v = SpeedValidator()
        caps = TTSEngineCapabilities(supports_speed=True)
        result = v.validate(_params(speed=0.25), caps)
        assert result == []


# ===================================================================
# SamplingBoundsValidator tests
# ===================================================================


class TestSamplingBoundsValidator:
    """Test SamplingBoundsValidator in isolation."""

    def test_top_k_within_bounds(self) -> None:
        """top_k=1000 (boundary) passes."""
        v = SamplingBoundsValidator()
        result = v.validate(_params(options={"top_k": 1000}), _all_supported_caps())
        assert result == []

    def test_top_k_above_bounds(self) -> None:
        """top_k=1001 is rejected."""
        v = SamplingBoundsValidator()
        result = v.validate(_params(options={"top_k": 1001}), _all_supported_caps())
        assert len(result) == 1
        assert "top_k" in result[0]

    def test_temperature_within_bounds(self) -> None:
        """temperature=2.0 (boundary) passes."""
        v = SamplingBoundsValidator()
        result = v.validate(_params(options={"temperature": 2.0}), _all_supported_caps())
        assert result == []

    def test_temperature_above_bounds(self) -> None:
        """temperature=2.5 is rejected."""
        v = SamplingBoundsValidator()
        result = v.validate(_params(options={"temperature": 2.5}), _all_supported_caps())
        assert len(result) == 1
        assert "temperature" in result[0]

    def test_top_p_within_bounds(self) -> None:
        """top_p=1.0 (boundary) passes."""
        v = SamplingBoundsValidator()
        result = v.validate(_params(options={"top_p": 1.0}), _all_supported_caps())
        assert result == []

    def test_top_p_above_bounds(self) -> None:
        """top_p=1.5 is rejected."""
        v = SamplingBoundsValidator()
        result = v.validate(_params(options={"top_p": 1.5}), _all_supported_caps())
        assert len(result) == 1
        assert "top_p" in result[0]

    def test_speed_below_minimum(self) -> None:
        """speed=0.1 is below 0.25 minimum."""
        v = SamplingBoundsValidator()
        result = v.validate(_params(speed=0.1), _all_supported_caps())
        assert len(result) == 1
        assert "speed" in result[0]

    def test_speed_above_maximum(self) -> None:
        """speed=5.0 is above 4.0 maximum."""
        v = SamplingBoundsValidator()
        result = v.validate(_params(speed=5.0), _all_supported_caps())
        assert len(result) == 1
        assert "speed" in result[0]

    def test_speed_at_boundaries(self) -> None:
        """speed=0.25 and speed=4.0 both pass."""
        v = SamplingBoundsValidator()
        assert v.validate(_params(speed=0.25), _all_supported_caps()) == []
        assert v.validate(_params(speed=4.0), _all_supported_caps()) == []

    def test_multiple_violations(self) -> None:
        """Multiple out-of-bounds values produce multiple errors."""
        v = SamplingBoundsValidator()
        params = _params(
            speed=10.0,
            options={"top_k": 5000, "temperature": 3.0},
        )
        result = v.validate(params, _all_supported_caps())
        assert len(result) == 3
        texts = " ".join(result)
        assert "top_k" in texts
        assert "temperature" in texts
        assert "speed" in texts

    def test_non_numeric_values_ignored(self) -> None:
        """Non-numeric option values are silently ignored."""
        v = SamplingBoundsValidator()
        params = _params(options={"top_k": "invalid", "temperature": None})
        result = v.validate(params, _all_supported_caps())
        assert result == []

    def test_no_options(self) -> None:
        """No options dict produces no errors (speed=1.0 default in bounds)."""
        v = SamplingBoundsValidator()
        result = v.validate(_params(), _all_supported_caps())
        assert result == []

    def test_empty_options(self) -> None:
        """Empty options dict produces no errors."""
        v = SamplingBoundsValidator()
        result = v.validate(_params(options={}), _all_supported_caps())
        assert result == []

    def test_top_k_zero_passes(self) -> None:
        """top_k=0 (engine default) passes."""
        v = SamplingBoundsValidator()
        result = v.validate(_params(options={"top_k": 0}), _all_supported_caps())
        assert result == []

    def test_top_k_negative_rejected(self) -> None:
        """Negative top_k is rejected (lower bound defense-in-depth)."""
        v = SamplingBoundsValidator()
        result = v.validate(_params(options={"top_k": -1}), _all_supported_caps())
        assert len(result) == 1
        assert "top_k" in result[0]
        assert ">= 0" in result[0]

    def test_temperature_negative_rejected(self) -> None:
        """Negative temperature is rejected (lower bound defense-in-depth)."""
        v = SamplingBoundsValidator()
        result = v.validate(_params(options={"temperature": -0.5}), _all_supported_caps())
        assert len(result) == 1
        assert "temperature" in result[0]
        assert ">= 0" in result[0]

    def test_top_p_negative_rejected(self) -> None:
        """Negative top_p is rejected (lower bound defense-in-depth)."""
        v = SamplingBoundsValidator()
        result = v.validate(_params(options={"top_p": -0.1}), _all_supported_caps())
        assert len(result) == 1
        assert "top_p" in result[0]
        assert ">= 0" in result[0]

    def test_all_negative_bounds_rejected(self) -> None:
        """All three negative sampling params are rejected together."""
        v = SamplingBoundsValidator()
        params = _params(options={"top_k": -10, "temperature": -1.0, "top_p": -0.5})
        result = v.validate(params, _all_supported_caps())
        assert len(result) == 3


# ===================================================================
# validate_params composition function tests
# ===================================================================


class TestValidateParams:
    """Test the validate_params runner function."""

    def test_empty_validators(self) -> None:
        """No validators produces no errors."""
        result = validate_params(_params(), _all_supported_caps(), ())
        assert result == []

    def test_single_validator(self) -> None:
        """Single validator errors are propagated."""
        caps = TTSEngineCapabilities(supports_speed=False)
        result = validate_params(_params(speed=2.0), caps, (SpeedValidator(),))
        assert len(result) == 1
        assert "speed" in result[0]

    def test_multiple_validators_accumulate(self) -> None:
        """Errors from multiple validators are accumulated."""
        caps = TTSEngineCapabilities(supports_speed=False)
        params = _params(speed=10.0, options={"top_k": 5000})
        result = validate_params(params, caps, (SpeedValidator(), SamplingBoundsValidator()))
        # SpeedValidator: 1 error (not supported)
        # SamplingBoundsValidator: 1 error (top_k out of bounds; speed bounds
        # skipped because supports_speed=False)
        assert len(result) == 2

    def test_validators_run_in_order(self) -> None:
        """Validators run in the order they appear in the tuple."""
        caps = TTSEngineCapabilities(supports_speed=True)
        result = validate_params(
            _params(speed=10.0, options={"top_k": 5000}),
            caps,
            (SpeedValidator(), SamplingBoundsValidator()),
        )
        # SpeedValidator: no error (supports_speed=True, speed != 1.0 is ok)
        # SamplingBoundsValidator: top_k first, then speed
        assert "top_k" in result[0]
        assert "speed" in result[1]


# ===================================================================
# get_validators_for_engine registry tests
# ===================================================================


class TestGetValidatorsForEngine:
    """Test the engine validator registry."""

    def test_unknown_engine_returns_defaults_only(self) -> None:
        """Unknown engine returns only the 3 default validators."""
        validators = get_validators_for_engine("unknown-engine")
        assert len(validators) == 3
        assert isinstance(validators[0], SpeedValidator)
        assert isinstance(validators[1], CapabilityValidator)
        assert isinstance(validators[2], SamplingBoundsValidator)

    def test_kokoro_returns_defaults_plus_engine_specific(self) -> None:
        """Kokoro returns 3 defaults + KokoroValidator."""
        validators = get_validators_for_engine("kokoro")
        assert len(validators) == 4
        assert isinstance(validators[3], KokoroValidator)

    def test_chatterbox_returns_defaults_plus_engine_specific(self) -> None:
        """Chatterbox-turbo returns 3 defaults + ChatterboxValidator."""
        validators = get_validators_for_engine("chatterbox")
        assert len(validators) == 4
        assert isinstance(validators[3], ChatterboxValidator)

    def test_qwen3_returns_defaults_plus_engine_specific(self) -> None:
        """Qwen3-TTS returns 3 defaults + Qwen3TTSValidator."""
        validators = get_validators_for_engine("qwen3-tts")
        assert len(validators) == 4
        assert isinstance(validators[3], Qwen3TTSValidator)

    def test_returns_tuple(self) -> None:
        """Return type is always a tuple."""
        validators = get_validators_for_engine("test")
        assert isinstance(validators, tuple)


# ===================================================================
# Backward compatibility tests
# ===================================================================


class TestBackwardCompatibility:
    """Ensure validate_params_against_capabilities wrapper works as before."""

    def test_wrapper_returns_empty_for_valid_params(self) -> None:
        """Backward-compat wrapper returns empty for valid params."""
        result = validate_params_against_capabilities(_params(), _all_supported_caps())
        assert result == []

    def test_wrapper_returns_errors_for_invalid_params(self) -> None:
        """Backward-compat wrapper returns errors for invalid params."""
        caps = TTSEngineCapabilities(supports_speed=False)
        result = validate_params_against_capabilities(_params(speed=1.5), caps)
        assert len(result) == 1
        assert "speed" in result[0]

    def test_wrapper_detects_bounds_violations(self) -> None:
        """Backward-compat wrapper includes SamplingBoundsValidator."""
        result = validate_params_against_capabilities(
            _params(options={"top_k": 5000}),
            _all_supported_caps(),
        )
        assert len(result) == 1
        assert "top_k" in result[0]

    def test_wrapper_detects_capability_violations(self) -> None:
        """Backward-compat wrapper includes CapabilityValidator."""
        caps = TTSEngineCapabilities(supports_voice_cloning=False)
        result = validate_params_against_capabilities(
            _params(options={"ref_audio": b"fake-audio"}),
            caps,
        )
        assert len(result) == 1
        assert "ref_audio" in result[0]


# ===================================================================
# CapabilityValidator tests
# ===================================================================


class TestCapabilityValidator:
    """Test CapabilityValidator — feature-gated param checks."""

    def test_ref_audio_rejected_without_voice_cloning(self) -> None:
        """ref_audio is rejected when engine lacks voice cloning."""
        v = CapabilityValidator()
        caps = TTSEngineCapabilities(supports_voice_cloning=False)
        result = v.validate(_params(options={"ref_audio": b"audio-data"}), caps)
        assert len(result) == 1
        assert "ref_audio" in result[0]
        assert "voice cloning" in result[0]

    def test_ref_text_rejected_without_voice_cloning(self) -> None:
        """ref_text is rejected when engine lacks voice cloning."""
        v = CapabilityValidator()
        caps = TTSEngineCapabilities(supports_voice_cloning=False)
        result = v.validate(_params(options={"ref_text": "hello"}), caps)
        assert len(result) == 1
        assert "ref_text" in result[0]

    def test_instruction_rejected_without_instruct(self) -> None:
        """instruction is rejected when engine lacks instruct support."""
        v = CapabilityValidator()
        caps = TTSEngineCapabilities(supports_instruct=False)
        result = v.validate(_params(options={"instruction": "speak like a pirate"}), caps)
        assert len(result) == 1
        assert "instruction" in result[0]
        assert "instruct" in result[0]

    def test_ref_audio_accepted_with_voice_cloning(self) -> None:
        """ref_audio passes when engine supports voice cloning."""
        v = CapabilityValidator()
        caps = TTSEngineCapabilities(supports_voice_cloning=True)
        result = v.validate(_params(options={"ref_audio": b"audio-data"}), caps)
        assert result == []

    def test_instruction_accepted_with_instruct(self) -> None:
        """instruction passes when engine supports instruct."""
        v = CapabilityValidator()
        caps = TTSEngineCapabilities(supports_instruct=True)
        result = v.validate(_params(options={"instruction": "speak softly"}), caps)
        assert result == []

    def test_multiple_capability_violations(self) -> None:
        """Multiple violations produce multiple errors."""
        v = CapabilityValidator()
        caps = TTSEngineCapabilities(
            supports_voice_cloning=False,
            supports_instruct=False,
        )
        params = _params(
            options={
                "ref_audio": b"audio",
                "ref_text": "text",
                "instruction": "be loud",
            }
        )
        result = v.validate(params, caps)
        assert len(result) == 3

    def test_no_options_passes(self) -> None:
        """No options dict produces no errors."""
        v = CapabilityValidator()
        caps = TTSEngineCapabilities(supports_voice_cloning=False)
        result = v.validate(_params(), caps)
        assert result == []

    def test_empty_ref_audio_not_flagged(self) -> None:
        """Empty bytes ref_audio is falsy — not flagged."""
        v = CapabilityValidator()
        caps = TTSEngineCapabilities(supports_voice_cloning=False)
        result = v.validate(_params(options={"ref_audio": b""}), caps)
        assert result == []

    def test_kokoro_caps_reject_voice_cloning_params(self) -> None:
        """Kokoro (no voice cloning) rejects ref_audio."""
        v = CapabilityValidator()
        result = v.validate(
            _params(options={"ref_audio": b"audio", "instruction": "test"}),
            _kokoro_caps(),
        )
        # Kokoro: supports_voice_cloning=False, supports_instruct=False (default)
        assert len(result) == 2

    def test_qwen3_caps_accept_voice_cloning_params(self) -> None:
        """Qwen3 (voice cloning + instruct) accepts both."""
        v = CapabilityValidator()
        result = v.validate(
            _params(options={"ref_audio": b"audio", "instruction": "test"}),
            _qwen3_caps(),
        )
        assert result == []

    def test_chatterbox_caps_accept_ref_audio_reject_instruction(self) -> None:
        """Chatterbox (voice cloning but no instruct) accepts ref_audio, rejects instruction."""
        v = CapabilityValidator()
        result = v.validate(
            _params(options={"ref_audio": b"audio", "instruction": "test"}),
            _chatterbox_caps(),
        )
        assert len(result) == 1
        assert "instruction" in result[0]


# ===================================================================
# KokoroValidator tests
# ===================================================================


class TestKokoroValidator:
    """Test KokoroValidator — reject no-op sampling params."""

    def test_temperature_rejected(self) -> None:
        """Non-zero temperature rejected on Kokoro."""
        v = KokoroValidator()
        result = v.validate(_params(options={"temperature": 0.8}), _kokoro_caps())
        assert len(result) == 1
        assert "temperature" in result[0]
        assert "Kokoro" in result[0]

    def test_top_k_rejected(self) -> None:
        """Non-zero top_k rejected on Kokoro."""
        v = KokoroValidator()
        result = v.validate(_params(options={"top_k": 50}), _kokoro_caps())
        assert len(result) == 1
        assert "top_k" in result[0]

    def test_top_p_rejected(self) -> None:
        """Non-zero top_p rejected on Kokoro."""
        v = KokoroValidator()
        result = v.validate(_params(options={"top_p": 0.95}), _kokoro_caps())
        assert len(result) == 1
        assert "top_p" in result[0]

    def test_zero_values_accepted(self) -> None:
        """Zero (proto default / not set) values are not flagged."""
        v = KokoroValidator()
        result = v.validate(
            _params(options={"temperature": 0, "top_k": 0, "top_p": 0}),
            _kokoro_caps(),
        )
        assert result == []

    def test_multiple_sampling_params_rejected(self) -> None:
        """All three sampling params rejected together."""
        v = KokoroValidator()
        params = _params(options={"temperature": 0.8, "top_k": 50, "top_p": 0.95})
        result = v.validate(params, _kokoro_caps())
        assert len(result) == 3

    def test_no_options_accepted(self) -> None:
        """No options dict produces no errors."""
        v = KokoroValidator()
        result = v.validate(_params(), _kokoro_caps())
        assert result == []

    def test_seed_not_checked(self) -> None:
        """KokoroValidator does NOT check seed (it's a default validator concern)."""
        v = KokoroValidator()
        result = v.validate(_params(options={"seed": 42}), _kokoro_caps())
        assert result == []

    def test_non_numeric_ignored(self) -> None:
        """Non-numeric values silently ignored."""
        v = KokoroValidator()
        result = v.validate(
            _params(options={"temperature": "high", "top_k": None}),
            _kokoro_caps(),
        )
        assert result == []

    def test_servicer_rejects_kokoro_temperature(self) -> None:
        """Servicer with engine='kokoro' rejects temperature via KokoroValidator."""
        # Verify the full chain fires for kokoro engine
        validators = get_validators_for_engine("kokoro")
        params = _params(options={"temperature": 0.8})
        result = validate_params(params, _kokoro_caps(), validators)
        assert any("temperature" in e and "Kokoro" in e for e in result)


# ===================================================================
# ChatterboxValidator tests
# ===================================================================


class TestChatterboxValidator:
    """Test ChatterboxValidator — reject seed."""

    def test_seed_rejected(self) -> None:
        """Non-zero seed rejected on Chatterbox."""
        v = ChatterboxValidator()
        result = v.validate(_params(options={"seed": 42}), _chatterbox_caps())
        assert len(result) == 1
        assert "seed" in result[0]
        assert "Chatterbox" in result[0]

    def test_seed_zero_accepted(self) -> None:
        """seed=0 (proto default / not set) is not flagged."""
        v = ChatterboxValidator()
        result = v.validate(_params(options={"seed": 0}), _chatterbox_caps())
        assert result == []

    def test_no_options_accepted(self) -> None:
        """No options dict produces no errors."""
        v = ChatterboxValidator()
        result = v.validate(_params(), _chatterbox_caps())
        assert result == []

    def test_temperature_not_checked(self) -> None:
        """ChatterboxValidator does NOT check temperature (Chatterbox supports it)."""
        v = ChatterboxValidator()
        result = v.validate(_params(options={"temperature": 0.8}), _chatterbox_caps())
        assert result == []

    def test_servicer_rejects_chatterbox_seed(self) -> None:
        """Full validator chain for chatterbox rejects seed."""
        validators = get_validators_for_engine("chatterbox")
        params = _params(options={"seed": 42})
        result = validate_params(params, _chatterbox_caps(), validators)
        assert any("seed" in e and "Chatterbox" in e for e in result)

    def test_servicer_accepts_chatterbox_temperature(self) -> None:
        """Full validator chain for chatterbox accepts temperature."""
        validators = get_validators_for_engine("chatterbox")
        params = _params(options={"temperature": 0.8})
        result = validate_params(params, _chatterbox_caps(), validators)
        assert result == []


# ===================================================================
# Qwen3TTSValidator tests
# ===================================================================


class TestQwen3TTSValidator:
    """Test Qwen3TTSValidator — instruction + text_normalization conflict."""

    def test_conflict_rejected(self) -> None:
        """instruction + text_normalization='off' is rejected."""
        v = Qwen3TTSValidator()
        params = _params(
            options={"instruction": "speak softly", "text_normalization": "off"},
        )
        result = v.validate(params, _qwen3_caps())
        assert len(result) == 1
        assert "instruction" in result[0]
        assert "text_normalization" in result[0]
        assert "Qwen3" in result[0]

    def test_instruction_only_accepted(self) -> None:
        """instruction without text_normalization='off' is fine."""
        v = Qwen3TTSValidator()
        params = _params(options={"instruction": "speak softly"})
        result = v.validate(params, _qwen3_caps())
        assert result == []

    def test_text_norm_off_only_accepted(self) -> None:
        """text_normalization='off' without instruction is fine."""
        v = Qwen3TTSValidator()
        params = _params(options={"text_normalization": "off"})
        result = v.validate(params, _qwen3_caps())
        assert result == []

    def test_text_norm_auto_with_instruction_accepted(self) -> None:
        """text_normalization='auto' + instruction is fine (no conflict)."""
        v = Qwen3TTSValidator()
        params = _params(
            options={"instruction": "speak softly", "text_normalization": "auto"},
        )
        result = v.validate(params, _qwen3_caps())
        assert result == []

    def test_text_norm_on_with_instruction_accepted(self) -> None:
        """text_normalization='on' + instruction is fine (no conflict)."""
        v = Qwen3TTSValidator()
        params = _params(
            options={"instruction": "be loud", "text_normalization": "on"},
        )
        result = v.validate(params, _qwen3_caps())
        assert result == []

    def test_no_options_accepted(self) -> None:
        """No options dict produces no errors."""
        v = Qwen3TTSValidator()
        result = v.validate(_params(), _qwen3_caps())
        assert result == []

    def test_empty_instruction_not_flagged(self) -> None:
        """Empty instruction string is falsy — not flagged even with off."""
        v = Qwen3TTSValidator()
        params = _params(
            options={"instruction": "", "text_normalization": "off"},
        )
        result = v.validate(params, _qwen3_caps())
        assert result == []

    def test_non_string_text_norm_ignored(self) -> None:
        """Non-string text_normalization values do not match 'off'."""
        v = Qwen3TTSValidator()
        params = _params(
            options={"instruction": "speak fast", "text_normalization": False},
        )
        result = v.validate(params, _qwen3_caps())
        assert result == []

    def test_servicer_rejects_qwen3_conflict(self) -> None:
        """Full validator chain for qwen3-tts rejects instruction + off conflict."""
        validators = get_validators_for_engine("qwen3-tts")
        params = _params(
            options={"instruction": "speak softly", "text_normalization": "off"},
        )
        result = validate_params(params, _qwen3_caps(), validators)
        assert any("instruction" in e and "Qwen3" in e for e in result)


# ===================================================================
# Qwen3-TTS full-chain validation tests
# ===================================================================


class TestQwen3FullChainValidation:
    """Qwen3-TTS: all params accepted when no conflict exists."""

    def test_qwen3_accepts_all_params_via_chain(self) -> None:
        """Full validator chain for qwen3-tts accepts all params (no conflict)."""
        validators = get_validators_for_engine("qwen3-tts")
        params = _params(
            speed=1.5,
            options={
                "seed": 42,
                "temperature": 0.9,
                "top_k": 50,
                "top_p": 0.95,
                "ref_audio": b"audio-data",
                "instruction": "speak softly",
            },
        )
        result = validate_params(params, _qwen3_caps(), validators)
        assert result == []

    def test_qwen3_still_bounded_by_safety(self) -> None:
        """Qwen3 params are still bounded by SamplingBoundsValidator."""
        validators = get_validators_for_engine("qwen3-tts")
        params = _params(options={"top_k": 5000})
        result = validate_params(params, _qwen3_caps(), validators)
        assert len(result) == 1
        assert "top_k" in result[0]


# ===================================================================
# _is_numeric helper tests (M-2: bool exclusion)
# ===================================================================


class TestIsNumeric:
    """Test _is_numeric helper guards against bool subclass of int."""

    def test_int_is_numeric(self) -> None:
        assert _is_numeric(42) is True

    def test_float_is_numeric(self) -> None:
        assert _is_numeric(3.14) is True

    def test_zero_is_numeric(self) -> None:
        assert _is_numeric(0) is True

    def test_negative_is_numeric(self) -> None:
        assert _is_numeric(-1) is True

    def test_bool_true_not_numeric(self) -> None:
        """bool subclass of int must be excluded."""
        assert _is_numeric(True) is False

    def test_bool_false_not_numeric(self) -> None:
        """bool subclass of int must be excluded."""
        assert _is_numeric(False) is False

    def test_string_not_numeric(self) -> None:
        assert _is_numeric("42") is False

    def test_none_not_numeric(self) -> None:
        assert _is_numeric(None) is False

    def test_list_not_numeric(self) -> None:
        assert _is_numeric([1]) is False


class TestBoolValuesRejectedBySamplingBounds:
    """Bool values must NOT pass numeric bounds checks (M-2)."""

    def test_temperature_bool_ignored(self) -> None:
        """temperature=True (evaluates as 1) should be ignored, not validated."""
        v = SamplingBoundsValidator()
        result = v.validate(_params(options={"temperature": True}), _all_supported_caps())
        assert result == []

    def test_top_k_bool_ignored(self) -> None:
        """top_k=True (evaluates as 1) should be ignored, not validated."""
        v = SamplingBoundsValidator()
        result = v.validate(_params(options={"top_k": True}), _all_supported_caps())
        assert result == []

    def test_top_p_bool_ignored(self) -> None:
        """top_p=True (evaluates as 1) should be ignored, not validated."""
        v = SamplingBoundsValidator()
        result = v.validate(_params(options={"top_p": True}), _all_supported_caps())
        assert result == []


class TestBoolValuesIgnoredByKokoro:
    """KokoroValidator must not reject bool values as sampling params."""

    def test_kokoro_temperature_bool_ignored(self) -> None:
        """temperature=True should NOT trigger Kokoro rejection."""
        v = KokoroValidator()
        result = v.validate(_params(options={"temperature": True}), _kokoro_caps())
        assert result == []


# ===================================================================
# register_engine_validators tests (C-7)
# ===================================================================


class TestRegisterEngineValidators:
    """Test runtime registration of engine-specific validators."""

    def test_register_new_engine(self) -> None:
        """New engine validators are discoverable after registration."""
        from macaw.workers.tts._validation import _ENGINE_VALIDATORS

        test_engine = "test-custom-engine-register"
        try:
            register_engine_validators(test_engine, (SpeedValidator(),))
            validators = get_validators_for_engine(test_engine)
            assert len(validators) == 4  # 3 defaults + 1 custom
        finally:
            _ENGINE_VALIDATORS.pop(test_engine, None)

    def test_register_duplicate_raises(self) -> None:
        """Registering validators for an existing engine raises ValueError."""
        import pytest

        with pytest.raises(ValueError, match="already registered"):
            register_engine_validators("kokoro", (SpeedValidator(),))

    def test_register_empty_tuple(self) -> None:
        """Registering empty tuple is allowed (no-op validators)."""
        from macaw.workers.tts._validation import _ENGINE_VALIDATORS

        test_engine = "test-empty-validators"
        try:
            register_engine_validators(test_engine, ())
            validators = get_validators_for_engine(test_engine)
            assert len(validators) == 3  # 3 defaults + 0 custom
        finally:
            _ENGINE_VALIDATORS.pop(test_engine, None)


# ===================================================================
# validate_params_against_capabilities engine param tests (M-1)
# ===================================================================


class TestWrapperEngineParam:
    """Test the engine kwarg on validate_params_against_capabilities."""

    def test_engine_none_uses_defaults_only(self) -> None:
        """engine=None runs only default validators."""
        params = _params(options={"temperature": 0.8})
        result = validate_params_against_capabilities(params, _kokoro_caps())
        # Default validators don't reject temperature
        assert result == []

    def test_engine_kokoro_includes_kokoro_validator(self) -> None:
        """engine='kokoro' runs KokoroValidator which rejects temperature."""
        params = _params(options={"temperature": 0.8})
        result = validate_params_against_capabilities(params, _kokoro_caps(), engine="kokoro")
        assert any("temperature" in e and "Kokoro" in e for e in result)

    def test_engine_unknown_uses_defaults_only(self) -> None:
        """Unknown engine name uses only default validators."""
        params = _params(options={"temperature": 0.8})
        result = validate_params_against_capabilities(
            params, _kokoro_caps(), engine="nonexistent-engine"
        )
        assert result == []

    def test_engine_chatterbox_rejects_seed(self) -> None:
        """engine='chatterbox' runs ChatterboxValidator which rejects seed."""
        params = _params(options={"seed": 42})
        result = validate_params_against_capabilities(
            params, _chatterbox_caps(), engine="chatterbox"
        )
        assert any("seed" in e and "Chatterbox" in e for e in result)


# ===================================================================
# SamplingBoundsValidator speed-skip tests (S-5)
# ===================================================================


class TestSpeedBoundsSkipWhenUnsupported:
    """SamplingBoundsValidator skips speed bounds when supports_speed=False."""

    def test_speed_bounds_skipped_when_unsupported(self) -> None:
        """Out-of-range speed produces no bounds error when unsupported."""
        v = SamplingBoundsValidator()
        caps = TTSEngineCapabilities(supports_speed=False)
        result = v.validate(_params(speed=10.0), caps)
        assert result == []

    def test_speed_bounds_checked_when_supported(self) -> None:
        """Out-of-range speed produces bounds error when supported."""
        v = SamplingBoundsValidator()
        caps = TTSEngineCapabilities(supports_speed=True)
        result = v.validate(_params(speed=10.0), caps)
        assert len(result) == 1
        assert "speed" in result[0]

    def test_dual_error_when_supported_and_out_of_range(self) -> None:
        """With supports_speed=True, both SpeedValidator and SamplingBoundsValidator
        can fire independently (speed=10.0 is out of range but 'supported')."""
        caps = TTSEngineCapabilities(supports_speed=True)
        params = _params(speed=10.0)
        # SpeedValidator: no error (speed IS supported)
        # SamplingBoundsValidator: 1 error (out of range)
        result = validate_params(params, caps, (SpeedValidator(), SamplingBoundsValidator()))
        assert len(result) == 1
        assert "between" in result[0]

    def test_unsupported_speed_only_one_error(self) -> None:
        """With supports_speed=False + speed!=1.0, only SpeedValidator fires."""
        caps = TTSEngineCapabilities(supports_speed=False)
        params = _params(speed=10.0)
        result = validate_params(params, caps, (SpeedValidator(), SamplingBoundsValidator()))
        assert len(result) == 1
        assert "not supported" in result[0]
