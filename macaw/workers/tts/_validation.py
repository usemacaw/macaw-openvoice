"""Validation of TTS parameters against engine capabilities.

Uses a composable **ParamValidator Protocol** (Strategy + Registry) so that
new engines can register custom validators without modifying existing code
(OCP).  Three default validators ship out of the box:

- ``SpeedValidator`` — rejects non-default speed when the engine lacks
  ``supports_speed``.
- ``CapabilityValidator`` — rejects feature-gated params (ref_audio,
  instruction) when the engine lacks the corresponding capability.
- ``SamplingBoundsValidator`` — enforces universal safety bounds on
  ``top_k``, ``temperature``, ``top_p`` and ``speed`` regardless of
  engine capabilities (defense-in-depth at the gRPC boundary).

Engine-specific validators are registered in ``_ENGINE_VALIDATORS``:

- ``kokoro`` — rejects sampling params that have no effect on a
  deterministic forward-pass engine (temperature, top_k, top_p).
- ``qwen3-tts`` — rejects the conflicting combination of a custom
  ``instruction`` with ``text_normalization="off"`` (the instruction
  takes priority and the normalization-off request is silently lost).
- ``chatterbox`` — rejects seed (not implemented, no
  reproducibility guarantee).
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Protocol, TypeGuard, runtime_checkable

if TYPE_CHECKING:
    from macaw._types import TTSEngineCapabilities
    from macaw.workers.tts.converters import SynthesizeParams


def _is_numeric(value: object) -> TypeGuard[int | float]:
    """Return True if *value* is a real number (int or float), excluding bool.

    Python's ``bool`` is a subclass of ``int``, so ``isinstance(True, int)``
    returns ``True``.  Passing ``temperature=True`` would silently evaluate
    as ``1`` and pass bounds checks — semantically incorrect.

    Uses ``TypeGuard`` so mypy narrows the type to ``int | float`` after
    a truthy check.
    """
    return isinstance(value, int | float) and not isinstance(value, bool)


# ── Safety bounds (universal, engine-independent) ──────────────────────
_MAX_TOP_K = 1000
_MAX_TEMPERATURE = 2.0
_MAX_TOP_P = 1.0
_MIN_SPEED = 0.25
_MAX_SPEED = 4.0


# ── Protocol ───────────────────────────────────────────────────────────


@runtime_checkable
class ParamValidator(Protocol):
    """Strategy interface for TTS parameter validation.

    Each validator inspects ``params`` and ``capabilities`` and returns a
    list of human-readable error messages.  An empty list means the
    validator found no issues.
    """

    def validate(
        self,
        params: SynthesizeParams,
        capabilities: TTSEngineCapabilities,
    ) -> list[str]: ...


# ── Default validators (apply to all engines) ─────────────────────────


class SpeedValidator:
    """Reject non-default speed when the engine does not support it."""

    def validate(
        self,
        params: SynthesizeParams,
        capabilities: TTSEngineCapabilities,
    ) -> list[str]:
        # Exact comparison safe: speed comes from proto float serialization
        if params.speed != 1.0 and not capabilities.supports_speed:
            return ["speed is not supported by this engine"]
        return []


class CapabilityValidator:
    """Reject params that require capabilities the engine does not have.

    Unlike sampling params (which are silent no-ops on deterministic
    engines), voice cloning and instruct params cause a fundamental
    feature mismatch: the user expects a behavior that will NOT happen.
    """

    def validate(
        self,
        params: SynthesizeParams,
        capabilities: TTSEngineCapabilities,
    ) -> list[str]:
        errors: list[str] = []
        options = params.options or {}

        if options.get("ref_audio") and not capabilities.supports_voice_cloning:
            errors.append("ref_audio requires an engine with voice cloning support")
        if options.get("ref_text") and not capabilities.supports_voice_cloning:
            errors.append("ref_text requires an engine with voice cloning support")
        if options.get("instruction") and not capabilities.supports_instruct:
            errors.append("instruction requires an engine with instruct support")

        return errors


class SamplingBoundsValidator:
    """Enforce universal safety bounds on sampling parameters.

    These bounds apply regardless of engine capabilities and act as a
    second defense layer behind Pydantic (which validates at the REST/WS
    boundary).  Workers can be called directly via
    ``MACAW_REMOTE_WORKERS``, bypassing Pydantic entirely.
    """

    def validate(
        self,
        params: SynthesizeParams,
        capabilities: TTSEngineCapabilities,
    ) -> list[str]:
        errors: list[str] = []
        options = params.options or {}

        # top_k
        raw_top_k = options.get("top_k")
        if _is_numeric(raw_top_k):
            if raw_top_k < 0:
                errors.append("top_k must be >= 0")
            elif raw_top_k > _MAX_TOP_K:
                errors.append(f"top_k must be <= {_MAX_TOP_K}")

        # temperature
        raw_temp = options.get("temperature")
        if _is_numeric(raw_temp):
            if raw_temp < 0:
                errors.append("temperature must be >= 0")
            elif raw_temp > _MAX_TEMPERATURE:
                errors.append(f"temperature must be <= {_MAX_TEMPERATURE}")

        # top_p
        raw_top_p = options.get("top_p")
        if _is_numeric(raw_top_p):
            if raw_top_p < 0:
                errors.append("top_p must be >= 0")
            elif raw_top_p > _MAX_TOP_P:
                errors.append(f"top_p must be <= {_MAX_TOP_P}")

        # speed (value bounds — only when engine supports speed; otherwise
        # SpeedValidator already rejects the non-default value and a second
        # "out of range" message would be confusing)
        if capabilities.supports_speed and (
            params.speed < _MIN_SPEED or params.speed > _MAX_SPEED
        ):
            errors.append(f"speed must be between {_MIN_SPEED} and {_MAX_SPEED}")

        return errors


# ── Engine-specific validators ─────────────────────────────────────────


class KokoroValidator:
    """Reject sampling params that have no effect on Kokoro.

    Kokoro is a deterministic forward-pass engine with no sampling step.
    Sending temperature, top_k, or top_p has zero effect on the output.
    Rejecting these explicitly helps users understand their params are
    being ignored instead of silently producing identical results.
    """

    def validate(
        self,
        params: SynthesizeParams,
        capabilities: TTSEngineCapabilities,
    ) -> list[str]:
        errors: list[str] = []
        options = params.options or {}

        raw_temp = options.get("temperature")
        if _is_numeric(raw_temp) and raw_temp != 0:
            errors.append("temperature has no effect on Kokoro (deterministic engine)")

        raw_top_k = options.get("top_k")
        if _is_numeric(raw_top_k) and raw_top_k != 0:
            errors.append("top_k has no effect on Kokoro (deterministic engine)")

        raw_top_p = options.get("top_p")
        if _is_numeric(raw_top_p) and raw_top_p != 0:
            errors.append("top_p has no effect on Kokoro (deterministic engine)")

        return errors


class Qwen3TTSValidator:
    """Reject conflicting instruction + text_normalization on Qwen3-TTS.

    Qwen3 implements ``text_normalization="off"`` by injecting a special
    instruction prompt internally.  When the user also provides a custom
    ``instruction``, the custom instruction takes priority and the
    ``text_normalization="off"`` request is silently lost.

    Rejecting this combination upfront prevents a subtle bug where the
    user believes both settings are active but only the instruction is.
    """

    def validate(
        self,
        params: SynthesizeParams,
        capabilities: TTSEngineCapabilities,
    ) -> list[str]:
        options = params.options or {}

        instruction = options.get("instruction")
        text_norm = options.get("text_normalization")

        if instruction and text_norm == "off":
            return [
                "instruction and text_normalization='off' conflict on Qwen3-TTS: "
                "custom instruction takes priority, text_normalization='off' "
                "will be ignored"
            ]

        return []


class ChatterboxValidator:
    """Reject seed on Chatterbox (not implemented, no reproducibility).

    Chatterbox does not extract or use the seed parameter.  A user
    sending seed expects reproducible output, which Chatterbox cannot
    guarantee.  Fail-fast with a clear message.
    """

    def validate(
        self,
        params: SynthesizeParams,
        capabilities: TTSEngineCapabilities,
    ) -> list[str]:
        options = params.options or {}

        raw_seed = options.get("seed")
        if _is_numeric(raw_seed) and raw_seed != 0:
            return ["seed is not supported by Chatterbox (reproducibility not guaranteed)"]

        return []


# ── Registry ───────────────────────────────────────────────────────────

# Singletons: all validators MUST be stateless (shared across requests).
_DEFAULT_VALIDATORS: tuple[ParamValidator, ...] = (
    SpeedValidator(),
    CapabilityValidator(),
    SamplingBoundsValidator(),
)

_ENGINE_VALIDATORS: dict[str, tuple[ParamValidator, ...]] = {
    "kokoro": (KokoroValidator(),),
    "qwen3-tts": (Qwen3TTSValidator(),),
    "chatterbox": (ChatterboxValidator(),),
}


def register_engine_validators(engine: str, validators: tuple[ParamValidator, ...]) -> None:
    """Register engine-specific validators at runtime.

    External engines loaded via ``python_package`` can call this during
    their ``post_load_hook()`` to register custom validators without
    modifying the built-in registry (OCP).

    Raises ``ValueError`` if the engine already has registered validators.
    """
    if engine in _ENGINE_VALIDATORS:
        msg = f"Validators already registered for engine '{engine}'"
        raise ValueError(msg)
    _ENGINE_VALIDATORS[engine] = validators


def get_validators_for_engine(engine: str) -> tuple[ParamValidator, ...]:
    """Return the full validator chain for *engine*.

    Concatenates ``_DEFAULT_VALIDATORS`` with any engine-specific
    validators registered in ``_ENGINE_VALIDATORS``.
    """
    extra = _ENGINE_VALIDATORS.get(engine, ())
    return _DEFAULT_VALIDATORS + extra


# ── Runner ─────────────────────────────────────────────────────────────


def validate_params(
    params: SynthesizeParams,
    capabilities: TTSEngineCapabilities,
    validators: tuple[ParamValidator, ...],
) -> list[str]:
    """Run *validators* in order and accumulate error messages."""
    errors: list[str] = []
    for validator in validators:
        errors.extend(validator.validate(params, capabilities))
    return errors


# ── Backward-compatible wrapper ────────────────────────────────────────


def validate_params_against_capabilities(
    params: SynthesizeParams,
    capabilities: TTSEngineCapabilities,
    *,
    engine: str | None = None,
) -> list[str]:
    """Check TTS params, optionally including engine-specific validators.

    When *engine* is provided, engine-specific validators (e.g. KokoroValidator)
    are appended to the default chain — matching the servicer's behavior exactly.
    When *engine* is ``None``, only the three default validators run.
    """
    validators = get_validators_for_engine(engine) if engine else _DEFAULT_VALIDATORS
    return validate_params(params, capabilities, validators)
