"""Validation of TTS parameters against engine capabilities.

Pure function without side effects. Returns a list of unsupported parameter
names so the servicer can reject requests with INVALID_ARGUMENT.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from macaw._types import TTSEngineCapabilities
    from macaw.workers.tts.converters import SynthesizeParams


def validate_params_against_capabilities(
    params: SynthesizeParams,
    capabilities: TTSEngineCapabilities,
) -> list[str]:
    """Check if the requested TTS params are supported by the engine.

    Only flags params that the client explicitly set (non-default values).
    Returns list of unsupported parameter names. Empty = all valid.
    """
    unsupported: list[str] = []
    options = params.options or {}

    # Speed: check only if non-default (not 1.0)
    if params.speed != 1.0 and not capabilities.supports_speed:
        unsupported.append("speed")

    # Options-based params: check only if present in options dict.
    #
    # ``seed`` and ``text_normalization`` are intentionally NOT validated:
    #
    # - ``seed``: A deterministic engine inherently satisfies the
    #   "reproducible output" intent, so sending ``seed`` is never an
    #   error — the backend logs and ignores it (see kokoro.py
    #   ``seed_ignored_deterministic_engine``).
    #
    # - ``text_normalization``: "on" is default behavior for all engines;
    #   "off" is best-effort with clear logging (see kokoro.py
    #   ``text_normalization_off_best_effort``).  Rejecting it blocks
    #   graceful degradation already implemented in the backend.
    if "temperature" in options and not capabilities.supports_temperature:
        unsupported.append("temperature")
    if "top_k" in options and not capabilities.supports_top_k:
        unsupported.append("top_k")
    if "top_p" in options and not capabilities.supports_top_p:
        unsupported.append("top_p")

    return unsupported
