"""Audio effects module — post-synthesis audio transformations.

Provides AudioEffect ABC, AudioEffectChain compositor, and concrete
effects (PitchShiftEffect, ReverbEffect). Effects are applied after
TTS synthesis and before codec encoding / transport.
"""

from __future__ import annotations

from macaw.audio_effects.chain import AudioEffectChain
from macaw.audio_effects.interface import AudioEffect

__all__ = [
    "AudioEffect",
    "AudioEffectChain",
    "create_effect_chain",
]


def create_effect_chain(
    *,
    pitch_shift_semitones: float = 0.0,
    reverb_room_size: float | None = None,
    reverb_damping: float | None = None,
    reverb_wet_dry_mix: float | None = None,
    sample_rate: int = 24000,
) -> AudioEffectChain | None:
    """Create an effect chain from parameters.

    Returns None if no effects are requested (all defaults).
    Order: pitch_shift first, then reverb (shift frequency before adding room acoustics).
    """
    effects: list[AudioEffect] = []

    if pitch_shift_semitones != 0.0:
        from macaw.audio_effects.pitch_shift import PitchShiftEffect

        effects.append(PitchShiftEffect(pitch_shift_semitones))

    has_reverb = any(v is not None for v in (reverb_room_size, reverb_damping, reverb_wet_dry_mix))
    if has_reverb:
        from macaw.audio_effects.reverb import ReverbEffect

        effects.append(
            ReverbEffect(
                room_size=reverb_room_size if reverb_room_size is not None else 0.5,
                damping=reverb_damping if reverb_damping is not None else 0.5,
                wet_dry_mix=reverb_wet_dry_mix if reverb_wet_dry_mix is not None else 0.3,
                sample_rate=sample_rate,
            )
        )

    if not effects:
        return None

    return AudioEffectChain(effects)
