"""Tests for audio effects: PitchShift, Reverb, AudioEffectChain, and factory."""

from __future__ import annotations

import numpy as np
import pytest

from macaw._audio_constants import PCM_INT16_SCALE
from macaw.audio_effects import create_effect_chain
from macaw.audio_effects.chain import AudioEffectChain, pcm16_bytes_to_float32
from macaw.audio_effects.interface import AudioEffect
from macaw.audio_effects.pitch_shift import PitchShiftEffect
from macaw.audio_effects.reverb import ReverbEffect
from macaw.workers.tts.audio_utils import float32_to_pcm16_bytes

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _sine_wave(freq_hz: float, duration_s: float, sample_rate: int) -> np.ndarray:
    """Generate a mono sine wave as float32 array."""
    t = np.arange(int(sample_rate * duration_s), dtype=np.float32) / sample_rate
    return np.sin(2 * np.pi * freq_hz * t).astype(np.float32)


def _dominant_frequency(audio: np.ndarray, sample_rate: int) -> float:
    """Return the dominant frequency in Hz using FFT."""
    fft = np.abs(np.fft.rfft(audio))
    freqs = np.fft.rfftfreq(len(audio), d=1.0 / sample_rate)
    return float(freqs[np.argmax(fft)])


# ---------------------------------------------------------------------------
# AudioEffect ABC
# ---------------------------------------------------------------------------


class TestAudioEffectABC:
    def test_cannot_instantiate_directly(self) -> None:
        with pytest.raises(TypeError):
            AudioEffect()  # type: ignore[abstract]

    def test_reset_default_is_noop(self) -> None:
        """Concrete subclass with reset() not overridden — should not raise."""

        class DummyEffect(AudioEffect):
            @property
            def name(self) -> str:
                return "dummy"

            def process(self, audio: np.ndarray, sample_rate: int) -> np.ndarray:
                return audio

        effect = DummyEffect()
        effect.reset()  # no-op, should not raise


# ---------------------------------------------------------------------------
# PitchShiftEffect
# ---------------------------------------------------------------------------


class TestPitchShiftEffect:
    def test_name(self) -> None:
        effect = PitchShiftEffect(semitones=3.0)
        assert effect.name == "pitch_shift"

    def test_pitch_shift_zero_is_identity(self) -> None:
        audio = _sine_wave(440, 0.5, 24000)
        effect = PitchShiftEffect(semitones=0.0)
        result = effect.process(audio, 24000)
        np.testing.assert_array_equal(result, audio)

    def test_pitch_shift_preserves_duration(self) -> None:
        audio = _sine_wave(440, 0.5, 24000)
        for semitones in (-6, -3, 3, 6, 12):
            effect = PitchShiftEffect(semitones=float(semitones))
            result = effect.process(audio, 24000)
            assert len(result) == len(audio), f"Duration changed at {semitones} semitones"

    def test_pitch_shift_up_increases_frequency(self) -> None:
        sr = 24000
        audio = _sine_wave(440, 0.5, sr)
        effect = PitchShiftEffect(semitones=12.0)
        result = effect.process(audio, sr)

        original_freq = _dominant_frequency(audio, sr)
        shifted_freq = _dominant_frequency(result, sr)

        # +12 semitones = 1 octave up = 2x frequency
        assert shifted_freq > original_freq * 1.5, (
            f"Expected ~{original_freq * 2:.0f}Hz, got {shifted_freq:.0f}Hz"
        )

    def test_pitch_shift_down_decreases_frequency(self) -> None:
        sr = 24000
        audio = _sine_wave(880, 0.5, sr)
        effect = PitchShiftEffect(semitones=-12.0)
        result = effect.process(audio, sr)

        original_freq = _dominant_frequency(audio, sr)
        shifted_freq = _dominant_frequency(result, sr)

        # -12 semitones = 1 octave down = 0.5x frequency
        assert shifted_freq < original_freq * 0.75, (
            f"Expected ~{original_freq * 0.5:.0f}Hz, got {shifted_freq:.0f}Hz"
        )

    def test_pitch_shift_empty_audio(self) -> None:
        audio = np.array([], dtype=np.float32)
        effect = PitchShiftEffect(semitones=5.0)
        result = effect.process(audio, 24000)
        assert len(result) == 0

    def test_pitch_shift_output_dtype(self) -> None:
        audio = _sine_wave(440, 0.1, 24000)
        effect = PitchShiftEffect(semitones=3.0)
        result = effect.process(audio, 24000)
        assert result.dtype == np.float32


# ---------------------------------------------------------------------------
# ReverbEffect
# ---------------------------------------------------------------------------


class TestReverbEffect:
    def test_name(self) -> None:
        effect = ReverbEffect()
        assert effect.name == "reverb"

    def test_reverb_preserves_length(self) -> None:
        """Reverb output should be same length as input (no tail extension)."""
        audio = _sine_wave(440, 0.5, 24000)
        effect = ReverbEffect(room_size=0.5, damping=0.5, wet_dry_mix=0.3)
        result = effect.process(audio, 24000)
        assert len(result) == len(audio)

    def test_reverb_wet_zero_is_dry(self) -> None:
        """wet_dry_mix=0 should return input unchanged (only dry signal)."""
        audio = _sine_wave(440, 0.5, 24000)
        effect = ReverbEffect(wet_dry_mix=0.0)
        result = effect.process(audio, 24000)
        np.testing.assert_allclose(result, audio, atol=1e-6)

    def test_reverb_modifies_signal_when_wet(self) -> None:
        """With non-zero wet mix, output should differ from input."""
        audio = _sine_wave(440, 0.5, 24000)
        effect = ReverbEffect(room_size=0.7, wet_dry_mix=0.5)
        result = effect.process(audio, 24000)
        assert not np.allclose(result, audio, atol=1e-3)

    def test_reverb_preserves_amplitude_range(self) -> None:
        """Output should be clipped to [-1, 1]."""
        audio = _sine_wave(440, 0.5, 24000) * 0.9
        effect = ReverbEffect(room_size=0.9, wet_dry_mix=0.5)
        result = effect.process(audio, 24000)
        assert result.max() <= 1.0
        assert result.min() >= -1.0

    def test_reverb_processes_silence(self) -> None:
        """Silence in should produce silence out (no noise injection)."""
        audio = np.zeros(12000, dtype=np.float32)
        effect = ReverbEffect(room_size=0.5, wet_dry_mix=0.5)
        result = effect.process(audio, 24000)
        np.testing.assert_allclose(result, 0.0, atol=1e-10)

    def test_reverb_reset_clears_state(self) -> None:
        """After reset, processing identical audio should give identical results."""
        audio = _sine_wave(440, 0.2, 24000)
        effect = ReverbEffect(room_size=0.5, wet_dry_mix=0.3)

        result1 = effect.process(audio, 24000)
        effect.reset()
        result2 = effect.process(audio, 24000)

        np.testing.assert_array_equal(result1, result2)

    def test_reverb_empty_audio(self) -> None:
        audio = np.array([], dtype=np.float32)
        effect = ReverbEffect()
        result = effect.process(audio, 24000)
        assert len(result) == 0

    def test_reverb_output_dtype(self) -> None:
        audio = _sine_wave(440, 0.1, 24000)
        effect = ReverbEffect(room_size=0.5, wet_dry_mix=0.3)
        result = effect.process(audio, 24000)
        assert result.dtype == np.float32

    def test_reverb_room_size_affects_output(self) -> None:
        """Different room sizes should produce different outputs."""
        audio = _sine_wave(440, 0.3, 24000)

        small = ReverbEffect(room_size=0.1, wet_dry_mix=0.5)
        large = ReverbEffect(room_size=0.9, wet_dry_mix=0.5)

        result_small = small.process(audio, 24000)
        result_large = large.process(audio, 24000)

        assert not np.allclose(result_small, result_large, atol=1e-3)


# ---------------------------------------------------------------------------
# AudioEffectChain
# ---------------------------------------------------------------------------


class TestAudioEffectChain:
    def test_empty_chain_is_passthrough(self) -> None:
        audio = _sine_wave(440, 0.1, 24000)
        chain = AudioEffectChain(effects=[])
        result = chain.process(audio, 24000)
        np.testing.assert_array_equal(result, audio)

    def test_chain_composes_effects(self) -> None:
        """Chain of two effects should be applied sequentially."""
        audio = _sine_wave(440, 0.3, 24000)

        # Pitch shift followed by a second pitch shift
        chain = AudioEffectChain(
            effects=[
                PitchShiftEffect(semitones=6.0),
                PitchShiftEffect(semitones=-6.0),
            ]
        )
        result = chain.process(audio, 24000)

        # Roundtrip should produce roughly similar frequency
        original_freq = _dominant_frequency(audio, 24000)
        result_freq = _dominant_frequency(result, 24000)
        assert abs(result_freq - original_freq) < 50  # within 50Hz tolerance

    def test_chain_reset_resets_all(self) -> None:
        """reset() should propagate to all effects."""
        reverb = ReverbEffect(room_size=0.5, wet_dry_mix=0.3)
        chain = AudioEffectChain(effects=[reverb])

        audio = _sine_wave(440, 0.2, 24000)
        result1 = chain.process(audio, 24000)
        chain.reset()
        result2 = chain.process(audio, 24000)

        np.testing.assert_array_equal(result1, result2)


# ---------------------------------------------------------------------------
# PCM16 bytes <-> float32 conversion
# ---------------------------------------------------------------------------


class TestPCM16Conversion:
    def test_pcm16_bytes_to_float32_roundtrip(self) -> None:
        """PCM16 -> float32 -> PCM16 should be lossless."""
        audio = _sine_wave(440, 0.1, 24000)
        pcm_bytes = float32_to_pcm16_bytes(audio)
        recovered = pcm16_bytes_to_float32(pcm_bytes)
        final_bytes = float32_to_pcm16_bytes(recovered)
        assert pcm_bytes == final_bytes

    def test_pcm16_bytes_to_float32_range(self) -> None:
        """Output should be in [-1.0, ~1.0] range."""
        # Create PCM with max/min values
        int16_data = np.array([32767, -32768, 0], dtype=np.int16)
        pcm_bytes = int16_data.tobytes()
        result = pcm16_bytes_to_float32(pcm_bytes)
        assert result[0] == pytest.approx(32767 / PCM_INT16_SCALE)
        assert result[1] == pytest.approx(-32768 / PCM_INT16_SCALE)
        assert result[2] == 0.0

    def test_process_bytes_roundtrip(self) -> None:
        """process_bytes with empty chain should preserve audio fidelity."""
        audio = _sine_wave(440, 0.1, 24000)
        pcm_bytes = float32_to_pcm16_bytes(audio)
        chain = AudioEffectChain(effects=[])
        result_bytes = chain.process_bytes(pcm_bytes, 24000)
        assert result_bytes == pcm_bytes

    def test_process_bytes_with_effect(self) -> None:
        """process_bytes should apply effects and return modified PCM16."""
        audio = _sine_wave(440, 0.1, 24000)
        pcm_bytes = float32_to_pcm16_bytes(audio)
        chain = AudioEffectChain(effects=[PitchShiftEffect(semitones=6.0)])
        result_bytes = chain.process_bytes(pcm_bytes, 24000)
        assert result_bytes != pcm_bytes
        assert len(result_bytes) == len(pcm_bytes)  # same duration


# ---------------------------------------------------------------------------
# Factory
# ---------------------------------------------------------------------------


class TestCreateEffectChain:
    def test_returns_none_for_defaults(self) -> None:
        result = create_effect_chain()
        assert result is None

    def test_returns_none_for_zero_pitch_no_reverb(self) -> None:
        result = create_effect_chain(pitch_shift_semitones=0.0)
        assert result is None

    def test_returns_chain_with_pitch_shift(self) -> None:
        result = create_effect_chain(pitch_shift_semitones=3.0)
        assert result is not None
        assert isinstance(result, AudioEffectChain)

    def test_returns_chain_with_reverb(self) -> None:
        result = create_effect_chain(reverb_room_size=0.5)
        assert result is not None

    def test_returns_chain_with_reverb_any_param(self) -> None:
        """Setting any single reverb param should trigger reverb."""
        for kwargs in [
            {"reverb_damping": 0.3},
            {"reverb_wet_dry_mix": 0.4},
        ]:
            result = create_effect_chain(**kwargs)
            assert result is not None, f"Expected chain for {kwargs}"

    def test_returns_chain_with_both_effects(self) -> None:
        result = create_effect_chain(
            pitch_shift_semitones=3.0,
            reverb_room_size=0.5,
        )
        assert result is not None

    def test_pitch_shift_before_reverb_in_chain(self) -> None:
        """Effects should be ordered: pitch_shift first, reverb second."""
        result = create_effect_chain(
            pitch_shift_semitones=3.0,
            reverb_room_size=0.5,
        )
        assert result is not None
        effects = result._effects
        assert len(effects) == 2
        assert isinstance(effects[0], PitchShiftEffect)
        assert isinstance(effects[1], ReverbEffect)


# ---------------------------------------------------------------------------
# AudioEffectsParams Pydantic model
# ---------------------------------------------------------------------------


class TestAudioEffectsParams:
    def test_default_values(self) -> None:
        from macaw.server.models.effects import AudioEffectsParams

        params = AudioEffectsParams()
        assert params.pitch_shift_semitones == 0.0
        assert params.reverb_room_size is None
        assert params.reverb_damping is None
        assert params.reverb_wet_dry_mix is None

    def test_validation_pitch_shift_range(self) -> None:
        from macaw.server.models.effects import AudioEffectsParams

        # Valid
        AudioEffectsParams(pitch_shift_semitones=-12.0)
        AudioEffectsParams(pitch_shift_semitones=12.0)

        # Invalid
        with pytest.raises(Exception):  # noqa: B017
            AudioEffectsParams(pitch_shift_semitones=-13.0)
        with pytest.raises(Exception):  # noqa: B017
            AudioEffectsParams(pitch_shift_semitones=13.0)

    def test_validation_reverb_range(self) -> None:
        from macaw.server.models.effects import AudioEffectsParams

        # Valid
        AudioEffectsParams(reverb_room_size=0.0, reverb_damping=1.0, reverb_wet_dry_mix=0.5)

        # Invalid
        with pytest.raises(Exception):  # noqa: B017
            AudioEffectsParams(reverb_room_size=-0.1)
        with pytest.raises(Exception):  # noqa: B017
            AudioEffectsParams(reverb_room_size=1.1)

    def test_speech_request_accepts_effects(self) -> None:
        from macaw.server.models.speech import SpeechRequest

        req = SpeechRequest(
            model="kokoro",
            input="Hello world",
            effects={"pitch_shift_semitones": 3.0},
        )
        assert req.effects is not None
        assert req.effects.pitch_shift_semitones == 3.0

    def test_speech_request_effects_optional(self) -> None:
        from macaw.server.models.speech import SpeechRequest

        req = SpeechRequest(model="kokoro", input="Hello")
        assert req.effects is None

    def test_tts_speak_command_accepts_effects(self) -> None:
        from macaw.server.models.events import TTSSpeakCommand

        cmd = TTSSpeakCommand(
            text="Hello",
            effects={"reverb_room_size": 0.5, "reverb_wet_dry_mix": 0.3},
        )
        assert cmd.effects is not None
        assert cmd.effects.reverb_room_size == 0.5

    def test_tts_speak_command_effects_optional(self) -> None:
        from macaw.server.models.events import TTSSpeakCommand

        cmd = TTSSpeakCommand(text="Hello")
        assert cmd.effects is None
