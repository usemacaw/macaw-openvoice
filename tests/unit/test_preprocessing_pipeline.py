"""Tests for the Audio Preprocessing Pipeline.

Tests AudioStage ABC, audio decode/encode, and AudioPreprocessingPipeline.
"""

from __future__ import annotations

import io
import math
import struct
import wave

import numpy as np
import pytest

from macaw.config.preprocessing import PreprocessingConfig
from macaw.exceptions import AudioFormatError
from macaw.preprocessing.audio_io import decode_audio, encode_pcm16
from macaw.preprocessing.pipeline import AudioPreprocessingPipeline
from macaw.preprocessing.stages import AudioStage

# --- Helpers ---


def make_wav_bytes(
    sample_rate: int = 16000,
    duration: float = 0.1,
    frequency: float = 440.0,
    amplitude: float = 0.5,
) -> bytes:
    """Create WAV PCM 16-bit bytes with sine tone."""
    n_samples = int(sample_rate * duration)
    samples = []
    for i in range(n_samples):
        t = i / sample_rate
        value = int(32767 * amplitude * math.sin(2 * math.pi * frequency * t))
        samples.append(value)

    buffer = io.BytesIO()
    with wave.open(buffer, "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(sample_rate)
        wf.writeframes(struct.pack(f"<{len(samples)}h", *samples))
    return buffer.getvalue()


def make_stereo_wav_bytes(
    sample_rate: int = 16000,
    duration: float = 0.1,
) -> bytes:
    """Create stereo WAV PCM 16-bit bytes."""
    n_samples = int(sample_rate * duration)
    samples = []
    for i in range(n_samples):
        t = i / sample_rate
        left = int(32767 * 0.5 * math.sin(2 * math.pi * 440.0 * t))
        right = int(32767 * 0.3 * math.sin(2 * math.pi * 880.0 * t))
        samples.extend([left, right])

    buffer = io.BytesIO()
    with wave.open(buffer, "wb") as wf:
        wf.setnchannels(2)
        wf.setsampwidth(2)
        wf.setframerate(sample_rate)
        wf.writeframes(struct.pack(f"<{len(samples)}h", *samples))
    return buffer.getvalue()


class PassthroughStage(AudioStage):
    """Stage that returns audio without modification."""

    @property
    def name(self) -> str:
        return "passthrough"

    def process(self, audio: np.ndarray, sample_rate: int) -> tuple[np.ndarray, int]:
        return audio, sample_rate


class GainStage(AudioStage):
    """Stage that multiplies amplitude by a factor."""

    def __init__(self, factor: float = 0.5) -> None:
        self._factor = factor

    @property
    def name(self) -> str:
        return "gain"

    def process(self, audio: np.ndarray, sample_rate: int) -> tuple[np.ndarray, int]:
        return audio * self._factor, sample_rate


class SampleRateChangeStage(AudioStage):
    """Stage that simulates sample rate change (truncates samples for testing)."""

    def __init__(self, target_sr: int) -> None:
        self._target_sr = target_sr

    @property
    def name(self) -> str:
        return "sr_change"

    def process(self, audio: np.ndarray, sample_rate: int) -> tuple[np.ndarray, int]:
        # Simulates simple resample by repetition/decimation
        ratio = self._target_sr / sample_rate
        new_length = int(len(audio) * ratio)
        indices = np.linspace(0, len(audio) - 1, new_length).astype(int)
        return audio[indices], self._target_sr


# --- AudioStage ABC Tests ---


class TestAudioStage:
    def test_cannot_instantiate_abc(self) -> None:
        """AudioStage is abstract and cannot be instantiated directly."""
        with pytest.raises(TypeError, match="abstract"):
            AudioStage()  # type: ignore[abstract]

    def test_passthrough_stage_implements_interface(self) -> None:
        """Concrete stage implementing AudioStage works correctly."""
        stage = PassthroughStage()
        audio = np.array([0.1, 0.2, 0.3], dtype=np.float32)
        result, sr = stage.process(audio, 16000)
        np.testing.assert_array_equal(result, audio)
        assert sr == 16000

    def test_stage_name_property(self) -> None:
        """Stage exposes identifier name."""
        stage = PassthroughStage()
        assert stage.name == "passthrough"

    def test_gain_stage_modifies_audio(self) -> None:
        """Stage can modify the processed audio."""
        stage = GainStage(factor=0.5)
        audio = np.array([1.0, -1.0, 0.5], dtype=np.float32)
        result, sr = stage.process(audio, 16000)
        expected = np.array([0.5, -0.5, 0.25], dtype=np.float32)
        np.testing.assert_array_almost_equal(result, expected)
        assert sr == 16000


# --- decode_audio Tests ---


class TestDecodeAudio:
    def test_decode_wav_16khz(self) -> None:
        """Decodes WAV PCM 16-bit, 16kHz correctly."""
        wav_bytes = make_wav_bytes(sample_rate=16000, duration=0.1)
        audio, sr = decode_audio(wav_bytes)
        assert sr == 16000
        assert audio.dtype == np.float32
        assert len(audio) == 1600  # 0.1s * 16000

    def test_decode_wav_8khz(self) -> None:
        """Decodes WAV 8kHz correctly."""
        wav_bytes = make_wav_bytes(sample_rate=8000, duration=0.1)
        audio, sr = decode_audio(wav_bytes)
        assert sr == 8000
        assert len(audio) == 800

    def test_decode_wav_44khz(self) -> None:
        """Decodes WAV 44.1kHz correctly."""
        wav_bytes = make_wav_bytes(sample_rate=44100, duration=0.1)
        audio, sr = decode_audio(wav_bytes)
        assert sr == 44100
        assert len(audio) == 4410

    def test_decode_stereo_to_mono(self) -> None:
        """Converts stereo audio to mono automatically."""
        stereo_bytes = make_stereo_wav_bytes(sample_rate=16000, duration=0.1)
        audio, sr = decode_audio(stereo_bytes)
        assert sr == 16000
        assert audio.ndim == 1  # Mono

    def test_decode_empty_bytes_raises(self) -> None:
        """Empty bytes raise AudioFormatError."""
        with pytest.raises(AudioFormatError, match="Empty audio"):
            decode_audio(b"")

    def test_decode_invalid_bytes_raises(self) -> None:
        """Invalid bytes raise AudioFormatError."""
        with pytest.raises(AudioFormatError):
            decode_audio(b"not audio data at all")

    def test_decode_returns_float32(self) -> None:
        """Decoded audio is always float32."""
        wav_bytes = make_wav_bytes()
        audio, _ = decode_audio(wav_bytes)
        assert audio.dtype == np.float32

    def test_decode_values_in_range(self) -> None:
        """Decoded values are in the range [-1.0, 1.0]."""
        wav_bytes = make_wav_bytes(amplitude=1.0)
        audio, _ = decode_audio(wav_bytes)
        assert np.all(audio >= -1.0)
        assert np.all(audio <= 1.0)


# --- encode_pcm16 Tests ---


class TestEncodePcm16:
    def test_encode_produces_valid_wav(self) -> None:
        """Encode produces valid WAV bytes that can be re-decoded."""
        audio = np.array([0.0, 0.5, -0.5, 1.0, -1.0], dtype=np.float32)
        wav_bytes = encode_pcm16(audio, 16000)

        # Should be re-decodable
        decoded, sr = decode_audio(wav_bytes)
        assert sr == 16000
        assert len(decoded) == 5

    def test_encode_clamps_values(self) -> None:
        """Encode clamps values outside [-1.0, 1.0] without overflow."""
        audio = np.array([2.0, -2.0, 0.0], dtype=np.float32)
        wav_bytes = encode_pcm16(audio, 16000)

        # Verify via wave stdlib that the WAV is valid
        with wave.open(io.BytesIO(wav_bytes), "rb") as wf:
            assert wf.getnchannels() == 1
            assert wf.getsampwidth() == 2
            assert wf.getframerate() == 16000
            assert wf.getnframes() == 3

    def test_encode_preserves_sample_rate(self) -> None:
        """Sample rate is preserved in the WAV header."""
        audio = np.zeros(100, dtype=np.float32)
        for sr in [8000, 16000, 44100, 48000]:
            wav_bytes = encode_pcm16(audio, sr)
            with wave.open(io.BytesIO(wav_bytes), "rb") as wf:
                assert wf.getframerate() == sr

    def test_roundtrip_decode_encode(self) -> None:
        """Roundtrip decode -> encode preserves audio (with PCM16 quantization)."""
        original_wav = make_wav_bytes(sample_rate=16000, duration=0.05)
        audio, sr = decode_audio(original_wav)
        re_encoded = encode_pcm16(audio, sr)
        re_decoded, sr2 = decode_audio(re_encoded)

        assert sr == sr2
        assert len(audio) == len(re_decoded)
        # PCM16 quantization tolerance (1/32768 ~= 3e-5)
        np.testing.assert_allclose(audio, re_decoded, atol=1e-4)


# --- AudioPreprocessingPipeline Tests ---


class TestAudioPreprocessingPipeline:
    def test_pipeline_zero_stages_returns_pcm16(self) -> None:
        """Pipeline with no stages decodes and re-encodes audio as PCM16 WAV."""
        config = PreprocessingConfig()
        pipeline = AudioPreprocessingPipeline(config)
        input_wav = make_wav_bytes(sample_rate=16000, duration=0.05)

        result = pipeline.process(input_wav)

        # Result should be valid WAV
        audio, sr = decode_audio(result)
        assert sr == 16000
        assert audio.dtype == np.float32

    def test_pipeline_with_passthrough_stage(self) -> None:
        """Pipeline with passthrough stage returns equivalent audio."""
        config = PreprocessingConfig()
        pipeline = AudioPreprocessingPipeline(config, stages=[PassthroughStage()])
        input_wav = make_wav_bytes(sample_rate=16000, duration=0.05)

        result = pipeline.process(input_wav)
        result_audio, result_sr = decode_audio(result)
        input_audio, input_sr = decode_audio(input_wav)

        assert result_sr == input_sr
        assert len(result_audio) == len(input_audio)
        np.testing.assert_allclose(result_audio, input_audio, atol=1e-4)

    def test_pipeline_chains_multiple_stages(self) -> None:
        """Pipeline executes stages in sequence."""
        config = PreprocessingConfig()
        # Two gain stages of 0.5 -> final result = 0.25 * original
        pipeline = AudioPreprocessingPipeline(
            config,
            stages=[GainStage(factor=0.5), GainStage(factor=0.5)],
        )
        input_wav = make_wav_bytes(sample_rate=16000, duration=0.05, amplitude=0.8)

        result = pipeline.process(input_wav)
        result_audio, _ = decode_audio(result)
        input_audio, _ = decode_audio(input_wav)

        # Resulting audio should be ~0.25 of original
        expected = input_audio * 0.25
        np.testing.assert_allclose(result_audio, expected, atol=1e-3)

    def test_pipeline_preserves_sample_rate_through_stages(self) -> None:
        """Sample rate is preserved when stages do not alter it."""
        config = PreprocessingConfig()
        pipeline = AudioPreprocessingPipeline(
            config,
            stages=[PassthroughStage(), GainStage(factor=0.8)],
        )
        input_wav = make_wav_bytes(sample_rate=44100, duration=0.05)

        result = pipeline.process(input_wav)
        _, result_sr = decode_audio(result)
        assert result_sr == 44100

    def test_pipeline_stage_can_change_sample_rate(self) -> None:
        """Stage can change sample rate and the pipeline propagates it."""
        config = PreprocessingConfig()
        pipeline = AudioPreprocessingPipeline(
            config,
            stages=[SampleRateChangeStage(target_sr=8000)],
        )
        input_wav = make_wav_bytes(sample_rate=16000, duration=0.1)

        result = pipeline.process(input_wav)
        _, result_sr = decode_audio(result)
        assert result_sr == 8000

    def test_pipeline_invalid_audio_raises(self) -> None:
        """Pipeline raises AudioFormatError for invalid audio."""
        config = PreprocessingConfig()
        pipeline = AudioPreprocessingPipeline(config)

        with pytest.raises(AudioFormatError):
            pipeline.process(b"not valid audio")

    def test_pipeline_empty_audio_raises(self) -> None:
        """Pipeline raises AudioFormatError for empty bytes."""
        config = PreprocessingConfig()
        pipeline = AudioPreprocessingPipeline(config)

        with pytest.raises(AudioFormatError, match="Empty audio"):
            pipeline.process(b"")

    def test_pipeline_config_accessible(self) -> None:
        """Config is accessible via property."""
        config = PreprocessingConfig(target_sample_rate=8000)
        pipeline = AudioPreprocessingPipeline(config)
        assert pipeline.config.target_sample_rate == 8000

    def test_pipeline_stages_property_returns_copy(self) -> None:
        """stages property returns a copy of the internal list."""
        config = PreprocessingConfig()
        stages = [PassthroughStage()]
        pipeline = AudioPreprocessingPipeline(config, stages=stages)

        returned = pipeline.stages
        returned.append(GainStage())  # Modify copy
        assert len(pipeline.stages) == 1  # Original does not change

    def test_pipeline_with_fixture_audio(self, audio_16khz: None) -> None:
        """Pipeline processes project audio fixtures."""
        from pathlib import Path

        fixture_path = Path(__file__).parent.parent / "fixtures" / "audio" / "sample_16khz.wav"
        if not fixture_path.exists():
            pytest.skip("Audio fixture not found")

        audio_bytes = fixture_path.read_bytes()
        config = PreprocessingConfig()
        pipeline = AudioPreprocessingPipeline(config)

        result = pipeline.process(audio_bytes)
        result_audio, result_sr = decode_audio(result)
        assert result_sr == 16000
        assert len(result_audio) > 0


# --- Tests for create_stages() factory (C-01 fix) ---


class TestCreateStagesFactory:
    """Tests that create_stages() returns independent stage instances."""

    def test_create_stages_returns_new_instances(self) -> None:
        """create_stages() returns deep-copied stages, not the originals."""
        from macaw.preprocessing.dc_remove import DCRemoveStage

        config = PreprocessingConfig()
        dc_stage = DCRemoveStage(cutoff_hz=20)
        pipeline = AudioPreprocessingPipeline(config, stages=[dc_stage])

        fresh = pipeline.create_stages()

        assert len(fresh) == 1
        assert fresh[0] is not dc_stage
        assert fresh[0].name == "dc_remove"

    def test_create_stages_independent_filter_state(self) -> None:
        """Two calls to create_stages() produce stages with independent state."""
        from macaw.preprocessing.dc_remove import DCRemoveStage

        config = PreprocessingConfig()
        dc_stage = DCRemoveStage(cutoff_hz=20)
        pipeline = AudioPreprocessingPipeline(config, stages=[dc_stage])

        stages_a = pipeline.create_stages()
        stages_b = pipeline.create_stages()

        # Process different audio through each set
        t = np.arange(int(16000 * 0.5)) / 16000
        audio_a = (0.5 * np.sin(2 * np.pi * 440 * t) + 0.2).astype(np.float32)
        audio_b = (0.3 * np.sin(2 * np.pi * 880 * t) - 0.1).astype(np.float32)

        stages_a[0].process(audio_a, 16000)
        stages_b[0].process(audio_b, 16000)

        # Filter state (zi) should be different since inputs differ
        dc_a = stages_a[0]
        dc_b = stages_b[0]
        assert dc_a._zi is not None  # type: ignore[union-attr]
        assert dc_b._zi is not None  # type: ignore[union-attr]
        assert not np.array_equal(dc_a._zi, dc_b._zi)  # type: ignore[union-attr]

    def test_concurrent_sessions_get_independent_dc_state(self) -> None:
        """Simulates two concurrent streaming sessions with independent stages.

        Each session processes different audio. The DC filter state of one
        session must NOT contaminate the other.
        """
        from macaw.preprocessing.dc_remove import DCRemoveStage
        from macaw.preprocessing.streaming import StreamingPreprocessor

        config = PreprocessingConfig()
        dc_stage = DCRemoveStage(cutoff_hz=20)
        pipeline = AudioPreprocessingPipeline(config, stages=[dc_stage])

        # Each session gets its own stages via create_stages()
        preprocessor_1 = StreamingPreprocessor(
            stages=pipeline.create_stages(), input_sample_rate=16000
        )
        preprocessor_2 = StreamingPreprocessor(
            stages=pipeline.create_stages(), input_sample_rate=16000
        )

        # Session 1: loud speech with DC offset
        n = int(16000 * 0.1)  # 100ms
        t = np.arange(n) / 16000
        audio_1 = (0.5 * np.sin(2 * np.pi * 440 * t) + 0.3).astype(np.float32)
        pcm_1 = (audio_1 * 32767).astype(np.int16).tobytes()

        # Session 2: quiet signal, no offset
        audio_2 = (0.05 * np.sin(2 * np.pi * 200 * t)).astype(np.float32)
        pcm_2 = (audio_2 * 32767).astype(np.int16).tobytes()

        # Interleave frames (simulating concurrent processing)
        result_1 = preprocessor_1.process_frame(pcm_1)
        result_2 = preprocessor_2.process_frame(pcm_2)

        # Results should be independent — session 2 should not inherit
        # the DC offset correction state from session 1
        assert len(result_1) > 0
        assert len(result_2) > 0
        # Session 2 had near-zero mean; result should also be near-zero
        assert abs(np.mean(result_2)) < 0.05

    def test_create_stages_resets_filter_state(self) -> None:
        """create_stages() calls reset() on each returned stage."""
        from macaw.preprocessing.dc_remove import DCRemoveStage

        config = PreprocessingConfig()
        dc_stage = DCRemoveStage(cutoff_hz=20)

        # Process some audio to populate filter state
        t = np.arange(int(16000 * 0.1)) / 16000
        audio = (0.5 * np.sin(2 * np.pi * 440 * t)).astype(np.float32)
        dc_stage.process(audio, 16000)
        assert dc_stage._zi is not None

        pipeline = AudioPreprocessingPipeline(config, stages=[dc_stage])
        fresh = pipeline.create_stages()

        # Fresh stage should have zi = None (reset was called)
        assert fresh[0]._zi is None  # type: ignore[union-attr]

    def test_create_stages_with_stateless_stage(self) -> None:
        """create_stages() works with stateless stages (reset is no-op)."""
        config = PreprocessingConfig()
        pipeline = AudioPreprocessingPipeline(config, stages=[PassthroughStage()])

        fresh = pipeline.create_stages()

        assert len(fresh) == 1
        assert fresh[0].name == "passthrough"
