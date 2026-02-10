"""Testes do Audio Preprocessing Pipeline.

Testa AudioStage ABC, decode/encode de audio e AudioPreprocessingPipeline.
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
    """Cria bytes WAV PCM 16-bit com tom senoidal."""
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
    """Cria bytes WAV PCM 16-bit stereo."""
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
    """Stage que retorna audio sem modificacao."""

    @property
    def name(self) -> str:
        return "passthrough"

    def process(self, audio: np.ndarray, sample_rate: int) -> tuple[np.ndarray, int]:
        return audio, sample_rate


class GainStage(AudioStage):
    """Stage que multiplica amplitude por um fator."""

    def __init__(self, factor: float = 0.5) -> None:
        self._factor = factor

    @property
    def name(self) -> str:
        return "gain"

    def process(self, audio: np.ndarray, sample_rate: int) -> tuple[np.ndarray, int]:
        return audio * self._factor, sample_rate


class SampleRateChangeStage(AudioStage):
    """Stage que simula mudanca de sample rate (trunca amostras para teste)."""

    def __init__(self, target_sr: int) -> None:
        self._target_sr = target_sr

    @property
    def name(self) -> str:
        return "sr_change"

    def process(self, audio: np.ndarray, sample_rate: int) -> tuple[np.ndarray, int]:
        # Simula resample simples por repeticao/decimacao
        ratio = self._target_sr / sample_rate
        new_length = int(len(audio) * ratio)
        indices = np.linspace(0, len(audio) - 1, new_length).astype(int)
        return audio[indices], self._target_sr


# --- Testes AudioStage ABC ---


class TestAudioStage:
    def test_cannot_instantiate_abc(self) -> None:
        """AudioStage e abstrato e nao pode ser instanciado diretamente."""
        with pytest.raises(TypeError, match="abstract"):
            AudioStage()  # type: ignore[abstract]

    def test_passthrough_stage_implements_interface(self) -> None:
        """Stage concreto implementando AudioStage funciona corretamente."""
        stage = PassthroughStage()
        audio = np.array([0.1, 0.2, 0.3], dtype=np.float32)
        result, sr = stage.process(audio, 16000)
        np.testing.assert_array_equal(result, audio)
        assert sr == 16000

    def test_stage_name_property(self) -> None:
        """Stage expoe nome identificador."""
        stage = PassthroughStage()
        assert stage.name == "passthrough"

    def test_gain_stage_modifies_audio(self) -> None:
        """Stage pode modificar o audio processado."""
        stage = GainStage(factor=0.5)
        audio = np.array([1.0, -1.0, 0.5], dtype=np.float32)
        result, sr = stage.process(audio, 16000)
        expected = np.array([0.5, -0.5, 0.25], dtype=np.float32)
        np.testing.assert_array_almost_equal(result, expected)
        assert sr == 16000


# --- Testes decode_audio ---


class TestDecodeAudio:
    def test_decode_wav_16khz(self) -> None:
        """Decodifica WAV PCM 16-bit, 16kHz corretamente."""
        wav_bytes = make_wav_bytes(sample_rate=16000, duration=0.1)
        audio, sr = decode_audio(wav_bytes)
        assert sr == 16000
        assert audio.dtype == np.float32
        assert len(audio) == 1600  # 0.1s * 16000

    def test_decode_wav_8khz(self) -> None:
        """Decodifica WAV 8kHz corretamente."""
        wav_bytes = make_wav_bytes(sample_rate=8000, duration=0.1)
        audio, sr = decode_audio(wav_bytes)
        assert sr == 8000
        assert len(audio) == 800

    def test_decode_wav_44khz(self) -> None:
        """Decodifica WAV 44.1kHz corretamente."""
        wav_bytes = make_wav_bytes(sample_rate=44100, duration=0.1)
        audio, sr = decode_audio(wav_bytes)
        assert sr == 44100
        assert len(audio) == 4410

    def test_decode_stereo_to_mono(self) -> None:
        """Converte audio stereo para mono automaticamente."""
        stereo_bytes = make_stereo_wav_bytes(sample_rate=16000, duration=0.1)
        audio, sr = decode_audio(stereo_bytes)
        assert sr == 16000
        assert audio.ndim == 1  # Mono

    def test_decode_empty_bytes_raises(self) -> None:
        """Bytes vazios levantam AudioFormatError."""
        with pytest.raises(AudioFormatError, match="vazio"):
            decode_audio(b"")

    def test_decode_invalid_bytes_raises(self) -> None:
        """Bytes invalidos levantam AudioFormatError."""
        with pytest.raises(AudioFormatError):
            decode_audio(b"not audio data at all")

    def test_decode_returns_float32(self) -> None:
        """Audio decodificado e sempre float32."""
        wav_bytes = make_wav_bytes()
        audio, _ = decode_audio(wav_bytes)
        assert audio.dtype == np.float32

    def test_decode_values_in_range(self) -> None:
        """Valores decodificados estao no range [-1.0, 1.0]."""
        wav_bytes = make_wav_bytes(amplitude=1.0)
        audio, _ = decode_audio(wav_bytes)
        assert np.all(audio >= -1.0)
        assert np.all(audio <= 1.0)


# --- Testes encode_pcm16 ---


class TestEncodePcm16:
    def test_encode_produces_valid_wav(self) -> None:
        """Encode produz bytes WAV validos que podem ser re-decodificados."""
        audio = np.array([0.0, 0.5, -0.5, 1.0, -1.0], dtype=np.float32)
        wav_bytes = encode_pcm16(audio, 16000)

        # Deve ser re-decodificavel
        decoded, sr = decode_audio(wav_bytes)
        assert sr == 16000
        assert len(decoded) == 5

    def test_encode_clamps_values(self) -> None:
        """Encode limita valores fora de [-1.0, 1.0] sem overflow."""
        audio = np.array([2.0, -2.0, 0.0], dtype=np.float32)
        wav_bytes = encode_pcm16(audio, 16000)

        # Verificar via wave stdlib que o WAV e valido
        with wave.open(io.BytesIO(wav_bytes), "rb") as wf:
            assert wf.getnchannels() == 1
            assert wf.getsampwidth() == 2
            assert wf.getframerate() == 16000
            assert wf.getnframes() == 3

    def test_encode_preserves_sample_rate(self) -> None:
        """Sample rate e preservado no header WAV."""
        audio = np.zeros(100, dtype=np.float32)
        for sr in [8000, 16000, 44100, 48000]:
            wav_bytes = encode_pcm16(audio, sr)
            with wave.open(io.BytesIO(wav_bytes), "rb") as wf:
                assert wf.getframerate() == sr

    def test_roundtrip_decode_encode(self) -> None:
        """Roundtrip decode -> encode preserva audio (com quantizacao PCM16)."""
        original_wav = make_wav_bytes(sample_rate=16000, duration=0.05)
        audio, sr = decode_audio(original_wav)
        re_encoded = encode_pcm16(audio, sr)
        re_decoded, sr2 = decode_audio(re_encoded)

        assert sr == sr2
        assert len(audio) == len(re_decoded)
        # Tolerancia de quantizacao PCM16 (1/32768 ~= 3e-5)
        np.testing.assert_allclose(audio, re_decoded, atol=1e-4)


# --- Testes AudioPreprocessingPipeline ---


class TestAudioPreprocessingPipeline:
    def test_pipeline_zero_stages_returns_pcm16(self) -> None:
        """Pipeline sem stages decodifica e re-codifica audio como PCM16 WAV."""
        config = PreprocessingConfig()
        pipeline = AudioPreprocessingPipeline(config)
        input_wav = make_wav_bytes(sample_rate=16000, duration=0.05)

        result = pipeline.process(input_wav)

        # Resultado deve ser WAV valido
        audio, sr = decode_audio(result)
        assert sr == 16000
        assert audio.dtype == np.float32

    def test_pipeline_with_passthrough_stage(self) -> None:
        """Pipeline com stage passthrough retorna audio equivalente."""
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
        """Pipeline executa stages em sequencia."""
        config = PreprocessingConfig()
        # Dois stages de ganho 0.5 -> resultado final = 0.25 * original
        pipeline = AudioPreprocessingPipeline(
            config,
            stages=[GainStage(factor=0.5), GainStage(factor=0.5)],
        )
        input_wav = make_wav_bytes(sample_rate=16000, duration=0.05, amplitude=0.8)

        result = pipeline.process(input_wav)
        result_audio, _ = decode_audio(result)
        input_audio, _ = decode_audio(input_wav)

        # Audio resultante deve ser ~0.25 do original
        expected = input_audio * 0.25
        np.testing.assert_allclose(result_audio, expected, atol=1e-3)

    def test_pipeline_preserves_sample_rate_through_stages(self) -> None:
        """Sample rate e preservado quando stages nao o alteram."""
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
        """Stage pode alterar sample rate e o pipeline propaga."""
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
        """Pipeline levanta AudioFormatError para audio invalido."""
        config = PreprocessingConfig()
        pipeline = AudioPreprocessingPipeline(config)

        with pytest.raises(AudioFormatError):
            pipeline.process(b"not valid audio")

    def test_pipeline_empty_audio_raises(self) -> None:
        """Pipeline levanta AudioFormatError para bytes vazios."""
        config = PreprocessingConfig()
        pipeline = AudioPreprocessingPipeline(config)

        with pytest.raises(AudioFormatError, match="vazio"):
            pipeline.process(b"")

    def test_pipeline_config_accessible(self) -> None:
        """Config e acessivel via property."""
        config = PreprocessingConfig(target_sample_rate=8000)
        pipeline = AudioPreprocessingPipeline(config)
        assert pipeline.config.target_sample_rate == 8000

    def test_pipeline_stages_property_returns_copy(self) -> None:
        """Property stages retorna copia da lista interna."""
        config = PreprocessingConfig()
        stages = [PassthroughStage()]
        pipeline = AudioPreprocessingPipeline(config, stages=stages)

        returned = pipeline.stages
        returned.append(GainStage())  # Modificar copia
        assert len(pipeline.stages) == 1  # Original nao muda

    def test_pipeline_with_fixture_audio(self, audio_16khz: None) -> None:
        """Pipeline processa fixtures de audio do projeto."""
        from pathlib import Path

        fixture_path = Path(__file__).parent.parent / "fixtures" / "audio" / "sample_16khz.wav"
        if not fixture_path.exists():
            pytest.skip("Fixture de audio nao encontrada")

        audio_bytes = fixture_path.read_bytes()
        config = PreprocessingConfig()
        pipeline = AudioPreprocessingPipeline(config)

        result = pipeline.process(audio_bytes)
        result_audio, result_sr = decode_audio(result)
        assert result_sr == 16000
        assert len(result_audio) > 0
