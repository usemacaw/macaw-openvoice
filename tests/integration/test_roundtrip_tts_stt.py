"""Teste de round-trip TTS -> STT (qualidade end-to-end).

Gera audio via KokoroBackend (TTS), alimenta ao FasterWhisperBackend (STT),
e compara o texto original com o texto transcrito.

Requer modelos reais instalados:
  - kokoro-v1 em ~/.macaw/models/kokoro-v1/
  - faster-whisper-tiny em ~/.macaw/models/faster-whisper-tiny/

Marcado como @pytest.mark.integration — nao roda no CI padrao.
"""

from __future__ import annotations

import os
import re
import struct

import numpy as np
import pytest

# Paths dos modelos
_KOKORO_MODEL_PATH = os.path.expanduser("~/.macaw/models/kokoro-v1")
_FW_MODEL_PATH = os.path.expanduser("~/.macaw/models/faster-whisper-tiny")

# Pular se modelos nao instalados
_HAS_KOKORO_MODEL = os.path.isdir(_KOKORO_MODEL_PATH) and os.path.isfile(
    os.path.join(_KOKORO_MODEL_PATH, "config.json")
)
_HAS_FW_MODEL = os.path.isdir(_FW_MODEL_PATH)

# Tentar importar engines (opcionais)
try:
    import kokoro as _kokoro_check  # noqa: F401

    _HAS_KOKORO_LIB = True
except ImportError:
    _HAS_KOKORO_LIB = False

try:
    import faster_whisper as _fw_check  # noqa: F401

    _HAS_FW_LIB = True
except ImportError:
    _HAS_FW_LIB = False

_SKIP_REASON = ""
if not _HAS_KOKORO_LIB:
    _SKIP_REASON = "kokoro nao instalado"
elif not _HAS_FW_LIB:
    _SKIP_REASON = "faster-whisper nao instalado"
elif not _HAS_KOKORO_MODEL:
    _SKIP_REASON = f"modelo kokoro-v1 nao encontrado em {_KOKORO_MODEL_PATH}"
elif not _HAS_FW_MODEL:
    _SKIP_REASON = f"modelo faster-whisper-tiny nao encontrado em {_FW_MODEL_PATH}"

pytestmark = [
    pytest.mark.integration,
    pytest.mark.skipif(bool(_SKIP_REASON), reason=_SKIP_REASON or "n/a"),
]


def _normalize_text(text: str) -> str:
    """Normaliza texto para comparacao: lowercase, sem pontuacao, trim."""
    text = text.lower().strip()
    text = re.sub(r"[^a-z0-9 ]", "", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def _word_overlap_ratio(original: str, transcribed: str) -> float:
    """Calcula a fracao de palavras do original presentes na transcricao.

    Retorna valor entre 0.0 e 1.0.
    """
    orig_words = set(_normalize_text(original).split())
    trans_words = set(_normalize_text(transcribed).split())
    if not orig_words:
        return 0.0
    return len(orig_words & trans_words) / len(orig_words)


def _pcm16_to_wav_bytes(pcm_data: bytes, sample_rate: int) -> bytes:
    """Converte PCM 16-bit mono para WAV em memoria."""
    num_channels = 1
    bits_per_sample = 16
    byte_rate = sample_rate * num_channels * bits_per_sample // 8
    block_align = num_channels * bits_per_sample // 8
    data_size = len(pcm_data)

    header = b"RIFF"
    header += struct.pack("<I", 36 + data_size)
    header += b"WAVE"
    header += b"fmt "
    header += struct.pack("<I", 16)
    header += struct.pack("<H", 1)
    header += struct.pack("<H", num_channels)
    header += struct.pack("<I", sample_rate)
    header += struct.pack("<I", byte_rate)
    header += struct.pack("<H", block_align)
    header += struct.pack("<H", bits_per_sample)
    header += b"data"
    header += struct.pack("<I", data_size)

    return header + pcm_data


class TestRoundtripTTSSTT:
    """Teste end-to-end: Text -> TTS (Kokoro) -> Audio -> STT (Faster-Whisper) -> Text."""

    @pytest.fixture(scope="class")
    async def tts_backend(self) -> object:
        """Carrega KokoroBackend com modelo real (uma vez por classe)."""
        from macaw.workers.tts.kokoro import KokoroBackend

        backend = KokoroBackend()
        await backend.load(
            _KOKORO_MODEL_PATH,
            {"device": "cpu", "lang_code": "a", "default_voice": "af_heart"},
        )
        yield backend
        await backend.unload()

    @pytest.fixture(scope="class")
    async def stt_backend(self) -> object:
        """Carrega FasterWhisperBackend com modelo real (uma vez por classe)."""
        from macaw.workers.stt.faster_whisper import FasterWhisperBackend

        backend = FasterWhisperBackend()
        await backend.load(
            _FW_MODEL_PATH,
            {"model_size": "tiny", "compute_type": "int8", "device": "cpu"},
        )
        yield backend
        await backend.unload()

    async def _synthesize_text(self, tts_backend: object, text: str) -> bytes:
        """Sintetiza texto e retorna audio PCM 16-bit concatenado."""
        from macaw.workers.tts.kokoro import KokoroBackend

        assert isinstance(tts_backend, KokoroBackend)
        chunks: list[bytes] = []
        async for chunk in tts_backend.synthesize(text, voice="default", speed=1.0):
            chunks.append(chunk)
        return b"".join(chunks)

    async def _transcribe_audio(
        self,
        stt_backend: object,
        pcm_data: bytes,
        *,
        language: str = "en",
    ) -> str:
        """Transcreve audio PCM 16-bit e retorna texto."""
        from macaw.workers.stt.faster_whisper import FasterWhisperBackend

        assert isinstance(stt_backend, FasterWhisperBackend)
        result = await stt_backend.transcribe_file(
            pcm_data,
            language=language,
        )
        return result.text

    async def _roundtrip(
        self,
        tts_backend: object,
        stt_backend: object,
        text: str,
        *,
        language: str = "en",
    ) -> tuple[str, float]:
        """Executa round-trip: text -> TTS -> STT -> text.

        Returns:
            Tupla (texto_transcrito, word_overlap_ratio).
        """
        # TTS: text -> PCM 16-bit at 24kHz
        pcm_24k = await self._synthesize_text(tts_backend, text)
        assert len(pcm_24k) > 0, f"TTS retornou audio vazio para: {text!r}"

        # Resample 24kHz -> 16kHz (STT espera 16kHz)
        pcm_16k = _resample_pcm16(pcm_24k, from_rate=24000, to_rate=16000)
        assert len(pcm_16k) > 0, "Resample retornou audio vazio"

        # STT: PCM 16-bit 16kHz -> text
        transcribed = await self._transcribe_audio(stt_backend, pcm_16k, language=language)

        overlap = _word_overlap_ratio(text, transcribed)
        return transcribed, overlap

    async def test_simple_greeting(self, tts_backend: object, stt_backend: object) -> None:
        """Frase simples de saudacao."""
        text = "Hello, how can I help you today?"
        transcribed, overlap = await self._roundtrip(tts_backend, stt_backend, text)

        assert len(transcribed) > 0, "STT retornou texto vazio"
        assert overlap >= 0.5, (
            f"Word overlap muito baixo ({overlap:.0%}). "
            f"Original: {text!r}, Transcrito: {transcribed!r}"
        )

    async def test_sentence_with_numbers(self, tts_backend: object, stt_backend: object) -> None:
        """Frase com numeros (desafio para TTS+STT)."""
        text = "Please transfer one thousand dollars to account number five."
        transcribed, overlap = await self._roundtrip(tts_backend, stt_backend, text)

        assert len(transcribed) > 0, "STT retornou texto vazio"
        assert overlap >= 0.4, (
            f"Word overlap muito baixo ({overlap:.0%}). "
            f"Original: {text!r}, Transcrito: {transcribed!r}"
        )

    async def test_pangram(self, tts_backend: object, stt_backend: object) -> None:
        """Pangrama classico — testa diversidade de fonemas."""
        text = "The quick brown fox jumps over the lazy dog."
        transcribed, overlap = await self._roundtrip(tts_backend, stt_backend, text)

        assert len(transcribed) > 0, "STT retornou texto vazio"
        assert overlap >= 0.5, (
            f"Word overlap muito baixo ({overlap:.0%}). "
            f"Original: {text!r}, Transcrito: {transcribed!r}"
        )

    async def test_tts_produces_valid_audio(self, tts_backend: object) -> None:
        """Verifica que TTS produz audio com amplitude razoavel (nao silencio)."""
        pcm_data = await self._synthesize_text(tts_backend, "Hello world")
        assert len(pcm_data) > 1000, "Audio muito curto"

        # Converter para numpy e verificar amplitude
        audio_array = np.frombuffer(pcm_data, dtype=np.int16).astype(np.float32)
        rms = np.sqrt(np.mean(audio_array**2))
        assert rms > 100.0, f"Audio parece silencio (RMS={rms:.1f})"

    async def test_roundtrip_reports_quality(
        self, tts_backend: object, stt_backend: object
    ) -> None:
        """Teste de relatorio: imprime metricas de qualidade para analise humana."""
        phrases = [
            "Hello, how can I help you today?",
            "Please transfer one thousand dollars.",
            "The quick brown fox jumps over the lazy dog.",
            "What is the balance on my checking account?",
            "Thank you for calling, have a nice day.",
        ]

        results: list[dict[str, object]] = []
        for phrase in phrases:
            transcribed, overlap = await self._roundtrip(tts_backend, stt_backend, phrase)
            results.append(
                {
                    "original": phrase,
                    "transcribed": transcribed,
                    "overlap": overlap,
                }
            )

        # Imprimir relatorio para analise humana
        print("\n" + "=" * 70)
        print("ROUND-TRIP TTS->STT QUALITY REPORT")
        print("=" * 70)
        for r in results:
            status = "OK" if float(str(r["overlap"])) >= 0.5 else "LOW"
            print(f"\n[{status}] Overlap: {float(str(r['overlap'])):.0%}")
            print(f"  Original:    {r['original']}")
            print(f"  Transcribed: {r['transcribed']}")
        print("=" * 70)

        # Pelo menos 3 de 5 frases devem ter overlap >= 50%
        good_count = sum(1 for r in results if float(str(r["overlap"])) >= 0.5)
        assert good_count >= 3, (
            f"Apenas {good_count}/5 frases com overlap >= 50%. "
            "Qualidade de audio TTS->STT abaixo do aceitavel."
        )


def _resample_pcm16(pcm_data: bytes, *, from_rate: int, to_rate: int) -> bytes:
    """Resample audio PCM 16-bit de from_rate para to_rate.

    Usa scipy se disponivel, senao faz downsampling simples por decimacao.
    """
    if from_rate == to_rate:
        return pcm_data

    audio: np.ndarray = np.frombuffer(pcm_data, dtype=np.int16).astype(np.float32)

    try:
        from scipy.signal import resample_poly

        gcd = int(np.gcd(to_rate, from_rate))
        up = to_rate // gcd
        down = from_rate // gcd
        resampled: np.ndarray = np.asarray(resample_poly(audio, up, down))
    except ImportError:
        # Fallback: decimacao simples (funciona para 24k->16k = ratio 2/3)
        ratio = to_rate / from_rate
        new_len = int(len(audio) * ratio)
        indices = np.linspace(0, len(audio) - 1, new_len)
        resampled = np.asarray(np.interp(indices, np.arange(len(audio)), audio))

    return resampled.astype(np.int16).tobytes()
