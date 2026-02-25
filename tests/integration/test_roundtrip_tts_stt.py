"""Round-trip TTS -> STT test (end-to-end quality).

Generates audio via KokoroBackend (TTS), feeds it to FasterWhisperBackend (STT),
and compares the original text with the transcribed text.

Requires real models installed:
  - kokoro-v1 at ~/.macaw/models/kokoro-v1/
  - faster-whisper-tiny at ~/.macaw/models/faster-whisper-tiny/

Marked as @pytest.mark.integration -- does not run in standard CI.
"""

from __future__ import annotations

import os
import re
import struct

import numpy as np
import pytest

# Model paths
_KOKORO_MODEL_PATH = os.path.expanduser("~/.macaw/models/kokoro-v1")
_FW_MODEL_PATH = os.path.expanduser("~/.macaw/models/faster-whisper-tiny")

# Skip if models not installed
_HAS_KOKORO_MODEL = os.path.isdir(_KOKORO_MODEL_PATH) and os.path.isfile(
    os.path.join(_KOKORO_MODEL_PATH, "config.json")
)
_HAS_FW_MODEL = os.path.isdir(_FW_MODEL_PATH)

# Try to import engines (optional)
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
    _SKIP_REASON = "kokoro not installed"
elif not _HAS_FW_LIB:
    _SKIP_REASON = "faster-whisper not installed"
elif not _HAS_KOKORO_MODEL:
    _SKIP_REASON = f"kokoro-v1 model not found at {_KOKORO_MODEL_PATH}"
elif not _HAS_FW_MODEL:
    _SKIP_REASON = f"faster-whisper-tiny model not found at {_FW_MODEL_PATH}"

pytestmark = [
    pytest.mark.integration,
    pytest.mark.skipif(bool(_SKIP_REASON), reason=_SKIP_REASON or "n/a"),
]


def _normalize_text(text: str) -> str:
    """Normalizes text for comparison: lowercase, no punctuation, trim."""
    text = text.lower().strip()
    text = re.sub(r"[^a-z0-9 ]", "", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def _word_overlap_ratio(original: str, transcribed: str) -> float:
    """Calculates the fraction of original words present in the transcription.

    Returns a value between 0.0 and 1.0.
    """
    orig_words = set(_normalize_text(original).split())
    trans_words = set(_normalize_text(transcribed).split())
    if not orig_words:
        return 0.0
    return len(orig_words & trans_words) / len(orig_words)


def _pcm16_to_wav_bytes(pcm_data: bytes, sample_rate: int) -> bytes:
    """Converts PCM 16-bit mono to WAV in memory."""
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
        """Loads KokoroBackend with real model (once per class)."""
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
        """Loads FasterWhisperBackend with real model (once per class)."""
        from macaw.workers.stt.faster_whisper import FasterWhisperBackend

        backend = FasterWhisperBackend()
        await backend.load(
            _FW_MODEL_PATH,
            {"model_size": "tiny", "compute_type": "int8", "device": "cpu"},
        )
        yield backend
        await backend.unload()

    async def _synthesize_text(self, tts_backend: object, text: str) -> bytes:
        """Synthesizes text and returns concatenated PCM 16-bit audio."""
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
        """Transcribes PCM 16-bit audio and returns text."""
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
        """Executes round-trip: text -> TTS -> STT -> text.

        Returns:
            Tuple (transcribed_text, word_overlap_ratio).
        """
        # TTS: text -> PCM 16-bit at 24kHz
        pcm_24k = await self._synthesize_text(tts_backend, text)
        assert len(pcm_24k) > 0, f"TTS returned empty audio for: {text!r}"

        # Resample 24kHz -> 16kHz (STT expects 16kHz)
        pcm_16k = _resample_pcm16(pcm_24k, from_rate=24000, to_rate=16000)
        assert len(pcm_16k) > 0, "Resample returned empty audio"

        # STT: PCM 16-bit 16kHz -> text
        transcribed = await self._transcribe_audio(stt_backend, pcm_16k, language=language)

        overlap = _word_overlap_ratio(text, transcribed)
        return transcribed, overlap

    async def test_simple_greeting(self, tts_backend: object, stt_backend: object) -> None:
        """Simple greeting phrase."""
        text = "Hello, how can I help you today?"
        transcribed, overlap = await self._roundtrip(tts_backend, stt_backend, text)

        assert len(transcribed) > 0, "STT returned empty text"
        assert overlap >= 0.5, (
            f"Word overlap too low ({overlap:.0%}). "
            f"Original: {text!r}, Transcribed: {transcribed!r}"
        )

    async def test_sentence_with_numbers(self, tts_backend: object, stt_backend: object) -> None:
        """Phrase with numbers (challenging for TTS+STT)."""
        text = "Please transfer one thousand dollars to account number five."
        transcribed, overlap = await self._roundtrip(tts_backend, stt_backend, text)

        assert len(transcribed) > 0, "STT returned empty text"
        assert overlap >= 0.4, (
            f"Word overlap too low ({overlap:.0%}). "
            f"Original: {text!r}, Transcribed: {transcribed!r}"
        )

    async def test_pangram(self, tts_backend: object, stt_backend: object) -> None:
        """Classic pangram -- tests phoneme diversity."""
        text = "The quick brown fox jumps over the lazy dog."
        transcribed, overlap = await self._roundtrip(tts_backend, stt_backend, text)

        assert len(transcribed) > 0, "STT returned empty text"
        assert overlap >= 0.5, (
            f"Word overlap too low ({overlap:.0%}). "
            f"Original: {text!r}, Transcribed: {transcribed!r}"
        )

    async def test_tts_produces_valid_audio(self, tts_backend: object) -> None:
        """Verifies that TTS produces audio with reasonable amplitude (not silence)."""
        pcm_data = await self._synthesize_text(tts_backend, "Hello world")
        assert len(pcm_data) > 1000, "Audio muito curto"

        # Convert to numpy and check amplitude
        audio_array = np.frombuffer(pcm_data, dtype=np.int16).astype(np.float32)
        rms = np.sqrt(np.mean(audio_array**2))
        assert rms > 100.0, f"Audio appears to be silence (RMS={rms:.1f})"

    async def test_roundtrip_reports_quality(
        self, tts_backend: object, stt_backend: object
    ) -> None:
        """Report test: prints quality metrics for human analysis."""
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

        # Print report for human analysis
        print("\n" + "=" * 70)
        print("ROUND-TRIP TTS->STT QUALITY REPORT")
        print("=" * 70)
        for r in results:
            status = "OK" if float(str(r["overlap"])) >= 0.5 else "LOW"
            print(f"\n[{status}] Overlap: {float(str(r['overlap'])):.0%}")
            print(f"  Original:    {r['original']}")
            print(f"  Transcribed: {r['transcribed']}")
        print("=" * 70)

        # At least 3 of 5 phrases must have overlap >= 50%
        good_count = sum(1 for r in results if float(str(r["overlap"])) >= 0.5)
        assert good_count >= 3, (
            f"Only {good_count}/5 phrases with overlap >= 50%. "
            "TTS->STT audio quality below acceptable threshold."
        )


def _resample_pcm16(pcm_data: bytes, *, from_rate: int, to_rate: int) -> bytes:
    """Resamples PCM 16-bit audio from from_rate to to_rate.

    Uses scipy if available, otherwise performs simple downsampling via decimation.
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
        # Fallback: simple decimation (works for 24k->16k = ratio 2/3)
        ratio = to_rate / from_rate
        new_len = int(len(audio) * ratio)
        indices = np.linspace(0, len(audio) - 1, new_len)
        resampled = np.asarray(np.interp(indices, np.arange(len(audio)), audio))

    return resampled.astype(np.int16).tobytes()
