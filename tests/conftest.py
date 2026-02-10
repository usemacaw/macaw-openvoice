"""Fixtures compartilhadas para todos os testes."""

from __future__ import annotations

import struct
from pathlib import Path

import pytest

from macaw._types import BatchResult, SegmentDetail, WordTimestamp

FIXTURES_DIR = Path(__file__).parent / "fixtures"
AUDIO_DIR = FIXTURES_DIR / "audio"
MANIFESTS_DIR = FIXTURES_DIR / "manifests"


@pytest.fixture
def audio_16khz() -> Path:
    """Caminho para audio sample PCM 16-bit, 16kHz, mono."""
    path = AUDIO_DIR / "sample_16khz.wav"
    assert path.exists(), f"Fixture de audio nao encontrada: {path}"
    return path


@pytest.fixture
def audio_8khz() -> Path:
    """Caminho para audio sample PCM 16-bit, 8kHz, mono."""
    path = AUDIO_DIR / "sample_8khz.wav"
    assert path.exists(), f"Fixture de audio nao encontrada: {path}"
    return path


@pytest.fixture
def audio_44khz() -> Path:
    """Caminho para audio sample PCM 16-bit, 44.1kHz, mono."""
    path = AUDIO_DIR / "sample_44khz.wav"
    assert path.exists(), f"Fixture de audio nao encontrada: {path}"
    return path


@pytest.fixture
def valid_stt_manifest_path() -> Path:
    """Caminho para manifesto STT valido."""
    path = MANIFESTS_DIR / "valid_stt.yaml"
    assert path.exists(), f"Fixture de manifesto nao encontrada: {path}"
    return path


@pytest.fixture
def valid_tts_manifest_path() -> Path:
    """Caminho para manifesto TTS valido."""
    path = MANIFESTS_DIR / "valid_tts.yaml"
    assert path.exists(), f"Fixture de manifesto nao encontrada: {path}"
    return path


@pytest.fixture
def minimal_manifest_path() -> Path:
    """Caminho para manifesto com campos minimos."""
    path = MANIFESTS_DIR / "minimal.yaml"
    assert path.exists(), f"Fixture de manifesto nao encontrada: {path}"
    return path


@pytest.fixture
def valid_stt_wenet_manifest_path() -> Path:
    """Caminho para manifesto STT WeNet CTC valido."""
    path = MANIFESTS_DIR / "valid_stt_wenet.yaml"
    assert path.exists(), f"Fixture de manifesto nao encontrada: {path}"
    return path


@pytest.fixture
def invalid_manifest_path() -> Path:
    """Caminho para manifesto invalido (campos obrigatorios faltando)."""
    path = MANIFESTS_DIR / "invalid_missing.yaml"
    assert path.exists(), f"Fixture de manifesto nao encontrada: {path}"
    return path


@pytest.fixture
def sample_audio_bytes() -> bytes:
    """1 segundo de audio PCM 16-bit, 16kHz, mono (tom 440Hz)."""
    sample_rate = 16000
    duration = 1.0
    frequency = 440.0
    import math

    samples = []
    for i in range(int(sample_rate * duration)):
        t = i / sample_rate
        value = int(32767 * 0.5 * math.sin(2 * math.pi * frequency * t))
        samples.append(value)
    return struct.pack(f"<{len(samples)}h", *samples)


@pytest.fixture
def sample_batch_result() -> BatchResult:
    """BatchResult de exemplo para testes de conversao."""
    return BatchResult(
        text="Ola, como posso ajudar?",
        language="pt",
        duration=2.5,
        segments=(
            SegmentDetail(
                id=0,
                start=0.0,
                end=2.5,
                text="Ola, como posso ajudar?",
                avg_logprob=-0.25,
                no_speech_prob=0.01,
                compression_ratio=1.1,
            ),
        ),
        words=(
            WordTimestamp(word="Ola", start=0.0, end=0.5, probability=0.95),
            WordTimestamp(word="como", start=0.6, end=0.9, probability=0.90),
            WordTimestamp(word="posso", start=1.0, end=1.3, probability=0.92),
            WordTimestamp(word="ajudar", start=1.4, end=2.5, probability=0.88),
        ),
    )
