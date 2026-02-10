"""Testes de integracao end-to-end do worker STT.

Requerem:
- faster-whisper instalado
- Modelo faster-whisper-tiny baixado
- GPU nao necessaria (usa CPU)

Executar com:
    python -m pytest tests/integration/ -m integration -v
"""

from __future__ import annotations

import asyncio
import math
import struct

import pytest

pytestmark = pytest.mark.integration


def _generate_speech_audio_bytes(duration: float = 1.0, sample_rate: int = 16000) -> bytes:
    """Gera audio PCM 16-bit sintetico (sine tone, nao fala real).

    Para testes de integracao reais, substituir por audio com fala.
    """
    samples = []
    for i in range(int(sample_rate * duration)):
        t = i / sample_rate
        value = int(32767 * 0.5 * math.sin(2 * math.pi * 440.0 * t))
        samples.append(value)
    return struct.pack(f"<{len(samples)}h", *samples)


class TestFasterWhisperBackendIntegration:
    """Testes que usam FasterWhisperBackend com modelo real."""

    async def test_load_and_health(self) -> None:
        from macaw.workers.stt.faster_whisper import FasterWhisperBackend

        backend = FasterWhisperBackend()
        await backend.load("tiny", {"model_size": "tiny", "compute_type": "int8", "device": "cpu"})

        health = await backend.health()
        assert health["status"] == "ok"

        await backend.unload()

    async def test_transcribe_sine_tone(self) -> None:
        """Sine tone nao contem fala — resultado esperado e texto vazio ou curto."""
        from macaw.workers.stt.faster_whisper import FasterWhisperBackend

        backend = FasterWhisperBackend()
        await backend.load("tiny", {"model_size": "tiny", "compute_type": "int8", "device": "cpu"})

        audio = _generate_speech_audio_bytes(duration=2.0)
        result = await backend.transcribe_file(audio)

        # Nao validamos texto especifico — sine tone gera output imprevisivel.
        # Validamos que o pipeline nao crasha e retorna tipos corretos.
        assert isinstance(result.text, str)
        assert result.language is not None
        assert result.duration > 0

        await backend.unload()

    async def test_transcribe_returns_segments(self) -> None:
        from macaw.workers.stt.faster_whisper import FasterWhisperBackend

        backend = FasterWhisperBackend()
        await backend.load("tiny", {"model_size": "tiny", "compute_type": "int8", "device": "cpu"})

        audio = _generate_speech_audio_bytes(duration=3.0)
        result = await backend.transcribe_file(audio)

        # O tipo de segmentos deve ser tuple
        assert isinstance(result.segments, tuple)

        await backend.unload()


class TestWorkerGRPCIntegration:
    """Testes que iniciam worker real como subprocess e comunicam via gRPC.

    Requerem modelo tiny disponivel.
    """

    async def test_worker_subprocess_health(self) -> None:
        """Testa spawn de worker, health check, e shutdown."""
        from macaw.workers.manager import WorkerManager, WorkerState

        manager = WorkerManager()
        handle = await manager.spawn_worker(
            model_name="tiny",
            port=50099,
            engine="faster-whisper",
            model_path="tiny",
            engine_config={
                "model_size": "tiny",
                "compute_type": "int8",
                "device": "cpu",
            },
        )

        # Esperar worker ficar READY (pode demorar com download de modelo)
        for _ in range(60):
            if handle.state == WorkerState.READY:
                break
            await asyncio.sleep(1.0)

        assert handle.state == WorkerState.READY

        await manager.stop_all()
        assert handle.state == WorkerState.STOPPED
