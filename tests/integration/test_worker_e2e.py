"""End-to-end integration tests for the STT worker.

Requirements:
- faster-whisper installed
- faster-whisper-tiny model downloaded
- GPU not required (uses CPU)

Run with:
    python -m pytest tests/integration/ -m integration -v
"""

from __future__ import annotations

import asyncio
import math
import struct

import pytest

pytestmark = pytest.mark.integration


def _generate_speech_audio_bytes(duration: float = 1.0, sample_rate: int = 16000) -> bytes:
    """Generates synthetic PCM 16-bit audio (sine tone, not real speech).

    For real integration tests, replace with speech audio.
    """
    samples = []
    for i in range(int(sample_rate * duration)):
        t = i / sample_rate
        value = int(32767 * 0.5 * math.sin(2 * math.pi * 440.0 * t))
        samples.append(value)
    return struct.pack(f"<{len(samples)}h", *samples)


class TestFasterWhisperBackendIntegration:
    """Tests using FasterWhisperBackend with a real model."""

    async def test_load_and_health(self) -> None:
        from macaw.workers.stt.faster_whisper import FasterWhisperBackend

        backend = FasterWhisperBackend()
        await backend.load("tiny", {"model_size": "tiny", "compute_type": "int8", "device": "cpu"})

        health = await backend.health()
        assert health["status"] == "ok"

        await backend.unload()

    async def test_transcribe_sine_tone(self) -> None:
        """Sine tone does not contain speech -- expected result is empty or short text."""
        from macaw.workers.stt.faster_whisper import FasterWhisperBackend

        backend = FasterWhisperBackend()
        await backend.load("tiny", {"model_size": "tiny", "compute_type": "int8", "device": "cpu"})

        audio = _generate_speech_audio_bytes(duration=2.0)
        result = await backend.transcribe_file(audio)

        # We don't validate specific text -- sine tone generates unpredictable output.
        # We validate that the pipeline doesn't crash and returns correct types.
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

        # Segments type must be tuple
        assert isinstance(result.segments, tuple)

        await backend.unload()


class TestWorkerGRPCIntegration:
    """Tests that start a real worker as subprocess and communicate via gRPC.

    Require tiny model to be available.
    """

    async def test_worker_subprocess_health(self) -> None:
        """Tests worker spawn, health check, and shutdown."""
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

        # Wait for worker to become READY (may take time with model download)
        for _ in range(60):
            if handle.state == WorkerState.READY:
                break
            await asyncio.sleep(1.0)

        assert handle.state == WorkerState.READY

        await manager.stop_all()
        assert handle.state == WorkerState.STOPPED
