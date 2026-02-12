---
title: Adding an Engine
sidebar_position: 4
---

# Adding an Engine

Macaw is **engine-agnostic**. Adding a new STT or TTS engine requires implementing the backend interface, registering it in the factory, creating a model manifest, and writing tests. Zero changes to the runtime core.

## Overview

```
Step 1: Implement the Backend interface (STTBackend or TTSBackend)
Step 2: Register in the factory function
Step 3: Create a model manifest (macaw.yaml)
Step 4: Declare dependencies (optional extra)
Step 5: Write tests
```

## Step 1: Implement the Backend

### STT Engine

Create a new file in `src/macaw/workers/stt/`:

```python title="src/macaw/workers/stt/my_engine.py"
from macaw.workers.stt.interface import (
    STTBackend,
    STTArchitecture,
    EngineCapabilities,
    BatchResult,
    TranscriptSegment,
)
from typing import AsyncIterator


class MyEngineBackend(STTBackend):
    """STT backend for MyEngine."""

    @property
    def architecture(self) -> STTArchitecture:
        # Choose one:
        # - STTArchitecture.ENCODER_DECODER  (like Whisper)
        # - STTArchitecture.CTC              (like WeNet)
        # - STTArchitecture.STREAMING_NATIVE (like Paraformer)
        return STTArchitecture.ENCODER_DECODER

    async def load(self, model_path: str, config: dict) -> None:
        """Load the model into memory."""
        # Initialize your engine here
        # config comes from macaw.yaml engine_config section
        ...

    def capabilities(self) -> EngineCapabilities:
        """Declare what this engine supports."""
        return EngineCapabilities(
            supports_hot_words=False,
            supports_initial_prompt=True,
            supports_batch=True,
            supports_word_timestamps=True,
            max_concurrent=1,  # GPU concurrency limit
        )

    async def transcribe_file(
        self,
        audio_data: bytes,
        language: str | None = None,
        initial_prompt: str | None = None,
        hot_words: list[str] | None = None,
        temperature: float = 0.0,
        word_timestamps: bool = False,
    ) -> BatchResult:
        """Transcribe a complete audio file."""
        # audio_data is PCM 16-bit, 16kHz, mono (already preprocessed)
        # Return BatchResult with text, segments, language, duration
        ...

    async def transcribe_stream(
        self,
        audio_chunks: AsyncIterator[bytes],
        language: str | None = None,
        initial_prompt: str | None = None,
        hot_words: list[str] | None = None,
    ) -> AsyncIterator[TranscriptSegment]:
        """Transcribe streaming audio."""
        # Yield TranscriptSegment for each partial/final result
        async for chunk in audio_chunks:
            # Process chunk, yield results
            ...

    async def unload(self) -> None:
        """Free model resources."""
        ...

    async def health(self) -> dict:
        """Return health status."""
        return {"status": "ok", "engine": "my_engine"}
```

### TTS Engine

Create a new file in `src/macaw/workers/tts/`:

```python title="src/macaw/workers/tts/my_tts.py"
from macaw.workers.tts.interface import TTSBackend, VoiceInfo
from typing import AsyncIterator


class MyTTSBackend(TTSBackend):
    """TTS backend for MyTTS."""

    async def load(self, model_path: str, config: dict) -> None:
        """Load the TTS model."""
        ...

    async def synthesize(
        self,
        text: str,
        voice: str = "default",
        sample_rate: int = 24000,
        speed: float = 1.0,
    ) -> AsyncIterator[bytes]:
        """Synthesize speech from text.

        Yields PCM 16-bit audio chunks for streaming with low TTFB.
        """
        # Process text and yield audio chunks
        # Each chunk should be ~4096 bytes for smooth streaming
        ...

    async def voices(self) -> list[VoiceInfo]:
        """List available voices."""
        return [
            VoiceInfo(id="default", name="Default", language="en"),
        ]

    async def unload(self) -> None:
        """Free model resources."""
        ...

    async def health(self) -> dict:
        """Return health status."""
        return {"status": "ok", "engine": "my_tts"}
```

:::info Streaming TTS
The `synthesize()` method returns an `AsyncIterator[bytes]` — not a single bytes object. This enables **streaming with low TTFB** (Time to First Byte). Yield audio chunks as they become available rather than waiting for the full synthesis to complete.
:::

## Step 2: Register in the Factory

Add your engine to the factory function that creates backends:

```python title="Registration"
# In the worker factory (e.g., _create_backend)
def _create_backend(engine: str) -> STTBackend:
    match engine:
        case "faster-whisper":
            from macaw.workers.stt.faster_whisper import FasterWhisperBackend
            return FasterWhisperBackend()
        case "wenet":
            from macaw.workers.stt.wenet import WeNetBackend
            return WeNetBackend()
        case "my-engine":  # Add your engine here
            from macaw.workers.stt.my_engine import MyEngineBackend
            return MyEngineBackend()
        case _:
            raise ValueError(f"Unknown engine: {engine}")
```

:::tip Lazy imports
Use lazy imports inside the `match` branches. This way, engine dependencies are only loaded when that specific engine is requested. Users who don't use your engine don't need to install its dependencies.
:::

## Step 3: Create a Model Manifest

Every model needs a `macaw.yaml` manifest in its model directory:

```yaml title="models/my-model/macaw.yaml"
name: my-model-large
type: stt                    # stt or tts
engine: my-engine            # matches the factory key
architecture: encoder-decoder # encoder-decoder, ctc, or streaming-native

capabilities:
  hot_words: false
  initial_prompt: true
  batch: true
  word_timestamps: true

engine_config:
  beam_size: 5
  vad_filter: false          # Always false — runtime handles VAD
  compute_type: float16
  device: cuda               # cuda or cpu

files:
  - model.bin
  - tokenizer.json
  - config.json
```

### Architecture Field

The `architecture` field tells the runtime how to adapt the streaming pipeline:

| Architecture | LocalAgreement | Cross-segment Context | Native Partials |
|-------------|:-:|:-:|:-:|
| `encoder-decoder` | Yes (confirms tokens across passes) | Yes (224 tokens from previous segment) | No |
| `ctc` | No (not needed) | No (`initial_prompt` not supported) | Yes |
| `streaming-native` | No (not needed) | No | Yes |

:::warning Set the right architecture
Choosing the wrong architecture will cause incorrect streaming behavior. If your engine produces native partial transcripts, use `ctc` or `streaming-native`. If it needs multiple inference passes to produce stable output, use `encoder-decoder`.
:::

### Engine Config

The `engine_config` section is passed directly to your `load()` method as a dict. Define whatever configuration your engine needs:

```python
async def load(self, model_path: str, config: dict) -> None:
    beam_size = config.get("beam_size", 5)
    compute_type = config.get("compute_type", "float16")
    device = config.get("device", "cuda")
    # Initialize engine with these settings
```

## Step 4: Declare Dependencies

If your engine requires additional Python packages, add them as an optional extra in `pyproject.toml`:

```toml title="pyproject.toml"
[project.optional-dependencies]
my-engine = ["my-engine-lib>=1.0"]

# Users install with:
# pip install macaw-openvoice[my-engine]
```

This keeps the base Macaw installation lightweight — users only install engine dependencies they actually use.

## Step 5: Write Tests

### Unit Tests

Test your backend in isolation with mocked inference:

```python title="tests/unit/workers/stt/test_my_engine.py"
import pytest
from unittest.mock import AsyncMock, patch

from macaw.workers.stt.my_engine import MyEngineBackend
from macaw.workers.stt.interface import STTArchitecture


class TestMyEngineBackend:
    def test_architecture(self):
        backend = MyEngineBackend()
        assert backend.architecture == STTArchitecture.ENCODER_DECODER

    def test_capabilities(self):
        backend = MyEngineBackend()
        caps = backend.capabilities()
        assert caps.supports_batch is True
        assert caps.max_concurrent == 1

    async def test_transcribe_file(self):
        backend = MyEngineBackend()
        # Mock the engine's inference
        with patch.object(backend, "_inference", new_callable=AsyncMock) as mock:
            mock.return_value = "Hello world"
            result = await backend.transcribe_file(b"fake_audio")
            assert result.text == "Hello world"

    async def test_health(self):
        backend = MyEngineBackend()
        status = await backend.health()
        assert status["status"] == "ok"
```

### Integration Tests

Test with a real model (mark as integration):

```python title="tests/integration/workers/stt/test_my_engine_integration.py"
import pytest

from macaw.workers.stt.my_engine import MyEngineBackend


@pytest.mark.integration
class TestMyEngineIntegration:
    async def test_transcribe_real_audio(self, audio_440hz_wav):
        backend = MyEngineBackend()
        await backend.load("path/to/model", {"device": "cpu"})
        try:
            result = await backend.transcribe_file(audio_440hz_wav)
            assert isinstance(result.text, str)
        finally:
            await backend.unload()
```

## Checklist

Before submitting your engine:

- [ ] Implements all abstract methods from `STTBackend` or `TTSBackend`
- [ ] `architecture` property returns the correct type
- [ ] `capabilities()` accurately reflects engine features
- [ ] `vad_filter: false` in the manifest (runtime handles VAD)
- [ ] Lazy import in the factory function
- [ ] Optional dependency declared in `pyproject.toml`
- [ ] Unit tests with mocked inference
- [ ] Integration tests marked with `@pytest.mark.integration`
- [ ] `health()` returns meaningful status

## What You Don't Need to Touch

The engine-agnostic design means you do **not** modify:

| Component | Reason |
|-----------|--------|
| API Server | Routes are engine-agnostic |
| Session Manager | Adapts automatically via `architecture` field |
| VAD Pipeline | Runs before audio reaches the engine |
| Preprocessing | Engines receive normalized PCM 16kHz |
| Postprocessing | ITN runs after transcription, independent of engine |
| Scheduler | Routes requests by model name, not engine type |
| CLI | Commands work with any registered model |
