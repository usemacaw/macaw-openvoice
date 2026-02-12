---
title: Faster-Whisper
sidebar_position: 2
---

# Faster-Whisper

Faster-Whisper is the primary STT engine in Macaw OpenVoice. It provides high-accuracy multilingual speech recognition using CTranslate2-optimized Whisper models. Five model variants are available in the official catalog, covering use cases from lightweight testing to production-grade transcription.

## Installation

```bash
pip install macaw-openvoice[faster-whisper]
```

This installs `faster-whisper>=1.1,<2.0` as an optional dependency.

## Architecture

Faster-Whisper uses the **encoder-decoder** architecture (based on OpenAI Whisper). This means the runtime adapts the streaming pipeline with:

- **LocalAgreement** — confirms tokens across multiple inference passes before emitting partials
- **Cross-segment context** — passes up to 224 tokens from the previous final transcript as `initial_prompt` to maintain continuity across segments
- **Accumulation** — audio is buffered for ~5 seconds before each inference pass (not frame-by-frame)

```
Audio → [5s accumulation] → Inference → LocalAgreement → Partial/Final
                                ↑
                    initial_prompt (224 tokens from previous final)
```

:::info Why accumulation?
Encoder-decoder models like Whisper process audio in fixed-length windows. Sending tiny chunks would produce poor results. The 5-second accumulation threshold balances latency with transcription quality.
:::

## Model Variants

### large-v3 {#large-v3}

The highest quality Faster-Whisper model. Best accuracy across 100+ languages.

| Property | Value |
|----------|-------|
| Catalog name | `faster-whisper-large-v3` |
| HuggingFace repo | `Systran/faster-whisper-large-v3` |
| Memory | 3,072 MB |
| GPU | Recommended |
| Load time | ~8 seconds |
| Languages | 100+ (auto-detect) |
| Translation | Yes (any → English) |

```bash
macaw pull faster-whisper-large-v3
```

**Best for:** Production workloads where accuracy matters most. Multilingual support. Translation tasks.

### medium {#medium}

Good balance between quality and speed. Suitable for production with GPU.

| Property | Value |
|----------|-------|
| Catalog name | `faster-whisper-medium` |
| HuggingFace repo | `Systran/faster-whisper-medium` |
| Memory | 1,536 MB |
| GPU | Recommended |
| Load time | ~5 seconds |
| Languages | 100+ (auto-detect) |
| Translation | Yes (any → English) |

```bash
macaw pull faster-whisper-medium
```

**Best for:** Production with moderate GPU resources. Near large-v3 quality at lower cost.

### small {#small}

Lightweight model that runs well on CPU. Good quality for common languages.

| Property | Value |
|----------|-------|
| Catalog name | `faster-whisper-small` |
| HuggingFace repo | `Systran/faster-whisper-small` |
| Memory | 512 MB |
| GPU | Optional |
| Load time | ~3 seconds |
| Languages | 100+ (auto-detect) |
| Translation | Yes (any → English) |

```bash
macaw pull faster-whisper-small
```

**Best for:** CPU-only deployments. Development and staging environments. Good speed/accuracy trade-off.

### tiny {#tiny}

Ultra-lightweight model for testing and prototyping. Fastest to load.

| Property | Value |
|----------|-------|
| Catalog name | `faster-whisper-tiny` |
| HuggingFace repo | `Systran/faster-whisper-tiny` |
| Memory | 256 MB |
| GPU | Optional |
| Load time | ~2 seconds |
| Languages | 100+ (auto-detect) |
| Translation | Yes (any → English) |

```bash
macaw pull faster-whisper-tiny
```

**Best for:** Quick testing and prototyping. CI/CD pipelines. Environments with minimal resources.

### distil-large-v3 {#distil-large-v3}

Distilled version of large-v3. Approximately 6x faster with only ~1% WER gap. English only.

| Property | Value |
|----------|-------|
| Catalog name | `distil-whisper-large-v3` |
| HuggingFace repo | `Systran/faster-distil-whisper-large-v3` |
| Memory | 1,536 MB |
| GPU | Recommended |
| Load time | ~5 seconds |
| Languages | English only |
| Translation | No |

```bash
macaw pull distil-whisper-large-v3
```

**Best for:** English-only production workloads where speed matters. High-throughput transcription. When large-v3 is too slow but you want near-equal quality.

## Capabilities

| Capability | Supported | Notes |
|------------|:---------:|-------|
| Streaming | Yes | 5s accumulation threshold |
| Batch inference | Yes | Via `POST /v1/audio/transcriptions` |
| Word timestamps | Yes | Per-word start/end/probability |
| Language detection | Yes | Automatic when `language` is `"auto"` or omitted |
| Translation | Yes | Any language → English (except distil-large-v3) |
| Initial prompt | Yes | Context string to guide transcription |
| Hot words | No | Workaround via `initial_prompt` prefix |
| Partial transcripts | Yes | Via LocalAgreement in streaming mode |

### Hot Words Workaround

Faster-Whisper does not support native keyword boosting. However, Macaw provides a workaround by prepending hot words to the `initial_prompt`:

```python
# In the backend, hot_words are converted to an initial_prompt prefix:
# hot_words=["Macaw", "OpenVoice"] → initial_prompt="Terms: Macaw, OpenVoice."
```

This biases the model toward recognizing these terms but is less reliable than native hot word support (see [WeNet](./wenet) for native hot words).

### Language Handling

| Input | Behavior |
|-------|----------|
| `"auto"` | Auto-detect language (passed as `None` to Faster-Whisper) |
| `"mixed"` | Auto-detect language (same as `"auto"`) |
| `"en"`, `"pt"`, etc. | Force specific language |
| Omitted | Auto-detect |

The model supports 100+ languages. The catalog manifests list `["auto", "en", "pt", "es", "ja", "zh"]` as common examples, but all Whisper-supported languages work.

## Engine Configuration

The `engine_config` section in the model manifest controls Faster-Whisper behavior:

```yaml title="engine_config defaults"
engine_config:
  model_size: "large-v3"    # Model size or path
  compute_type: "float16"   # float16, int8, int8_float16, float32
  device: "auto"            # "auto", "cpu", "cuda"
  beam_size: 5              # Beam search width
  vad_filter: false         # Always false — VAD is handled by the runtime
```

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `model_size` | string | (from catalog) | Whisper model size or path to model directory |
| `compute_type` | string | `"float16"` | Quantization type. `float16` for GPU, `int8` for CPU |
| `device` | string | `"auto"` | Inference device. `"auto"` selects GPU if available |
| `beam_size` | int | `5` | Beam search width. Higher = more accurate, slower |
| `vad_filter` | bool | `false` | Internal VAD filter. **Always `false`** — Macaw handles VAD |

:::warning Never set `vad_filter: true`
The Macaw runtime runs its own VAD pipeline (energy pre-filter + Silero VAD). Enabling the internal Faster-Whisper VAD filter would duplicate the work and produce inconsistent behavior.
:::

## Usage Examples

### Batch Transcription (REST API)

```bash title="Transcribe a file"
curl -X POST http://localhost:8000/v1/audio/transcriptions \
  -F "file=@meeting.wav" \
  -F "model=faster-whisper-large-v3" \
  -F "language=auto" \
  -F "response_format=verbose_json"
```

```bash title="Translate to English"
curl -X POST http://localhost:8000/v1/audio/translations \
  -F "file=@audio_pt.wav" \
  -F "model=faster-whisper-large-v3"
```

### Streaming (WebSocket)

```python title="Python WebSocket client"
import asyncio
import json
import websockets

async def stream_audio():
    uri = "ws://localhost:8000/v1/realtime"
    async with websockets.connect(uri) as ws:
        # Configure session
        await ws.send(json.dumps({
            "type": "session.configure",
            "model": "faster-whisper-large-v3",
            "language": "auto"
        }))

        # Stream audio chunks (16-bit PCM, 16kHz, mono)
        with open("audio.raw", "rb") as f:
            while chunk := f.read(3200):  # 100ms chunks
                await ws.send(chunk)
                
                # Check for transcripts
                try:
                    msg = await asyncio.wait_for(ws.recv(), timeout=0.1)
                    event = json.loads(msg)
                    if event["type"] == "transcript.partial":
                        print(f"  ...{event['text']}")
                    elif event["type"] == "transcript.final":
                        print(f"  >> {event['text']}")
                except asyncio.TimeoutError:
                    pass

asyncio.run(stream_audio())
```

### CLI

```bash title="Transcribe via CLI"
macaw transcribe meeting.wav --model faster-whisper-large-v3

# With word timestamps
macaw transcribe meeting.wav --model faster-whisper-large-v3 --word-timestamps
```

## Comparison of Variants

| | tiny | small | medium | large-v3 | distil-large-v3 |
|---|:-:|:-:|:-:|:-:|:-:|
| **Memory** | 256 MB | 512 MB | 1,536 MB | 3,072 MB | 1,536 MB |
| **GPU needed** | No | No | Recommended | Recommended | Recommended |
| **Load time** | ~2s | ~3s | ~5s | ~8s | ~5s |
| **Languages** | 100+ | 100+ | 100+ | 100+ | English only |
| **Translation** | Yes | Yes | Yes | Yes | No |
| **Relative speed** | Fastest | Fast | Moderate | Slowest | ~6x faster than large-v3 |
| **Best for** | Testing | CPU deploy | Balanced | Max accuracy | Fast English |

## Manifest Reference

Every Faster-Whisper model in the catalog uses the same manifest structure. Here is the full manifest for `faster-whisper-large-v3`:

```yaml title="macaw.yaml (faster-whisper-large-v3)"
name: faster-whisper-large-v3
version: "3.0.0"
engine: faster-whisper
type: stt
description: "Faster Whisper Large V3 - encoder-decoder STT"

capabilities:
  streaming: true
  architecture: encoder-decoder
  languages: ["auto", "en", "pt", "es", "ja", "zh"]
  word_timestamps: true
  translation: true
  partial_transcripts: true
  hot_words: false
  batch_inference: true
  language_detection: true
  initial_prompt: true

resources:
  memory_mb: 3072
  gpu_required: false
  gpu_recommended: true
  load_time_seconds: 8

engine_config:
  model_size: "large-v3"
  compute_type: "float16"
  device: "auto"
  beam_size: 5
  vad_filter: false
```
