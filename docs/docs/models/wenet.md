---
title: WeNet
sidebar_position: 3
---

# WeNet

WeNet is a CTC-based STT engine optimized for low-latency streaming and Chinese speech recognition. Unlike Faster-Whisper, WeNet produces **native partial transcripts** frame-by-frame without requiring LocalAgreement, making it ideal for real-time applications where latency is critical.

:::info Bring your own model
WeNet has **no pre-configured models** in the Macaw catalog. You provide your own WeNet model and create a `macaw.yaml` manifest for it. See [Creating a Manifest](#creating-a-manifest) below.
:::

## Installation

```bash
pip install macaw-openvoice[wenet]
```

This installs `wenet>=2.0,<3.0` as an optional dependency.

## Architecture

WeNet uses the **CTC (Connectionist Temporal Classification)** architecture. This means the runtime adapts the streaming pipeline with:

- **No LocalAgreement** — CTC produces native partial transcripts directly
- **No cross-segment context** — CTC does not support `initial_prompt` conditioning
- **No accumulation** — each chunk is processed immediately (frame-by-frame, minimum 160ms)

```
Audio → [immediate processing] → Native CTC partials → Final
           (160ms minimum)
```

### Faster-Whisper vs. WeNet Streaming

| Behavior | Faster-Whisper (Encoder-Decoder) | WeNet (CTC) |
|----------|:-:|:-:|
| Audio buffering | 5s accumulation | Frame-by-frame (160ms min) |
| Partial generation | Via LocalAgreement | Native |
| Cross-segment context | 224 tokens via initial_prompt | Not supported |
| First partial latency | ~5 seconds | ~160 milliseconds |
| Best for | Accuracy | Low latency |

## Capabilities

| Capability | Supported | Notes |
|------------|:---------:|-------|
| Streaming | Yes | Native frame-by-frame partials |
| Batch inference | Yes | Via `POST /v1/audio/transcriptions` |
| Word timestamps | Yes | From token-level output |
| Language detection | No | Language is fixed per model |
| Translation | No | |
| Initial prompt | No | CTC does not support conditioning |
| Hot words | Yes | Native keyword boosting via context biasing |
| Partial transcripts | Yes | Native CTC partials |

### Native Hot Words

WeNet supports native keyword boosting (context biasing), unlike Faster-Whisper which uses an `initial_prompt` workaround. This makes hot word recognition more reliable for domain-specific vocabulary:

```json title="WebSocket session.configure"
{
  "type": "session.configure",
  "model": "my-wenet-model",
  "hot_words": ["CPF", "CNPJ", "PIX"]
}
```

### Language Handling

| Input | Behavior |
|-------|----------|
| `"auto"` | Falls back to `"zh"` (Chinese) |
| `"mixed"` | Falls back to `"zh"` (Chinese) |
| `"zh"`, `"en"`, etc. | Uses the specified language |
| Omitted | Falls back to `"zh"` |

WeNet models are typically trained for a specific language (most commonly Chinese). The `language` parameter is informational — the model always uses the language it was trained for.

### Device Handling

| Input | Behavior |
|-------|----------|
| `"auto"` | Maps to `"cpu"` |
| `"cpu"` | CPU inference |
| `"cuda"` | GPU inference |
| `"cuda:0"` | Specific GPU |

:::tip
Unlike Faster-Whisper where `"auto"` selects GPU if available, WeNet's `"auto"` always maps to `"cpu"`. Explicitly set `device: "cuda"` if you want GPU inference.
:::

## Creating a Manifest

Since WeNet has no catalog entries, you must create a `macaw.yaml` manifest manually in your model directory:

```yaml title="~/.macaw/models/my-wenet-model/macaw.yaml"
name: my-wenet-model
version: "1.0.0"
engine: wenet
type: stt
description: "Custom WeNet CTC model for Mandarin"

capabilities:
  streaming: true
  architecture: ctc
  languages: ["zh"]
  word_timestamps: true
  translation: false
  partial_transcripts: true
  hot_words: true
  batch_inference: true
  language_detection: false
  initial_prompt: false

resources:
  memory_mb: 512
  gpu_required: false
  gpu_recommended: false
  load_time_seconds: 3

engine_config:
  language: "chinese"
  device: "cpu"
```

### Manifest Fields for WeNet

| Field | Required | Description |
|-------|:--------:|-------------|
| `capabilities.architecture` | Yes | Must be `ctc` |
| `capabilities.hot_words` | Yes | Set to `true` — WeNet supports native hot words |
| `capabilities.initial_prompt` | Yes | Must be `false` — CTC does not support conditioning |
| `capabilities.translation` | Yes | Must be `false` — WeNet does not translate |
| `capabilities.language_detection` | Yes | Must be `false` — WeNet does not auto-detect language |
| `engine_config.language` | No | Default language for the model (default: `"chinese"`) |
| `engine_config.device` | No | Inference device (default: `"cpu"`) |

## Setting Up a WeNet Model

1. **Download or train a WeNet model** — obtain a model directory with the required files (model weights, config, etc.)

2. **Create the model directory:**
   ```bash
   mkdir -p ~/.macaw/models/my-wenet-model
   ```

3. **Copy model files** into the directory

4. **Create the manifest:**
   ```bash title="Create macaw.yaml"
   cat > ~/.macaw/models/my-wenet-model/macaw.yaml << 'EOF'
   name: my-wenet-model
   version: "1.0.0"
   engine: wenet
   type: stt
   description: "Custom WeNet model"
   capabilities:
     streaming: true
     architecture: ctc
     languages: ["zh"]
     word_timestamps: true
     translation: false
     partial_transcripts: true
     hot_words: true
     batch_inference: true
     language_detection: false
     initial_prompt: false
   resources:
     memory_mb: 512
     gpu_required: false
     gpu_recommended: false
     load_time_seconds: 3
   engine_config:
     language: "chinese"
     device: "cpu"
   EOF
   ```

5. **Verify the model is detected:**
   ```bash
   macaw list
   # Should show: my-wenet-model  wenet  stt  ctc
   ```

6. **Test transcription:**
   ```bash
   macaw transcribe audio_zh.wav --model my-wenet-model
   ```

## Usage Examples

### Batch Transcription

```bash title="Transcribe a Chinese audio file"
curl -X POST http://localhost:8000/v1/audio/transcriptions \
  -F "file=@audio_zh.wav" \
  -F "model=my-wenet-model"
```

### Streaming (WebSocket)

```python title="Low-latency streaming with WeNet"
import asyncio
import json
import websockets

async def stream_low_latency():
    uri = "ws://localhost:8000/v1/realtime"
    async with websockets.connect(uri) as ws:
        # Configure with WeNet model and hot words
        await ws.send(json.dumps({
            "type": "session.configure",
            "model": "my-wenet-model",
            "hot_words": ["CPF", "CNPJ", "PIX"]
        }))

        # Stream audio — partials arrive within ~160ms
        with open("audio.raw", "rb") as f:
            while chunk := f.read(3200):  # 100ms chunks
                await ws.send(chunk)
                
                try:
                    msg = await asyncio.wait_for(ws.recv(), timeout=0.05)
                    event = json.loads(msg)
                    if event["type"] == "transcript.partial":
                        print(f"  ...{event['text']}")
                    elif event["type"] == "transcript.final":
                        print(f"  >> {event['text']}")
                except asyncio.TimeoutError:
                    pass

asyncio.run(stream_low_latency())
```

## Engine Configuration Reference

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `language` | string | `"chinese"` | Model language (informational) |
| `device` | string | `"cpu"` | Inference device (`"cpu"`, `"cuda"`, `"auto"` → `"cpu"`) |

## When to Choose WeNet

**Choose WeNet when:**
- You need the lowest possible latency for streaming (partials in ~160ms vs ~5s for Faster-Whisper)
- Your application is Chinese-focused
- You need reliable native hot word support for domain-specific vocabulary
- You have your own trained WeNet model

**Choose Faster-Whisper instead when:**
- You need multilingual support (100+ languages)
- You need translation capabilities
- You want ready-to-use catalog models (no manual setup)
- Accuracy is more important than latency
