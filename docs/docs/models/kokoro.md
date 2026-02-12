---
title: Kokoro TTS
sidebar_position: 4
---

# Kokoro TTS

Kokoro is a lightweight neural text-to-speech engine with 82 million parameters. It supports 9 languages, multiple voices, and produces high-quality 24kHz audio. Kokoro is the default TTS engine in Macaw OpenVoice, used for full-duplex voice interactions via WebSocket and REST speech synthesis.

## Installation

```bash
pip install macaw-openvoice[kokoro]
```

This installs `kokoro>=0.1,<1.0` as an optional dependency.

## Model

| Property | Value |
|----------|-------|
| Catalog name | `kokoro-v1` |
| HuggingFace repo | `hexgrad/Kokoro-82M` |
| Parameters | 82M |
| Memory | 512 MB |
| GPU | Recommended |
| Load time | ~3 seconds |
| Output sample rate | 24,000 Hz |
| Output format | 16-bit PCM |
| Chunk size | 4,096 bytes (~85ms at 24kHz) |
| API version | Kokoro v0.9.4 |

```bash
macaw pull kokoro-v1
```

## Languages

Kokoro supports 9 languages, identified by a single-character prefix in the voice name:

| Prefix | Language | Example Voice |
|:------:|----------|---------------|
| `a` | English (American) | `af_heart` |
| `b` | English (British) | `bf_emma` |
| `e` | Spanish | `ef_dora` |
| `f` | French | `ff_siwis` |
| `h` | Hindi | `hf_alpha` |
| `i` | Italian | `if_sara` |
| `j` | Japanese | `jf_alpha` |
| `p` | Portuguese | `pf_dora` |
| `z` | Chinese | `zf_xiaobei` |

The language is selected automatically based on the voice prefix. When loading the model, the `lang_code` in `engine_config` sets the default pipeline language.

## Voices

### Naming Convention

Kokoro voices follow a strict naming pattern:

```
[language][gender][_name]
```

| Position | Meaning | Values |
|----------|---------|--------|
| 1st character | Language prefix | `a`, `b`, `e`, `f`, `h`, `i`, `j`, `p`, `z` |
| 2nd character | Gender | `f` = female, `m` = male |
| Rest | Voice name | Unique identifier (e.g., `_heart`, `_emma`) |

**Examples:**
- `af_heart` — American English, female, "heart"
- `bm_george` — British English, male, "george"
- `pf_dora` — Portuguese, female, "dora"
- `jf_alpha` — Japanese, female, "alpha"

### Default Voice

The default voice is **`af_heart`** (American English, female). This is used when:
- `voice` is set to `"default"` in the API request
- No voice is specified

### Voice Resolution

When a voice is requested, Kokoro resolves it in this order:

1. `"default"` → uses the configured `default_voice` (default: `af_heart`)
2. Simple name (e.g., `"af_heart"`) → looks for `<voices_dir>/af_heart.pt`
3. Absolute path or `.pt` extension → uses as-is

Voice files are `.pt` (PyTorch) files stored in the `voices/` subdirectory of the model path.

### Voice Discovery

The backend scans `<model_path>/voices/*.pt` to discover available voices. Each `.pt` file becomes a selectable voice:

```bash title="List available voices"
ls ~/.macaw/models/kokoro-v1/voices/
# af_heart.pt  af_sky.pt  am_adam.pt  bf_emma.pt  ...
```

## Usage Examples

### REST API — Speech Synthesis

```bash title="Generate speech (WAV)"
curl -X POST http://localhost:8000/v1/audio/speech \
  -H "Content-Type: application/json" \
  -d '{
    "model": "kokoro-v1",
    "input": "Hello! Welcome to Macaw OpenVoice.",
    "voice": "af_heart"
  }' \
  --output speech.wav
```

```bash title="Generate speech (raw PCM)"
curl -X POST http://localhost:8000/v1/audio/speech \
  -H "Content-Type: application/json" \
  -d '{
    "model": "kokoro-v1",
    "input": "Olá! Bem-vindo ao Macaw OpenVoice.",
    "voice": "pf_dora",
    "response_format": "pcm"
  }' \
  --output speech.raw
```

```bash title="Adjust speed"
curl -X POST http://localhost:8000/v1/audio/speech \
  -H "Content-Type: application/json" \
  -d '{
    "model": "kokoro-v1",
    "input": "Speaking slowly for clarity.",
    "voice": "af_heart",
    "speed": 0.8
  }' \
  --output slow.wav
```

### Full-Duplex (WebSocket)

In a full-duplex WebSocket session, TTS is triggered via the `tts.speak` command:

```python title="Full-duplex voice assistant"
import asyncio
import json
import websockets

async def voice_assistant():
    uri = "ws://localhost:8000/v1/realtime"
    async with websockets.connect(uri) as ws:
        # Configure session with STT + TTS models
        await ws.send(json.dumps({
            "type": "session.configure",
            "model": "faster-whisper-large-v3",
            "tts_model": "kokoro-v1",
            "tts_voice": "af_heart"
        }))

        # When user says something and you get a final transcript...
        # Send TTS response:
        await ws.send(json.dumps({
            "type": "tts.speak",
            "text": "I heard you! Let me help with that."
        }))

        # Receive TTS audio as binary frames
        while True:
            msg = await ws.recv()
            if isinstance(msg, bytes):
                # Binary frame = TTS audio (16-bit PCM, 24kHz)
                play_audio(msg)
            else:
                event = json.loads(msg)
                if event["type"] == "tts.speaking_start":
                    print("TTS started")
                elif event["type"] == "tts.speaking_end":
                    print("TTS finished")
                    break
```

:::info Mute-on-speak
During TTS playback, the STT pipeline automatically mutes (discards incoming audio frames). This prevents the system from transcribing its own speech output. Unmute is guaranteed via `try/finally` — even if TTS crashes, the microphone is restored.
:::

### Python SDK

```python title="Direct synthesis"
import httpx

async def synthesize():
    async with httpx.AsyncClient() as client:
        response = await client.post(
            "http://localhost:8000/v1/audio/speech",
            json={
                "model": "kokoro-v1",
                "input": "Macaw OpenVoice converts text to natural speech.",
                "voice": "af_heart",
                "speed": 1.0,
            },
        )
        with open("output.wav", "wb") as f:
            f.write(response.content)
```

## Engine Configuration

```yaml title="engine_config in macaw.yaml"
engine_config:
  device: "auto"           # "auto" (→ cpu), "cpu", "cuda"
  default_voice: "af_heart" # Default voice when "default" is requested
  sample_rate: 24000        # Output sample rate (fixed at 24kHz)
  lang_code: "a"            # Default pipeline language prefix
```

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `device` | string | `"auto"` | Inference device. `"auto"` maps to `"cpu"` |
| `default_voice` | string | `"af_heart"` | Voice used when `"default"` is requested |
| `sample_rate` | int | `24000` | Output sample rate (always 24kHz) |
| `lang_code` | string | `"a"` | Default language prefix for the KPipeline |

## Streaming Behavior

Kokoro synthesizes text in a streaming fashion using `AsyncIterator[bytes]`:

1. The full text is passed to Kokoro's `KPipeline`
2. KPipeline processes text in segments, yielding `(graphemes, phonemes, audio)` tuples
3. Audio arrays are concatenated and converted to 16-bit PCM
4. The PCM data is chunked into **4,096-byte segments** (~85ms of audio at 24kHz)
5. Chunks are yielded one at a time for low time-to-first-byte (TTFB)

```
Text → KPipeline → [segment audio] → float32→PCM16 → 4096-byte chunks → Client
```

### Speed Control

The `speed` parameter adjusts synthesis speed:

| Value | Effect |
|-------|--------|
| `0.25` | 4x slower (minimum) |
| `1.0` | Normal speed (default) |
| `2.0` | 2x faster |
| `4.0` | 4x faster (maximum) |

## Response Formats

The `POST /v1/audio/speech` endpoint supports two output formats:

| Format | Content-Type | Description |
|--------|-------------|-------------|
| `wav` (default) | `audio/wav` | WAV file with header (44 bytes header + PCM data) |
| `pcm` | `audio/pcm` | Raw 16-bit PCM, little-endian, mono, 24kHz |

## Manifest Reference

```yaml title="macaw.yaml (kokoro-v1)"
name: kokoro-v1
version: "1.0.0"
engine: kokoro
type: tts
description: "Kokoro TTS - neural text-to-speech"

capabilities:
  streaming: true
  languages: ["en", "pt", "ja"]

resources:
  memory_mb: 512
  gpu_required: false
  gpu_recommended: true
  load_time_seconds: 3

engine_config:
  device: "auto"
  default_voice: "af_heart"
  sample_rate: 24000
  lang_code: "a"
```

## Key Behaviors

- **TTS is stateless per request.** Each `tts.speak` or `POST /v1/audio/speech` is independent. The Session Manager is not used for TTS.
- **New `tts.speak` cancels the previous.** If a `tts.speak` command arrives while another is in progress, the previous one is cancelled automatically.
- **Binary WebSocket frames are directional.** Server→client binary frames are always TTS audio. Client→server binary frames are always STT audio. No ambiguity.
- **TTS worker is a separate subprocess.** Runs on a different gRPC port (default 50052 vs 50051 for STT). Crash does not affect the runtime.
