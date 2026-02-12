---
title: Supported Models
sidebar_position: 1
---

# Supported Models

Macaw OpenVoice is **engine-agnostic** — it supports multiple STT and TTS engines through a unified backend interface. Each engine runs as an isolated gRPC subprocess, and the runtime adapts its pipeline automatically based on the model's architecture.

## Model Catalog

These are the official models available via `macaw pull`:

### STT Models

| Model | Engine | Architecture | Memory | GPU | Languages | Translation |
|-------|--------|:---:|:---:|:---:|-----------|:---:|
| [`faster-whisper-large-v3`](/docs/models/faster-whisper#large-v3) | Faster-Whisper | encoder-decoder | 3,072 MB | Recommended | 100+ (auto-detect) | Yes |
| [`faster-whisper-medium`](/docs/models/faster-whisper#medium) | Faster-Whisper | encoder-decoder | 1,536 MB | Recommended | 100+ (auto-detect) | Yes |
| [`faster-whisper-small`](/docs/models/faster-whisper#small) | Faster-Whisper | encoder-decoder | 512 MB | Optional | 100+ (auto-detect) | Yes |
| [`faster-whisper-tiny`](/docs/models/faster-whisper#tiny) | Faster-Whisper | encoder-decoder | 256 MB | Optional | 100+ (auto-detect) | Yes |
| [`distil-whisper-large-v3`](/docs/models/faster-whisper#distil-large-v3) | Faster-Whisper | encoder-decoder | 1,536 MB | Recommended | English only | No |

### TTS Models

| Model | Engine | Memory | GPU | Languages | Default Voice |
|-------|--------|:---:|:---:|-----------|---------------|
| [`kokoro-v1`](/docs/models/kokoro) | Kokoro | 512 MB | Recommended | 9 languages | `af_heart` |

### VAD (Internal)

| Model | Purpose | Memory | GPU | Cost |
|-------|---------|:---:|:---:|:---:|
| [Silero VAD](/docs/models/silero-vad) | Voice Activity Detection | ~50 MB | Not needed | ~2ms/frame |

:::info WeNet — bring your own model
[WeNet](/docs/models/wenet) is a supported engine but has no pre-configured models in the catalog. You provide your own WeNet model and create a `macaw.yaml` manifest for it.
:::

## Quick Install

```bash title="Install a model from the catalog"
macaw pull faster-whisper-large-v3
```

```bash title="List installed models"
macaw list
```

```bash title="Inspect model details"
macaw inspect faster-whisper-large-v3
```

```bash title="Remove a model"
macaw remove faster-whisper-large-v3
```

Models are downloaded from HuggingFace Hub and stored in `~/.macaw/models/` by default.

## Engine Comparison

### STT Engines

| Feature | Faster-Whisper | WeNet |
|---------|:-:|:-:|
| Architecture | Encoder-decoder | CTC |
| Streaming partials | Via LocalAgreement | Native |
| Hot words | Via `initial_prompt` workaround | Native keyword boosting |
| Cross-segment context | Yes (224 tokens) | No |
| Language detection | Yes | No |
| Translation | Yes (to English) | No |
| Word timestamps | Yes | Yes |
| Batch inference | Yes | Yes |
| Best for | Accuracy, multilingual | Low latency, Chinese |

### How Architecture Affects the Pipeline

The `architecture` field in the model manifest tells the runtime how to adapt its streaming pipeline:

| | Encoder-Decoder | CTC | Streaming-Native |
|---|:-:|:-:|:-:|
| **LocalAgreement** | Yes — confirms tokens across multiple inference passes | No | No |
| **Cross-segment context** | Yes — 224 tokens from previous final as `initial_prompt` | No | No |
| **Native partials** | No — runtime generates partials via LocalAgreement | Yes | Yes |
| **Accumulation** | 5s chunks before inference | Frame-by-frame (160ms minimum) | Frame-by-frame |
| **Example** | Faster-Whisper | WeNet | Paraformer (future) |

:::tip Choosing a model
- **Best accuracy**: `faster-whisper-large-v3` — highest quality, 100+ languages
- **Best speed/accuracy trade-off**: `faster-whisper-small` — runs on CPU, good quality
- **Fastest startup**: `faster-whisper-tiny` — 256 MB, loads in ~2s
- **English only, fast**: `distil-whisper-large-v3` — 6x faster than large-v3, ~1% WER gap
- **Low-latency streaming**: WeNet (CTC) — frame-by-frame native partials
- **Chinese focus**: WeNet — optimized for Chinese with native hot word support
:::

## Model Manifest

Every model has a `macaw.yaml` manifest that describes its capabilities, resource requirements, and engine configuration. See [Configuration](/docs/getting-started/configuration) for the full manifest format.

```yaml title="Example: macaw.yaml"
name: faster-whisper-large-v3
version: "1.0.0"
type: stt
engine: faster-whisper

capabilities:
  architecture: encoder-decoder
  streaming: true
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

## Dependencies

Each engine has its own optional dependency group. Install only what you need:

| Extra | Command | What It Installs |
|-------|---------|-----------------|
| `faster-whisper` | `pip install macaw-openvoice[faster-whisper]` | `faster-whisper>=1.1,<2.0` |
| `wenet` | `pip install macaw-openvoice[wenet]` | `wenet>=2.0,<3.0` |
| `kokoro` | `pip install macaw-openvoice[kokoro]` | `kokoro>=0.1,<1.0` |
| `huggingface` | `pip install macaw-openvoice[huggingface]` | `huggingface_hub>=0.20,<1.0` |
| `itn` | `pip install macaw-openvoice[itn]` | `nemo_text_processing>=1.1,<2.0` |

```bash title="Install everything for a typical deployment"
pip install macaw-openvoice[server,grpc,faster-whisper,kokoro,huggingface]
```

## Adding Your Own Engine

Macaw is designed to make adding new engines straightforward — approximately 400-700 lines of code with zero changes to the runtime core. See the [Adding an Engine](/docs/guides/adding-engine) guide.
