---
title: Configuration
sidebar_position: 3
---

# Configuration

Macaw OpenVoice uses a combination of model manifests, runtime defaults, and environment variables for configuration.

---

## Model Manifests

Each engine model is described by a `macaw.yaml` manifest file. This file declares the model's capabilities and how the runtime should interact with it.

```yaml title="Example: macaw.yaml for Faster-Whisper"
name: faster-whisper-large-v3
type: stt
engine: faster-whisper
architecture: encoder-decoder
languages:
  - en
  - pt
  - es
options:
  beam_size: 5
  vad_filter: false      # VAD is handled by the runtime, not the engine
  word_timestamps: false
```

### Key Fields

| Field | Type | Description |
|---|---|---|
| `name` | string | Unique model identifier |
| `type` | string | `stt` or `tts` |
| `engine` | string | Engine backend (`faster-whisper`, `wenet`, `kokoro`) |
| `architecture` | string | `encoder-decoder`, `ctc`, or `streaming-native` |
| `languages` | list | Supported language codes |
| `options` | dict | Engine-specific configuration |

:::warning
Always set `vad_filter: false` in your manifest. The VAD is managed by the Macaw runtime -- enabling the engine's internal VAD would duplicate the work and cause unpredictable behavior.
:::

---

## Runtime Configuration

Runtime behavior is controlled through server startup options:

```bash title="Start with custom settings"
macaw serve --host 0.0.0.0 --port 8000
```

### Server Options

| Option | Default | Description |
|---|---|---|
| `--host` | `127.0.0.1` | Bind address |
| `--port` | `8000` | HTTP port |
| `--workers` | `1` | Uvicorn workers |

### Scheduler Settings

The scheduler manages request prioritization and batching:

| Setting | Default | Description |
|---|---|---|
| Aging timeout | `30.0s` | Max time a request waits in queue |
| Batch window | `75ms` | Time window to accumulate batch requests |
| Batch max size | `8` | Maximum requests per batch |

:::tip
Streaming WebSocket requests bypass the scheduler entirely -- they use a direct gRPC streaming connection for minimum latency.
:::

---

## Pipeline Configuration

### Preprocessing

The audio preprocessing pipeline runs **before** VAD and is not configurable per-request -- it ensures all audio reaches the VAD and engine in a consistent format:

1. **Resample** to 16 kHz mono
2. **DC removal** (high-pass filter)
3. **Gain normalization**

### VAD Settings

VAD can be configured per WebSocket session via the `session.configure` command:

```json title="WebSocket session configuration"
{
  "type": "session.configure",
  "vad": {
    "sensitivity": "normal"
  },
  "language": "en",
  "hot_words": ["Macaw", "OpenVoice"]
}
```

| VAD Setting | Options | Description |
|---|---|---|
| `sensitivity` | `high`, `normal`, `low` | Controls speech detection threshold |

### Post-Processing (ITN)

Inverse Text Normalization converts spoken numbers and patterns to their written form. It is applied **only to final transcripts**, never to partials.

| Input | Output |
|---|---|
| "two thousand twenty six" | "2026" |
| "ten dollars and fifty cents" | "$10.50" |
| "one two three four" | "1234" |

:::info
ITN requires the `itn` extra: `pip install macaw-openvoice[itn]`. If not installed, transcripts are returned as-is (fail-open behavior).
:::

---

## Environment Variables

| Variable | Description | Default |
|---|---|---|
| `MACAW_MODELS_DIR` | Directory for model files | `~/.macaw/models` |
| `MACAW_LOG_LEVEL` | Logging level | `INFO` |
| `MACAW_STT_PORT` | gRPC port for STT worker | `50051` |
| `MACAW_TTS_PORT` | gRPC port for TTS worker | `50052` |

---

## Next Steps

- **[Architecture Overview](../architecture/overview)** -- Understand the runtime design
- **[Adding an Engine](../guides/adding-engine)** -- Add custom STT or TTS engines
