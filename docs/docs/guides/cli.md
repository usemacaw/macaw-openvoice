---
title: CLI Reference
sidebar_position: 5
---

# CLI Reference

Macaw ships with an Ollama-style CLI for managing models, running the server, and transcribing audio. All commands are available via the `macaw` binary.

```bash
macaw --help
```

## Commands

### `macaw serve`

Start the API server and gRPC workers.

```bash title="Basic usage"
macaw serve
```

```bash title="Custom host and port"
macaw serve --host 0.0.0.0 --port 9000
```

| Flag | Default | Description |
|------|---------|-------------|
| `--host` | `127.0.0.1` | Bind address |
| `--port` | `8000` | HTTP port |
| `--models-dir` | `~/.macaw/models` | Models directory |
| `--cors-origins` | `*` | Allowed CORS origins |
| `--log-format` | `text` | Log format (`text` or `json`) |
| `--log-level` | `info` | Log level (`debug`, `info`, `warning`, `error`) |

The server starts:
1. FastAPI HTTP server on the specified port
2. STT gRPC worker on port 50051
3. TTS gRPC worker on port 50052

:::tip Production
For production deployments, use structured logging and bind to all interfaces:
```bash
macaw serve --host 0.0.0.0 --log-format json --log-level info
```
:::

---

### `macaw transcribe`

Transcribe an audio file or stream from the microphone.

```bash title="Transcribe a file"
macaw transcribe recording.wav --model faster-whisper-large-v3
```

```bash title="Stream from microphone"
macaw transcribe --stream --model faster-whisper-large-v3
```

| Flag | Short | Default | Description |
|------|:---:|---------|-------------|
| `--model` | `-m` | — | Model to use (required) |
| `--format` | | `json` | Output format (`json`, `text`, `verbose_json`, `srt`, `vtt`) |
| `--language` | `-l` | auto | Language code (e.g., `en`, `pt`) |
| `--no-itn` | | — | Disable Inverse Text Normalization |
| `--hot-words` | | — | Comma-separated hot words |
| `--stream` | | — | Stream from microphone instead of file |
| `--server` | | `http://localhost:8000` | Server URL (connects to running server) |

**File mode** sends the audio to the REST API for batch transcription. **Stream mode** connects to the WebSocket endpoint for real-time transcription.

```bash title="With hot words and language"
macaw transcribe call.wav -m faster-whisper-large-v3 -l pt --hot-words "Macaw,OpenVoice"
```

```bash title="Output as subtitles"
macaw transcribe video.wav -m faster-whisper-large-v3 --format srt
```

---

### `macaw translate`

Translate audio to English.

```bash title="Translate Portuguese audio"
macaw translate reuniao.wav --model faster-whisper-large-v3
```

| Flag | Short | Default | Description |
|------|:---:|---------|-------------|
| `--model` | `-m` | — | Model to use (required) |
| `--format` | | `json` | Output format |
| `--no-itn` | | — | Disable ITN |
| `--hot-words` | | — | Comma-separated hot words |
| `--server` | | `http://localhost:8000` | Server URL |

:::info English output only
Translation always produces English text, regardless of the source language. This matches the OpenAI API behavior.
:::

---

### `macaw list`

List installed models.

```bash
macaw list
```

```
NAME                        TYPE   ENGINE            SIZE
faster-whisper-large-v3     stt    faster-whisper    3.1 GB
kokoro                      tts    kokoro            982 MB
wenet-chinese               stt    wenet            1.2 GB
```

| Flag | Default | Description |
|------|---------|-------------|
| `--models-dir` | `~/.macaw/models` | Models directory to scan |

---

### `macaw inspect`

Show detailed information about a model.

```bash
macaw inspect faster-whisper-large-v3
```

```
Name:           faster-whisper-large-v3
Type:           stt
Engine:         faster-whisper
Architecture:   encoder-decoder
Size:           3.1 GB

Capabilities:
  Hot words:          false (via initial_prompt workaround)
  Initial prompt:     true
  Batch:              true
  Word timestamps:    true
  Max concurrent:     1

Engine Config:
  beam_size:          5
  vad_filter:         false
  compute_type:       float16
  device:             cuda
```

| Flag | Default | Description |
|------|---------|-------------|
| `--models-dir` | `~/.macaw/models` | Models directory |

---

### `macaw ps`

List models loaded on a running server.

```bash
macaw ps
```

```
NAME                        TYPE   ENGINE            STATUS
faster-whisper-large-v3     stt    faster-whisper    ready
kokoro                      tts    kokoro            ready
```

| Flag | Default | Description |
|------|---------|-------------|
| `--server` | `http://localhost:8000` | Server URL to query |

This queries the `GET /v1/models` endpoint on the running server.

---

### `macaw pull`

Download a model from HuggingFace Hub.

```bash title="Download a model"
macaw pull faster-whisper-large-v3
```

```bash title="Force re-download"
macaw pull faster-whisper-large-v3 --force
```

| Flag | Default | Description |
|------|---------|-------------|
| `--models-dir` | `~/.macaw/models` | Download destination |
| `--force` | — | Re-download even if already exists |

---

### `macaw remove`

Remove an installed model.

```bash title="Remove with confirmation"
macaw remove faster-whisper-large-v3
```

```bash title="Skip confirmation"
macaw remove faster-whisper-large-v3 --yes
```

| Flag | Short | Default | Description |
|------|:---:|---------|-------------|
| `--models-dir` | | `~/.macaw/models` | Models directory |
| `--yes` | `-y` | — | Skip confirmation prompt |

## Typical Workflow

```bash title="1. Install a model"
macaw pull faster-whisper-large-v3

# 2. Verify it's installed
macaw list

# 3. Start the server
macaw serve

# 4. In another terminal — check loaded models
macaw ps

# 5. Transcribe a file
macaw transcribe audio.wav -m faster-whisper-large-v3

# 6. Stream from microphone
macaw transcribe --stream -m faster-whisper-large-v3

# 7. Translate foreign audio
macaw translate foreign.wav -m faster-whisper-large-v3
```

## Environment Variables

CLI commands respect these environment variables:

| Variable | Default | Description |
|----------|---------|-------------|
| `MACAW_MODELS_DIR` | `~/.macaw/models` | Default models directory |
| `MACAW_SERVER_URL` | `http://localhost:8000` | Default server URL for client commands |
| `MACAW_LOG_LEVEL` | `info` | Default log level |
| `MACAW_LOG_FORMAT` | `text` | Default log format |
