---
title: Installation
sidebar_position: 1
---

# Installation

Macaw OpenVoice requires **Python 3.11+** and uses pip extras to install only the engines you need.

---

## Prerequisites

| Requirement | Minimum | Recommended |
|---|---|---|
| Python | 3.11 | 3.12 |
| pip | 21.0+ | latest |
| OS | Linux, macOS | Linux (for GPU support) |
| CUDA | Optional | 12.x (for GPU inference) |

:::info
Macaw runs on CPU by default. GPU support depends on the engine -- Faster-Whisper uses CTranslate2 which supports CUDA out of the box.
:::

---

## Install with pip

The simplest way to get started:

```bash title="Minimal install (STT only)"
pip install macaw-openvoice[server,grpc,faster-whisper]
```

```bash title="Full install (STT + TTS + ITN)"
pip install macaw-openvoice[server,grpc,faster-whisper,kokoro,itn]
```

### Available Extras

| Extra | What it adds | Size |
|---|---|---|
| `server` | FastAPI + Uvicorn (required for serving) | ~20 MB |
| `grpc` | gRPC runtime for worker communication | ~15 MB |
| `faster-whisper` | Faster-Whisper STT engine | ~100 MB |
| `wenet` | WeNet CTC STT engine | ~80 MB |
| `kokoro` | Kokoro TTS engine | ~50 MB |
| `itn` | NeMo Inverse Text Normalization | ~200 MB |
| `stream` | Microphone streaming via sounddevice | ~5 MB |
| `dev` | Development tools (ruff, mypy, pytest) | ~50 MB |

---

## Install with uv (recommended for development)

[uv](https://github.com/astral-sh/uv) is significantly faster than pip and handles virtual environments automatically:

```bash title="Create a virtual environment and install"
uv venv --python 3.12
uv sync --all-extras
```

```bash title="Activate the environment"
source .venv/bin/activate
```

---

## GPU Setup

For GPU-accelerated inference with Faster-Whisper:

1. Install CUDA drivers for your GPU
2. Install the CUDA-enabled version of CTranslate2:

```bash
pip install ctranslate2
```

:::warning
Ensure your CUDA version matches the CTranslate2 build. Check compatibility at the [CTranslate2 releases page](https://github.com/OpenNMT/CTranslate2/releases).
:::

---

## Verify Installation

```bash title="Check that Macaw is installed correctly"
macaw --help
```

You should see:

```
Usage: macaw [OPTIONS] COMMAND [ARGS]...

  Macaw OpenVoice CLI

Commands:
  serve        Start the API server
  transcribe   Transcribe an audio file
  translate    Translate audio to English
  list         List installed models
  pull         Download a model
  inspect      Show model details
```

---

## Next Steps

- **[Quickstart](quickstart)** -- Run your first transcription
- **[Configuration](configuration)** -- Customize runtime settings
