---
title: Welcome to Macaw OpenVoice
sidebar_label: Welcome
sidebar_position: 1
---

# Welcome to Macaw OpenVoice

Macaw OpenVoice is an open-source **voice runtime** real-time Speech-to-Text and Text-to-Speech with OpenAI-compatible API, streaming session control, and extensible execution architecture.

> Macaw is **not** a fork, wrapper, or thin layer on top of existing projects. It is the **runtime layer** that sits between inference engines and production -- handling session management, audio preprocessing, post-processing, scheduling, observability, and a unified CLI.

---

## Capabilities

| Capability | Description |
|---|---|
| **OpenAI-Compatible API** | `POST /v1/audio/transcriptions`, `/translations`, `/speech` -- existing SDKs work out of the box |
| **Real-Time Streaming** | Partial and final transcripts via WebSocket with sub-300ms TTFB |
| **Full-Duplex** | Simultaneous STT + TTS on a single WebSocket with mute-on-speak safety |
| **Multi-Engine** | Faster-Whisper (encoder-decoder), WeNet (CTC), Kokoro (TTS) through one interface |
| **Session Manager** | 6-state machine, ring buffer, WAL-based crash recovery, backpressure control |
| **Voice Activity Detection** | Silero VAD with energy pre-filter and configurable sensitivity levels |
| **Audio Preprocessing** | Automatic resample, DC removal, and gain normalization to 16 kHz |
| **Post-Processing** | Inverse Text Normalization via NeMo (e.g., "two thousand" becomes "2000") |
| **Hot Words** | Domain-specific keyword boosting per session |
| **CLI** | Ollama-style UX -- `macaw serve`, `macaw transcribe`, `macaw list`, `macaw pull` |
| **Observability** | Prometheus metrics for TTFB, session duration, VAD events, TTS latency |

---

## Supported Engines

| Engine | Type | Architecture | Streaming | Hot Words |
|---|---|---|---|---|
| [Faster-Whisper](https://github.com/SYSTRAN/faster-whisper) | STT | Encoder-Decoder | LocalAgreement | via `initial_prompt` |
| [WeNet](https://github.com/wenet-e2e/wenet) | STT | CTC | Native partials | Native keyword boosting |
| [Kokoro](https://github.com/hexgrad/kokoro) | TTS | Neural | Chunked streaming | -- |

:::tip Adding new engines
Adding a new STT or TTS engine requires approximately 400-700 lines of code and **zero changes to the runtime core**. See the [Adding an Engine](guides/adding-engine) guide.
:::

---

## How It Works

```
              Clients (REST / WebSocket / CLI)
                          |
              +-----------+-----------+
              |     API Server        |
              |  (FastAPI + Uvicorn)  |
              +-----------+-----------+
                          |
              +-----------+-----------+
              |      Scheduler        |
              |  Priority . Batching  |
              |  Cancellation . TTFB  |
              +-----+----------+------+
                    |          |
           +--------+--+  +---+--------+
           | STT Worker |  | TTS Worker |
           |  (gRPC)    |  |  (gRPC)    |
           +------------+  +------------+
           | Faster-    |  | Kokoro     |
           | Whisper    |  +------------+
           | WeNet      |
           +------------+
```

Workers run as **isolated gRPC subprocesses**. If a worker crashes, the runtime recovers automatically via the WAL -- no data is lost, no segments are duplicated.

---

## Quick Example

```bash title="Install and start"
pip install macaw-openvoice[server,grpc,faster-whisper]
macaw serve
```

```bash title="Transcribe a file"
curl -X POST http://localhost:8000/v1/audio/transcriptions \
  -F file=@audio.wav \
  -F model=faster-whisper-large-v3
```

```python title="Using the OpenAI SDK"
from openai import OpenAI

client = OpenAI(base_url="http://localhost:8000/v1", api_key="not-needed")

result = client.audio.transcriptions.create(
    model="faster-whisper-large-v3",
    file=open("audio.wav", "rb"),
)
print(result.text)
```

---

## Next Steps

- **[Installation](getting-started/installation)** -- Set up Python, install Macaw, and configure your first engine
- **[Quickstart](getting-started/quickstart)** -- Run your first transcription in under 5 minutes
- **[Streaming STT](guides/streaming-stt)** -- Connect via WebSocket for real-time transcription
- **[Full-Duplex](guides/full-duplex)** -- Build voice assistants with simultaneous STT and TTS
- **[API Reference](api-reference/rest-api)** -- Complete endpoint documentation
- **[Architecture](architecture/overview)** -- Understand how the runtime is structured

---

## Contact

- **Website:** [usemacaw.io](https://usemacaw.io)
- **Email:** [hello@usemacaw.io](mailto:hello@usemacaw.io)
