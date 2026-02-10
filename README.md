<p align="center">
  <img src="docs/static/img/logo-256.png" alt="Macaw OpenVoice" width="128" />
</p>

<h1 align="center">Macaw OpenVoice</h1>

<p align="center">
  <strong>Voice runtime (STT + TTS) with OpenAI-compatible API</strong>
</p>

<p align="center">
  <a href="https://github.com/useMacaw/macaw-openvoice/releases"><img src="https://img.shields.io/badge/version-1.0.0-blue?style=flat-square" alt="Version 1.0.0"></a>
  <a href="LICENSE"><img src="https://img.shields.io/badge/license-Apache_2.0-blue?style=flat-square" alt="License"></a>
  <a href="https://www.python.org/downloads/"><img src="https://img.shields.io/badge/python-3.11%2B-blue?style=flat-square" alt="Python 3.11+"></a>
  <a href="https://github.com/useMacaw/macaw-openvoice/actions"><img src="https://img.shields.io/badge/tests-1686_passed-brightgreen?style=flat-square" alt="Tests"></a>
  <a href="https://pypi.org/project/macaw-openvoice/"><img src="https://img.shields.io/badge/pypi-macaw--openvoice-orange?style=flat-square" alt="PyPI"></a>
</p>

<p align="center">
  <a href="#quick-start">Quick Start</a> &middot;
  <a href="#features">Features</a> &middot;
  <a href="#architecture">Architecture</a> &middot;
  <a href="#api-compatibility">API Docs</a> &middot;
  <a href="#demo">Demo</a> &middot;
  <a href="https://macaw-voice.github.io/macaw-openvoice">Full Documentation</a>
</p>

---

Macaw OpenVoice is a **voice runtime built from scratch** in Python. It orchestrates inference engines (Faster-Whisper, WeNet, Silero VAD, Kokoro) within a single process and serves both STT and TTS through an OpenAI-compatible API and an Ollama-inspired CLI.

Macaw is **not** a fork, wrapper, or extension of existing projects. It is the **runtime layer** that sits between inference engines and production: session management, preprocessing, post-processing, scheduling, observability, and a unified CLI.


## Features

- **OpenAI-compatible API** — `POST /v1/audio/transcriptions`, `/translations`, `/speech`, `WS /v1/realtime`
- **Full-duplex STT + TTS** — simultaneous speech-to-text and text-to-speech on the same WebSocket
- **Real-time streaming** — partial and final transcripts via WebSocket with <300ms TTFB
- **Multi-engine** — Faster-Whisper (encoder-decoder), WeNet (CTC), Kokoro (TTS) with a single interface
- **Session Manager** — 6-state machine, ring buffer, WAL, crash recovery without segment duplication
- **Voice Activity Detection** — Silero VAD + energy pre-filter, sensitivity levels (high/normal/low)
- **Audio preprocessing** — resample, DC remove, gain normalize (any sample rate to 16kHz)
- **Post-processing** — Inverse Text Normalization via NeMo ("dois mil" to "2000")
- **Mute-on-speak** — STT pauses during TTS to prevent feedback loops
- **Hot words** — domain-specific keyword boosting per session
- **CLI** — `macaw serve`, `macaw transcribe`, `macaw translate`, `macaw list` (Ollama-style UX)
- **Observability** — Prometheus metrics for TTFB, session duration, VAD events, TTS latency

## Quick Start

```bash
# Install
pip install macaw-openvoice[server,grpc,faster-whisper]

# Pull a model
macaw pull faster-whisper-tiny

# Start the runtime
macaw serve
```

```
$ macaw serve
  ╔══════════════════════════════════════════════╗
  ║         Macaw OpenVoice v1.0.0              ║
  ╚══════════════════════════════════════════════╝

INFO     Scanning models in ~/.macaw/models
INFO     Found 2 model(s): faster-whisper-tiny (STT), kokoro-v1 (TTS)
INFO     Spawning STT worker   faster-whisper-tiny  port=50051  engine=faster-whisper
INFO     Spawning TTS worker   kokoro-v1            port=50052  engine=kokoro
INFO     Scheduler started     aging=30.0s  batch_ms=75.0  batch_max=8
INFO     Uvicorn running on http://127.0.0.1:8000
```

### Transcribe a file

```bash
# Via REST API
curl -X POST http://localhost:8000/v1/audio/transcriptions \
  -F file=@audio.wav \
  -F model=faster-whisper-tiny

# Via CLI
macaw transcribe audio.wav --model faster-whisper-tiny
```

### Streaming via WebSocket

```bash
wscat -c "ws://localhost:8000/v1/realtime?model=faster-whisper-tiny"
# Send binary audio frames, receive JSON transcript events
```

### Text-to-Speech

```bash
curl -X POST http://localhost:8000/v1/audio/speech \
  -H "Content-Type: application/json" \
  -d '{"model": "kokoro-v1", "input": "Hello, how can I help you?", "voice": "default"}' \
  --output speech.wav
```

## Architecture

```
                         Clients
          CLI / REST / WebSocket (full-duplex)
                           |
                           v
  +----------------------------------------------------+
  |              API Server (FastAPI)                    |
  |                                                    |
  |  POST /v1/audio/transcriptions    (STT batch)      |
  |  POST /v1/audio/translations      (STT translate)  |
  |  POST /v1/audio/speech            (TTS)            |
  |  WS   /v1/realtime                (STT+TTS)        |
  +----------------------------------------------------+
  |              Scheduler                              |
  |  Priority queue (realtime > batch), cancellation,   |
  |  dynamic batching, latency tracking                 |
  +----------------------------------------------------+
  |              Model Registry                         |
  |  Declarative manifest (macaw.yaml), lifecycle        |
  +----------+-------------------+---------------------+
             |                   |
    +--------+--------+  +------+-------+
    |  STT Workers    |  |  TTS Workers |
    |  (subprocess    |  |  (subprocess |
    |   gRPC)         |  |   gRPC)      |
    |                 |  |              |
    | Faster-Whisper  |  | Kokoro       |
    | WeNet           |  |              |
    +-----------------+  +--------------+
             |
  +----------+-------------------------------------+
  |  Audio Preprocessing Pipeline                   |
  |  Resample -> DC Remove -> Gain Normalize        |
  +------------------------------------------------+
  |  Session Manager (STT only)                     |
  |  6 states, ring buffer, WAL, LocalAgreement,    |
  |  cross-segment context, crash recovery          |
  +------------------------------------------------+
  |  VAD (Energy Pre-filter + Silero VAD)           |
  +------------------------------------------------+
  |  Post-Processing (ITN via NeMo)                 |
  +------------------------------------------------+
```

## Demo
<p align="center">
  <img src="docs/static/img/screen.png" alt="Macaw OpenVoice Demo" width="720" />
</p>


## Supported Models

| Engine | Type | Architecture | Partials | Hot Words | Status |
|--------|------|-------------|----------|-----------|--------|
| [Faster-Whisper](https://github.com/SYSTRAN/faster-whisper) | STT | encoder-decoder | LocalAgreement | via initial_prompt | Supported |
| [WeNet](https://github.com/wenet-e2e/wenet) | STT | CTC | native | native keyword boosting | Supported |
| [Kokoro](https://github.com/hexgrad/kokoro) | TTS | neural | — | — | Supported |

Adding a new engine requires ~400-700 lines of code and zero changes to the runtime core. See the [Adding an Engine](https://macaw-voice.github.io/macaw-openvoice/docs/guides/adding-engine) guide.

## API Compatibility

Macaw implements the [OpenAI Audio API](https://platform.openai.com/docs/api-reference/audio) contract, so existing SDKs work without modification:

```python
from openai import OpenAI

client = OpenAI(base_url="http://localhost:8000/v1", api_key="not-needed")

# Transcription
result = client.audio.transcriptions.create(
    model="faster-whisper-tiny",
    file=open("audio.wav", "rb"),
)
print(result.text)

# Text-to-Speech
response = client.audio.speech.create(
    model="kokoro-v1",
    input="Hello, how can I help you?",
    voice="default",
)
response.stream_to_file("output.wav")
```

## WebSocket Protocol

The `/v1/realtime` endpoint supports full-duplex STT + TTS:

```
Client -> Server:
  Binary frames     PCM 16-bit audio (any sample rate)
  session.configure  Configure VAD, language, hot words, TTS model
  tts.speak          Trigger text-to-speech synthesis
  tts.cancel         Cancel active TTS

Server -> Client:
  session.created     Session established
  vad.speech_start    Speech detected
  transcript.partial  Intermediate hypothesis
  transcript.final    Confirmed segment (with ITN)
  vad.speech_end      Speech ended
  tts.speaking_start  TTS started (STT muted)
  Binary frames       TTS audio output
  tts.speaking_end    TTS finished (STT unmuted)
  error               Error with recoverable flag
```

## CLI

```bash
macaw serve                                   # Start API server
macaw transcribe audio.wav                    # Transcribe file
macaw transcribe audio.wav --format srt       # Generate subtitles
macaw transcribe --stream                     # Stream from microphone
macaw translate audio.wav                     # Translate to English
macaw list                                    # List installed models
macaw pull faster-whisper-tiny                # Download a model
macaw inspect faster-whisper-tiny             # Model details
```

## Demo

An interactive demo with a React/Next.js frontend is included:

```bash
./demo/start.sh
```

This starts the FastAPI backend (port 9000) and the Next.js frontend (port 3000) together. The demo includes a dashboard for batch transcriptions, real-time streaming STT with VAD visualization, and a TTS playground. See [demo/README.md](demo/README.md) for details.

## Development

```bash
# Setup (requires Python 3.11+ and uv)
uv venv --python 3.12
uv sync --all-extras

# Development workflow
make check       # format + lint + typecheck
make test-unit   # unit tests (preferred during development)
make test        # all tests (1686 passing)
make ci          # full pipeline: format + lint + typecheck + test
```

## Documentation

Full documentation is available at **[macaw-voice.github.io/macaw-openvoice](https://macaw-voice.github.io/macaw-openvoice)**.

- [Getting Started](https://macaw-voice.github.io/macaw-openvoice/docs/getting-started/installation)
- [Streaming Guide](https://macaw-voice.github.io/macaw-openvoice/docs/guides/streaming-stt)
- [Full-Duplex Guide](https://macaw-voice.github.io/macaw-openvoice/docs/guides/full-duplex)
- [Adding an Engine](https://macaw-voice.github.io/macaw-openvoice/docs/guides/adding-engine)
- [API Reference](https://macaw-voice.github.io/macaw-openvoice/docs/api-reference/rest-api)
- [Architecture](https://macaw-voice.github.io/macaw-openvoice/docs/architecture/overview)

## Contributing

We welcome contributions! Please read our [Contributing Guide](CONTRIBUTING.md) before submitting a pull request.

## License

[Apache License 2.0](LICENSE)
