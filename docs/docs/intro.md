---
title: Introduction
sidebar_position: 1
---

Macaw OpenVoice is aruntime for voice pipelines. It orchestrates STT and TTS engines behind an OpenAI-compatible API, with streaming, VAD, preprocessing, and session management built in.

## Why Macaw

- One runtime for batch and streaming speech
- Full-duplex STT and TTS on the same WebSocket
- Engine-agnostic architecture for Faster-Whisper, WeNet, and Kokoro
- Production-grade session manager with recovery and backpressure

## Quick start

```bash
pip install Macaw-openvoice[server,grpc,faster-whisper]

Macaw serve

curl -X POST http://localhost:8000/v1/audio/transcriptions \
  -F file=@audio.wav \
  -F model=faster-whisper-large-v3
```

Next steps:

- Install the runtime: [Getting Started](getting-started/installation)
- Stream audio in real time: [Streaming STT](guides/streaming-stt)
- Add a new engine: [Adding an Engine](guides/adding-engine)
