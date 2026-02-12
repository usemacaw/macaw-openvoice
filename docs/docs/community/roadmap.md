---
title: Roadmap
sidebar_position: 3
---

# Roadmap

Macaw OpenVoice has completed all 9 milestones of the initial Product Requirements Document. The runtime is fully functional with STT, TTS, and full-duplex capabilities.

## Completed Milestones

| Phase | Milestone | What Was Delivered |
|:---:|-----------|-------------------|
| 1 | **M1 — API Server** | FastAPI with health endpoint, CORS, OpenAI-compatible structure |
| 1 | **M2 — Model Registry** | `macaw.yaml` manifests, model lifecycle, architecture field |
| 2 | **M3 — Scheduler** | Priority queue, cancellation, dynamic batching, latency tracking |
| 2 | **M4 — STT Workers** | gRPC subprocess workers, Faster-Whisper backend, crash recovery |
| 3 | **M5 — Streaming STT** | WebSocket `/v1/realtime`, VAD pipeline, streaming preprocessor |
| 3 | **M6 — Session Manager** | State machine (6 states), ring buffer, WAL, backpressure |
| 4 | **M7 — Multi-Engine** | WeNet CTC backend, pipeline adaptation by architecture |
| 4 | **M8 — TTS** | Kokoro TTS backend, `POST /v1/audio/speech`, gRPC TTS worker |
| 5 | **M9 — Full-Duplex** | Mute-on-speak, `tts.speak`/`tts.cancel`, STT+TTS on same WebSocket |

## Current State

- **1,600+ tests** passing (unit + integration)
- **3 STT architectures** supported: encoder-decoder, CTC, streaming-native
- **2 STT engines**: Faster-Whisper, WeNet
- **1 TTS engine**: Kokoro (9 languages)
- **Full-duplex** voice interactions on a single WebSocket
- **OpenAI-compatible** REST API
- **Ollama-style** CLI

## What's Next

The following areas are under consideration for future development. These are not commitments — they represent directions the project may explore based on community feedback and priorities.

### Engine Ecosystem

| Feature | Description |
|---------|-------------|
| Paraformer backend | Streaming-native architecture support |
| Piper TTS | Lightweight TTS alternative for CPU-only deployments |
| Whisper.cpp | GGML-based inference without Python/CUDA dependency |
| Multi-model serving | Load multiple models per worker type |

### Scalability

| Feature | Description |
|---------|-------------|
| Worker pooling | Multiple worker instances per engine for higher throughput |
| Horizontal scaling | Multiple runtime instances behind a load balancer |
| GPU sharing | Time-slice GPU across STT and TTS workers |
| Kubernetes operator | Automated deployment with GPU scheduling |

### Features

| Feature | Description |
|---------|-------------|
| Speaker diarization | Identify and label different speakers |
| Word-level timestamps | Per-word timing in streaming mode |
| Custom vocabularies | User-defined vocabularies beyond hot words |
| Audio streaming output | Server-Sent Events for TTS as an alternative to WebSocket |
| Barge-in | Client interrupts TTS to speak (currently requires `tts.cancel`) |

### Observability

| Feature | Description |
|---------|-------------|
| OpenTelemetry | Distributed tracing across runtime and workers |
| Dashboard templates | Pre-built Grafana dashboards for Prometheus metrics |
| Structured audit logging | Request/response logging for compliance |

## Contributing

Want to help shape the roadmap? See the [Contributing Guide](./contributing) to get started, or open an issue on [GitHub](https://github.com/macaw-voice/macaw-openvoice/issues) to discuss new ideas.
