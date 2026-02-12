---
title: Architecture Overview
sidebar_position: 1
---

# Architecture Overview

Macaw OpenVoice is a **unified voice runtime** that orchestrates STT (Speech-to-Text) and TTS (Text-to-Speech) engines through a single process with isolated gRPC workers. It provides an OpenAI-compatible API while keeping engines modular and crash-isolated.

## High-Level Architecture

```
                         ┌─────────────────────────────────────────────┐
                         │               Macaw Runtime                │
                         │                                            │
  ┌──────────┐     ┌─────┴─────┐     ┌───────────┐     ┌───────────┐ │
  │  Clients  │────▶│ API Server │────▶│ Scheduler │────▶│ STT Worker│ │ (subprocess)
  │           │     │  (FastAPI) │     │           │     │  gRPC:50051│ │
  │ REST      │     │            │     │ Priority  │     └───────────┘ │
  │ WebSocket │     │ /v1/audio/ │     │ Queue     │                   │
  │ CLI       │     │ /v1/realtime│    │ Batching  │     ┌───────────┐ │
  └──────────┘     │            │     │ Cancel    │────▶│ TTS Worker│ │ (subprocess)
                    └─────┬─────┘     └───────────┘     │  gRPC:50052│ │
                          │                              └───────────┘ │
                    ┌─────┴──────────────┐                             │
                    │  Session Manager   │                             │
                    │  (streaming only)  │                             │
                    │                    │                             │
                    │  State Machine     │                             │
                    │  Ring Buffer       │                             │
                    │  WAL Recovery      │                             │
                    └────────────────────┘                             │
                         │                                            │
                    ┌────┴───────────────────────────────┐            │
                    │         Audio Pipeline             │            │
                    │  Preprocessing → VAD → Postprocess │            │
                    └────────────────────────────────────┘            │
                         └────────────────────────────────────────────┘
```

## Core Layers

### API Server

The FastAPI server exposes three types of interfaces:

| Interface | Endpoint | Use Case |
|-----------|----------|----------|
| REST (batch) | `POST /v1/audio/transcriptions` | File transcription |
| REST (batch) | `POST /v1/audio/translations` | File translation to English |
| REST (batch) | `POST /v1/audio/speech` | Text-to-speech synthesis |
| WebSocket | `WS /v1/realtime` | Streaming STT + full-duplex TTS |
| Health | `GET /health`, `GET /v1/models` | Monitoring and model listing |

All REST endpoints are **OpenAI API-compatible** — existing OpenAI client libraries work without modification.

### Scheduler

The Scheduler routes **batch** (REST) requests to gRPC workers. It provides:

- **Priority queue** with two levels: `REALTIME` and `BATCH`
- **Cancellation** for queued and in-flight requests
- **Dynamic batching** to group requests by model
- **Latency tracking** with TTL-based cleanup

:::warning Streaming bypasses the Scheduler
WebSocket streaming uses `StreamingGRPCClient` directly — it does **not** pass through the priority queue. The Scheduler is only for REST batch requests.
:::

See [Scheduling](./scheduling) for details.

### Session Manager

The Session Manager coordinates **streaming STT only**. Each WebSocket connection gets its own session with:

- **State machine** — 6 states: `INIT → ACTIVE → SILENCE → HOLD → CLOSING → CLOSED`
- **Ring buffer** — pre-allocated circular buffer for audio frames (zero allocations during streaming)
- **WAL** — in-memory Write-Ahead Log for crash recovery
- **Backpressure** — rate limiting at 1.2x real-time, frame dropping when overloaded

:::info TTS is stateless
TTS does not use the Session Manager. Each `tts.speak` request is independent — no state is carried between synthesis calls.
:::

See [Session Manager](./session-manager) for details.

### Audio Pipeline

The audio pipeline runs **in the runtime**, not in the engine. This guarantees consistent behavior across all engines.

```
Input Audio → Resample (16kHz) → DC Remove → Gain Normalize → VAD → Engine
                                                                         ↓
                                                              Raw Text → ITN → Output
```

| Stage | Layer | Description |
|-------|-------|-------------|
| Resample | Preprocessing | Convert to 16kHz mono via `scipy.signal.resample_poly` |
| DC Remove | Preprocessing | 2nd-order Butterworth HPF at 20Hz |
| Gain Normalize | Preprocessing | Peak normalization to -3.0 dBFS |
| Energy Pre-filter | VAD | RMS + spectral flatness check (~0.1ms) |
| Silero VAD | VAD | Neural speech probability (~2ms on CPU) |
| ITN | Postprocessing | Inverse Text Normalization via NeMo (fail-open) |

See [VAD Pipeline](./vad-pipeline) for details.

### Workers

Workers are **gRPC subprocesses**. A worker crash does not bring down the runtime — the Session Manager recovers by resending uncommitted audio from the ring buffer.

| Worker | Port | Protocol | Engines |
|--------|------|----------|---------|
| STT | 50051 | Bidirectional streaming | Faster-Whisper, WeNet |
| TTS | 50052 | Server streaming | Kokoro |

**Worker lifecycle:**

```
STARTING → READY → BUSY → STOPPING → STOPPED
              ↑       │
              └───────┘
              (on idle)

          CRASHED → (auto-restart, max 3 in 60s)
```

The WorkerManager handles health probing (exponential backoff, 30s timeout), graceful shutdown (SIGTERM → 5s wait → SIGKILL), and automatic restart with rate limiting.

### Model Registry

The Registry manages model manifests (`macaw.yaml` files) and lifecycle. Models declare their `architecture` field, which tells the runtime how to adapt the pipeline:

| Architecture | Example | LocalAgreement | Cross-segment Context | Native Partials |
|-------------|---------|:-:|:-:|:-:|
| `encoder-decoder` | Faster-Whisper | Yes | Yes (224 tokens) | No |
| `ctc` | WeNet | No | No | Yes |
| `streaming-native` | Paraformer | No | No | Yes |

## Data Flow

### Batch Request (REST)

```
Client → POST /v1/audio/transcriptions
       → Preprocessing pipeline (resample, DC remove, normalize)
       → Scheduler priority queue
       → gRPC TranscribeFile to STT worker
       → Postprocessing (ITN)
       → JSON response to client
```

### Streaming Request (WebSocket)

```
Client → WS /v1/realtime
       → Session created (state: INIT)
       → Binary frames arrive
       → StreamingPreprocessor (per-frame)
       → VAD (energy pre-filter → Silero)
       → SPEECH_START → state: ACTIVE
       → Frames written to ring buffer
       → Frames sent via StreamingGRPCClient to STT worker
       → Partial/final transcripts sent back to client
       → SPEECH_END → state: SILENCE
       → ITN applied on final transcripts only
```

### Full-Duplex (STT + TTS)

```
Client sends audio (STT)  ──────────────────────────────▶  partials/finals
Client sends tts.speak     ──▶  mute STT
                                 ──▶  gRPC Synthesize to TTS worker
                                 ──▶  tts.speaking_start event
                                 ──▶  binary audio chunks (server → client)
                                 ──▶  tts.speaking_end event
                                 ──▶  unmute STT (guaranteed via try/finally)
```

## Key Design Decisions

| Decision | Rationale |
|----------|-----------|
| **Single process, subprocess workers** | Crash isolation without distributed system complexity |
| **VAD in runtime, not engine** | Consistent behavior across all engines |
| **Preprocessing before VAD** | Normalized audio ensures stable VAD thresholds |
| **Streaming bypasses Scheduler** | Direct gRPC connection avoids queue latency for real-time |
| **Mute-on-speak for full-duplex** | Prevents TTS audio from feeding back into STT |
| **Pipeline adapts by architecture** | Encoder-decoder gets LocalAgreement; CTC uses native partials |
| **ITN on finals only** | Partials are unstable — ITN would produce confusing output |
| **In-memory WAL** | Fast recovery without disk I/O overhead |
| **gRPC stream break as heartbeat** | No separate health polling needed for crash detection |
