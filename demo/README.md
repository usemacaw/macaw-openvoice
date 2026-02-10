# Macaw OpenVoice Demo

End-to-end demo application with a FastAPI backend and a React/Next.js frontend (shadcn/Radix UI components).

## Quickstart

```bash
./demo/start.sh
```

This starts the backend (port 9000) and frontend (port 3000) together. Ctrl+C stops both.

## Structure

```
demo/
  backend/          # FastAPI: registry, gRPC workers, scheduler, Macaw routes + /demo/*
  frontend/         # Next.js 14: dashboard, streaming STT, TTS playback
  start.sh     # Script that starts backend + frontend
```

## Prerequisites

- Python 3.12+ (via `uv`)
- Node.js 18+
- At least one STT model installed in `~/.macaw/models` (or configure `DEMO_MODELS_DIR`)

## Installation

### Backend

```bash
# From the project root
cd /path/to/macaw-openvoice
.venv/bin/pip install -e ".[server,grpc]"
```

### Frontend

```bash
cd demo/frontend
npm install
```

## Running Separately

### Backend only

```bash
# Option 1: script
SKIP_FRONTEND=1 ./demo/start.sh

# Option 2: direct
.venv/bin/python -m uvicorn demo.backend.app:app --reload --host 127.0.0.1 --port 9000
```

### Frontend only

```bash
# Option 1: script
SKIP_BACKEND=1 ./demo/start.sh

# Option 2: direct
cd demo/frontend
NEXT_PUBLIC_DEMO_API=http://localhost:9000 npm run dev
```

## Environment Variables

| Variable | Default | Description |
|----------|---------|-----------|
| `DEMO_HOST` | `127.0.0.1` | Backend host |
| `DEMO_PORT` | `9000` | Backend port |
| `FRONTEND_PORT` | `3000` | Next.js frontend port |
| `DEMO_MODELS_DIR` | `~/.macaw/models` | Models directory |
| `DEMO_ALLOWED_ORIGINS` | `http://localhost:3000` | CORS origins (comma-separated) |
| `DEMO_BATCH_ACCUMULATE_MS` | `75` | Batcher accumulation time |
| `DEMO_BATCH_MAX_SIZE` | `8` | Maximum batch size |
| `UVICORN_RELOAD` | `1` | Backend hot-reload (`0` to disable) |
| `SKIP_FRONTEND` | `0` | `1` to start only the backend |
| `SKIP_BACKEND` | `0` | `1` to start only the frontend |
| `NEXT_PUBLIC_DEMO_API` | `http://localhost:9000` | Backend URL for the frontend |

## Demo Features

The interface has 3 tabs:

### Dashboard (Batch STT)
- List of installed models (STT/TTS) with architecture badges
- Scheduler queue metrics (depth, priority)
- Audio file upload for batch transcription
- Jobs table with status, result, and cancellation

### Streaming STT
- Real-time microphone recording via Web Audio API
- WebSocket `/v1/realtime` with JSON event protocol
- Visual indicators: connection, VAD (speech detected), waveform
- Partial and final transcriptions with auto-scroll

### Text-to-Speech
- Speech synthesis via `POST /v1/audio/speech`
- Controls: voice, speed (0.5x-2.0x)
- Integrated audio player
- TTFB measurement (Time to First Byte)

## Endpoints

### Macaw routes (prefix `/api`)
- `POST /api/v1/audio/transcriptions` — batch transcription
- `POST /api/v1/audio/translations` — translation to English
- `POST /api/v1/audio/speech` — TTS synthesis
- `WS /api/v1/realtime` — streaming STT + TTS full-duplex
- `GET /api/health` — health check

### Demo routes
- `GET /demo/models` — list registry models
- `GET /demo/queue` — queue metrics
- `GET /demo/jobs` — list jobs
- `POST /demo/jobs` — submit transcription job
- `POST /demo/jobs/{id}/cancel` — cancel job

## Observability

- Swagger UI: http://localhost:9000/docs
- Health: http://localhost:9000/api/health
- Real scheduler with PriorityQueue, CancellationManager, BatchAccumulator, and LatencyTracker
- gRPC workers isolated as subprocesses
