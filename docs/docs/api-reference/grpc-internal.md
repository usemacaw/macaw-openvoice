---
title: gRPC (Internal)
sidebar_position: 3
---

# gRPC Internal Protocol

Macaw uses gRPC for communication between the runtime process and its worker subprocesses. This protocol is **internal** and not intended for direct client use.

:::warning
The gRPC API is an implementation detail. Use the [REST API](rest-api) or [WebSocket Protocol](websocket-protocol) for client integrations.
:::

---

## Overview

Each engine runs in an isolated subprocess that exposes a gRPC server. The runtime connects as a gRPC client.

```
Runtime Process                    Worker Subprocess
+------------------+              +------------------+
|  API Server      |   gRPC      |  STT Backend     |
|  Scheduler       | <========> |  (Faster-Whisper) |
|  Session Manager |   :50051   |                    |
+------------------+              +------------------+

+------------------+              +------------------+
|  API Server      |   gRPC      |  TTS Backend     |
|                  | <========> |  (Kokoro)         |
|                  |   :50052   |                    |
+------------------+              +------------------+
```

---

## Proto Definitions

### STT Worker

```
src/macaw/proto/stt_worker.proto
```

The STT worker uses a **bidirectional streaming** RPC for real-time transcription:

- **Client stream**: Audio chunks (PCM bytes)
- **Server stream**: Transcription results (partial and final)

The gRPC stream itself serves as the health check mechanism -- a broken stream indicates a crashed worker, triggering automatic recovery.

### TTS Worker

```
src/macaw/proto/tts_worker.proto
```

The TTS worker uses a **server-side streaming** RPC:

- **Request**: Text input, voice ID, and synthesis parameters
- **Server stream**: Audio chunks for low-latency streaming

---

## Worker Lifecycle

1. **Spawn**: Runtime launches worker as a subprocess on a designated port
2. **Ready**: Worker loads the model and starts the gRPC server
3. **Serve**: Runtime sends requests over gRPC streams
4. **Crash/Recovery**: If the stream breaks, the runtime respawns the worker and replays uncommitted data from the WAL

### Default Ports

| Worker | Port |
|---|---|
| STT | `50051` |
| TTS | `50052` |

---

## Regenerating Stubs

If you modify the proto files, regenerate the Python stubs:

```bash
make proto
```

This runs `grpcio-tools` to generate `*_pb2.py` and `*_pb2_grpc.py` files in `src/macaw/proto/`.
