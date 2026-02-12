---
title: Streaming STT
sidebar_position: 2
---

# Streaming STT

Macaw provides real-time speech-to-text via WebSocket at `/v1/realtime`. Audio frames are sent as binary messages and transcription events are returned as JSON.

## Quick Start

### Using wscat

```bash title="Connect and stream"
wscat -c "ws://localhost:8000/v1/realtime?model=faster-whisper-large-v3"
```

### Using Python

```python title="stream_audio.py"
import asyncio
import json
import websockets

async def stream_microphone():
    uri = "ws://localhost:8000/v1/realtime?model=faster-whisper-large-v3"

    async with websockets.connect(uri) as ws:
        # Wait for session.created
        msg = json.loads(await ws.recv())
        print(f"Session: {msg['session_id']}")

        # Send audio frames (16-bit PCM, 16kHz)
        # In production, read from microphone
        with open("audio.raw", "rb") as f:
            while chunk := f.read(3200):  # 100ms frames
                await ws.send(chunk)
                await asyncio.sleep(0.1)

                # Check for transcription events (non-blocking)
                try:
                    response = await asyncio.wait_for(ws.recv(), timeout=0.01)
                    event = json.loads(response)
                    if event["type"] == "transcript.partial":
                        print(f"  ...{event['text']}", end="\r")
                    elif event["type"] == "transcript.final":
                        print(f"  {event['text']}")
                except asyncio.TimeoutError:
                    pass

asyncio.run(stream_microphone())
```

### Using the CLI

```bash title="Stream from microphone"
macaw transcribe --stream --model faster-whisper-large-v3
```

## Connection

### URL Format

```
ws://HOST:PORT/v1/realtime?model=MODEL&language=LANG
```

| Parameter | Required | Default | Description |
|-----------|:---:|---------|-------------|
| `model` | Yes | — | STT model name |
| `language` | No | auto | ISO 639-1 code (e.g., `en`, `pt`) |

### Session Created

After connecting, the server immediately sends a `session.created` event:

```json title="Server → Client"
{
  "type": "session.created",
  "session_id": "sess_a1b2c3d4"
}
```

Save the `session_id` for logging and debugging.

## Audio Format

Send audio as **binary WebSocket frames**:

| Property | Value |
|----------|-------|
| Encoding | PCM 16-bit signed, little-endian |
| Sample rate | 16,000 Hz |
| Channels | Mono |
| Frame size | Recommended: 3,200 bytes (100ms) |

:::tip Preprocessing is automatic
If your audio isn't exactly 16kHz mono, the `StreamingPreprocessor` will resample it automatically. However, sending pre-formatted audio avoids unnecessary processing.
:::

## Transcription Events

### Partial Transcripts

Emitted in real-time as speech is being recognized. These are **unstable** — text may change as more context arrives:

```json title="Server → Client"
{
  "type": "transcript.partial",
  "text": "hello how are",
  "segment_id": 1
}
```

### Final Transcripts

Emitted when a speech segment ends (VAD detects silence). These are **stable** — the text will not change:

```json title="Server → Client"
{
  "type": "transcript.final",
  "text": "Hello, how are you doing today?",
  "segment_id": 1,
  "start": 0.5,
  "end": 2.8,
  "confidence": 0.94
}
```

:::info ITN on finals only
Inverse Text Normalization (e.g., "one hundred" → "100") is applied **only** to final transcripts. Partials return raw text because they change too frequently for ITN to be useful.
:::

## Session Configuration

After connecting, you can adjust session settings dynamically:

```json title="Client → Server"
{
  "type": "session.configure",
  "language": "pt",
  "vad_sensitivity": "high",
  "hot_words": ["Macaw", "OpenVoice", "gRPC"],
  "enable_itn": true
}
```

| Field | Type | Description |
|-------|------|-------------|
| `language` | string | Change language mid-session |
| `vad_sensitivity` | string | `"high"`, `"normal"`, or `"low"` |
| `hot_words` | string[] | Domain-specific terms to boost recognition |
| `enable_itn` | boolean | Enable/disable Inverse Text Normalization |
| `model_tts` | string | Set TTS model for full-duplex (see [Full-Duplex](./full-duplex)) |

## Buffer Management

### Manual Commit

Force the audio buffer to commit and produce a final transcript, even without a VAD silence event:

```json title="Client → Server"
{
  "type": "input_audio_buffer.commit"
}
```

This is useful when you know the user has finished speaking (e.g., they pressed a "done" button) but the VAD hasn't detected silence yet.

## Closing the Session

### Graceful Close

```json title="Client → Server"
{
  "type": "session.close"
}
```

The server flushes remaining data, emits any final transcripts, and sends:

```json title="Server → Client"
{
  "type": "session.closed",
  "session_id": "sess_a1b2c3d4",
  "reason": "client_close"
}
```

### Cancel

```json title="Client → Server"
{
  "type": "session.cancel"
}
```

Immediately closes the session without flushing. Pending transcripts are discarded.

## Backpressure

If the client sends audio faster than real-time (e.g., reading from a file without throttling), the server applies backpressure:

### Rate Limit Warning

```json title="Server → Client"
{
  "type": "session.rate_limit",
  "delay_ms": 50,
  "message": "Audio arriving faster than 1.2x real-time"
}
```

**Action:** slow down your send rate by the suggested `delay_ms`.

### Frames Dropped

```json title="Server → Client"
{
  "type": "session.frames_dropped",
  "dropped_ms": 200,
  "message": "Backlog exceeded 10s, frames dropped"
}
```

**Action:** this is informational — frames have already been dropped. Reduce send rate to prevent further drops.

:::warning Throttle file streaming
When streaming from a file (not a microphone), add `asyncio.sleep(0.1)` between 100ms frames to simulate real-time. Without throttling, the server will trigger backpressure.
:::

## Error Handling

### Error Events

```json title="Server → Client"
{
  "type": "error",
  "code": "worker_unavailable",
  "message": "STT worker not available for model faster-whisper-large-v3",
  "recoverable": true
}
```

| Field | Description |
|-------|-------------|
| `code` | Machine-readable error code |
| `message` | Human-readable description |
| `recoverable` | `true` if the client can retry or continue |

### Common Errors

| Code | Recoverable | Description |
|------|:---:|-------------|
| `model_not_found` | No | Requested model is not loaded |
| `worker_unavailable` | Yes | Worker crashed, recovery in progress |
| `session_timeout` | No | Session exceeded idle timeout |
| `invalid_command` | Yes | Unrecognized JSON command |

### Reconnection

If the WebSocket disconnects unexpectedly:

1. Reconnect with the same parameters
2. A new `session_id` will be assigned
3. Previous session state is not preserved — this is a fresh session

## Inactivity Timeout

The server monitors session activity:

| Parameter | Value |
|-----------|-------|
| Heartbeat ping | Every 10s |
| Auto-close timeout | 60s of inactivity |

If no audio frames arrive for 60 seconds, the server closes the session automatically.

## Next Steps

| Goal | Guide |
|------|-------|
| Add TTS to the same connection | [Full-Duplex](./full-duplex) |
| Batch file transcription instead | [Batch Transcription](./batch-transcription) |
| Full protocol reference | [WebSocket Protocol](../api-reference/websocket-protocol) |
