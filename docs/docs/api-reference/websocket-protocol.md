---
title: WebSocket Protocol
sidebar_position: 2
---

# WebSocket Protocol

The `/v1/realtime` endpoint supports real-time bidirectional audio streaming with JSON control messages and binary audio frames.

---

## Connecting

```bash title="Connect with wscat"
wscat -c "ws://localhost:8000/v1/realtime?model=faster-whisper-large-v3"
```

```python title="Connect with Python websockets"
import websockets

async with websockets.connect(
    "ws://localhost:8000/v1/realtime?model=faster-whisper-large-v3"
) as ws:
    # Send audio frames, receive events
    ...
```

### Query Parameters

| Parameter | Required | Description |
|---|---|---|
| `model` | Yes | STT model ID |

---

## Message Flow

```
Client                                          Server
  |                                                |
  |  ---- [connect] ---->                          |
  |                          <---- session.created |
  |                                                |
  |  ---- session.configure ---->                  |
  |                                                |
  |  ---- [binary PCM frames] ---->                |
  |                          <---- vad.speech_start|
  |                          <---- transcript.partial
  |                          <---- transcript.partial
  |                          <---- transcript.final|
  |                          <---- vad.speech_end  |
  |                                                |
  |  ---- tts.speak ---->                          |
  |                          <---- tts.speaking_start
  |                          <---- [binary audio]  |
  |                          <---- tts.speaking_end|
  |                                                |
  |  ---- [close] ---->                            |
```

---

## Client to Server Messages

### Binary Frames (Audio)

Send raw PCM audio as binary WebSocket frames:

| Property | Value |
|---|---|
| Format | PCM 16-bit signed integer |
| Sample rate | Any (resampled automatically to 16 kHz) |
| Channels | Mono (or first channel extracted) |

:::tip
You can send audio at any sample rate -- the runtime automatically resamples to 16 kHz before processing.
:::

### session.configure

Configure the session after connecting. Optional -- defaults are used if not sent.

```json
{
  "type": "session.configure",
  "vad": {
    "sensitivity": "normal"
  },
  "language": "en",
  "hot_words": ["Macaw", "OpenVoice", "transcription"],
  "tts_model": "kokoro-v1"
}
```

| Field | Type | Description |
|---|---|---|
| `vad.sensitivity` | string | `high`, `normal`, or `low` |
| `language` | string | ISO 639-1 language code |
| `hot_words` | string[] | Domain-specific keywords to boost |
| `tts_model` | string | TTS model for full-duplex mode |

### tts.speak

Trigger text-to-speech synthesis. The server will stream audio back as binary frames.

```json
{
  "type": "tts.speak",
  "text": "Hello, how can I help you?",
  "voice": "default"
}
```

| Field | Type | Description |
|---|---|---|
| `text` | string | Text to synthesize |
| `voice` | string | Voice identifier |

:::warning
Sending a new `tts.speak` while one is already active **cancels the previous one**. TTS requests do not queue -- only the latest one is processed.
:::

### tts.cancel

Cancel the currently active TTS synthesis.

```json
{
  "type": "tts.cancel"
}
```

---

## Server to Client Events

### session.created

Sent immediately after the WebSocket connection is established.

```json
{
  "type": "session.created",
  "session_id": "abc123"
}
```

### vad.speech_start

Speech activity detected. The runtime has started buffering audio for transcription.

```json
{
  "type": "vad.speech_start",
  "timestamp": 1234567890.123
}
```

### transcript.partial

Intermediate transcription hypothesis. Updated as more audio arrives. **Unstable** -- may change with subsequent partials.

```json
{
  "type": "transcript.partial",
  "text": "Hello how can"
}
```

:::info
Partials are best-effort hypotheses. Never apply post-processing (ITN) to partials -- they are too unstable for reliable formatting.
:::

### transcript.final

Confirmed transcription segment. This is the stable, post-processed result.

```json
{
  "type": "transcript.final",
  "text": "Hello, how can I help you today?",
  "language": "en",
  "duration": 3.42
}
```

### vad.speech_end

Speech activity has ended.

```json
{
  "type": "vad.speech_end",
  "timestamp": 1234567890.456
}
```

### tts.speaking_start

TTS synthesis has begun. **STT is automatically muted** during TTS to prevent feedback loops.

```json
{
  "type": "tts.speaking_start"
}
```

### Binary Frames (TTS Audio)

During TTS synthesis, the server sends binary WebSocket frames containing audio data:

| Direction | Content |
|---|---|
| Server to client (binary) | Always TTS audio |
| Client to server (binary) | Always STT audio |

:::tip
There is no ambiguity in binary frame direction -- server-to-client binary frames are **always** TTS audio, and client-to-server binary frames are **always** STT audio.
:::

### tts.speaking_end

TTS synthesis is complete. STT is **automatically unmuted** and resumes processing audio.

```json
{
  "type": "tts.speaking_end"
}
```

### error

An error occurred during processing.

```json
{
  "type": "error",
  "message": "Worker connection lost",
  "recoverable": true
}
```

| Field | Type | Description |
|---|---|---|
| `message` | string | Human-readable error description |
| `recoverable` | boolean | Whether the session can continue |

---

## Full-Duplex Mode

When a `tts_model` is configured, the WebSocket operates in full-duplex mode:

1. Client streams audio for STT continuously
2. Client sends `tts.speak` to trigger speech synthesis
3. Server automatically **mutes STT** during TTS playback (via `try/finally` -- unmute is guaranteed even if TTS crashes)
4. After TTS completes, STT **resumes automatically**

See the [Full-Duplex Guide](../guides/full-duplex) for implementation details.
