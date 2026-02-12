---
title: Full-Duplex STT + TTS
sidebar_position: 3
---

# Full-Duplex STT + TTS

Macaw supports **full-duplex** voice interactions on a single WebSocket connection. The client streams audio for STT while simultaneously receiving synthesized speech from TTS — all on the same `/v1/realtime` endpoint.

## How It Works

The key mechanism is **mute-on-speak**: when TTS is active, STT is muted to prevent the synthesized audio from being fed back into the speech recognizer.

```
Timeline ─────────────────────────────────────────────────▶

Client sends audio (STT) ████████████░░░░░░░░░░░████████████
                                     │          │
                              tts.speak    tts.speaking_end
                                     │          │
Server sends audio (TTS)             │██████████│
                                     │          │
STT active                   ████████│  muted   │████████████
```

### Flow

1. Client streams audio frames for STT (binary messages)
2. Client sends a `tts.speak` command (JSON message)
3. Server **mutes** STT — incoming audio frames are dropped
4. Server emits `tts.speaking_start` event
5. Server streams TTS audio as binary frames (server → client)
6. When synthesis completes, server emits `tts.speaking_end`
7. Server **unmutes** STT — audio processing resumes
8. Client continues streaming audio for STT

:::tip Directionality is unambiguous
- **Binary frames client → server** are always STT audio
- **Binary frames server → client** are always TTS audio
- **Text frames** (both directions) are always JSON events/commands
:::

## Setup

### 1. Connect to the WebSocket

```python title="Connect with STT model"
import asyncio
import json
import websockets

async def full_duplex():
    uri = "ws://localhost:8000/v1/realtime?model=faster-whisper-large-v3"
    async with websockets.connect(uri) as ws:
        # Wait for session.created
        event = json.loads(await ws.recv())
        print(f"Session: {event['session_id']}")
```

### 2. Configure TTS Model

Set the TTS model for the session:

```python title="Configure TTS"
        await ws.send(json.dumps({
            "type": "session.configure",
            "model_tts": "kokoro"
        }))
```

:::info Auto-discovery
If you don't set `model_tts`, the server will auto-discover the first available TTS model from the registry when you send a `tts.speak` command.
:::

### 3. Request Speech Synthesis

```python title="Send tts.speak"
        await ws.send(json.dumps({
            "type": "tts.speak",
            "text": "Hello! How can I help you today?",
            "voice": "af_heart"
        }))
```

### 4. Handle Events and Audio

```python title="Event loop"
        async for message in ws:
            if isinstance(message, bytes):
                # TTS audio chunk (PCM 16-bit, 24kHz)
                play_audio(message)
            else:
                event = json.loads(message)
                match event["type"]:
                    case "transcript.partial":
                        print(f"  ...{event['text']}", end="\r")
                    case "transcript.final":
                        print(f"  User: {event['text']}")
                        # Generate response and speak it
                        response = get_llm_response(event["text"])
                        await ws.send(json.dumps({
                            "type": "tts.speak",
                            "text": response
                        }))
                    case "tts.speaking_start":
                        print("  [Speaking...]")
                    case "tts.speaking_end":
                        print(f"  [Done, {event['duration_ms']}ms]")
```

## Commands

### `tts.speak`

Request speech synthesis:

```json title="Client → Server"
{
  "type": "tts.speak",
  "text": "Hello, how can I help you?",
  "voice": "af_heart"
}
```

| Field | Type | Default | Description |
|-------|------|---------|-------------|
| `text` | string | *required* | Text to synthesize |
| `voice` | string | `"default"` | Voice ID (see [available voices](#available-voices)) |

:::warning Auto-cancellation
If a `tts.speak` command arrives while a previous synthesis is still in progress, the **previous one is cancelled automatically**. TTS commands do not accumulate — only the latest one plays.
:::

### `tts.cancel`

Cancel the current TTS synthesis:

```json title="Client → Server"
{
  "type": "tts.cancel"
}
```

This immediately:
1. Stops sending audio chunks
2. Unmutes STT
3. Emits `tts.speaking_end` with `"cancelled": true`

## Events

### `tts.speaking_start`

Emitted when the first audio chunk is ready to send:

```json title="Server → Client"
{
  "type": "tts.speaking_start",
  "text": "Hello, how can I help you?"
}
```

At this point, STT is muted and audio chunks will follow.

### `tts.speaking_end`

Emitted when synthesis completes (or is cancelled):

```json title="Server → Client"
{
  "type": "tts.speaking_end",
  "duration_ms": 1250,
  "cancelled": false
}
```

| Field | Type | Description |
|-------|------|-------------|
| `duration_ms` | int | Total duration of audio sent |
| `cancelled` | bool | `true` if stopped early via `tts.cancel` or new `tts.speak` |

After this event, STT is unmuted and audio processing resumes.

## TTS Audio Format

TTS audio chunks are sent as binary WebSocket frames:

| Property | Value |
|----------|-------|
| Encoding | PCM 16-bit signed, little-endian |
| Sample rate | 24,000 Hz (Kokoro default) |
| Channels | Mono |
| Chunk size | ~4,096 bytes (~85ms at 24kHz) |

:::info Different sample rates
STT input is 16kHz, but TTS output is 24kHz (Kokoro's native rate). The client is responsible for handling both sample rates appropriately (e.g., separate audio output streams).
:::

## Mute-on-Speak Details

The mute mechanism ensures STT doesn't hear the TTS output:

```
tts.speak received
       │
       ▼
  ┌──────────┐
  │ mute()   │  STT frames dropped (counter incremented)
  └────┬─────┘
       │
       ▼
  ┌──────────────────────┐
  │ Stream TTS audio     │  Binary frames server → client
  │ chunks to client     │
  └────┬─────────────────┘
       │
       ▼ (in finally block — always executes)
  ┌──────────┐
  │ unmute() │  STT processing resumes
  └──────────┘
```

### Guarantees

| Property | Guarantee |
|----------|-----------|
| Unmute on completion | Always — via `try/finally` |
| Unmute on TTS error | Always — via `try/finally` |
| Unmute on cancel | Always — via `try/finally` |
| Unmute on WebSocket close | Always — session cleanup |
| Idempotent | `mute()` and `unmute()` can be called multiple times |

:::warning
The `try/finally` pattern is critical. If TTS crashes mid-synthesis, the `finally` block still calls `unmute()`. Without this, a TTS error would permanently mute STT for the session.
:::

## Available Voices

Kokoro supports multiple languages and voices. The voice ID prefix determines the language:

| Prefix | Language | Example |
|--------|----------|---------|
| `a` | English (US) | `af_heart`, `am_adam` |
| `b` | English (UK) | `bf_emma`, `bm_george` |
| `e` | Spanish | `ef_dora`, `em_alex` |
| `f` | French | `ff_siwis` |
| `h` | Hindi | `hf_alpha`, `hm_omega` |
| `i` | Italian | `if_sara`, `im_nicola` |
| `j` | Japanese | `jf_alpha`, `jm_omega` |
| `p` | Portuguese | `pf_dora`, `pm_alex` |
| `z` | Chinese | `zf_xiaobei`, `zm_yunjian` |

The second character indicates gender: `f` = female, `m` = male.

**Default voice:** `af_heart` (English US, female)

## Complete Example

```python title="voice_assistant.py"
import asyncio
import json
import websockets

async def voice_assistant():
    uri = "ws://localhost:8000/v1/realtime?model=faster-whisper-large-v3"

    async with websockets.connect(uri) as ws:
        # Wait for session
        event = json.loads(await ws.recv())
        print(f"Connected: {event['session_id']}")

        # Configure TTS
        await ws.send(json.dumps({
            "type": "session.configure",
            "model_tts": "kokoro",
            "vad_sensitivity": "normal",
            "enable_itn": True
        }))

        # Greet the user
        await ws.send(json.dumps({
            "type": "tts.speak",
            "text": "Hi! I'm ready to help. Go ahead and speak.",
            "voice": "af_heart"
        }))

        # Main loop: listen for events
        async for message in ws:
            if isinstance(message, bytes):
                # TTS audio — send to speaker
                play_audio(message)
                continue

            event = json.loads(message)

            if event["type"] == "transcript.final":
                user_text = event["text"]
                print(f"User: {user_text}")

                # Get response from your LLM
                response = await get_llm_response(user_text)
                print(f"Assistant: {response}")

                # Speak the response
                await ws.send(json.dumps({
                    "type": "tts.speak",
                    "text": response,
                    "voice": "af_heart"
                }))

            elif event["type"] == "tts.speaking_end":
                if event.get("cancelled"):
                    print("  (interrupted)")

asyncio.run(voice_assistant())
```

## Next Steps

| Goal | Guide |
|------|-------|
| WebSocket protocol reference | [WebSocket Protocol](../api-reference/websocket-protocol) |
| Understanding mute and session state | [Session Manager](../architecture/session-manager) |
| Batch transcription instead | [Batch Transcription](./batch-transcription) |
