---
title: Quickstart
sidebar_position: 2
---

# Quickstart

Get from zero to your first transcription in under 5 minutes.

---

## Step 1: Install

```bash
pip install macaw-openvoice[server,grpc,faster-whisper]
```

:::tip
If you plan to use TTS as well, add the `kokoro` extra:
```bash
pip install macaw-openvoice[server,grpc,faster-whisper,kokoro]
```
:::

---

## Step 2: Start the Server

```bash
macaw serve
```

You should see output like this:

```
  ╔═══════════════════════════════════════╗
  ║       Macaw OpenVoice v1.0.0         ║
  ╚═══════════════════════════════════════╝

INFO  Scanning models in ~/.macaw/models
INFO  Found 2 model(s): faster-whisper-tiny (STT), kokoro-v1 (TTS)
INFO  Spawning STT worker   port=50051  engine=faster-whisper
INFO  Spawning TTS worker   port=50052  engine=kokoro
INFO  Scheduler started     aging=30.0s  batch_ms=75.0  batch_max=8
INFO  Uvicorn running on http://127.0.0.1:8000
```

The server is now ready to accept requests on port **8000**.

---

## Step 3: Transcribe Audio

### Via REST API (curl)

```bash
curl -X POST http://localhost:8000/v1/audio/transcriptions \
  -F file=@audio.wav \
  -F model=faster-whisper-large-v3
```

**Response:**

```json
{
  "text": "Hello, how can I help you today?"
}
```

### Via CLI

```bash
macaw transcribe audio.wav --model faster-whisper-large-v3
```

### Via OpenAI Python SDK

```python
from openai import OpenAI

client = OpenAI(base_url="http://localhost:8000/v1", api_key="not-needed")

result = client.audio.transcriptions.create(
    model="faster-whisper-large-v3",
    file=open("audio.wav", "rb"),
)
print(result.text)
```

:::info
Macaw implements the OpenAI Audio API contract, so any OpenAI-compatible client library works without modification. Just change the `base_url`.
:::

---

## Step 4: Try Real-Time Streaming

Connect via WebSocket for live transcription:

```bash
wscat -c "ws://localhost:8000/v1/realtime?model=faster-whisper-large-v3"
```

Send binary PCM audio frames and receive JSON transcript events:

```json
{"type": "transcript.partial", "text": "Hello how"}
{"type": "transcript.final", "text": "Hello, how can I help you today?"}
```

See the [Streaming STT guide](../guides/streaming-stt) for the full protocol.

---

## Step 5: Text-to-Speech

Generate speech from text:

```bash
curl -X POST http://localhost:8000/v1/audio/speech \
  -H "Content-Type: application/json" \
  -d '{"model": "kokoro-v1", "input": "Hello, welcome to Macaw!", "voice": "default"}' \
  --output speech.wav
```

Or use the OpenAI SDK:

```python
response = client.audio.speech.create(
    model="kokoro-v1",
    input="Hello, welcome to Macaw!",
    voice="default",
)
response.stream_to_file("output.wav")
```

---

## What's Next?

| Want to... | Read... |
|---|---|
| Stream audio in real time | [Streaming STT](../guides/streaming-stt) |
| Build a voice assistant with STT + TTS | [Full-Duplex Guide](../guides/full-duplex) |
| Transcribe files in batch | [Batch Transcription](../guides/batch-transcription) |
| Understand the architecture | [Architecture Overview](../architecture/overview) |
| Add a new engine | [Adding an Engine](../guides/adding-engine) |
| Use the CLI | [CLI Reference](../guides/cli) |
