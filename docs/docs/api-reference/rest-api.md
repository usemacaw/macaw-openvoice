---
title: REST API
sidebar_position: 1
---

# REST API Reference

Macaw implements the [OpenAI Audio API](https://platform.openai.com/docs/api-reference/audio) contract. Existing OpenAI client libraries work without modification -- just change the `base_url`.

---

## Endpoints Overview

| Method | Path | Description |
|---|---|---|
| `POST` | `/v1/audio/transcriptions` | Transcribe audio to text |
| `POST` | `/v1/audio/translations` | Translate audio to English |
| `POST` | `/v1/audio/speech` | Generate speech from text |
| `GET` | `/health` | Health check |

---

## POST /v1/audio/transcriptions

Transcribe an audio file into text.

### Request

| Field | Type | Required | Description |
|---|---|---|---|
| `file` | file | Yes | Audio file (WAV, MP3, FLAC, OGG, WebM) |
| `model` | string | Yes | Model ID (e.g., `faster-whisper-large-v3`) |
| `language` | string | No | ISO 639-1 language code |
| `prompt` | string | No | Context or hot words for the model |
| `response_format` | string | No | `json` (default), `text`, `srt`, `vtt`, `verbose_json` |
| `temperature` | float | No | Sampling temperature (0.0 - 1.0) |

### Examples

```bash title="Basic transcription"
curl -X POST http://localhost:8000/v1/audio/transcriptions \
  -F file=@audio.wav \
  -F model=faster-whisper-large-v3
```

```bash title="With language and format options"
curl -X POST http://localhost:8000/v1/audio/transcriptions \
  -F file=@audio.wav \
  -F model=faster-whisper-large-v3 \
  -F language=en \
  -F response_format=verbose_json
```

```python title="Python (OpenAI SDK)"
from openai import OpenAI

client = OpenAI(base_url="http://localhost:8000/v1", api_key="not-needed")

result = client.audio.transcriptions.create(
    model="faster-whisper-large-v3",
    file=open("audio.wav", "rb"),
    language="en",
    response_format="verbose_json",
)
print(result.text)
```

### Response

```json title="json format (default)"
{
  "text": "Hello, how can I help you today?"
}
```

```json title="verbose_json format"
{
  "task": "transcribe",
  "language": "en",
  "duration": 3.42,
  "text": "Hello, how can I help you today?",
  "segments": [
    {
      "id": 0,
      "start": 0.0,
      "end": 3.42,
      "text": "Hello, how can I help you today?"
    }
  ]
}
```

---

## POST /v1/audio/translations

Translate audio from any supported language to English.

### Request

| Field | Type | Required | Description |
|---|---|---|---|
| `file` | file | Yes | Audio file |
| `model` | string | Yes | Model ID |
| `prompt` | string | No | Context for the model |
| `response_format` | string | No | Same options as transcriptions |
| `temperature` | float | No | Sampling temperature |

### Example

```bash
curl -X POST http://localhost:8000/v1/audio/translations \
  -F file=@audio_portuguese.wav \
  -F model=faster-whisper-large-v3
```

### Response

```json
{
  "text": "Hello, how can I help you today?"
}
```

:::info
Translation always outputs English text, regardless of the source language.
:::

---

## POST /v1/audio/speech

Generate speech audio from text.

### Request

| Field | Type | Required | Description |
|---|---|---|---|
| `model` | string | Yes | TTS model ID (e.g., `kokoro-v1`) |
| `input` | string | Yes | Text to synthesize |
| `voice` | string | Yes | Voice identifier (e.g., `default`) |
| `response_format` | string | No | `wav` (default) or `pcm` |

### Examples

```bash title="Generate WAV file"
curl -X POST http://localhost:8000/v1/audio/speech \
  -H "Content-Type: application/json" \
  -d '{"model": "kokoro-v1", "input": "Hello, welcome to Macaw!", "voice": "default"}' \
  --output speech.wav
```

```bash title="Generate raw PCM"
curl -X POST http://localhost:8000/v1/audio/speech \
  -H "Content-Type: application/json" \
  -d '{"model": "kokoro-v1", "input": "Hello!", "voice": "default", "response_format": "pcm"}' \
  --output speech.pcm
```

```python title="Python (OpenAI SDK)"
response = client.audio.speech.create(
    model="kokoro-v1",
    input="Hello, welcome to Macaw!",
    voice="default",
)
response.stream_to_file("output.wav")
```

### Response

The response body is the audio file in the requested format.

| Format | Content-Type | Description |
|---|---|---|
| `wav` | `audio/wav` | WAV with headers (default) |
| `pcm` | `audio/pcm` | Raw PCM 16-bit, 16kHz, mono |

---

## GET /health

Returns the runtime health status.

```bash
curl http://localhost:8000/health
```

```json
{
  "status": "ok"
}
```

---

## Error Responses

All endpoints return standard HTTP error codes with a JSON body:

```json
{
  "error": {
    "message": "Model 'nonexistent' not found",
    "type": "model_not_found",
    "code": 404
  }
}
```

| Status | Meaning |
|---|---|
| `400` | Invalid request (missing fields, bad format) |
| `404` | Model not found |
| `422` | Validation error |
| `500` | Internal server error |
| `503` | Worker unavailable |
