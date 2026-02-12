---
title: Batch Transcription
sidebar_position: 1
---

# Batch Transcription

Macaw's REST API is **OpenAI-compatible** — you can use the official OpenAI SDK or any HTTP client to transcribe and translate audio files.

## Transcription

### Using curl

```bash title="Basic transcription"
curl -X POST http://localhost:8000/v1/audio/transcriptions \
  -F file=@meeting.wav \
  -F model=faster-whisper-large-v3
```

```json title="Response"
{
  "text": "Hello, welcome to the meeting. Let's get started."
}
```

### Using the OpenAI SDK

```python title="Python"
from openai import OpenAI

client = OpenAI(
    base_url="http://localhost:8000/v1",
    api_key="not-needed"  # Macaw doesn't require auth
)

with open("meeting.wav", "rb") as f:
    transcript = client.audio.transcriptions.create(
        model="faster-whisper-large-v3",
        file=f
    )

print(transcript.text)
```

:::tip Drop-in replacement
Since Macaw implements the OpenAI Audio API, switching from OpenAI's hosted API is a one-line change — just update the `base_url`.
:::

## Request Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `file` | binary | *required* | Audio file (WAV, MP3, FLAC, OGG, WebM) |
| `model` | string | *required* | Model name (e.g., `faster-whisper-large-v3`) |
| `language` | string | auto | ISO 639-1 language code (e.g., `en`, `pt`, `es`) |
| `prompt` | string | — | Context hint for the model (hot words, domain terms) |
| `response_format` | string | `json` | Output format (see below) |
| `temperature` | float | `0.0` | Sampling temperature (0.0 = deterministic) |
| `itn` | boolean | `true` | Apply Inverse Text Normalization |

## Response Formats

### `json` (default)

```bash
curl -X POST http://localhost:8000/v1/audio/transcriptions \
  -F file=@audio.wav \
  -F model=faster-whisper-large-v3 \
  -F response_format=json
```

```json
{
  "text": "The total is one hundred and fifty dollars."
}
```

### `verbose_json`

Includes segment-level detail with timestamps and confidence scores:

```bash
curl -X POST http://localhost:8000/v1/audio/transcriptions \
  -F file=@audio.wav \
  -F model=faster-whisper-large-v3 \
  -F response_format=verbose_json
```

```json
{
  "text": "The total is one hundred and fifty dollars.",
  "segments": [
    {
      "id": 0,
      "start": 0.0,
      "end": 2.5,
      "text": "The total is one hundred and fifty dollars.",
      "avg_logprob": -0.15,
      "no_speech_prob": 0.02
    }
  ],
  "language": "en",
  "duration": 2.5
}
```

### `text`

Returns plain text without JSON wrapping:

```bash
curl -X POST http://localhost:8000/v1/audio/transcriptions \
  -F file=@audio.wav \
  -F model=faster-whisper-large-v3 \
  -F response_format=text
```

```
The total is one hundred and fifty dollars.
```

### `srt`

SubRip subtitle format:

```
1
00:00:00,000 --> 00:00:02,500
The total is one hundred and fifty dollars.
```

### `vtt`

WebVTT subtitle format:

```
WEBVTT

00:00:00.000 --> 00:00:02.500
The total is one hundred and fifty dollars.
```

## Translation

The translation endpoint translates audio from **any supported language to English**:

```bash title="Translate Portuguese audio to English"
curl -X POST http://localhost:8000/v1/audio/translations \
  -F file=@reuniao.wav \
  -F model=faster-whisper-large-v3
```

```json
{
  "text": "Hello, welcome to the meeting. Let's get started."
}
```

:::info Translation target
Translation always produces English output. This matches the OpenAI API behavior. The source language is detected automatically.
:::

The translation endpoint accepts the same parameters as transcription (except `language`, which is ignored since the output is always English).

## Inverse Text Normalization (ITN)

By default, Macaw applies ITN to final transcripts, converting spoken forms to written forms:

| Spoken | Written (with ITN) |
|--------|-------------------|
| "one hundred and fifty dollars" | "$150" |
| "january twenty third twenty twenty five" | "January 23, 2025" |
| "five five five one two three four" | "555-1234" |

To disable ITN:

```bash
curl -X POST http://localhost:8000/v1/audio/transcriptions \
  -F file=@audio.wav \
  -F model=faster-whisper-large-v3 \
  -F itn=false
```

:::tip ITN is fail-open
ITN uses NeMo Text Processing. If NeMo is not installed or fails, the raw text passes through unchanged — no errors are raised.
:::

## Cancellation

Long-running requests can be cancelled using the request ID:

```bash title="Start a transcription"
# The request_id is returned in the response headers
curl -v -X POST http://localhost:8000/v1/audio/transcriptions \
  -F file=@long_recording.wav \
  -F model=faster-whisper-large-v3
```

```bash title="Cancel it"
curl -X POST http://localhost:8000/v1/audio/transcriptions/req_abc123/cancel
```

```json title="Response"
{
  "request_id": "req_abc123",
  "cancelled": true
}
```

Cancellation is **idempotent** — cancelling an already-completed or already-cancelled request returns `cancelled: false`.

## Supported Audio Formats

| Format | MIME Type | Notes |
|--------|-----------|-------|
| WAV | `audio/wav` | Preferred — no transcoding needed |
| MP3 | `audio/mpeg` | Decoded automatically |
| FLAC | `audio/flac` | Lossless, good for archival |
| OGG | `audio/ogg` | Opus/Vorbis codec |
| WebM | `audio/webm` | Common from browsers |

:::warning Preprocessing is automatic
All audio is automatically resampled to 16kHz mono, DC-filtered, and gain-normalized before reaching the engine. You don't need to preprocess your files.
:::

## Error Responses

| Status | Description |
|:---:|-------------|
| 400 | Invalid request (missing file, unsupported format, unknown model) |
| 413 | File too large |
| 503 | No workers available for the requested model |
| 500 | Internal error during transcription |

```json title="Error response"
{
  "error": {
    "message": "Model 'unknown-model' not found",
    "type": "invalid_request_error",
    "code": "model_not_found"
  }
}
```

## Next Steps

| Goal | Guide |
|------|-------|
| Real-time streaming transcription | [Streaming STT](./streaming-stt) |
| Text-to-speech synthesis | [REST API - Speech](../api-reference/rest-api#post-v1audiospeech) |
| Full-duplex voice interaction | [Full-Duplex](./full-duplex) |
