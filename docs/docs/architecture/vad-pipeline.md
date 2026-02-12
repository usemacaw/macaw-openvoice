---
title: VAD Pipeline
sidebar_position: 3
---

# VAD Pipeline

Macaw runs all audio preprocessing and Voice Activity Detection (VAD) **in the runtime**, not in the engine. This guarantees consistent behavior regardless of which STT engine is active.

:::warning Preprocessing comes before VAD
Audio must be normalized before reaching Silero VAD. Without normalization, VAD thresholds become unreliable across different audio sources and recording conditions.
:::

## Pipeline Overview

```
Raw Audio Input
       │
       ▼
┌──────────────┐
│   Resample   │  16kHz mono
│   (~0.5ms)   │
└──────┬───────┘
       │
       ▼
┌──────────────┐
│  DC Remove   │  Butterworth HPF @ 20Hz
│   (~0.1ms)   │
└──────┬───────┘
       │
       ▼
┌──────────────┐
│    Gain      │  Peak normalize to -3dBFS
│  Normalize   │
│   (~0.1ms)   │
└──────┬───────┘
       │
       ▼
┌──────────────┐
│   Energy     │  RMS + spectral flatness
│  Pre-filter  │  (if silence → skip Silero)
│   (~0.1ms)   │
└──────┬───────┘
       │ (only if energy detected)
       ▼
┌──────────────┐
│  Silero VAD  │  Neural speech probability
│   (~2ms)     │
└──────┬───────┘
       │
       ▼
  VAD Decision
  (SPEECH_START / SPEECH_END)
```

## Stage 1: Resample

Converts input audio to **16kHz mono**, the standard format expected by all STT engines.

| Property | Value |
|----------|-------|
| Method | `scipy.signal.resample_poly` (polyphase filter) |
| Target rate | 16,000 Hz |
| Channel handling | Multi-channel averaged to mono |
| Quality | High-quality polyphase resampling (no aliasing) |

```python title="What it does"
# Input: any sample rate, any channels
# Output: 16kHz mono float32
audio, sample_rate = resample_stage.process(audio, original_sample_rate)
# sample_rate is now 16000
```

## Stage 2: DC Remove

Removes DC offset using a **2nd-order Butterworth high-pass filter** at 20Hz. DC offset is common in low-quality microphones and can bias VAD energy calculations.

| Property | Value |
|----------|-------|
| Filter type | Butterworth high-pass (2nd order) |
| Cutoff frequency | 20 Hz |
| Implementation | `scipy.signal.sosfilt` with cached SOS coefficients |

:::tip Why 20Hz?
Human speech starts around 85Hz (male fundamental) to 255Hz (female fundamental). A 20Hz cutoff removes DC and sub-bass rumble without affecting any speech content.
:::

## Stage 3: Gain Normalize

Peak normalization ensures audio reaches the VAD at a consistent level, regardless of the original recording volume.

| Property | Value |
|----------|-------|
| Method | Peak normalization |
| Target level | -3.0 dBFS (default) |
| Clip protection | Yes — output clamped to [-1.0, 1.0] |

Without gain normalization, a quiet recording might produce energy levels below the VAD threshold, causing missed speech detection. A loud recording might trigger false positives.

## Stage 4: Energy Pre-filter

The energy pre-filter is a **fast, cheap check** (~0.1ms) that gates access to the more expensive Silero VAD (~2ms). If the frame is clearly silence, Silero is never called.

### How It Works

The pre-filter combines two measurements:

1. **RMS energy (dBFS)** — overall loudness of the frame
2. **Spectral flatness** — how "noise-like" vs "tonal" the signal is

```
                          RMS Energy
                              │
              ┌───────────────┼───────────────┐
              │               │               │
         Below threshold   Above threshold    │
              │               │               │
              ▼               ▼               │
          SILENCE        Check spectral       │
         (skip Silero)   flatness             │
                              │               │
                    ┌─────────┼─────────┐     │
                    │                   │     │
               Flatness > 0.8     Flatness ≤ 0.8
               (white noise)      (tonal/speech)
                    │                   │
                    ▼                   ▼
                SILENCE          PASS TO SILERO
               (skip Silero)
```

### Sensitivity Levels

Energy thresholds vary by sensitivity level:

| Sensitivity | Energy Threshold | Effect |
|-------------|:---:|--------|
| `HIGH` | -50 dBFS | Detects very quiet speech. More false positives |
| `NORMAL` | -40 dBFS | Balanced default |
| `LOW` | -30 dBFS | Requires louder speech. Fewer false positives |

**Spectral flatness threshold:** 0.8 (fixed). Values above 0.8 indicate white noise or silence — signals with no tonal content.

## Stage 5: Silero VAD

The Silero VAD model is the final decision-maker. It uses a neural network to compute a **speech probability** for each frame.

| Property | Value |
|----------|-------|
| Model | `snakers4/silero-vad` via `torch.hub` |
| Loading | Lazy (loaded on first use) |
| Thread safety | `threading.Lock` |
| Frame size | 512 samples (32ms at 16kHz) |
| Large frames | Split into 512-sample sub-frames, max probability returned |
| Cost | ~2ms per frame on CPU |

### Speech Probability Thresholds

| Sensitivity | Threshold | Meaning |
|-------------|:---:|---------|
| `HIGH` | 0.3 | Low bar — detects quiet/uncertain speech |
| `NORMAL` | 0.5 | Balanced default |
| `LOW` | 0.7 | High bar — only clear speech triggers |

:::info Sensitivity affects both stages
The sensitivity level controls thresholds in **both** the energy pre-filter and Silero VAD. Setting `HIGH` makes both stages more permissive.
:::

## Debounce and Duration Limits

The VAD detector applies debounce to prevent rapid state toggling:

| Parameter | Default | Description |
|-----------|:---:|-------------|
| `min_speech_duration_ms` | 250ms | Minimum speech before emitting `SPEECH_START` |
| `min_silence_duration_ms` | 300ms | Minimum silence before emitting `SPEECH_END` |
| `max_speech_duration_ms` | 30,000ms | Maximum speech segment (forces `SPEECH_END`) |

### Why max speech duration?

Encoder-decoder models like Whisper have a fixed context window (30 seconds). If speech exceeds this window, the segment is force-ended to trigger transcription before the buffer overflows. The Session Manager handles cross-segment context to maintain continuity.

## VAD Events

The VAD emits two event types:

```python
@dataclass
class VADEvent:
    event_type: str     # "SPEECH_START" or "SPEECH_END"
    timestamp_ms: int   # Monotonic timestamp
```

These events drive state transitions in the [Session Manager](./session-manager):

| Event | Session Transition |
|-------|-------------------|
| `SPEECH_START` | `INIT → ACTIVE`, `SILENCE → ACTIVE`, `HOLD → ACTIVE` |
| `SPEECH_END` | `ACTIVE → SILENCE` |

## Streaming vs Batch

The preprocessing pipeline has two modes:

### Batch (REST API)

Used for file uploads via `POST /v1/audio/transcriptions`:

```python title="AudioPreprocessingPipeline"
# Decodes entire file (WAV/FLAC/OGG), applies all stages, outputs PCM 16-bit WAV
pipeline = AudioPreprocessingPipeline(stages=[resample, dc_remove, gain_normalize])
processed_audio = pipeline.process(uploaded_file)
```

Supported input formats: WAV, FLAC, OGG (via `libsndfile` with `stdlib wave` fallback).

### Streaming (WebSocket)

Used for real-time audio via `WS /v1/realtime`:

```python title="StreamingPreprocessor"
# Processes one frame at a time
# Input: raw PCM int16 bytes
# Output: float32 16kHz mono
preprocessor = StreamingPreprocessor(stages=[resample, dc_remove, gain_normalize])
processed_frame = preprocessor.process_frame(raw_pcm_bytes)
```

Each WebSocket connection gets its own `StreamingPreprocessor` instance to maintain per-connection filter state (DC remove uses stateful IIR filters).

## Configuration

VAD settings can be adjusted per session via the `session.configure` WebSocket command:

```json title="Client → Server"
{
  "type": "session.configure",
  "vad_sensitivity": "high",
  "hot_words": ["Macaw", "OpenVoice"]
}
```

:::warning Engine VAD must be disabled
Always set `vad_filter: false` in the engine manifest. The runtime manages VAD — enabling the engine's built-in VAD (e.g., Faster-Whisper's `vad_filter`) would duplicate the work and cause unpredictable behavior.
:::
