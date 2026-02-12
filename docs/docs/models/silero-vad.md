---
title: Silero VAD
sidebar_position: 5
---

# Silero VAD

Silero VAD (Voice Activity Detection) is the neural speech detector used internally by Macaw OpenVoice. It determines which audio frames contain speech and which are silence, enabling the runtime to process only relevant audio. Silero VAD is not a user-installable model — it is bundled with the runtime and downloaded automatically via `torch.hub`.

:::info Internal component
Silero VAD is not something you `macaw pull`. It is loaded automatically when the runtime starts a streaming session. You configure its behavior through the `vad_sensitivity` setting in `session.configure`.
:::

## How It Works

Macaw uses a **two-stage VAD pipeline** that combines a fast energy pre-filter with the Silero neural classifier:

```
Audio Frame
    │
    ▼
┌──────────────────────┐
│  Energy Pre-Filter   │  ~0.1ms/frame
│  (RMS + Spectral     │
│   Flatness)          │
│                      │
│  Low energy + flat   │──── Silence (skip Silero)
│  spectrum?           │
└──────────┬───────────┘
           │ Non-silence
           ▼
┌──────────────────────┐
│  Silero VAD          │  ~2ms/frame
│  (Neural classifier) │
│                      │
│  Speech probability  │
│  > threshold?        │
└──────────┬───────────┘
           │
           ▼
┌──────────────────────┐
│  Debounce            │
│  (VADDetector)       │
│                      │
│  Confirmed state     │
│  transition?         │
└──────────┬───────────┘
           │
           ▼
    VADEvent (SPEECH_START / SPEECH_END)
```

This two-stage design reduces unnecessary Silero invocations by 60-70% in noisy environments, since obvious silence is filtered out at the energy level without invoking the neural model.

## Stage 1: Energy Pre-Filter

The energy pre-filter (`EnergyPreFilter`) uses two metrics to classify obvious silence:

### RMS Energy (dBFS)

Computes the Root Mean Square energy of the frame and converts to dBFS (decibels relative to full scale). Frames below the energy threshold are candidates for silence.

| Sensitivity | Energy Threshold | Description |
|:-----------:|:----------------:|-------------|
| HIGH | -50 dBFS | Very sensitive — detects whispers |
| NORMAL | -40 dBFS | Normal conversation (default) |
| LOW | -30 dBFS | Noisy environments, call centers |

### Spectral Flatness

After the energy check, the pre-filter computes spectral flatness (ratio of geometric mean to arithmetic mean of the magnitude spectrum). A value above **0.8** indicates a flat spectrum (white noise or silence), while tonal speech typically has low spectral flatness (~0.1-0.5).

A frame is classified as **silence** only when **both** conditions are met:
- RMS energy < threshold (dBFS)
- Spectral flatness > 0.8

**Cost:** ~0.1ms per frame.

## Stage 2: Silero VAD Classifier

Frames that pass the energy pre-filter are sent to the Silero neural classifier (`SileroVADClassifier`). It returns a speech probability between 0.0 and 1.0.

### Speech Probability Thresholds

| Sensitivity | Threshold | Behavior |
|:-----------:|:---------:|----------|
| HIGH | 0.3 | Detects soft speech, whispers — more false positives |
| NORMAL | 0.5 | Balanced for normal conversation (default) |
| LOW | 0.7 | Requires clear speech — fewer false positives, may miss quiet speakers |

A frame is classified as **speech** when `probability > threshold`.

### Frame Processing

- **Expected frame size:** 512 samples (32ms at 16kHz)
- **Large frames:** automatically split into 512-sample sub-frames, processed sequentially (preserving Silero's internal temporal state). Returns the **maximum** probability among sub-frames
- **Sample rate:** 16,000 Hz (required — validated on initialization)

### Model Loading

Silero VAD is **lazy-loaded** on the first call to `get_speech_probability()`:

- Downloaded via `torch.hub.load("snakers4/silero-vad", "silero_vad")`
- Cached by PyTorch's hub mechanism (typically in `~/.cache/torch/hub/`)
- **Thread-safe** — uses `threading.Lock` with double-check locking pattern
- Can be preloaded with `await classifier.preload()` to avoid first-call latency

**Cost:** ~2ms per frame on CPU.

## Stage 3: Debounce (VADDetector)

The `VADDetector` orchestrates both stages and applies debounce to prevent rapid state changes from producing noisy events.

### Debounce Parameters

| Parameter | Default | Description |
|-----------|:-------:|-------------|
| `min_speech_duration_ms` | 250ms | Consecutive speech frames required before emitting `SPEECH_START` |
| `min_silence_duration_ms` | 300ms | Consecutive silence frames required before emitting `SPEECH_END` |
| `max_speech_duration_ms` | 30,000ms | Maximum continuous speech before forcing `SPEECH_END` |

### State Machine

```
                250ms consecutive speech
    SILENCE ──────────────────────────────► SPEAKING
       ▲                                      │
       │                                      │
       │         300ms consecutive silence     │
       ◄──────────────────────────────────────┘
                        OR
                  30s max speech duration
```

### Events

| Event | When |
|-------|------|
| `SPEECH_START` | After `min_speech_duration_ms` of consecutive speech |
| `SPEECH_END` | After `min_silence_duration_ms` of consecutive silence during speech |
| `SPEECH_END` (forced) | After `max_speech_duration_ms` of continuous speech |

Each event includes a `timestamp_ms` computed from total processed samples.

## Configuration

VAD sensitivity is configured per session via the WebSocket `session.configure` command:

```json title="WebSocket session.configure"
{
  "type": "session.configure",
  "model": "faster-whisper-large-v3",
  "vad_sensitivity": "normal"
}
```

Valid values: `"high"`, `"normal"` (default), `"low"`.

Changing the sensitivity adjusts **both** the energy pre-filter threshold and the Silero speech probability threshold simultaneously.

### Sensitivity Guide

| Environment | Recommended | Why |
|-------------|:-----------:|-----|
| Quiet office, banking app | HIGH | Detects soft-spoken customers, whispers |
| Normal conversation | NORMAL | Balanced for typical voice interactions |
| Call center, noisy background | LOW | Reduces false triggers from background noise |

## Performance

| Metric | Value |
|--------|-------|
| Energy pre-filter cost | ~0.1ms/frame |
| Silero classifier cost | ~2ms/frame |
| Total cost (silence frame) | ~0.1ms (Silero skipped) |
| Total cost (speech frame) | ~2.1ms |
| Model memory | ~50 MB |
| GPU required | No |
| False positive reduction | 60-70% in noisy environments |

## Dependencies

Silero VAD requires PyTorch:

```bash
pip install torch
```

PyTorch is not listed as a direct Macaw dependency — it is typically installed as a transitive dependency of the STT or TTS engines (Faster-Whisper, Kokoro). If you are using a minimal installation, ensure `torch` is available.

## Key Design Decisions

- **VAD runs in the runtime, not in the engine.** The Macaw runtime owns the VAD pipeline. Engines receive only speech audio. This ensures consistent behavior across all STT engines.
- **Preprocessing comes before VAD.** Audio must be normalized (DC removal, gain normalization, resample to 16kHz) before reaching the VAD, otherwise Silero's thresholds produce inconsistent results.
- **Never enable engine-internal VAD.** The `vad_filter` in Faster-Whisper's engine config is always `false`. Enabling it would duplicate the VAD work and create conflicts.
- **Energy pre-filter is a performance optimization, not a replacement.** It reduces Silero invocations for obvious silence but never classifies speech on its own. Only Silero can confirm speech.
- **Debounce uses sample counts, not timers.** The debounce counters accumulate actual processed samples, making the timing deterministic regardless of processing speed.
