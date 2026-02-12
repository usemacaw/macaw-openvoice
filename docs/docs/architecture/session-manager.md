---
title: Session Manager
sidebar_position: 2
---

# Session Manager

The Session Manager is the core component for **streaming STT**. It coordinates audio buffering, speech detection, worker communication, and crash recovery for each WebSocket connection.

:::info STT only
The Session Manager is used exclusively for streaming STT. TTS is stateless per request — each `tts.speak` call is independent.
:::

## State Machine

Each streaming session progresses through a 6-state finite state machine. Transitions are validated — invalid transitions raise `InvalidTransitionError`, and `CLOSED` is terminal.

```
                     ┌─────────────────────────┐
                     │                         │
                     ▼                         │
    ┌──────┐    ┌────────┐    ┌─────────┐    ┌──────┐
    │ INIT │───▶│ ACTIVE │───▶│ SILENCE │───▶│ HOLD │
    └──┬───┘    └───┬────┘    └────┬────┘    └──┬───┘
       │            │              │             │
       │            │              │         ┌───┴────┐
       │            │              └────────▶│CLOSING │
       │            │                        └───┬────┘
       │            │                            │
       ▼            ▼                            ▼
    ┌──────────────────────────────────────────────┐
    │                   CLOSED                     │
    └──────────────────────────────────────────────┘
```

### States

| State | Description | Behavior |
|-------|-------------|----------|
| `INIT` | Session created, waiting for first speech | Frames preprocessed but not sent to worker |
| `ACTIVE` | Speech detected, actively transcribing | Frames written to ring buffer and sent to gRPC worker |
| `SILENCE` | Speech ended, waiting for next speech | Final transcript emitted, worker stream may close |
| `HOLD` | Extended silence, conserving resources | Frames **not** sent to worker (saves GPU). Worker stream closed |
| `CLOSING` | Graceful shutdown in progress | Flushing remaining data, preparing to close |
| `CLOSED` | Terminal state | No transitions allowed. Session resources released |

### Timeouts

Each state has a configurable timeout that triggers an automatic transition:

| State | Default Timeout | Transition Target |
|-------|:-:|---|
| `INIT` | 30s | `CLOSED` (no speech detected) |
| `SILENCE` | 30s | `HOLD` (extended silence) |
| `HOLD` | 300s (5 min) | `CLOSING` (session idle too long) |
| `CLOSING` | 2s | `CLOSED` (flush complete) |

### Triggers

| Trigger | Transition |
|---------|-----------|
| `SPEECH_START` (VAD) | `INIT → ACTIVE` or `SILENCE → ACTIVE` or `HOLD → ACTIVE` |
| `SPEECH_END` (VAD) | `ACTIVE → SILENCE` |
| Silence timeout | `SILENCE → HOLD` |
| Hold timeout | `HOLD → CLOSING` |
| `session.close` command | Any → `CLOSING → CLOSED` |
| Init timeout | `INIT → CLOSED` |

## Ring Buffer

The ring buffer is a **pre-allocated circular buffer** that stores audio frames during streaming. It is designed for zero allocations during operation.

### Specifications

| Property | Value |
|----------|-------|
| Default capacity | 1,920,000 bytes (60s at 16kHz, 16-bit) |
| Allocation | Pre-allocated at session start |
| Offset tracking | Absolute (`total_written`), monotonically increasing |
| Overwrite protection | Read fence prevents overwriting uncommitted data |
| Force commit threshold | 90% of capacity |

### Read Fence

The read fence (`_read_fence`) marks the boundary between committed and uncommitted data:

```
┌──────────────────────────────────────────────────────┐
│ Ring Buffer                                          │
│                                                      │
│  [committed]  │  [uncommitted]  │  [free space]      │
│               ▲                 ▲                     │
│          read_fence       write_pos                   │
│                                                      │
│  ◀── safe to overwrite    never overwrite ──▶        │
└──────────────────────────────────────────────────────┘
```

:::warning
Never overwrite data past `last_committed_offset` — this data is needed for recovery. If a write would overwrite uncommitted data, `BufferOverrunError` is raised.
:::

### Force Commit

When uncommitted data exceeds **90%** of buffer capacity, the ring buffer triggers a force commit:

1. The `on_force_commit` callback fires **synchronously** from `write()`
2. The callback sets a `_force_commit_pending` flag
3. `process_frame()` (async) checks this flag and commits the segment
4. This prevents buffer overrun while keeping the write path non-blocking

## WAL (Write-Ahead Log)

The WAL provides crash recovery using an **in-memory, single-record, overwrite** strategy.

### Checkpoint Structure

```python
@dataclass(frozen=True, slots=True)
class WALCheckpoint:
    segment_id: int       # Current speech segment
    buffer_offset: int    # Ring buffer position
    timestamp_ms: int     # Monotonic timestamp (never wall-clock)
```

:::tip Why monotonic time?
The WAL uses `time.monotonic()` instead of `time.time()`. This ensures checkpoint consistency even if the system clock is adjusted (NTP sync, DST changes).
:::

### Atomicity

WAL updates are atomic via Python reference assignment within the single asyncio event loop. No locks are needed — the GIL and single-threaded event loop guarantee consistency.

## Recovery

When a gRPC worker crashes (detected via stream break), the Session Manager recovers automatically:

```
Worker crash detected (gRPC stream break)
         │
         ▼
  ┌─────────────────────┐
  │ Set _recovering flag │  (prevents recursion)
  └──────────┬──────────┘
             │
             ▼
  ┌─────────────────────┐
  │ Open new gRPC stream │  (to restarted worker)
  └──────────┬──────────┘
             │
             ▼
  ┌─────────────────────┐
  │ Read WAL checkpoint  │  (get segment_id, buffer_offset)
  └──────────┬──────────┘
             │
             ▼
  ┌─────────────────────────────┐
  │ Resend uncommitted data     │  (ring_buffer.get_uncommitted())
  │ from ring buffer            │
  └──────────┬──────────────────┘
             │
             ▼
  ┌─────────────────────┐
  │ Resume normal flow   │  (clear _recovering flag)
  └─────────────────────┘
```

| Property | Value |
|----------|-------|
| Recovery timeout | 10s |
| Anti-recursion | `_recovering` flag prevents nested recovery attempts |
| Data guarantee | Only uncommitted data is resent (no duplicates) |
| Detection method | gRPC bidirectional stream break |

## Backpressure

The backpressure controller prevents the client from overwhelming the system when audio arrives faster than real-time.

### Thresholds

| Parameter | Value |
|-----------|-------|
| Rate limit | 1.2x real-time |
| Max backlog | 10s of audio |
| Burst detection window | 5s sliding window |
| Rate limit cooldown | 1s between emissions |
| Minimum wall-clock before checks | 0.5s |

### Actions

When thresholds are exceeded, the backpressure controller emits one of two actions:

| Action | Event | Description |
|--------|-------|-------------|
| `RateLimitAction` | `session.rate_limit` | Client should slow down. Includes `delay_ms` hint |
| `FramesDroppedAction` | `session.frames_dropped` | Frames were dropped. Includes `dropped_ms` |

## Mute-on-Speak

For full-duplex operation, the Session Manager supports muting STT while TTS is active:

```python
# In the TTS speak task (simplified)
try:
    session.mute()      # STT frames dropped
    # ... stream TTS audio to client ...
finally:
    session.unmute()    # STT always resumes, even on error
```

When muted:
- Incoming audio frames are **dropped** without processing
- The `stt_muted_frames_total` metric is incremented
- Unmute is **guaranteed** via `try/finally` — even if TTS crashes

## Metrics

The Session Manager exposes 9 Prometheus metrics (optional — graceful degradation if `prometheus_client` is not installed):

| Metric | Type | Description |
|--------|------|-------------|
| `stt_ttfb_seconds` | Histogram | Time to first byte (first partial transcript) |
| `stt_final_delay_seconds` | Histogram | Time from speech end to final transcript |
| `stt_active_sessions` | Gauge | Currently active streaming sessions |
| `stt_vad_events_total` | Counter | VAD events by type (speech_start, speech_end) |
| `stt_session_duration_seconds` | Histogram | Total session duration |
| `stt_segments_force_committed_total` | Counter | Ring buffer force commits |
| `stt_confidence_avg` | Histogram | Average transcript confidence |
| `stt_worker_recoveries_total` | Counter | Worker crash recoveries |
| `stt_muted_frames_total` | Counter | Frames dropped due to mute |

## Pipeline Adaptation

The `StreamingSession` adapts its behavior based on the engine's `architecture` field:

### Encoder-Decoder (Whisper)

- **LocalAgreement** — compares tokens across consecutive inference passes. Only tokens confirmed by `min_confirm_passes` (default: 2) passes are emitted as partials. `flush()` on speech end emits remaining tokens as final
- **Cross-segment context** — last 224 tokens (half of Whisper's 448 context window) from the previous final are used as `initial_prompt` for the next segment

### CTC (WeNet)

- **Native partials** — CTC produces real-time partial transcripts directly
- **No LocalAgreement** — not needed, partials are native
- **No cross-segment context** — CTC does not support `initial_prompt`
- Hot words with native support use the engine's built-in mechanism
