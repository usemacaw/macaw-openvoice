---
title: Scheduling
sidebar_position: 4
---

# Scheduling

The Scheduler routes **batch** (REST API) requests to gRPC workers. It provides priority queuing, request cancellation, dynamic batching, and latency tracking.

:::warning Streaming bypasses the Scheduler
WebSocket streaming uses `StreamingGRPCClient` directly. The Scheduler and its priority queue are **only** for REST batch requests (`POST /v1/audio/transcriptions`, `/translations`, `/speech`).
:::

## Components

```
                ┌─────────────────────────────────────────────┐
                │                 Scheduler                   │
                │                                             │
  REST Request ─┤  ┌──────────────────┐  ┌────────────────┐  │
                │  │ BatchAccumulator  │  │ PriorityQueue  │  │
                │  │                   │──▶│                │  │
                │  │ Group by model    │  │ REALTIME first │  │
                │  │ Flush: 50ms / 8   │  │ FIFO within    │  │
                │  └──────────────────┘  │ level          │  │
                │                        └───────┬────────┘  │
                │                                │           │
                │  ┌──────────────────┐  ┌───────▼────────┐  │
                │  │ CancellationMgr  │  │ Dispatch Loop  │  │
                │  │                   │  │                │  │
                │  │ Queue + in-flight │  │ gRPC channel   │  │
                │  │ Cancel via gRPC   │  │ pool           │  │
                │  └──────────────────┘  └───────┬────────┘  │
                │                                │           │
                │  ┌──────────────────┐          │           │
                │  │ LatencyTracker   │◀─────────┘           │
                │  │                   │                      │
                │  │ 4 phases, 60s TTL│                      │
                │  └──────────────────┘                      │
                └─────────────────────────────────────────────┘
                                         │
                                         ▼
                                   gRPC Workers
```

## Priority Queue

The queue has **two priority levels** with FIFO ordering within each level:

| Level | Value | Use Case |
|-------|:---:|----------|
| `REALTIME` | 0 | High-priority requests |
| `BATCH` | 1 | Standard file transcriptions |

### Aging

To prevent starvation, `BATCH` requests that have been queued for more than **30 seconds** are automatically promoted to `REALTIME` priority. The `aging_promotions_total` metric tracks how often this occurs.

### Request Structure

Each queued request carries:

```python
@dataclass
class ScheduledRequest:
    request_id: str
    priority: Priority           # REALTIME or BATCH
    audio_data: bytes
    model: str
    cancel_event: asyncio.Event  # Set to cancel
    result_future: asyncio.Future # Resolved when complete
    enqueue_time: float          # For aging calculation
```

## Batch Accumulator

The BatchAccumulator groups `BATCH` requests by model to improve GPU utilization:

| Parameter | Value | Description |
|-----------|:---:|-------------|
| Flush timer | 50ms | Maximum wait before flushing a partial batch |
| Max batch size | 8 | Maximum requests per batch |
| Model grouping | Per-model | Different models are batched separately |

### Flush Triggers

A batch is flushed (sent to the queue) when **any** of these conditions is met:

1. **Timer expires** — 50ms since the first request in the batch
2. **Batch full** — 8 requests accumulated
3. **Model mismatch** — new request targets a different model

:::info REALTIME bypasses batching
Only `BATCH` priority requests go through the accumulator. `REALTIME` requests are sent directly to the priority queue.
:::

### Flush Lifecycle

The flush callback (`_dispatch_batch`) is fired by an asyncio timer. If the Scheduler stops before the timer fires, `stop()` performs a manual flush to avoid losing queued requests.

## Cancellation Manager

The CancellationManager handles cancellation for both **queued** and **in-flight** requests.

### Cancellation Flow

```
cancel(request_id)
       │
       ├─── Request in queue?
       │         │
       │         ▼
       │    Set cancel_event
       │    Remove from queue
       │    Remove from tracking
       │
       └─── Request in-flight?
                  │
                  ▼
             Set cancel_event
             Send gRPC Cancel RPC to worker (100ms timeout)
             Remove from tracking
             (best-effort — cannot interrupt CUDA kernels)
```

| Property | Value |
|----------|-------|
| Queue cancel | Immediate — request removed from queue |
| In-flight cancel | Best-effort — gRPC `Cancel` RPC with 100ms timeout |
| Idempotent | Yes — cancelling an already-cancelled request is a no-op |
| Tracking | Entry removed on cancel. `unregister()` is no-op if already cancelled |

### REST API

Cancellation is exposed via the REST endpoint:

```bash title="Cancel a request"
curl -X POST http://localhost:8000/v1/audio/transcriptions/{request_id}/cancel
```

```json title="Response"
{
  "request_id": "req_abc123",
  "cancelled": true
}
```

## Dispatch Loop

The dispatch loop runs as a background asyncio task and processes the priority queue:

1. Dequeue next request (REALTIME first, then BATCH, FIFO within each)
2. Check if request was cancelled (skip if so)
3. Acquire gRPC channel from the pool
4. Send `TranscribeFile` RPC to worker
5. Track latency phases
6. Resolve the `result_future` with the transcription result
7. Apply postprocessing (ITN) if enabled

### Timeouts

Request timeout is calculated dynamically:

```
timeout = max(30s, audio_duration_estimate × 2.0)
```

This ensures long audio files get proportionally more time while maintaining a reasonable minimum.

### Graceful Shutdown

When `stop()` is called:

1. Flush any pending batches in the BatchAccumulator
2. Signal the dispatch loop to stop
3. Wait up to **10 seconds** for in-flight requests to complete
4. Cancel remaining requests

## Streaming gRPC Client

For WebSocket streaming (which bypasses the Scheduler), Macaw uses `StreamingGRPCClient` with a `StreamHandle` abstraction:

```python title="StreamHandle API"
handle = await client.open_stream(model, session_id, language)

# Send audio frames
await handle.send_frame(audio_data, is_last=False)

# Receive transcript events
async for event in handle.receive_events():
    # TranscriptSegment with text, is_final, confidence, etc.
    ...

# Close gracefully
await handle.close()  # Sends is_last=True + done_writing()

# Or cancel
await handle.cancel()  # Target: ≤50ms
```

### gRPC Keepalive

Aggressive keepalive settings prevent stream drops:

| Parameter | Value |
|-----------|-------|
| `keepalive_time` | 10s |
| `keepalive_timeout` | 5s |

## Latency Tracker

The LatencyTracker measures request duration across 4 phases:

```
  start()        dequeued()      grpc_started()     complete()
     │               │                │                 │
     ▼               ▼                ▼                 ▼
┌─────────┐    ┌───────────┐    ┌──────────┐    ┌───────────┐
│ Enqueue │───▶│ Queue Wait│───▶│ gRPC Time│───▶│   Done    │
└─────────┘    └───────────┘    └──────────┘    └───────────┘
     │                                               │
     └──────────── total_time ──────────────────────┘
```

| Phase | Measurement |
|-------|-------------|
| `queue_wait` | Time spent waiting in the priority queue |
| `grpc_time` | Time spent in the gRPC call to the worker |
| `total_time` | End-to-end from enqueue to completion |

### TTL

Entries expire after **60 seconds**. `cleanup()` runs periodically to remove entries for requests that never completed (e.g., cancelled, timed out).

## Metrics

### Scheduler Metrics

| Metric | Type | Description |
|--------|------|-------------|
| `scheduler_queue_depth` | Gauge | Current queue depth by priority level |
| `scheduler_queue_wait_seconds` | Histogram | Time spent in queue |
| `scheduler_grpc_duration_seconds` | Histogram | gRPC call duration |
| `scheduler_cancel_latency_seconds` | Histogram | Time to cancel a request |
| `scheduler_batch_size` | Histogram | Batch sizes dispatched |
| `scheduler_requests_total` | Counter | Total requests by status (completed, cancelled, failed) |
| `scheduler_aging_promotions_total` | Counter | BATCH → REALTIME promotions |

### TTS Metrics

| Metric | Type | Description |
|--------|------|-------------|
| `tts_ttfb_seconds` | Histogram | Time to first audio byte |
| `tts_synthesis_duration_seconds` | Histogram | Total synthesis time |
| `tts_requests_total` | Counter | Total TTS requests |
| `tts_active_sessions` | Gauge | Currently active TTS sessions |

:::tip Metrics are optional
All metrics use `try/except ImportError` with a `HAS_METRICS` flag. If `prometheus_client` is not installed, metrics are silently skipped. Always check `if metric is not None` before observing.
:::
