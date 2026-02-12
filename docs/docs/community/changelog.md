---
title: Changelog
sidebar_position: 2
---

# Changelog

All notable changes to Macaw OpenVoice are documented here. This project follows [Semantic Versioning](https://semver.org/) and the [Keep a Changelog](https://keepachangelog.com/) format.

## [Unreleased]

### Added

- Full-duplex STT + TTS on a single WebSocket connection (M9)
- Mute-on-speak mechanism with guaranteed unmute via try/finally
- TTS cancel and auto-cancel of previous synthesis
- `tts.speaking_start` and `tts.speaking_end` WebSocket events
- KokoroBackend with 9-language support and streaming synthesis
- TTS gRPC worker on port 50052
- `POST /v1/audio/speech` REST endpoint (OpenAI-compatible)
- TTS metrics (TTFB, synthesis duration, requests, active sessions)
- Session backpressure controller (rate limit at 1.2x real-time)
- `session.configure` command for dynamic session settings
- `input_audio_buffer.commit` command for manual buffer commit
- `macaw pull` and `macaw remove` CLI commands
- `macaw ps` command to list models on a running server

### Changed

- WebSocket protocol extended with TTS commands and events
- Session Manager now supports mute/unmute for full-duplex
- Scheduler metrics made optional (graceful degradation without prometheus_client)

---

## Milestone History

| Milestone | Description | Status |
|-----------|-------------|:---:|
| M1 | FastAPI server + health endpoint | Done |
| M2 | Model Registry + manifests | Done |
| M3 | Scheduler + priority queue + cancellation | Done |
| M4 | gRPC STT workers (Faster-Whisper) | Done |
| M5 | Streaming STT via WebSocket | Done |
| M6 | Session Manager (state machine, ring buffer, WAL) | Done |
| M7 | WeNet CTC engine + pipeline adaptation | Done |
| M8 | TTS engine (Kokoro) + REST endpoint | Done |
| M9 | Full-duplex STT + TTS | Done |

:::info
For the complete commit history, see the [GitHub repository](https://github.com/macaw-voice/macaw-openvoice).
:::
