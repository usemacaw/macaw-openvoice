---
title: Contributing
sidebar_position: 1
---

# Contributing

Thank you for your interest in contributing to Macaw OpenVoice! This guide covers everything you need to set up a development environment, run tests, and submit changes.

## Development Setup

### Prerequisites

| Tool | Version | Purpose |
|------|---------|---------|
| Python | 3.12+ | Runtime (project requires >=3.11) |
| uv | latest | Fast Python package manager |
| make | any | Build automation |
| git | any | Version control |

### Clone and Install

```bash title="Clone the repository"
git clone https://github.com/macaw-voice/macaw-openvoice.git
cd macaw-openvoice
```

```bash title="Create virtual environment and install dependencies"
uv venv --python 3.12
source .venv/bin/activate
uv pip install -e ".[dev]"
```

### Verify Setup

```bash title="Run the full check pipeline"
make check   # format + lint + typecheck
make test-unit  # unit tests
```

If both pass, you're ready to contribute.

## Development Workflow

### 1. Create a Branch

```bash
git checkout -b feat/my-feature
```

Branch naming follows conventional prefixes:

| Prefix | Use Case |
|--------|----------|
| `feat/` | New features |
| `fix/` | Bug fixes |
| `refactor/` | Code restructuring |
| `test/` | Test additions/improvements |
| `docs/` | Documentation changes |

### 2. Make Changes

Follow the [Code Style](#code-style) guidelines below.

### 3. Run Checks

```bash title="During development — run unit tests (fast)"
make test-unit
```

```bash title="Before committing — run everything"
make ci   # format + lint + typecheck + all tests
```

### 4. Commit

Commits follow [Conventional Commits](https://www.conventionalcommits.org/):

```bash
git commit -m "feat: add support for Paraformer streaming"
git commit -m "fix: prevent ring buffer overrun on force commit"
git commit -m "test: add integration tests for WeNet CTC partials"
```

| Type | Description |
|------|-------------|
| `feat:` | New feature |
| `fix:` | Bug fix |
| `refactor:` | Code change that neither fixes a bug nor adds a feature |
| `test:` | Adding or updating tests |
| `docs:` | Documentation only |

### 5. Submit a Pull Request

Push your branch and open a PR against `main`. Include:

- Clear description of what changed and why
- Reference to any related issues
- Test evidence (new tests or existing tests passing)

## Make Targets

All `make` targets use `.venv/bin/` automatically — no need to activate the venv manually.

| Target | Description |
|--------|-------------|
| `make check` | Format + lint + typecheck |
| `make test` | All tests |
| `make test-unit` | Unit tests only (use during development) |
| `make test-integration` | Integration tests only |
| `make test-fast` | All tests except `@pytest.mark.slow` |
| `make ci` | Full pipeline: format + lint + typecheck + test |
| `make proto` | Regenerate protobuf stubs |

:::tip Use `make test-unit` during development
The full test suite includes integration tests that may require models and GPU. Unit tests run in seconds and catch most issues.
:::

## Code Style

### General Rules

- **Python 3.12** with strict mypy typing
- **Async-first** — all public interfaces are `async`
- **Formatting** — ruff (format + lint)
- **Imports** — absolute from `macaw.` (e.g., `from macaw.registry import Registry`)
- **Naming** — `snake_case` for functions/variables, `PascalCase` for classes
- **Docstrings** — only on public interfaces (ABCs) and non-obvious functions
- **No obvious comments** — code should be self-explanatory
- **Errors** — typed domain exceptions, never generic `Exception`

### Testing Guidelines

| Rule | Details |
|------|---------|
| Framework | pytest + pytest-asyncio with `asyncio_mode = "auto"` |
| No `@pytest.mark.asyncio` | Auto mode handles it |
| Async HTTP tests | `httpx.AsyncClient` with `ASGITransport` |
| Error handler tests | `ASGITransport(raise_app_exceptions=False)` |
| Fixtures | `tests/conftest.py` (auto-generated sine tones) |
| Mocks | `unittest.mock` for inference engines |
| Integration tests | `@pytest.mark.integration` marker |
| Pattern | Arrange-Act-Assert |

### Running Individual Tests

```bash title="Run a specific test"
.venv/bin/python -m pytest tests/unit/test_foo.py::test_bar -q
```

```bash title="Run with verbose output"
.venv/bin/python -m pytest tests/unit/test_foo.py -v
```

## Project Structure

```
src/macaw/
├── server/           # FastAPI — REST + WebSocket endpoints
│   └── routes/       # transcriptions, translations, speech, health, realtime
├── scheduler/        # Priority queue, cancellation, batching, latency tracking
├── registry/         # Model Registry (macaw.yaml, lifecycle)
├── workers/          # Subprocess gRPC management
│   ├── stt/          # STTBackend interface + implementations
│   └── tts/          # TTSBackend interface + implementations
├── preprocessing/    # Audio pipeline (resample, DC remove, gain normalize)
├── postprocessing/   # Text pipeline (ITN via NeMo, fail-open)
├── vad/              # Voice Activity Detection (energy + Silero)
├── session/          # Session Manager (state machine, ring buffer, WAL)
├── cli/              # CLI commands (click)
└── proto/            # gRPC protobuf definitions
```

```
tests/
├── unit/             # Fast, no I/O, mocked dependencies
├── integration/      # Real dependencies, may need GPU/models
└── conftest.py       # Shared fixtures
```

## Common Pitfalls

These are the most common issues contributors encounter. Read these before diving into the code:

:::warning Things that will bite you
- **gRPC streams are the heartbeat.** Don't implement separate health check polling — the stream break is the crash detection.
- **Ring buffer has a read fence.** Never overwrite data past `last_committed_offset`.
- **ITN only on `transcript.final`.** Never apply ITN to partials.
- **Preprocessing before VAD.** Audio must be normalized before Silero VAD.
- **`vad_filter: false` in manifests.** Runtime handles VAD, not the engine.
- **Session Manager is STT only.** TTS is stateless per request.
- **LocalAgreement is for encoder-decoder only.** CTC has native partials.
- **Streaming bypasses the Scheduler.** WebSocket uses `StreamingGRPCClient` directly.
:::

## Getting Help

- Open an issue on [GitHub](https://github.com/macaw-voice/macaw-openvoice/issues) for bugs and feature requests
- Check existing issues and PRs before creating duplicates
- For architecture questions, review the [Architecture Overview](../architecture/overview)
