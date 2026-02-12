# Contributing to Macaw OpenVoice

Thank you for your interest in contributing to Macaw OpenVoice! This guide will help you get started.

## Prerequisites

- **Python 3.11+** (3.12 recommended)
- **[uv](https://docs.astral.sh/uv/)** for virtual environment and dependency management
- **make** for development workflow
- **Git** with conventional commit knowledge

## Development Setup

```bash
# Clone the repository
git clone https://github.com/useMacaw/Macaw-openvoice.git
cd Macaw-openvoice

# Create virtual environment with Python 3.12
uv venv --python 3.12

# Install all dependencies (including dev tools)
uv sync --all-extras

# Verify everything works
make ci
```

## Development Workflow

### Running checks

```bash
make check       # format (ruff) + lint (ruff) + typecheck (mypy strict)
make test-unit   # unit tests only (preferred during development)
make test        # all tests (1600+)
make ci          # full pipeline: format + lint + typecheck + test
```

### Running individual tests

```bash
.venv/bin/python -m pytest tests/unit/test_foo.py::test_bar -q
```

### Generating protobuf stubs

```bash
make proto
```

## Code Style

### Formatting and linting

- **ruff** handles both formatting and linting
- **mypy** runs in strict mode — all code must pass `mypy --strict`
- Run `make check` before committing

### Python conventions

- `from __future__ import annotations` is required in **all** source files
- Use `TYPE_CHECKING` blocks for imports used only in type hints (ruff TCH rules)
- Dataclasses use `frozen=True, slots=True` for immutable value objects
- Domain-specific exceptions in `src/macaw/exceptions.py` — never raise bare `Exception`
- Async-first: all public interfaces are `async`
- Imports are absolute from `macaw.` (e.g., `from macaw.registry import Registry`)

### Naming

- `snake_case` for functions and variables
- `PascalCase` for classes
- Descriptive names over short names (`user_metadata` not `metadata`)

## Testing Guidelines

- Framework: **pytest** with **pytest-asyncio** (`asyncio_mode = "auto"` — no `@pytest.mark.asyncio` needed)
- Test names describe behavior: `test_transcription_fails_when_model_not_found`
- Follow **Arrange-Act-Assert** (AAA) pattern
- Each test tests **one thing**
- Tests must be independent — no shared mutable state between tests
- Use `unittest.mock` for inference engines in unit tests
- Mark integration tests with `@pytest.mark.integration`
- Never commit real audio files — generate test audio in fixtures

### Test structure

```
tests/
  unit/           # Mirrors src/macaw/ structure
  integration/    # Tests requiring external resources
  fixtures/       # Shared test fixtures (audio, manifests)
  conftest.py     # Shared fixtures
```

## Pull Request Process

### Branch naming

Use descriptive branch names:
- `feat/add-paraformer-backend`
- `fix/ring-buffer-force-commit-race`
- `docs/update-websocket-protocol`

### Commit messages

Follow [Conventional Commits](https://www.conventionalcommits.org/):

```
feat: add Paraformer STT backend
fix: resolve race condition in ring buffer force commit
refactor: extract common audio decoding logic
test: add streaming stability test for 30-minute session
docs: update WebSocket protocol reference
```

### PR checklist

Before submitting your PR, ensure:

- [ ] `make check` passes (format + lint + typecheck)
- [ ] `make test-unit` passes with no new failures
- [ ] New code has unit tests covering business logic
- [ ] CHANGELOG.md is updated under `[Unreleased]`
- [ ] No secrets or credentials in committed files
- [ ] PR description explains the "why", not just the "what"

### CI checks

All PRs must pass:
1. **ruff** — formatting and linting
2. **mypy --strict** — type checking
3. **pytest** — unit and integration tests
4. **Build verification** — wheel builds correctly

## Releasing

Releases are triggered by pushing a git tag. The CI validates, builds, publishes to PyPI, and pushes Docker images to GHCR automatically.

### Prerequisites

- Push access to `main` branch
- One-time: [configure PyPI trusted publisher](https://docs.pypi.org/trusted-publishers/adding-a-publisher/) for `useMacaw/Macaw-openvoice`, workflow `release.yml`

### Step-by-step

**1. Decide the version number**

Follow [Semantic Versioning](https://semver.org/):

| Change | Example | Version bump |
|--------|---------|-------------|
| Breaking API change | Remove endpoint, change response format | Major (`1.0.0` → `2.0.0`) |
| New feature, backward-compatible | Add new endpoint, new engine | Minor (`0.1.0` → `0.2.0`) |
| Bug fix, backward-compatible | Fix race condition, typo | Patch (`0.1.0` → `0.1.2`) |

**2. Update version in two files**

Both must match exactly — the CI verifies this.

`pyproject.toml`:
```toml
[project]
version = "0.2.0"
```

`src/macaw/__init__.py`:
```python
__version__ = "0.2.0"
```

**3. Update CHANGELOG.md**

Move entries from `[Unreleased]` to a new versioned section with today's date. Keep the `[Unreleased]` header empty for future changes.

```markdown
## [Unreleased]

## [0.2.0] - 2026-02-15

### Added
- ...

### Fixed
- ...
```

The CI verifies that `CHANGELOG.md` contains an entry matching `[0.2.0]`.

**4. Commit the version bump**

```bash
git add pyproject.toml src/macaw/__init__.py CHANGELOG.md
git commit -m "release: v0.2.0"
```

**5. Create and push the tag**

```bash
git tag v0.2.0
git push origin main
git push origin v0.2.0
```

The tag **must** start with `v` (e.g., `v0.2.0`, not `0.2.0`).

**6. Verify the release**

The tag push triggers the `release.yml` workflow with 3 jobs:

| Job | What it does | Artifacts |
|-----|-------------|-----------|
| `validate` | Lint, typecheck, unit tests | — |
| `release` | Build wheel, create GitHub Release, publish to PyPI | `.whl`, `.tar.gz` on GitHub + PyPI |
| `docker` | Build and push CPU + GPU images to GHCR | `ghcr.io/useMacaw/Macaw-openvoice:0.2.0` |

Monitor at: `https://github.com/useMacaw/Macaw-openvoice/actions`

After completion, verify:
- GitHub Release: `https://github.com/useMacaw/Macaw-openvoice/releases/tag/v0.2.0`
- PyPI: `https://pypi.org/project/Macaw-openvoice/0.2.0/`
- GHCR: `docker pull ghcr.io/useMacaw/Macaw-openvoice:0.2.0`
- GHCR GPU: `docker pull ghcr.io/useMacaw/Macaw-openvoice:0.2.0-gpu`

### What can go wrong

| Error | Cause | Fix |
|-------|-------|-----|
| "Tag vX.Y.Z does not match pyproject.toml" | Version mismatch between tag, `pyproject.toml`, or `__init__.py` | Delete the tag (`git tag -d vX.Y.Z && git push origin :refs/tags/vX.Y.Z`), fix versions, re-tag |
| "CHANGELOG.md does not contain entry for version" | Missing `## [X.Y.Z]` section in CHANGELOG | Same: delete tag, add CHANGELOG entry, re-tag |
| PyPI publish fails | Trusted publisher not configured | Follow [setup guide](https://docs.pypi.org/trusted-publishers/adding-a-publisher/) |
| Docker push fails | GHCR permissions | Ensure repo has `packages: write` permission (set in workflow) |

### Docker image tags

Each release produces 4 Docker tags:

| Tag | Platform | Dockerfile |
|-----|----------|-----------|
| `X.Y.Z` | linux/amd64, linux/arm64 | `Dockerfile` (CPU) |
| `latest` | linux/amd64, linux/arm64 | `Dockerfile` (CPU) |
| `X.Y.Z-gpu` | linux/amd64 | `Dockerfile.gpu` (CUDA 12.4) |
| `latest-gpu` | linux/amd64 | `Dockerfile.gpu` (CUDA 12.4) |

## Adding a New STT Engine

If you're adding a new STT engine (e.g., Paraformer, Wav2Vec2), follow the step-by-step guide in [docs/ADDING_ENGINE.md](docs/ADDING_ENGINE.md). It requires changes to exactly 4 files plus tests — zero changes to the runtime core.

## Architecture

Before making significant changes, familiarize yourself with:
- [Architecture Document](docs/ARCHITECTURE.md) — system design and component interactions
- [PRD](docs/PRD.md) — product requirements and design decisions (ADRs)

## Getting Help

- Open a [GitHub Issue](https://github.com/useMacaw/Macaw-openvoice/issues) for bug reports or feature requests
- Use [GitHub Discussions](https://github.com/useMacaw/Macaw-openvoice/discussions) for questions
- For general inquiries, reach out at [hello@usemacaw.io](mailto:hello@usemacaw.io)
- Visit our website at [usemacaw.io](https://usemacaw.io)

## Code of Conduct

This project follows the [Contributor Covenant Code of Conduct](CODE_OF_CONDUCT.md). By participating, you agree to uphold this code.
