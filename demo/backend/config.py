"""Configuration for the demo backend."""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from collections.abc import Sequence


def _parse_origins(raw: str | None) -> list[str]:
    if not raw:
        return ["http://localhost:3000"]
    parts: Sequence[str] = [item.strip() for item in raw.split(",") if item.strip()]
    return list(parts) or ["http://localhost:3000"]


@dataclass(slots=True)
class DemoConfig:
    """Configuration parameters loaded from environment variables."""

    models_dir: Path = field(
        default_factory=lambda: Path(os.getenv("DEMO_MODELS_DIR", "~/.macaw/models")).expanduser(),
    )
    host: str = field(default_factory=lambda: os.getenv("DEMO_HOST", "127.0.0.1"))
    port: int = field(default_factory=lambda: int(os.getenv("DEMO_PORT", "9000")))
    worker_base_port: int = field(
        default_factory=lambda: int(os.getenv("DEMO_WORKER_BASE_PORT", "55051")),
    )
    aging_threshold_s: float = field(
        default_factory=lambda: float(os.getenv("DEMO_AGING_THRESHOLD_S", "30.0")),
    )
    batch_accumulate_ms: float = field(
        default_factory=lambda: float(os.getenv("DEMO_BATCH_ACCUMULATE_MS", "75.0")),
    )
    batch_max_size: int = field(
        default_factory=lambda: int(os.getenv("DEMO_BATCH_MAX_SIZE", "8")),
    )
    allowed_origins: list[str] = field(
        default_factory=lambda: _parse_origins(os.getenv("DEMO_ALLOWED_ORIGINS")),
    )
