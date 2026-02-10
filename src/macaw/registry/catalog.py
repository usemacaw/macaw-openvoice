"""Catalog of official Macaw OpenVoice models.

Reads the declarative catalog (catalog.yaml) with models available for download
via HuggingFace Hub. Each entry contains repository, engine, type, and manifest.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml

from macaw.logging import get_logger

logger = get_logger("registry.catalog")

_CATALOG_PATH = Path(__file__).parent / "catalog.yaml"


@dataclass(frozen=True, slots=True)
class CatalogEntry:
    """Model entry in the catalog."""

    name: str
    repo: str
    engine: str
    model_type: str
    architecture: str | None = None
    description: str = ""
    manifest: dict[str, Any] = field(default_factory=dict)


class ModelCatalog:
    """Catalog of official models available for download.

    Reads catalog.yaml and provides lookup by model name.
    """

    def __init__(self, catalog_path: str | Path | None = None) -> None:
        self._path = Path(catalog_path) if catalog_path else _CATALOG_PATH
        self._entries: dict[str, CatalogEntry] = {}

    def load(self) -> None:
        """Load catalog from YAML file."""
        if not self._path.exists():
            msg = f"Catalog not found: {self._path}"
            raise FileNotFoundError(msg)

        raw = self._path.read_text(encoding="utf-8")
        data = yaml.safe_load(raw)

        if not isinstance(data, dict) or "models" not in data:
            msg = f"Invalid catalog: {self._path} (missing 'models' field)"
            raise ValueError(msg)

        models = data["models"]
        if not isinstance(models, dict):
            msg = f"Invalid catalog: {self._path} ('models' must be a mapping)"
            raise ValueError(msg)

        self._entries.clear()
        for name, info in models.items():
            if not isinstance(info, dict):
                logger.warning("catalog_entry_invalid", name=name)
                continue

            entry = CatalogEntry(
                name=str(name),
                repo=str(info.get("repo", "")),
                engine=str(info.get("engine", "")),
                model_type=str(info.get("type", "")),
                architecture=info.get("architecture"),
                description=str(info.get("description", "")),
                manifest=info.get("manifest", {}),
            )
            self._entries[name] = entry

        logger.info("catalog_loaded", models_count=len(self._entries))

    def get(self, model_name: str) -> CatalogEntry | None:
        """Return catalog entry by name, or None if not found."""
        return self._entries.get(model_name)

    def list_models(self) -> list[CatalogEntry]:
        """Return all catalog entries."""
        return list(self._entries.values())

    def has_model(self, model_name: str) -> bool:
        """Check if a model exists in the catalog."""
        return model_name in self._entries
