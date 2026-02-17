"""Download models from the HuggingFace Hub.

Uses the huggingface_hub library to download model repositories
with a progress bar and manifest validation after copying.
"""

from __future__ import annotations

import shutil
from pathlib import Path
from typing import TYPE_CHECKING

import yaml

from macaw.config.manifest import MANIFEST_FILENAME, ModelManifest
from macaw.config.settings import get_settings
from macaw.logging import get_logger

if TYPE_CHECKING:
    from macaw.registry.catalog import CatalogEntry

logger = get_logger("registry.downloader")

_DEFAULT_MODELS_DIR = get_settings().worker.models_path


class ModelDownloader:
    """Download models from the HuggingFace Hub to the local directory.

    Flow:
    1. Create model directory at models_dir/<model_name>/
    2. Download full repository from HuggingFace Hub via snapshot_download
    3. Generate macaw.yaml from the manifest embedded in the catalog
    4. Validate manifest after generation
    """

    def __init__(self, models_dir: str | Path | None = None) -> None:
        self._models_dir = Path(models_dir) if models_dir else _DEFAULT_MODELS_DIR

    @property
    def models_dir(self) -> Path:
        """Base models directory."""
        return self._models_dir

    def is_installed(self, model_name: str) -> bool:
        """Check if a model is already installed."""
        model_dir = self._models_dir / model_name
        manifest_path = model_dir / MANIFEST_FILENAME
        return manifest_path.exists()

    def download(
        self,
        entry: CatalogEntry,
        *,
        force: bool = False,
    ) -> Path:
        """Download model from the HuggingFace Hub.

        Args:
            entry: Catalog entry with repository and manifest.
            force: If True, overwrite existing model.

        Returns:
            Path to the installed model directory.

        Raises:
            RuntimeError: If huggingface_hub is not installed.
            FileExistsError: If model is already installed and force=False.
        """
        try:
            from huggingface_hub import snapshot_download
        except ImportError:
            msg = "huggingface_hub is not installed. Install with: pip install huggingface_hub"
            raise RuntimeError(msg) from None

        model_dir = self._models_dir / entry.name

        if model_dir.exists() and not force:
            manifest_path = model_dir / MANIFEST_FILENAME
            if manifest_path.exists():
                msg = (
                    f"Model '{entry.name}' is already installed at {model_dir}. "
                    f"Use --force to reinstall."
                )
                raise FileExistsError(msg)

        # Create parent directory
        self._models_dir.mkdir(parents=True, exist_ok=True)

        # Clean existing directory if force
        if model_dir.exists() and force:
            shutil.rmtree(model_dir)

        logger.info(
            "download_starting",
            model=entry.name,
            repo=entry.repo,
            target=str(model_dir),
        )

        # Download via HuggingFace Hub
        downloaded_path = snapshot_download(
            repo_id=entry.repo,
            local_dir=str(model_dir),
        )

        logger.info("download_complete", model=entry.name, path=downloaded_path)

        # Generate macaw.yaml from the catalog manifest
        self._write_manifest(model_dir, entry)

        # Validate generated manifest
        manifest_path = model_dir / MANIFEST_FILENAME
        ModelManifest.from_yaml_path(manifest_path)

        logger.info("manifest_validated", model=entry.name)

        return model_dir

    def remove(self, model_name: str) -> bool:
        """Remove an installed model.

        Args:
            model_name: Model name to remove.

        Returns:
            True if removed, False if it did not exist.
        """
        model_dir = self._models_dir / model_name
        if not model_dir.exists():
            return False

        shutil.rmtree(model_dir)
        logger.info("model_removed", model=model_name, path=str(model_dir))
        return True

    def _write_manifest(self, model_dir: Path, entry: CatalogEntry) -> None:
        """Write macaw.yaml to the model directory."""
        manifest_data = entry.manifest
        if not manifest_data:
            msg = f"Catalog has no manifest for model '{entry.name}'"
            raise ValueError(msg)

        manifest_path = model_dir / MANIFEST_FILENAME
        manifest_path.write_text(
            yaml.dump(manifest_data, default_flow_style=False, allow_unicode=True),
            encoding="utf-8",
        )
        logger.info("manifest_written", model=entry.name, path=str(manifest_path))
