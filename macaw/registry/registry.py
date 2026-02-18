"""Model Registry — discovers and provides installed model manifests."""

from __future__ import annotations

from pathlib import Path

from macaw.config.manifest import MANIFEST_FILENAME, ModelManifest
from macaw.engines import ENGINE_PACKAGE
from macaw.exceptions import ManifestParseError, ManifestValidationError, ModelNotFoundError
from macaw.logging import get_logger

logger = get_logger("registry")


class ModelRegistry:
    """Discover and provide installed model manifests.

    Scans a models directory for macaw.yaml in each subdirectory.
    Does not manage lifecycle (load/unload) — only reports which models
    exist and their configurations.
    """

    def __init__(self, models_dir: str | Path) -> None:
        self._models_dir = Path(models_dir)
        self._manifests: dict[str, ModelManifest] = {}
        self._model_paths: dict[str, Path] = {}

    async def scan(self) -> None:
        """Scan models_dir and load all manifests.

        For each subdirectory containing macaw.yaml:
        1. Parse with ModelManifest.from_yaml_path()
        2. Index by manifest.name (not by directory name)
        3. Ignore subdirectories without macaw.yaml (log debug)
        4. Ignore invalid manifests (log error, do not crash)
        """
        self._manifests.clear()
        self._model_paths.clear()

        if not self._models_dir.exists():
            logger.warning("models_dir_not_found", path=str(self._models_dir))
            return

        for subdir in sorted(self._models_dir.iterdir()):
            if not subdir.is_dir():
                continue

            manifest_path = subdir / MANIFEST_FILENAME
            if not manifest_path.exists():
                logger.debug("no_manifest", dir=str(subdir))
                continue

            try:
                manifest = ModelManifest.from_yaml_path(manifest_path)
                self._manifests[manifest.name] = manifest
                self._model_paths[manifest.name] = subdir

                if manifest.engine not in ENGINE_PACKAGE and not manifest.python_package:
                    logger.warning(
                        "unknown_engine",
                        name=manifest.name,
                        engine=manifest.engine,
                        hint="Engine not in built-in registry. Worker may fail to start.",
                    )

                logger.info("model_found", name=manifest.name, engine=manifest.engine)
            except (ManifestParseError, ManifestValidationError, ValueError) as exc:
                logger.error("manifest_error", path=str(manifest_path), error=str(exc))

    def get_manifest(self, model_name: str) -> ModelManifest:
        """Return model manifest.

        Raises:
            ModelNotFoundError: If the model is not found.
        """
        manifest = self._manifests.get(model_name)
        if manifest is None:
            raise ModelNotFoundError(model_name)
        return manifest

    def list_models(self) -> list[ModelManifest]:
        """Return list of all installed models."""
        return list(self._manifests.values())

    def has_model(self, model_name: str) -> bool:
        """Check if model exists in the registry."""
        return model_name in self._manifests

    def get_model_path(self, model_name: str) -> Path:
        """Return the model directory path.

        Raises:
            ModelNotFoundError: If the model is not found.
        """
        path = self._model_paths.get(model_name)
        if path is None:
            raise ModelNotFoundError(model_name)
        return path
