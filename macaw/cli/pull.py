"""`macaw pull` command — downloads models from the HuggingFace Hub."""

from __future__ import annotations

import sys

import click

from macaw.cli.main import cli
from macaw.cli.serve import DEFAULT_MODELS_DIR


def _install_engine_deps(engine: str) -> bool:
    """Provision an isolated venv for *engine*.

    Returns ``True`` if provisioning succeeded or venv already exists.
    Falls back to parent-process pip install if venv provisioning is disabled.
    """
    from macaw.backends.venv_manager import VenvManager
    from macaw.config.settings import get_settings
    from macaw.exceptions import VenvProvisionError

    settings = get_settings().backend
    manager = VenvManager(settings.venv_base_path, uv_path=settings.uv_path)

    if manager.exists(engine):
        click.echo(f"Backend venv for '{engine}' already exists.")
        return True

    click.echo(f"Provisioning venv for engine '{engine}'...")
    try:
        manager.provision(engine)
        click.echo(f"Backend venv for '{engine}' provisioned.")
        return True
    except VenvProvisionError as exc:
        click.echo(f"Warning: {exc}", err=True)
        click.echo(
            f"Install manually: macaw backends install {engine}",
            err=True,
        )
        return False


@cli.command()
@click.argument("model_name")
@click.option(
    "--models-dir",
    default=DEFAULT_MODELS_DIR,
    show_default=True,
    help="Directory where the models will be installed.",
)
@click.option(
    "--force",
    is_flag=True,
    default=False,
    help="Overwrite an existing model.",
)
def pull(model_name: str, models_dir: str, force: bool) -> None:
    """Downloads a model from the HuggingFace Hub."""
    from pathlib import Path

    from macaw.registry.catalog import ModelCatalog
    from macaw.registry.downloader import ModelDownloader

    # Load catalog
    catalog = ModelCatalog()
    try:
        catalog.load()
    except (FileNotFoundError, ValueError) as e:
        click.echo(f"Failed to load catalog: {e}", err=True)
        sys.exit(1)

    # Check if model exists in the catalog
    entry = catalog.get(model_name)
    if entry is None:
        click.echo(f"Error: model '{model_name}' not found in the catalog.", err=True)
        click.echo("", err=True)
        click.echo("Available models:", err=True)
        for m in catalog.list_models():
            click.echo(f"  {m.name:<30} {m.description}", err=True)
        sys.exit(1)

    # Download
    downloader = ModelDownloader(Path(models_dir).expanduser())

    if downloader.is_installed(model_name) and not force:
        click.echo(f"Model '{model_name}' is already installed.")
        _install_engine_deps(entry.engine)
        click.echo("Use --force to reinstall.")
        return

    click.echo(f"Downloading {model_name} from {entry.repo}...")

    try:
        model_dir = downloader.download(entry, force=force)
    except RuntimeError as e:
        click.echo(f"Error: {e}", err=True)
        sys.exit(1)
    except Exception as e:
        click.echo(f"Failed to download the model: {e}", err=True)
        sys.exit(1)

    click.echo(f"Model installed in {model_dir}")

    # Auto-install engine dependencies
    _install_engine_deps(entry.engine)

    click.echo("Run 'macaw serve' to start the server.")
