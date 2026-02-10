"""`macaw pull` command — downloads models from the HuggingFace Hub."""

from __future__ import annotations

import sys

import click

from macaw.cli.main import cli
from macaw.cli.serve import DEFAULT_MODELS_DIR


@cli.command()
@click.argument("model_name")
@click.option(
    "--models-dir",
    default=DEFAULT_MODELS_DIR,
    show_default=True,
    help="Directory to install models.",
)
@click.option(
    "--force",
    is_flag=True,
    default=False,
    help="Overwrite existing model.",
)
def pull(model_name: str, models_dir: str, force: bool) -> None:
    """Download a model from the HuggingFace Hub.

    Available models: faster-whisper-tiny, faster-whisper-small,
    faster-whisper-medium, faster-whisper-large-v3, distil-whisper-large-v3,
    kokoro-v1.

    Example: macaw pull faster-whisper-tiny
    """
    from pathlib import Path

    from macaw.registry.catalog import ModelCatalog
    from macaw.registry.downloader import ModelDownloader

    # Load catalog
    catalog = ModelCatalog()
    try:
        catalog.load()
    except (FileNotFoundError, ValueError) as e:
        click.echo(f"Error loading catalog: {e}", err=True)
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
        click.echo("Use --force to reinstall.")
        return

    click.echo(f"Downloading {model_name} from {entry.repo}...")

    try:
        model_dir = downloader.download(entry, force=force)
    except RuntimeError as e:
        click.echo(f"Error: {e}", err=True)
        sys.exit(1)
    except Exception as e:
        click.echo(f"Error downloading model: {e}", err=True)
        sys.exit(1)

    click.echo(f"Model installed in {model_dir}")
    click.echo("Run 'macaw serve' to start the server.")
