"""`macaw remove` command — removes an installed model."""

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
    help="Directory with installed models.",
)
@click.option(
    "--yes",
    "-y",
    is_flag=True,
    default=False,
    help="Confirm removal without prompting.",
)
def remove(model_name: str, models_dir: str, yes: bool) -> None:
    """Remove an installed model.

    Example: macaw remove faster-whisper-tiny
    """
    from pathlib import Path

    from macaw.registry.downloader import ModelDownloader

    downloader = ModelDownloader(Path(models_dir).expanduser())

    if not downloader.is_installed(model_name):
        click.echo(f"Erro: modelo '{model_name}' nao esta instalado.", err=True)
        sys.exit(1)

    if not yes and not click.confirm(f"Remover o modelo '{model_name}'?", default=False):
        click.echo("Cancelado.")
        return

    removed = downloader.remove(model_name)
    if removed:
        click.echo(f"Modelo '{model_name}' removido.")
    else:
        click.echo(f"Erro ao remover modelo '{model_name}'.", err=True)
        sys.exit(1)
