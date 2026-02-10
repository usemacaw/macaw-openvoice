"""`macaw list` and `macaw inspect` commands — model management."""

from __future__ import annotations

import asyncio
import sys
from pathlib import Path

import click

from macaw.cli.main import cli
from macaw.cli.serve import DEFAULT_MODELS_DIR


@cli.command(name="list")
@click.option(
    "--models-dir",
    default=DEFAULT_MODELS_DIR,
    show_default=True,
    help="Diretorio com os modelos instalados.",
)
def list_models(models_dir: str) -> None:
    """Lista modelos instalados."""
    asyncio.run(_list_models(models_dir))


async def _list_models(models_dir: str) -> None:
    from macaw.registry.registry import ModelRegistry

    models_path = Path(models_dir).expanduser()
    registry = ModelRegistry(models_path)
    await registry.scan()

    models = registry.list_models()
    if not models:
        click.echo("Nenhum modelo instalado.")
        click.echo(f"Diretorio de modelos: {models_path}")
        return

    # Header
    name_w = max(len(m.name) for m in models)
    name_w = max(name_w, 4)  # min "NAME"
    header = f"{'NAME':<{name_w}}  {'TYPE':<5}  {'ENGINE':<16}  {'ARCHITECTURE':<18}  {'MEMORY'}"
    click.echo(header)

    for m in models:
        arch = m.capabilities.architecture.value if m.capabilities.architecture else "—"
        memory = f"{m.resources.memory_mb} MB"
        click.echo(
            f"{m.name:<{name_w}}  {m.model_type.value:<5}  {m.engine:<16}  {arch:<18}  {memory}"
        )


@cli.command()
@click.argument("model_name")
@click.option(
    "--models-dir",
    default=DEFAULT_MODELS_DIR,
    show_default=True,
    help="Diretorio com os modelos instalados.",
)
def inspect(model_name: str, models_dir: str) -> None:
    """Mostra detalhes de um modelo instalado."""
    asyncio.run(_inspect(model_name, models_dir))


async def _inspect(model_name: str, models_dir: str) -> None:
    from macaw.exceptions import ModelNotFoundError
    from macaw.registry.registry import ModelRegistry

    models_path = Path(models_dir).expanduser()
    registry = ModelRegistry(models_path)
    await registry.scan()

    try:
        manifest = registry.get_manifest(model_name)
    except ModelNotFoundError:
        click.echo(f"Erro: modelo '{model_name}' nao encontrado.", err=True)
        click.echo(
            f"Execute 'macaw list --models-dir {models_dir}' para ver os modelos instalados.",
            err=True,
        )
        sys.exit(1)

    arch = manifest.capabilities.architecture.value if manifest.capabilities.architecture else "—"
    languages = (
        ", ".join(manifest.capabilities.languages) if manifest.capabilities.languages else "—"
    )
    gpu_req = "Yes" if manifest.resources.gpu_required else "No"
    gpu_rec = "Yes" if manifest.resources.gpu_recommended else "No"

    capabilities: list[str] = []
    if manifest.capabilities.streaming:
        capabilities.append("streaming")
    if manifest.capabilities.word_timestamps:
        capabilities.append("word_timestamps")
    if manifest.capabilities.translation:
        capabilities.append("translation")
    if manifest.capabilities.batch_inference:
        capabilities.append("batch_inference")
    if manifest.capabilities.hot_words:
        capabilities.append("hot_words")
    if manifest.capabilities.language_detection:
        capabilities.append("language_detection")

    cap_str = ", ".join(capabilities) if capabilities else "—"

    click.echo(f"Name:            {manifest.name}")
    click.echo(f"Type:            {manifest.model_type.value}")
    click.echo(f"Engine:          {manifest.engine}")
    click.echo(f"Version:         {manifest.version}")
    click.echo(f"Architecture:    {arch}")
    click.echo(f"Languages:       {languages}")
    click.echo(f"Memory:          {manifest.resources.memory_mb} MB")
    click.echo(f"GPU Required:    {gpu_req}")
    click.echo(f"GPU Recommended: {gpu_rec}")
    click.echo(f"Capabilities:    {cap_str}")
    if manifest.description:
        click.echo(f"Description:     {manifest.description}")
