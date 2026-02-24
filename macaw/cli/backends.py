"""`macaw backends` commands — manage per-engine venv environments."""

from __future__ import annotations

import sys

import click

from macaw.cli.main import cli


@cli.group()
def backends() -> None:
    """Manage backend engine environments (venvs)."""


def _validate_known_engine(engine: str) -> None:
    """Exit with error if *engine* is not in ENGINE_EXTRAS."""
    from macaw.engines import ENGINE_EXTRAS

    if engine not in ENGINE_EXTRAS:
        click.echo(f"Error: unknown engine '{engine}'.", err=True)
        click.echo(f"Available engines: {', '.join(sorted(ENGINE_EXTRAS))}", err=True)
        sys.exit(1)


@backends.command("install")
@click.argument("engine")
def install(engine: str) -> None:
    """Pre-provision an isolated venv for ENGINE."""
    from macaw.backends.venv_manager import VenvManager
    from macaw.config.settings import get_settings
    from macaw.exceptions import VenvProvisionError

    _validate_known_engine(engine)

    settings = get_settings().backend
    manager = VenvManager(settings.venv_base_path, uv_path=settings.uv_path)

    if manager.exists(engine):
        click.echo(f"Backend venv for '{engine}' already exists.")
        return

    click.echo(f"Provisioning venv for '{engine}'...")
    try:
        python_path = manager.provision(engine)
        click.echo(f"Backend venv provisioned: {python_path}")
    except VenvProvisionError as exc:
        click.echo(f"Error: {exc}", err=True)
        sys.exit(1)


@backends.command("list")
def list_backends() -> None:
    """List all engines and their venv status."""
    from macaw.backends.venv_manager import VenvManager
    from macaw.config.settings import get_settings
    from macaw.engines import ENGINE_EXTRAS

    settings = get_settings().backend
    manager = VenvManager(settings.venv_base_path, uv_path=settings.uv_path)

    click.echo(f"{'Engine':<20} {'Status':<15} {'Path'}")
    click.echo("-" * 70)
    for engine in sorted(ENGINE_EXTRAS):
        if manager.exists(engine):
            click.echo(f"{engine:<20} {'provisioned':<15} {manager.venv_dir(engine)}")
        else:
            click.echo(f"{engine:<20} {'not provisioned':<15} -")


@backends.command("remove")
@click.argument("engine")
@click.option("--yes", "-y", is_flag=True, help="Skip confirmation.")
def remove(engine: str, yes: bool) -> None:
    """Remove the venv for ENGINE."""
    from macaw.backends.venv_manager import VenvManager
    from macaw.config.settings import get_settings

    _validate_known_engine(engine)

    settings = get_settings().backend
    manager = VenvManager(settings.venv_base_path, uv_path=settings.uv_path)

    if not manager.exists(engine):
        click.echo(f"No venv found for '{engine}'.")
        return

    if not yes:
        click.confirm(f"Remove venv for '{engine}'?", abort=True)

    manager.remove(engine)
    click.echo(f"Backend venv for '{engine}' removed.")


@backends.command("status")
@click.argument("engine")
def status(engine: str) -> None:
    """Show detailed status for ENGINE's venv."""
    from macaw.backends.venv_manager import VenvManager
    from macaw.config.settings import get_settings

    _validate_known_engine(engine)

    settings = get_settings().backend
    manager = VenvManager(settings.venv_base_path, uv_path=settings.uv_path)

    if not manager.exists(engine):
        click.echo(f"No provisioned venv for '{engine}'.")
        return

    click.echo(f"Engine:     {engine}")
    click.echo(f"Venv dir:   {manager.venv_dir(engine)}")
    click.echo(f"Python:     {manager.venv_python(engine)}")

    marker = manager.read_marker(engine)
    if marker:
        click.echo(f"Extra:      {marker.get('extra', 'unknown')}")
        click.echo(f"Provisioned: {marker.get('provisioned_at', 'unknown')}")

    available = manager.is_engine_available_in_venv(engine)
    click.echo(f"Available:  {'yes' if available else 'no'}")
