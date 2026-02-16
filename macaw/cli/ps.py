"""`macaw ps` command — lists models loaded on the server."""

from __future__ import annotations

import sys

import click

from macaw.cli.main import cli
from macaw.cli.transcribe import DEFAULT_SERVER_URL


@cli.command()
@click.option(
    "--server",
    default=DEFAULT_SERVER_URL,
    show_default=True,
    help="Macaw server URL.",
)
def ps(server: str) -> None:
    """Lists models loaded on the server.

    Requires the server to be running (macaw serve).
    """
    import httpx

    url = f"{server}/v1/models"

    try:
        response = httpx.get(url, timeout=10.0)
    except httpx.ConnectError:
        click.echo(
            f"Error: server not available at {server}. Run 'macaw serve' first.",
            err=True,
        )
        sys.exit(1)

    if response.status_code != 200:
        click.echo(f"Error ({response.status_code}): {response.text}", err=True)
        sys.exit(1)

    data = response.json()
    models = data.get("data", [])

    if not models:
        click.echo("No models loaded.")
        return

    # Header
    name_w = max(len(m.get("id", "")) for m in models)
    name_w = max(name_w, 4)
    header = f"{'NAME':<{name_w}}  {'TYPE':<5}  {'ENGINE':<16}"
    click.echo(header)

    for m in models:
        name = m.get("id", "?")
        model_type = m.get("type", "?")
        engine = m.get("engine", "?")
        click.echo(f"{name:<{name_w}}  {model_type:<5}  {engine:<16}")
