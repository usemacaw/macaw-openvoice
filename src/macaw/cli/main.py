"""Main CLI command group for Macaw OpenVoice."""

from __future__ import annotations

import click

import macaw


@click.group()
@click.version_option(version=macaw.__version__, prog_name="macaw")
def cli() -> None:
    """Macaw OpenVoice — Voice runtime (STT + TTS)."""
