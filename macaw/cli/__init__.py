"""Macaw OpenVoice CLI.

Registers all commands in the main group.
"""

from macaw.cli.backends import backends
from macaw.cli.main import cli
from macaw.cli.models import catalog, inspect, list_models
from macaw.cli.ps import ps
from macaw.cli.pull import pull
from macaw.cli.remove import remove
from macaw.cli.serve import serve
from macaw.cli.transcribe import transcribe, translate

__all__ = [
    "backends",
    "catalog",
    "cli",
    "inspect",
    "list_models",
    "ps",
    "pull",
    "remove",
    "serve",
    "transcribe",
    "translate",
]
