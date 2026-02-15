"""CLI do Macaw OpenVoice.

Registra todos os comandos no grupo principal.
"""

from macaw.cli.main import cli
from macaw.cli.models import catalog, inspect, list_models
from macaw.cli.ps import ps
from macaw.cli.pull import pull
from macaw.cli.remove import remove
from macaw.cli.serve import serve
from macaw.cli.transcribe import transcribe, translate

__all__ = [
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
