#!/bin/sh
# Redirect to the main install script.
# Quick install:
#   curl -fsSL https://raw.githubusercontent.com/macaw-voice/Macaw-openvoice/main/install.sh | sh

set -eu

# If running from a local clone, use the local script
SCRIPT_DIR="$(cd "$(dirname "$0")" 2>/dev/null && pwd)"
if [ -f "$SCRIPT_DIR/scripts/install.sh" ]; then
    exec sh "$SCRIPT_DIR/scripts/install.sh" "$@"
fi

# Otherwise, download and run the script from GitHub
curl -fsSL https://raw.githubusercontent.com/macaw-voice/macaw-openvoice/main/scripts/install.sh | sh
