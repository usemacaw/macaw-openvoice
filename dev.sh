#!/usr/bin/env bash
set -euo pipefail

VENV=".venv"
PYTHON="$VENV/bin/python"

# ── Venv check ──────────────────────────────────────────────────
if [ ! -f "$PYTHON" ]; then
    echo "Venv not found at $VENV/"
    echo "Run first:  ./setup.sh"
    exit 1
fi

# Detect whether to use uv or pip (venv created with uv does not include pip)
if command -v uv &>/dev/null; then
    PIP="uv pip"
elif "$PYTHON" -m pip --version &>/dev/null; then
    PIP="$PYTHON -m pip"
else
    echo "Neither uv nor pip found. Install uv: curl -LsSf https://astral.sh/uv/install.sh | sh"
    exit 1
fi

PY_VERSION=$("$PYTHON" --version 2>&1)

usage() {
    echo "Macaw OpenVoice — Dev Helper ($PY_VERSION)"
    echo ""
    echo "Usage: ./dev.sh <command> [args...]"
    echo ""
    echo "Commands:"
    echo "  install [extras]    Install the project in editable mode"
    echo "                      ./dev.sh install          → pip install -e ."
    echo "                      ./dev.sh install all      → pip install -e '.[all]'"
    echo "                      ./dev.sh install dev,grpc → pip install -e '.[dev,grpc]'"
    echo "  pip <args>          Run pip (via uv) in the venv"
    echo "                      ./dev.sh pip list"
    echo "                      ./dev.sh pip install numpy"
    echo "  test [args]         Run pytest (unit tests)"
    echo "                      ./dev.sh test"
    echo "                      ./dev.sh test tests/unit/test_types.py -v"
    echo "  lint                Run ruff check + format check"
    echo "  format              Format code with ruff"
    echo "  typecheck           Run mypy"
    echo "  check               Run lint + typecheck + test (full CI)"
    echo "  proto               Generate protobuf stubs"
    echo "  python <args>       Run python from the venv"
    echo "                      ./dev.sh python -c 'import macaw; print(macaw.__version__)'"
    echo "  shell               Open shell with venv activated"
    echo "  info                Show environment information"
    echo ""
}

cmd_install() {
    local extras="${1:-}"
    if [ -z "$extras" ]; then
        echo "==> $PIP install -e ."
        $PIP install -e . --python "$PYTHON"
    else
        echo "==> $PIP install -e '.[$extras]'"
        $PIP install -e ".[$extras]" --python "$PYTHON"
    fi
}

cmd_pip() {
    $PIP "$@" --python "$PYTHON"
}

cmd_test() {
    if [ $# -eq 0 ]; then
        "$PYTHON" -m pytest tests/unit/ -v
    else
        "$PYTHON" -m pytest "$@"
    fi
}

cmd_lint() {
    echo "=== Ruff Check ==="
    "$PYTHON" -m ruff check src/ tests/
    echo "=== Ruff Format Check ==="
    "$PYTHON" -m ruff format --check src/ tests/
}

cmd_format() {
    "$PYTHON" -m ruff format src/ tests/
    "$PYTHON" -m ruff check --fix src/ tests/
}

cmd_typecheck() {
    "$PYTHON" -m mypy src/
}

cmd_check() {
    cmd_lint
    echo ""
    cmd_typecheck
    echo ""
    cmd_test "$@"
}

cmd_proto() {
    if [ -f "scripts/generate_proto.sh" ]; then
        bash scripts/generate_proto.sh
    else
        echo "Script scripts/generate_proto.sh not found."
        exit 1
    fi
}

cmd_python() {
    "$PYTHON" "$@"
}

cmd_shell() {
    echo "Activating venv ($PY_VERSION)..."
    echo "Use 'exit' to leave."
    # shellcheck disable=SC1091
    exec bash --init-file <(echo "source $VENV/bin/activate && echo 'Venv activated: $PY_VERSION'")
}

cmd_info() {
    echo "Macaw OpenVoice — Development Environment"
    echo ""
    echo "Python:    $("$PYTHON" --version 2>&1)"
    echo "Path:      $(realpath "$PYTHON")"
    echo "Installer: $PIP"
    echo "Venv:      $(realpath "$VENV")"
    echo "Project:   $(pwd)"
    echo ""
    echo "Installed packages (Macaw*):"
    $PIP list --python "$PYTHON" 2>/dev/null | grep -i Macaw || echo "  (none)"
}

# ── Dispatch ─────────────────────────────────────────────────────
if [ $# -eq 0 ]; then
    usage
    exit 0
fi

COMMAND="$1"
shift

case "$COMMAND" in
    install)    cmd_install "$@" ;;
    pip)        cmd_pip "$@" ;;
    test)       cmd_test "$@" ;;
    lint)       cmd_lint ;;
    format)     cmd_format ;;
    typecheck)  cmd_typecheck ;;
    check)      cmd_check "$@" ;;
    proto)      cmd_proto ;;
    python)     cmd_python "$@" ;;
    shell)      cmd_shell ;;
    info)       cmd_info ;;
    help|-h|--help) usage ;;
    *)
        echo "Unknown command: $COMMAND"
        echo ""
        usage
        exit 1
        ;;
esac
