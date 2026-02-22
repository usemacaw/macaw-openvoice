#!/usr/bin/env bash
# ==============================================================================
# Macaw OpenVoice Demo — Start Script
#
# Starts the backend (FastAPI + gRPC workers) and the frontend (Next.js) in parallel.
# Ctrl+C stops both.
#
# Usage:
#   ./demo/start.sh                  # defaults
#   DEMO_PORT=8080 ./demo/start.sh   # backend on port 8080
#   SKIP_FRONTEND=1 ./demo/start.sh  # backend only
#   SKIP_BACKEND=1  ./demo/start.sh  # frontend only
# ==============================================================================
set -euo pipefail

# ---------------------------------------------------------------------------
# Resolve paths relative to the project root
# ---------------------------------------------------------------------------
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"

VENV_PYTHON="$PROJECT_ROOT/.venv/bin/python"
FRONTEND_DIR="$SCRIPT_DIR/frontend"

# ---------------------------------------------------------------------------
# Configuration via environment variables (with defaults)
# ---------------------------------------------------------------------------
DEMO_HOST="${DEMO_HOST:-127.0.0.1}"
DEMO_PORT="${DEMO_PORT:-9000}"
FRONTEND_PORT="${FRONTEND_PORT:-3000}"
SKIP_FRONTEND="${SKIP_FRONTEND:-0}"
SKIP_BACKEND="${SKIP_BACKEND:-0}"
UVICORN_RELOAD="${UVICORN_RELOAD:-1}"

# ---------------------------------------------------------------------------
# Colors for output
# ---------------------------------------------------------------------------
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
CYAN='\033[0;36m'
NC='\033[0m' # No Color
BOLD='\033[1m'

log_info()  { echo -e "${GREEN}[INFO]${NC}  $*"; }
log_warn()  { echo -e "${YELLOW}[WARN]${NC}  $*"; }
log_error() { echo -e "${RED}[ERROR]${NC} $*"; }
log_step()  { echo -e "${CYAN}[STEP]${NC}  $*"; }

# ---------------------------------------------------------------------------
# Banner
# ---------------------------------------------------------------------------
echo -e "${BOLD}"
echo "  ╔══════════════════════════════════════════════╗"
echo "  ║         Macaw OpenVoice — Demo                ║"
echo "  ║    Voice runtime (STT + TTS)           ║"
echo "  ╚══════════════════════════════════════════════╝"
echo -e "${NC}"

# ---------------------------------------------------------------------------
# Pre-checks
# ---------------------------------------------------------------------------
log_step "Verifying prerequisites..."

if [ "$SKIP_BACKEND" != "1" ]; then
    if [ ! -f "$VENV_PYTHON" ]; then
        log_error "Venv not found at $PROJECT_ROOT/.venv/"
        log_error "Run: cd $PROJECT_ROOT && uv venv --python 3.12 && uv pip install -e '.[server,grpc,dev]'"
        exit 1
    fi

    # Check if the Macaw package is installed
    if ! "$VENV_PYTHON" -c "import macaw" 2>/dev/null; then
        log_error "Package 'macaw' not found in venv."
        log_error "Run: cd $PROJECT_ROOT && .venv/bin/pip install -e '.[server,grpc]'"
        exit 1
    fi

    MACAW_VERSION=$("$VENV_PYTHON" -c "import macaw; print(macaw.__version__)" 2>/dev/null || echo "unknown")
    log_info "Macaw OpenVoice v${MACAW_VERSION}"

    # Check if models are installed
    MODELS_DIR="${DEMO_MODELS_DIR:-$HOME/.macaw/models}"
    if [ ! -d "$MODELS_DIR" ]; then
        log_warn "Models directory not found: $MODELS_DIR"
        log_warn "Create the directory and install at least one STT model."
        log_warn "  mkdir -p $MODELS_DIR"
        log_warn "  macaw pull faster-whisper-tiny"
    else
        MODEL_COUNT=$(find "$MODELS_DIR" -name "macaw.yaml" 2>/dev/null | wc -l)
        if [ "$MODEL_COUNT" -eq 0 ]; then
            log_warn "No models found in $MODELS_DIR"
            log_warn "Install at least one STT model: macaw pull faster-whisper-tiny"
        else
            log_info "Models found: $MODEL_COUNT (in $MODELS_DIR)"
        fi
    fi
fi

if [ "$SKIP_FRONTEND" != "1" ]; then
    if ! command -v node &>/dev/null; then
        log_error "Node.js not found. Install Node.js 18+."
        exit 1
    fi
    NODE_VERSION=$(node --version)
    log_info "Node.js $NODE_VERSION"

    if [ ! -d "$FRONTEND_DIR/node_modules" ]; then
        log_step "Installing frontend dependencies..."
        (cd "$FRONTEND_DIR" && npm install --silent)
        log_info "Frontend dependencies installed."
    fi
fi

# ---------------------------------------------------------------------------
# Trap to clean up processes on exit (Ctrl+C)
# ---------------------------------------------------------------------------
PIDS=()

cleanup() {
    echo ""
    log_step "Shutting down processes..."
    for pid in "${PIDS[@]}"; do
        if kill -0 "$pid" 2>/dev/null; then
            kill "$pid" 2>/dev/null || true
        fi
    done
    # Wait for all to finish
    for pid in "${PIDS[@]}"; do
        wait "$pid" 2>/dev/null || true
    done
    log_info "Demo stopped."
}

trap cleanup EXIT INT TERM

# ---------------------------------------------------------------------------
# Start backend
# ---------------------------------------------------------------------------
if [ "$SKIP_BACKEND" != "1" ]; then
    log_step "Starting backend at http://${DEMO_HOST}:${DEMO_PORT} ..."

    UVICORN_ARGS=(
        "$VENV_PYTHON" -m uvicorn
        demo.backend.app:app
        --host "$DEMO_HOST"
        --port "$DEMO_PORT"
    )

    if [ "$UVICORN_RELOAD" != "0" ]; then
        UVICORN_ARGS+=(--reload)
    fi

    (cd "$PROJECT_ROOT" && "${UVICORN_ARGS[@]}" 2>&1 | while IFS= read -r line; do
        echo -e "${BLUE}[backend]${NC} $line"
    done) &
    PIDS+=($!)

    # Wait for backend to be ready (max 30s)
    log_step "Waiting for backend to be ready..."
    for i in $(seq 1 30); do
        if curl -sf "http://${DEMO_HOST}:${DEMO_PORT}/api/health" >/dev/null 2>&1; then
            log_info "Backend ready!"
            break
        fi
        if [ "$i" -eq 30 ]; then
            log_warn "Backend still not responding after 30s. Continuing..."
        fi
        sleep 1
    done
fi

# ---------------------------------------------------------------------------
# Start frontend
# ---------------------------------------------------------------------------
if [ "$SKIP_FRONTEND" != "1" ]; then
    log_step "Starting frontend at http://localhost:${FRONTEND_PORT} ..."

    NEXT_PUBLIC_DEMO_API="http://${DEMO_HOST}:${DEMO_PORT}" \
        PORT="$FRONTEND_PORT" \
        npm --prefix "$FRONTEND_DIR" run dev 2>&1 | while IFS= read -r line; do
            echo -e "${CYAN}[frontend]${NC} $line"
        done &
    PIDS+=($!)
fi

# ---------------------------------------------------------------------------
# Summary
# ---------------------------------------------------------------------------
echo ""
echo -e "${BOLD}══════════════════════════════════════════════════${NC}"
if [ "$SKIP_BACKEND" != "1" ]; then
    echo -e "  ${GREEN}Backend${NC}:   http://${DEMO_HOST}:${DEMO_PORT}"
    echo -e "  ${GREEN}Health${NC}:    http://${DEMO_HOST}:${DEMO_PORT}/api/health"
    echo -e "  ${GREEN}Docs${NC}:      http://${DEMO_HOST}:${DEMO_PORT}/docs"
fi
if [ "$SKIP_FRONTEND" != "1" ]; then
    echo -e "  ${CYAN}Frontend${NC}:  http://localhost:${FRONTEND_PORT}"
fi
echo -e "${BOLD}══════════════════════════════════════════════════${NC}"
echo -e "  Ctrl+C to stop."
echo ""

# ---------------------------------------------------------------------------
# Keep script running until Ctrl+C
# ---------------------------------------------------------------------------
wait
