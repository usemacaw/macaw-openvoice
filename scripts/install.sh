#!/bin/sh
# This script installs Macaw OpenVoice on Linux and macOS.
# It uses uv (Astral) to install Python 3.12 and the macaw-openvoice package.
#
# Quick install:
#   curl -fsSL https://raw.githubusercontent.com/macaw-voice/macaw-openvoice/main/install.sh | sh
#
# Environment variables:
#   MACAW_VERSION       Pin to a specific version (default: latest)
#   MACAW_INSTALL_DIR   Custom install directory (default: /opt/macaw)
#   MACAW_EXTRAS        Pip extras to install (default: server,grpc)
#   MACAW_NO_SERVICE    Skip systemd service setup (set to any value)

set -eu

red="$( (/usr/bin/tput bold || :; /usr/bin/tput setaf 1 || :) 2>&-)"
green="$( (/usr/bin/tput bold || :; /usr/bin/tput setaf 2 || :) 2>&-)"
plain="$( (/usr/bin/tput sgr0 || :) 2>&-)"

status() { echo ">>> $*" >&2; }
error() { echo "${red}ERROR:${plain} $*"; exit 1; }
warning() { echo "${red}WARNING:${plain} $*"; }
success() { echo "${green}$*${plain}"; }

TEMP_DIR=$(mktemp -d)
cleanup() { rm -rf "$TEMP_DIR"; }
trap cleanup EXIT

available() { command -v "$1" >/dev/null; }

OS="$(uname -s)"
ARCH=$(uname -m)
case "$ARCH" in
    x86_64) ARCH="amd64" ;;
    aarch64|arm64) ARCH="arm64" ;;
    *) error "Unsupported architecture: $ARCH" ;;
esac

# Configuration
MACAW_INSTALL_DIR="${MACAW_INSTALL_DIR:-/opt/macaw}"
MACAW_EXTRAS="${MACAW_EXTRAS:-server,grpc}"
MACAW_VERSION="${MACAW_VERSION:-}"
PYTHON_VERSION="3.12"

###########################################
# macOS
###########################################

if [ "$OS" = "Darwin" ]; then
    if ! available curl; then
        error "curl is required but not found. Please install it first."
    fi

    status "Installing Macaw OpenVoice on macOS..."

    # Install uv if not present
    if ! available uv; then
        status "Installing uv (Python package manager)..."
        curl -LsSf https://astral.sh/uv/install.sh | sh
        # Source the env to get uv in PATH
        if [ -f "$HOME/.local/bin/env" ]; then
            . "$HOME/.local/bin/env"
        elif [ -f "$HOME/.cargo/env" ]; then
            . "$HOME/.cargo/env"
        fi
        export PATH="$HOME/.local/bin:$PATH"
    fi

    if ! available uv; then
        error "Failed to install uv. Please install it manually: https://docs.astral.sh/uv/getting-started/installation/"
    fi

    status "uv version: $(uv --version)"

    # Create install directory
    MACAW_INSTALL_DIR="${MACAW_INSTALL_DIR:-$HOME/.macaw}"
    mkdir -p "$MACAW_INSTALL_DIR"

    # Create venv with Python 3.12
    status "Creating Python $PYTHON_VERSION environment in $MACAW_INSTALL_DIR..."
    uv venv --python "$PYTHON_VERSION" "$MACAW_INSTALL_DIR/.venv"

    # Install macaw-openvoice
    MACAW_PKG="macaw-openvoice[$MACAW_EXTRAS]"
    if [ -n "$MACAW_VERSION" ]; then
        MACAW_PKG="macaw-openvoice[$MACAW_EXTRAS]==$MACAW_VERSION"
    fi
    status "Installing $MACAW_PKG..."
    uv pip install --python "$MACAW_INSTALL_DIR/.venv/bin/python" "$MACAW_PKG"

    # Symlink to /usr/local/bin
    status "Adding 'macaw' command to PATH..."
    mkdir -p "/usr/local/bin" 2>/dev/null || sudo mkdir -p "/usr/local/bin"
    ln -sf "$MACAW_INSTALL_DIR/.venv/bin/macaw" "/usr/local/bin/macaw" 2>/dev/null || \
        sudo ln -sf "$MACAW_INSTALL_DIR/.venv/bin/macaw" "/usr/local/bin/macaw"

    # GPU detection
    if system_profiler SPDisplaysDataType 2>/dev/null | grep -qi "nvidia\|cuda"; then
        status "NVIDIA GPU detected. To install GPU acceleration:"
        echo "  uv pip install --python $MACAW_INSTALL_DIR/.venv/bin/python 'macaw-openvoice[faster-whisper]'"
    fi

    success "Install complete. Run 'macaw serve' to start the API server."
    echo "  API will be available at http://127.0.0.1:8000"
    echo ""
    echo "  Quick start:"
    echo "    macaw serve &"
    echo "    macaw pull faster-whisper-large-v3"
    echo "    macaw transcribe audio.wav"
    exit 0
fi

###########################################
# Linux
###########################################

[ "$OS" = "Linux" ] || error 'This script is intended to run on Linux and macOS only.'

IS_WSL2=false
KERN=$(uname -r)
case "$KERN" in
    *icrosoft*WSL2 | *icrosoft*wsl2) IS_WSL2=true ;;
    *icrosoft) error "Microsoft WSL1 is not currently supported. Please use WSL2 with 'wsl --set-version <distro> 2'" ;;
    *) ;;
esac

SUDO=
if [ "$(id -u)" -ne 0 ]; then
    if ! available sudo; then
        error "This script requires superuser permissions. Please re-run as root."
    fi
    SUDO="sudo"
fi

if ! available curl; then
    error "curl is required but not found. Please install it first."
fi

status "Installing Macaw OpenVoice on Linux ($ARCH)..."

# Install uv if not present
if ! available uv; then
    status "Installing uv (Python package manager)..."
    curl -LsSf https://astral.sh/uv/install.sh | sh
    # Source the env to get uv in PATH
    if [ -f "$HOME/.local/bin/env" ]; then
        . "$HOME/.local/bin/env"
    elif [ -f "$HOME/.cargo/env" ]; then
        . "$HOME/.cargo/env"
    fi
    export PATH="$HOME/.local/bin:$PATH"
fi

if ! available uv; then
    error "Failed to install uv. Please install it manually: https://docs.astral.sh/uv/getting-started/installation/"
fi

status "uv version: $(uv --version)"

# Create install directory
status "Creating install directory at $MACAW_INSTALL_DIR..."
$SUDO mkdir -p "$MACAW_INSTALL_DIR"
$SUDO chown "$(id -u):$(id -g)" "$MACAW_INSTALL_DIR"

# Create venv with Python 3.12
status "Creating Python $PYTHON_VERSION environment..."
uv venv --python "$PYTHON_VERSION" "$MACAW_INSTALL_DIR/.venv"

# Install macaw-openvoice
MACAW_PKG="macaw-openvoice[$MACAW_EXTRAS]"
if [ -n "$MACAW_VERSION" ]; then
    MACAW_PKG="macaw-openvoice[$MACAW_EXTRAS]==$MACAW_VERSION"
fi
status "Installing $MACAW_PKG..."
uv pip install --python "$MACAW_INSTALL_DIR/.venv/bin/python" "$MACAW_PKG"

# Symlink to PATH
for BINDIR in /usr/local/bin /usr/bin /bin; do
    echo "$PATH" | grep -q "$BINDIR" && break || continue
done

status "Adding 'macaw' command to PATH in $BINDIR..."
$SUDO ln -sf "$MACAW_INSTALL_DIR/.venv/bin/macaw" "$BINDIR/macaw"

install_success() {
    success "Install complete."
    echo "  The Macaw OpenVoice API is available at http://127.0.0.1:8000"
    echo ""
    echo "  Quick start:"
    echo "    macaw serve &"
    echo "    macaw pull faster-whisper-large-v3"
    echo "    macaw transcribe audio.wav"
}

# Configure systemd service (optional)
configure_systemd() {
    if [ -n "${MACAW_NO_SERVICE:-}" ]; then
        status "Skipping systemd service setup (MACAW_NO_SERVICE is set)."
        return
    fi

    if ! id macaw >/dev/null 2>&1; then
        status "Creating macaw user..."
        $SUDO useradd -r -s /bin/false -U -m -d /var/lib/macaw macaw
    fi
    if getent group render >/dev/null 2>&1; then
        status "Adding macaw user to render group..."
        $SUDO usermod -a -G render macaw
    fi
    if getent group video >/dev/null 2>&1; then
        status "Adding macaw user to video group..."
        $SUDO usermod -a -G video macaw
    fi

    status "Adding current user to macaw group..."
    $SUDO usermod -a -G macaw "$(whoami)"

    status "Creating macaw systemd service..."
    cat <<EOF | $SUDO tee /etc/systemd/system/macaw.service >/dev/null
[Unit]
Description=Macaw OpenVoice Service
After=network-online.target

[Service]
ExecStart=$MACAW_INSTALL_DIR/.venv/bin/macaw serve --host 0.0.0.0 --port 8000
User=macaw
Group=macaw
Restart=always
RestartSec=3
Environment="PATH=$MACAW_INSTALL_DIR/.venv/bin:$PATH"
Environment="MACAW_MODELS_DIR=/var/lib/macaw/models"
WorkingDirectory=/var/lib/macaw

[Install]
WantedBy=default.target
EOF

    # Create models directory
    $SUDO mkdir -p /var/lib/macaw/models
    $SUDO chown -R macaw:macaw /var/lib/macaw

    SYSTEMCTL_RUNNING="$(systemctl is-system-running || true)"
    case $SYSTEMCTL_RUNNING in
        running|degraded)
            status "Enabling and starting macaw service..."
            $SUDO systemctl daemon-reload
            $SUDO systemctl enable macaw

            start_service() { $SUDO systemctl restart macaw; }
            trap start_service EXIT
            ;;
        *)
            warning "systemd is not running."
            if [ "$IS_WSL2" = true ]; then
                warning "See https://learn.microsoft.com/en-us/windows/wsl/systemd#how-to-enable-systemd to enable it."
            fi
            ;;
    esac
}

if available systemctl; then
    configure_systemd
fi

# GPU detection
check_gpu() {
    case $1 in
        lspci)
            case $2 in
                nvidia) available lspci && lspci -d '10de:' | grep -q 'NVIDIA' || return 1 ;;
            esac ;;
        lshw)
            case $2 in
                nvidia) available lshw && $SUDO lshw -c display -numeric -disable network | grep -q 'vendor: .* \[10DE\]' || return 1 ;;
            esac ;;
        nvidia-smi) available nvidia-smi || return 1 ;;
    esac
}

if check_gpu nvidia-smi; then
    status "NVIDIA GPU detected with drivers installed."
    status "Installing GPU-accelerated STT engine..."
    uv pip install --python "$MACAW_INSTALL_DIR/.venv/bin/python" "macaw-openvoice[faster-whisper]" || \
        warning "Failed to install faster-whisper GPU extras. You can install manually later."
elif (check_gpu lspci nvidia 2>/dev/null || check_gpu lshw nvidia 2>/dev/null); then
    warning "NVIDIA GPU detected but nvidia-smi not found. Install NVIDIA drivers first."
    echo "  After installing drivers, run:"
    echo "    uv pip install --python $MACAW_INSTALL_DIR/.venv/bin/python 'macaw-openvoice[faster-whisper]'"
elif [ "$IS_WSL2" = true ] && available nvidia-smi; then
    status "NVIDIA GPU detected via WSL2 passthrough."
    status "Installing GPU-accelerated STT engine..."
    uv pip install --python "$MACAW_INSTALL_DIR/.venv/bin/python" "macaw-openvoice[faster-whisper]" || \
        warning "Failed to install faster-whisper GPU extras. You can install manually later."
else
    status "No NVIDIA GPU detected. Macaw will run in CPU-only mode."
    echo "  CPU-only mode is fully functional but slower for large models."
fi

install_success
