<#
.SYNOPSIS
    Install, upgrade, or uninstall Macaw OpenVoice on Windows.

.DESCRIPTION
    Downloads uv (if not present) and installs Macaw OpenVoice in an isolated
    Python 3.12 virtual environment.

    Quick install:

        irm https://raw.githubusercontent.com/useMacaw/Macaw-openvoice/main/scripts/install.ps1 | iex

    Specific version:

        $env:Macaw_VERSION="0.1.0"; irm https://raw.githubusercontent.com/useMacaw/Macaw-openvoice/main/scripts/install.ps1 | iex

    Custom install directory:

        $env:Macaw_INSTALL_DIR="D:\Macaw"; irm https://raw.githubusercontent.com/useMacaw/Macaw-openvoice/main/scripts/install.ps1 | iex

    Uninstall:

        $env:Macaw_UNINSTALL=1; irm https://raw.githubusercontent.com/useMacaw/Macaw-openvoice/main/scripts/install.ps1 | iex

    Environment variables:

        Macaw_VERSION       Pin to a specific version (default: latest)
        Macaw_INSTALL_DIR   Custom install directory
        Macaw_EXTRAS        Pip extras to install (default: server,grpc)
        Macaw_UNINSTALL     Set to 1 to uninstall Macaw OpenVoice

.EXAMPLE
    irm https://raw.githubusercontent.com/useMacaw/Macaw-openvoice/main/scripts/install.ps1 | iex

.LINK
    https://github.com/useMacaw/Macaw-openvoice
#>

$ErrorActionPreference = "Stop"
$ProgressPreference = "SilentlyContinue"

# --------------------------------------------------------------------------
# Configuration from environment variables
# --------------------------------------------------------------------------

$Version    = if ($env:Macaw_VERSION) { $env:Macaw_VERSION } else { "" }
$InstallDir = if ($env:Macaw_INSTALL_DIR) { $env:Macaw_INSTALL_DIR } else { "" }
$Extras     = if ($env:Macaw_EXTRAS) { $env:Macaw_EXTRAS } else { "server,grpc" }
$Uninstall  = $env:Macaw_UNINSTALL -eq "1"

$PythonVersion = "3.12"

# --------------------------------------------------------------------------
# Helpers
# --------------------------------------------------------------------------

function Write-Step {
    param([string]$Message)
    Write-Host ">>> $Message" -ForegroundColor Cyan
}

function Write-Success {
    param([string]$Message)
    Write-Host $Message -ForegroundColor Green
}

function Get-MacawDir {
    if ($InstallDir) {
        return $InstallDir
    }
    return Join-Path $env:LOCALAPPDATA "Programs\Macaw"
}

function Test-UvAvailable {
    try {
        $null = Get-Command uv -ErrorAction Stop
        return $true
    } catch {
        return $false
    }
}

# --------------------------------------------------------------------------
# Uninstall
# --------------------------------------------------------------------------

function Invoke-Uninstall {
    $MacawDir = Get-MacawDir

    if (-not (Test-Path $MacawDir)) {
        Write-Host "Macaw OpenVoice is not installed at $MacawDir."
        return
    }

    Write-Step "Uninstalling Macaw OpenVoice from $MacawDir..."

    # Remove from PATH
    $userPath = [Environment]::GetEnvironmentVariable("Path", "User")
    $scriptsDir = Join-Path $MacawDir ".venv\Scripts"
    if ($userPath -and $userPath.Contains($scriptsDir)) {
        $newPath = ($userPath -split ";" | Where-Object { $_ -ne $scriptsDir }) -join ";"
        [Environment]::SetEnvironmentVariable("Path", $newPath, "User")
        Write-Host "  Removed $scriptsDir from user PATH."
    }

    # Remove install directory
    Remove-Item -Recurse -Force $MacawDir -ErrorAction SilentlyContinue
    Write-Host "  Removed $MacawDir."

    Write-Success "Macaw OpenVoice has been uninstalled."
}

# --------------------------------------------------------------------------
# Install
# --------------------------------------------------------------------------

function Invoke-Install {
    $MacawDir = Get-MacawDir

    Write-Step "Installing Macaw OpenVoice..."

    # --- Install uv if not present ---
    if (-not (Test-UvAvailable)) {
        Write-Step "Installing uv (Python package manager)..."
        try {
            Invoke-Expression "& { $(Invoke-RestMethod https://astral.sh/uv/install.ps1) }"
        } catch {
            throw "Failed to install uv. Please install it manually: https://docs.astral.sh/uv/getting-started/installation/"
        }

        # Refresh PATH for current session
        $uvDir = Join-Path $env:LOCALAPPDATA "uv"
        if (Test-Path $uvDir) {
            $env:PATH = "$uvDir;$env:PATH"
        }
        $cargoDir = Join-Path $env:USERPROFILE ".cargo\bin"
        if (Test-Path $cargoDir) {
            $env:PATH = "$cargoDir;$env:PATH"
        }
    }

    if (-not (Test-UvAvailable)) {
        throw "uv is not available after installation. Please install it manually: https://docs.astral.sh/uv/getting-started/installation/"
    }

    $uvVersion = & uv --version
    Write-Host "  uv version: $uvVersion"

    # --- Create install directory ---
    Write-Step "Creating install directory at $MacawDir..."
    if (-not (Test-Path $MacawDir)) {
        New-Item -ItemType Directory -Path $MacawDir -Force | Out-Null
    }

    # --- Create venv with Python 3.12 ---
    Write-Step "Creating Python $PythonVersion environment..."
    & uv venv --python $PythonVersion (Join-Path $MacawDir ".venv")
    if ($LASTEXITCODE -ne 0) {
        throw "Failed to create Python $PythonVersion virtual environment."
    }

    # --- Install Macaw-openvoice ---
    $MacawPkg = "Macaw-openvoice[$Extras]"
    if ($Version) {
        $MacawPkg = "Macaw-openvoice[$Extras]==$Version"
    }
    Write-Step "Installing $MacawPkg..."
    $pythonExe = Join-Path $MacawDir ".venv\Scripts\python.exe"
    & uv pip install --python $pythonExe $MacawPkg
    if ($LASTEXITCODE -ne 0) {
        throw "Failed to install $MacawPkg."
    }

    # --- Add to PATH ---
    $scriptsDir = Join-Path $MacawDir ".venv\Scripts"
    $userPath = [Environment]::GetEnvironmentVariable("Path", "User")
    if (-not $userPath.Contains($scriptsDir)) {
        Write-Step "Adding Macaw to user PATH..."
        [Environment]::SetEnvironmentVariable("Path", "$scriptsDir;$userPath", "User")
        $env:PATH = "$scriptsDir;$env:PATH"
        Write-Host "  Added $scriptsDir to user PATH."
    }

    # --- GPU detection ---
    try {
        $nvsmi = Get-Command nvidia-smi -ErrorAction Stop
        Write-Step "NVIDIA GPU detected. Installing GPU-accelerated STT engine..."
        & uv pip install --python $pythonExe "macaw-openvoice[faster-whisper]"
        if ($LASTEXITCODE -ne 0) {
            Write-Host "  WARNING: Failed to install faster-whisper GPU extras. You can install manually later." -ForegroundColor Yellow
        }
    } catch {
        Write-Host "  No NVIDIA GPU detected. Macaw will run in CPU-only mode."
        Write-Host "  CPU-only mode is fully functional but slower for large models."
    }

    # --- Success ---
    Write-Success "Install complete. Run 'Macaw serve' to start the API server."
    Write-Host "  API will be available at http://127.0.0.1:8000"
    Write-Host ""
    Write-Host "  Quick start:"
    Write-Host "    macaw serve"
    Write-Host "    macaw pull faster-whisper-large-v3"
    Write-Host "    macaw transcribe audio.wav"
    Write-Host ""
    Write-Host "  NOTE: You may need to restart your terminal for PATH changes to take effect."
}

# --------------------------------------------------------------------------
# Main
# --------------------------------------------------------------------------

if ($Uninstall) {
    Invoke-Uninstall
} else {
    Invoke-Install
}
