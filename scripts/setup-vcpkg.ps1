# PowerShell script to set up vcpkg for PyFlare development
# Run this script in PowerShell to install vcpkg and PyFlare dependencies

$ErrorActionPreference = "Stop"

$VCPKG_ROOT = "$env:USERPROFILE\.vcpkg"
$VCPKG_REPO = "https://github.com/microsoft/vcpkg.git"

Write-Host "PyFlare vcpkg Setup Script" -ForegroundColor Cyan
Write-Host "==========================" -ForegroundColor Cyan
Write-Host ""

# Check if vcpkg is already installed
if (Test-Path "$VCPKG_ROOT\vcpkg.exe") {
    Write-Host "vcpkg is already installed at $VCPKG_ROOT" -ForegroundColor Green
} else {
    Write-Host "Installing vcpkg to $VCPKG_ROOT..." -ForegroundColor Yellow

    # Clone vcpkg
    if (Test-Path $VCPKG_ROOT) {
        Remove-Item -Recurse -Force $VCPKG_ROOT
    }

    git clone $VCPKG_REPO $VCPKG_ROOT

    if ($LASTEXITCODE -ne 0) {
        Write-Host "Failed to clone vcpkg repository" -ForegroundColor Red
        exit 1
    }

    # Bootstrap vcpkg
    Push-Location $VCPKG_ROOT
    & .\bootstrap-vcpkg.bat
    Pop-Location

    if ($LASTEXITCODE -ne 0) {
        Write-Host "Failed to bootstrap vcpkg" -ForegroundColor Red
        exit 1
    }

    Write-Host "vcpkg installed successfully" -ForegroundColor Green
}

# Set environment variable
Write-Host ""
Write-Host "Setting VCPKG_ROOT environment variable..." -ForegroundColor Yellow
[Environment]::SetEnvironmentVariable("VCPKG_ROOT", $VCPKG_ROOT, "User")
$env:VCPKG_ROOT = $VCPKG_ROOT

# Add vcpkg to PATH for current session
$env:PATH = "$VCPKG_ROOT;$env:PATH"

Write-Host ""
Write-Host "Installing PyFlare dependencies..." -ForegroundColor Yellow
Write-Host "This may take several minutes on first run..." -ForegroundColor Yellow
Write-Host ""

# Get the script directory and navigate to project root
$ScriptDir = Split-Path -Parent $MyInvocation.MyCommand.Path
$ProjectRoot = Split-Path -Parent $ScriptDir

Push-Location $ProjectRoot

# Install dependencies using manifest mode
& "$VCPKG_ROOT\vcpkg.exe" install --triplet x64-windows

if ($LASTEXITCODE -ne 0) {
    Write-Host "Failed to install vcpkg dependencies" -ForegroundColor Red
    Pop-Location
    exit 1
}

Pop-Location

Write-Host ""
Write-Host "Setup complete!" -ForegroundColor Green
Write-Host ""
Write-Host "To build PyFlare with vcpkg, use:" -ForegroundColor Cyan
Write-Host "  cmake --preset windows-msvc" -ForegroundColor White
Write-Host "  cmake --build build/windows-msvc" -ForegroundColor White
Write-Host ""
Write-Host "Or configure your IDE to use the vcpkg toolchain:" -ForegroundColor Cyan
Write-Host "  CMAKE_TOOLCHAIN_FILE=$VCPKG_ROOT\scripts\buildsystems\vcpkg.cmake" -ForegroundColor White
