#!/bin/bash
# Script to set up vcpkg for PyFlare development
# Run this script to install vcpkg and PyFlare dependencies

set -e

VCPKG_ROOT="${VCPKG_ROOT:-$HOME/.vcpkg}"
VCPKG_REPO="https://github.com/microsoft/vcpkg.git"

echo "PyFlare vcpkg Setup Script"
echo "=========================="
echo ""

# Check if vcpkg is already installed
if [ -f "$VCPKG_ROOT/vcpkg" ]; then
    echo "vcpkg is already installed at $VCPKG_ROOT"
else
    echo "Installing vcpkg to $VCPKG_ROOT..."

    # Clone vcpkg
    if [ -d "$VCPKG_ROOT" ]; then
        rm -rf "$VCPKG_ROOT"
    fi

    git clone "$VCPKG_REPO" "$VCPKG_ROOT"

    # Bootstrap vcpkg
    pushd "$VCPKG_ROOT" > /dev/null
    ./bootstrap-vcpkg.sh
    popd > /dev/null

    echo "vcpkg installed successfully"
fi

# Set environment variable
echo ""
echo "Setting VCPKG_ROOT environment variable..."

# Add to shell profile if not already present
SHELL_PROFILE=""
if [ -f "$HOME/.zshrc" ]; then
    SHELL_PROFILE="$HOME/.zshrc"
elif [ -f "$HOME/.bashrc" ]; then
    SHELL_PROFILE="$HOME/.bashrc"
elif [ -f "$HOME/.bash_profile" ]; then
    SHELL_PROFILE="$HOME/.bash_profile"
fi

if [ -n "$SHELL_PROFILE" ]; then
    if ! grep -q "VCPKG_ROOT" "$SHELL_PROFILE"; then
        echo "" >> "$SHELL_PROFILE"
        echo "# vcpkg" >> "$SHELL_PROFILE"
        echo "export VCPKG_ROOT=\"$VCPKG_ROOT\"" >> "$SHELL_PROFILE"
        echo "export PATH=\"\$VCPKG_ROOT:\$PATH\"" >> "$SHELL_PROFILE"
        echo "Added VCPKG_ROOT to $SHELL_PROFILE"
    fi
fi

export VCPKG_ROOT="$VCPKG_ROOT"
export PATH="$VCPKG_ROOT:$PATH"

echo ""
echo "Installing PyFlare dependencies..."
echo "This may take several minutes on first run..."
echo ""

# Get the script directory and navigate to project root
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"

pushd "$PROJECT_ROOT" > /dev/null

# Determine triplet
if [ "$(uname)" == "Darwin" ]; then
    TRIPLET="x64-osx"
    if [ "$(uname -m)" == "arm64" ]; then
        TRIPLET="arm64-osx"
    fi
else
    TRIPLET="x64-linux"
fi

# Install dependencies using manifest mode
"$VCPKG_ROOT/vcpkg" install --triplet "$TRIPLET"

popd > /dev/null

echo ""
echo "Setup complete!"
echo ""
echo "To build PyFlare with vcpkg, use:"
echo "  cmake --preset dev-vcpkg"
echo "  cmake --build build/dev-vcpkg"
echo ""
echo "Or configure your IDE to use the vcpkg toolchain:"
echo "  CMAKE_TOOLCHAIN_FILE=\$VCPKG_ROOT/scripts/buildsystems/vcpkg.cmake"
