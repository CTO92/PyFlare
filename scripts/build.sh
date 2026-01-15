#!/usr/bin/env bash
# PyFlare Build Script

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"
BUILD_DIR="${PROJECT_ROOT}/build"

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Default options
BUILD_TYPE="Release"
BUILD_TESTS="ON"
CLEAN_BUILD=false
JOBS=$(nproc 2>/dev/null || sysctl -n hw.ncpu 2>/dev/null || echo 4)

usage() {
    echo "Usage: $0 [OPTIONS]"
    echo ""
    echo "Options:"
    echo "  -t, --type TYPE     Build type: Debug, Release, RelWithDebInfo (default: Release)"
    echo "  -c, --clean         Clean build directory before building"
    echo "  -j, --jobs N        Number of parallel jobs (default: auto-detect)"
    echo "  --no-tests          Don't build tests"
    echo "  -h, --help          Show this help message"
    echo ""
}

log_info() {
    echo -e "${GREEN}[INFO]${NC} $1"
}

log_warn() {
    echo -e "${YELLOW}[WARN]${NC} $1"
}

log_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

# Parse arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        -t|--type)
            BUILD_TYPE="$2"
            shift 2
            ;;
        -c|--clean)
            CLEAN_BUILD=true
            shift
            ;;
        -j|--jobs)
            JOBS="$2"
            shift 2
            ;;
        --no-tests)
            BUILD_TESTS="OFF"
            shift
            ;;
        -h|--help)
            usage
            exit 0
            ;;
        *)
            log_error "Unknown option: $1"
            usage
            exit 1
            ;;
    esac
done

# Validate build type
case $BUILD_TYPE in
    Debug|Release|RelWithDebInfo|MinSizeRel)
        ;;
    *)
        log_error "Invalid build type: $BUILD_TYPE"
        exit 1
        ;;
esac

log_info "Building PyFlare"
log_info "  Build type: $BUILD_TYPE"
log_info "  Build tests: $BUILD_TESTS"
log_info "  Parallel jobs: $JOBS"

# Clean if requested
if [ "$CLEAN_BUILD" = true ]; then
    log_info "Cleaning build directory..."
    rm -rf "$BUILD_DIR"
fi

# Create build directory
mkdir -p "$BUILD_DIR"

# Configure
log_info "Configuring..."
cmake -B "$BUILD_DIR" -S "$PROJECT_ROOT" \
    -DCMAKE_BUILD_TYPE="$BUILD_TYPE" \
    -DPYFLARE_BUILD_TESTS="$BUILD_TESTS"

# Build
log_info "Building..."
cmake --build "$BUILD_DIR" --parallel "$JOBS"

log_info "Build complete!"
log_info "Binaries are in: $BUILD_DIR/bin"
log_info "Libraries are in: $BUILD_DIR/lib"
