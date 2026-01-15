#!/usr/bin/env bash
# PyFlare Test Script

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"
BUILD_DIR="${PROJECT_ROOT}/build"

# Colors
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m'

# Options
RUN_CPP_TESTS=true
RUN_PYTHON_TESTS=true
VERBOSE=false
COVERAGE=false

usage() {
    echo "Usage: $0 [OPTIONS]"
    echo ""
    echo "Options:"
    echo "  --cpp-only         Only run C++ tests"
    echo "  --python-only      Only run Python tests"
    echo "  -v, --verbose      Verbose test output"
    echo "  --coverage         Generate coverage report"
    echo "  -h, --help         Show this help message"
    echo ""
}

log_info() {
    echo -e "${GREEN}[INFO]${NC} $1"
}

log_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

# Parse arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --cpp-only)
            RUN_PYTHON_TESTS=false
            shift
            ;;
        --python-only)
            RUN_CPP_TESTS=false
            shift
            ;;
        -v|--verbose)
            VERBOSE=true
            shift
            ;;
        --coverage)
            COVERAGE=true
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

EXIT_CODE=0

# Run C++ tests
if [ "$RUN_CPP_TESTS" = true ]; then
    log_info "Running C++ tests..."

    if [ ! -d "$BUILD_DIR" ]; then
        log_error "Build directory not found. Run build.sh first."
        exit 1
    fi

    CTEST_ARGS="--test-dir $BUILD_DIR --output-on-failure"
    if [ "$VERBOSE" = true ]; then
        CTEST_ARGS="$CTEST_ARGS --verbose"
    fi

    if ! ctest $CTEST_ARGS; then
        log_error "C++ tests failed"
        EXIT_CODE=1
    else
        log_info "C++ tests passed"
    fi
fi

# Run Python tests
if [ "$RUN_PYTHON_TESTS" = true ]; then
    log_info "Running Python tests..."

    PYTHON_SDK_DIR="${PROJECT_ROOT}/sdk/python"

    if [ ! -d "$PYTHON_SDK_DIR" ]; then
        log_error "Python SDK directory not found"
        exit 1
    fi

    cd "$PYTHON_SDK_DIR"

    PYTEST_ARGS=""
    if [ "$VERBOSE" = true ]; then
        PYTEST_ARGS="-v"
    fi
    if [ "$COVERAGE" = true ]; then
        PYTEST_ARGS="$PYTEST_ARGS --cov=pyflare --cov-report=html --cov-report=term"
    fi

    if ! python -m pytest $PYTEST_ARGS; then
        log_error "Python tests failed"
        EXIT_CODE=1
    else
        log_info "Python tests passed"
    fi

    cd "$PROJECT_ROOT"
fi

if [ $EXIT_CODE -eq 0 ]; then
    log_info "All tests passed!"
else
    log_error "Some tests failed"
fi

exit $EXIT_CODE
