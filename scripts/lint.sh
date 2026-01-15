#!/usr/bin/env bash
# PyFlare Lint Script

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"

# Colors
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m'

# Options
FIX=false
CPP_ONLY=false
PYTHON_ONLY=false
JS_ONLY=false

usage() {
    echo "Usage: $0 [OPTIONS]"
    echo ""
    echo "Options:"
    echo "  --fix              Auto-fix issues where possible"
    echo "  --cpp-only         Only lint C++ code"
    echo "  --python-only      Only lint Python code"
    echo "  --js-only          Only lint JavaScript/TypeScript code"
    echo "  -h, --help         Show this help message"
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
        --fix)
            FIX=true
            shift
            ;;
        --cpp-only)
            CPP_ONLY=true
            shift
            ;;
        --python-only)
            PYTHON_ONLY=true
            shift
            ;;
        --js-only)
            JS_ONLY=true
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

# C++ linting
lint_cpp() {
    log_info "Linting C++ code..."

    # Find all C++ files
    CPP_FILES=$(find "$PROJECT_ROOT/src" -name "*.cpp" -o -name "*.h" 2>/dev/null || true)

    if [ -z "$CPP_FILES" ]; then
        log_warn "No C++ files found"
        return 0
    fi

    # clang-format
    if command -v clang-format &> /dev/null; then
        if [ "$FIX" = true ]; then
            echo "$CPP_FILES" | xargs clang-format -i
            log_info "C++ files formatted"
        else
            if ! echo "$CPP_FILES" | xargs clang-format --dry-run --Werror 2>/dev/null; then
                log_error "C++ formatting issues found. Run with --fix to auto-fix."
                EXIT_CODE=1
            fi
        fi
    else
        log_warn "clang-format not found, skipping C++ formatting"
    fi

    # clang-tidy (if build exists)
    if [ -f "$PROJECT_ROOT/build/compile_commands.json" ] && command -v clang-tidy &> /dev/null; then
        log_info "Running clang-tidy..."
        if ! echo "$CPP_FILES" | xargs clang-tidy -p "$PROJECT_ROOT/build" 2>/dev/null; then
            log_warn "clang-tidy found issues"
        fi
    fi
}

# Python linting
lint_python() {
    log_info "Linting Python code..."

    PYTHON_DIR="$PROJECT_ROOT/sdk/python"

    if [ ! -d "$PYTHON_DIR" ]; then
        log_warn "Python SDK directory not found"
        return 0
    fi

    cd "$PYTHON_DIR"

    # Ruff
    if command -v ruff &> /dev/null; then
        if [ "$FIX" = true ]; then
            ruff check --fix . || true
            log_info "Python files linted with ruff"
        else
            if ! ruff check .; then
                log_error "Python linting issues found. Run with --fix to auto-fix."
                EXIT_CODE=1
            fi
        fi
    else
        log_warn "ruff not found, skipping Python linting"
    fi

    # Black
    if command -v black &> /dev/null; then
        if [ "$FIX" = true ]; then
            black .
            log_info "Python files formatted with black"
        else
            if ! black --check .; then
                log_error "Python formatting issues found. Run with --fix to auto-fix."
                EXIT_CODE=1
            fi
        fi
    else
        log_warn "black not found, skipping Python formatting"
    fi

    # MyPy
    if command -v mypy &> /dev/null; then
        if ! mypy pyflare --ignore-missing-imports 2>/dev/null; then
            log_warn "mypy found type issues"
        fi
    fi

    cd "$PROJECT_ROOT"
}

# JavaScript/TypeScript linting
lint_js() {
    log_info "Linting JavaScript/TypeScript code..."

    UI_DIR="$PROJECT_ROOT/ui"

    if [ ! -d "$UI_DIR" ]; then
        log_warn "UI directory not found"
        return 0
    fi

    cd "$UI_DIR"

    # Check if node_modules exists
    if [ ! -d "node_modules" ]; then
        log_warn "node_modules not found. Run 'npm install' first."
        cd "$PROJECT_ROOT"
        return 0
    fi

    # ESLint
    if [ "$FIX" = true ]; then
        npm run lint:fix || true
        log_info "JS/TS files linted"
    else
        if ! npm run lint; then
            log_error "JS/TS linting issues found. Run with --fix to auto-fix."
            EXIT_CODE=1
        fi
    fi

    # Prettier
    if [ "$FIX" = true ]; then
        npm run format || true
        log_info "JS/TS files formatted"
    fi

    cd "$PROJECT_ROOT"
}

# Run linters based on options
if [ "$CPP_ONLY" = true ]; then
    lint_cpp
elif [ "$PYTHON_ONLY" = true ]; then
    lint_python
elif [ "$JS_ONLY" = true ]; then
    lint_js
else
    lint_cpp
    lint_python
    lint_js
fi

if [ $EXIT_CODE -eq 0 ]; then
    log_info "All linting checks passed!"
else
    log_error "Linting found issues"
fi

exit $EXIT_CODE
