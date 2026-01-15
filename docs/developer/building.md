# Building PyFlare

This guide covers building PyFlare from source, including the C++ collector and Python SDK.

## Prerequisites

### Required Tools

- **CMake** 3.20 or later
- **C++20 compiler**:
  - GCC 11+ on Linux
  - Clang 14+ on macOS
  - MSVC 2022 on Windows
- **Python** 3.10+ (for SDK)
- **vcpkg** (for C++ dependency management)
- **Docker** (optional, for containerized builds)

### Installing Prerequisites

#### Ubuntu/Debian

```bash
sudo apt-get update
sudo apt-get install -y \
    build-essential \
    cmake \
    ninja-build \
    git \
    pkg-config \
    curl \
    python3 \
    python3-pip \
    python3-venv
```

#### macOS

```bash
brew install cmake ninja python@3.11
xcode-select --install  # For Clang
```

#### Windows

```powershell
# Install Visual Studio 2022 with C++ workload
# Or use winget:
winget install Microsoft.VisualStudio.2022.BuildTools
winget install Kitware.CMake
winget install Python.Python.3.11
```

---

## Setting Up vcpkg

PyFlare uses vcpkg for C++ dependency management.

### Clone vcpkg

```bash
git clone https://github.com/Microsoft/vcpkg.git ~/vcpkg
cd ~/vcpkg
./bootstrap-vcpkg.sh  # Linux/macOS
# bootstrap-vcpkg.bat  # Windows
```

### Set Environment Variable

```bash
# Add to ~/.bashrc or ~/.zshrc
export VCPKG_ROOT=~/vcpkg
export PATH=$VCPKG_ROOT:$PATH
```

### Install Dependencies

The project includes a `vcpkg.json` manifest. Dependencies are installed automatically during CMake configuration, or manually:

```bash
cd ~/vcpkg
./vcpkg install \
    abseil \
    spdlog \
    nlohmann-json \
    yaml-cpp \
    cli11 \
    gtest

# Optional dependencies for full functionality
./vcpkg install grpc         # gRPC support
./vcpkg install rdkafka      # Kafka support
./vcpkg install cpp-httplib  # HTTP support
```

---

## Building the Collector

### Basic Build

```bash
cd PyFlare

# Configure
cmake -B build \
    -DCMAKE_BUILD_TYPE=Release \
    -DCMAKE_TOOLCHAIN_FILE=$VCPKG_ROOT/scripts/buildsystems/vcpkg.cmake

# Build
cmake --build build --parallel

# The binary is at: build/bin/pyflare_collector
```

### Build Options

| Option | Default | Description |
|--------|---------|-------------|
| `CMAKE_BUILD_TYPE` | `Release` | Build type: Debug, Release, RelWithDebInfo |
| `PYFLARE_BUILD_TESTS` | `ON` | Build unit tests |
| `PYFLARE_BUILD_BENCHMARKS` | `OFF` | Build benchmarks |
| `PYFLARE_ENABLE_SANITIZERS` | `OFF` | Enable address/thread sanitizers |

```bash
# Debug build with tests
cmake -B build \
    -DCMAKE_BUILD_TYPE=Debug \
    -DPYFLARE_BUILD_TESTS=ON \
    -DCMAKE_TOOLCHAIN_FILE=$VCPKG_ROOT/scripts/buildsystems/vcpkg.cmake

# Release build without tests
cmake -B build \
    -DCMAKE_BUILD_TYPE=Release \
    -DPYFLARE_BUILD_TESTS=OFF \
    -DCMAKE_TOOLCHAIN_FILE=$VCPKG_ROOT/scripts/buildsystems/vcpkg.cmake
```

### Using Ninja (Faster Builds)

```bash
cmake -B build -G Ninja \
    -DCMAKE_BUILD_TYPE=Release \
    -DCMAKE_TOOLCHAIN_FILE=$VCPKG_ROOT/scripts/buildsystems/vcpkg.cmake

cmake --build build
```

### Conditional Features

The collector automatically detects available dependencies:

```bash
# Check which features are enabled
cmake -B build \
    -DCMAKE_TOOLCHAIN_FILE=$VCPKG_ROOT/scripts/buildsystems/vcpkg.cmake \
    2>&1 | grep "PYFLARE"

# Output:
# -- PYFLARE_HAS_GRPC: ON
# -- PYFLARE_HAS_RDKAFKA: ON
# -- PYFLARE_HAS_HTTPLIB: ON
```

---

## Building the Python SDK

### Development Installation

```bash
cd sdk/python

# Create virtual environment
python -m venv .venv
source .venv/bin/activate  # Linux/macOS
# .venv\Scripts\activate   # Windows

# Install in development mode
pip install -e ".[dev]"
```

### Dependencies

The SDK has minimal core dependencies:

```
opentelemetry-api>=1.20.0
opentelemetry-sdk>=1.20.0
opentelemetry-exporter-otlp>=1.20.0
```

Optional dependencies for integrations (install from repository root):

```bash
cd sdk/python
pip install -e ".[openai]"      # OpenAI integration
pip install -e ".[anthropic]"   # Anthropic integration
pip install -e ".[all]"         # All integrations
pip install -e ".[dev]"         # Development dependencies
```

---

## Running Tests

### C++ Tests

```bash
cd PyFlare

# Build with tests
cmake -B build \
    -DPYFLARE_BUILD_TESTS=ON \
    -DCMAKE_TOOLCHAIN_FILE=$VCPKG_ROOT/scripts/buildsystems/vcpkg.cmake
cmake --build build

# Run all tests
cd build
ctest --output-on-failure

# Run specific test
./bin/sampler_test
./bin/batcher_test
```

### Python Tests

```bash
cd sdk/python
source .venv/bin/activate

# Run all tests
pytest -v

# Run with coverage
pytest --cov=pyflare --cov-report=html

# Run specific test file
pytest tests/test_sdk.py -v

# Run specific test
pytest tests/test_sdk.py::TestPyFlare::test_initialization -v
```

### Integration Tests

```bash
# Start the collector
./build/bin/pyflare_collector --config config/collector.yaml &

# Run integration tests
cd tests/integration
pytest -v --collector-url=http://localhost:4318
```

---

## Docker Builds

### Building Images

```bash
cd PyFlare

# Build collector image
docker build -t pyflare-collector:latest \
    -f deploy/docker/Dockerfile.collector .

# Build all images with docker-compose
cd deploy/docker
docker-compose build
```

### Multi-Platform Builds

```bash
# Create builder
docker buildx create --name pyflare-builder --use

# Build for multiple platforms
docker buildx build \
    --platform linux/amd64,linux/arm64 \
    -t pyflare-collector:latest \
    -f deploy/docker/Dockerfile.collector \
    --push .
```

### Running with Docker Compose

```bash
cd deploy/docker

# Start all services
docker-compose up -d

# View logs
docker-compose logs -f collector

# Stop services
docker-compose down
```

---

## IDE Setup

### Visual Studio Code

Recommended extensions:
- C/C++ (Microsoft)
- CMake Tools (Microsoft)
- Python (Microsoft)
- Pylance (Microsoft)

Create `.vscode/settings.json`:

```json
{
    "cmake.configureSettings": {
        "CMAKE_TOOLCHAIN_FILE": "${env:VCPKG_ROOT}/scripts/buildsystems/vcpkg.cmake"
    },
    "python.defaultInterpreterPath": "${workspaceFolder}/sdk/python/.venv/bin/python",
    "python.testing.pytestEnabled": true,
    "python.testing.pytestArgs": ["sdk/python/tests"]
}
```

### CLion

1. Open the project
2. Go to Settings → Build → CMake
3. Add CMake option: `-DCMAKE_TOOLCHAIN_FILE=$VCPKG_ROOT/scripts/buildsystems/vcpkg.cmake`

### PyCharm

1. Open `sdk/python` as a project
2. Configure Python interpreter to use the virtual environment
3. Enable pytest as the test runner

---

## Troubleshooting

### CMake Can't Find vcpkg

```bash
# Ensure VCPKG_ROOT is set
echo $VCPKG_ROOT

# If not set, add to shell profile:
export VCPKG_ROOT=~/vcpkg

# Or specify directly in cmake:
cmake -B build \
    -DCMAKE_TOOLCHAIN_FILE=~/vcpkg/scripts/buildsystems/vcpkg.cmake
```

### Missing Dependencies

```bash
# Reinstall vcpkg dependencies
cd $VCPKG_ROOT
./vcpkg remove abseil spdlog nlohmann-json yaml-cpp cli11 gtest
./vcpkg install abseil spdlog nlohmann-json yaml-cpp cli11 gtest
```

### gRPC Build Issues

gRPC can be slow to build. Consider using pre-built binaries:

```bash
# Ubuntu
sudo apt-get install libgrpc++-dev protobuf-compiler-grpc

# Or install via vcpkg (slower but more portable)
vcpkg install grpc
```

### Python Import Errors

```bash
# Ensure you're in the virtual environment
source sdk/python/.venv/bin/activate
which python  # Should show .venv path

# Reinstall in development mode
pip install -e ".[dev]"
```

### Test Failures

```bash
# Run tests with verbose output
ctest --output-on-failure -V

# For Python, show full traceback
pytest -v --tb=long
```

---

## Continuous Integration

### GitHub Actions Example

```yaml
# .github/workflows/build.yml
name: Build

on: [push, pull_request]

jobs:
  build-cpp:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4

      - name: Setup vcpkg
        uses: lukka/run-vcpkg@v11
        with:
          vcpkgGitCommitId: '...'

      - name: Configure
        run: |
          cmake -B build \
            -DCMAKE_BUILD_TYPE=Release \
            -DPYFLARE_BUILD_TESTS=ON \
            -DCMAKE_TOOLCHAIN_FILE=$VCPKG_ROOT/scripts/buildsystems/vcpkg.cmake

      - name: Build
        run: cmake --build build --parallel

      - name: Test
        run: cd build && ctest --output-on-failure

  build-python:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4

      - name: Setup Python
        uses: actions/setup-python@v5
        with:
          python-version: '3.11'

      - name: Install dependencies
        run: |
          cd sdk/python
          pip install -e ".[dev]"

      - name: Run tests
        run: |
          cd sdk/python
          pytest -v --cov=pyflare
```

---

## Release Process

### Versioning

PyFlare follows semantic versioning (MAJOR.MINOR.PATCH).

### Creating a Release

1. Update version in:
   - `CMakeLists.txt`
   - `sdk/python/pyflare/__init__.py`
   - `CHANGELOG.md`

2. Create and push tag:
   ```bash
   git tag -a v1.0.0 -m "Release 1.0.0"
   git push origin v1.0.0
   ```

3. Build Docker images:
   ```bash
   docker build -t pyflare-collector:1.0.0 -f deploy/docker/Dockerfile.collector .
   ```

---

## Next Steps

- [Architecture](./architecture.md) - Understand the system design
- [Component Guide](./components.md) - Learn about each component
- [Extension Guide](./extending.md) - Add new features
