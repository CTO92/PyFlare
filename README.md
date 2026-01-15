# PyFlare

**OpenTelemetry-native observability platform for AI/ML workloads**

PyFlare provides deep observability into AI/ML applications, including traditional ML, deep learning, and LLM applications. It extends the PyFlame ecosystem philosophy - breaking vendor lock-in for AI infrastructure - from training into production monitoring.

## Release

PyFlare is released as a pre-release V1.0.0-alpha and will likely contain bugs. Please use the "issues" tab to report them. Use at your own risk.

## About OA Quantum Labs

PyFlame is developed by **[OA Quantum Labs](https://oaqlabs.com)**, a specialized engineering firm focused on high-performance computing and tooling that breaks vendor lock-in.

### What We Do

In this context We help organizations unlock the full potential of specialized hardware through custom developer tools, optimized frameworks, and performance engineering:

- **Custom Framework Development** — Native tooling designed for your specific accelerator architecture
- **Performance Optimization** — Squeeze maximum throughput from your existing hardware investments
- **Migration & Porting** — Adapt existing ML workloads to new accelerator platforms
- **Training & Enablement** — Get your team productive on specialized hardware faster

### Why Work With Us

PyFlare demonstrates our approach: rather than forcing general-purpose tools onto specialized hardware, we build native solutions that leverage the unique strengths of each architecture. The result is dramatically better performance and a more intuitive developer experience.

If your organization is working with specialized AI accelerators, FPGAs, or custom silicon, we'd love to discuss how purpose-built tooling could transform your development workflow.

### Get In Touch

**Danny Wall** — CTO, OA Quantum Labs
[dwall@oaqlabs.com](mailto:dwall@oaqlabs.com) | [oaqlabs.com](https://oaqlabs.com)

## Support the Project

We welcome financial support in the effort. Learn how you can help at https://oaqlabs.com/pyflare (bottom of page)

## Features

- **Deep Model Introspection**: Understand *why* models make specific decisions
- **Multi-Model Coverage**: Unified observability for ML, deep learning, and LLMs
- **Intelligent Drift Detection**: Embedding, feature, concept, and prediction drift with PSI, MMD, and KS tests
- **LLM Evaluations**: Hallucination detection, RAG quality, toxicity scoring, semantic similarity
- **Safety Analysis**: PII detection, prompt injection detection, content safety scoring
- **Cost Intelligence**: Granular per-request cost tracking, budget management, and attribution
- **Root Cause Analysis**: AI-driven failure clustering, slice analysis, and causal factor identification
- **Intelligent Alerting**: Rule-based alerts with deduplication, silences, and maintenance windows
- **Intelligence Pipeline**: Unified orchestration of drift, evaluation, safety, and RCA
- **Zero Vendor Lock-In**: OpenTelemetry native, standard data formats

## Quick Start

### Python SDK

```bash
# Clone the repository
git clone https://github.com/oaqlabs/pyflare.git
cd pyflare

# Install the Python SDK from source
cd sdk/python
pip install -e .
```

```python
from pyflare import PyFlare, trace

# Initialize
pyflare = PyFlare(
    service_name="my-ml-service",
    endpoint="http://localhost:4317",
)

# Trace your inference
@trace(model_id="gpt-4")
def chat(messages):
    return openai.chat.completions.create(
        model="gpt-4",
        messages=messages
    )
```

### Docker Compose

```bash
cd deploy/docker
docker-compose up -d
```

This starts:
- PyFlare Collector (OTLP receiver)
- PyFlare Processor (drift detection, evaluation)
- PyFlare Query API
- PyFlare Web UI
- Kafka, ClickHouse, Qdrant, Redis

Access the UI at http://localhost:3000

## Architecture

```
┌─────────────────┐
│  ML Application │
│  (Python SDK)   │
└────────┬────────┘
         │ OTLP
         ▼
┌─────────────────┐
│ PyFlare Collector│
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│  Apache Kafka   │
└────────┬────────┘
         │
         ▼
┌─────────────────────────────────────┐
│       Intelligence Pipeline         │
│  ┌────────┐ ┌────────┐ ┌────────┐  │
│  │ Drift  │ │  Eval  │ │ Safety │  │
│  │Detector│ │ Engine │ │Analyzer│  │
│  └───┬────┘ └───┬────┘ └───┬────┘  │
│      └────┬─────┴─────┬────┘       │
│           ▼           ▼            │
│      ┌────────┐  ┌────────┐        │
│      │  RCA   │  │ Alert  │        │
│      │ Engine │  │ Service│        │
│      └────────┘  └────────┘        │
└────────────────┬────────────────────┘
                 ▼
    ┌─────────────────────────┐
    │        Storage          │
    │ (ClickHouse/Qdrant/Redis)│
    └────────────┬────────────┘
                 │
                 ▼
    ┌─────────────────────────┐
    │       Query API         │
    │ (REST: /intelligence,   │
    │  /alerts, /rca, /drift) │
    └────────────┬────────────┘
                 │
                 ▼
    ┌─────────────────────────┐
    │     Web UI Dashboard    │
    │ (Intelligence, Alerts,  │
    │  RCA Explorer)          │
    └─────────────────────────┘
```

## Building from Source

### Prerequisites

- CMake 3.20+
- C++20 compiler (GCC 11+, Clang 14+, MSVC 2022+)
- Python 3.10+
- Node.js 20+

### Build

```bash
# C++ components
./scripts/build.sh

# Python SDK (from repository root)
cd sdk/python
pip install -e ".[dev]"
cd ../..

# Web UI (from repository root)
cd ui
npm install
npm run build
```

### Run Tests

```bash
./scripts/test.sh
```

## Documentation

- [Technical Architecture](Technical_Architecture.md)
- [API Reference](docs/api-reference.md)
- [SDK Documentation](sdk/python/README.md)
- [Deployment Guide](docs/deployment.md)

## Project Structure

```
pyflare/
├── src/                    # C++ source code
│   ├── collector/          # OTLP collector
│   ├── processor/          # Stream processors
│   │   ├── drift/          # Drift detection (KS, PSI, MMD)
│   │   ├── eval/           # Evaluators (hallucination, RAG, safety)
│   │   ├── rca/            # Root cause analysis (clustering, slices)
│   │   ├── cost/           # Cost tracking & budgets
│   │   ├── alerting/       # Alert rules, deduplication, notifications
│   │   └── intelligence/   # Unified intelligence pipeline
│   ├── storage/            # Storage clients (ClickHouse, Qdrant, Redis)
│   ├── query/              # Query API
│   │   └── handlers/       # REST handlers (intelligence, alerts, RCA)
│   └── common/             # Shared utilities
├── sdk/
│   └── python/             # Python SDK
├── ui/                     # React web UI
│   └── src/components/     # Intelligence Dashboard, Alerts Panel, RCA Explorer
├── deploy/
│   ├── docker/             # Docker configs
│   └── kubernetes/         # Helm charts
├── tests/                  # Test suites
└── scripts/                # Build scripts
```

## Contributing

We welcome contributions! Please see [CONTRIBUTING.md](CONTRIBUTING.md) for guidelines.

## License

Apache 2.0 - See [LICENSE](LICENSE) for details.
