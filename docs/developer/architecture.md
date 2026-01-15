# PyFlare Architecture Overview

This document provides a comprehensive overview of PyFlare's Phase 1 architecture, designed for developers who want to understand, maintain, or extend the codebase.

## System Overview

PyFlare is an OpenTelemetry-native observability platform optimized for AI/ML workloads. Phase 1 implements the core telemetry collection pipeline.

```
┌─────────────────────────────────────────────────────────────────────────┐
│                           PyFlare System                                │
├─────────────────────────────────────────────────────────────────────────┤
│                                                                         │
│  ┌──────────────┐    ┌──────────────────────────────────────────────┐  │
│  │  Python SDK  │───▶│              PyFlare Collector               │  │
│  │  (pyflare)   │    │                                              │  │
│  └──────────────┘    │  ┌─────────┐  ┌─────────┐  ┌─────────────┐  │  │
│                      │  │  OTLP   │─▶│ Sampler │─▶│   Batcher   │  │  │
│                      │  │Receiver │  │         │  │             │  │  │
│                      │  └─────────┘  └─────────┘  └──────┬──────┘  │  │
│                      │                                    │         │  │
│                      │                           ┌────────▼───────┐ │  │
│                      │                           │ Kafka Exporter │ │  │
│                      │                           └────────┬───────┘ │  │
│                      └────────────────────────────────────┼─────────┘  │
│                                                           │            │
│                                                           ▼            │
│                                                    ┌────────────┐      │
│                                                    │   Kafka    │      │
│                                                    └────────────┘      │
│                                                                         │
└─────────────────────────────────────────────────────────────────────────┘
```

## Core Design Principles

### 1. OpenTelemetry Native

PyFlare embraces OpenTelemetry as the standard for telemetry data:
- Uses OTLP (OpenTelemetry Protocol) for data ingestion
- Extends OTel with AI/ML-specific semantic conventions
- Compatible with existing OTel instrumentation

### 2. Modular Pipeline Architecture

The collector follows a pipeline pattern:
```
Receive → Sample → Batch → Export
```

Each stage is a separate, testable component with well-defined interfaces.

### 3. Conditional Compilation

The C++ collector uses conditional compilation to handle optional dependencies:
- `PYFLARE_HAS_GRPC` - gRPC support
- `PYFLARE_HAS_RDKAFKA` - Kafka support
- `PYFLARE_HAS_HTTPLIB` - HTTP support

This allows building a minimal collector for development or a full-featured version for production.

### 4. PIMPL Pattern

All major components use the PIMPL (Pointer to Implementation) pattern:
```cpp
// Header (public interface)
class Component {
public:
    Component(const Config& config);
    ~Component();
    void DoSomething();
private:
    class Impl;
    std::unique_ptr<Impl> impl_;
};
```

Benefits:
- Compilation firewall (faster builds)
- ABI stability
- Clean public interfaces
- Implementation hiding

---

## Component Architecture

### Collector (Orchestrator)

The `Collector` class is the main orchestrator that:
1. Loads configuration from YAML
2. Creates and wires components
3. Manages component lifecycle
4. Handles graceful shutdown

**File:** `src/collector/collector.cpp`

```
                    ┌──────────────────┐
                    │    Collector     │
                    │                  │
                    │  ┌────────────┐  │
         config.yaml│  │   Config   │  │
        ────────────┼─▶│   Loader   │  │
                    │  └────────────┘  │
                    │                  │
                    │  Component Lifecycle:
                    │  1. Initialize   │
                    │  2. Start        │
                    │  3. Run          │
                    │  4. Shutdown     │
                    └──────────────────┘
```

### OTLP Receiver

Accepts telemetry data via gRPC (port 4317) and HTTP (port 4318).

**File:** `src/collector/otlp_receiver.cpp`

```
         gRPC:4317                    HTTP:4318
              │                            │
              ▼                            ▼
    ┌─────────────────┐         ┌─────────────────┐
    │  gRPC Service   │         │   HTTP Server   │
    │  TraceService   │         │   /v1/traces    │
    └────────┬────────┘         └────────┬────────┘
             │                           │
             │      ┌─────────────┐      │
             └─────▶│   Parser    │◀─────┘
                    │  (Protobuf  │
                    │   or JSON)  │
                    └──────┬──────┘
                           │
                           ▼
                    ┌─────────────┐
                    │  Callback   │──▶ Sampler
                    │ span_callback_
                    └─────────────┘
```

Key design decisions:
- Callback-based: Components don't know about each other
- Thread-safe: Mutex protection for concurrent requests
- Protocol-agnostic: Same processing path for gRPC and HTTP

### Sampler

Decides which spans to keep based on configurable strategies.

**File:** `src/collector/sampler.cpp`

```
                    ┌─────────────────────────────────┐
                    │           Sampler               │
                    │                                 │
       Span ───────▶│  ┌─────────────────────────┐   │
                    │  │   Sampling Strategy     │   │
                    │  ├─────────────────────────┤   │
                    │  │ • AlwaysOn              │   │
                    │  │ • AlwaysOff             │   │
                    │  │ • Probabilistic (hash)  │   │──▶ Keep/Drop
                    │  │ • RateLimiting (bucket) │   │
                    │  │ • ParentBased           │   │
                    │  │ • Composite             │   │
                    │  └─────────────────────────┘   │
                    │                                 │
                    │  Service-specific overrides:   │
                    │  service_rates_["svc"] = 0.5   │
                    │                                 │
                    └─────────────────────────────────┘
```

Sampling strategies:
- **AlwaysOn/Off**: Simple pass-through or drop
- **Probabilistic**: FNV-1a hash of trace ID for deterministic sampling
- **RateLimiting**: Token bucket algorithm for rate limiting
- **ParentBased**: Respects parent span's sampling decision
- **Composite**: Per-service sampling rates

### Batcher

Accumulates spans into batches for efficient export.

**File:** `src/collector/batcher.cpp`

```
                         ┌───────────────────────────────┐
                         │           Batcher             │
                         │                               │
        Spans ──────────▶│  ┌─────────────────────────┐ │
         (1-N)           │  │     Pending Queue       │ │
                         │  │   std::deque<Span>      │ │
                         │  └───────────┬─────────────┘ │
                         │              │               │
                         │   ┌──────────┴──────────┐    │
                         │   │                     │    │
                         │   ▼                     ▼    │
                         │ Size >= max?      Timeout?   │
                         │   │                     │    │
                         │   └──────────┬──────────┘    │
                         │              │               │
                         │              ▼               │
                         │  ┌─────────────────────────┐ │
                         │  │     Batch Ready         │ │
                         │  │   batch_callback_()     │ │──▶ Exporter
                         │  └─────────────────────────┘ │
                         │                               │
                         │  Backpressure:               │
                         │  if queue > max_queue_size   │
                         │    → block or drop oldest    │
                         └───────────────────────────────┘
```

Threading model:
- Timer thread: Periodic flush on timeout
- Worker threads: Parallel batch processing
- Mutex protection: Thread-safe queue access

### Kafka Exporter

Publishes batches to Kafka topics.

**File:** `src/collector/kafka_exporter.cpp`

```
                    ┌─────────────────────────────────┐
                    │        Kafka Exporter           │
                    │                                 │
       Batch ──────▶│  ┌─────────────────────────┐   │
                    │  │      Serializer         │   │
                    │  │  (JSON or Protobuf)     │   │
                    │  └───────────┬─────────────┘   │
                    │              │                 │
                    │              ▼                 │
                    │  ┌─────────────────────────┐   │
                    │  │    librdkafka Producer  │   │
                    │  │                         │   │
                    │  │  • Compression (lz4)    │   │──▶ Kafka
                    │  │  • Batching             │   │    Brokers
                    │  │  • Retries              │   │
                    │  │  • Idempotence          │   │
                    │  └─────────────────────────┘   │
                    │                                 │
                    │  Topics:                       │
                    │  • pyflare.traces             │
                    │  • pyflare.metrics            │
                    │  • pyflare.logs               │
                    └─────────────────────────────────┘
```

Key features:
- Asynchronous delivery with callbacks
- Configurable compression
- Exactly-once semantics (idempotent producer)
- Automatic retry with backoff

---

## Data Model

### Span Structure

The core data structure representing a trace span:

```cpp
struct Span {
    std::string trace_id;           // 128-bit trace identifier
    std::string span_id;            // 64-bit span identifier
    std::string parent_span_id;     // Parent span (empty if root)
    std::string name;               // Operation name
    SpanKind kind;                  // CLIENT, SERVER, INTERNAL, etc.
    uint64_t start_time_unix_nano;  // Start timestamp
    uint64_t end_time_unix_nano;    // End timestamp
    StatusCode status_code;         // OK, ERROR, UNSET
    std::string status_message;     // Error message if applicable

    // Attributes
    std::unordered_map<std::string, AttributeValue> attributes;

    // Resource info
    std::string service_name;
    std::unordered_map<std::string, std::string> resource_attributes;
};
```

### AI/ML Semantic Conventions

PyFlare extends OpenTelemetry with AI/ML-specific attributes:

| Attribute | Type | Description |
|-----------|------|-------------|
| `pyflare.model.id` | string | Model identifier |
| `pyflare.model.version` | string | Model version |
| `pyflare.model.provider` | string | Provider (openai, anthropic, etc.) |
| `pyflare.inference.type` | string | llm, embedding, classification, etc. |
| `pyflare.tokens.input` | int | Input token count |
| `pyflare.tokens.output` | int | Output token count |
| `pyflare.cost.micros` | int | Cost in micro-dollars |
| `pyflare.user.id` | string | User attribution |
| `pyflare.feature.id` | string | Feature attribution |

---

## Python SDK Architecture

The Python SDK provides the client-side instrumentation.

```
                    ┌─────────────────────────────────────────┐
                    │            Python SDK                   │
                    │                                         │
                    │  ┌─────────────────────────────────────┐│
                    │  │            PyFlare                  ││
                    │  │                                     ││
                    │  │  • TracerProvider setup             ││
                    │  │  • SpanProcessor (batch)            ││
                    │  │  • OTLPSpanExporter                 ││
                    │  │  • Context management               ││
                    │  └─────────────────────────────────────┘│
                    │                                         │
                    │  ┌──────────────┐  ┌──────────────────┐ │
                    │  │  Decorators  │  │  Integrations    │ │
                    │  │              │  │                  │ │
                    │  │  @trace      │  │  OpenAI          │ │
                    │  │  @trace_llm  │  │  Anthropic       │ │
                    │  │  @trace_embed│  │  LangChain       │ │
                    │  └──────────────┘  └──────────────────┘ │
                    │                                         │
                    │  ┌──────────────────────────────────┐   │
                    │  │        Cost Calculator           │   │
                    │  │                                  │   │
                    │  │  Model pricing database          │   │
                    │  │  Automatic cost attribution      │   │
                    │  └──────────────────────────────────┘   │
                    │                                         │
                    └─────────────────────────────────────────┘
```

### SDK Components

1. **PyFlare Class**: Core SDK initialization and configuration
2. **Decorators**: `@trace`, `@trace_llm`, `@trace_embedding`
3. **Context Managers**: `with pyflare.span()`, `with pyflare.llm_span()`
4. **Integrations**: Auto-instrumentation for popular libraries
5. **Cost Calculator**: LLM cost tracking and attribution

---

## Threading Model

### Collector Threading

```
┌─────────────────────────────────────────────────────────────────────┐
│                        Collector Threads                            │
├─────────────────────────────────────────────────────────────────────┤
│                                                                     │
│  Main Thread                                                        │
│  ├── Signal handling                                                │
│  └── Lifecycle management                                           │
│                                                                     │
│  gRPC Server Threads (if enabled)                                   │
│  └── Per-connection request handling                                │
│                                                                     │
│  HTTP Server Thread (if enabled)                                    │
│  └── Request handling                                               │
│                                                                     │
│  Batcher Threads                                                    │
│  ├── Timer thread (periodic flush)                                  │
│  └── Worker threads (batch processing)                              │
│                                                                     │
│  Kafka Producer Thread                                              │
│  └── Async message delivery                                         │
│                                                                     │
└─────────────────────────────────────────────────────────────────────┘
```

### Thread Safety

All components use appropriate synchronization:
- `std::mutex` for critical sections
- `std::condition_variable` for signaling
- `std::atomic` for flags
- Lock-free queues where appropriate

---

## Configuration Flow

```
┌─────────────────┐
│  config.yaml    │
└────────┬────────┘
         │
         ▼
┌─────────────────┐     ┌─────────────────┐
│  YAML Parser    │────▶│  CollectorConfig │
│  (yaml-cpp)     │     │                 │
└─────────────────┘     │  • ReceiverConfig
                        │  • SamplerConfig
         │              │  • BatcherConfig
         │              │  • KafkaConfig
         ▼              │  • GeneralConfig
┌─────────────────┐     └─────────────────┘
│  Environment    │              │
│  Variables      │──────────────┘
│  Override       │
└─────────────────┘
         │
         ▼
┌─────────────────┐
│  CLI Arguments  │
│  Override       │
└─────────────────┘
```

Configuration priority (highest to lowest):
1. CLI arguments
2. Environment variables
3. YAML configuration file
4. Default values

---

## Error Handling

PyFlare uses `absl::Status` for error handling:

```cpp
absl::Status Collector::Start() {
    // Initialize components with error propagation
    auto status = receiver_->Start();
    if (!status.ok()) {
        return status;
    }

    status = kafka_exporter_->Connect();
    if (!status.ok()) {
        // Graceful degradation: log warning but continue
        spdlog::warn("Kafka connection failed: {}", status.message());
    }

    return absl::OkStatus();
}
```

Error handling philosophy:
- **Fail fast** for configuration errors
- **Graceful degradation** for runtime errors
- **Comprehensive logging** for debugging
- **Status propagation** through the call stack

---

## Build System

PyFlare uses CMake with vcpkg for dependency management.

```
PyFlare/
├── CMakeLists.txt          # Root CMake configuration
├── vcpkg.json              # vcpkg manifest
├── src/
│   ├── collector/
│   │   └── CMakeLists.txt  # Collector build config
│   └── common/
│       └── CMakeLists.txt  # Common library
├── tests/
│   └── CMakeLists.txt      # Test configuration
└── deploy/
    └── docker/
        └── Dockerfile.*    # Container builds
```

Key CMake features:
- vcpkg toolchain integration
- Conditional feature compilation
- Out-of-source builds
- CTest integration

---

## Phase 2 Architecture

Phase 2 extends PyFlare with advanced analytics, evaluation, and root cause analysis capabilities.

### Phase 2 System Overview

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                        PyFlare Phase 2 System                               │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│  ┌──────────────┐                                                           │
│  │  Python SDK  │────────────────────────────────────────────────┐          │
│  │  (pyflare)   │                                                │          │
│  └──────────────┘                                                │          │
│         │                                                        │          │
│         │ OTLP                                                   │          │
│         ▼                                                        │          │
│  ┌──────────────────────────────────────────────────┐            │          │
│  │              PyFlare Collector                    │            │          │
│  │  [Receiver] → [Sampler] → [Batcher] → [Exporter] │            │          │
│  └──────────────────────────────────────────────────┘            │          │
│         │                                                        │          │
│         │ Kafka                                                  │          │
│         ▼                                                        │          │
│  ┌──────────────────────────────────────────────────────────────────────┐  │
│  │                     Message Processing Pipeline                       │  │
│  │  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐  │  │
│  │  │   Drift     │  │   Cost      │  │ Evaluation  │  │    RCA      │  │  │
│  │  │  Detection  │  │  Tracking   │  │  Pipeline   │  │  Pipeline   │  │  │
│  │  │             │  │             │  │             │  │             │  │  │
│  │  │ • PSI       │  │ • Token     │  │ • LLM Judge │  │ • Slice     │  │  │
│  │  │ • MMD       │  │   Extract   │  │ • RAG Eval  │  │   Analyzer  │  │  │
│  │  │ • KS Test   │  │ • Budget    │  │ • Toxicity  │  │ • Pattern   │  │  │
│  │  │             │  │   Manager   │  │ • PII       │  │   Detector  │  │  │
│  │  └──────┬──────┘  └──────┬──────┘  └──────┬──────┘  └──────┬──────┘  │  │
│  └─────────┼────────────────┼────────────────┼────────────────┼─────────┘  │
│            │                │                │                │            │
│            ▼                ▼                ▼                ▼            │
│  ┌──────────────────────────────────────────────────────────────────────┐  │
│  │                        Storage Layer                                  │  │
│  │  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐                │  │
│  │  │  ClickHouse  │  │    Qdrant    │  │    Redis     │                │  │
│  │  │   (OLAP)     │  │  (Vectors)   │  │   (Cache)    │                │  │
│  │  │              │  │              │  │              │                │  │
│  │  │ • Traces     │  │ • Embeddings │  │ • Budget     │                │  │
│  │  │ • Costs      │  │ • References │  │   Counters   │                │  │
│  │  │ • Alerts     │  │ • Clusters   │  │ • Rate Limit │                │  │
│  │  └──────────────┘  └──────────────┘  └──────────────┘                │  │
│  └──────────────────────────────────────────────────────────────────────┘  │
│            │                                                               │
│            ▼                                                               │
│  ┌──────────────────────────────────────────────────────────────────────┐  │
│  │                        Query API                                      │  │
│  │  • /api/v1/traces    • /api/v1/drift    • /api/v1/costs              │  │
│  │  • /api/v1/evaluations                  • /api/v1/rca                │  │
│  └──────────────────────────────────────────────────────────────────────┘  │
│            │                                                               │
│            ▼                                                               │
│  ┌──────────────────────────────────────────────────────────────────────┐  │
│  │                          Web UI                                       │  │
│  │  • TraceViewer    • DriftHeatmap    • CostCharts    • RCAExplorer   │  │
│  └──────────────────────────────────────────────────────────────────────┘  │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
```

### Phase 2 Component Architecture

#### Stream Processing Pipeline

The processing pipeline consumes messages from Kafka and routes them through specialized processors:

```
                    ┌─────────────────────────────────────────────────┐
                    │             Message Pipeline                    │
                    │                                                 │
       Kafka ──────▶│  ┌─────────────────────────────────────────┐   │
                    │  │          Kafka Consumer                  │   │
                    │  │  • Consumer group: pyflare-processors    │   │
                    │  │  • Topics: pyflare.traces, pyflare.evals │   │
                    │  └─────────────────┬───────────────────────┘   │
                    │                    │                           │
                    │                    ▼                           │
                    │  ┌─────────────────────────────────────────┐   │
                    │  │        Pipeline Orchestrator             │   │
                    │  │  • Routes spans to processors            │   │
                    │  │  • Manages processor lifecycle           │   │
                    │  │  • Handles backpressure                  │   │
                    │  └─────────────────┬───────────────────────┘   │
                    │                    │                           │
                    │       ┌────────────┼────────────┐              │
                    │       ▼            ▼            ▼              │
                    │  ┌─────────┐  ┌─────────┐  ┌─────────┐        │
                    │  │  Drift  │  │  Eval   │  │  Cost   │        │
                    │  │ Proc.   │  │ Proc.   │  │ Proc.   │        │
                    │  └────┬────┘  └────┬────┘  └────┬────┘        │
                    │       └────────────┼────────────┘              │
                    │                    ▼                           │
                    │            ┌───────────────┐                   │
                    │            │    Storage    │                   │
                    │            │    Sinks      │                   │
                    │            └───────────────┘                   │
                    └─────────────────────────────────────────────────┘
```

#### Drift Detection Architecture

```
                    ┌─────────────────────────────────────────────────┐
                    │           Drift Detection Pipeline              │
                    │                                                 │
       Spans ──────▶│  ┌─────────────────────────────────────────┐   │
                    │  │         Feature Extractor                │   │
                    │  │  • Extract numerical features            │   │
                    │  │  • Extract categorical features          │   │
                    │  │  • Extract embeddings                    │   │
                    │  └─────────────────┬───────────────────────┘   │
                    │                    │                           │
                    │       ┌────────────┼────────────┐              │
                    │       ▼            ▼            ▼              │
                    │  ┌─────────┐  ┌─────────┐  ┌─────────┐        │
                    │  │   PSI   │  │   MMD   │  │ KS Test │        │
                    │  │Detector │  │Detector │  │Detector │        │
                    │  └────┬────┘  └────┬────┘  └────┬────┘        │
                    │       │            │            │              │
                    │       └────────────┼────────────┘              │
                    │                    ▼                           │
                    │  ┌─────────────────────────────────────────┐   │
                    │  │         Reference Store                  │   │
                    │  │  • Qdrant for embeddings                 │   │
                    │  │  • Redis for distributions               │   │
                    │  └─────────────────────────────────────────┘   │
                    │                    │                           │
                    │                    ▼                           │
                    │  ┌─────────────────────────────────────────┐   │
                    │  │         Alert Generator                  │   │
                    │  │  • Webhook notifications                 │   │
                    │  │  • ClickHouse storage                    │   │
                    │  └─────────────────────────────────────────┘   │
                    └─────────────────────────────────────────────────┘
```

#### Evaluation Pipeline Architecture

```
                    ┌─────────────────────────────────────────────────┐
                    │            Evaluation Pipeline                  │
                    │                                                 │
       Record ─────▶│  ┌─────────────────────────────────────────┐   │
                    │  │         Evaluation Router                │   │
                    │  │  • Route to appropriate evaluators       │   │
                    │  │  • Batch for efficiency                  │   │
                    │  └─────────────────┬───────────────────────┘   │
                    │                    │                           │
                    │   ┌────────────────┼────────────────┐          │
                    │   ▼                ▼                ▼          │
                    │  ┌──────────┐ ┌──────────┐ ┌──────────┐       │
                    │  │ LLM Judge│ │RAG Eval  │ │Toxicity  │       │
                    │  │          │ │          │ │          │       │
                    │  │ • GPT-4o │ │ • Context│ │ • Word   │       │
                    │  │ • Claude │ │   Relev. │ │   Lists  │       │
                    │  │ • Custom │ │ • Faith- │ │ • ML     │       │
                    │  │          │ │   fulness│ │   Model  │       │
                    │  └────┬─────┘ └────┬─────┘ └────┬─────┘       │
                    │       │            │            │              │
                    │       └────────────┼────────────┘              │
                    │                    ▼                           │
                    │  ┌─────────────────────────────────────────┐   │
                    │  │         Result Aggregator                │   │
                    │  │  • Combine evaluator results             │   │
                    │  │  • Calculate overall score               │   │
                    │  │  • Generate explanation                  │   │
                    │  └─────────────────────────────────────────┘   │
                    └─────────────────────────────────────────────────┘
```

#### Root Cause Analysis Architecture

```
                    ┌─────────────────────────────────────────────────┐
                    │              RCA Pipeline                       │
                    │                                                 │
    ClickHouse ────▶│  ┌─────────────────────────────────────────┐   │
                    │  │          Slice Analyzer                  │   │
                    │  │  • Analyze by dimension                  │   │
                    │  │  • Calculate deviation                   │   │
                    │  │  • Statistical significance              │   │
                    │  └─────────────────┬───────────────────────┘   │
                    │                    │                           │
                    │                    ▼                           │
                    │  ┌─────────────────────────────────────────┐   │
                    │  │         Pattern Detector                 │   │
                    │  │  • Temporal patterns                     │   │
                    │  │  • Error spikes                          │   │
                    │  │  • Correlation analysis                  │   │
                    │  └─────────────────┬───────────────────────┘   │
                    │                    │                           │
                    │                    ▼                           │
                    │  ┌─────────────────────────────────────────┐   │
                    │  │        Failure Clusterer                 │   │
                    │  │  • Text similarity clustering            │   │
                    │  │  • Embedding clustering                  │   │
                    │  │  • Representative error extraction       │   │
                    │  └─────────────────┬───────────────────────┘   │
                    │                    │                           │
                    │                    ▼                           │
                    │  ┌─────────────────────────────────────────┐   │
                    │  │        Action Generator                  │   │
                    │  │  • Suggest remediation                   │   │
                    │  │  • Prioritize by impact                  │   │
                    │  └─────────────────────────────────────────┘   │
                    └─────────────────────────────────────────────────┘
```

### Storage Schema

#### ClickHouse Tables

```sql
-- Traces table
CREATE TABLE traces (
    trace_id String,
    span_id String,
    parent_span_id String,
    service_name String,
    model_id String,
    start_time DateTime64(9),
    end_time DateTime64(9),
    latency_ms UInt64,
    status Enum('ok', 'error'),
    error_type Nullable(String),
    input_tokens UInt32,
    output_tokens UInt32,
    cost_micros UInt64,
    eval_score Nullable(Float64),
    drift_score Nullable(Float64),
    user_id Nullable(String),
    feature_id Nullable(String)
) ENGINE = MergeTree()
ORDER BY (service_name, model_id, start_time);

-- Costs aggregation table
CREATE TABLE costs (
    timestamp DateTime,
    model_id String,
    user_id String,
    feature_id String,
    request_count UInt64,
    total_tokens UInt64,
    total_cost_micros UInt64
) ENGINE = SummingMergeTree()
ORDER BY (timestamp, model_id, user_id, feature_id);

-- Drift alerts table
CREATE TABLE drift_alerts (
    id UUID,
    model_id String,
    drift_type Enum('feature', 'embedding', 'concept', 'prediction'),
    score Float64,
    severity Enum('low', 'medium', 'high'),
    affected_features Array(String),
    timestamp DateTime64(3)
) ENGINE = MergeTree()
ORDER BY (model_id, timestamp);
```

### Web UI Component Architecture

```
                    ┌─────────────────────────────────────────────────┐
                    │               React UI Architecture             │
                    │                                                 │
                    │  ┌───────────────────────────────────────────┐ │
                    │  │                 App Shell                  │ │
                    │  │  • Navigation    • Layout    • Theme       │ │
                    │  └───────────────────────────────────────────┘ │
                    │                      │                         │
                    │      ┌───────────────┼───────────────┐         │
                    │      ▼               ▼               ▼         │
                    │  ┌─────────┐   ┌─────────┐   ┌─────────┐      │
                    │  │Trace    │   │Drift    │   │Cost     │      │
                    │  │Viewer   │   │Heatmap  │   │Charts   │      │
                    │  └────┬────┘   └────┬────┘   └────┬────┘      │
                    │       │             │             │            │
                    │       └─────────────┼─────────────┘            │
                    │                     ▼                          │
                    │  ┌───────────────────────────────────────────┐ │
                    │  │              Custom Hooks                  │ │
                    │  │  useTraces  useDrift  useCosts  useRCA    │ │
                    │  └───────────────────────────────────────────┘ │
                    │                     │                          │
                    │                     ▼                          │
                    │  ┌───────────────────────────────────────────┐ │
                    │  │              API Service                   │ │
                    │  │  • Type-safe requests                      │ │
                    │  │  • Error handling                          │ │
                    │  │  • Caching                                 │ │
                    │  └───────────────────────────────────────────┘ │
                    │                     │                          │
                    │                     ▼                          │
                    │  ┌───────────────────────────────────────────┐ │
                    │  │              Query API                     │ │
                    │  └───────────────────────────────────────────┘ │
                    └─────────────────────────────────────────────────┘
```

---

## Next Steps

- [Component Guide](./components.md) - Deep dive into each component
- [Extension Guide](./extending.md) - How to add new features
- [Building](./building.md) - Build instructions
