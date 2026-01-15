# PyFlare Development Guide

> **Purpose**: This document serves as the primary instruction set for Claude Code to begin development of PyFlare, an open-source AI/ML observability platform.

---

## Project Overview

### What is PyFlare?

PyFlare is an open-source, OpenTelemetry-native observability platform purpose-built for AI/ML workloads. It extends the PyFlame ecosystem philosophy — breaking vendor lock-in for AI infrastructure — from training into production monitoring.

### Core Value Proposition

- **Deep Model Introspection**: Understand *why* models make specific decisions, not just what decisions they made
- **Multi-Model Coverage**: Unified observability for traditional ML, deep learning, and LLM applications
- **Production-First Design**: Built for scale from day one, handling millions of inferences per second
- **Zero Vendor Lock-In**: OpenTelemetry native, standard data formats, full data portability
- **Self-Hostable**: Run entirely on your infrastructure for complete data sovereignty

### Ecosystem Positioning

PyFlare completes the PyFlame family:

| Component | Purpose |
|-----------|---------|
| **PyFlame** | Train models on Cerebras without CUDA lock-in |
| **PyFlameRT** | Deploy models with optimized inference |
| **PyFlameVision** | Computer vision acceleration |
| **PyFlameAudio** | Audio signal processing |
| **PyFlare** | Observe and debug models in production |

---

## Technical Architecture

### High-Level Data Flow

```
┌─────────────────────────────────────────────────────────────────┐
│                        ML APPLICATION                           │
│  ┌─────────┐ ┌─────────┐ ┌─────────┐ ┌─────────┐ ┌─────────┐   │
│  │PyFlame  │ │PyTorch  │ │LangChain│ │ OpenAI  │ │ Custom  │   │
│  └────┬────┘ └────┬────┘ └────┬────┘ └────┬────┘ └────┬────┘   │
└───────┼──────────┼──────────┼──────────┼──────────┼───────────┘
        └──────────┴──────────┼──────────┴──────────┘
                              ▼
                    ┌─────────────────┐
                    │  PyFlare SDK    │  (OpenTelemetry)
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
           ┌────────────────┬┴─────────────────┐
           ▼                ▼                  ▼
    ┌────────────┐  ┌────────────┐     ┌────────────┐
    │   Drift    │  │ Evaluator  │     │   Cost     │
    │  Detector  │  │  Engine    │     │  Tracker   │
    └─────┬──────┘  └─────┬──────┘     └─────┬──────┘
          └───────────────┼──────────────────┘
                          ▼
           ┌──────────────┴──────────────┐
           ▼                             ▼
    ┌────────────┐                ┌────────────┐
    │ClickHouse  │                │   Qdrant   │
    │ (metrics)  │                │(embeddings)│
    └─────┬──────┘                └─────┬──────┘
          └───────────────┬─────────────┘
                          ▼
                    ┌────────────┐
                    │ Query API  │
                    └─────┬──────┘
           ┌──────────────┴──────────────┐
           ▼                             ▼
    ┌────────────┐                ┌────────────┐
    │ PyFlare UI │                │  Grafana   │
    └────────────┘                └────────────┘
```

### Layer Descriptions

1. **Collection Layer**: PyFlare SDKs instrument ML code with minimal overhead. Traces, metrics, and logs are exported via OpenTelemetry Protocol (OTLP) to the PyFlare Collector, which handles batching, sampling, and enrichment.

2. **Transport Layer**: Apache Kafka provides durable, ordered message delivery. This decouples collection from processing, enabling horizontal scaling and replay capabilities for debugging.

3. **Processing Layer**: Stream processors consume from Kafka to perform real-time analysis: drift detection, anomaly scoring, cost calculation, and evaluation. Results are written to storage and trigger alerts when thresholds are exceeded.

4. **Storage Layer**: ClickHouse stores structured telemetry data with aggressive compression. Qdrant stores embeddings for semantic analysis. Both support high-cardinality queries without cost explosion.

5. **Query Layer**: A unified query API abstracts over storage backends, providing SQL-like syntax for ad-hoc analysis. Pre-computed materialized views accelerate common dashboard queries.

6. **Presentation Layer**: The PyFlare UI provides ML-specific visualizations: trace waterfalls, drift heatmaps, embedding projections, and cost breakdowns. A Grafana plugin enables integration with existing monitoring infrastructure.

---

## Development Languages & Rationale

### Primary Languages

| Component | Language | Rationale |
|-----------|----------|-----------|
| **Core Platform** | **C++** | Ecosystem consistency with PyFlame family; shared tooling, build systems, and developer familiarity; proven performance for high-throughput data ingestion |
| **SDKs** | **Python** (primary) | Ubiquitous in ML; decorator-based instrumentation; async support |
| **Query Engine** | **C++ + SQL** | DataFusion-style analytical queries; familiar SQL interface for users |
| **Web UI** | **TypeScript/React** | Modern SPA framework; rich visualization libraries; type safety |
| **ML Analysis** | **Python** | Drift detection algorithms; embedding analysis; statistical tests |

### Why C++ (Not Rust)

The PyFlame ecosystem (PyFlame, PyFlameRT, PyFlameVision, PyFlameAudio) is built entirely in C++. Choosing C++ for PyFlare provides:

- **Ecosystem Consistency**: Shared tooling, build systems (CMake), and coding patterns
- **Developer Continuity**: Contributors familiar with PyFlame can work on PyFlare immediately
- **Code Sharing**: Common utilities (logging, memory management, error handling) can be shared
- **FFI Simplicity**: Python bindings via pybind11 are already proven in the ecosystem
- **Mature Libraries**: OpenTelemetry C++ SDK, ClickHouse C++ client are production-ready

---

## Technology Stack

### Core Infrastructure

| Layer | Technology | Purpose |
|-------|------------|---------|
| Instrumentation | OpenTelemetry | Industry standard for telemetry collection; prevents vendor lock-in |
| Data Transport | Apache Kafka | High-throughput streaming; decouples collection from storage |
| Primary Storage | ClickHouse | Columnar OLAP; 10-100x better compression than alternatives |
| Vector Storage | Qdrant | Embedding storage for semantic search and drift analysis |
| Cache | Redis | Real-time metrics aggregation; session state; rate limiting |
| Visualization | Custom + Grafana | Native UI for ML-specific views; Grafana plugin for existing dashboards |

### Required Dependencies

```yaml
# Core C++ Dependencies
cpp:
  - opentelemetry-cpp: "^1.14.0"
  - clickhouse-cpp: "^2.5.0"
  - librdkafka: "^2.3.0"
  - grpc: "^1.60.0"
  - protobuf: "^25.0"
  - abseil-cpp: "^20240116"
  - nlohmann_json: "^3.11.0"
  - spdlog: "^1.13.0"
  - fmt: "^10.2.0"

# Python SDK Dependencies
python:
  - opentelemetry-api: "^1.23.0"
  - opentelemetry-sdk: "^1.23.0"
  - opentelemetry-exporter-otlp: "^1.23.0"
  - pydantic: "^2.6.0"
  - httpx: "^0.27.0"
  - numpy: "^1.26.0"

# Web UI Dependencies
node:
  - react: "^18.2.0"
  - typescript: "^5.3.0"
  - tailwindcss: "^3.4.0"
  - recharts: "^2.12.0"
  - tanstack/react-query: "^5.24.0"
```

---

## Core Capabilities to Implement

### 1. Intelligent Drift Detection ✅ IMPLEMENTED

Multi-dimensional drift detection with advanced statistical methods:

- **Embedding Drift**: Maximum Mean Discrepancy (MMD) with RBF kernel for vector space shift detection
- **Feature Drift**: Kolmogorov-Smirnov tests, Population Stability Index (PSI) for categorical features
- **Concept Drift**: Joint distribution analysis detecting input-output relationship changes
- **Prediction Drift**: Output distribution monitoring with early warning capabilities
- **Correlation Analysis**: Multi-dimensional drift severity scoring across all drift types

```cpp
// Example drift detector interface
namespace pyflare::drift {

class DriftDetector {
public:
    virtual ~DriftDetector() = default;
    
    // Register a reference distribution (e.g., from training data)
    virtual void set_reference(const Distribution& ref) = 0;
    
    // Compute drift score for a new batch
    virtual DriftResult compute(const Distribution& current) = 0;
    
    // Get drift type
    virtual DriftType type() const = 0;
};

class EmbeddingDriftDetector : public DriftDetector {
    // Implements cosine similarity, MMD, or other embedding-specific metrics
};

class FeatureDriftDetector : public DriftDetector {
    // Implements KS test, PSI, chi-squared for tabular features
};

}  // namespace pyflare::drift
```

### 2. Hallucination & Failure Detection ✅ IMPLEMENTED

Comprehensive LLM evaluation and safety analysis:

- **Hallucination Scoring**: LLM-as-judge evaluation with configurable rubrics and semantic verification
- **RAG Quality Analysis**: Relevance scoring, groundedness checking, context utilization metrics
- **Toxicity & Safety**: Multi-category toxicity detection with configurable thresholds
- **Prompt Injection Detection**: Pattern-based and semantic detection of adversarial inputs
- **PII Detection**: Automatic identification of sensitive data in inputs/outputs
- **Semantic Similarity**: Embedding-based coherence and consistency checking

```cpp
// Example evaluator interface
namespace pyflare::eval {

class Evaluator {
public:
    virtual ~Evaluator() = default;
    
    // Evaluate a single inference
    virtual EvalResult evaluate(const InferenceRecord& record) = 0;
    
    // Batch evaluation
    virtual std::vector<EvalResult> evaluate_batch(
        const std::vector<InferenceRecord>& records) = 0;
};

class HallucinationEvaluator : public Evaluator {
    // Uses LLM-as-judge or embedding-based factuality checking
};

class RAGEvaluator : public Evaluator {
    // Evaluates retrieval quality, context relevance, answer groundedness
};

}  // namespace pyflare::eval
```

### 3. Root Cause Analysis Engine ✅ IMPLEMENTED

Intelligent automated root cause analysis with causal reasoning:

- **Multi-Phase Analysis**: Systematic investigation through data collection, pattern detection, causal analysis, and recommendation phases
- **Anomaly Clustering**: DBSCAN-based clustering of failures with pattern extraction
- **Slice Analysis**: Automatic identification of underperforming data segments with impact scoring
- **Causal Factor Identification**: Root cause detection with confidence scoring and evidence collection
- **Actionable Recommendations**: Prioritized remediation suggestions with expected impact
- **Temporal Correlation**: Link model behavior changes to deployment events and external factors

```cpp
namespace pyflare::rca {

class RootCauseAnalyzer {
public:
    // Analyze a set of failures and identify common patterns
    virtual RCAReport analyze(const std::vector<FailureRecord>& failures) = 0;
    
    // Find underperforming slices in the data
    virtual std::vector<Slice> find_problematic_slices(
        const Dataset& data,
        const std::string& metric) = 0;
    
    // Generate counterfactual explanations
    virtual Counterfactual explain(
        const InferenceRecord& record,
        const std::string& target_outcome) = 0;
};

}  // namespace pyflare::rca
```

### 4. Unified Tracing

End-to-end visibility across complex ML pipelines:

- **Multi-Step Agent Tracing**: Follow agent workflows from input through tool calls to response
- **RAG Pipeline Visibility**: Trace query → embedding → retrieval → reranking → generation
- **Model Cascade Tracking**: Monitor multi-model architectures
- **Latency Breakdown**: Identify bottlenecks at each inference stage

### 5. Cost Intelligence

Granular cost tracking and optimization:

- **Per-Request Cost Attribution**: Track costs by user, feature, model version, or custom dimensions
- **Token Economics**: Detailed input/output token analysis with cost projections
- **Budget Alerts**: Configurable thresholds with automatic notifications
- **Optimization Recommendations**: AI-driven suggestions for prompt optimization, caching, model selection

### 6. Intelligent Alerting System ✅ IMPLEMENTED

Comprehensive alerting with noise reduction:

- **Rule Types**: Threshold, anomaly detection, rate-based, pattern matching, and composite rules
- **Alert Deduplication**: Fingerprint-based deduplication with configurable windows
- **Alert Grouping**: Automatic grouping by labels, model, or custom dimensions
- **Silences**: Time-based or matcher-based alert suppression
- **Maintenance Windows**: Scheduled maintenance periods with automatic alert suppression
- **Multi-Channel Notifications**: Slack, PagerDuty, webhooks, email with retry logic
- **Rate Limiting**: Configurable rate limits per channel to prevent notification storms

```cpp
// Alert rule configuration
namespace pyflare::alerting {

struct AlertRule {
    std::string id;
    std::string name;
    RuleType type;  // kThreshold, kAnomaly, kRate, kPattern, kComposite
    AlertSeverity severity;
    std::chrono::seconds evaluation_interval;
    bool enabled;
};

}  // namespace pyflare::alerting
```

### 7. Intelligence Pipeline ✅ IMPLEMENTED

Unified orchestration of all intelligence components:

- **Trace Analysis**: Automatic analysis of incoming traces through all processors
- **Model Health Scoring**: Composite health scores based on drift, evaluation, and safety metrics
- **System Health Aggregation**: Platform-wide health monitoring across all models
- **Background Processing**: Async workers for non-blocking analysis
- **Result Correlation**: Cross-component result aggregation and causality detection

```cpp
// Intelligence pipeline configuration
namespace pyflare::intelligence {

struct IntelligenceResult {
    std::string trace_id;
    std::string model_id;
    double health_score;
    DriftAnalysisResult drift;
    EvaluationResult evaluation;
    SafetyResult safety;
    std::optional<RCAReport> rca;
};

}  // namespace pyflare::intelligence
```

---

## Project Structure

```
pyflare/
├── CMakeLists.txt
├── README.md
├── LICENSE                      # Apache 2.0
├── docs/
│   └── ...
├── src/
│   ├── collector/               # OTLP collector service
│   │   ├── CMakeLists.txt
│   │   ├── collector.cpp
│   │   ├── collector.h
│   │   ├── otlp_receiver.cpp
│   │   ├── otlp_receiver.h
│   │   ├── kafka_exporter.cpp
│   │   └── kafka_exporter.h
│   ├── processor/               # Stream processing
│   │   ├── CMakeLists.txt
│   │   ├── drift/               # Drift detection (Phase 3)
│   │   │   ├── drift_detector.h
│   │   │   ├── embedding_drift.cpp/.h
│   │   │   ├── feature_drift.cpp/.h
│   │   │   ├── concept_drift.cpp/.h
│   │   │   ├── prediction_drift.cpp/.h
│   │   │   ├── psi_detector.cpp/.h       # Population Stability Index
│   │   │   ├── mmd_detector.cpp/.h       # Maximum Mean Discrepancy
│   │   │   └── reference_store.cpp/.h    # Qdrant reference storage
│   │   ├── eval/                # Evaluators (Phase 3)
│   │   │   ├── evaluator.h
│   │   │   ├── hallucination.cpp/.h
│   │   │   ├── rag_quality.cpp/.h
│   │   │   ├── toxicity.cpp/.h
│   │   │   ├── safety_analyzer.cpp/.h    # PII, injection detection
│   │   │   └── semantic_similarity.cpp/.h
│   │   ├── rca/                 # Root Cause Analysis (Phase 3)
│   │   │   ├── rca_service.cpp/.h        # Main RCA orchestration
│   │   │   ├── analyzer.h
│   │   │   ├── clustering.cpp/.h         # Failure clustering
│   │   │   ├── slice_finder.cpp/.h       # Problematic slice detection
│   │   │   ├── pattern_detector.cpp/.h   # Pattern extraction
│   │   │   └── counterfactual.cpp
│   │   ├── alerting/            # Alerting System (Phase 3)
│   │   │   ├── alert_service.cpp/.h      # Main alert service
│   │   │   ├── alert_rules.cpp/.h        # Rule engine
│   │   │   └── deduplicator.cpp/.h       # Dedup, silences, maintenance
│   │   ├── intelligence/        # Intelligence Pipeline (Phase 3)
│   │   │   └── intelligence_pipeline.cpp/.h
│   │   └── cost/
│   │       ├── tracker.h
│   │       └── tracker.cpp
│   ├── storage/                 # Storage adapters
│   │   ├── CMakeLists.txt
│   │   ├── clickhouse/
│   │   │   ├── client.cpp
│   │   │   └── client.h
│   │   ├── qdrant/
│   │   │   ├── client.cpp
│   │   │   └── client.h
│   │   └── redis/
│   │       ├── client.cpp
│   │       └── client.h
│   ├── query/                   # Query API
│   │   ├── CMakeLists.txt
│   │   ├── api.cpp
│   │   ├── api.h
│   │   ├── sql_parser.cpp
│   │   └── handlers/            # REST API Handlers (Phase 3)
│   │       ├── intelligence_handler.cpp/.h
│   │       ├── alerts_handler.cpp/.h
│   │       └── rca_handler.cpp/.h
│   └── common/                  # Shared utilities
│       ├── CMakeLists.txt
│       ├── logging.h
│       ├── metrics.h
│       └── config.h
├── sdk/
│   └── python/                  # Python SDK
│       ├── pyproject.toml
│       ├── pyflare/
│       │   ├── __init__.py
│       │   ├── sdk.py
│       │   ├── decorators.py
│       │   ├── exporters.py
│       │   └── integrations/
│       │       ├── __init__.py
│       │       ├── langchain.py
│       │       ├── openai.py
│       │       ├── pytorch.py
│       │       └── pyflame.py
│       └── tests/
├── ui/                          # Web UI
│   ├── package.json
│   ├── tsconfig.json
│   ├── src/
│   │   ├── App.tsx
│   │   ├── components/          # UI Components (Phase 3)
│   │   │   ├── IntelligenceDashboard.tsx
│   │   │   ├── AlertsPanel.tsx
│   │   │   └── RCAExplorer.tsx
│   │   ├── pages/
│   │   ├── services/
│   │   │   └── api.ts
│   │   └── api/
│   └── public/
├── deploy/
│   ├── docker/
│   │   ├── Dockerfile.collector
│   │   ├── Dockerfile.processor
│   │   ├── Dockerfile.query
│   │   └── docker-compose.yml
│   └── kubernetes/
│       └── helm/
│           └── pyflare/
├── tests/
│   ├── unit/
│   │   └── processor/
│   │       ├── intelligence/
│   │       │   └── intelligence_pipeline_test.cpp
│   │       └── alerting/
│   │           └── alerting_test.cpp
│   ├── integration/
│   └── e2e/
└── scripts/
    ├── build.sh
    ├── test.sh
    └── lint.sh
```

---

## Development Phases

### Phase 1: Foundation (Weeks 1-4)

1. **Project Setup**
   - Initialize CMake build system with proper C++20 configuration
   - Set up CI/CD pipeline (GitHub Actions)
   - Configure linting (clang-format, clang-tidy) and testing (Google Test)

2. **Core Collector**
   - Implement OTLP gRPC receiver
   - Basic Kafka producer
   - Configuration management

3. **Python SDK (Basic)**
   - OpenTelemetry-based instrumentation
   - Simple decorator API: `@pyflare.trace`
   - OTLP exporter to collector

### Phase 2: Storage & Processing (Weeks 5-8)

1. **Storage Layer**
   - ClickHouse schema design for traces, metrics, logs
   - Qdrant integration for embeddings
   - Data retention policies

2. **Stream Processing**
   - Kafka consumer framework
   - Basic drift detection (feature drift)
   - Cost tracking pipeline

3. **Query API**
   - REST API for trace retrieval
   - Basic SQL query support
   - GraphQL schema (optional)

### Phase 3: Intelligence (Weeks 9-12) ✅ COMPLETE

1. **Advanced Drift Detection** ✅
   - Embedding drift with Maximum Mean Discrepancy (MMD) and RBF kernel
   - Concept drift detection with joint distribution analysis
   - Prediction drift monitoring
   - Population Stability Index (PSI) for categorical features
   - Multi-dimensional drift correlation

2. **Enhanced Evaluators** ✅
   - Hallucination detection with LLM-as-judge
   - RAG quality metrics (relevance, groundedness, context utilization)
   - Toxicity detection with multi-category scoring
   - Semantic similarity evaluation
   - Safety analysis (PII detection, prompt injection, content safety)

3. **Intelligent Root Cause Analysis** ✅
   - Multi-phase analysis engine
   - Failure clustering with pattern detection
   - Data slice analysis for underperforming segments
   - Causal factor identification with confidence scoring
   - Actionable recommendations generation

4. **Alerting System** ✅
   - Rule-based alerting (threshold, anomaly, rate, pattern, composite)
   - Alert deduplication and grouping
   - Silences and maintenance windows
   - Multi-channel notifications (Slack, PagerDuty, webhooks, email)
   - Rate limiting and escalation

5. **Intelligence Pipeline** ✅
   - Unified orchestration of all intelligence components
   - Model-level health scoring
   - System-wide health aggregation
   - Real-time processing with background workers

6. **API Extensions** ✅
   - `/api/v1/intelligence/*` - Intelligence operations
   - `/api/v1/alerts/*` - Alert management, rules, silences
   - `/api/v1/rca/*` - Root cause analysis endpoints

7. **UI Components** ✅
   - Intelligence Dashboard with system/model health
   - Alerts Panel with rules and silence management
   - RCA Explorer for root cause investigation

### Phase 4: UI & Polish (Weeks 13-16)

1. **Web UI**
   - Trace explorer
   - Drift dashboards
   - Cost analytics

2. **Integrations**
   - LangChain integration
   - OpenAI integration
   - PyFlame native integration

3. **Documentation & Testing**
   - API documentation
   - User guides
   - Performance benchmarks

---

## Platform Support

### Deployment Options

| Platform | Support Details |
|----------|-----------------|
| **Self-Hosted** | Docker Compose for development; Kubernetes (Helm charts) for production |
| **AWS** | Native integration with SageMaker, Bedrock, EKS; CloudFormation templates |
| **GCP** | Integration with Vertex AI, GKE, Cloud Run; Terraform modules |
| **Azure** | Support for Azure ML, AKS, Azure OpenAI; ARM templates |
| **On-Premise** | Air-gapped deployment support for regulated industries |

### Framework Integrations

- **Deep Learning**: PyFlame (native), PyTorch, TensorFlow, JAX
- **LLM Frameworks**: LangChain, LlamaIndex, DSPy, Haystack
- **LLM Providers**: OpenAI, Anthropic, Google (Gemini), Mistral, Cohere, AWS Bedrock, Azure OpenAI
- **Traditional ML**: scikit-learn, XGBoost, LightGBM, CatBoost
- **Orchestration**: Airflow, Prefect, Dagster, Kubeflow

---

## Coding Standards

### C++ Standards

- Use C++20 features where appropriate
- Follow Google C++ Style Guide with PyFlame-specific modifications
- Use `namespace pyflare` for all code
- Prefer `std::unique_ptr` and `std::shared_ptr` over raw pointers
- Use `absl::StatusOr<T>` for functions that can fail
- All public APIs must be documented with Doxygen comments

### Python Standards

- Python 3.10+ required
- Use type hints throughout
- Follow PEP 8 with Black formatting
- Use Pydantic for data validation
- Async-first design for I/O operations

### Testing Requirements

- Minimum 80% code coverage for new code
- Unit tests for all public APIs
- Integration tests for cross-component interactions
- Performance benchmarks for critical paths

---

## Licensing

- **Core Platform**: Apache 2.0 License — fully open source, no restrictions on commercial use
- **Enterprise Features (Future)**: Optional paid add-ons for SSO/SAML, advanced RBAC, priority support

---

## Getting Started (For Claude Code)

1. Begin by creating the project structure as outlined above
2. Start with the `src/common/` utilities (logging, config)
3. Implement the collector OTLP receiver
4. Create the basic Python SDK with trace decorator
5. Add ClickHouse storage integration
6. Iterate from there based on the development phases

### First Commands

```bash
# Create project structure
mkdir -p pyflare/{src,sdk,ui,deploy,tests,scripts,docs}
mkdir -p pyflare/src/{collector,processor,storage,query,common}
mkdir -p pyflare/src/processor/{drift,eval,rca,cost}
mkdir -p pyflare/sdk/python/pyflare/integrations
mkdir -p pyflare/deploy/{docker,kubernetes}

# Initialize CMake
cd pyflare
touch CMakeLists.txt

# Initialize Python SDK
cd sdk/python
touch pyproject.toml
```

---

## Questions for Development

If clarification is needed during development, prioritize:

1. **Performance requirements**: What's the target throughput (inferences/second)?
2. **Initial integration priority**: Which framework integration should be first?
3. **UI priority**: Should UI development happen in parallel or after backend is stable?
4. **Cloud-first vs self-hosted-first**: Which deployment model to optimize for initially?

---

*This document is the authoritative guide for PyFlare development. Update it as architecture decisions are made and requirements evolve.*