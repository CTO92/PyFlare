# PyFlare Technical Architecture

> **Document Version**: 1.1
> **Status**: Phase 3 Complete
> **Last Updated**: 2026-01-15

---

## Table of Contents

1. [Executive Summary](#executive-summary)
2. [System Architecture](#system-architecture)
3. [Component Specifications](#component-specifications)
4. [Data Models & Storage Schemas](#data-models--storage-schemas)
5. [API Contracts](#api-contracts)
6. [Communication Protocols](#communication-protocols)
7. [Security Architecture](#security-architecture)
8. [Development Plan](#development-plan)
9. [Testing Strategy](#testing-strategy)
10. [Deployment Architecture](#deployment-architecture)
11. [Appendices](#appendices)

---

## 1. Executive Summary

PyFlare is an OpenTelemetry-native observability platform for AI/ML workloads. This document defines the technical architecture, component specifications, and development roadmap for building a production-grade system capable of handling millions of inferences per second.

### Key Architectural Decisions

| Decision | Choice | Rationale |
|----------|--------|-----------|
| Core Language | C++20 | Ecosystem consistency with PyFlame; proven performance |
| Telemetry Standard | OpenTelemetry | Industry standard; prevents vendor lock-in |
| Message Transport | Apache Kafka | High throughput; replay capability; proven at scale |
| Metrics Storage | ClickHouse | Columnar OLAP; excellent compression; fast aggregations |
| Vector Storage | Qdrant | Purpose-built for embeddings; efficient ANN search |
| API Protocol | gRPC + REST | gRPC for performance; REST for accessibility |
| Configuration | YAML + Environment | Human-readable; container-friendly |

### Non-Functional Requirements

| Requirement | Target |
|-------------|--------|
| Throughput | 1M+ inferences/second (clustered) |
| Latency (p99) | < 10ms for trace ingestion |
| Storage Efficiency | 10:1 compression ratio minimum |
| Availability | 99.9% uptime (self-hosted) |
| Data Retention | Configurable (default 30 days hot, 1 year cold) |

---

## 2. System Architecture

### 2.1 High-Level Architecture Diagram

```
┌─────────────────────────────────────────────────────────────────────────────────┐
│                              CLIENT APPLICATIONS                                 │
│  ┌──────────┐ ┌──────────┐ ┌──────────┐ ┌──────────┐ ┌──────────┐              │
│  │ PyFlame  │ │ PyTorch  │ │LangChain │ │  OpenAI  │ │  Custom  │              │
│  │   App    │ │   App    │ │   App    │ │   App    │ │   App    │              │
│  └────┬─────┘ └────┬─────┘ └────┬─────┘ └────┬─────┘ └────┬─────┘              │
│       │            │            │            │            │                     │
│       └────────────┴────────────┼────────────┴────────────┘                     │
│                                 │                                               │
│                    ┌────────────▼────────────┐                                  │
│                    │     PyFlare SDK         │                                  │
│                    │  (Python/OpenTelemetry) │                                  │
│                    └────────────┬────────────┘                                  │
└─────────────────────────────────┼───────────────────────────────────────────────┘
                                  │ OTLP (gRPC/HTTP)
                                  ▼
┌─────────────────────────────────────────────────────────────────────────────────┐
│                           COLLECTION LAYER                                       │
│  ┌──────────────────────────────────────────────────────────────────────────┐   │
│  │                        PyFlare Collector Cluster                          │   │
│  │  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐      │   │
│  │  │ Collector-1 │  │ Collector-2 │  │ Collector-3 │  │ Collector-N │      │   │
│  │  │             │  │             │  │             │  │             │      │   │
│  │  │ ┌─────────┐ │  │ ┌─────────┐ │  │ ┌─────────┐ │  │ ┌─────────┐ │      │   │
│  │  │ │  OTLP   │ │  │ │  OTLP   │ │  │ │  OTLP   │ │  │ │  OTLP   │ │      │   │
│  │  │ │Receiver │ │  │ │Receiver │ │  │ │Receiver │ │  │ │Receiver │ │      │   │
│  │  │ └────┬────┘ │  │ └────┬────┘ │  │ └────┬────┘ │  │ └────┬────┘ │      │   │
│  │  │      │      │  │      │      │  │      │      │  │      │      │      │   │
│  │  │ ┌────▼────┐ │  │ ┌────▼────┐ │  │ ┌────▼────┐ │  │ ┌────▼────┐ │      │   │
│  │  │ │ Batcher │ │  │ │ Batcher │ │  │ │ Batcher │ │  │ │ Batcher │ │      │   │
│  │  │ │& Sampler│ │  │ │& Sampler│ │  │ │& Sampler│ │  │ │& Sampler│ │      │   │
│  │  │ └────┬────┘ │  │ └────┬────┘ │  │ └────┬────┘ │  │ └────┬────┘ │      │   │
│  │  │      │      │  │      │      │  │      │      │  │      │      │      │   │
│  │  │ ┌────▼────┐ │  │ ┌────▼────┐ │  │ ┌────▼────┐ │  │ ┌────▼────┐ │      │   │
│  │  │ │  Kafka  │ │  │ │  Kafka  │ │  │ │  Kafka  │ │  │ │  Kafka  │ │      │   │
│  │  │ │Producer │ │  │ │Producer │ │  │ │Producer │ │  │ │Producer │ │      │   │
│  │  │ └─────────┘ │  │ └─────────┘ │  │ └─────────┘ │  │ └─────────┘ │      │   │
│  │  └─────────────┘  └─────────────┘  └─────────────┘  └─────────────┘      │   │
│  └──────────────────────────────────────────────────────────────────────────┘   │
└─────────────────────────────────┬───────────────────────────────────────────────┘
                                  │
                                  ▼
┌─────────────────────────────────────────────────────────────────────────────────┐
│                           TRANSPORT LAYER                                        │
│  ┌──────────────────────────────────────────────────────────────────────────┐   │
│  │                         Apache Kafka Cluster                              │   │
│  │                                                                           │   │
│  │   Topics:                                                                 │   │
│  │   ┌─────────────────┐  ┌─────────────────┐  ┌─────────────────┐          │   │
│  │   │ pyflare.traces  │  │ pyflare.metrics │  │  pyflare.logs   │          │   │
│  │   │   (partitioned  │  │   (partitioned  │  │   (partitioned  │          │   │
│  │   │    by trace_id) │  │    by model_id) │  │    by service)  │          │   │
│  │   └────────┬────────┘  └────────┬────────┘  └────────┬────────┘          │   │
│  │            │                    │                    │                    │   │
│  │   ┌────────▼────────┐  ┌────────▼────────┐  ┌────────▼────────┐          │   │
│  │   │pyflare.enriched │  │pyflare.alerts   │  │pyflare.embeddings│         │   │
│  │   │    .traces      │  │                 │  │                 │          │   │
│  │   └─────────────────┘  └─────────────────┘  └─────────────────┘          │   │
│  └──────────────────────────────────────────────────────────────────────────┘   │
└─────────────────────────────────┬───────────────────────────────────────────────┘
                                  │
                                  ▼
┌─────────────────────────────────────────────────────────────────────────────────┐
│                          PROCESSING LAYER                                        │
│                                                                                  │
│  ┌────────────────┐  ┌────────────────┐  ┌────────────────┐  ┌────────────────┐ │
│  │ Drift Detector │  │   Evaluator    │  │  Cost Tracker  │  │  RCA Engine    │ │
│  │    Service     │  │    Service     │  │    Service     │  │    Service     │ │
│  │                │  │                │  │                │  │                │ │
│  │ - Embedding    │  │ - Hallucination│  │ - Token Count  │  │ - Clustering   │ │
│  │ - Feature      │  │ - RAG Quality  │  │ - Cost Calc    │  │ - Slice Find   │ │
│  │ - Concept      │  │ - Toxicity     │  │ - Budget Check │  │ - Counterfact  │ │
│  │ - Prediction   │  │ - Custom       │  │ - Attribution  │  │ - Correlation  │ │
│  └───────┬────────┘  └───────┬────────┘  └───────┬────────┘  └───────┬────────┘ │
│          │                   │                   │                   │          │
│          └───────────────────┴─────────┬─────────┴───────────────────┘          │
│                                        │                                         │
└────────────────────────────────────────┼────────────────────────────────────────┘
                                         │
                                         ▼
┌─────────────────────────────────────────────────────────────────────────────────┐
│                           STORAGE LAYER                                          │
│                                                                                  │
│  ┌─────────────────────────────────┐    ┌─────────────────────────────────┐     │
│  │         ClickHouse Cluster      │    │         Qdrant Cluster          │     │
│  │                                 │    │                                 │     │
│  │  ┌───────────┐ ┌───────────┐   │    │  ┌───────────┐ ┌───────────┐   │     │
│  │  │  Traces   │ │  Metrics  │   │    │  │ Inference │ │ Reference │   │     │
│  │  │  Table    │ │  Table    │   │    │  │Embeddings │ │Embeddings │   │     │
│  │  └───────────┘ └───────────┘   │    │  └───────────┘ └───────────┘   │     │
│  │  ┌───────────┐ ┌───────────┐   │    │  ┌───────────┐ ┌───────────┐   │     │
│  │  │   Logs    │ │   Costs   │   │    │  │   Drift   │ │  Anomaly  │   │     │
│  │  │  Table    │ │  Table    │   │    │  │  Vectors  │ │  Vectors  │   │     │
│  │  └───────────┘ └───────────┘   │    │  └───────────┘ └───────────┘   │     │
│  │  ┌───────────┐ ┌───────────┐   │    │                                 │     │
│  │  │  Alerts   │ │Materialized│  │    │                                 │     │
│  │  │  Table    │ │   Views   │   │    │                                 │     │
│  │  └───────────┘ └───────────┘   │    │                                 │     │
│  └─────────────────────────────────┘    └─────────────────────────────────┘     │
│                                                                                  │
│  ┌─────────────────────────────────┐                                            │
│  │            Redis                │                                            │
│  │  - Rate Limiting                │                                            │
│  │  - Session State                │                                            │
│  │  - Real-time Aggregations       │                                            │
│  │  - Cache Layer                  │                                            │
│  └─────────────────────────────────┘                                            │
└─────────────────────────────────────────────────────────────────────────────────┘
                                         │
                                         ▼
┌─────────────────────────────────────────────────────────────────────────────────┐
│                            QUERY LAYER                                           │
│  ┌──────────────────────────────────────────────────────────────────────────┐   │
│  │                         PyFlare Query API                                 │   │
│  │                                                                           │   │
│  │  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐      │   │
│  │  │  REST API   │  │  gRPC API   │  │ SQL Engine  │  │  GraphQL    │      │   │
│  │  │  (Public)   │  │ (Internal)  │  │  (Query)    │  │ (Optional)  │      │   │
│  │  └─────────────┘  └─────────────┘  └─────────────┘  └─────────────┘      │   │
│  │                                                                           │   │
│  │  ┌─────────────────────────────────────────────────────────────────┐     │   │
│  │  │                    Query Optimizer & Planner                     │     │   │
│  │  │  - Query parsing and validation                                  │     │   │
│  │  │  - Materialized view routing                                     │     │   │
│  │  │  - Cross-storage query federation                                │     │   │
│  │  └─────────────────────────────────────────────────────────────────┘     │   │
│  └──────────────────────────────────────────────────────────────────────────┘   │
└─────────────────────────────────────────────────────────────────────────────────┘
                                         │
                                         ▼
┌─────────────────────────────────────────────────────────────────────────────────┐
│                         PRESENTATION LAYER                                       │
│                                                                                  │
│  ┌─────────────────────────────────┐    ┌─────────────────────────────────┐     │
│  │        PyFlare Web UI           │    │      Grafana Integration        │     │
│  │                                 │    │                                 │     │
│  │  ┌───────────┐ ┌───────────┐   │    │  ┌───────────┐ ┌───────────┐   │     │
│  │  │  Trace    │ │   Drift   │   │    │  │  PyFlare  │ │  Custom   │   │     │
│  │  │ Explorer  │ │ Dashboard │   │    │  │  Plugin   │ │Dashboards │   │     │
│  │  └───────────┘ └───────────┘   │    │  └───────────┘ └───────────┘   │     │
│  │  ┌───────────┐ ┌───────────┐   │    │                                 │     │
│  │  │   Cost    │ │  Alerts   │   │    │                                 │     │
│  │  │ Analytics │ │  Center   │   │    │                                 │     │
│  │  └───────────┘ └───────────┘   │    │                                 │     │
│  └─────────────────────────────────┘    └─────────────────────────────────┘     │
└─────────────────────────────────────────────────────────────────────────────────┘
```

### 2.2 Component Interaction Flow

```
                                    Request Flow
                                    ============

  [ML App] ──OTLP──► [Collector] ──Kafka──► [Processor] ──Write──► [Storage]
                          │                      │                      │
                          │                      │                      │
                          ▼                      ▼                      ▼
                    [Sampling]             [Analysis]              [Indexing]
                    [Batching]             [Alerting]             [Compaction]
                    [Enrichment]           [Scoring]              [Retention]


                                    Query Flow
                                    ==========

  [UI/API] ──Query──► [Query API] ──SQL──► [ClickHouse]
      │                    │                    │
      │                    │                    │
      │                    └──Vector──► [Qdrant]
      │                    │
      │                    └──Cache──► [Redis]
      │
      └──Subscribe──► [WebSocket] ──Stream──► [Kafka]
```

---

## 3. Component Specifications

### 3.1 PyFlare Collector

The collector is the entry point for all telemetry data. It receives OTLP data, processes it, and forwards to Kafka.

#### 3.1.1 Architecture

```cpp
namespace pyflare::collector {

// Core collector class
class Collector {
public:
    struct Config {
        std::string listen_address = "0.0.0.0:4317";  // gRPC
        std::string http_address = "0.0.0.0:4318";    // HTTP
        size_t max_batch_size = 1000;
        std::chrono::milliseconds batch_timeout{100};
        double sampling_rate = 1.0;
        std::vector<std::string> kafka_brokers;
    };

    explicit Collector(Config config);

    // Lifecycle
    absl::Status Start();
    absl::Status Shutdown();

    // Health
    bool IsHealthy() const;
    CollectorStats GetStats() const;

private:
    std::unique_ptr<OtlpReceiver> otlp_receiver_;
    std::unique_ptr<Batcher> batcher_;
    std::unique_ptr<Sampler> sampler_;
    std::unique_ptr<KafkaExporter> kafka_exporter_;
};

}  // namespace pyflare::collector
```

#### 3.1.2 Sub-components

| Component | Responsibility |
|-----------|----------------|
| `OtlpReceiver` | Accept OTLP/gRPC and OTLP/HTTP requests |
| `Batcher` | Aggregate spans/metrics into batches for efficient Kafka writes |
| `Sampler` | Implement head-based and tail-based sampling strategies |
| `Enricher` | Add metadata (hostname, version, environment) |
| `KafkaExporter` | Produce messages to appropriate Kafka topics |

#### 3.1.3 Configuration Schema

```yaml
# collector.yaml
collector:
  # OTLP receiver configuration
  otlp:
    grpc:
      endpoint: "0.0.0.0:4317"
      max_recv_msg_size_mib: 16
      max_concurrent_streams: 100
    http:
      endpoint: "0.0.0.0:4318"
      cors:
        allowed_origins: ["*"]

  # Batching configuration
  batcher:
    max_batch_size: 1000
    timeout_ms: 100
    max_queue_size: 10000

  # Sampling configuration
  sampling:
    default_rate: 1.0  # 100% sampling by default
    rules:
      - service: "high-volume-service"
        rate: 0.1  # 10% sampling
      - attribute:
          key: "priority"
          value: "high"
        rate: 1.0  # Always sample high priority

  # Kafka exporter configuration
  kafka:
    brokers:
      - "kafka-1:9092"
      - "kafka-2:9092"
      - "kafka-3:9092"
    topics:
      traces: "pyflare.traces"
      metrics: "pyflare.metrics"
      logs: "pyflare.logs"
    producer:
      compression: "lz4"
      batch_size: 16384
      linger_ms: 5
      acks: "all"
```

---

### 3.2 Stream Processors

#### 3.2.1 Drift Detector Service

```cpp
namespace pyflare::drift {

// Drift detection result
struct DriftResult {
    DriftType type;
    double score;          // 0.0 - 1.0
    double threshold;
    bool is_drifted;
    std::string explanation;
    std::chrono::system_clock::time_point detected_at;
    std::unordered_map<std::string, double> feature_scores;  // Per-feature breakdown
};

enum class DriftType {
    kFeature,      // Input distribution shift
    kEmbedding,    // Vector space shift
    kConcept,      // Input-output relationship change
    kPrediction    // Output distribution shift
};

// Abstract drift detector interface
class DriftDetector {
public:
    virtual ~DriftDetector() = default;

    // Set reference distribution (from training data)
    virtual absl::Status SetReference(const Distribution& reference) = 0;

    // Compute drift for a batch of data
    virtual absl::StatusOr<DriftResult> Compute(
        const std::vector<DataPoint>& current_batch) = 0;

    // Get detector type
    virtual DriftType Type() const = 0;

    // Serialize/deserialize state
    virtual absl::StatusOr<std::string> SerializeState() const = 0;
    virtual absl::Status LoadState(std::string_view state) = 0;
};

// Feature drift using statistical tests
class FeatureDriftDetector : public DriftDetector {
public:
    struct Config {
        StatisticalTest test = StatisticalTest::kKolmogorovSmirnov;
        double significance_level = 0.05;
        size_t min_sample_size = 100;
    };

    explicit FeatureDriftDetector(Config config);
    // ... implementation
};

// Embedding drift using distance metrics
class EmbeddingDriftDetector : public DriftDetector {
public:
    struct Config {
        DistanceMetric metric = DistanceMetric::kCosineSimilarity;
        double threshold = 0.1;
        size_t reference_sample_size = 10000;
    };

    explicit EmbeddingDriftDetector(Config config);
    // ... implementation
};

// Drift detection service (Kafka consumer)
class DriftDetectorService {
public:
    struct Config {
        std::vector<std::string> kafka_brokers;
        std::string consumer_group = "drift-detector";
        std::string input_topic = "pyflare.traces";
        std::string output_topic = "pyflare.alerts";
        std::chrono::seconds window_size{300};  // 5-minute windows
    };

    explicit DriftDetectorService(Config config);

    absl::Status Start();
    absl::Status Stop();

    // Register detectors for specific models
    void RegisterDetector(
        const std::string& model_id,
        std::unique_ptr<DriftDetector> detector);

private:
    void ProcessBatch(const std::vector<TraceRecord>& records);
    void EmitAlert(const std::string& model_id, const DriftResult& result);
};

}  // namespace pyflare::drift
```

#### 3.2.2 Evaluator Service

```cpp
namespace pyflare::eval {

// Evaluation result
struct EvalResult {
    std::string evaluator_type;
    double score;              // 0.0 - 1.0 (higher = better)
    std::string verdict;       // "pass", "fail", "warn"
    std::string explanation;
    std::unordered_map<std::string, std::string> metadata;
};

// Inference record for evaluation
struct InferenceRecord {
    std::string trace_id;
    std::string model_id;
    std::string input;
    std::string output;
    std::optional<std::string> expected_output;
    std::optional<std::vector<std::string>> retrieved_contexts;  // For RAG
    std::unordered_map<std::string, std::string> attributes;
};

// Abstract evaluator interface
class Evaluator {
public:
    virtual ~Evaluator() = default;

    virtual absl::StatusOr<EvalResult> Evaluate(
        const InferenceRecord& record) = 0;

    virtual absl::StatusOr<std::vector<EvalResult>> EvaluateBatch(
        const std::vector<InferenceRecord>& records) = 0;

    virtual std::string Type() const = 0;
};

// LLM-as-judge hallucination detector
class HallucinationEvaluator : public Evaluator {
public:
    struct Config {
        std::string judge_model = "gpt-4";  // or local model
        std::string judge_endpoint;
        std::string rubric_template;
        double threshold = 0.7;
    };

    explicit HallucinationEvaluator(Config config);
    // ... implementation
};

// RAG quality evaluator
class RAGEvaluator : public Evaluator {
public:
    struct Config {
        bool check_relevance = true;
        bool check_groundedness = true;
        bool check_context_utilization = true;
        double relevance_threshold = 0.6;
    };

    explicit RAGEvaluator(Config config);
    // ... implementation
};

// Toxicity evaluator
class ToxicityEvaluator : public Evaluator {
public:
    struct Config {
        std::string model_path;  // Local classifier model
        std::vector<std::string> categories = {
            "hate", "harassment", "violence", "sexual", "self-harm"
        };
        std::unordered_map<std::string, double> thresholds;
    };

    explicit ToxicityEvaluator(Config config);
    // ... implementation
};

}  // namespace pyflare::eval
```

#### 3.2.3 Cost Tracker Service

```cpp
namespace pyflare::cost {

// Cost calculation result
struct CostResult {
    std::string trace_id;
    std::string model_id;

    // Token counts
    int64_t input_tokens;
    int64_t output_tokens;
    int64_t total_tokens;

    // Costs in USD (micro-dollars for precision)
    int64_t input_cost_micros;
    int64_t output_cost_micros;
    int64_t total_cost_micros;

    // Attribution dimensions
    std::string user_id;
    std::string feature_id;
    std::string environment;

    std::chrono::system_clock::time_point timestamp;
};

// Model pricing configuration
struct ModelPricing {
    std::string model_id;
    std::string provider;
    int64_t input_cost_per_million_tokens;   // micro-dollars
    int64_t output_cost_per_million_tokens;  // micro-dollars
    std::chrono::system_clock::time_point effective_from;
};

class CostTracker {
public:
    struct Config {
        std::vector<std::string> kafka_brokers;
        std::string input_topic = "pyflare.traces";
        std::string output_topic = "pyflare.costs";
        std::string pricing_config_path;
    };

    explicit CostTracker(Config config);

    // Calculate cost for a single inference
    absl::StatusOr<CostResult> Calculate(const TraceRecord& record);

    // Update pricing configuration
    absl::Status UpdatePricing(const ModelPricing& pricing);

    // Budget alerting
    void SetBudgetAlert(
        const std::string& dimension,
        const std::string& value,
        int64_t threshold_micros,
        std::function<void(const BudgetAlert&)> callback);

private:
    std::unordered_map<std::string, ModelPricing> pricing_map_;
    // ... implementation
};

}  // namespace pyflare::cost
```

#### 3.2.4 Root Cause Analysis Engine

```cpp
namespace pyflare::rca {

// Failure record for analysis
struct FailureRecord {
    std::string trace_id;
    std::string model_id;
    std::string failure_type;
    std::string error_message;
    InferenceRecord inference;
    std::chrono::system_clock::time_point timestamp;
};

// Data slice with performance metrics
struct Slice {
    std::string name;
    std::unordered_map<std::string, std::string> filters;
    size_t sample_count;
    double metric_value;
    double baseline_value;
    double deviation;  // How far from baseline
    double confidence;
};

// Counterfactual explanation
struct Counterfactual {
    std::string original_input;
    std::string modified_input;
    std::string original_output;
    std::string target_output;
    std::vector<std::string> changes_made;
    double confidence;
};

// RCA report
struct RCAReport {
    std::vector<std::string> trace_ids_analyzed;
    std::chrono::system_clock::time_point analysis_time;

    // Identified patterns
    struct Pattern {
        std::string description;
        std::vector<std::string> affected_trace_ids;
        double frequency;
        std::string suggested_action;
    };
    std::vector<Pattern> patterns;

    // Problematic slices
    std::vector<Slice> problematic_slices;

    // Temporal correlations
    struct Correlation {
        std::string event_type;
        std::chrono::system_clock::time_point event_time;
        double correlation_score;
    };
    std::vector<Correlation> temporal_correlations;
};

class RootCauseAnalyzer {
public:
    struct Config {
        size_t min_failures_for_analysis = 10;
        size_t max_slices_to_report = 20;
        double slice_deviation_threshold = 0.2;  // 20% worse than baseline
    };

    explicit RootCauseAnalyzer(Config config);

    // Analyze a set of failures
    absl::StatusOr<RCAReport> Analyze(
        const std::vector<FailureRecord>& failures);

    // Find underperforming slices
    absl::StatusOr<std::vector<Slice>> FindProblematicSlices(
        const std::string& model_id,
        const std::string& metric,
        const TimeRange& range);

    // Generate counterfactual explanation
    absl::StatusOr<Counterfactual> GenerateCounterfactual(
        const InferenceRecord& record,
        const std::string& target_outcome);
};

}  // namespace pyflare::rca
```

---

### 3.3 Storage Layer

#### 3.3.1 ClickHouse Client

```cpp
namespace pyflare::storage {

class ClickHouseClient {
public:
    struct Config {
        std::string host = "localhost";
        uint16_t port = 9000;
        std::string database = "pyflare";
        std::string user = "default";
        std::string password;
        size_t max_connections = 10;
        std::chrono::seconds connection_timeout{30};
    };

    explicit ClickHouseClient(Config config);

    // Connection management
    absl::Status Connect();
    absl::Status Disconnect();
    bool IsConnected() const;

    // Write operations
    absl::Status InsertTraces(const std::vector<TraceRecord>& traces);
    absl::Status InsertMetrics(const std::vector<MetricRecord>& metrics);
    absl::Status InsertLogs(const std::vector<LogRecord>& logs);
    absl::Status InsertCosts(const std::vector<CostResult>& costs);

    // Query operations
    absl::StatusOr<QueryResult> Execute(const std::string& sql);
    absl::StatusOr<QueryResult> ExecuteWithParams(
        const std::string& sql,
        const std::vector<QueryParam>& params);

    // Batch operations
    class BatchInserter {
    public:
        void Add(const TraceRecord& record);
        absl::Status Flush();
    };
    std::unique_ptr<BatchInserter> CreateBatchInserter(
        const std::string& table);
};

}  // namespace pyflare::storage
```

#### 3.3.2 Qdrant Client

```cpp
namespace pyflare::storage {

class QdrantClient {
public:
    struct Config {
        std::string host = "localhost";
        uint16_t port = 6334;
        std::string api_key;
        bool use_tls = false;
    };

    explicit QdrantClient(Config config);

    // Collection management
    absl::Status CreateCollection(
        const std::string& name,
        size_t vector_size,
        DistanceMetric metric = DistanceMetric::kCosine);

    absl::Status DeleteCollection(const std::string& name);

    // Vector operations
    absl::Status Upsert(
        const std::string& collection,
        const std::vector<VectorPoint>& points);

    absl::StatusOr<std::vector<SearchResult>> Search(
        const std::string& collection,
        const std::vector<float>& query_vector,
        size_t limit,
        const std::optional<Filter>& filter = std::nullopt);

    // Batch search for drift detection
    absl::StatusOr<std::vector<std::vector<SearchResult>>> BatchSearch(
        const std::string& collection,
        const std::vector<std::vector<float>>& query_vectors,
        size_t limit);
};

struct VectorPoint {
    std::string id;
    std::vector<float> vector;
    std::unordered_map<std::string, std::string> payload;
};

struct SearchResult {
    std::string id;
    float score;
    std::unordered_map<std::string, std::string> payload;
};

}  // namespace pyflare::storage
```

---

### 3.4 Query API

```cpp
namespace pyflare::query {

// Query request/response types
struct QueryRequest {
    std::string sql;
    std::vector<QueryParam> params;
    std::optional<size_t> limit;
    std::optional<size_t> offset;
    std::optional<std::string> format;  // "json", "csv", "arrow"
};

struct QueryResponse {
    std::vector<std::string> columns;
    std::vector<std::vector<Value>> rows;
    size_t total_rows;
    std::chrono::milliseconds execution_time;
};

// REST API handlers
class QueryAPI {
public:
    struct Config {
        std::string listen_address = "0.0.0.0:8080";
        size_t max_query_size = 1024 * 1024;  // 1MB
        std::chrono::seconds query_timeout{30};
        size_t max_result_rows = 10000;
    };

    explicit QueryAPI(
        Config config,
        std::shared_ptr<ClickHouseClient> clickhouse,
        std::shared_ptr<QdrantClient> qdrant,
        std::shared_ptr<RedisClient> redis);

    absl::Status Start();
    absl::Status Stop();

    // API Endpoints (implemented as handlers)

    // Traces
    // GET /api/v1/traces
    // GET /api/v1/traces/{trace_id}
    // GET /api/v1/traces/{trace_id}/spans

    // Metrics
    // GET /api/v1/metrics
    // GET /api/v1/metrics/timeseries

    // Drift
    // GET /api/v1/drift/{model_id}
    // GET /api/v1/drift/{model_id}/history

    // Costs
    // GET /api/v1/costs
    // GET /api/v1/costs/breakdown
    // GET /api/v1/costs/forecast

    // Alerts
    // GET /api/v1/alerts
    // POST /api/v1/alerts/rules

    // SQL Query
    // POST /api/v1/query

private:
    std::unique_ptr<SqlParser> sql_parser_;
    std::unique_ptr<QueryOptimizer> optimizer_;
    // ... handlers
};

// SQL parser for PyFlare query language
class SqlParser {
public:
    // Parse and validate SQL
    absl::StatusOr<ParsedQuery> Parse(const std::string& sql);

    // Supported query types
    enum class QueryType {
        kSelect,
        kAggregate,
        kTimeSeries,
        kTopK,
        kDistinct
    };
};

}  // namespace pyflare::query
```

---

## 4. Data Models & Storage Schemas

### 4.1 ClickHouse Schemas

#### 4.1.1 Traces Table

```sql
-- Main traces table with ReplacingMergeTree for deduplication
CREATE TABLE pyflare.traces
(
    -- Identity
    trace_id          String,
    span_id           String,
    parent_span_id    Nullable(String),

    -- Timing
    start_time        DateTime64(9),  -- Nanosecond precision
    end_time          DateTime64(9),
    duration_ns       UInt64,

    -- Classification
    service_name      LowCardinality(String),
    operation_name    String,
    span_kind         LowCardinality(String),  -- 'client', 'server', 'producer', 'consumer', 'internal'
    status_code       LowCardinality(String),  -- 'ok', 'error', 'unset'
    status_message    Nullable(String),

    -- ML-specific fields
    model_id          LowCardinality(String),
    model_version     LowCardinality(String),
    inference_type    LowCardinality(String),  -- 'llm', 'embedding', 'classification', 'regression'

    -- Input/Output (stored compressed)
    input_preview     String,              -- First 1000 chars
    output_preview    String,              -- First 1000 chars
    input_hash        String,              -- For deduplication analysis

    -- Token metrics (for LLMs)
    input_tokens      Nullable(UInt32),
    output_tokens     Nullable(UInt32),
    total_tokens      Nullable(UInt32),

    -- Cost (micro-dollars)
    cost_micros       Nullable(UInt64),

    -- Attributes (flexible key-value)
    attributes        Map(String, String),

    -- Resource attributes
    resource          Map(String, String),

    -- Events within span
    events            Array(Tuple(
        time DateTime64(9),
        name String,
        attributes Map(String, String)
    )),

    -- Partition and sort keys
    _partition_date   Date DEFAULT toDate(start_time),

    -- Ingestion metadata
    _ingested_at      DateTime64(3) DEFAULT now64(3)
)
ENGINE = ReplacingMergeTree(_ingested_at)
PARTITION BY toYYYYMM(_partition_date)
ORDER BY (service_name, model_id, start_time, trace_id, span_id)
TTL _partition_date + INTERVAL 30 DAY
SETTINGS index_granularity = 8192;

-- Indexes for common queries
ALTER TABLE pyflare.traces ADD INDEX idx_trace_id trace_id TYPE bloom_filter(0.01) GRANULARITY 1;
ALTER TABLE pyflare.traces ADD INDEX idx_model_id model_id TYPE set(100) GRANULARITY 1;
ALTER TABLE pyflare.traces ADD INDEX idx_status status_code TYPE set(10) GRANULARITY 1;
```

#### 4.1.2 Metrics Table

```sql
-- Time-series metrics table
CREATE TABLE pyflare.metrics
(
    -- Identity
    metric_name       LowCardinality(String),

    -- Dimensions
    service_name      LowCardinality(String),
    model_id          LowCardinality(String),
    model_version     LowCardinality(String),
    environment       LowCardinality(String),

    -- Timing
    timestamp         DateTime64(3),

    -- Values
    value_type        LowCardinality(String),  -- 'gauge', 'counter', 'histogram'
    value             Float64,

    -- Histogram-specific
    histogram_count   Nullable(UInt64),
    histogram_sum     Nullable(Float64),
    histogram_buckets Array(Tuple(Float64, UInt64)),  -- (upper_bound, count)

    -- Attributes
    attributes        Map(String, String),

    -- Partition
    _partition_date   Date DEFAULT toDate(timestamp)
)
ENGINE = MergeTree()
PARTITION BY toYYYYMM(_partition_date)
ORDER BY (metric_name, service_name, model_id, timestamp)
TTL _partition_date + INTERVAL 90 DAY;
```

#### 4.1.3 Costs Table

```sql
-- Cost tracking table with rollup support
CREATE TABLE pyflare.costs
(
    -- Identity
    trace_id          String,

    -- Timing
    timestamp         DateTime64(3),

    -- Model info
    model_id          LowCardinality(String),
    model_version     LowCardinality(String),
    provider          LowCardinality(String),

    -- Token counts
    input_tokens      UInt32,
    output_tokens     UInt32,
    total_tokens      UInt32,

    -- Costs (micro-dollars for precision)
    input_cost_micros   UInt64,
    output_cost_micros  UInt64,
    total_cost_micros   UInt64,

    -- Attribution dimensions
    user_id           String,
    feature_id        LowCardinality(String),
    environment       LowCardinality(String),
    team              LowCardinality(String),

    -- Custom dimensions
    dimensions        Map(String, String),

    -- Partition
    _partition_date   Date DEFAULT toDate(timestamp)
)
ENGINE = SummingMergeTree()
PARTITION BY toYYYYMM(_partition_date)
ORDER BY (model_id, user_id, feature_id, toStartOfHour(timestamp))
TTL _partition_date + INTERVAL 365 DAY;

-- Materialized view for hourly rollup
CREATE MATERIALIZED VIEW pyflare.costs_hourly_mv
ENGINE = SummingMergeTree()
PARTITION BY toYYYYMM(hour)
ORDER BY (model_id, user_id, feature_id, hour)
AS SELECT
    model_id,
    user_id,
    feature_id,
    environment,
    toStartOfHour(timestamp) AS hour,
    sum(input_tokens) AS input_tokens,
    sum(output_tokens) AS output_tokens,
    sum(total_tokens) AS total_tokens,
    sum(input_cost_micros) AS input_cost_micros,
    sum(output_cost_micros) AS output_cost_micros,
    sum(total_cost_micros) AS total_cost_micros,
    count() AS request_count
FROM pyflare.costs
GROUP BY model_id, user_id, feature_id, environment, hour;
```

#### 4.1.4 Alerts Table

```sql
CREATE TABLE pyflare.alerts
(
    -- Identity
    alert_id          UUID DEFAULT generateUUIDv4(),

    -- Timing
    triggered_at      DateTime64(3),
    resolved_at       Nullable(DateTime64(3)),

    -- Alert info
    alert_type        LowCardinality(String),  -- 'drift', 'cost', 'error_rate', 'latency', 'custom'
    severity          LowCardinality(String),  -- 'critical', 'warning', 'info'
    status            LowCardinality(String),  -- 'firing', 'resolved', 'acknowledged'

    -- Context
    model_id          LowCardinality(String),
    service_name      LowCardinality(String),

    -- Details
    title             String,
    description       String,
    metric_value      Float64,
    threshold_value   Float64,

    -- Metadata
    labels            Map(String, String),
    annotations       Map(String, String),

    -- Related traces
    sample_trace_ids  Array(String),

    -- Partition
    _partition_date   Date DEFAULT toDate(triggered_at)
)
ENGINE = ReplacingMergeTree(triggered_at)
PARTITION BY toYYYYMM(_partition_date)
ORDER BY (alert_type, model_id, triggered_at, alert_id);
```

### 4.2 Qdrant Collections

#### 4.2.1 Inference Embeddings

```json
{
  "collection_name": "inference_embeddings",
  "vectors": {
    "size": 1536,
    "distance": "Cosine"
  },
  "payload_schema": {
    "trace_id": "keyword",
    "model_id": "keyword",
    "timestamp": "datetime",
    "embedding_type": "keyword",
    "input_hash": "keyword"
  },
  "optimizers_config": {
    "indexing_threshold": 20000
  },
  "hnsw_config": {
    "m": 16,
    "ef_construct": 100
  }
}
```

#### 4.2.2 Reference Embeddings (for drift detection)

```json
{
  "collection_name": "reference_embeddings",
  "vectors": {
    "size": 1536,
    "distance": "Cosine"
  },
  "payload_schema": {
    "model_id": "keyword",
    "dataset_version": "keyword",
    "created_at": "datetime",
    "sample_index": "integer"
  }
}
```

### 4.3 Kafka Topics Schema

#### 4.3.1 Topic Configuration

```yaml
topics:
  pyflare.traces:
    partitions: 32
    replication_factor: 3
    retention_ms: 604800000  # 7 days
    cleanup_policy: delete
    compression_type: lz4

  pyflare.metrics:
    partitions: 16
    replication_factor: 3
    retention_ms: 259200000  # 3 days
    cleanup_policy: delete
    compression_type: lz4

  pyflare.logs:
    partitions: 16
    replication_factor: 3
    retention_ms: 259200000  # 3 days
    cleanup_policy: delete
    compression_type: lz4

  pyflare.alerts:
    partitions: 8
    replication_factor: 3
    retention_ms: 2592000000  # 30 days
    cleanup_policy: compact

  pyflare.embeddings:
    partitions: 16
    replication_factor: 3
    retention_ms: 86400000  # 1 day
    cleanup_policy: delete
    compression_type: zstd  # Better for embeddings
```

#### 4.3.2 Message Schemas (Protobuf)

```protobuf
syntax = "proto3";

package pyflare.v1;

import "google/protobuf/timestamp.proto";

// Trace record for Kafka
message TraceRecord {
  string trace_id = 1;
  string span_id = 2;
  optional string parent_span_id = 3;

  google.protobuf.Timestamp start_time = 4;
  google.protobuf.Timestamp end_time = 5;

  string service_name = 6;
  string operation_name = 7;
  SpanKind span_kind = 8;
  StatusCode status_code = 9;
  optional string status_message = 10;

  // ML-specific
  string model_id = 11;
  string model_version = 12;
  InferenceType inference_type = 13;

  // Content
  string input_preview = 14;
  string output_preview = 15;

  // Tokens
  optional uint32 input_tokens = 16;
  optional uint32 output_tokens = 17;

  // Attributes
  map<string, string> attributes = 18;
  map<string, string> resource = 19;

  repeated Event events = 20;
}

enum SpanKind {
  SPAN_KIND_UNSPECIFIED = 0;
  SPAN_KIND_INTERNAL = 1;
  SPAN_KIND_SERVER = 2;
  SPAN_KIND_CLIENT = 3;
  SPAN_KIND_PRODUCER = 4;
  SPAN_KIND_CONSUMER = 5;
}

enum StatusCode {
  STATUS_CODE_UNSET = 0;
  STATUS_CODE_OK = 1;
  STATUS_CODE_ERROR = 2;
}

enum InferenceType {
  INFERENCE_TYPE_UNSPECIFIED = 0;
  INFERENCE_TYPE_LLM = 1;
  INFERENCE_TYPE_EMBEDDING = 2;
  INFERENCE_TYPE_CLASSIFICATION = 3;
  INFERENCE_TYPE_REGRESSION = 4;
  INFERENCE_TYPE_OBJECT_DETECTION = 5;
  INFERENCE_TYPE_CUSTOM = 6;
}

message Event {
  google.protobuf.Timestamp time = 1;
  string name = 2;
  map<string, string> attributes = 3;
}

// Alert message
message AlertMessage {
  string alert_id = 1;
  google.protobuf.Timestamp triggered_at = 2;

  AlertType alert_type = 3;
  Severity severity = 4;

  string model_id = 5;
  string service_name = 6;

  string title = 7;
  string description = 8;
  double metric_value = 9;
  double threshold_value = 10;

  map<string, string> labels = 11;
  repeated string sample_trace_ids = 12;
}

enum AlertType {
  ALERT_TYPE_UNSPECIFIED = 0;
  ALERT_TYPE_DRIFT = 1;
  ALERT_TYPE_COST = 2;
  ALERT_TYPE_ERROR_RATE = 3;
  ALERT_TYPE_LATENCY = 4;
  ALERT_TYPE_CUSTOM = 5;
}

enum Severity {
  SEVERITY_UNSPECIFIED = 0;
  SEVERITY_INFO = 1;
  SEVERITY_WARNING = 2;
  SEVERITY_CRITICAL = 3;
}
```

---

## 5. API Contracts

### 5.1 REST API Specification

#### 5.1.1 Base Configuration

```yaml
openapi: "3.1.0"
info:
  title: PyFlare API
  version: "1.0.0"
  description: PyFlare Observability Platform API

servers:
  - url: http://localhost:8080/api/v1
    description: Local development
  - url: https://pyflare.example.com/api/v1
    description: Production

security:
  - bearerAuth: []
  - apiKeyAuth: []
```

#### 5.1.2 Traces API

```yaml
paths:
  /traces:
    get:
      summary: List traces
      parameters:
        - name: service
          in: query
          schema:
            type: string
        - name: model_id
          in: query
          schema:
            type: string
        - name: start_time
          in: query
          schema:
            type: string
            format: date-time
        - name: end_time
          in: query
          schema:
            type: string
            format: date-time
        - name: status
          in: query
          schema:
            type: string
            enum: [ok, error]
        - name: limit
          in: query
          schema:
            type: integer
            default: 100
            maximum: 1000
        - name: offset
          in: query
          schema:
            type: integer
            default: 0
      responses:
        "200":
          description: List of traces
          content:
            application/json:
              schema:
                $ref: "#/components/schemas/TraceListResponse"

  /traces/{trace_id}:
    get:
      summary: Get trace by ID
      parameters:
        - name: trace_id
          in: path
          required: true
          schema:
            type: string
      responses:
        "200":
          description: Trace details
          content:
            application/json:
              schema:
                $ref: "#/components/schemas/Trace"
        "404":
          description: Trace not found

  /traces/{trace_id}/spans:
    get:
      summary: Get all spans for a trace
      parameters:
        - name: trace_id
          in: path
          required: true
          schema:
            type: string
      responses:
        "200":
          description: List of spans
          content:
            application/json:
              schema:
                type: array
                items:
                  $ref: "#/components/schemas/Span"
```

#### 5.1.3 Drift API

```yaml
paths:
  /drift/{model_id}:
    get:
      summary: Get current drift status for a model
      parameters:
        - name: model_id
          in: path
          required: true
          schema:
            type: string
      responses:
        "200":
          description: Current drift status
          content:
            application/json:
              schema:
                $ref: "#/components/schemas/DriftStatus"

  /drift/{model_id}/history:
    get:
      summary: Get drift history for a model
      parameters:
        - name: model_id
          in: path
          required: true
          schema:
            type: string
        - name: drift_type
          in: query
          schema:
            type: string
            enum: [feature, embedding, concept, prediction]
        - name: start_time
          in: query
          schema:
            type: string
            format: date-time
        - name: end_time
          in: query
          schema:
            type: string
            format: date-time
      responses:
        "200":
          description: Drift history
          content:
            application/json:
              schema:
                $ref: "#/components/schemas/DriftHistory"

  /drift/{model_id}/reference:
    post:
      summary: Upload reference distribution for drift detection
      parameters:
        - name: model_id
          in: path
          required: true
          schema:
            type: string
      requestBody:
        required: true
        content:
          application/json:
            schema:
              $ref: "#/components/schemas/ReferenceDistribution"
      responses:
        "201":
          description: Reference distribution uploaded
```

#### 5.1.4 Costs API

```yaml
paths:
  /costs:
    get:
      summary: Get cost summary
      parameters:
        - name: start_time
          in: query
          required: true
          schema:
            type: string
            format: date-time
        - name: end_time
          in: query
          required: true
          schema:
            type: string
            format: date-time
        - name: group_by
          in: query
          schema:
            type: array
            items:
              type: string
              enum: [model_id, user_id, feature_id, environment, hour, day]
      responses:
        "200":
          description: Cost summary
          content:
            application/json:
              schema:
                $ref: "#/components/schemas/CostSummary"

  /costs/breakdown:
    get:
      summary: Get detailed cost breakdown
      parameters:
        - name: dimension
          in: query
          required: true
          schema:
            type: string
            enum: [model, user, feature, team]
        - name: start_time
          in: query
          required: true
          schema:
            type: string
            format: date-time
        - name: end_time
          in: query
          required: true
          schema:
            type: string
            format: date-time
        - name: limit
          in: query
          schema:
            type: integer
            default: 10
      responses:
        "200":
          description: Cost breakdown
          content:
            application/json:
              schema:
                $ref: "#/components/schemas/CostBreakdown"

  /costs/forecast:
    get:
      summary: Get cost forecast
      parameters:
        - name: model_id
          in: query
          schema:
            type: string
        - name: horizon_days
          in: query
          schema:
            type: integer
            default: 30
      responses:
        "200":
          description: Cost forecast
          content:
            application/json:
              schema:
                $ref: "#/components/schemas/CostForecast"
```

#### 5.1.5 Query API

```yaml
paths:
  /query:
    post:
      summary: Execute SQL query
      requestBody:
        required: true
        content:
          application/json:
            schema:
              type: object
              required:
                - sql
              properties:
                sql:
                  type: string
                  description: SQL query to execute
                  example: "SELECT model_id, count() FROM traces WHERE start_time > now() - INTERVAL 1 HOUR GROUP BY model_id"
                params:
                  type: array
                  items:
                    type: object
                limit:
                  type: integer
                  maximum: 10000
                format:
                  type: string
                  enum: [json, csv, arrow]
                  default: json
      responses:
        "200":
          description: Query results
          content:
            application/json:
              schema:
                $ref: "#/components/schemas/QueryResult"
        "400":
          description: Invalid query
```

#### 5.1.6 Intelligence API (Phase 3)

```yaml
paths:
  /intelligence/health:
    get:
      summary: Get system-wide intelligence health
      responses:
        "200":
          description: System health metrics
          content:
            application/json:
              schema:
                $ref: "#/components/schemas/SystemHealth"

  /intelligence/health/{model_id}:
    get:
      summary: Get health metrics for a specific model
      parameters:
        - name: model_id
          in: path
          required: true
          schema:
            type: string
      responses:
        "200":
          description: Model health metrics
          content:
            application/json:
              schema:
                $ref: "#/components/schemas/ModelHealth"

  /intelligence/analyze:
    post:
      summary: Analyze a trace through the intelligence pipeline
      requestBody:
        required: true
        content:
          application/json:
            schema:
              $ref: "#/components/schemas/AnalyzeRequest"
      responses:
        "200":
          description: Intelligence analysis result
          content:
            application/json:
              schema:
                $ref: "#/components/schemas/IntelligenceResult"

  /intelligence/models:
    get:
      summary: List all registered models with health status
      responses:
        "200":
          description: List of models
          content:
            application/json:
              schema:
                type: array
                items:
                  $ref: "#/components/schemas/ModelHealth"

  /intelligence/stats:
    get:
      summary: Get pipeline processing statistics
      responses:
        "200":
          description: Pipeline statistics
          content:
            application/json:
              schema:
                $ref: "#/components/schemas/PipelineStats"
```

#### 5.1.7 Alerts API (Phase 3)

```yaml
paths:
  /alerts:
    get:
      summary: List active alerts
      parameters:
        - name: status
          in: query
          schema:
            type: string
            enum: [firing, resolved, acknowledged]
        - name: severity
          in: query
          schema:
            type: string
            enum: [info, warning, critical]
        - name: model_id
          in: query
          schema:
            type: string
        - name: limit
          in: query
          schema:
            type: integer
            default: 100
      responses:
        "200":
          description: List of alerts
          content:
            application/json:
              schema:
                $ref: "#/components/schemas/AlertListResponse"

  /alerts/{alert_id}:
    get:
      summary: Get alert details
      parameters:
        - name: alert_id
          in: path
          required: true
          schema:
            type: string
      responses:
        "200":
          description: Alert details
          content:
            application/json:
              schema:
                $ref: "#/components/schemas/Alert"

  /alerts/{alert_id}/acknowledge:
    post:
      summary: Acknowledge an alert
      parameters:
        - name: alert_id
          in: path
          required: true
          schema:
            type: string
      responses:
        "200":
          description: Alert acknowledged

  /alerts/{alert_id}/resolve:
    post:
      summary: Resolve an alert
      parameters:
        - name: alert_id
          in: path
          required: true
          schema:
            type: string
      responses:
        "200":
          description: Alert resolved

  /alerts/rules:
    get:
      summary: List alert rules
      responses:
        "200":
          description: List of alert rules
          content:
            application/json:
              schema:
                type: array
                items:
                  $ref: "#/components/schemas/AlertRule"
    post:
      summary: Create alert rule
      requestBody:
        required: true
        content:
          application/json:
            schema:
              $ref: "#/components/schemas/AlertRuleCreate"
      responses:
        "201":
          description: Rule created
          content:
            application/json:
              schema:
                $ref: "#/components/schemas/AlertRule"

  /alerts/rules/{rule_id}:
    get:
      summary: Get rule details
      parameters:
        - name: rule_id
          in: path
          required: true
          schema:
            type: string
      responses:
        "200":
          description: Rule details
          content:
            application/json:
              schema:
                $ref: "#/components/schemas/AlertRule"
    put:
      summary: Update rule
      parameters:
        - name: rule_id
          in: path
          required: true
          schema:
            type: string
      requestBody:
        required: true
        content:
          application/json:
            schema:
              $ref: "#/components/schemas/AlertRuleUpdate"
      responses:
        "200":
          description: Rule updated
    delete:
      summary: Delete rule
      parameters:
        - name: rule_id
          in: path
          required: true
          schema:
            type: string
      responses:
        "204":
          description: Rule deleted

  /alerts/silences:
    get:
      summary: List silences
      responses:
        "200":
          description: List of silences
          content:
            application/json:
              schema:
                type: array
                items:
                  $ref: "#/components/schemas/Silence"
    post:
      summary: Create silence
      requestBody:
        required: true
        content:
          application/json:
            schema:
              $ref: "#/components/schemas/SilenceCreate"
      responses:
        "201":
          description: Silence created

  /alerts/silences/{silence_id}:
    delete:
      summary: Delete silence
      parameters:
        - name: silence_id
          in: path
          required: true
          schema:
            type: string
      responses:
        "204":
          description: Silence deleted

  /alerts/maintenance:
    get:
      summary: List maintenance windows
      responses:
        "200":
          description: List of maintenance windows
          content:
            application/json:
              schema:
                type: array
                items:
                  $ref: "#/components/schemas/MaintenanceWindow"
    post:
      summary: Create maintenance window
      requestBody:
        required: true
        content:
          application/json:
            schema:
              $ref: "#/components/schemas/MaintenanceWindowCreate"
      responses:
        "201":
          description: Maintenance window created

  /alerts/maintenance/{window_id}:
    delete:
      summary: Delete maintenance window
      parameters:
        - name: window_id
          in: path
          required: true
          schema:
            type: string
      responses:
        "204":
          description: Maintenance window deleted

  /alerts/stats:
    get:
      summary: Get alert statistics
      responses:
        "200":
          description: Alert statistics
          content:
            application/json:
              schema:
                $ref: "#/components/schemas/AlertStats"
```

#### 5.1.8 RCA API (Phase 3)

```yaml
paths:
  /rca/analyze:
    post:
      summary: Run root cause analysis
      requestBody:
        required: true
        content:
          application/json:
            schema:
              type: object
              properties:
                model_id:
                  type: string
                time_range:
                  $ref: "#/components/schemas/TimeRange"
                failure_type:
                  type: string
      responses:
        "200":
          description: RCA report
          content:
            application/json:
              schema:
                $ref: "#/components/schemas/RCAReport"

  /rca/patterns:
    get:
      summary: Get detected failure patterns
      parameters:
        - name: model_id
          in: query
          schema:
            type: string
      responses:
        "200":
          description: List of patterns
          content:
            application/json:
              schema:
                type: array
                items:
                  $ref: "#/components/schemas/Pattern"

  /rca/clusters:
    get:
      summary: Get failure clusters
      parameters:
        - name: model_id
          in: query
          schema:
            type: string
      responses:
        "200":
          description: List of failure clusters
          content:
            application/json:
              schema:
                type: array
                items:
                  $ref: "#/components/schemas/FailureCluster"

  /rca/slices:
    get:
      summary: Get problematic data slices
      parameters:
        - name: model_id
          in: query
          schema:
            type: string
        - name: metric
          in: query
          schema:
            type: string
      responses:
        "200":
          description: List of problematic slices
          content:
            application/json:
              schema:
                type: array
                items:
                  $ref: "#/components/schemas/ProblematicSlice"
```

### 5.2 Response Schemas

```yaml
components:
  schemas:
    Trace:
      type: object
      properties:
        trace_id:
          type: string
        service_name:
          type: string
        model_id:
          type: string
        start_time:
          type: string
          format: date-time
        end_time:
          type: string
          format: date-time
        duration_ms:
          type: number
        status:
          type: string
          enum: [ok, error]
        span_count:
          type: integer
        input_preview:
          type: string
        output_preview:
          type: string
        input_tokens:
          type: integer
        output_tokens:
          type: integer
        cost_usd:
          type: number
        attributes:
          type: object
          additionalProperties:
            type: string

    DriftStatus:
      type: object
      properties:
        model_id:
          type: string
        overall_status:
          type: string
          enum: [healthy, warning, drifted]
        drift_scores:
          type: object
          properties:
            feature:
              $ref: "#/components/schemas/DriftScore"
            embedding:
              $ref: "#/components/schemas/DriftScore"
            concept:
              $ref: "#/components/schemas/DriftScore"
            prediction:
              $ref: "#/components/schemas/DriftScore"
        last_updated:
          type: string
          format: date-time

    DriftScore:
      type: object
      properties:
        score:
          type: number
          minimum: 0
          maximum: 1
        threshold:
          type: number
        is_drifted:
          type: boolean
        trend:
          type: string
          enum: [stable, increasing, decreasing]
        feature_breakdown:
          type: object
          additionalProperties:
            type: number

    CostSummary:
      type: object
      properties:
        total_cost_usd:
          type: number
        total_requests:
          type: integer
        total_input_tokens:
          type: integer
        total_output_tokens:
          type: integer
        average_cost_per_request_usd:
          type: number
        by_period:
          type: array
          items:
            type: object
            properties:
              period:
                type: string
              cost_usd:
                type: number
              requests:
                type: integer

    QueryResult:
      type: object
      properties:
        columns:
          type: array
          items:
            type: object
            properties:
              name:
                type: string
              type:
                type: string
        rows:
          type: array
          items:
            type: array
        total_rows:
          type: integer
        execution_time_ms:
          type: number

    # Phase 3 Schemas

    SystemHealth:
      type: object
      properties:
        overall_health:
          type: number
        models_with_drift:
          type: integer
        total_active_alerts:
          type: integer
        models_analyzed:
          type: integer
        avg_health_score:
          type: number
        last_update:
          type: integer

    ModelHealth:
      type: object
      properties:
        model_id:
          type: string
        health_score:
          type: number
        has_active_drift:
          type: boolean
        active_alerts:
          type: integer
        recent_safety_issues:
          type: integer
        avg_evaluation_score:
          type: number
        last_analyzed:
          type: integer

    IntelligenceResult:
      type: object
      properties:
        trace_id:
          type: string
        model_id:
          type: string
        analyzed_at:
          type: integer
        health_score:
          type: number
        drift:
          $ref: "#/components/schemas/DriftAnalysis"
        evaluation:
          $ref: "#/components/schemas/EvaluationResult"
        safety:
          $ref: "#/components/schemas/SafetyResult"

    DriftAnalysis:
      type: object
      properties:
        drift_detected:
          type: boolean
        overall_severity:
          type: number
        drifted_dimensions:
          type: array
          items:
            type: string
        causes:
          type: array
          items:
            type: string

    EvaluationResult:
      type: object
      properties:
        overall_score:
          type: number
        passed:
          type: boolean
        issues:
          type: array
          items:
            type: string

    SafetyResult:
      type: object
      properties:
        is_safe:
          type: boolean
        risk_score:
          type: number
        detected_issues:
          type: array
          items:
            type: string
        risk_level:
          type: string
          enum: [low, medium, high, critical]

    Alert:
      type: object
      properties:
        id:
          type: string
        name:
          type: string
        severity:
          type: string
          enum: [info, warning, critical]
        status:
          type: string
          enum: [firing, resolved, acknowledged]
        model_id:
          type: string
        description:
          type: string
        fired_at:
          type: integer
        resolved_at:
          type: integer
        labels:
          type: object
          additionalProperties:
            type: string

    AlertRule:
      type: object
      properties:
        id:
          type: string
        name:
          type: string
        type:
          type: string
          enum: [threshold, anomaly, rate, pattern, composite]
        severity:
          type: string
          enum: [info, warning, critical]
        enabled:
          type: boolean
        evaluation_interval:
          type: integer
        config:
          type: object

    Silence:
      type: object
      properties:
        id:
          type: string
        matchers:
          type: array
          items:
            type: object
            properties:
              name:
                type: string
              value:
                type: string
              is_regex:
                type: boolean
        starts_at:
          type: integer
        ends_at:
          type: integer
        created_by:
          type: string
        comment:
          type: string

    MaintenanceWindow:
      type: object
      properties:
        id:
          type: string
        name:
          type: string
        starts_at:
          type: integer
        ends_at:
          type: integer
        affected_models:
          type: array
          items:
            type: string
        created_by:
          type: string

    RCAReport:
      type: object
      properties:
        id:
          type: string
        model_id:
          type: string
        analysis_time:
          type: integer
        patterns:
          type: array
          items:
            $ref: "#/components/schemas/Pattern"
        clusters:
          type: array
          items:
            $ref: "#/components/schemas/FailureCluster"
        problematic_slices:
          type: array
          items:
            $ref: "#/components/schemas/ProblematicSlice"
        root_causes:
          type: array
          items:
            $ref: "#/components/schemas/RootCause"
        recommendations:
          type: array
          items:
            $ref: "#/components/schemas/Recommendation"

    Pattern:
      type: object
      properties:
        id:
          type: string
        type:
          type: string
        description:
          type: string
        severity:
          type: number
        affected_traces:
          type: integer
        suggested_actions:
          type: array
          items:
            type: string

    FailureCluster:
      type: object
      properties:
        id:
          type: string
        name:
          type: string
        size:
          type: integer
        representative_error:
          type: string
        common_keywords:
          type: array
          items:
            type: string
        severity:
          type: number

    ProblematicSlice:
      type: object
      properties:
        id:
          type: string
        name:
          type: string
        dimension:
          type: string
        dimension_value:
          type: string
        metric:
          type: string
        metric_value:
          type: number
        baseline:
          type: number
        deviation_percentage:
          type: number
        impact_score:
          type: number
        is_significant:
          type: boolean

    RootCause:
      type: object
      properties:
        id:
          type: string
        category:
          type: string
        description:
          type: string
        confidence:
          type: number
        evidence:
          type: array
          items:
            type: string
        related_patterns:
          type: array
          items:
            type: string

    Recommendation:
      type: object
      properties:
        id:
          type: string
        priority:
          type: integer
        action:
          type: string
        rationale:
          type: string
        expected_impact:
          type: string
        related_causes:
          type: array
          items:
            type: string

    PipelineStats:
      type: object
      properties:
        total_processed:
          type: integer
        drift_detections:
          type: integer
        safety_issues:
          type: integer
        evaluation_failures:
          type: integer
        rca_triggered:
          type: integer
        alerts_generated:
          type: integer
        avg_processing_time_ms:
          type: number
        p99_processing_time_ms:
          type: number
        queue_depth:
          type: integer
        component_health:
          type: object
          properties:
            drift_service:
              type: boolean
            eval_service:
              type: boolean
            rca_service:
              type: boolean
            alert_service:
              type: boolean
```

---

## 6. Communication Protocols

### 6.1 OTLP Protocol

PyFlare collectors accept telemetry via the OpenTelemetry Protocol (OTLP) over both gRPC and HTTP.

#### 6.1.1 gRPC Endpoints

| Endpoint | Service | Method |
|----------|---------|--------|
| `:4317` | `opentelemetry.proto.collector.trace.v1.TraceService` | `Export` |
| `:4317` | `opentelemetry.proto.collector.metrics.v1.MetricsService` | `Export` |
| `:4317` | `opentelemetry.proto.collector.logs.v1.LogsService` | `Export` |

#### 6.1.2 HTTP Endpoints

| Endpoint | Method | Content-Type |
|----------|--------|--------------|
| `:4318/v1/traces` | POST | `application/x-protobuf`, `application/json` |
| `:4318/v1/metrics` | POST | `application/x-protobuf`, `application/json` |
| `:4318/v1/logs` | POST | `application/x-protobuf`, `application/json` |

### 6.2 Internal gRPC Services

```protobuf
syntax = "proto3";

package pyflare.internal.v1;

// Service for inter-component communication
service ProcessorService {
  // Stream processed traces to storage
  rpc StreamTraces(stream TraceRecord) returns (StreamResponse);

  // Get processing status
  rpc GetStatus(StatusRequest) returns (StatusResponse);
}

service QueryService {
  // Execute query across storage backends
  rpc Query(QueryRequest) returns (QueryResponse);

  // Stream query results
  rpc StreamQuery(QueryRequest) returns (stream QueryRow);
}

service AlertService {
  // Subscribe to alerts
  rpc SubscribeAlerts(AlertSubscription) returns (stream AlertMessage);

  // Acknowledge alert
  rpc AcknowledgeAlert(AcknowledgeRequest) returns (AcknowledgeResponse);
}
```

### 6.3 WebSocket Protocol (Real-time Updates)

```typescript
// WebSocket message types for real-time UI updates

interface WebSocketMessage {
  type: 'trace' | 'metric' | 'alert' | 'drift' | 'cost';
  payload: unknown;
  timestamp: string;
}

// Subscribe to real-time updates
interface SubscribeMessage {
  action: 'subscribe';
  channels: string[];  // e.g., ['traces:model-123', 'alerts:*']
}

// Unsubscribe
interface UnsubscribeMessage {
  action: 'unsubscribe';
  channels: string[];
}

// New trace notification
interface TraceNotification {
  type: 'trace';
  payload: {
    trace_id: string;
    model_id: string;
    status: 'ok' | 'error';
    duration_ms: number;
    timestamp: string;
  };
}

// Alert notification
interface AlertNotification {
  type: 'alert';
  payload: {
    alert_id: string;
    severity: 'info' | 'warning' | 'critical';
    title: string;
    model_id: string;
    timestamp: string;
  };
}
```

---

## 7. Security Architecture

### 7.1 Authentication

```yaml
authentication:
  # API Key authentication
  api_key:
    enabled: true
    header_name: "X-API-Key"
    storage: "redis"  # or "database"

  # JWT authentication
  jwt:
    enabled: true
    issuer: "pyflare"
    audience: "pyflare-api"
    algorithms: ["RS256"]
    public_key_path: "/etc/pyflare/keys/public.pem"
    token_expiry: "24h"

  # OAuth2/OIDC (Enterprise)
  oauth2:
    enabled: false
    provider: "okta"  # or "auth0", "azure-ad"
    client_id: ""
    authorization_endpoint: ""
    token_endpoint: ""
    userinfo_endpoint: ""
```

### 7.2 Authorization (RBAC)

```yaml
roles:
  - name: admin
    permissions:
      - "*"

  - name: developer
    permissions:
      - "traces:read"
      - "metrics:read"
      - "costs:read"
      - "drift:read"
      - "alerts:read"
      - "query:execute"

  - name: viewer
    permissions:
      - "traces:read"
      - "metrics:read"
      - "costs:read:own"  # Only own costs
      - "drift:read"

  - name: operator
    permissions:
      - "traces:read"
      - "metrics:read"
      - "alerts:*"
      - "drift:read"
      - "drift:configure"
```

### 7.3 Data Security

```cpp
namespace pyflare::security {

// PII detection and handling
class PIIHandler {
public:
    enum class Action {
        kAllow,     // No action
        kMask,      // Replace with ***
        kHash,      // Replace with hash
        kRedact     // Remove entirely
    };

    struct Config {
        Action default_action = Action::kMask;
        std::vector<std::string> patterns;  // Regex patterns for PII
        std::unordered_map<std::string, Action> field_actions;
    };

    // Process input/output before storage
    std::string Process(const std::string& content);
};

// Encryption at rest
class EncryptionHandler {
public:
    struct Config {
        std::string key_provider;  // "aws-kms", "gcp-kms", "local"
        std::string key_id;
        std::string algorithm = "AES-256-GCM";
    };

    absl::StatusOr<std::string> Encrypt(std::string_view plaintext);
    absl::StatusOr<std::string> Decrypt(std::string_view ciphertext);
};

}  // namespace pyflare::security
```

---

## 8. Development Plan

### 8.1 Phase 1: Foundation

**Objective**: Establish project infrastructure and core data ingestion pipeline.

#### 8.1.1 Project Setup

| Task | Description | Dependencies |
|------|-------------|--------------|
| 1.1.1 | Initialize repository structure | None |
| 1.1.2 | Configure CMake build system (C++20) | 1.1.1 |
| 1.1.3 | Set up vcpkg/Conan for dependency management | 1.1.2 |
| 1.1.4 | Configure clang-format and clang-tidy | 1.1.1 |
| 1.1.5 | Set up Google Test framework | 1.1.2 |
| 1.1.6 | Create GitHub Actions CI pipeline | 1.1.4, 1.1.5 |
| 1.1.7 | Configure Docker build environment | 1.1.2 |

#### 8.1.2 Common Utilities

| Task | Description | Dependencies |
|------|-------------|--------------|
| 1.2.1 | Implement logging wrapper (spdlog) | 1.1.2 |
| 1.2.2 | Implement configuration loader (YAML) | 1.1.2 |
| 1.2.3 | Create error handling utilities (absl::Status) | 1.1.2 |
| 1.2.4 | Implement metrics collection (internal) | 1.2.1 |
| 1.2.5 | Create thread pool implementation | 1.1.2 |
| 1.2.6 | Write unit tests for utilities | 1.2.1-1.2.5 |

#### 8.1.3 OTLP Collector

| Task | Description | Dependencies |
|------|-------------|--------------|
| 1.3.1 | Implement gRPC server skeleton | 1.2.1-1.2.3 |
| 1.3.2 | Implement OTLP trace receiver | 1.3.1 |
| 1.3.3 | Implement OTLP metrics receiver | 1.3.1 |
| 1.3.4 | Implement OTLP logs receiver | 1.3.1 |
| 1.3.5 | Add HTTP/JSON receiver support | 1.3.2-1.3.4 |
| 1.3.6 | Implement batching logic | 1.3.2 |
| 1.3.7 | Implement sampling strategies | 1.3.6 |
| 1.3.8 | Write integration tests | 1.3.1-1.3.7 |

#### 8.1.4 Kafka Integration

| Task | Description | Dependencies |
|------|-------------|--------------|
| 1.4.1 | Implement Kafka producer wrapper | 1.2.1-1.2.3 |
| 1.4.2 | Define Protobuf message schemas | 1.1.2 |
| 1.4.3 | Implement serialization logic | 1.4.1, 1.4.2 |
| 1.4.4 | Add producer batching and compression | 1.4.1 |
| 1.4.5 | Implement error handling and retries | 1.4.1 |
| 1.4.6 | Connect collector to Kafka exporter | 1.3.6, 1.4.3 |
| 1.4.7 | Write integration tests with Kafka | 1.4.6 |

#### 8.1.5 Python SDK (Basic)

| Task | Description | Dependencies |
|------|-------------|--------------|
| 1.5.1 | Initialize Python package structure | None |
| 1.5.2 | Configure pyproject.toml with dependencies | 1.5.1 |
| 1.5.3 | Implement core SDK class | 1.5.2 |
| 1.5.4 | Implement `@pyflare.trace` decorator | 1.5.3 |
| 1.5.5 | Implement OTLP exporter configuration | 1.5.3 |
| 1.5.6 | Add context propagation | 1.5.4 |
| 1.5.7 | Write unit tests | 1.5.3-1.5.6 |
| 1.5.8 | Write integration tests with collector | 1.5.7, 1.3.8 |

#### Phase 1 Deliverables

- Working OTLP collector accepting traces, metrics, logs
- Kafka message pipeline with Protobuf serialization
- Basic Python SDK with trace decorator
- CI/CD pipeline with automated testing
- Docker images for collector

---

### 8.2 Phase 2: Storage & Processing

**Objective**: Implement persistent storage and basic stream processing.

#### 8.2.1 ClickHouse Integration

| Task | Description | Dependencies |
|------|-------------|--------------|
| 2.1.1 | Implement ClickHouse client wrapper | Phase 1 |
| 2.1.2 | Create database schema migrations | 2.1.1 |
| 2.1.3 | Implement traces table and writer | 2.1.2 |
| 2.1.4 | Implement metrics table and writer | 2.1.2 |
| 2.1.5 | Implement logs table and writer | 2.1.2 |
| 2.1.6 | Create materialized views for common queries | 2.1.3-2.1.5 |
| 2.1.7 | Implement batch insert optimization | 2.1.3 |
| 2.1.8 | Add connection pooling | 2.1.1 |
| 2.1.9 | Write integration tests | 2.1.3-2.1.8 |

#### 8.2.2 Qdrant Integration

| Task | Description | Dependencies |
|------|-------------|--------------|
| 2.2.1 | Implement Qdrant client wrapper | Phase 1 |
| 2.2.2 | Create collection schemas | 2.2.1 |
| 2.2.3 | Implement vector upsert operations | 2.2.2 |
| 2.2.4 | Implement vector search operations | 2.2.2 |
| 2.2.5 | Add batch operations support | 2.2.3 |
| 2.2.6 | Write integration tests | 2.2.3-2.2.5 |

#### 8.2.3 Stream Processing Framework

| Task | Description | Dependencies |
|------|-------------|--------------|
| 2.3.1 | Implement Kafka consumer framework | Phase 1 |
| 2.3.2 | Create processor base class | 2.3.1 |
| 2.3.3 | Implement consumer group management | 2.3.1 |
| 2.3.4 | Add offset management and commits | 2.3.1 |
| 2.3.5 | Implement storage writer processor | 2.3.2, 2.1.7 |
| 2.3.6 | Add dead letter queue handling | 2.3.1 |
| 2.3.7 | Write integration tests | 2.3.5-2.3.6 |

#### 8.2.4 Cost Tracking

| Task | Description | Dependencies |
|------|-------------|--------------|
| 2.4.1 | Define cost calculation interfaces | 2.3.2 |
| 2.4.2 | Implement token counting logic | 2.4.1 |
| 2.4.3 | Create pricing configuration loader | 2.4.1 |
| 2.4.4 | Implement cost calculation processor | 2.4.2, 2.4.3 |
| 2.4.5 | Create costs table and writer | 2.1.2, 2.4.4 |
| 2.4.6 | Implement cost aggregation views | 2.4.5 |
| 2.4.7 | Write unit and integration tests | 2.4.4-2.4.6 |

#### 8.2.5 Query API (Basic)

| Task | Description | Dependencies |
|------|-------------|--------------|
| 2.5.1 | Implement HTTP server (REST) | Phase 1 |
| 2.5.2 | Create trace query endpoints | 2.5.1, 2.1.3 |
| 2.5.3 | Create metrics query endpoints | 2.5.1, 2.1.4 |
| 2.5.4 | Create cost query endpoints | 2.5.1, 2.4.5 |
| 2.5.5 | Implement basic SQL query endpoint | 2.5.1, 2.1.1 |
| 2.5.6 | Add pagination support | 2.5.2-2.5.5 |
| 2.5.7 | Implement API authentication | 2.5.1 |
| 2.5.8 | Write API integration tests | 2.5.2-2.5.7 |
| 2.5.9 | Generate OpenAPI documentation | 2.5.8 |

#### Phase 2 Deliverables

- ClickHouse storage with optimized schemas
- Qdrant integration for embeddings
- Stream processing framework consuming from Kafka
- Cost tracking pipeline
- REST API for trace and cost queries
- API documentation

---

### 8.3 Phase 3: Intelligence ✅ COMPLETE

**Objective**: Implement ML-specific analysis capabilities.

**Status**: All tasks completed. Phase 3 implementation includes:
- Advanced drift detection (embedding, concept, prediction, feature with PSI/MMD/KS)
- Enhanced evaluators (hallucination, RAG, toxicity, safety, semantic similarity)
- Intelligent RCA with causal analysis and recommendations
- Full alerting system with rules, deduplication, silences, maintenance windows
- Intelligence pipeline orchestrating all components
- REST API handlers for intelligence, alerts, and RCA
- UI components (Intelligence Dashboard, Alerts Panel, RCA Explorer)
- Unit tests for pipeline and alerting

#### 8.3.1 Feature Drift Detection ✅

| Task | Description | Dependencies |
|------|-------------|--------------|
| 3.1.1 | Implement distribution representation | Phase 2 |
| 3.1.2 | Implement Kolmogorov-Smirnov test | 3.1.1 |
| 3.1.3 | Implement Population Stability Index | 3.1.1 |
| 3.1.4 | Implement Chi-squared test (categorical) | 3.1.1 |
| 3.1.5 | Create drift detector service | 3.1.2-3.1.4 |
| 3.1.6 | Implement reference distribution storage | 3.1.5, 2.2.3 |
| 3.1.7 | Add windowed drift computation | 3.1.5 |
| 3.1.8 | Write unit and integration tests | 3.1.5-3.1.7 |

#### 8.3.2 Embedding Drift Detection ✅

| Task | Description | Dependencies | Status |
|------|-------------|--------------|--------|
| 3.2.1 | Implement cosine similarity drift | 3.1.1 | ✅ |
| 3.2.2 | Implement MMD (Maximum Mean Discrepancy) | 3.1.1 | ✅ |
| 3.2.3 | Create embedding drift detector | 3.2.1, 3.2.2 | ✅ |
| 3.2.4 | Integrate with Qdrant for reference embeddings | 3.2.3, 2.2.4 | ✅ |
| 3.2.5 | Add incremental update support | 3.2.3 | ✅ |
| 3.2.6 | Write tests | 3.2.3-3.2.5 | ✅ |

#### 8.3.3 Alerting System ✅

| Task | Description | Dependencies | Status |
|------|-------------|--------------|--------|
| 3.3.1 | Define alert rule schema | 3.1.5 | ✅ |
| 3.3.2 | Implement alert rule engine | 3.3.1 | ✅ |
| 3.3.3 | Create alerts table and writer | 3.3.2, 2.1.2 | ✅ |
| 3.3.4 | Implement alert deduplication | 3.3.2 | ✅ |
| 3.3.5 | Add webhook notification support | 3.3.2 | ✅ |
| 3.3.6 | Add Slack/PagerDuty integration | 3.3.5 | ✅ |
| 3.3.7 | Create alert API endpoints | 3.3.3, 2.5.1 | ✅ |
| 3.3.8 | Write tests | 3.3.2-3.3.7 | ✅ |

**Additional Phase 3 Alerting Features Implemented:**
- Silence management (create, list, delete)
- Maintenance windows (create, list, delete)
- Multi-channel notifications (Slack, PagerDuty, webhooks, email)
- Rate limiting and escalation
- Alert grouping by labels

#### 8.3.4 Evaluators ✅

| Task | Description | Dependencies | Status |
|------|-------------|--------------|--------|
| 3.4.1 | Define evaluator interface | Phase 2 | ✅ |
| 3.4.2 | Implement hallucination evaluator (Python) | 3.4.1 | ✅ |
| 3.4.3 | Implement RAG quality evaluator (Python) | 3.4.1 | ✅ |
| 3.4.4 | Implement toxicity evaluator | 3.4.1 | ✅ |
| 3.4.5 | Create evaluator service (calls Python) | 3.4.2-3.4.4 | ✅ |
| 3.4.6 | Add async evaluation pipeline | 3.4.5 | ✅ |
| 3.4.7 | Store evaluation results in ClickHouse | 3.4.5, 2.1.3 | ✅ |
| 3.4.8 | Write tests | 3.4.5-3.4.7 | ✅ |

**Additional Phase 3 Evaluator Features Implemented:**
- Safety analyzer (PII detection, prompt injection, content safety)
- Semantic similarity evaluator
- Multi-category toxicity scoring

#### 8.3.5 Root Cause Analysis ✅

| Task | Description | Dependencies | Status |
|------|-------------|--------------|--------|
| 3.5.1 | Implement failure clustering (Python) | Phase 2 | ✅ |
| 3.5.2 | Implement slice finder algorithm | 3.5.1 | ✅ |
| 3.5.3 | Create RCA service | 3.5.1, 3.5.2 | ✅ |
| 3.5.4 | Add temporal correlation analysis | 3.5.3 | ✅ |
| 3.5.5 | Create RCA API endpoints | 3.5.3, 2.5.1 | ✅ |
| 3.5.6 | Write tests | 3.5.3-3.5.5 | ✅ |

**Additional Phase 3 RCA Features Implemented:**
- Multi-phase analysis engine
- Causal factor identification with confidence scoring
- Actionable recommendations generation
- Pattern detection and failure clustering

#### 8.3.6 Intelligence Pipeline ✅ (NEW)

| Task | Description | Dependencies | Status |
|------|-------------|--------------|--------|
| 3.6.1 | Create unified intelligence pipeline | 3.1-3.5 | ✅ |
| 3.6.2 | Implement model health scoring | 3.6.1 | ✅ |
| 3.6.3 | Add system health aggregation | 3.6.2 | ✅ |
| 3.6.4 | Create intelligence API handler | 3.6.1 | ✅ |
| 3.6.5 | Write pipeline tests | 3.6.1-3.6.4 | ✅ |

#### 8.3.7 UI Components ✅ (NEW)

| Task | Description | Dependencies | Status |
|------|-------------|--------------|--------|
| 3.7.1 | Create Intelligence Dashboard | 3.6.4 | ✅ |
| 3.7.2 | Create Alerts Panel | 3.3.7 | ✅ |
| 3.7.3 | Create RCA Explorer | 3.5.5 | ✅ |

#### Phase 3 Deliverables ✅ ALL COMPLETE

- **Drift Detection**: Feature, embedding, concept, and prediction drift with PSI, MMD, KS tests
- **Alerting System**: Rule engine (threshold, anomaly, rate, pattern, composite), deduplication, silences, maintenance windows, multi-channel notifications (Slack, PagerDuty, webhooks, email)
- **Evaluators**: Hallucination detection, RAG quality metrics, toxicity scoring, semantic similarity, safety analysis (PII, prompt injection)
- **Root Cause Analysis**: Multi-phase analysis engine, failure clustering, slice analysis, causal factor identification, recommendations
- **Intelligence Pipeline**: Unified orchestration of all components, model/system health scoring
- **API Endpoints**: `/api/v1/intelligence/*`, `/api/v1/alerts/*`, `/api/v1/rca/*`
- **UI Components**: Intelligence Dashboard, Alerts Panel, RCA Explorer
- **Tests**: Unit tests for intelligence pipeline and alerting system

---

### 8.4 Phase 4: UI & Polish

**Objective**: Build user interface and finalize integrations.

#### 8.4.1 Web UI Foundation

| Task | Description | Dependencies |
|------|-------------|--------------|
| 4.1.1 | Initialize React/TypeScript project | None |
| 4.1.2 | Set up Tailwind CSS | 4.1.1 |
| 4.1.3 | Configure React Query for data fetching | 4.1.1 |
| 4.1.4 | Implement authentication flow | 4.1.1, 2.5.7 |
| 4.1.5 | Create layout and navigation | 4.1.2 |
| 4.1.6 | Implement API client | 4.1.3 |

#### 8.4.2 Trace Explorer

| Task | Description | Dependencies |
|------|-------------|--------------|
| 4.2.1 | Create trace list view | 4.1.5, 4.1.6 |
| 4.2.2 | Implement trace search and filters | 4.2.1 |
| 4.2.3 | Create trace detail view | 4.2.1 |
| 4.2.4 | Implement span waterfall visualization | 4.2.3 |
| 4.2.5 | Add trace comparison view | 4.2.3 |
| 4.2.6 | Implement real-time trace streaming | 4.2.1 |

#### 8.4.3 Drift Dashboard

| Task | Description | Dependencies |
|------|-------------|--------------|
| 4.3.1 | Create drift overview page | 4.1.5, 4.1.6 |
| 4.3.2 | Implement drift score charts | 4.3.1 |
| 4.3.3 | Create drift history timeline | 4.3.1 |
| 4.3.4 | Add feature-level drift breakdown | 4.3.2 |
| 4.3.5 | Implement drift alerts panel | 4.3.1 |

#### 8.4.4 Cost Analytics

| Task | Description | Dependencies |
|------|-------------|--------------|
| 4.4.1 | Create cost overview page | 4.1.5, 4.1.6 |
| 4.4.2 | Implement cost breakdown charts | 4.4.1 |
| 4.4.3 | Create cost trend analysis | 4.4.1 |
| 4.4.4 | Add budget tracking visualization | 4.4.1 |
| 4.4.5 | Implement cost attribution table | 4.4.2 |

#### 8.4.5 SDK Integrations

| Task | Description | Dependencies |
|------|-------------|--------------|
| 4.5.1 | Implement LangChain integration | Phase 1 SDK |
| 4.5.2 | Implement OpenAI integration | Phase 1 SDK |
| 4.5.3 | Implement PyTorch integration | Phase 1 SDK |
| 4.5.4 | Implement PyFlame native integration | Phase 1 SDK |
| 4.5.5 | Write integration documentation | 4.5.1-4.5.4 |
| 4.5.6 | Create integration examples | 4.5.1-4.5.4 |

#### 8.4.6 Grafana Plugin

| Task | Description | Dependencies |
|------|-------------|--------------|
| 4.6.1 | Initialize Grafana plugin project | Phase 2 |
| 4.6.2 | Implement data source plugin | 4.6.1 |
| 4.6.3 | Create trace panel plugin | 4.6.2 |
| 4.6.4 | Create drift panel plugin | 4.6.2 |
| 4.6.5 | Write plugin documentation | 4.6.2-4.6.4 |

#### 8.4.7 Documentation & Testing

| Task | Description | Dependencies |
|------|-------------|--------------|
| 4.7.1 | Write API reference documentation | All APIs |
| 4.7.2 | Create getting started guide | All components |
| 4.7.3 | Write deployment guide | All components |
| 4.7.4 | Create architecture documentation | All components |
| 4.7.5 | Run performance benchmarks | All components |
| 4.7.6 | Run security audit | All components |
| 4.7.7 | Create end-to-end test suite | All components |

#### Phase 4 Deliverables

- Complete web UI with trace explorer, drift dashboards, cost analytics
- Framework integrations (LangChain, OpenAI, PyTorch, PyFlame)
- Grafana plugin for existing dashboards
- Complete documentation
- Performance benchmarks
- Production-ready release

---

### 8.5 Dependency Graph

```
Phase 1: Foundation
├── 1.1 Project Setup ──────────────────┐
├── 1.2 Common Utilities ◄──────────────┤
├── 1.3 OTLP Collector ◄────────────────┤
├── 1.4 Kafka Integration ◄─────────────┤
└── 1.5 Python SDK (Basic) ◄────────────┘

Phase 2: Storage & Processing
├── 2.1 ClickHouse Integration ◄────────┬── Phase 1
├── 2.2 Qdrant Integration ◄────────────┤
├── 2.3 Stream Processing Framework ◄───┤
├── 2.4 Cost Tracking ◄─────────────────┤
└── 2.5 Query API (Basic) ◄─────────────┘

Phase 3: Intelligence
├── 3.1 Feature Drift Detection ◄───────┬── Phase 2
├── 3.2 Embedding Drift Detection ◄─────┤
├── 3.3 Alerting System ◄───────────────┤
├── 3.4 Evaluators ◄────────────────────┤
└── 3.5 Root Cause Analysis ◄───────────┘

Phase 4: UI & Polish
├── 4.1 Web UI Foundation ◄─────────────┬── Phase 3
├── 4.2 Trace Explorer ◄────────────────┤
├── 4.3 Drift Dashboard ◄───────────────┤
├── 4.4 Cost Analytics ◄────────────────┤
├── 4.5 SDK Integrations ◄──────────────┤
├── 4.6 Grafana Plugin ◄────────────────┤
└── 4.7 Documentation & Testing ◄───────┘
```

---

## 9. Testing Strategy

### 9.1 Test Categories

| Category | Scope | Tools | Coverage Target |
|----------|-------|-------|-----------------|
| Unit Tests | Individual functions/classes | Google Test (C++), pytest (Python) | 80%+ |
| Integration Tests | Component interactions | Docker Compose, testcontainers | Critical paths |
| E2E Tests | Full system flows | Playwright (UI), custom harness | Happy paths |
| Performance Tests | Throughput, latency | k6, custom benchmarks | SLA targets |
| Security Tests | Vulnerabilities | OWASP ZAP, static analysis | No critical/high |

### 9.2 Test Infrastructure

```yaml
# docker-compose.test.yml
version: "3.8"

services:
  kafka:
    image: confluentinc/cp-kafka:7.5.0
    environment:
      KAFKA_NODE_ID: 1
      KAFKA_PROCESS_ROLES: broker,controller
      KAFKA_LISTENERS: PLAINTEXT://0.0.0.0:9092,CONTROLLER://0.0.0.0:9093
      KAFKA_CONTROLLER_QUORUM_VOTERS: 1@kafka:9093
      CLUSTER_ID: test-cluster-id

  clickhouse:
    image: clickhouse/clickhouse-server:24.1
    ports:
      - "9000:9000"
      - "8123:8123"

  qdrant:
    image: qdrant/qdrant:v1.7.0
    ports:
      - "6333:6333"
      - "6334:6334"

  redis:
    image: redis:7-alpine
    ports:
      - "6379:6379"
```

### 9.3 Continuous Integration

```yaml
# .github/workflows/ci.yml
name: CI

on:
  push:
    branches: [main, develop]
  pull_request:
    branches: [main]

jobs:
  cpp-build-test:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4

      - name: Install dependencies
        run: |
          sudo apt-get update
          sudo apt-get install -y cmake ninja-build clang-17

      - name: Configure
        run: cmake -B build -G Ninja -DCMAKE_CXX_COMPILER=clang++-17

      - name: Build
        run: cmake --build build

      - name: Test
        run: ctest --test-dir build --output-on-failure

      - name: Upload coverage
        uses: codecov/codecov-action@v3

  python-test:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4

      - name: Set up Python
        uses: actions/setup-python@v5
        with:
          python-version: "3.11"

      - name: Install dependencies
        run: |
          cd sdk/python
          pip install -e ".[dev]"

      - name: Run tests
        run: pytest --cov=pyflare --cov-report=xml

      - name: Upload coverage
        uses: codecov/codecov-action@v3

  integration-test:
    runs-on: ubuntu-latest
    services:
      kafka:
        image: confluentinc/cp-kafka:7.5.0
      clickhouse:
        image: clickhouse/clickhouse-server:24.1
      qdrant:
        image: qdrant/qdrant:v1.7.0
    steps:
      - uses: actions/checkout@v4
      - name: Run integration tests
        run: ./scripts/integration-test.sh

  lint:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4

      - name: C++ lint
        run: |
          clang-format --dry-run --Werror src/**/*.cpp src/**/*.h

      - name: Python lint
        run: |
          cd sdk/python
          pip install ruff black
          ruff check .
          black --check .
```

---

## 10. Deployment Architecture

### 10.1 Docker Compose (Development)

```yaml
# deploy/docker/docker-compose.yml
version: "3.8"

services:
  collector:
    build:
      context: ../..
      dockerfile: deploy/docker/Dockerfile.collector
    ports:
      - "4317:4317"  # gRPC
      - "4318:4318"  # HTTP
    environment:
      - PYFLARE_KAFKA_BROKERS=kafka:9092
      - PYFLARE_LOG_LEVEL=info
    depends_on:
      - kafka

  processor:
    build:
      context: ../..
      dockerfile: deploy/docker/Dockerfile.processor
    environment:
      - PYFLARE_KAFKA_BROKERS=kafka:9092
      - PYFLARE_CLICKHOUSE_HOST=clickhouse
      - PYFLARE_QDRANT_HOST=qdrant
    depends_on:
      - kafka
      - clickhouse
      - qdrant

  query-api:
    build:
      context: ../..
      dockerfile: deploy/docker/Dockerfile.query
    ports:
      - "8080:8080"
    environment:
      - PYFLARE_CLICKHOUSE_HOST=clickhouse
      - PYFLARE_QDRANT_HOST=qdrant
      - PYFLARE_REDIS_HOST=redis
    depends_on:
      - clickhouse
      - qdrant
      - redis

  ui:
    build:
      context: ../../ui
      dockerfile: Dockerfile
    ports:
      - "3000:3000"
    environment:
      - PYFLARE_API_URL=http://query-api:8080
    depends_on:
      - query-api

  # Infrastructure
  kafka:
    image: confluentinc/cp-kafka:7.5.0
    environment:
      KAFKA_NODE_ID: 1
      KAFKA_PROCESS_ROLES: broker,controller
      KAFKA_LISTENERS: PLAINTEXT://0.0.0.0:9092,CONTROLLER://0.0.0.0:9093
      KAFKA_ADVERTISED_LISTENERS: PLAINTEXT://kafka:9092
      KAFKA_CONTROLLER_LISTENER_NAMES: CONTROLLER
      KAFKA_CONTROLLER_QUORUM_VOTERS: 1@kafka:9093
      CLUSTER_ID: pyflare-dev-cluster
    ports:
      - "9092:9092"

  clickhouse:
    image: clickhouse/clickhouse-server:24.1
    ports:
      - "9000:9000"
      - "8123:8123"
    volumes:
      - clickhouse_data:/var/lib/clickhouse

  qdrant:
    image: qdrant/qdrant:v1.7.0
    ports:
      - "6333:6333"
      - "6334:6334"
    volumes:
      - qdrant_data:/qdrant/storage

  redis:
    image: redis:7-alpine
    ports:
      - "6379:6379"

volumes:
  clickhouse_data:
  qdrant_data:
```

### 10.2 Kubernetes (Production)

```yaml
# deploy/kubernetes/helm/pyflare/values.yaml
global:
  image:
    registry: ghcr.io/pyflare
    tag: "1.0.0"

collector:
  replicaCount: 3
  resources:
    requests:
      cpu: 500m
      memory: 512Mi
    limits:
      cpu: 2000m
      memory: 2Gi
  autoscaling:
    enabled: true
    minReplicas: 3
    maxReplicas: 20
    targetCPUUtilizationPercentage: 70
  service:
    type: LoadBalancer
    ports:
      grpc: 4317
      http: 4318

processor:
  replicaCount: 3
  resources:
    requests:
      cpu: 1000m
      memory: 1Gi
    limits:
      cpu: 4000m
      memory: 4Gi
  autoscaling:
    enabled: true
    minReplicas: 3
    maxReplicas: 10

queryApi:
  replicaCount: 2
  resources:
    requests:
      cpu: 500m
      memory: 512Mi
    limits:
      cpu: 2000m
      memory: 2Gi
  ingress:
    enabled: true
    className: nginx
    hosts:
      - host: api.pyflare.example.com
        paths:
          - path: /
            pathType: Prefix

ui:
  replicaCount: 2
  resources:
    requests:
      cpu: 100m
      memory: 128Mi
  ingress:
    enabled: true
    className: nginx
    hosts:
      - host: pyflare.example.com
        paths:
          - path: /
            pathType: Prefix

# External dependencies (typically managed separately)
kafka:
  enabled: false
  externalBrokers:
    - kafka-1.example.com:9092
    - kafka-2.example.com:9092
    - kafka-3.example.com:9092

clickhouse:
  enabled: false
  externalHost: clickhouse.example.com

qdrant:
  enabled: false
  externalHost: qdrant.example.com

redis:
  enabled: true
  architecture: standalone
  auth:
    enabled: true
```

### 10.3 Resource Sizing Guide

| Component | Small (<10K req/s) | Medium (<100K req/s) | Large (<1M req/s) |
|-----------|-------------------|---------------------|------------------|
| **Collector** | 2x (2 CPU, 2GB) | 5x (4 CPU, 4GB) | 20x (4 CPU, 8GB) |
| **Processor** | 2x (2 CPU, 4GB) | 5x (4 CPU, 8GB) | 10x (8 CPU, 16GB) |
| **Query API** | 2x (1 CPU, 1GB) | 3x (2 CPU, 2GB) | 5x (4 CPU, 4GB) |
| **Kafka** | 3 brokers | 5 brokers | 10+ brokers |
| **ClickHouse** | 1 node (8 CPU, 32GB) | 3 nodes (16 CPU, 64GB) | Cluster |
| **Qdrant** | 1 node (4 CPU, 16GB) | 3 nodes (8 CPU, 32GB) | Cluster |
| **Redis** | 1 node | Sentinel | Cluster |

---

## 11. Appendices

### A. Glossary

| Term | Definition |
|------|------------|
| **Drift** | Change in data distribution over time that may degrade model performance |
| **Embedding** | Dense vector representation of data (text, images, etc.) |
| **Hallucination** | LLM generating false or unsupported information |
| **OTLP** | OpenTelemetry Protocol - standard for transmitting telemetry data |
| **RAG** | Retrieval-Augmented Generation - combining retrieval with generation |
| **Span** | A single unit of work in a trace |
| **Trace** | End-to-end record of a request through a distributed system |

### B. Configuration Reference

See [Configuration Reference](./docs/configuration.md) for complete configuration options.

### C. API Reference

See [API Reference](./docs/api-reference.md) for complete API documentation.

### D. Changelog

| Version | Date | Changes |
|---------|------|---------|
| 1.0.0 | TBD | Initial release |

---

*This document is maintained as part of the PyFlare project. For questions or contributions, see the project repository.*
