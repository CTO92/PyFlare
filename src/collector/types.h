#pragma once

/// @file types.h
/// @brief Common types for the PyFlare collector

#include <chrono>
#include <cstdint>
#include <functional>
#include <map>
#include <optional>
#include <string>
#include <variant>
#include <vector>

namespace pyflare::collector {

/// Attribute value type
using AttributeValue = std::variant<
    std::string,
    bool,
    int64_t,
    double,
    std::vector<std::string>,
    std::vector<int64_t>,
    std::vector<double>,
    std::vector<bool>>;

/// Span kind
enum class SpanKind {
    kUnspecified = 0,
    kInternal = 1,
    kServer = 2,
    kClient = 3,
    kProducer = 4,
    kConsumer = 5
};

/// Status code
enum class StatusCode {
    kUnset = 0,
    kOk = 1,
    kError = 2
};

/// Inference type
enum class InferenceType {
    kUnspecified = 0,
    kLlm = 1,
    kEmbedding = 2,
    kClassification = 3,
    kRegression = 4,
    kObjectDetection = 5,
    kSegmentation = 6,
    kGeneration = 7,
    kCustom = 8
};

/// Token usage for LLM inference
struct TokenUsage {
    uint32_t input_tokens = 0;
    uint32_t output_tokens = 0;
    uint32_t total_tokens = 0;
};

/// ML-specific attributes
struct MLAttributes {
    std::string model_id;
    std::string model_version;
    std::string model_provider;
    InferenceType inference_type = InferenceType::kUnspecified;
    std::optional<TokenUsage> token_usage;
    std::string input_preview;
    std::string output_preview;
    std::string input_hash;
    uint64_t cost_micros = 0;
    uint32_t embedding_dimensions = 0;
    std::map<std::string, AttributeValue> custom_attributes;
};

/// Span event
struct SpanEvent {
    uint64_t timestamp_ns = 0;
    std::string name;
    std::map<std::string, AttributeValue> attributes;
};

/// Span link
struct SpanLink {
    std::string trace_id;
    std::string span_id;
    std::string trace_state;
    std::map<std::string, AttributeValue> attributes;
};

/// Span status
struct SpanStatus {
    StatusCode code = StatusCode::kUnset;
    std::string message;
};

/// Resource attributes
struct Resource {
    std::map<std::string, AttributeValue> attributes;
    std::string schema_url;
};

/// Instrumentation scope
struct InstrumentationScope {
    std::string name;
    std::string version;
    std::map<std::string, AttributeValue> attributes;
};

/// Span data structure
struct Span {
    // Identity
    std::string trace_id;      // 16 bytes hex encoded
    std::string span_id;       // 8 bytes hex encoded
    std::string parent_span_id;
    std::string trace_state;

    // Basic info
    std::string name;
    SpanKind kind = SpanKind::kInternal;

    // Timing (nanoseconds since Unix epoch)
    uint64_t start_time_ns = 0;
    uint64_t end_time_ns = 0;

    // Attributes
    std::map<std::string, AttributeValue> attributes;

    // Events and links
    std::vector<SpanEvent> events;
    std::vector<SpanLink> links;

    // Status
    SpanStatus status;

    // ML-specific (PyFlare extension)
    std::optional<MLAttributes> ml_attributes;

    // Resource and scope (populated during enrichment)
    std::optional<Resource> resource;
    std::optional<InstrumentationScope> scope;

    /// Get duration in nanoseconds
    uint64_t DurationNs() const {
        return end_time_ns > start_time_ns ? end_time_ns - start_time_ns : 0;
    }

    /// Get duration in milliseconds
    double DurationMs() const {
        return static_cast<double>(DurationNs()) / 1'000'000.0;
    }
};

/// Metric data point
struct MetricDataPoint {
    uint64_t timestamp_ns = 0;
    std::string metric_name;
    std::map<std::string, std::string> labels;
    double value = 0.0;

    enum class Type {
        kGauge,
        kCounter,
        kHistogram,
        kSummary
    };
    Type type = Type::kGauge;

    // Histogram-specific
    std::vector<double> histogram_bounds;
    std::vector<uint64_t> histogram_counts;
    uint64_t histogram_count = 0;
    double histogram_sum = 0.0;
};

/// Log record
struct LogRecord {
    uint64_t timestamp_ns = 0;
    std::string trace_id;
    std::string span_id;
    std::string severity;
    std::string body;
    std::map<std::string, AttributeValue> attributes;
    std::optional<Resource> resource;
};

/// Callback types
using SpanCallback = std::function<void(std::vector<Span>&&)>;
using MetricCallback = std::function<void(std::vector<MetricDataPoint>&&)>;
using LogCallback = std::function<void(std::vector<LogRecord>&&)>;

}  // namespace pyflare::collector
