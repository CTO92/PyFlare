#pragma once

/// @file temporal_analyzer.h
/// @brief Temporal correlation analysis for root cause identification
///
/// Analyzes time-based patterns and correlations between events
/// to identify potential causal relationships.

#include <chrono>
#include <memory>
#include <mutex>
#include <optional>
#include <string>
#include <unordered_map>
#include <vector>

#include <absl/status/status.h>
#include <absl/status/statusor.h>

#include "storage/clickhouse/client.h"

namespace pyflare::rca {

/// @brief An event in the temporal analysis
struct TemporalEvent {
    std::string event_id;
    std::string event_type;  // "error", "drift", "deployment", "config_change", etc.
    std::string model_id;

    std::chrono::system_clock::time_point timestamp;

    /// Severity (0.0 - 1.0)
    double severity = 0.0;

    /// Additional attributes
    std::unordered_map<std::string, std::string> attributes;

    /// Description
    std::string description;
};

/// @brief Correlation between two event types
struct TemporalCorrelation {
    std::string event_type_a;
    std::string event_type_b;

    /// Correlation coefficient (-1.0 to 1.0)
    double correlation = 0.0;

    /// Time lag where correlation is strongest (positive = A leads B)
    std::chrono::minutes lag{0};

    /// P-value for statistical significance
    double p_value = 1.0;

    /// Whether correlation is statistically significant
    bool is_significant = false;

    /// Direction of relationship
    enum class Direction {
        kNone,       ///< No relationship
        kALeadsB,    ///< Event A typically precedes B
        kBLeadsA,    ///< Event B typically precedes A
        kConcurrent  ///< Events occur together
    };
    Direction direction = Direction::kNone;

    /// Strength interpretation
    std::string strength;  // "strong", "moderate", "weak", "none"

    /// Human-readable explanation
    std::string explanation;
};

/// @brief A detected temporal pattern
struct TemporalPattern {
    std::string pattern_id;
    std::string pattern_type;  // "spike", "trend", "periodicity", "anomaly"

    /// Events involved in the pattern
    std::vector<std::string> event_types;

    /// Time window where pattern occurs
    std::chrono::system_clock::time_point start_time;
    std::chrono::system_clock::time_point end_time;

    /// Pattern characteristics
    double frequency = 0.0;  ///< For periodic patterns
    double trend_slope = 0.0;  ///< For trend patterns
    double anomaly_score = 0.0;  ///< For anomaly patterns

    /// Confidence in pattern detection
    double confidence = 0.0;

    /// Description
    std::string description;
};

/// @brief Configuration for temporal analysis
struct TemporalAnalyzerConfig {
    /// Analysis time window
    std::chrono::hours analysis_window = std::chrono::hours(24);

    /// Resolution for time series analysis
    std::chrono::minutes resolution = std::chrono::minutes(5);

    /// Maximum lag to consider for correlations
    std::chrono::hours max_lag = std::chrono::hours(4);

    /// P-value threshold for significance
    double p_value_threshold = 0.05;

    /// Minimum correlation coefficient for "significant"
    double min_correlation = 0.3;

    /// Minimum events to compute correlation
    size_t min_events = 10;

    /// Event types to include in analysis
    std::vector<std::string> event_types = {
        "error", "drift", "latency_spike", "cost_spike",
        "deployment", "config_change", "model_update"
    };

    /// Enable pattern detection
    bool detect_patterns = true;

    /// Enable anomaly detection
    bool detect_anomalies = true;

    /// Anomaly detection sensitivity (0.0 - 1.0)
    double anomaly_sensitivity = 0.8;
};

/// @brief Temporal correlation analyzer
///
/// Identifies time-based relationships between events to help
/// understand root causes of issues.
///
/// Capabilities:
/// - Cross-correlation between event time series
/// - Lead/lag relationship detection
/// - Pattern detection (spikes, trends, periodicity)
/// - Anomaly detection in event sequences
///
/// Example:
/// @code
///   TemporalAnalyzerConfig config;
///   config.analysis_window = std::chrono::hours(48);
///   auto analyzer = std::make_unique<TemporalAnalyzer>(clickhouse, config);
///   analyzer->Initialize();
///
///   // Find correlations for a model
///   auto correlations = analyzer->FindCorrelations("my-model");
///   for (const auto& corr : *correlations) {
///       if (corr.is_significant && corr.direction == Direction::kALeadsB) {
///           LOG(INFO) << corr.event_type_a << " appears to cause "
///                     << corr.event_type_b << " (lag: " << corr.lag.count() << "m)";
///       }
///   }
/// @endcode
class TemporalAnalyzer {
public:
    TemporalAnalyzer(
        std::shared_ptr<storage::ClickHouseClient> clickhouse,
        TemporalAnalyzerConfig config = {});
    ~TemporalAnalyzer();

    // Disable copy
    TemporalAnalyzer(const TemporalAnalyzer&) = delete;
    TemporalAnalyzer& operator=(const TemporalAnalyzer&) = delete;

    /// @brief Initialize analyzer
    absl::Status Initialize();

    // =========================================================================
    // Correlation Analysis
    // =========================================================================

    /// @brief Find all correlations for a model
    /// @param model_id Model to analyze
    absl::StatusOr<std::vector<TemporalCorrelation>> FindCorrelations(
        const std::string& model_id);

    /// @brief Find correlation between two specific event types
    /// @param model_id Model to analyze
    /// @param event_type_a First event type
    /// @param event_type_b Second event type
    absl::StatusOr<TemporalCorrelation> FindCorrelation(
        const std::string& model_id,
        const std::string& event_type_a,
        const std::string& event_type_b);

    /// @brief Find events that precede a target event type
    /// @param model_id Model to analyze
    /// @param target_event Target event type to explain
    /// @param max_results Maximum number of results
    absl::StatusOr<std::vector<TemporalCorrelation>> FindPrecedingEvents(
        const std::string& model_id,
        const std::string& target_event,
        size_t max_results = 10);

    // =========================================================================
    // Pattern Detection
    // =========================================================================

    /// @brief Detect temporal patterns for a model
    /// @param model_id Model to analyze
    absl::StatusOr<std::vector<TemporalPattern>> DetectPatterns(
        const std::string& model_id);

    /// @brief Detect anomalies in event sequence
    /// @param model_id Model to analyze
    absl::StatusOr<std::vector<TemporalPattern>> DetectAnomalies(
        const std::string& model_id);

    /// @brief Check if there's a trend in event frequency
    /// @param model_id Model to analyze
    /// @param event_type Event type to check
    absl::StatusOr<TemporalPattern> DetectTrend(
        const std::string& model_id,
        const std::string& event_type);

    // =========================================================================
    // Event Management
    // =========================================================================

    /// @brief Add an event for analysis
    /// @param event Event to add
    absl::Status AddEvent(const TemporalEvent& event);

    /// @brief Query events from storage
    /// @param model_id Model ID
    /// @param event_type Event type (empty for all)
    /// @param start Start time
    /// @param end End time
    absl::StatusOr<std::vector<TemporalEvent>> QueryEvents(
        const std::string& model_id,
        const std::string& event_type = "",
        std::optional<std::chrono::system_clock::time_point> start = std::nullopt,
        std::optional<std::chrono::system_clock::time_point> end = std::nullopt);

    // =========================================================================
    // Time Series Analysis
    // =========================================================================

    /// @brief Get event count time series
    /// @param model_id Model ID
    /// @param event_type Event type
    /// @param resolution Time resolution
    absl::StatusOr<std::vector<std::pair<std::chrono::system_clock::time_point, double>>>
    GetTimeSeries(
        const std::string& model_id,
        const std::string& event_type,
        std::chrono::minutes resolution = std::chrono::minutes(5));

    /// @brief Compute cross-correlation between two time series
    /// @param series_a First time series
    /// @param series_b Second time series
    /// @param max_lag Maximum lag to consider
    absl::StatusOr<std::vector<std::pair<int, double>>> ComputeCrossCorrelation(
        const std::vector<double>& series_a,
        const std::vector<double>& series_b,
        size_t max_lag);

    // =========================================================================
    // Configuration
    // =========================================================================

    /// @brief Update configuration
    void SetConfig(TemporalAnalyzerConfig config);

    /// @brief Get configuration
    const TemporalAnalyzerConfig& GetConfig() const { return config_; }

    // =========================================================================
    // Statistics
    // =========================================================================

    /// @brief Get analysis statistics
    struct Stats {
        size_t correlations_computed = 0;
        size_t patterns_detected = 0;
        size_t anomalies_detected = 0;
        size_t events_processed = 0;
        double avg_analysis_time_ms = 0.0;
    };
    Stats GetStats() const;

    /// @brief Reset statistics
    void ResetStats();

private:
    // Time series computation
    std::vector<double> ComputeEventCounts(
        const std::vector<TemporalEvent>& events,
        std::chrono::system_clock::time_point start,
        std::chrono::system_clock::time_point end,
        std::chrono::minutes resolution);

    // Statistical computations
    double ComputePearsonCorrelation(
        const std::vector<double>& x,
        const std::vector<double>& y);

    double ComputePValue(double correlation, size_t n);

    std::pair<int, double> FindBestLag(
        const std::vector<double>& series_a,
        const std::vector<double>& series_b,
        size_t max_lag);

    // Pattern detection helpers
    std::optional<TemporalPattern> DetectSpike(
        const std::vector<double>& series,
        std::chrono::system_clock::time_point start,
        std::chrono::minutes resolution);

    std::optional<TemporalPattern> DetectPeriodicity(
        const std::vector<double>& series,
        std::chrono::system_clock::time_point start,
        std::chrono::minutes resolution);

    double ComputeTrendSlope(const std::vector<double>& series);

    // Anomaly detection helpers
    std::vector<size_t> DetectAnomalyPoints(
        const std::vector<double>& series,
        double sensitivity);

    // Result building
    TemporalCorrelation BuildCorrelation(
        const std::string& event_type_a,
        const std::string& event_type_b,
        double correlation,
        int lag_minutes,
        double p_value);

    // Query helpers
    std::string BuildEventQuery(
        const std::string& model_id,
        const std::string& event_type,
        std::chrono::system_clock::time_point start,
        std::chrono::system_clock::time_point end);

    std::shared_ptr<storage::ClickHouseClient> clickhouse_;
    TemporalAnalyzerConfig config_;

    // In-memory event buffer
    std::vector<TemporalEvent> event_buffer_;
    mutable std::mutex buffer_mutex_;

    // Statistics
    Stats stats_;
    mutable std::mutex stats_mutex_;

    bool initialized_ = false;
};

/// @brief Factory function to create temporal analyzer
std::unique_ptr<TemporalAnalyzer> CreateTemporalAnalyzer(
    std::shared_ptr<storage::ClickHouseClient> clickhouse,
    TemporalAnalyzerConfig config = {});

/// @brief Convert correlation direction to string
std::string DirectionToString(TemporalCorrelation::Direction direction);

/// @brief Interpret correlation strength
std::string InterpretCorrelation(double correlation);

}  // namespace pyflare::rca
