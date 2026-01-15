#pragma once

/// @file pattern_detector.h
/// @brief Failure pattern detection for root cause analysis

#include <chrono>
#include <memory>
#include <string>
#include <unordered_map>
#include <vector>

#include <absl/status/statusor.h>

#include "processor/rca/analyzer.h"

namespace pyflare::rca {

/// @brief Pattern types detected by the analyzer
enum class PatternType {
    kErrorSpike,           ///< Sudden increase in errors
    kLatencyDegradation,   ///< Gradual latency increase
    kCostAnomaly,          ///< Unusual cost patterns
    kQualityDrop,          ///< Quality metric degradation
    kInputDrift,           ///< Input distribution shift
    kOutputDrift,          ///< Output distribution shift
    kCorrelatedFailures,   ///< Failures that occur together
    kPeriodicIssue,        ///< Issues recurring at intervals
    kNewFailureMode,       ///< Previously unseen error type
    kCustom
};

/// @brief Detected pattern details
struct Pattern {
    std::string id;
    PatternType type = PatternType::kCustom;
    std::string description;

    // Timing
    std::chrono::system_clock::time_point first_seen;
    std::chrono::system_clock::time_point last_seen;
    std::chrono::seconds duration{0};

    // Impact
    size_t affected_traces = 0;
    size_t affected_users = 0;
    double severity = 0.0;  ///< 0.0 - 1.0

    // Associated data
    std::vector<std::string> trace_ids;
    std::vector<std::string> error_messages;
    std::unordered_map<std::string, std::string> common_attributes;

    // Correlation
    std::vector<std::string> correlated_patterns;
    double correlation_strength = 0.0;

    // Suggested actions
    std::vector<std::string> suggested_actions;
    std::string root_cause_hypothesis;
};

/// @brief Configuration for pattern detector
struct PatternDetectorConfig {
    /// Minimum failures to consider as a pattern
    size_t min_pattern_size = 5;

    /// Time window for pattern detection
    std::chrono::hours detection_window = std::chrono::hours(24);

    /// Error rate increase threshold to detect spike
    double error_spike_threshold = 2.0;  // 2x increase

    /// Latency increase threshold
    double latency_degradation_threshold = 1.5;  // 50% increase

    /// Quality drop threshold
    double quality_drop_threshold = 0.2;  // 20% drop

    /// Minimum correlation coefficient for related patterns
    double min_correlation = 0.5;

    /// Enable ML-based pattern detection
    bool use_ml_detection = false;

    /// Pattern types to detect
    std::vector<PatternType> enabled_patterns = {
        PatternType::kErrorSpike,
        PatternType::kLatencyDegradation,
        PatternType::kQualityDrop,
        PatternType::kCorrelatedFailures
    };
};

/// @brief Pattern detector for identifying failure patterns
///
/// Detects various failure patterns:
/// - Error spikes and sudden degradations
/// - Gradual quality drift
/// - Correlated failures across dimensions
/// - Periodic/recurring issues
///
/// Example usage:
/// @code
///   PatternDetectorConfig config;
///   auto detector = std::make_unique<PatternDetector>(clickhouse, config);
///
///   // Detect patterns in recent failures
///   auto patterns = detector->DetectPatterns("my-model");
///
///   for (const auto& pattern : *patterns) {
///       std::cout << pattern.description << " (severity: "
///                 << pattern.severity << ")\n";
///       for (const auto& action : pattern.suggested_actions) {
///           std::cout << "  - " << action << "\n";
///       }
///   }
/// @endcode
class PatternDetector {
public:
    PatternDetector(std::shared_ptr<storage::ClickHouseClient> clickhouse,
                    PatternDetectorConfig config = {});
    ~PatternDetector();

    // Disable copy
    PatternDetector(const PatternDetector&) = delete;
    PatternDetector& operator=(const PatternDetector&) = delete;

    /// @brief Initialize the detector
    absl::Status Initialize();

    // =========================================================================
    // Pattern Detection
    // =========================================================================

    /// @brief Detect all patterns for a model
    /// @param model_id Model ID (empty for all models)
    absl::StatusOr<std::vector<Pattern>> DetectPatterns(
        const std::string& model_id = "");

    /// @brief Detect a specific pattern type
    absl::StatusOr<std::vector<Pattern>> DetectPatternType(
        const std::string& model_id,
        PatternType type);

    /// @brief Detect patterns from a set of failures
    absl::StatusOr<std::vector<Pattern>> DetectPatternsFromFailures(
        const std::vector<FailureRecord>& failures);

    /// @brief Find correlated patterns
    absl::StatusOr<std::vector<std::pair<Pattern, Pattern>>> FindCorrelatedPatterns(
        const std::vector<Pattern>& patterns);

    // =========================================================================
    // Specific Pattern Detection
    // =========================================================================

    /// @brief Detect error rate spikes
    absl::StatusOr<std::vector<Pattern>> DetectErrorSpikes(
        const std::string& model_id);

    /// @brief Detect latency degradation
    absl::StatusOr<std::vector<Pattern>> DetectLatencyDegradation(
        const std::string& model_id);

    /// @brief Detect quality drops
    absl::StatusOr<std::vector<Pattern>> DetectQualityDrops(
        const std::string& model_id);

    /// @brief Detect periodic issues
    absl::StatusOr<std::vector<Pattern>> DetectPeriodicIssues(
        const std::string& model_id);

    /// @brief Detect new failure modes
    absl::StatusOr<std::vector<Pattern>> DetectNewFailureModes(
        const std::string& model_id);

    // =========================================================================
    // Analysis Helpers
    // =========================================================================

    /// @brief Extract common attributes from traces
    std::unordered_map<std::string, std::string> ExtractCommonAttributes(
        const std::vector<std::string>& trace_ids);

    /// @brief Generate root cause hypothesis
    std::string GenerateHypothesis(const Pattern& pattern);

    /// @brief Generate suggested actions for a pattern
    std::vector<std::string> GenerateSuggestedActions(const Pattern& pattern);

    // =========================================================================
    // Configuration
    // =========================================================================

    /// @brief Get configuration
    const PatternDetectorConfig& GetConfig() const { return config_; }

    // =========================================================================
    // Helpers
    // =========================================================================

    /// @brief Convert pattern type to string
    static std::string PatternTypeToString(PatternType type);

    /// @brief Convert string to pattern type
    static PatternType StringToPatternType(const std::string& str);

private:
    /// @brief Calculate time series correlation
    double CalculateCorrelation(
        const std::vector<double>& series1,
        const std::vector<double>& series2);

    /// @brief Group failures by common attributes
    std::vector<std::vector<FailureRecord>> GroupFailures(
        const std::vector<FailureRecord>& failures);

    /// @brief Calculate pattern severity
    double CalculateSeverity(const Pattern& pattern);

    std::shared_ptr<storage::ClickHouseClient> clickhouse_;
    PatternDetectorConfig config_;
};

/// @brief Create a pattern detector with default configuration
std::unique_ptr<PatternDetector> CreatePatternDetector(
    std::shared_ptr<storage::ClickHouseClient> clickhouse,
    PatternDetectorConfig config = {});

}  // namespace pyflare::rca
