#pragma once

/// @file alert_rules.h
/// @brief Alert rules engine for configurable alerting
///
/// Provides a flexible rule-based alerting system that supports:
/// - Threshold-based rules
/// - Anomaly detection rules
/// - Composite rules (AND/OR combinations)
/// - Rate-based rules
/// - Pattern matching rules

#include <chrono>
#include <functional>
#include <memory>
#include <mutex>
#include <optional>
#include <string>
#include <unordered_map>
#include <variant>
#include <vector>

#include <absl/status/status.h>
#include <absl/status/statusor.h>

namespace pyflare::alerting {

/// @brief Alert severity levels
enum class AlertSeverity {
    kInfo,      ///< Informational
    kWarning,   ///< Warning
    kError,     ///< Error
    kCritical   ///< Critical
};

/// @brief Alert rule types
enum class RuleType {
    kThreshold,    ///< Simple threshold comparison
    kAnomaly,      ///< Statistical anomaly detection
    kRate,         ///< Rate of change
    kPattern,      ///< Pattern matching
    kComposite     ///< Combination of rules
};

/// @brief Threshold comparison operators
enum class ComparisonOp {
    kGreaterThan,
    kGreaterThanOrEqual,
    kLessThan,
    kLessThanOrEqual,
    kEqual,
    kNotEqual
};

/// @brief Metric value for evaluation
struct MetricValue {
    std::string name;
    double value = 0.0;
    std::chrono::system_clock::time_point timestamp;
    std::unordered_map<std::string, std::string> labels;
};

/// @brief Alert event generated when rule triggers
struct AlertEvent {
    std::string alert_id;
    std::string rule_id;
    std::string rule_name;
    AlertSeverity severity = AlertSeverity::kWarning;

    std::chrono::system_clock::time_point triggered_at;
    std::chrono::system_clock::time_point resolved_at;

    bool is_firing = true;
    bool is_resolved = false;

    std::string title;
    std::string description;

    /// Metric that triggered the alert
    std::string metric_name;
    double metric_value = 0.0;
    double threshold_value = 0.0;

    /// Context
    std::string model_id;
    std::unordered_map<std::string, std::string> labels;
    std::unordered_map<std::string, std::string> annotations;

    /// Fingerprint for deduplication
    std::string fingerprint;
};

/// @brief Threshold rule configuration
struct ThresholdRuleConfig {
    std::string metric_name;
    ComparisonOp comparison = ComparisonOp::kGreaterThan;
    double threshold = 0.0;

    /// Duration metric must exceed threshold
    std::chrono::seconds for_duration = std::chrono::seconds(0);

    /// Number of consecutive violations before firing
    size_t consecutive_count = 1;
};

/// @brief Anomaly rule configuration
struct AnomalyRuleConfig {
    std::string metric_name;

    /// Standard deviations from mean
    double std_dev_threshold = 3.0;

    /// Window for baseline calculation
    std::chrono::hours baseline_window = std::chrono::hours(24);

    /// Minimum samples for baseline
    size_t min_samples = 100;

    /// Detect spikes (positive), dips (negative), or both
    enum class Direction { kBoth, kUp, kDown };
    Direction direction = Direction::kBoth;
};

/// @brief Rate rule configuration
struct RateRuleConfig {
    std::string metric_name;

    /// Time window for rate calculation
    std::chrono::seconds window = std::chrono::minutes(5);

    /// Threshold for rate of change (per second)
    double rate_threshold = 0.0;

    /// Compare rate
    ComparisonOp comparison = ComparisonOp::kGreaterThan;

    /// Absolute or percentage rate
    bool use_percentage = false;
};

/// @brief Pattern rule configuration
struct PatternRuleConfig {
    std::string metric_name;

    /// Pattern type
    enum class PatternType {
        kSpike,           ///< Sudden increase then decrease
        kDip,             ///< Sudden decrease then increase
        kStep,            ///< Sudden level shift
        kTrend,           ///< Consistent upward/downward trend
        kOscillation      ///< Alternating values
    };
    PatternType pattern = PatternType::kSpike;

    /// Pattern detection sensitivity (0.0 - 1.0)
    double sensitivity = 0.7;

    /// Minimum duration for pattern
    std::chrono::seconds min_duration = std::chrono::minutes(5);
};

/// @brief Composite rule configuration
struct CompositeRuleConfig {
    /// Logical operator
    enum class LogicalOp { kAnd, kOr };
    LogicalOp op = LogicalOp::kAnd;

    /// Child rule IDs
    std::vector<std::string> rule_ids;

    /// All rules must fire within this window for AND
    std::chrono::seconds correlation_window = std::chrono::minutes(5);
};

/// @brief Alert rule definition
struct AlertRule {
    std::string rule_id;
    std::string name;
    std::string description;

    RuleType type = RuleType::kThreshold;
    AlertSeverity severity = AlertSeverity::kWarning;

    /// Type-specific configuration
    std::variant<
        ThresholdRuleConfig,
        AnomalyRuleConfig,
        RateRuleConfig,
        PatternRuleConfig,
        CompositeRuleConfig
    > config;

    /// Rule scope
    std::vector<std::string> model_ids;  ///< Empty = all models
    std::unordered_map<std::string, std::string> label_matchers;

    /// Alert metadata
    std::unordered_map<std::string, std::string> annotations;

    /// Enable/disable
    bool enabled = true;

    /// Timestamps
    std::chrono::system_clock::time_point created_at;
    std::chrono::system_clock::time_point updated_at;
};

/// @brief Rule evaluation result
struct RuleEvaluation {
    std::string rule_id;
    bool fired = false;
    bool pending = false;  ///< For duration-based rules
    double current_value = 0.0;
    double threshold_value = 0.0;
    std::string reason;
    std::chrono::system_clock::time_point evaluated_at;
};

/// @brief Alert rules engine
///
/// Evaluates alert rules against metric values and generates alerts.
///
/// Example:
/// @code
///   AlertRulesEngine engine;
///   engine.Initialize();
///
///   // Add threshold rule
///   AlertRule rule;
///   rule.rule_id = "high_error_rate";
///   rule.name = "High Error Rate";
///   rule.type = RuleType::kThreshold;
///   ThresholdRuleConfig config;
///   config.metric_name = "error_rate";
///   config.comparison = ComparisonOp::kGreaterThan;
///   config.threshold = 0.1;
///   rule.config = config;
///   engine.AddRule(rule);
///
///   // Evaluate metrics
///   MetricValue metric;
///   metric.name = "error_rate";
///   metric.value = 0.15;
///   auto alerts = engine.Evaluate({metric});
/// @endcode
class AlertRulesEngine {
public:
    AlertRulesEngine();
    ~AlertRulesEngine();

    // Disable copy
    AlertRulesEngine(const AlertRulesEngine&) = delete;
    AlertRulesEngine& operator=(const AlertRulesEngine&) = delete;

    /// @brief Initialize engine
    absl::Status Initialize();

    // =========================================================================
    // Rule Management
    // =========================================================================

    /// @brief Add alert rule
    absl::Status AddRule(const AlertRule& rule);

    /// @brief Update existing rule
    absl::Status UpdateRule(const AlertRule& rule);

    /// @brief Remove rule
    absl::Status RemoveRule(const std::string& rule_id);

    /// @brief Get rule by ID
    absl::StatusOr<AlertRule> GetRule(const std::string& rule_id) const;

    /// @brief List all rules
    std::vector<AlertRule> ListRules() const;

    /// @brief List rules by type
    std::vector<AlertRule> ListRulesByType(RuleType type) const;

    /// @brief Enable/disable rule
    absl::Status SetRuleEnabled(const std::string& rule_id, bool enabled);

    // =========================================================================
    // Evaluation
    // =========================================================================

    /// @brief Evaluate all rules against metrics
    /// @param metrics Current metric values
    /// @return List of generated alerts
    std::vector<AlertEvent> Evaluate(const std::vector<MetricValue>& metrics);

    /// @brief Evaluate specific rule
    /// @param rule_id Rule to evaluate
    /// @param metrics Current metric values
    absl::StatusOr<RuleEvaluation> EvaluateRule(
        const std::string& rule_id,
        const std::vector<MetricValue>& metrics);

    /// @brief Record metric for anomaly/rate detection
    /// @param metric Metric value to record
    void RecordMetric(const MetricValue& metric);

    // =========================================================================
    // Persistence
    // =========================================================================

    /// @brief Load rules from YAML file
    absl::Status LoadFromFile(const std::string& path);

    /// @brief Save rules to YAML file
    absl::Status SaveToFile(const std::string& path) const;

    /// @brief Load rules from JSON string
    absl::Status LoadFromJson(const std::string& json);

    /// @brief Export rules to JSON string
    absl::StatusOr<std::string> ExportToJson() const;

    // =========================================================================
    // Statistics
    // =========================================================================

    /// @brief Engine statistics
    struct Stats {
        size_t total_rules = 0;
        size_t enabled_rules = 0;
        size_t evaluations = 0;
        size_t alerts_generated = 0;
        std::unordered_map<RuleType, size_t> rules_by_type;
        std::unordered_map<AlertSeverity, size_t> alerts_by_severity;
    };
    Stats GetStats() const;

    /// @brief Reset statistics
    void ResetStats();

private:
    // Rule evaluation by type
    RuleEvaluation EvaluateThresholdRule(
        const AlertRule& rule,
        const std::vector<MetricValue>& metrics);

    RuleEvaluation EvaluateAnomalyRule(
        const AlertRule& rule,
        const std::vector<MetricValue>& metrics);

    RuleEvaluation EvaluateRateRule(
        const AlertRule& rule,
        const std::vector<MetricValue>& metrics);

    RuleEvaluation EvaluatePatternRule(
        const AlertRule& rule,
        const std::vector<MetricValue>& metrics);

    RuleEvaluation EvaluateCompositeRule(
        const AlertRule& rule,
        const std::vector<MetricValue>& metrics);

    // Helper functions
    std::optional<MetricValue> FindMetric(
        const std::vector<MetricValue>& metrics,
        const std::string& name) const;

    bool CompareValues(double value, ComparisonOp op, double threshold) const;

    std::string GenerateAlertId() const;
    std::string GenerateFingerprint(const AlertRule& rule,
                                    const MetricValue& metric) const;

    AlertEvent CreateAlertEvent(const AlertRule& rule,
                                const RuleEvaluation& eval,
                                const MetricValue& metric);

    // Rule state tracking
    struct RuleState {
        std::chrono::system_clock::time_point first_violation;
        size_t consecutive_violations = 0;
        bool is_firing = false;
        std::string last_fingerprint;
    };
    std::unordered_map<std::string, RuleState> rule_states_;

    // Metric history for anomaly/rate detection
    struct MetricHistory {
        std::vector<MetricValue> values;
        double mean = 0.0;
        double std_dev = 0.0;
        std::chrono::system_clock::time_point last_updated;
    };
    std::unordered_map<std::string, MetricHistory> metric_history_;

    // Rules storage
    std::unordered_map<std::string, AlertRule> rules_;
    mutable std::mutex rules_mutex_;

    // Statistics
    Stats stats_;
    mutable std::mutex stats_mutex_;

    bool initialized_ = false;
};

/// @brief Factory function
std::unique_ptr<AlertRulesEngine> CreateAlertRulesEngine();

/// @brief Convert severity to string
std::string AlertSeverityToString(AlertSeverity severity);

/// @brief Convert string to severity
AlertSeverity StringToAlertSeverity(const std::string& str);

/// @brief Convert rule type to string
std::string RuleTypeToString(RuleType type);

/// @brief Convert string to rule type
RuleType StringToRuleType(const std::string& str);

/// @brief Convert comparison operator to string
std::string ComparisonOpToString(ComparisonOp op);

/// @brief Convert string to comparison operator
ComparisonOp StringToComparisonOp(const std::string& str);

/// @brief Validate alert rule
absl::Status ValidateRule(const AlertRule& rule);

/// @brief Create simple threshold rule
AlertRule CreateThresholdRule(
    const std::string& name,
    const std::string& metric_name,
    ComparisonOp comparison,
    double threshold,
    AlertSeverity severity = AlertSeverity::kWarning);

/// @brief Create anomaly detection rule
AlertRule CreateAnomalyRule(
    const std::string& name,
    const std::string& metric_name,
    double std_dev_threshold = 3.0,
    AlertSeverity severity = AlertSeverity::kWarning);

/// @brief Create rate-based rule
AlertRule CreateRateRule(
    const std::string& name,
    const std::string& metric_name,
    double rate_threshold,
    std::chrono::seconds window = std::chrono::minutes(5),
    AlertSeverity severity = AlertSeverity::kWarning);

}  // namespace pyflare::alerting
