/// @file pattern_detector.cpp
/// @brief Pattern detection implementation

#include "processor/rca/pattern_detector.h"

#include <algorithm>
#include <cmath>
#include <numeric>
#include <sstream>

#include <spdlog/spdlog.h>

namespace pyflare::rca {

// =============================================================================
// PatternDetector Implementation
// =============================================================================

PatternDetector::PatternDetector(
    std::shared_ptr<storage::ClickHouseClient> clickhouse,
    PatternDetectorConfig config)
    : clickhouse_(std::move(clickhouse)),
      config_(std::move(config)) {}

PatternDetector::~PatternDetector() = default;

absl::Status PatternDetector::Initialize() {
    spdlog::info("PatternDetector initialized with {} enabled patterns",
                 config_.enabled_patterns.size());
    return absl::OkStatus();
}

absl::StatusOr<std::vector<Pattern>> PatternDetector::DetectPatterns(
    const std::string& model_id) {

    std::vector<Pattern> all_patterns;

    for (auto pattern_type : config_.enabled_patterns) {
        auto patterns = DetectPatternType(model_id, pattern_type);
        if (patterns.ok()) {
            for (auto& p : *patterns) {
                all_patterns.push_back(std::move(p));
            }
        }
    }

    // Sort by severity
    std::sort(all_patterns.begin(), all_patterns.end(),
              [](const auto& a, const auto& b) {
                  return a.severity > b.severity;
              });

    return all_patterns;
}

absl::StatusOr<std::vector<Pattern>> PatternDetector::DetectPatternType(
    const std::string& model_id,
    PatternType type) {

    switch (type) {
        case PatternType::kErrorSpike:
            return DetectErrorSpikes(model_id);
        case PatternType::kLatencyDegradation:
            return DetectLatencyDegradation(model_id);
        case PatternType::kQualityDrop:
            return DetectQualityDrops(model_id);
        case PatternType::kPeriodicIssue:
            return DetectPeriodicIssues(model_id);
        case PatternType::kNewFailureMode:
            return DetectNewFailureModes(model_id);
        default:
            return std::vector<Pattern>{};
    }
}

absl::StatusOr<std::vector<Pattern>> PatternDetector::DetectPatternsFromFailures(
    const std::vector<FailureRecord>& failures) {

    if (failures.size() < config_.min_pattern_size) {
        return std::vector<Pattern>{};
    }

    std::vector<Pattern> patterns;

    // Group failures by common attributes
    auto groups = GroupFailures(failures);

    for (const auto& group : groups) {
        if (group.size() >= config_.min_pattern_size) {
            Pattern pattern;
            pattern.id = "pattern_" + std::to_string(patterns.size());
            pattern.type = PatternType::kCorrelatedFailures;
            pattern.affected_traces = group.size();

            // Extract common attributes
            for (const auto& failure : group) {
                pattern.trace_ids.push_back(failure.trace_id);
                pattern.error_messages.push_back(failure.error_message);
            }

            // Set timing
            if (!group.empty()) {
                pattern.first_seen = group.front().timestamp;
                pattern.last_seen = group.back().timestamp;
                pattern.duration = std::chrono::duration_cast<std::chrono::seconds>(
                    pattern.last_seen - pattern.first_seen);
            }

            // Calculate severity and generate actions
            pattern.severity = CalculateSeverity(pattern);
            pattern.root_cause_hypothesis = GenerateHypothesis(pattern);
            pattern.suggested_actions = GenerateSuggestedActions(pattern);

            patterns.push_back(std::move(pattern));
        }
    }

    return patterns;
}

absl::StatusOr<std::vector<std::pair<Pattern, Pattern>>>
PatternDetector::FindCorrelatedPatterns(const std::vector<Pattern>& patterns) {

    std::vector<std::pair<Pattern, Pattern>> correlations;

    for (size_t i = 0; i < patterns.size(); ++i) {
        for (size_t j = i + 1; j < patterns.size(); ++j) {
            // Check temporal overlap
            bool overlaps =
                patterns[i].first_seen <= patterns[j].last_seen &&
                patterns[j].first_seen <= patterns[i].last_seen;

            if (overlaps) {
                // Calculate trace overlap
                std::set<std::string> traces_i(
                    patterns[i].trace_ids.begin(), patterns[i].trace_ids.end());
                std::set<std::string> traces_j(
                    patterns[j].trace_ids.begin(), patterns[j].trace_ids.end());

                std::vector<std::string> common;
                std::set_intersection(
                    traces_i.begin(), traces_i.end(),
                    traces_j.begin(), traces_j.end(),
                    std::back_inserter(common));

                double overlap_ratio =
                    static_cast<double>(common.size()) /
                    std::min(traces_i.size(), traces_j.size());

                if (overlap_ratio >= config_.min_correlation) {
                    correlations.emplace_back(patterns[i], patterns[j]);
                }
            }
        }
    }

    return correlations;
}

absl::StatusOr<std::vector<Pattern>> PatternDetector::DetectErrorSpikes(
    const std::string& model_id) {

    std::vector<Pattern> patterns;

    // Query for hourly error rates
    std::ostringstream query;
    query << "SELECT "
          << "toStartOfHour(timestamp) as hour, "
          << "count(*) as total, "
          << "countIf(is_error = 1) as errors, "
          << "errors / total as error_rate "
          << "FROM traces "
          << "WHERE timestamp >= now() - INTERVAL "
          << config_.detection_window.count() << " HOUR";

    if (!model_id.empty()) {
        query << " AND model_id = '" << model_id << "'";
    }

    query << " GROUP BY hour ORDER BY hour";

    auto result = clickhouse_->ExecuteQuery(query.str());
    if (!result.ok()) {
        return result.status();
    }

    // Analyze for spikes (placeholder - would parse actual results)
    // In production, compare each hour to baseline and detect 2x+ increases

    return patterns;
}

absl::StatusOr<std::vector<Pattern>> PatternDetector::DetectLatencyDegradation(
    const std::string& model_id) {

    std::vector<Pattern> patterns;

    std::ostringstream query;
    query << "SELECT "
          << "toStartOfHour(timestamp) as hour, "
          << "quantile(0.95)(latency_ms) as p95_latency "
          << "FROM traces "
          << "WHERE timestamp >= now() - INTERVAL "
          << config_.detection_window.count() << " HOUR";

    if (!model_id.empty()) {
        query << " AND model_id = '" << model_id << "'";
    }

    query << " GROUP BY hour ORDER BY hour";

    auto result = clickhouse_->ExecuteQuery(query.str());
    if (!result.ok()) {
        return result.status();
    }

    // Analyze for degradation trends
    // In production, fit linear regression and detect positive slopes

    return patterns;
}

absl::StatusOr<std::vector<Pattern>> PatternDetector::DetectQualityDrops(
    const std::string& model_id) {

    std::vector<Pattern> patterns;

    std::ostringstream query;
    query << "SELECT "
          << "toStartOfHour(timestamp) as hour, "
          << "avg(eval_score) as avg_quality "
          << "FROM traces "
          << "WHERE timestamp >= now() - INTERVAL "
          << config_.detection_window.count() << " HOUR"
          << " AND eval_score IS NOT NULL";

    if (!model_id.empty()) {
        query << " AND model_id = '" << model_id << "'";
    }

    query << " GROUP BY hour ORDER BY hour";

    auto result = clickhouse_->ExecuteQuery(query.str());
    if (!result.ok()) {
        return result.status();
    }

    return patterns;
}

absl::StatusOr<std::vector<Pattern>> PatternDetector::DetectPeriodicIssues(
    const std::string& model_id) {

    std::vector<Pattern> patterns;

    // Query error rates by hour of day across multiple days
    std::ostringstream query;
    query << "SELECT "
          << "toHour(timestamp) as hour_of_day, "
          << "avg(toUInt8(is_error = 1)) as error_rate "
          << "FROM traces "
          << "WHERE timestamp >= now() - INTERVAL 7 DAY";

    if (!model_id.empty()) {
        query << " AND model_id = '" << model_id << "'";
    }

    query << " GROUP BY hour_of_day ORDER BY hour_of_day";

    auto result = clickhouse_->ExecuteQuery(query.str());
    if (!result.ok()) {
        return result.status();
    }

    // Analyze for periodic patterns using FFT or autocorrelation
    // In production, detect recurring patterns at specific times

    return patterns;
}

absl::StatusOr<std::vector<Pattern>> PatternDetector::DetectNewFailureModes(
    const std::string& model_id) {

    std::vector<Pattern> patterns;

    // Compare recent errors to historical errors
    std::ostringstream query;
    query << "SELECT "
          << "error_type, "
          << "count(*) as count "
          << "FROM traces "
          << "WHERE is_error = 1 "
          << "AND timestamp >= now() - INTERVAL "
          << config_.detection_window.count() << " HOUR";

    if (!model_id.empty()) {
        query << " AND model_id = '" << model_id << "'";
    }

    query << " GROUP BY error_type "
          << "HAVING count >= " << config_.min_pattern_size;

    auto result = clickhouse_->ExecuteQuery(query.str());
    if (!result.ok()) {
        return result.status();
    }

    return patterns;
}

std::unordered_map<std::string, std::string>
PatternDetector::ExtractCommonAttributes(const std::vector<std::string>& trace_ids) {

    std::unordered_map<std::string, std::string> common;

    // Query would look up attributes for all trace IDs
    // and find values that appear in most/all traces

    return common;
}

std::string PatternDetector::GenerateHypothesis(const Pattern& pattern) {
    std::ostringstream hypothesis;

    switch (pattern.type) {
        case PatternType::kErrorSpike:
            hypothesis << "Sudden increase in errors may indicate: "
                       << "deployment issue, upstream service failure, "
                       << "or data quality problem";
            break;

        case PatternType::kLatencyDegradation:
            hypothesis << "Gradual latency increase may indicate: "
                       << "resource exhaustion, memory leak, "
                       << "or increased load";
            break;

        case PatternType::kQualityDrop:
            hypothesis << "Quality degradation may indicate: "
                       << "model drift, prompt template changes, "
                       << "or data distribution shift";
            break;

        case PatternType::kCorrelatedFailures:
            hypothesis << "Correlated failures may share: "
                       << "common input characteristics, "
                       << "infrastructure dependency, or timing";
            break;

        case PatternType::kPeriodicIssue:
            hypothesis << "Periodic issues may be caused by: "
                       << "scheduled jobs, traffic patterns, "
                       << "or resource contention";
            break;

        default:
            hypothesis << "Pattern detected - further investigation needed";
    }

    return hypothesis.str();
}

std::vector<std::string> PatternDetector::GenerateSuggestedActions(
    const Pattern& pattern) {

    std::vector<std::string> actions;

    switch (pattern.type) {
        case PatternType::kErrorSpike:
            actions.push_back("Review recent deployments and changes");
            actions.push_back("Check upstream service health");
            actions.push_back("Analyze error messages for common causes");
            break;

        case PatternType::kLatencyDegradation:
            actions.push_back("Check system resource utilization");
            actions.push_back("Review query patterns for inefficiencies");
            actions.push_back("Consider scaling resources");
            break;

        case PatternType::kQualityDrop:
            actions.push_back("Compare recent prompts to baseline");
            actions.push_back("Check for input distribution changes");
            actions.push_back("Evaluate need for model retraining");
            break;

        case PatternType::kCorrelatedFailures:
            actions.push_back("Identify common attributes in failing requests");
            actions.push_back("Test with similar inputs in isolation");
            actions.push_back("Add input validation for problematic patterns");
            break;

        case PatternType::kPeriodicIssue:
            actions.push_back("Correlate with scheduled jobs and cron tasks");
            actions.push_back("Check for traffic pattern changes");
            actions.push_back("Review rate limiting and throttling");
            break;

        default:
            actions.push_back("Investigate affected traces");
            actions.push_back("Monitor for recurrence");
    }

    return actions;
}

// =============================================================================
// Static Methods
// =============================================================================

std::string PatternDetector::PatternTypeToString(PatternType type) {
    switch (type) {
        case PatternType::kErrorSpike: return "error_spike";
        case PatternType::kLatencyDegradation: return "latency_degradation";
        case PatternType::kCostAnomaly: return "cost_anomaly";
        case PatternType::kQualityDrop: return "quality_drop";
        case PatternType::kInputDrift: return "input_drift";
        case PatternType::kOutputDrift: return "output_drift";
        case PatternType::kCorrelatedFailures: return "correlated_failures";
        case PatternType::kPeriodicIssue: return "periodic_issue";
        case PatternType::kNewFailureMode: return "new_failure_mode";
        case PatternType::kCustom: return "custom";
    }
    return "unknown";
}

PatternType PatternDetector::StringToPatternType(const std::string& str) {
    if (str == "error_spike") return PatternType::kErrorSpike;
    if (str == "latency_degradation") return PatternType::kLatencyDegradation;
    if (str == "cost_anomaly") return PatternType::kCostAnomaly;
    if (str == "quality_drop") return PatternType::kQualityDrop;
    if (str == "input_drift") return PatternType::kInputDrift;
    if (str == "output_drift") return PatternType::kOutputDrift;
    if (str == "correlated_failures") return PatternType::kCorrelatedFailures;
    if (str == "periodic_issue") return PatternType::kPeriodicIssue;
    if (str == "new_failure_mode") return PatternType::kNewFailureMode;
    return PatternType::kCustom;
}

// =============================================================================
// Private Methods
// =============================================================================

double PatternDetector::CalculateCorrelation(
    const std::vector<double>& series1,
    const std::vector<double>& series2) {

    if (series1.size() != series2.size() || series1.empty()) {
        return 0.0;
    }

    size_t n = series1.size();

    double sum1 = std::accumulate(series1.begin(), series1.end(), 0.0);
    double sum2 = std::accumulate(series2.begin(), series2.end(), 0.0);
    double mean1 = sum1 / n;
    double mean2 = sum2 / n;

    double cov = 0.0, var1 = 0.0, var2 = 0.0;
    for (size_t i = 0; i < n; ++i) {
        double d1 = series1[i] - mean1;
        double d2 = series2[i] - mean2;
        cov += d1 * d2;
        var1 += d1 * d1;
        var2 += d2 * d2;
    }

    double denom = std::sqrt(var1 * var2);
    return denom > 0 ? cov / denom : 0.0;
}

std::vector<std::vector<FailureRecord>> PatternDetector::GroupFailures(
    const std::vector<FailureRecord>& failures) {

    std::vector<std::vector<FailureRecord>> groups;

    // Group by error type
    std::unordered_map<std::string, std::vector<FailureRecord>> by_type;
    for (const auto& f : failures) {
        by_type[f.failure_type].push_back(f);
    }

    for (auto& [type, group] : by_type) {
        if (group.size() >= config_.min_pattern_size) {
            groups.push_back(std::move(group));
        }
    }

    return groups;
}

double PatternDetector::CalculateSeverity(const Pattern& pattern) {
    // Combine multiple factors
    double trace_factor = std::min(1.0, pattern.affected_traces / 100.0);
    double user_factor = std::min(1.0, pattern.affected_users / 50.0);
    double duration_factor = std::min(1.0, pattern.duration.count() / 3600.0);

    return (trace_factor + user_factor + duration_factor) / 3.0;
}

// =============================================================================
// Factory Function
// =============================================================================

std::unique_ptr<PatternDetector> CreatePatternDetector(
    std::shared_ptr<storage::ClickHouseClient> clickhouse,
    PatternDetectorConfig config) {
    return std::make_unique<PatternDetector>(std::move(clickhouse),
                                              std::move(config));
}

}  // namespace pyflare::rca
