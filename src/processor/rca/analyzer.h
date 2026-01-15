#pragma once

/// @file analyzer.h
/// @brief Root Cause Analysis engine for PyFlare

#include <chrono>
#include <memory>
#include <string>
#include <vector>

#include <absl/status/statusor.h>

#include "eval/evaluator.h"

namespace pyflare::rca {

/// @brief A failure record for RCA
struct FailureRecord {
    std::string trace_id;
    std::string model_id;
    std::string failure_type;
    std::string error_message;
    eval::InferenceRecord inference;
    std::chrono::system_clock::time_point timestamp;
};

/// @brief A data slice with performance metrics
struct Slice {
    std::string name;
    std::unordered_map<std::string, std::string> filters;
    size_t sample_count;
    double metric_value;
    double baseline_value;
    double deviation;
    double confidence;
};

/// @brief RCA analysis report
struct RCAReport {
    std::vector<std::string> trace_ids_analyzed;
    std::chrono::system_clock::time_point analysis_time;

    struct Pattern {
        std::string description;
        std::vector<std::string> affected_trace_ids;
        double frequency;
        std::string suggested_action;
    };
    std::vector<Pattern> patterns;

    std::vector<Slice> problematic_slices;
};

/// @brief Root cause analyzer
class RootCauseAnalyzer {
public:
    struct Config {
        size_t min_failures_for_analysis = 10;
        size_t max_slices_to_report = 20;
        double slice_deviation_threshold = 0.2;
    };

    explicit RootCauseAnalyzer(Config config = {});
    ~RootCauseAnalyzer();

    /// @brief Analyze failures and identify patterns
    absl::StatusOr<RCAReport> Analyze(const std::vector<FailureRecord>& failures);

    /// @brief Find underperforming slices
    absl::StatusOr<std::vector<Slice>> FindProblematicSlices(
        const std::string& model_id,
        const std::string& metric);

private:
    Config config_;
    class Impl;
    std::unique_ptr<Impl> impl_;
};

}  // namespace pyflare::rca
