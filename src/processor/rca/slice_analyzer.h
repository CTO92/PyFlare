#pragma once

/// @file slice_analyzer.h
/// @brief Data slice analysis for root cause analysis in PyFlare

#include <chrono>
#include <functional>
#include <memory>
#include <optional>
#include <string>
#include <unordered_map>
#include <vector>

#include <absl/status/statusor.h>

#include "processor/rca/analyzer.h"
#include "storage/clickhouse/client.h"

namespace pyflare::rca {

/// @brief Slice dimension types
enum class SliceDimension {
    kModel,
    kUser,
    kFeature,
    kInputLength,
    kOutputLength,
    kLatency,
    kTimeOfDay,
    kDayOfWeek,
    kPromptTemplate,
    kProvider,
    kEnvironment,
    kCustom
};

/// @brief Metric types for slice analysis
enum class SliceMetric {
    kErrorRate,
    kLatencyP50,
    kLatencyP95,
    kLatencyP99,
    kCost,
    kTokenUsage,
    kToxicityRate,
    kHallucinationRate,
    kDriftScore,
    kCustom
};

/// @brief Detailed slice definition
struct SliceDefinition {
    std::string name;
    SliceDimension dimension = SliceDimension::kCustom;
    std::string dimension_value;

    // For range-based dimensions (latency, input length)
    std::optional<double> range_min;
    std::optional<double> range_max;

    // For categorical dimensions
    std::vector<std::string> categories;

    // Custom SQL filter
    std::string custom_filter;
};

/// @brief Slice analysis result
struct SliceAnalysisResult {
    SliceDefinition definition;
    SliceMetric metric;

    // Statistics
    size_t sample_count = 0;
    double metric_value = 0.0;
    double baseline_value = 0.0;
    double deviation = 0.0;
    double deviation_percentage = 0.0;

    // Statistical significance
    double p_value = 0.0;
    double confidence_interval_lower = 0.0;
    double confidence_interval_upper = 0.0;
    bool is_statistically_significant = false;

    // Impact assessment
    double impact_score = 0.0;  ///< Combination of deviation and sample size
    size_t affected_users = 0;
    int64_t cost_impact_micros = 0;

    // Comparison
    std::string comparison_description;
    std::vector<std::string> related_slices;
};

/// @brief Configuration for slice analyzer
struct SliceAnalyzerConfig {
    /// Minimum samples for a slice to be analyzed
    size_t min_samples = 100;

    /// Maximum number of slices to analyze
    size_t max_slices = 50;

    /// Deviation threshold for reporting (percentage)
    double deviation_threshold = 0.1;

    /// P-value threshold for significance
    double p_value_threshold = 0.05;

    /// Time window for analysis
    std::chrono::hours analysis_window = std::chrono::hours(24);

    /// Dimensions to analyze automatically
    std::vector<SliceDimension> auto_dimensions = {
        SliceDimension::kModel,
        SliceDimension::kUser,
        SliceDimension::kFeature,
        SliceDimension::kInputLength,
        SliceDimension::kTimeOfDay
    };

    /// Metrics to analyze automatically
    std::vector<SliceMetric> auto_metrics = {
        SliceMetric::kErrorRate,
        SliceMetric::kLatencyP95,
        SliceMetric::kHallucinationRate
    };

    /// Custom slice definitions
    std::vector<SliceDefinition> custom_slices;
};

/// @brief Data slice analyzer for identifying problematic segments
///
/// Automatically discovers and analyzes data slices to identify:
/// - Segments with higher error rates
/// - Latency outliers
/// - Cost anomalies
/// - Quality degradation
///
/// Uses statistical testing to ensure findings are significant.
///
/// Example usage:
/// @code
///   SliceAnalyzerConfig config;
///   config.deviation_threshold = 0.2;  // 20% deviation
///   auto analyzer = std::make_unique<SliceAnalyzer>(clickhouse, config);
///
///   // Find problematic slices for error rate
///   auto results = analyzer->AnalyzeSlices("my-model", SliceMetric::kErrorRate);
///
///   for (const auto& result : *results) {
///       if (result.is_statistically_significant) {
///           std::cout << result.definition.name << ": "
///                     << result.deviation_percentage << "% higher error rate\n";
///       }
///   }
/// @endcode
class SliceAnalyzer {
public:
    SliceAnalyzer(std::shared_ptr<storage::ClickHouseClient> clickhouse,
                  SliceAnalyzerConfig config = {});
    ~SliceAnalyzer();

    // Disable copy
    SliceAnalyzer(const SliceAnalyzer&) = delete;
    SliceAnalyzer& operator=(const SliceAnalyzer&) = delete;

    /// @brief Initialize the analyzer
    absl::Status Initialize();

    // =========================================================================
    // Slice Analysis
    // =========================================================================

    /// @brief Analyze all configured slices for a metric
    /// @param model_id Model to analyze (empty for all models)
    /// @param metric Metric to analyze
    absl::StatusOr<std::vector<SliceAnalysisResult>> AnalyzeSlices(
        const std::string& model_id,
        SliceMetric metric);

    /// @brief Analyze a specific dimension
    /// @param model_id Model to analyze
    /// @param dimension Dimension to slice by
    /// @param metric Metric to analyze
    absl::StatusOr<std::vector<SliceAnalysisResult>> AnalyzeDimension(
        const std::string& model_id,
        SliceDimension dimension,
        SliceMetric metric);

    /// @brief Analyze a custom slice
    /// @param definition Custom slice definition
    /// @param metric Metric to analyze
    absl::StatusOr<SliceAnalysisResult> AnalyzeCustomSlice(
        const SliceDefinition& definition,
        SliceMetric metric);

    /// @brief Find top problematic slices across all dimensions
    /// @param model_id Model to analyze (empty for all)
    /// @param limit Maximum number of slices to return
    absl::StatusOr<std::vector<SliceAnalysisResult>> FindTopProblematicSlices(
        const std::string& model_id = "",
        size_t limit = 10);

    // =========================================================================
    // Baseline Management
    // =========================================================================

    /// @brief Get baseline metric value
    /// @param model_id Model ID
    /// @param metric Metric type
    absl::StatusOr<double> GetBaseline(const std::string& model_id,
                                        SliceMetric metric);

    /// @brief Update baseline from recent data
    absl::Status UpdateBaseline(const std::string& model_id,
                                 SliceMetric metric);

    /// @brief Set custom baseline value
    absl::Status SetBaseline(const std::string& model_id,
                              SliceMetric metric,
                              double value);

    // =========================================================================
    // Dimension Utilities
    // =========================================================================

    /// @brief Get unique values for a dimension
    absl::StatusOr<std::vector<std::string>> GetDimensionValues(
        const std::string& model_id,
        SliceDimension dimension);

    /// @brief Get suggested bin edges for numeric dimensions
    absl::StatusOr<std::vector<double>> GetSuggestedBins(
        const std::string& model_id,
        SliceDimension dimension,
        size_t num_bins = 10);

    // =========================================================================
    // Configuration
    // =========================================================================

    /// @brief Add custom slice definition
    void AddCustomSlice(const SliceDefinition& definition);

    /// @brief Get configuration
    const SliceAnalyzerConfig& GetConfig() const { return config_; }

    // =========================================================================
    // Helpers
    // =========================================================================

    /// @brief Convert dimension to string
    static std::string DimensionToString(SliceDimension dimension);

    /// @brief Convert string to dimension
    static SliceDimension StringToDimension(const std::string& str);

    /// @brief Convert metric to string
    static std::string MetricToString(SliceMetric metric);

    /// @brief Convert string to metric
    static SliceMetric StringToMetric(const std::string& str);

    /// @brief Get SQL column name for dimension
    static std::string DimensionToColumn(SliceDimension dimension);

    /// @brief Get SQL aggregation for metric
    static std::string MetricToAggregation(SliceMetric metric);

private:
    /// @brief Build query for slice analysis
    std::string BuildSliceQuery(const std::string& model_id,
                                 SliceDimension dimension,
                                 SliceMetric metric);

    /// @brief Calculate statistical significance
    void CalculateSignificance(SliceAnalysisResult& result,
                                size_t total_samples);

    /// @brief Calculate impact score
    void CalculateImpact(SliceAnalysisResult& result);

    std::shared_ptr<storage::ClickHouseClient> clickhouse_;
    SliceAnalyzerConfig config_;

    // Cached baselines
    std::unordered_map<std::string, double> baselines_;
    mutable std::mutex baselines_mutex_;
};

/// @brief Create slice definitions for common dimensions
std::vector<SliceDefinition> CreateDefaultSliceDefinitions();

}  // namespace pyflare::rca
