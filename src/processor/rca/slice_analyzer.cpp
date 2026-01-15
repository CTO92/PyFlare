/// @file slice_analyzer.cpp
/// @brief Data slice analyzer implementation
///
/// SECURITY: All queries use parameterized execution to prevent SQL injection.
/// Custom filters use a safe expression parser instead of raw SQL.

#include "processor/rca/slice_analyzer.h"

#include <algorithm>
#include <cmath>
#include <numeric>
#include <regex>
#include <sstream>

#include <spdlog/spdlog.h>

namespace pyflare::rca {

// =============================================================================
// SECURITY: Input Validation and Safe Expression Parsing
// =============================================================================

/// @brief Maximum allowed model_id length
static constexpr size_t kMaxModelIdLength = 256;

/// @brief Validate model_id format (alphanumeric, hyphens, underscores, dots)
static bool IsValidModelId(const std::string& model_id) {
    if (model_id.empty() || model_id.size() > kMaxModelIdLength) {
        return false;
    }
    static const std::regex model_id_regex("^[a-zA-Z0-9_\\-\\.]+$");
    return std::regex_match(model_id, model_id_regex);
}

/// @brief Allowed columns for custom filter expressions
static const std::unordered_set<std::string> kAllowedFilterColumns = {
    "model_id", "user_id", "feature_id", "provider", "environment",
    "status", "error_type", "input_tokens", "output_tokens", "latency_ms",
    "cost_micros", "eval_score", "drift_score", "toxicity_score"
};

/// @brief Allowed operators for custom filter expressions
static const std::unordered_set<std::string> kAllowedFilterOperators = {
    "=", "!=", "<>", "<", ">", "<=", ">=", "IN", "NOT IN", "LIKE", "NOT LIKE",
    "BETWEEN", "IS NULL", "IS NOT NULL"
};

/// @brief Parse and validate a custom filter expression
/// Returns a parameterized query clause and parameters, or error
/// Format: "column operator value" or "column operator value AND/OR ..."
static absl::StatusOr<std::pair<std::string, std::vector<storage::QueryParam>>>
ParseSafeFilterExpression(const std::string& filter, int& param_counter) {
    if (filter.empty()) {
        return std::make_pair(std::string(""), std::vector<storage::QueryParam>{});
    }

    // SECURITY: Maximum filter length to prevent DoS
    if (filter.size() > 1024) {
        return absl::InvalidArgumentError("Filter expression too long");
    }

    std::vector<storage::QueryParam> params;
    std::ostringstream result;

    // Simple tokenizer for filter expressions
    // Format: column op value [AND|OR column op value]*
    std::istringstream stream(filter);
    std::string token;
    bool first_condition = true;

    while (stream >> token) {
        // Check for logical operators
        std::string upper_token = token;
        std::transform(upper_token.begin(), upper_token.end(), upper_token.begin(), ::toupper);

        if (upper_token == "AND" || upper_token == "OR") {
            if (first_condition) {
                return absl::InvalidArgumentError("Filter cannot start with AND/OR");
            }
            result << " " << upper_token << " ";
            continue;
        }

        // Parse column name
        std::string column = token;
        std::transform(column.begin(), column.end(), column.begin(), ::tolower);

        if (kAllowedFilterColumns.find(column) == kAllowedFilterColumns.end()) {
            return absl::InvalidArgumentError(
                absl::StrCat("Invalid filter column: ", column));
        }

        // Parse operator
        std::string op;
        if (!(stream >> op)) {
            return absl::InvalidArgumentError("Missing operator in filter");
        }
        std::string upper_op = op;
        std::transform(upper_op.begin(), upper_op.end(), upper_op.begin(), ::toupper);

        // Handle multi-word operators
        if (upper_op == "NOT") {
            std::string next_word;
            if (!(stream >> next_word)) {
                return absl::InvalidArgumentError("Incomplete operator");
            }
            std::transform(next_word.begin(), next_word.end(), next_word.begin(), ::toupper);
            upper_op = upper_op + " " + next_word;
        } else if (upper_op == "IS") {
            std::string next_word;
            if (!(stream >> next_word)) {
                return absl::InvalidArgumentError("Incomplete operator");
            }
            std::transform(next_word.begin(), next_word.end(), next_word.begin(), ::toupper);
            upper_op = upper_op + " " + next_word;
            if (upper_op == "IS NOT") {
                std::string null_word;
                if (!(stream >> null_word)) {
                    return absl::InvalidArgumentError("Expected NULL after IS NOT");
                }
                upper_op = upper_op + " " + null_word;
            }
        }

        if (kAllowedFilterOperators.find(upper_op) == kAllowedFilterOperators.end()) {
            return absl::InvalidArgumentError(
                absl::StrCat("Invalid filter operator: ", op));
        }

        // Handle IS NULL / IS NOT NULL (no value needed)
        if (upper_op == "IS NULL" || upper_op == "IS NOT NULL") {
            if (!first_condition) {
                // Already have space from AND/OR
            }
            result << column << " " << upper_op;
            first_condition = false;
            continue;
        }

        // Parse value
        std::string value;
        if (!(stream >> value)) {
            return absl::InvalidArgumentError("Missing value in filter");
        }

        // Remove quotes if present
        if ((value.front() == '\'' && value.back() == '\'') ||
            (value.front() == '"' && value.back() == '"')) {
            value = value.substr(1, value.size() - 2);
        }

        // Build parameterized condition
        std::string param_name = "filter_param_" + std::to_string(param_counter++);

        // Determine type based on column
        std::string param_type = "String";
        if (column == "input_tokens" || column == "output_tokens" ||
            column == "latency_ms" || column == "cost_micros") {
            param_type = "Int64";
        } else if (column == "eval_score" || column == "drift_score" ||
                   column == "toxicity_score") {
            param_type = "Float64";
        }

        result << column << " " << upper_op << " {" << param_name << ":" << param_type << "}";
        params.push_back({param_name, value, param_type});
        first_condition = false;
    }

    return std::make_pair(result.str(), params);
}

// =============================================================================
// SliceAnalyzer Implementation
// =============================================================================

SliceAnalyzer::SliceAnalyzer(
    std::shared_ptr<storage::ClickHouseClient> clickhouse,
    SliceAnalyzerConfig config)
    : clickhouse_(std::move(clickhouse)),
      config_(std::move(config)) {}

SliceAnalyzer::~SliceAnalyzer() = default;

absl::Status SliceAnalyzer::Initialize() {
    if (!clickhouse_) {
        return absl::FailedPreconditionError("ClickHouse client not provided");
    }

    // Initialize baselines for common models
    spdlog::info("SliceAnalyzer initialized with {} auto-dimensions, {} auto-metrics",
                 config_.auto_dimensions.size(), config_.auto_metrics.size());
    return absl::OkStatus();
}

absl::StatusOr<std::vector<SliceAnalysisResult>> SliceAnalyzer::AnalyzeSlices(
    const std::string& model_id,
    SliceMetric metric) {

    std::vector<SliceAnalysisResult> all_results;

    // Analyze each configured dimension
    for (auto dimension : config_.auto_dimensions) {
        auto dim_results = AnalyzeDimension(model_id, dimension, metric);
        if (dim_results.ok()) {
            for (auto& result : *dim_results) {
                all_results.push_back(std::move(result));
            }
        } else {
            spdlog::warn("Failed to analyze dimension {}: {}",
                         DimensionToString(dimension),
                         dim_results.status().message());
        }
    }

    // Analyze custom slices
    for (const auto& slice_def : config_.custom_slices) {
        auto result = AnalyzeCustomSlice(slice_def, metric);
        if (result.ok()) {
            all_results.push_back(std::move(*result));
        }
    }

    // Sort by impact score
    std::sort(all_results.begin(), all_results.end(),
              [](const auto& a, const auto& b) {
                  return a.impact_score > b.impact_score;
              });

    // Limit results
    if (all_results.size() > config_.max_slices) {
        all_results.resize(config_.max_slices);
    }

    return all_results;
}

absl::StatusOr<std::vector<SliceAnalysisResult>> SliceAnalyzer::AnalyzeDimension(
    const std::string& model_id,
    SliceDimension dimension,
    SliceMetric metric) {

    // SECURITY: Validate model_id format
    if (!model_id.empty() && !IsValidModelId(model_id)) {
        return absl::InvalidArgumentError("Invalid model_id format");
    }

    std::vector<SliceAnalysisResult> results;

    // Get baseline
    auto baseline_result = GetBaseline(model_id, metric);
    double baseline = baseline_result.ok() ? *baseline_result : 0.0;

    // SECURITY: Build and execute query with parameters
    auto [query, params] = BuildSliceQueryWithParams(model_id, dimension, metric);

    auto query_result = clickhouse_->ExecuteWithParams(query, params);
    if (!query_result.ok()) {
        return query_result.status();
    }

    // Parse results (assuming JSON format)
    // In production, this would use proper result parsing
    // For now, create mock results based on dimension

    auto dim_values = GetDimensionValues(model_id, dimension);
    if (!dim_values.ok()) {
        return dim_values.status();
    }

    // Calculate total samples for significance testing
    size_t total_samples = 0;
    for (const auto& val : *dim_values) {
        total_samples += config_.min_samples;  // Placeholder
    }

    for (const auto& value : *dim_values) {
        SliceAnalysisResult result;

        result.definition.name = DimensionToString(dimension) + "=" + value;
        result.definition.dimension = dimension;
        result.definition.dimension_value = value;
        result.metric = metric;

        // These would come from actual query results
        result.sample_count = config_.min_samples;
        result.metric_value = baseline;  // Placeholder
        result.baseline_value = baseline;
        result.deviation = 0.0;
        result.deviation_percentage = 0.0;

        // Calculate significance
        CalculateSignificance(result, total_samples);

        // Calculate impact
        CalculateImpact(result);

        // Only include if significant deviation
        if (std::abs(result.deviation_percentage) >= config_.deviation_threshold * 100) {
            results.push_back(std::move(result));
        }
    }

    return results;
}

absl::StatusOr<SliceAnalysisResult> SliceAnalyzer::AnalyzeCustomSlice(
    const SliceDefinition& definition,
    SliceMetric metric) {

    SliceAnalysisResult result;
    result.definition = definition;
    result.metric = metric;

    // SECURITY: Build custom query with parameterized filter
    std::ostringstream query;
    std::vector<storage::QueryParam> params;
    int param_counter = 0;

    query << "SELECT " << MetricToAggregation(metric) << " as metric_value, "
          << "count(*) as sample_count "
          << "FROM traces "
          << "WHERE 1=1";

    // SECURITY: Parse custom_filter through safe expression parser instead of raw SQL
    if (!definition.custom_filter.empty()) {
        auto filter_result = ParseSafeFilterExpression(definition.custom_filter, param_counter);
        if (!filter_result.ok()) {
            spdlog::warn("Invalid custom filter expression: {}",
                         filter_result.status().message());
            return filter_result.status();
        }

        auto [filter_clause, filter_params] = *filter_result;
        if (!filter_clause.empty()) {
            query << " AND (" << filter_clause << ")";
            for (const auto& p : filter_params) {
                params.push_back(p);
            }
        }
    }

    // Add time window with parameter
    query << " AND timestamp >= now() - INTERVAL {window_hours:Int64} HOUR";
    params.push_back({"window_hours", std::to_string(config_.analysis_window.count()), "Int64"});

    auto query_result = clickhouse_->ExecuteWithParams(query.str(), params);
    if (!query_result.ok()) {
        return query_result.status();
    }

    // Parse result (placeholder)
    result.sample_count = config_.min_samples;
    result.baseline_value = 0.0;
    result.metric_value = 0.0;

    return result;
}

absl::StatusOr<std::vector<SliceAnalysisResult>> SliceAnalyzer::FindTopProblematicSlices(
    const std::string& model_id,
    size_t limit) {

    // SECURITY: Validate model_id format
    if (!model_id.empty() && !IsValidModelId(model_id)) {
        return absl::InvalidArgumentError("Invalid model_id format");
    }

    // SECURITY: Limit result count to prevent DoS
    if (limit > 1000) {
        limit = 1000;
    }

    std::vector<SliceAnalysisResult> all_results;

    // Analyze all metrics
    for (auto metric : config_.auto_metrics) {
        auto results = AnalyzeSlices(model_id, metric);
        if (results.ok()) {
            for (auto& r : *results) {
                if (r.is_statistically_significant) {
                    all_results.push_back(std::move(r));
                }
            }
        }
    }

    // Sort by impact
    std::sort(all_results.begin(), all_results.end(),
              [](const auto& a, const auto& b) {
                  return a.impact_score > b.impact_score;
              });

    // Limit
    if (all_results.size() > limit) {
        all_results.resize(limit);
    }

    return all_results;
}

absl::StatusOr<double> SliceAnalyzer::GetBaseline(const std::string& model_id,
                                                   SliceMetric metric) {
    // SECURITY: Validate model_id format
    if (!model_id.empty() && !IsValidModelId(model_id)) {
        return absl::InvalidArgumentError("Invalid model_id format");
    }

    std::string key = model_id + ":" + MetricToString(metric);

    {
        std::lock_guard<std::mutex> lock(baselines_mutex_);
        auto it = baselines_.find(key);
        if (it != baselines_.end()) {
            return it->second;
        }
    }

    // SECURITY: Use parameterized query to prevent SQL injection
    std::ostringstream query;
    std::vector<storage::QueryParam> params;

    query << "SELECT " << MetricToAggregation(metric)
          << " FROM traces WHERE 1=1";

    if (!model_id.empty()) {
        query << " AND model_id = {model_id:String}";
        params.push_back({"model_id", model_id, "String"});
    }

    query << " AND timestamp >= now() - INTERVAL {window_hours:Int64} HOUR";
    params.push_back({"window_hours", std::to_string(config_.analysis_window.count() * 7), "Int64"});

    auto result = clickhouse_->ExecuteWithParams(query.str(), params);
    if (!result.ok()) {
        return result.status();
    }

    // Parse and cache (placeholder value)
    double baseline = 0.0;
    {
        std::lock_guard<std::mutex> lock(baselines_mutex_);
        baselines_[key] = baseline;
    }

    return baseline;
}

absl::Status SliceAnalyzer::UpdateBaseline(const std::string& model_id,
                                            SliceMetric metric) {
    std::string key = model_id + ":" + MetricToString(metric);

    // Remove cached baseline to force recalculation
    {
        std::lock_guard<std::mutex> lock(baselines_mutex_);
        baselines_.erase(key);
    }

    // Recalculate
    auto result = GetBaseline(model_id, metric);
    return result.status();
}

absl::Status SliceAnalyzer::SetBaseline(const std::string& model_id,
                                         SliceMetric metric,
                                         double value) {
    std::string key = model_id + ":" + MetricToString(metric);

    std::lock_guard<std::mutex> lock(baselines_mutex_);
    baselines_[key] = value;

    return absl::OkStatus();
}

absl::StatusOr<std::vector<std::string>> SliceAnalyzer::GetDimensionValues(
    const std::string& model_id,
    SliceDimension dimension) {

    // SECURITY: Validate model_id format
    if (!model_id.empty() && !IsValidModelId(model_id)) {
        return absl::InvalidArgumentError("Invalid model_id format");
    }

    std::string column = DimensionToColumn(dimension);

    // SECURITY: Use parameterized query
    std::ostringstream query;
    std::vector<storage::QueryParam> params;

    query << "SELECT DISTINCT " << column << " FROM traces WHERE 1=1";

    if (!model_id.empty()) {
        query << " AND model_id = {model_id:String}";
        params.push_back({"model_id", model_id, "String"});
    }

    query << " AND timestamp >= now() - INTERVAL {window_hours:Int64} HOUR"
          << " LIMIT 100";
    params.push_back({"window_hours", std::to_string(config_.analysis_window.count()), "Int64"});

    auto result = clickhouse_->ExecuteWithParams(query.str(), params);
    if (!result.ok()) {
        return result.status();
    }

    // Parse results (placeholder)
    std::vector<std::string> values;
    // In production, parse actual query results

    return values;
}

absl::StatusOr<std::vector<double>> SliceAnalyzer::GetSuggestedBins(
    const std::string& model_id,
    SliceDimension dimension,
    size_t num_bins) {

    // SECURITY: Validate model_id format
    if (!model_id.empty() && !IsValidModelId(model_id)) {
        return absl::InvalidArgumentError("Invalid model_id format");
    }

    // SECURITY: Limit num_bins to prevent DoS
    if (num_bins > 100) {
        num_bins = 100;
    }

    std::string column = DimensionToColumn(dimension);

    // SECURITY: Use parameterized query
    std::ostringstream query;
    std::vector<storage::QueryParam> params;

    query << "SELECT quantiles(";
    for (size_t i = 0; i <= num_bins; ++i) {
        if (i > 0) query << ", ";
        query << static_cast<double>(i) / num_bins;
    }
    query << ")(" << column << ") FROM traces WHERE 1=1";

    if (!model_id.empty()) {
        query << " AND model_id = {model_id:String}";
        params.push_back({"model_id", model_id, "String"});
    }

    auto result = clickhouse_->ExecuteWithParams(query.str(), params);
    if (!result.ok()) {
        return result.status();
    }

    // Parse results (placeholder)
    std::vector<double> bins;
    for (size_t i = 0; i <= num_bins; ++i) {
        bins.push_back(static_cast<double>(i) * 100);  // Placeholder
    }

    return bins;
}

void SliceAnalyzer::AddCustomSlice(const SliceDefinition& definition) {
    config_.custom_slices.push_back(definition);
}

// =============================================================================
// Static Methods
// =============================================================================

std::string SliceAnalyzer::DimensionToString(SliceDimension dimension) {
    switch (dimension) {
        case SliceDimension::kModel: return "model";
        case SliceDimension::kUser: return "user";
        case SliceDimension::kFeature: return "feature";
        case SliceDimension::kInputLength: return "input_length";
        case SliceDimension::kOutputLength: return "output_length";
        case SliceDimension::kLatency: return "latency";
        case SliceDimension::kTimeOfDay: return "time_of_day";
        case SliceDimension::kDayOfWeek: return "day_of_week";
        case SliceDimension::kPromptTemplate: return "prompt_template";
        case SliceDimension::kProvider: return "provider";
        case SliceDimension::kEnvironment: return "environment";
        case SliceDimension::kCustom: return "custom";
    }
    return "unknown";
}

SliceDimension SliceAnalyzer::StringToDimension(const std::string& str) {
    if (str == "model") return SliceDimension::kModel;
    if (str == "user") return SliceDimension::kUser;
    if (str == "feature") return SliceDimension::kFeature;
    if (str == "input_length") return SliceDimension::kInputLength;
    if (str == "output_length") return SliceDimension::kOutputLength;
    if (str == "latency") return SliceDimension::kLatency;
    if (str == "time_of_day") return SliceDimension::kTimeOfDay;
    if (str == "day_of_week") return SliceDimension::kDayOfWeek;
    if (str == "prompt_template") return SliceDimension::kPromptTemplate;
    if (str == "provider") return SliceDimension::kProvider;
    if (str == "environment") return SliceDimension::kEnvironment;
    return SliceDimension::kCustom;
}

std::string SliceAnalyzer::MetricToString(SliceMetric metric) {
    switch (metric) {
        case SliceMetric::kErrorRate: return "error_rate";
        case SliceMetric::kLatencyP50: return "latency_p50";
        case SliceMetric::kLatencyP95: return "latency_p95";
        case SliceMetric::kLatencyP99: return "latency_p99";
        case SliceMetric::kCost: return "cost";
        case SliceMetric::kTokenUsage: return "token_usage";
        case SliceMetric::kToxicityRate: return "toxicity_rate";
        case SliceMetric::kHallucinationRate: return "hallucination_rate";
        case SliceMetric::kDriftScore: return "drift_score";
        case SliceMetric::kCustom: return "custom";
    }
    return "unknown";
}

SliceMetric SliceAnalyzer::StringToMetric(const std::string& str) {
    if (str == "error_rate") return SliceMetric::kErrorRate;
    if (str == "latency_p50") return SliceMetric::kLatencyP50;
    if (str == "latency_p95") return SliceMetric::kLatencyP95;
    if (str == "latency_p99") return SliceMetric::kLatencyP99;
    if (str == "cost") return SliceMetric::kCost;
    if (str == "token_usage") return SliceMetric::kTokenUsage;
    if (str == "toxicity_rate") return SliceMetric::kToxicityRate;
    if (str == "hallucination_rate") return SliceMetric::kHallucinationRate;
    if (str == "drift_score") return SliceMetric::kDriftScore;
    return SliceMetric::kCustom;
}

std::string SliceAnalyzer::DimensionToColumn(SliceDimension dimension) {
    switch (dimension) {
        case SliceDimension::kModel: return "model_id";
        case SliceDimension::kUser: return "user_id";
        case SliceDimension::kFeature: return "feature_id";
        case SliceDimension::kInputLength: return "input_tokens";
        case SliceDimension::kOutputLength: return "output_tokens";
        case SliceDimension::kLatency: return "latency_ms";
        case SliceDimension::kTimeOfDay: return "toHour(timestamp)";
        case SliceDimension::kDayOfWeek: return "toDayOfWeek(timestamp)";
        case SliceDimension::kPromptTemplate: return "prompt_template";
        case SliceDimension::kProvider: return "provider";
        case SliceDimension::kEnvironment: return "environment";
        default: return "model_id";
    }
}

std::string SliceAnalyzer::MetricToAggregation(SliceMetric metric) {
    switch (metric) {
        case SliceMetric::kErrorRate:
            return "avg(toUInt8(is_error))";
        case SliceMetric::kLatencyP50:
            return "quantile(0.5)(latency_ms)";
        case SliceMetric::kLatencyP95:
            return "quantile(0.95)(latency_ms)";
        case SliceMetric::kLatencyP99:
            return "quantile(0.99)(latency_ms)";
        case SliceMetric::kCost:
            return "sum(cost_micros)";
        case SliceMetric::kTokenUsage:
            return "avg(total_tokens)";
        case SliceMetric::kToxicityRate:
            return "avg(toUInt8(toxicity_score > 0.5))";
        case SliceMetric::kHallucinationRate:
            return "avg(toUInt8(hallucination_score > 0.5))";
        case SliceMetric::kDriftScore:
            return "avg(drift_score)";
        default:
            return "count(*)";
    }
}

// =============================================================================
// Private Methods
// =============================================================================

/// @brief Build a slice query with parameterized execution
/// @note This is now a private helper that returns query and params
std::pair<std::string, std::vector<storage::QueryParam>>
SliceAnalyzer::BuildSliceQueryWithParams(const std::string& model_id,
                                          SliceDimension dimension,
                                          SliceMetric metric) {
    std::ostringstream query;
    std::vector<storage::QueryParam> params;

    std::string dim_col = DimensionToColumn(dimension);
    std::string agg = MetricToAggregation(metric);

    query << "SELECT "
          << dim_col << " as slice_value, "
          << agg << " as metric_value, "
          << "count(*) as sample_count "
          << "FROM traces "
          << "WHERE timestamp >= now() - INTERVAL {window_hours:Int64} HOUR";
    params.push_back({"window_hours", std::to_string(config_.analysis_window.count()), "Int64"});

    if (!model_id.empty()) {
        query << " AND model_id = {model_id:String}";
        params.push_back({"model_id", model_id, "String"});
    }

    query << " GROUP BY " << dim_col
          << " HAVING sample_count >= {min_samples:UInt64}"
          << " ORDER BY metric_value DESC"
          << " LIMIT {max_slices:UInt64}";
    params.push_back({"min_samples", std::to_string(config_.min_samples), "UInt64"});
    params.push_back({"max_slices", std::to_string(config_.max_slices), "UInt64"});

    return {query.str(), params};
}

// DEPRECATED: Use BuildSliceQueryWithParams instead
std::string SliceAnalyzer::BuildSliceQuery(const std::string& model_id,
                                            SliceDimension dimension,
                                            SliceMetric metric) {
    // This method is kept for backward compatibility but should not be used
    // Use BuildSliceQueryWithParams for secure parameterized queries
    auto [query, params] = BuildSliceQueryWithParams(model_id, dimension, metric);
    // WARNING: This loses parameterization - only use for logging/debugging
    return query;
}

void SliceAnalyzer::CalculateSignificance(SliceAnalysisResult& result,
                                           size_t total_samples) {
    // Calculate deviation
    if (result.baseline_value != 0) {
        result.deviation = result.metric_value - result.baseline_value;
        result.deviation_percentage =
            (result.deviation / result.baseline_value) * 100.0;
    }

    // Simplified significance test
    // In production, use proper statistical tests (t-test, chi-squared, etc.)
    double effect_size = std::abs(result.deviation_percentage) / 100.0;
    double sample_ratio = static_cast<double>(result.sample_count) / total_samples;

    // Rough p-value estimate (placeholder)
    result.p_value = std::exp(-effect_size * sample_ratio * 10);
    result.is_statistically_significant = result.p_value < config_.p_value_threshold;

    // Confidence interval (placeholder)
    double margin = 1.96 * result.baseline_value / std::sqrt(result.sample_count);
    result.confidence_interval_lower = result.metric_value - margin;
    result.confidence_interval_upper = result.metric_value + margin;
}

void SliceAnalyzer::CalculateImpact(SliceAnalysisResult& result) {
    // Impact = deviation * log(sample_count)
    // This weights larger slices more heavily
    double deviation_factor = std::abs(result.deviation_percentage) / 100.0;
    double size_factor = std::log(result.sample_count + 1);

    result.impact_score = deviation_factor * size_factor;

    // Build comparison description
    std::ostringstream desc;
    if (result.deviation_percentage > 0) {
        desc << std::fixed << std::setprecision(1)
             << result.deviation_percentage << "% higher";
    } else {
        desc << std::fixed << std::setprecision(1)
             << std::abs(result.deviation_percentage) << "% lower";
    }
    desc << " than baseline (" << result.sample_count << " samples)";
    result.comparison_description = desc.str();
}

// =============================================================================
// Factory Function
// =============================================================================

std::vector<SliceDefinition> CreateDefaultSliceDefinitions() {
    std::vector<SliceDefinition> definitions;

    // Model dimension
    SliceDefinition model_slice;
    model_slice.name = "By Model";
    model_slice.dimension = SliceDimension::kModel;
    definitions.push_back(model_slice);

    // User dimension
    SliceDefinition user_slice;
    user_slice.name = "By User";
    user_slice.dimension = SliceDimension::kUser;
    definitions.push_back(user_slice);

    // Input length buckets
    SliceDefinition short_input;
    short_input.name = "Short Inputs (<100 tokens)";
    short_input.dimension = SliceDimension::kInputLength;
    short_input.range_max = 100;
    definitions.push_back(short_input);

    SliceDefinition medium_input;
    medium_input.name = "Medium Inputs (100-1000 tokens)";
    medium_input.dimension = SliceDimension::kInputLength;
    medium_input.range_min = 100;
    medium_input.range_max = 1000;
    definitions.push_back(medium_input);

    SliceDefinition long_input;
    long_input.name = "Long Inputs (>1000 tokens)";
    long_input.dimension = SliceDimension::kInputLength;
    long_input.range_min = 1000;
    definitions.push_back(long_input);

    return definitions;
}

}  // namespace pyflare::rca
