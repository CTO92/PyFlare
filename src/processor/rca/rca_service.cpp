/// @file rca_service.cpp
/// @brief Root Cause Analysis orchestration service implementation

#include "processor/rca/rca_service.h"

#include <algorithm>
#include <iomanip>
#include <random>
#include <sstream>

#include <nlohmann/json.hpp>

namespace pyflare::rca {

using json = nlohmann::json;

RCAService::RCAService(
    std::shared_ptr<storage::ClickHouseClient> clickhouse,
    std::shared_ptr<storage::QdrantClient> qdrant,
    std::shared_ptr<storage::RedisClient> redis,
    RCAServiceConfig config)
    : clickhouse_(std::move(clickhouse)),
      qdrant_(std::move(qdrant)),
      redis_(std::move(redis)),
      config_(std::move(config)) {}

RCAService::~RCAService() = default;

absl::Status RCAService::Initialize() {
    if (initialized_) {
        return absl::OkStatus();
    }

    // Initialize slice analyzer
    slice_analyzer_ = std::make_unique<SliceAnalyzer>(
        clickhouse_, config_.slice_config);
    auto status = slice_analyzer_->Initialize();
    if (!status.ok()) {
        return status;
    }

    // Initialize temporal analyzer
    temporal_analyzer_ = std::make_unique<TemporalAnalyzer>(
        clickhouse_, config_.temporal_config);
    status = temporal_analyzer_->Initialize();
    if (!status.ok()) {
        return status;
    }

    // Initialize counterfactual generator
    counterfactual_generator_ = std::make_unique<CounterfactualGenerator>(
        qdrant_, config_.counterfactual_config);
    status = counterfactual_generator_->Initialize();
    if (!status.ok()) {
        return status;
    }

    initialized_ = true;
    return absl::OkStatus();
}

absl::StatusOr<RCAReport> RCAService::Analyze(const std::string& model_id) {
    RCAReport::Trigger trigger;
    trigger.type = "manual";
    trigger.description = "Manual RCA analysis triggered";
    trigger.severity = 0.5;

    return AnalyzeWithTrigger(model_id, trigger);
}

absl::StatusOr<RCAReport> RCAService::AnalyzeWithTrigger(
    const std::string& model_id,
    const RCAReport::Trigger& trigger) {

    auto now = std::chrono::system_clock::now();
    auto start = now - config_.default_analysis_window;

    return AnalyzeTimeRange(model_id, start, now);
}

absl::StatusOr<RCAReport> RCAService::AnalyzeTimeRange(
    const std::string& model_id,
    std::chrono::system_clock::time_point start,
    std::chrono::system_clock::time_point end) {

    if (!initialized_) {
        return absl::FailedPreconditionError("RCAService not initialized");
    }

    auto analysis_start = std::chrono::steady_clock::now();

    // Create report
    RCAReport report;
    report.report_id = GenerateReportId();
    report.model_id = model_id;
    report.generated_at = std::chrono::system_clock::now();
    report.analysis_start = start;
    report.analysis_end = end;
    report.status = RCAReport::Status::kInProgress;

    // Default trigger if not set
    report.trigger.type = "manual";
    report.trigger.description = "Analysis requested";
    report.trigger.severity = 0.5;

    // Phase 1: Slice Analysis
    auto slice_result = RunSliceAnalysis(model_id, start, end);
    if (!slice_result.ok()) {
        report.status = RCAReport::Status::kFailed;
        report.error_message = std::string(slice_result.status().message());
        return report;
    }
    report.problematic_slices = *slice_result;

    // Phase 2: Temporal Analysis
    auto temporal_result = RunTemporalAnalysis(model_id, start, end);
    if (!temporal_result.ok()) {
        report.status = RCAReport::Status::kFailed;
        report.error_message = std::string(temporal_result.status().message());
        return report;
    }
    report.temporal_correlations = *temporal_result;

    // Phase 3: Counterfactual Analysis
    auto counterfactual_result = RunCounterfactualAnalysis(
        model_id, report.problematic_slices);
    if (!counterfactual_result.ok()) {
        // Non-fatal - continue without counterfactuals
        report.counterfactuals = {};
    } else {
        report.counterfactuals = *counterfactual_result;
    }

    // Phase 4: Identify Causal Factors
    report.causal_factors = IdentifyCausalFactors(
        report.problematic_slices,
        report.temporal_correlations,
        report.counterfactuals);

    // Phase 5: Generate Root Causes
    report.root_causes = GenerateRootCauses(report.causal_factors);

    // Phase 6: Generate Recommendations
    report.recommendations = GenerateRecommendations(
        report.causal_factors, report.problematic_slices);

    // Phase 7: Generate Summary and Key Findings
    report.key_findings = GenerateKeyFindings(report);
    report.summary = GenerateSummary(report);

    // Calculate confidence
    double confidence = 0.5;  // Base confidence
    if (!report.causal_factors.empty()) {
        confidence += 0.2;
    }
    if (!report.temporal_correlations.empty()) {
        confidence += 0.15;
    }
    if (!report.counterfactuals.empty()) {
        confidence += 0.15;
    }
    report.confidence = std::min(confidence, 1.0);

    // Calculate analysis duration
    auto analysis_end = std::chrono::steady_clock::now();
    report.analysis_duration = std::chrono::duration_cast<std::chrono::milliseconds>(
        analysis_end - analysis_start);

    // Count samples analyzed
    for (const auto& slice : report.problematic_slices) {
        report.samples_analyzed += slice.sample_count;
    }

    // Mark as completed
    report.status = RCAReport::Status::kCompleted;

    // Persist report
    if (config_.persist_reports) {
        auto persist_status = PersistReport(report);
        if (!persist_status.ok()) {
            // Log but don't fail the analysis
        }
    }

    // Update statistics
    {
        std::lock_guard<std::mutex> lock(stats_mutex_);
        stats_.total_analyses++;
        stats_.successful_analyses++;
        stats_.last_analysis = std::chrono::system_clock::now();

        // Update running average
        double new_time = static_cast<double>(report.analysis_duration.count());
        stats_.avg_analysis_time_ms =
            (stats_.avg_analysis_time_ms * (stats_.total_analyses - 1) + new_time) /
            stats_.total_analyses;
    }

    // Notify callbacks
    {
        std::lock_guard<std::mutex> lock(callbacks_mutex_);
        for (const auto& callback : report_callbacks_) {
            callback(report);
        }
    }

    return report;
}

absl::StatusOr<RCAReport> RCAService::QuickAnalyze(const std::string& model_id) {
    if (!initialized_) {
        return absl::FailedPreconditionError("RCAService not initialized");
    }

    auto now = std::chrono::system_clock::now();
    auto start = now - std::chrono::hours(6);  // Shorter window for quick analysis

    RCAReport report;
    report.report_id = GenerateReportId();
    report.model_id = model_id;
    report.generated_at = now;
    report.analysis_start = start;
    report.analysis_end = now;
    report.status = RCAReport::Status::kInProgress;

    // Only run slice analysis for quick mode
    auto slice_result = RunSliceAnalysis(model_id, start, now);
    if (!slice_result.ok()) {
        report.status = RCAReport::Status::kFailed;
        report.error_message = std::string(slice_result.status().message());
        return report;
    }
    report.problematic_slices = *slice_result;

    // Generate quick summary
    if (!report.problematic_slices.empty()) {
        std::ostringstream summary;
        summary << "Quick analysis found " << report.problematic_slices.size()
                << " problematic data slices. ";

        const auto& top_slice = report.problematic_slices[0];
        summary << "Top issue: " << top_slice.slice_definition
                << " with severity " << std::fixed << std::setprecision(2)
                << top_slice.severity_score;

        report.summary = summary.str();
        report.root_causes.push_back(
            "Performance degradation detected in specific data segments");
    } else {
        report.summary = "Quick analysis found no significant issues.";
    }

    report.confidence = 0.5;  // Lower confidence for quick analysis
    report.status = RCAReport::Status::kCompleted;

    return report;
}

absl::StatusOr<std::vector<SliceAnalysisResult>> RCAService::AnalyzeSlices(
    const std::string& model_id) {

    if (!initialized_) {
        return absl::FailedPreconditionError("RCAService not initialized");
    }

    auto now = std::chrono::system_clock::now();
    auto start = now - config_.default_analysis_window;

    return RunSliceAnalysis(model_id, start, now);
}

absl::StatusOr<std::vector<TemporalCorrelation>> RCAService::AnalyzeTemporalCorrelations(
    const std::string& model_id) {

    if (!initialized_) {
        return absl::FailedPreconditionError("RCAService not initialized");
    }

    auto now = std::chrono::system_clock::now();
    auto start = now - config_.default_analysis_window;

    return RunTemporalAnalysis(model_id, start, now);
}

absl::StatusOr<std::vector<Counterfactual>> RCAService::GenerateCounterfactuals(
    const eval::InferenceRecord& record,
    const std::string& target_outcome) {

    if (!initialized_) {
        return absl::FailedPreconditionError("RCAService not initialized");
    }

    return counterfactual_generator_->Generate(record, target_outcome);
}

absl::StatusOr<RCAReport> RCAService::GetReport(const std::string& report_id) {
    // Try cache first
    if (config_.cache_reports) {
        auto cached = LoadCachedReport(report_id);
        if (cached.ok() && cached->has_value()) {
            return cached->value();
        }
    }

    // Query from ClickHouse
    if (!clickhouse_) {
        return absl::NotFoundError("Report not found: " + report_id);
    }

    std::string query = "SELECT report_json FROM rca_reports WHERE report_id = '" +
                        report_id + "' LIMIT 1";

    auto result = clickhouse_->Query(query);
    if (!result.ok()) {
        return result.status();
    }

    if (result->empty()) {
        return absl::NotFoundError("Report not found: " + report_id);
    }

    // Parse JSON
    return DeserializeReport((*result)[0]["report_json"]);
}

absl::StatusOr<std::vector<RCAReport>> RCAService::ListReports(
    const std::string& model_id,
    size_t limit) {

    if (!clickhouse_) {
        return absl::FailedPreconditionError("ClickHouse client not available");
    }

    std::string query = "SELECT report_json FROM rca_reports WHERE model_id = '" +
                        model_id + "' ORDER BY generated_at DESC LIMIT " +
                        std::to_string(limit);

    auto result = clickhouse_->Query(query);
    if (!result.ok()) {
        return result.status();
    }

    std::vector<RCAReport> reports;
    for (const auto& row : *result) {
        auto report = DeserializeReport(row.at("report_json"));
        if (report.ok()) {
            reports.push_back(*report);
        }
    }

    return reports;
}

absl::Status RCAService::DeleteReport(const std::string& report_id) {
    if (!clickhouse_) {
        return absl::FailedPreconditionError("ClickHouse client not available");
    }

    std::string query = "ALTER TABLE rca_reports DELETE WHERE report_id = '" +
                        report_id + "'";

    return clickhouse_->Execute(query);
}

void RCAService::OnReportComplete(std::function<void(const RCAReport&)> callback) {
    std::lock_guard<std::mutex> lock(callbacks_mutex_);
    report_callbacks_.push_back(std::move(callback));
}

void RCAService::ClearCallbacks() {
    std::lock_guard<std::mutex> lock(callbacks_mutex_);
    report_callbacks_.clear();
}

void RCAService::SetConfig(RCAServiceConfig config) {
    config_ = std::move(config);

    // Update component configs
    if (slice_analyzer_) {
        slice_analyzer_->SetConfig(config_.slice_config);
    }
    if (temporal_analyzer_) {
        temporal_analyzer_->SetConfig(config_.temporal_config);
    }
    if (counterfactual_generator_) {
        counterfactual_generator_->SetConfig(config_.counterfactual_config);
    }
}

RCAService::Stats RCAService::GetStats() const {
    std::lock_guard<std::mutex> lock(stats_mutex_);
    return stats_;
}

void RCAService::ResetStats() {
    std::lock_guard<std::mutex> lock(stats_mutex_);
    stats_ = Stats{};
}

// Private methods

absl::StatusOr<std::vector<SliceAnalysisResult>> RCAService::RunSliceAnalysis(
    const std::string& model_id,
    std::chrono::system_clock::time_point start,
    std::chrono::system_clock::time_point end) {

    if (!slice_analyzer_) {
        return absl::FailedPreconditionError("Slice analyzer not initialized");
    }

    auto results = slice_analyzer_->Analyze(model_id, start, end);
    if (!results.ok()) {
        return results.status();
    }

    // Sort by severity and limit
    auto sorted = *results;
    std::sort(sorted.begin(), sorted.end(),
              [](const SliceAnalysisResult& a, const SliceAnalysisResult& b) {
                  return a.severity_score > b.severity_score;
              });

    if (sorted.size() > config_.max_slices) {
        sorted.resize(config_.max_slices);
    }

    return sorted;
}

absl::StatusOr<std::vector<TemporalCorrelation>> RCAService::RunTemporalAnalysis(
    const std::string& model_id,
    std::chrono::system_clock::time_point start,
    std::chrono::system_clock::time_point end) {

    if (!temporal_analyzer_) {
        return absl::FailedPreconditionError("Temporal analyzer not initialized");
    }

    auto results = temporal_analyzer_->Analyze(model_id, start, end);
    if (!results.ok()) {
        return results.status();
    }

    // Sort by correlation strength and limit
    auto sorted = *results;
    std::sort(sorted.begin(), sorted.end(),
              [](const TemporalCorrelation& a, const TemporalCorrelation& b) {
                  return std::abs(a.correlation_coefficient) >
                         std::abs(b.correlation_coefficient);
              });

    if (sorted.size() > config_.max_correlations) {
        sorted.resize(config_.max_correlations);
    }

    return sorted;
}

absl::StatusOr<std::vector<Counterfactual>> RCAService::RunCounterfactualAnalysis(
    const std::string& model_id,
    const std::vector<SliceAnalysisResult>& problematic_slices) {

    if (!counterfactual_generator_) {
        return absl::FailedPreconditionError("Counterfactual generator not initialized");
    }

    // Get representative problematic inferences
    auto inferences = QueryProblematicInferences(
        model_id, problematic_slices, config_.max_counterfactuals);

    if (!inferences.ok()) {
        return inferences.status();
    }

    std::vector<Counterfactual> all_counterfactuals;

    for (const auto& inference : *inferences) {
        auto counterfactuals = counterfactual_generator_->Generate(
            inference, "successful_outcome");

        if (counterfactuals.ok() && !counterfactuals->empty()) {
            all_counterfactuals.insert(
                all_counterfactuals.end(),
                counterfactuals->begin(),
                counterfactuals->end());

            if (all_counterfactuals.size() >= config_.max_counterfactuals) {
                break;
            }
        }
    }

    return all_counterfactuals;
}

std::vector<RCAReport::CausalFactor> RCAService::IdentifyCausalFactors(
    const std::vector<SliceAnalysisResult>& slices,
    const std::vector<TemporalCorrelation>& correlations,
    const std::vector<Counterfactual>& counterfactuals) {

    std::vector<RCAReport::CausalFactor> factors;

    // Extract factors from slice analysis
    for (const auto& slice : slices) {
        if (slice.severity_score > 0.5) {
            RCAReport::CausalFactor factor;
            factor.factor = "Data segment: " + slice.slice_definition;
            factor.contribution = slice.severity_score;
            factor.evidence = "Performance degradation: " +
                              std::to_string(static_cast<int>(
                                  slice.baseline_performance - slice.current_performance * 100)) +
                              "% drop";
            factor.category = "data";
            factors.push_back(factor);
        }
    }

    // Extract factors from temporal correlations
    for (const auto& corr : correlations) {
        if (std::abs(corr.correlation_coefficient) > 0.6) {
            RCAReport::CausalFactor factor;
            factor.factor = corr.event_a_type + " correlates with " + corr.event_b_type;
            factor.contribution = std::abs(corr.correlation_coefficient);
            factor.evidence = "Correlation: " +
                              std::to_string(corr.correlation_coefficient) +
                              " (lag: " + std::to_string(corr.lag_seconds.count()) + "s)";

            // Categorize based on event types
            if (corr.event_a_type.find("deployment") != std::string::npos ||
                corr.event_b_type.find("deployment") != std::string::npos) {
                factor.category = "infrastructure";
            } else if (corr.event_a_type.find("data") != std::string::npos) {
                factor.category = "data";
            } else {
                factor.category = "model";
            }

            factors.push_back(factor);
        }
    }

    // Extract factors from counterfactuals
    for (const auto& cf : counterfactuals) {
        if (cf.plausibility_score > 0.7) {
            for (const auto& change : cf.changes) {
                RCAReport::CausalFactor factor;
                factor.factor = "Input modification: " + change.attribute;
                factor.contribution = change.importance;
                factor.evidence = "Counterfactual: " + change.original_value +
                                  " -> " + change.new_value;
                factor.category = "data";
                factors.push_back(factor);
            }
        }
    }

    // Sort by contribution
    std::sort(factors.begin(), factors.end(),
              [](const RCAReport::CausalFactor& a, const RCAReport::CausalFactor& b) {
                  return a.contribution > b.contribution;
              });

    return factors;
}

std::vector<std::string> RCAService::GenerateRootCauses(
    const std::vector<RCAReport::CausalFactor>& factors) {

    std::vector<std::string> causes;
    std::unordered_map<std::string, std::vector<const RCAReport::CausalFactor*>> by_category;

    // Group by category
    for (const auto& factor : factors) {
        by_category[factor.category].push_back(&factor);
    }

    // Generate cause statements by category
    if (auto it = by_category.find("data"); it != by_category.end() && !it->second.empty()) {
        std::ostringstream cause;
        cause << "Data quality issues detected in " << it->second.size() << " areas";
        if (!it->second.empty()) {
            cause << ": " << it->second[0]->factor;
        }
        causes.push_back(cause.str());
    }

    if (auto it = by_category.find("model"); it != by_category.end() && !it->second.empty()) {
        std::ostringstream cause;
        cause << "Model behavior anomalies: " << it->second[0]->factor;
        causes.push_back(cause.str());
    }

    if (auto it = by_category.find("infrastructure"); it != by_category.end() && !it->second.empty()) {
        std::ostringstream cause;
        cause << "Infrastructure changes correlated with issues: " << it->second[0]->factor;
        causes.push_back(cause.str());
    }

    if (auto it = by_category.find("external"); it != by_category.end() && !it->second.empty()) {
        std::ostringstream cause;
        cause << "External factors identified: " << it->second[0]->factor;
        causes.push_back(cause.str());
    }

    // Add high-contribution individual causes
    for (const auto& factor : factors) {
        if (factor.contribution > 0.8 && causes.size() < 5) {
            causes.push_back("High-impact factor: " + factor.factor);
        }
    }

    return causes;
}

std::vector<std::string> RCAService::GenerateRecommendations(
    const std::vector<RCAReport::CausalFactor>& factors,
    const std::vector<SliceAnalysisResult>& slices) {

    std::vector<std::string> recommendations;

    // Category-based recommendations
    std::unordered_map<std::string, bool> has_category;
    for (const auto& factor : factors) {
        has_category[factor.category] = true;
    }

    if (has_category["data"]) {
        recommendations.push_back(
            "Review and validate input data quality for affected segments");
        recommendations.push_back(
            "Implement data validation rules for identified problematic patterns");
    }

    if (has_category["model"]) {
        recommendations.push_back(
            "Consider retraining or fine-tuning the model on recent data");
        recommendations.push_back(
            "Review model performance metrics across different input types");
    }

    if (has_category["infrastructure"]) {
        recommendations.push_back(
            "Investigate recent infrastructure changes that coincide with issues");
        recommendations.push_back(
            "Implement rollback procedures for configuration changes");
    }

    // Slice-specific recommendations
    for (const auto& slice : slices) {
        if (slice.severity_score > 0.8) {
            recommendations.push_back(
                "Urgent: Address high-severity issues in segment: " +
                slice.slice_definition);
        }
    }

    // General recommendations
    if (recommendations.empty()) {
        recommendations.push_back(
            "Continue monitoring system performance");
        recommendations.push_back(
            "Set up alerts for similar patterns in the future");
    }

    return recommendations;
}

std::string RCAService::GenerateSummary(const RCAReport& report) {
    std::ostringstream summary;

    summary << "Root Cause Analysis for model '" << report.model_id << "' ";
    summary << "analyzed " << report.samples_analyzed << " samples ";
    summary << "over " << std::chrono::duration_cast<std::chrono::hours>(
                              report.analysis_end - report.analysis_start).count()
            << " hours. ";

    if (!report.root_causes.empty()) {
        summary << "Primary root cause: " << report.root_causes[0] << ". ";
    }

    if (!report.problematic_slices.empty()) {
        summary << "Identified " << report.problematic_slices.size()
                << " problematic data segments. ";
    }

    if (!report.temporal_correlations.empty()) {
        summary << "Found " << report.temporal_correlations.size()
                << " temporal correlations. ";
    }

    summary << "Analysis confidence: " << std::fixed << std::setprecision(0)
            << (report.confidence * 100) << "%.";

    return summary.str();
}

std::vector<std::string> RCAService::GenerateKeyFindings(const RCAReport& report) {
    std::vector<std::string> findings;

    // Top problematic slice
    if (!report.problematic_slices.empty()) {
        const auto& top = report.problematic_slices[0];
        std::ostringstream finding;
        finding << "Most affected segment: " << top.slice_definition
                << " (severity: " << std::fixed << std::setprecision(2)
                << top.severity_score << ")";
        findings.push_back(finding.str());
    }

    // Strongest correlation
    if (!report.temporal_correlations.empty()) {
        const auto& top = report.temporal_correlations[0];
        std::ostringstream finding;
        finding << "Strongest correlation: " << top.event_a_type << " <-> "
                << top.event_b_type << " (r=" << std::fixed << std::setprecision(2)
                << top.correlation_coefficient << ")";
        findings.push_back(finding.str());
    }

    // Top causal factor
    if (!report.causal_factors.empty()) {
        const auto& top = report.causal_factors[0];
        findings.push_back("Top causal factor: " + top.factor +
                           " (" + top.category + ")");
    }

    // Counterfactual insight
    if (!report.counterfactuals.empty()) {
        findings.push_back("Counterfactual analysis suggests input modifications " +
                           std::string("could improve outcomes"));
    }

    return findings;
}

std::string RCAService::GenerateReportId() {
    static std::random_device rd;
    static std::mt19937 gen(rd());
    static std::uniform_int_distribution<uint64_t> dis;

    std::ostringstream id;
    id << "rca-" << std::hex << dis(gen);
    return id.str();
}

absl::Status RCAService::PersistReport(const RCAReport& report) {
    if (!clickhouse_) {
        return absl::FailedPreconditionError("ClickHouse client not available");
    }

    std::string report_json = SerializeReport(report);

    auto generated_at_sec = std::chrono::duration_cast<std::chrono::seconds>(
        report.generated_at.time_since_epoch()).count();

    std::ostringstream query;
    query << "INSERT INTO rca_reports (report_id, model_id, generated_at, "
          << "status, confidence, report_json) VALUES ('"
          << report.report_id << "', '"
          << report.model_id << "', "
          << "toDateTime(" << generated_at_sec << "), '"
          << ReportStatusToString(report.status) << "', "
          << report.confidence << ", '"
          << report_json << "')";

    return clickhouse_->Execute(query.str());
}

absl::StatusOr<std::optional<RCAReport>> RCAService::LoadCachedReport(
    const std::string& report_id) {

    if (!redis_) {
        return std::nullopt;
    }

    std::string key = "rca:report:" + report_id;
    auto result = redis_->Get(key);

    if (!result.ok() || !result->has_value()) {
        return std::nullopt;
    }

    auto report = DeserializeReport(**result);
    if (!report.ok()) {
        return std::nullopt;
    }

    return *report;
}

absl::StatusOr<std::vector<eval::InferenceRecord>> RCAService::QueryProblematicInferences(
    const std::string& model_id,
    const std::vector<SliceAnalysisResult>& slices,
    size_t limit) {

    if (!clickhouse_) {
        return absl::FailedPreconditionError("ClickHouse client not available");
    }

    // Build query for problematic inferences
    std::ostringstream query;
    query << "SELECT * FROM traces WHERE model_id = '" << model_id << "' "
          << "AND eval_passed = 0 "
          << "ORDER BY timestamp DESC "
          << "LIMIT " << limit;

    auto result = clickhouse_->Query(query.str());
    if (!result.ok()) {
        return result.status();
    }

    // Convert to InferenceRecord (simplified)
    std::vector<eval::InferenceRecord> records;
    for (const auto& row : *result) {
        eval::InferenceRecord record;
        record.trace_id = row.at("trace_id");
        if (row.count("input")) record.input = row.at("input");
        if (row.count("output")) record.output = row.at("output");
        records.push_back(record);
    }

    return records;
}

// Factory function
std::unique_ptr<RCAService> CreateRCAService(
    std::shared_ptr<storage::ClickHouseClient> clickhouse,
    std::shared_ptr<storage::QdrantClient> qdrant,
    std::shared_ptr<storage::RedisClient> redis,
    RCAServiceConfig config) {

    return std::make_unique<RCAService>(
        std::move(clickhouse),
        std::move(qdrant),
        std::move(redis),
        std::move(config));
}

// Utility functions
std::string ReportStatusToString(RCAReport::Status status) {
    switch (status) {
        case RCAReport::Status::kPending: return "pending";
        case RCAReport::Status::kInProgress: return "in_progress";
        case RCAReport::Status::kCompleted: return "completed";
        case RCAReport::Status::kFailed: return "failed";
        default: return "unknown";
    }
}

std::string SerializeReport(const RCAReport& report) {
    json j;

    j["report_id"] = report.report_id;
    j["model_id"] = report.model_id;
    j["generated_at"] = std::chrono::duration_cast<std::chrono::seconds>(
        report.generated_at.time_since_epoch()).count();

    j["trigger"] = {
        {"type", report.trigger.type},
        {"event_id", report.trigger.event_id},
        {"description", report.trigger.description},
        {"severity", report.trigger.severity}
    };

    j["analysis_start"] = std::chrono::duration_cast<std::chrono::seconds>(
        report.analysis_start.time_since_epoch()).count();
    j["analysis_end"] = std::chrono::duration_cast<std::chrono::seconds>(
        report.analysis_end.time_since_epoch()).count();

    j["summary"] = report.summary;
    j["key_findings"] = report.key_findings;
    j["root_causes"] = report.root_causes;
    j["recommendations"] = report.recommendations;

    // Causal factors
    json factors_json = json::array();
    for (const auto& factor : report.causal_factors) {
        factors_json.push_back({
            {"factor", factor.factor},
            {"contribution", factor.contribution},
            {"evidence", factor.evidence},
            {"category", factor.category}
        });
    }
    j["causal_factors"] = factors_json;

    j["confidence"] = report.confidence;
    j["samples_analyzed"] = report.samples_analyzed;
    j["analysis_duration_ms"] = report.analysis_duration.count();
    j["status"] = ReportStatusToString(report.status);
    j["error_message"] = report.error_message;

    return j.dump();
}

absl::StatusOr<RCAReport> DeserializeReport(const std::string& json_str) {
    try {
        json j = json::parse(json_str);

        RCAReport report;
        report.report_id = j.value("report_id", "");
        report.model_id = j.value("model_id", "");
        report.generated_at = std::chrono::system_clock::time_point(
            std::chrono::seconds(j.value("generated_at", 0)));

        if (j.contains("trigger")) {
            report.trigger.type = j["trigger"].value("type", "");
            report.trigger.event_id = j["trigger"].value("event_id", "");
            report.trigger.description = j["trigger"].value("description", "");
            report.trigger.severity = j["trigger"].value("severity", 0.0);
        }

        report.analysis_start = std::chrono::system_clock::time_point(
            std::chrono::seconds(j.value("analysis_start", 0)));
        report.analysis_end = std::chrono::system_clock::time_point(
            std::chrono::seconds(j.value("analysis_end", 0)));

        report.summary = j.value("summary", "");
        report.key_findings = j.value("key_findings", std::vector<std::string>{});
        report.root_causes = j.value("root_causes", std::vector<std::string>{});
        report.recommendations = j.value("recommendations", std::vector<std::string>{});

        if (j.contains("causal_factors")) {
            for (const auto& f : j["causal_factors"]) {
                RCAReport::CausalFactor factor;
                factor.factor = f.value("factor", "");
                factor.contribution = f.value("contribution", 0.0);
                factor.evidence = f.value("evidence", "");
                factor.category = f.value("category", "");
                report.causal_factors.push_back(factor);
            }
        }

        report.confidence = j.value("confidence", 0.0);
        report.samples_analyzed = j.value("samples_analyzed", 0);
        report.analysis_duration = std::chrono::milliseconds(
            j.value("analysis_duration_ms", 0));

        std::string status_str = j.value("status", "pending");
        if (status_str == "in_progress") {
            report.status = RCAReport::Status::kInProgress;
        } else if (status_str == "completed") {
            report.status = RCAReport::Status::kCompleted;
        } else if (status_str == "failed") {
            report.status = RCAReport::Status::kFailed;
        } else {
            report.status = RCAReport::Status::kPending;
        }

        report.error_message = j.value("error_message", "");

        return report;
    } catch (const json::exception& e) {
        return absl::InvalidArgumentError(
            std::string("Failed to parse RCA report JSON: ") + e.what());
    }
}

}  // namespace pyflare::rca
