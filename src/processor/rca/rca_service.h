#pragma once

/// @file rca_service.h
/// @brief Root Cause Analysis orchestration service
///
/// Coordinates all RCA components to generate comprehensive
/// root cause analysis reports for model issues.

#include <chrono>
#include <functional>
#include <memory>
#include <mutex>
#include <string>
#include <unordered_map>
#include <vector>

#include <absl/status/status.h>
#include <absl/status/statusor.h>

#include "processor/rca/analyzer.h"
#include "processor/rca/counterfactual.h"
#include "processor/rca/slice_analyzer.h"
#include "processor/rca/temporal_analyzer.h"
#include "storage/clickhouse/client.h"
#include "storage/qdrant/client.h"
#include "storage/redis/client.h"

namespace pyflare::rca {

/// @brief RCA report structure
struct RCAReport {
    std::string report_id;
    std::string model_id;
    std::chrono::system_clock::time_point generated_at;

    /// Trigger information
    struct Trigger {
        std::string type;  // "drift", "error_spike", "manual", "scheduled"
        std::string event_id;
        std::string description;
        double severity = 0.0;
    };
    Trigger trigger;

    /// Time window analyzed
    std::chrono::system_clock::time_point analysis_start;
    std::chrono::system_clock::time_point analysis_end;

    /// Summary
    std::string summary;
    std::vector<std::string> key_findings;
    std::vector<std::string> root_causes;
    std::vector<std::string> recommendations;

    /// Detailed analysis results
    std::vector<SliceAnalysisResult> problematic_slices;
    std::vector<TemporalCorrelation> temporal_correlations;
    std::vector<Counterfactual> counterfactuals;

    /// Causal factors
    struct CausalFactor {
        std::string factor;
        double contribution = 0.0;  ///< 0.0 - 1.0
        std::string evidence;
        std::string category;  // "data", "model", "infrastructure", "external"
    };
    std::vector<CausalFactor> causal_factors;

    /// Confidence in the analysis
    double confidence = 0.0;

    /// Metrics
    size_t samples_analyzed = 0;
    std::chrono::milliseconds analysis_duration{0};

    /// Status
    enum class Status {
        kPending,
        kInProgress,
        kCompleted,
        kFailed
    };
    Status status = Status::kPending;
    std::string error_message;
};

/// @brief Configuration for RCA service
struct RCAServiceConfig {
    /// Analysis time window
    std::chrono::hours default_analysis_window = std::chrono::hours(24);

    /// Maximum slices to analyze
    size_t max_slices = 50;

    /// Maximum counterfactuals to generate
    size_t max_counterfactuals = 5;

    /// Maximum temporal correlations
    size_t max_correlations = 20;

    /// Minimum severity to trigger automatic RCA
    double auto_trigger_severity = 0.7;

    /// Enable automatic RCA on drift detection
    bool auto_rca_on_drift = true;

    /// Enable automatic RCA on error spikes
    bool auto_rca_on_errors = true;

    /// Store reports to ClickHouse
    bool persist_reports = true;

    /// Cache reports in Redis
    bool cache_reports = true;

    /// Report cache TTL
    std::chrono::hours report_cache_ttl = std::chrono::hours(24);

    /// Slice analyzer configuration
    SliceAnalyzerConfig slice_config;

    /// Counterfactual generator configuration
    CounterfactualConfig counterfactual_config;

    /// Temporal analyzer configuration
    TemporalAnalyzerConfig temporal_config;
};

/// @brief Root Cause Analysis service
///
/// Orchestrates all RCA components to provide comprehensive
/// analysis of model issues.
///
/// Components:
/// - Slice Analyzer: Identifies problematic data segments
/// - Temporal Analyzer: Finds time-based correlations
/// - Counterfactual Generator: Creates explanatory examples
///
/// Example:
/// @code
///   RCAServiceConfig config;
///   RCAService service(clickhouse, qdrant, redis, config);
///   service.Initialize();
///
///   // Trigger RCA for a model
///   auto report = service.Analyze("my-model");
///
///   // Or trigger on specific event
///   RCAReport::Trigger trigger;
///   trigger.type = "error_spike";
///   trigger.severity = 0.8;
///   auto report = service.AnalyzeWithTrigger("my-model", trigger);
///
///   // Access findings
///   for (const auto& cause : report->root_causes) {
///       LOG(INFO) << "Root cause: " << cause;
///   }
/// @endcode
class RCAService {
public:
    RCAService(
        std::shared_ptr<storage::ClickHouseClient> clickhouse,
        std::shared_ptr<storage::QdrantClient> qdrant,
        std::shared_ptr<storage::RedisClient> redis,
        RCAServiceConfig config = {});
    ~RCAService();

    // Disable copy
    RCAService(const RCAService&) = delete;
    RCAService& operator=(const RCAService&) = delete;

    /// @brief Initialize the service
    absl::Status Initialize();

    // =========================================================================
    // Analysis API
    // =========================================================================

    /// @brief Run full RCA analysis for a model
    /// @param model_id Model to analyze
    absl::StatusOr<RCAReport> Analyze(const std::string& model_id);

    /// @brief Run RCA with specific trigger
    /// @param model_id Model to analyze
    /// @param trigger Trigger information
    absl::StatusOr<RCAReport> AnalyzeWithTrigger(
        const std::string& model_id,
        const RCAReport::Trigger& trigger);

    /// @brief Run RCA for specific time window
    /// @param model_id Model to analyze
    /// @param start Start time
    /// @param end End time
    absl::StatusOr<RCAReport> AnalyzeTimeRange(
        const std::string& model_id,
        std::chrono::system_clock::time_point start,
        std::chrono::system_clock::time_point end);

    /// @brief Run quick RCA (subset of full analysis)
    /// @param model_id Model to analyze
    absl::StatusOr<RCAReport> QuickAnalyze(const std::string& model_id);

    // =========================================================================
    // Component-Specific Analysis
    // =========================================================================

    /// @brief Run only slice analysis
    absl::StatusOr<std::vector<SliceAnalysisResult>> AnalyzeSlices(
        const std::string& model_id);

    /// @brief Run only temporal analysis
    absl::StatusOr<std::vector<TemporalCorrelation>> AnalyzeTemporalCorrelations(
        const std::string& model_id);

    /// @brief Generate counterfactuals for a specific inference
    absl::StatusOr<std::vector<Counterfactual>> GenerateCounterfactuals(
        const eval::InferenceRecord& record,
        const std::string& target_outcome);

    // =========================================================================
    // Report Management
    // =========================================================================

    /// @brief Get report by ID
    absl::StatusOr<RCAReport> GetReport(const std::string& report_id);

    /// @brief List reports for a model
    absl::StatusOr<std::vector<RCAReport>> ListReports(
        const std::string& model_id,
        size_t limit = 10);

    /// @brief Delete a report
    absl::Status DeleteReport(const std::string& report_id);

    // =========================================================================
    // Callbacks
    // =========================================================================

    /// @brief Register callback for report completion
    void OnReportComplete(std::function<void(const RCAReport&)> callback);

    /// @brief Clear callbacks
    void ClearCallbacks();

    // =========================================================================
    // Configuration
    // =========================================================================

    /// @brief Update configuration
    void SetConfig(RCAServiceConfig config);

    /// @brief Get configuration
    const RCAServiceConfig& GetConfig() const { return config_; }

    // =========================================================================
    // Statistics
    // =========================================================================

    /// @brief Get service statistics
    struct Stats {
        size_t total_analyses = 0;
        size_t successful_analyses = 0;
        size_t failed_analyses = 0;
        double avg_analysis_time_ms = 0.0;
        std::chrono::system_clock::time_point last_analysis;
    };
    Stats GetStats() const;

    /// @brief Reset statistics
    void ResetStats();

private:
    // Analysis phases
    absl::StatusOr<std::vector<SliceAnalysisResult>> RunSliceAnalysis(
        const std::string& model_id,
        std::chrono::system_clock::time_point start,
        std::chrono::system_clock::time_point end);

    absl::StatusOr<std::vector<TemporalCorrelation>> RunTemporalAnalysis(
        const std::string& model_id,
        std::chrono::system_clock::time_point start,
        std::chrono::system_clock::time_point end);

    absl::StatusOr<std::vector<Counterfactual>> RunCounterfactualAnalysis(
        const std::string& model_id,
        const std::vector<SliceAnalysisResult>& problematic_slices);

    // Root cause identification
    std::vector<RCAReport::CausalFactor> IdentifyCausalFactors(
        const std::vector<SliceAnalysisResult>& slices,
        const std::vector<TemporalCorrelation>& correlations,
        const std::vector<Counterfactual>& counterfactuals);

    std::vector<std::string> GenerateRootCauses(
        const std::vector<RCAReport::CausalFactor>& factors);

    std::vector<std::string> GenerateRecommendations(
        const std::vector<RCAReport::CausalFactor>& factors,
        const std::vector<SliceAnalysisResult>& slices);

    std::string GenerateSummary(const RCAReport& report);

    std::vector<std::string> GenerateKeyFindings(const RCAReport& report);

    // Report management
    std::string GenerateReportId();
    absl::Status PersistReport(const RCAReport& report);
    absl::StatusOr<std::optional<RCAReport>> LoadCachedReport(
        const std::string& report_id);

    // Query helpers
    absl::StatusOr<std::vector<eval::InferenceRecord>> QueryProblematicInferences(
        const std::string& model_id,
        const std::vector<SliceAnalysisResult>& slices,
        size_t limit);

    // Storage clients
    std::shared_ptr<storage::ClickHouseClient> clickhouse_;
    std::shared_ptr<storage::QdrantClient> qdrant_;
    std::shared_ptr<storage::RedisClient> redis_;
    RCAServiceConfig config_;

    // Analyzers
    std::unique_ptr<SliceAnalyzer> slice_analyzer_;
    std::unique_ptr<TemporalAnalyzer> temporal_analyzer_;
    std::unique_ptr<CounterfactualGenerator> counterfactual_generator_;

    // Callbacks
    std::vector<std::function<void(const RCAReport&)>> report_callbacks_;
    mutable std::mutex callbacks_mutex_;

    // Statistics
    Stats stats_;
    mutable std::mutex stats_mutex_;

    bool initialized_ = false;
};

/// @brief Factory function to create RCA service
std::unique_ptr<RCAService> CreateRCAService(
    std::shared_ptr<storage::ClickHouseClient> clickhouse,
    std::shared_ptr<storage::QdrantClient> qdrant,
    std::shared_ptr<storage::RedisClient> redis,
    RCAServiceConfig config = {});

/// @brief Convert report status to string
std::string ReportStatusToString(RCAReport::Status status);

/// @brief Serialize RCA report to JSON
std::string SerializeReport(const RCAReport& report);

/// @brief Deserialize RCA report from JSON
absl::StatusOr<RCAReport> DeserializeReport(const std::string& json);

}  // namespace pyflare::rca
