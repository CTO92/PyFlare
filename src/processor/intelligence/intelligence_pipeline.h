#pragma once

/// @file intelligence_pipeline.h
/// @brief Intelligence pipeline orchestrating all analysis components
///
/// Provides unified orchestration of:
/// - Drift detection (multi-dimensional)
/// - Evaluations (semantic, custom, safety)
/// - Root cause analysis
/// - Alerting
///
/// The pipeline processes inference records through all intelligence
/// components and generates actionable insights.

#include <atomic>
#include <chrono>
#include <condition_variable>
#include <functional>
#include <memory>
#include <mutex>
#include <queue>
#include <string>
#include <thread>
#include <unordered_map>
#include <vector>

#include <absl/status/status.h>
#include <absl/status/statusor.h>

#include "processor/alerting/alert_service.h"
#include "processor/drift/drift_service.h"
#include "processor/drift/multi_dim_analyzer.h"
#include "processor/eval/custom_evaluator.h"
#include "processor/eval/evaluator.h"
#include "processor/eval/prompt_injection_detector.h"
#include "processor/eval/semantic_similarity.h"
#include "processor/rca/rca_service.h"
#include "storage/clickhouse/client.h"
#include "storage/qdrant/client.h"
#include "storage/redis/client.h"

namespace pyflare::intelligence {

/// @brief Intelligence analysis result for a single inference
struct IntelligenceResult {
    std::string trace_id;
    std::string model_id;
    std::chrono::system_clock::time_point analyzed_at;

    /// Drift analysis
    struct DriftResult {
        bool drift_detected = false;
        double overall_severity = 0.0;
        std::vector<std::string> drifted_dimensions;
        std::vector<std::string> causes;
    };
    DriftResult drift;

    /// Evaluation results
    struct EvaluationResult {
        double overall_score = 0.0;
        std::unordered_map<std::string, double> evaluator_scores;
        std::vector<std::string> issues;
        bool passed = true;
    };
    EvaluationResult evaluation;

    /// Safety analysis
    struct SafetyResult {
        bool is_safe = true;
        double risk_score = 0.0;
        std::vector<std::string> detected_issues;
        std::string risk_level;
    };
    SafetyResult safety;

    /// RCA insights (if applicable)
    struct RCAInsights {
        bool rca_triggered = false;
        std::vector<std::string> root_causes;
        std::vector<std::string> recommendations;
        double confidence = 0.0;
    };
    RCAInsights rca;

    /// Alerts generated
    std::vector<alerting::AlertEvent> alerts;

    /// Overall health score (0-1, higher is better)
    double health_score = 1.0;

    /// Processing metadata
    std::chrono::milliseconds processing_time{0};
};

/// @brief Batch intelligence result
struct BatchIntelligenceResult {
    std::vector<IntelligenceResult> results;

    /// Aggregate metrics
    size_t total_processed = 0;
    size_t drift_detected_count = 0;
    size_t safety_issues_count = 0;
    size_t evaluation_failures = 0;

    double avg_health_score = 0.0;
    double avg_processing_time_ms = 0.0;

    std::chrono::system_clock::time_point batch_start;
    std::chrono::system_clock::time_point batch_end;
};

/// @brief Pipeline configuration
struct IntelligencePipelineConfig {
    /// Enable/disable components
    bool enable_drift_detection = true;
    bool enable_evaluations = true;
    bool enable_safety_checks = true;
    bool enable_rca = true;
    bool enable_alerting = true;

    /// Processing settings
    size_t batch_size = 100;
    std::chrono::seconds processing_timeout = std::chrono::seconds(30);

    /// Worker threads for parallel processing
    size_t worker_threads = 4;

    /// Component configurations
    drift::MultiDimAnalyzerConfig drift_config;
    eval::PromptInjectionConfig safety_config;
    eval::SemanticSimilarityConfig similarity_config;
    rca::RCAServiceConfig rca_config;
    alerting::AlertServiceConfig alert_config;

    /// Auto-trigger RCA on issues
    bool auto_rca_on_drift = true;
    bool auto_rca_on_safety_issues = true;
    double rca_severity_threshold = 0.7;

    /// Persist results
    bool persist_results = true;
    bool cache_results = true;
    std::chrono::hours result_cache_ttl = std::chrono::hours(24);
};

/// @brief Intelligence pipeline
///
/// Orchestrates all intelligence analysis components into a unified
/// processing pipeline.
///
/// Example:
/// @code
///   IntelligencePipelineConfig config;
///   config.enable_drift_detection = true;
///   config.enable_safety_checks = true;
///
///   IntelligencePipeline pipeline(clickhouse, qdrant, redis, config);
///   pipeline.Initialize();
///   pipeline.Start();
///
///   // Process single inference
///   eval::InferenceRecord record;
///   record.input = "What is the capital of France?";
///   record.output = "The capital of France is Paris.";
///   auto result = pipeline.Process(record);
///
///   // Check results
///   if (result.drift.drift_detected) {
///       LOG(WARNING) << "Drift detected!";
///   }
///   if (!result.safety.is_safe) {
///       LOG(WARNING) << "Safety issue: " << result.safety.detected_issues[0];
///   }
///
///   // Process batch
///   auto batch_result = pipeline.ProcessBatch(records);
/// @endcode
class IntelligencePipeline {
public:
    IntelligencePipeline(
        std::shared_ptr<storage::ClickHouseClient> clickhouse,
        std::shared_ptr<storage::QdrantClient> qdrant,
        std::shared_ptr<storage::RedisClient> redis,
        IntelligencePipelineConfig config = {});
    ~IntelligencePipeline();

    // Disable copy
    IntelligencePipeline(const IntelligencePipeline&) = delete;
    IntelligencePipeline& operator=(const IntelligencePipeline&) = delete;

    /// @brief Initialize pipeline and all components
    absl::Status Initialize();

    /// @brief Start background workers
    absl::Status Start();

    /// @brief Stop pipeline
    void Stop();

    /// @brief Check if running
    bool IsRunning() const { return running_.load(); }

    // =========================================================================
    // Processing API
    // =========================================================================

    /// @brief Process single inference record
    IntelligenceResult Process(const eval::InferenceRecord& record);

    /// @brief Process batch of records
    BatchIntelligenceResult ProcessBatch(
        const std::vector<eval::InferenceRecord>& records);

    /// @brief Async process with callback
    void ProcessAsync(const eval::InferenceRecord& record,
                      std::function<void(const IntelligenceResult&)> callback);

    // =========================================================================
    // Model Management
    // =========================================================================

    /// @brief Register model for drift detection
    absl::Status RegisterModel(const std::string& model_id,
                               const drift::ReferenceData& reference);

    /// @brief Update model reference data
    absl::Status UpdateModelReference(const std::string& model_id,
                                      const drift::ReferenceData& reference);

    /// @brief Remove model
    absl::Status RemoveModel(const std::string& model_id);

    /// @brief List registered models
    std::vector<std::string> ListModels() const;

    // =========================================================================
    // Evaluator Management
    // =========================================================================

    /// @brief Add custom evaluator
    absl::Status AddEvaluator(const std::string& name,
                              std::unique_ptr<eval::Evaluator> evaluator);

    /// @brief Remove evaluator
    absl::Status RemoveEvaluator(const std::string& name);

    /// @brief List evaluators
    std::vector<std::string> ListEvaluators() const;

    /// @brief Enable/disable evaluator
    absl::Status SetEvaluatorEnabled(const std::string& name, bool enabled);

    // =========================================================================
    // Alert Rule Management
    // =========================================================================

    /// @brief Add alert rule
    absl::Status AddAlertRule(const alerting::AlertRule& rule);

    /// @brief Remove alert rule
    absl::Status RemoveAlertRule(const std::string& rule_id);

    /// @brief List alert rules
    std::vector<alerting::AlertRule> ListAlertRules() const;

    // =========================================================================
    // RCA Management
    // =========================================================================

    /// @brief Trigger manual RCA
    absl::StatusOr<rca::RCAReport> TriggerRCA(const std::string& model_id);

    /// @brief Get RCA report
    absl::StatusOr<rca::RCAReport> GetRCAReport(const std::string& report_id);

    /// @brief List RCA reports
    std::vector<rca::RCAReport> ListRCAReports(const std::string& model_id,
                                                size_t limit = 10);

    // =========================================================================
    // Insights & Analytics
    // =========================================================================

    /// @brief Get model health summary
    struct ModelHealthSummary {
        std::string model_id;
        double health_score = 1.0;
        bool has_active_drift = false;
        size_t active_alerts = 0;
        size_t recent_safety_issues = 0;
        double avg_evaluation_score = 0.0;
        std::chrono::system_clock::time_point last_analyzed;
    };
    absl::StatusOr<ModelHealthSummary> GetModelHealth(
        const std::string& model_id) const;

    /// @brief Get overall system health
    struct SystemHealthSummary {
        double overall_health = 1.0;
        size_t models_with_drift = 0;
        size_t total_active_alerts = 0;
        size_t models_analyzed = 0;
        double avg_health_score = 0.0;
        std::chrono::system_clock::time_point last_update;
    };
    SystemHealthSummary GetSystemHealth() const;

    // =========================================================================
    // Callbacks
    // =========================================================================

    /// @brief Register callback for intelligence results
    void OnResult(std::function<void(const IntelligenceResult&)> callback);

    /// @brief Register callback for drift detection
    void OnDrift(std::function<void(const std::string&, const drift::MultiDimensionalDriftStatus&)> callback);

    /// @brief Register callback for safety issues
    void OnSafetyIssue(std::function<void(const eval::InjectionDetectionResult&)> callback);

    /// @brief Clear callbacks
    void ClearCallbacks();

    // =========================================================================
    // Statistics
    // =========================================================================

    struct Stats {
        // Processing stats
        size_t total_processed = 0;
        size_t drift_detections = 0;
        size_t safety_issues = 0;
        size_t evaluation_failures = 0;
        size_t rca_triggered = 0;
        size_t alerts_generated = 0;

        // Performance stats
        double avg_processing_time_ms = 0.0;
        double p99_processing_time_ms = 0.0;
        size_t queue_depth = 0;

        // Component stats
        bool drift_service_healthy = true;
        bool eval_service_healthy = true;
        bool rca_service_healthy = true;
        bool alert_service_healthy = true;

        std::chrono::system_clock::time_point last_processed;
    };
    Stats GetStats() const;

    void ResetStats();

    // =========================================================================
    // Configuration
    // =========================================================================

    void SetConfig(IntelligencePipelineConfig config);
    const IntelligencePipelineConfig& GetConfig() const { return config_; }

private:
    // Processing phases
    void RunDriftDetection(const eval::InferenceRecord& record,
                           IntelligenceResult& result);
    void RunEvaluations(const eval::InferenceRecord& record,
                        IntelligenceResult& result);
    void RunSafetyChecks(const eval::InferenceRecord& record,
                         IntelligenceResult& result);
    void RunRCAIfNeeded(const eval::InferenceRecord& record,
                        IntelligenceResult& result);
    void GenerateAlerts(const eval::InferenceRecord& record,
                        IntelligenceResult& result);
    double ComputeHealthScore(const IntelligenceResult& result);

    // Background workers
    void ProcessingLoop();
    void AsyncProcessingLoop();

    // Persistence
    absl::Status PersistResult(const IntelligenceResult& result);

    // Convert to metrics for alerting
    std::vector<alerting::MetricValue> ResultToMetrics(
        const IntelligenceResult& result);

    // Storage clients
    std::shared_ptr<storage::ClickHouseClient> clickhouse_;
    std::shared_ptr<storage::QdrantClient> qdrant_;
    std::shared_ptr<storage::RedisClient> redis_;
    IntelligencePipelineConfig config_;

    // Components
    std::unique_ptr<drift::MultiDimDriftAnalyzer> drift_analyzer_;
    std::unique_ptr<eval::PromptInjectionDetector> injection_detector_;
    std::unique_ptr<eval::SemanticSimilarityScorer> similarity_scorer_;
    std::unique_ptr<rca::RCAService> rca_service_;
    std::unique_ptr<alerting::AlertService> alert_service_;

    // Custom evaluators
    std::unordered_map<std::string, std::unique_ptr<eval::Evaluator>> evaluators_;
    std::unordered_map<std::string, bool> evaluator_enabled_;
    mutable std::mutex evaluators_mutex_;

    // Model health tracking
    std::unordered_map<std::string, ModelHealthSummary> model_health_;
    mutable std::mutex health_mutex_;

    // Async processing queue
    struct AsyncTask {
        eval::InferenceRecord record;
        std::function<void(const IntelligenceResult&)> callback;
    };
    std::queue<AsyncTask> async_queue_;
    std::mutex queue_mutex_;
    std::condition_variable queue_cv_;

    // Workers
    std::vector<std::thread> workers_;
    std::atomic<bool> running_{false};

    // Callbacks
    std::vector<std::function<void(const IntelligenceResult&)>> result_callbacks_;
    std::vector<std::function<void(const std::string&, const drift::MultiDimensionalDriftStatus&)>> drift_callbacks_;
    std::vector<std::function<void(const eval::InjectionDetectionResult&)>> safety_callbacks_;
    mutable std::mutex callbacks_mutex_;

    // Statistics
    Stats stats_;
    mutable std::mutex stats_mutex_;
    std::vector<double> processing_times_;  // For percentile calculation

    bool initialized_ = false;
};

/// @brief Factory function
std::unique_ptr<IntelligencePipeline> CreateIntelligencePipeline(
    std::shared_ptr<storage::ClickHouseClient> clickhouse,
    std::shared_ptr<storage::QdrantClient> qdrant,
    std::shared_ptr<storage::RedisClient> redis,
    IntelligencePipelineConfig config = {});

/// @brief Serialize intelligence result to JSON
std::string SerializeIntelligenceResult(const IntelligenceResult& result);

/// @brief Deserialize intelligence result from JSON
absl::StatusOr<IntelligenceResult> DeserializeIntelligenceResult(
    const std::string& json);

}  // namespace pyflare::intelligence
