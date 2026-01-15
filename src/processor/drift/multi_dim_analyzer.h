#pragma once

/// @file multi_dim_analyzer.h
/// @brief Multi-dimensional drift analysis and correlation
///
/// Provides unified drift detection across multiple dimensions:
/// - Feature drift (PSI, KS test)
/// - Embedding drift (MMD)
/// - Concept drift (DDM, EDDM, ADWIN, Page-Hinkley)
/// - Prediction drift (output distribution changes)

#include <chrono>
#include <functional>
#include <memory>
#include <mutex>
#include <string>
#include <unordered_map>
#include <vector>

#include <absl/status/status.h>
#include <absl/status/statusor.h>

#include "processor/drift/concept_drift_detector.h"
#include "processor/drift/drift_detector.h"
#include "processor/drift/mmd_detector.h"
#include "processor/drift/prediction_drift_detector.h"
#include "processor/drift/psi_detector.h"
#include "processor/drift/reference_store.h"
#include "storage/clickhouse/client.h"
#include "storage/qdrant/client.h"
#include "storage/redis/client.h"

namespace pyflare::drift {

/// @brief Reference data for model registration
struct ReferenceData {
    std::string model_id;
    std::string version;
    std::string source;  // "training", "validation", "production"

    /// Feature distributions (for PSI)
    std::vector<std::vector<double>> feature_values;
    std::vector<std::string> feature_names;

    /// Embedding vectors (for MMD)
    std::vector<std::vector<float>> embeddings;

    /// Output data (for prediction drift)
    std::vector<std::string> text_outputs;
    std::vector<std::string> class_labels;
    std::vector<double> numeric_outputs;

    /// Performance data (for concept drift)
    std::vector<bool> correctness_labels;
    std::vector<double> error_values;

    /// Metadata
    std::chrono::system_clock::time_point created_at;
    size_t sample_count = 0;
};

/// @brief Combined drift status across all dimensions
struct MultiDimensionalDriftStatus {
    std::string model_id;
    std::chrono::system_clock::time_point timestamp;

    /// Individual drift scores per dimension
    struct DimensionScore {
        std::string dimension_type;  // "feature", "embedding", "concept", "prediction"
        double score = 0.0;
        double threshold = 0.0;
        bool is_drifted = false;
        std::string explanation;

        /// Detector-specific details
        std::string detector_name;
        std::unordered_map<std::string, double> sub_scores;
    };
    std::vector<DimensionScore> dimension_scores;

    /// Aggregated status
    bool has_any_drift = false;
    double overall_score = 0.0;  ///< Weighted combination of all scores
    std::string severity;        ///< "none", "low", "medium", "high", "critical"

    /// Drift correlations (which dimensions drift together)
    struct Correlation {
        std::string dimension_a;
        std::string dimension_b;
        double correlation_coefficient = 0.0;
        bool are_correlated = false;
        std::string explanation;
    };
    std::vector<Correlation> correlations;

    /// Suggested causes and actions
    std::vector<std::string> likely_causes;
    std::vector<std::string> recommended_actions;

    /// Statistics
    size_t samples_analyzed = 0;
    std::chrono::milliseconds analysis_duration{0};
};

/// @brief Configuration for multi-dimensional analysis
struct MultiDimAnalyzerConfig {
    /// Enable feature drift detection (PSI)
    bool enable_feature_drift = true;

    /// Enable embedding drift detection (MMD)
    bool enable_embedding_drift = true;

    /// Enable concept drift detection
    bool enable_concept_drift = true;

    /// Enable prediction drift detection
    bool enable_prediction_drift = true;

    /// Weights for overall score computation
    double feature_weight = 0.25;
    double embedding_weight = 0.25;
    double concept_weight = 0.25;
    double prediction_weight = 0.25;

    /// Thresholds for severity levels
    double low_threshold = 0.1;
    double medium_threshold = 0.25;
    double high_threshold = 0.5;
    double critical_threshold = 0.75;

    /// Correlation threshold (r value)
    double correlation_threshold = 0.5;

    /// Minimum samples for reliable analysis
    size_t min_samples = 50;

    /// Cache results in Redis
    bool enable_caching = true;

    /// Cache TTL
    std::chrono::seconds cache_ttl = std::chrono::hours(1);

    /// Store results to ClickHouse
    bool persist_results = true;

    /// Individual detector configs
    PSIConfig psi_config;
    MMDConfig mmd_config;
    ConceptDriftConfig concept_config;
    PredictionDriftConfig prediction_config;
};

/// @brief Multi-dimensional drift analyzer
///
/// Provides a unified view of drift across all dimensions and identifies
/// relationships between different types of drift.
///
/// Example:
/// @code
///   MultiDimAnalyzerConfig config;
///   auto analyzer = std::make_unique<MultiDimDriftAnalyzer>(
///       clickhouse, qdrant, redis, config);
///
///   // Register model with reference data
///   analyzer->RegisterModel("my-model", reference_data);
///
///   // Analyze current data
///   auto status = analyzer->Analyze("my-model", current_data);
///
///   if (status->has_any_drift) {
///       for (const auto& dim : status->dimension_scores) {
///           if (dim.is_drifted) {
///               LOG(WARNING) << dim.dimension_type << " drift: " << dim.explanation;
///           }
///       }
///   }
/// @endcode
class MultiDimDriftAnalyzer {
public:
    MultiDimDriftAnalyzer(
        std::shared_ptr<storage::ClickHouseClient> clickhouse,
        std::shared_ptr<storage::QdrantClient> qdrant,
        std::shared_ptr<storage::RedisClient> redis,
        MultiDimAnalyzerConfig config = {});
    ~MultiDimDriftAnalyzer();

    // Disable copy
    MultiDimDriftAnalyzer(const MultiDimDriftAnalyzer&) = delete;
    MultiDimDriftAnalyzer& operator=(const MultiDimDriftAnalyzer&) = delete;

    /// @brief Initialize all detectors
    absl::Status Initialize();

    // =========================================================================
    // Model Management
    // =========================================================================

    /// @brief Register a model with reference data
    /// @param model_id Model identifier
    /// @param reference Reference data for baseline
    absl::Status RegisterModel(
        const std::string& model_id,
        const ReferenceData& reference);

    /// @brief Update reference data for a model
    /// @param model_id Model identifier
    /// @param reference New reference data
    /// @param merge If true, merge with existing; otherwise replace
    absl::Status UpdateReference(
        const std::string& model_id,
        const ReferenceData& reference,
        bool merge = true);

    /// @brief Check if model is registered
    bool IsModelRegistered(const std::string& model_id) const;

    /// @brief Get registered models
    std::vector<std::string> GetRegisteredModels() const;

    /// @brief Unregister a model
    absl::Status UnregisterModel(const std::string& model_id);

    // =========================================================================
    // Drift Analysis
    // =========================================================================

    /// @brief Run full drift analysis on current data
    /// @param model_id Model to analyze
    /// @param current_data Current data batch
    absl::StatusOr<MultiDimensionalDriftStatus> Analyze(
        const std::string& model_id,
        const std::vector<DataPoint>& current_data);

    /// @brief Run analysis from stored data (time range)
    /// @param model_id Model to analyze
    /// @param start Start time
    /// @param end End time
    absl::StatusOr<MultiDimensionalDriftStatus> AnalyzeTimeRange(
        const std::string& model_id,
        std::chrono::system_clock::time_point start,
        std::chrono::system_clock::time_point end);

    /// @brief Run analysis on text outputs
    /// @param model_id Model to analyze
    /// @param inputs Input texts
    /// @param outputs Output texts
    /// @param embeddings Optional embeddings
    absl::StatusOr<MultiDimensionalDriftStatus> AnalyzeTextOutputs(
        const std::string& model_id,
        const std::vector<std::string>& inputs,
        const std::vector<std::string>& outputs,
        const std::vector<std::vector<float>>& embeddings = {});

    /// @brief Run analysis on classification outputs
    /// @param model_id Model to analyze
    /// @param features Feature vectors
    /// @param predictions Predicted classes
    /// @param actual Optional actual classes for concept drift
    absl::StatusOr<MultiDimensionalDriftStatus> AnalyzeClassification(
        const std::string& model_id,
        const std::vector<std::vector<double>>& features,
        const std::vector<std::string>& predictions,
        const std::vector<std::string>& actual = {});

    // =========================================================================
    // Trend Analysis
    // =========================================================================

    /// @brief Get drift trend over time
    /// @param model_id Model to analyze
    /// @param start Start time
    /// @param end End time
    /// @param interval Interval between data points
    absl::StatusOr<std::vector<MultiDimensionalDriftStatus>> GetDriftTrend(
        const std::string& model_id,
        std::chrono::system_clock::time_point start,
        std::chrono::system_clock::time_point end,
        std::chrono::minutes interval);

    /// @brief Compute correlations between drift types
    /// @param model_id Model to analyze
    /// @param history_hours Hours of history to analyze
    absl::StatusOr<std::vector<MultiDimensionalDriftStatus::Correlation>>
    ComputeCorrelations(
        const std::string& model_id,
        size_t history_hours = 24);

    // =========================================================================
    // Callbacks and Notifications
    // =========================================================================

    /// @brief Register callback for drift detection
    /// @param callback Function to call when drift is detected
    void OnDriftDetected(
        std::function<void(const MultiDimensionalDriftStatus&)> callback);

    /// @brief Clear all callbacks
    void ClearCallbacks();

    // =========================================================================
    // Configuration
    // =========================================================================

    /// @brief Update configuration
    void SetConfig(MultiDimAnalyzerConfig config);

    /// @brief Get configuration
    const MultiDimAnalyzerConfig& GetConfig() const { return config_; }

    // =========================================================================
    // Statistics
    // =========================================================================

    /// @brief Get analyzer statistics
    struct Stats {
        size_t total_analyses = 0;
        size_t drift_detections = 0;
        size_t models_registered = 0;
        double avg_analysis_time_ms = 0.0;
        std::chrono::system_clock::time_point last_analysis;
    };
    Stats GetStats() const;

private:
    // Individual dimension computations
    absl::StatusOr<MultiDimensionalDriftStatus::DimensionScore> ComputeFeatureDrift(
        const std::string& model_id,
        const std::vector<DataPoint>& data);

    absl::StatusOr<MultiDimensionalDriftStatus::DimensionScore> ComputeEmbeddingDrift(
        const std::string& model_id,
        const std::vector<std::vector<float>>& embeddings);

    absl::StatusOr<MultiDimensionalDriftStatus::DimensionScore> ComputeConceptDrift(
        const std::string& model_id,
        const std::vector<bool>& correctness);

    absl::StatusOr<MultiDimensionalDriftStatus::DimensionScore> ComputePredictionDrift(
        const std::string& model_id,
        const std::vector<std::string>& outputs);

    // Aggregation and analysis
    double ComputeOverallScore(
        const std::vector<MultiDimensionalDriftStatus::DimensionScore>& scores);

    std::string DetermineSeverity(double overall_score);

    std::vector<std::string> GenerateLikelyCauses(
        const std::vector<MultiDimensionalDriftStatus::DimensionScore>& scores);

    std::vector<std::string> GenerateRecommendations(
        const std::vector<MultiDimensionalDriftStatus::DimensionScore>& scores);

    // Correlation analysis
    double ComputePearsonCorrelation(
        const std::vector<double>& x,
        const std::vector<double>& y);

    // Persistence
    absl::Status PersistResult(const MultiDimensionalDriftStatus& status);
    absl::StatusOr<std::optional<MultiDimensionalDriftStatus>> LoadCachedResult(
        const std::string& model_id);

    // Query data from ClickHouse
    absl::StatusOr<std::vector<DataPoint>> QueryDataFromClickHouse(
        const std::string& model_id,
        std::chrono::system_clock::time_point start,
        std::chrono::system_clock::time_point end);

    // Build status object
    MultiDimensionalDriftStatus BuildStatus(
        const std::string& model_id,
        const std::vector<MultiDimensionalDriftStatus::DimensionScore>& scores,
        size_t samples_analyzed,
        std::chrono::milliseconds duration);

    // Storage clients
    std::shared_ptr<storage::ClickHouseClient> clickhouse_;
    std::shared_ptr<storage::QdrantClient> qdrant_;
    std::shared_ptr<storage::RedisClient> redis_;
    MultiDimAnalyzerConfig config_;

    // Reference store
    std::unique_ptr<ReferenceStore> reference_store_;

    // Detectors per model
    struct ModelDetectors {
        std::unique_ptr<PSIDriftDetector> psi_detector;
        std::unique_ptr<MMDDriftDetector> mmd_detector;
        std::unique_ptr<ConceptDriftDetector> concept_detector;
        std::unique_ptr<PredictionDriftDetector> prediction_detector;

        bool is_initialized = false;
        std::chrono::system_clock::time_point registered_at;
        size_t analysis_count = 0;
    };
    std::unordered_map<std::string, ModelDetectors> model_detectors_;
    mutable std::mutex detectors_mutex_;

    // Callbacks
    std::vector<std::function<void(const MultiDimensionalDriftStatus&)>> drift_callbacks_;
    mutable std::mutex callbacks_mutex_;

    // Statistics
    Stats stats_;
    mutable std::mutex stats_mutex_;

    bool initialized_ = false;
};

/// @brief Factory function to create multi-dimensional analyzer
std::unique_ptr<MultiDimDriftAnalyzer> CreateMultiDimAnalyzer(
    std::shared_ptr<storage::ClickHouseClient> clickhouse,
    std::shared_ptr<storage::QdrantClient> qdrant,
    std::shared_ptr<storage::RedisClient> redis,
    MultiDimAnalyzerConfig config = {});

}  // namespace pyflare::drift
