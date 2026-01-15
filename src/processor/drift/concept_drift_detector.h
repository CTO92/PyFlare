#pragma once

/// @file concept_drift_detector.h
/// @brief Concept drift detection for input-output relationship changes
///
/// Implements multiple algorithms for detecting concept drift:
/// - DDM (Drift Detection Method)
/// - EDDM (Early Drift Detection Method)
/// - ADWIN (Adaptive Windowing)
/// - Page-Hinkley test

#include <chrono>
#include <deque>
#include <memory>
#include <mutex>
#include <optional>
#include <random>
#include <string>
#include <unordered_map>
#include <vector>

#include <absl/status/status.h>
#include <absl/status/statusor.h>

#include "processor/drift/drift_detector.h"

namespace pyflare::drift {

/// @brief Methods for concept drift detection
enum class ConceptDriftMethod {
    kDDM,           ///< Drift Detection Method (error rate monitoring)
    kEDDM,          ///< Early Drift Detection Method (distance-based)
    kADWIN,         ///< Adaptive Windowing
    kPageHinkley,   ///< Page-Hinkley test for mean shift
    kEmbeddingDelta ///< Embedding space relationship change
};

/// @brief Configuration for concept drift detection
struct ConceptDriftConfig {
    ConceptDriftMethod method = ConceptDriftMethod::kADWIN;

    /// Window size for streaming algorithms
    size_t window_size = 1000;

    /// Threshold for drift detection (method-specific)
    double threshold = 0.1;

    /// Confidence level for statistical tests
    double confidence_level = 0.95;

    /// DDM-specific: warning level threshold (standard deviations)
    double ddm_warning_level = 2.0;

    /// DDM-specific: drift level threshold (standard deviations)
    double ddm_drift_level = 3.0;

    /// ADWIN-specific: delta parameter for confidence
    double adwin_delta = 0.002;

    /// Page-Hinkley: detection threshold
    double ph_threshold = 50.0;

    /// Page-Hinkley: magnitude of allowed change
    double ph_alpha = 0.005;

    /// Minimum samples before detection is active
    size_t min_samples = 30;

    /// Performance metric to monitor (for DDM/EDDM)
    std::string performance_metric = "accuracy";
};

/// @brief Concept drift detection result
struct ConceptDriftResult {
    bool drift_detected = false;
    bool warning_detected = false;

    double drift_score = 0.0;
    double current_error_rate = 0.0;
    double baseline_error_rate = 0.0;

    ConceptDriftMethod method_used;
    size_t samples_analyzed = 0;
    size_t drift_point = 0;  ///< Index where drift was detected

    std::string explanation;

    /// For ADWIN: detected change points
    std::vector<size_t> change_points;

    /// Per-feature contribution to drift (if available)
    std::unordered_map<std::string, double> feature_contributions;

    /// Statistical details
    double p_value = 0.0;
    double statistic = 0.0;
};

/// @brief Concept drift detector using online learning algorithms
///
/// Monitors the input-output relationship for changes. Unlike feature
/// drift which tracks input distribution, concept drift detects when
/// the same inputs start producing different outputs.
///
/// Implements multiple detection methods:
/// - DDM (Drift Detection Method): Monitors error rate
/// - EDDM (Early DDM): More sensitive to gradual drift
/// - ADWIN (Adaptive Windowing): Automatically adjusts window size
/// - Page-Hinkley: Sequential change detection
///
/// Example:
/// @code
///   ConceptDriftConfig config;
///   config.method = ConceptDriftMethod::kADWIN;
///   auto detector = std::make_unique<ConceptDriftDetector>(config);
///
///   // Feed performance observations
///   for (const auto& result : inference_results) {
///       auto drift_result = detector->Update(result.correct);
///       if (drift_result->drift_detected) {
///           // Model behavior has changed
///       }
///   }
/// @endcode
class ConceptDriftDetector : public DriftDetector {
public:
    explicit ConceptDriftDetector(ConceptDriftConfig config = {});
    ~ConceptDriftDetector() override;

    // Disable copy
    ConceptDriftDetector(const ConceptDriftDetector&) = delete;
    ConceptDriftDetector& operator=(const ConceptDriftDetector&) = delete;

    // ===========================================================================
    // DriftDetector Interface
    // ===========================================================================

    /// @brief Set reference distribution (converts to error observations)
    absl::Status SetReference(const Distribution& reference) override;

    /// @brief Compute drift for batch of data points
    absl::StatusOr<DriftResult> Compute(
        const std::vector<DataPoint>& current_batch) override;

    DriftType Type() const override { return DriftType::kConcept; }
    std::string Name() const override { return "ConceptDriftDetector"; }

    absl::StatusOr<std::string> SerializeState() const override;
    absl::Status LoadState(std::string_view state) override;

    double GetThreshold() const override { return config_.threshold; }
    void SetThreshold(double threshold) override { config_.threshold = threshold; }

    // ===========================================================================
    // Streaming API
    // ===========================================================================

    /// @brief Update with single observation (streaming)
    /// @param correct True if prediction was correct
    /// @return Drift detection result after this observation
    absl::StatusOr<ConceptDriftResult> Update(bool correct);

    /// @brief Update with error value (for regression)
    /// @param error Prediction error value (e.g., MSE for single sample)
    absl::StatusOr<ConceptDriftResult> UpdateWithError(double error);

    /// @brief Update with inference record (full context)
    /// @param input Input text/data
    /// @param output Model output
    /// @param expected Expected output for comparison
    absl::StatusOr<ConceptDriftResult> UpdateWithRecord(
        const std::string& input,
        const std::string& output,
        const std::string& expected);

    // ===========================================================================
    // Batch API
    // ===========================================================================

    /// @brief Analyze batch for concept drift
    /// @param predictions Vector of (correct: bool) results
    absl::StatusOr<ConceptDriftResult> AnalyzeBatch(
        const std::vector<bool>& predictions);

    /// @brief Analyze using embeddings (input-output relationship)
    /// @param input_embeddings Input embedding vectors
    /// @param output_embeddings Output embedding vectors
    absl::StatusOr<ConceptDriftResult> AnalyzeEmbeddings(
        const std::vector<std::vector<float>>& input_embeddings,
        const std::vector<std::vector<float>>& output_embeddings);

    // ===========================================================================
    // State Management
    // ===========================================================================

    /// @brief Reset detector state
    void Reset();

    /// @brief Get current state statistics
    struct State {
        size_t total_samples = 0;
        double error_rate = 0.0;
        double error_rate_std = 0.0;
        bool in_warning_state = false;
        size_t warning_start_index = 0;
        size_t samples_since_warning = 0;

        // DDM/EDDM specific
        double min_error_rate = 1.0;
        double min_std = 1.0;
        size_t min_error_index = 0;

        // ADWIN specific
        size_t window_size = 0;
        double window_mean = 0.0;

        // Page-Hinkley specific
        double cumulative_sum = 0.0;
        double min_cumulative = 0.0;
    };
    State GetState() const;

    /// @brief Get configuration
    const ConceptDriftConfig& GetConfig() const { return config_; }

    /// @brief Change detection method
    void SetMethod(ConceptDriftMethod method);

private:
    // DDM (Drift Detection Method) implementation
    ConceptDriftResult UpdateDDM(double error);

    // EDDM (Early DDM) implementation
    ConceptDriftResult UpdateEDDM(double error);

    // ADWIN (Adaptive Windowing) implementation
    ConceptDriftResult UpdateADWIN(double value);

    // Page-Hinkley implementation
    ConceptDriftResult UpdatePageHinkley(double value);

    // Helper: Check if minimum samples reached
    bool HasMinimumSamples() const;

    // Helper: Build result object
    ConceptDriftResult BuildResult(bool drift, bool warning,
                                    const std::string& explanation);

    ConceptDriftConfig config_;

    // State variables
    size_t n_samples_ = 0;
    double sum_errors_ = 0.0;
    double sum_squared_errors_ = 0.0;

    // DDM state
    double ddm_p_min_ = std::numeric_limits<double>::max();
    double ddm_s_min_ = std::numeric_limits<double>::max();
    size_t ddm_min_index_ = 0;
    bool ddm_in_warning_ = false;
    size_t ddm_warning_index_ = 0;

    // EDDM state
    double eddm_m2_min_ = std::numeric_limits<double>::max();
    double eddm_d_min_ = std::numeric_limits<double>::max();
    size_t eddm_last_error_index_ = 0;
    std::deque<size_t> eddm_distances_;

    // ADWIN state (bucket-based)
    struct ADWINBucket {
        double total = 0.0;
        double variance = 0.0;
        size_t count = 0;
    };
    std::deque<ADWINBucket> adwin_buckets_;
    size_t adwin_width_ = 0;
    double adwin_total_ = 0.0;
    double adwin_variance_ = 0.0;

    // Page-Hinkley state
    double ph_sum_ = 0.0;
    double ph_mean_ = 0.0;
    double ph_min_ = 0.0;

    // Thread safety
    mutable std::mutex state_mutex_;
};

/// @brief Factory function to create concept drift detector
std::unique_ptr<DriftDetector> CreateConceptDriftDetector(
    ConceptDriftConfig config = {});

/// @brief Convert method enum to string
std::string ConceptDriftMethodToString(ConceptDriftMethod method);

/// @brief Convert string to method enum
ConceptDriftMethod StringToConceptDriftMethod(const std::string& str);

}  // namespace pyflare::drift
