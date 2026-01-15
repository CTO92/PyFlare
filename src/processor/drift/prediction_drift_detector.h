#pragma once

/// @file prediction_drift_detector.h
/// @brief Prediction/output drift detection for model outputs
///
/// Monitors changes in model output distributions independently of input changes.

#include <chrono>
#include <memory>
#include <mutex>
#include <optional>
#include <string>
#include <unordered_map>
#include <vector>

#include <absl/status/status.h>
#include <absl/status/statusor.h>

#include "processor/drift/drift_detector.h"

namespace pyflare::drift {

/// @brief Types of output to monitor
enum class OutputType {
    kClassification,  ///< Discrete class labels
    kRegression,      ///< Continuous values
    kText,            ///< Text outputs (LLM)
    kEmbedding,       ///< Embedding vectors
    kProbability      ///< Probability distributions
};

/// @brief Configuration for prediction drift
struct PredictionDriftConfig {
    OutputType output_type = OutputType::kText;

    /// Threshold for drift detection
    double threshold = 0.1;

    /// For classification: monitor class distribution
    bool monitor_class_distribution = true;

    /// For regression: monitor output statistics
    bool monitor_output_statistics = true;

    /// For text: use embedding similarity
    bool use_text_embeddings = true;

    /// For text: monitor output length distribution
    bool monitor_output_length = true;

    /// For text: monitor vocabulary distribution
    bool monitor_vocabulary = false;

    /// Window size for comparison
    size_t window_size = 1000;

    /// Minimum samples before detection
    size_t min_samples = 50;

    /// Number of bins for continuous outputs
    size_t num_bins = 20;

    /// P-value threshold for statistical tests
    double p_value_threshold = 0.05;

    /// Embedding dimension (for text mode)
    size_t embedding_dimension = 1536;
};

/// @brief Prediction drift result
struct PredictionDriftResult {
    bool drift_detected = false;
    double drift_score = 0.0;
    double threshold = 0.0;

    OutputType output_type;

    // For classification
    std::unordered_map<std::string, double> class_distribution_current;
    std::unordered_map<std::string, double> class_distribution_reference;
    double class_distribution_divergence = 0.0;

    // For regression
    double mean_current = 0.0;
    double mean_reference = 0.0;
    double std_current = 0.0;
    double std_reference = 0.0;
    double ks_statistic = 0.0;

    // For text
    double embedding_drift = 0.0;
    double length_drift = 0.0;
    double vocabulary_drift = 0.0;

    // Statistical details
    double p_value = 0.0;

    std::string explanation;

    // Top changed classes (for classification)
    std::vector<std::pair<std::string, double>> top_changed_classes;
};

/// @brief Prediction drift detector
///
/// Monitors the output distribution of a model for changes.
/// Supports multiple output types:
/// - Classification: KL divergence on class distribution
/// - Regression: Statistical tests on output distribution
/// - Text: Embedding-based similarity + structural analysis
///
/// Example:
/// @code
///   PredictionDriftConfig config;
///   config.output_type = OutputType::kText;
///   auto detector = std::make_unique<PredictionDriftDetector>(config);
///
///   // Set reference from training/validation outputs
///   detector->SetReferenceTexts(validation_outputs);
///
///   // Monitor production outputs
///   auto result = detector->ComputeTextDrift(production_outputs);
/// @endcode
class PredictionDriftDetector : public DriftDetector {
public:
    explicit PredictionDriftDetector(PredictionDriftConfig config = {});
    ~PredictionDriftDetector() override;

    // Disable copy
    PredictionDriftDetector(const PredictionDriftDetector&) = delete;
    PredictionDriftDetector& operator=(const PredictionDriftDetector&) = delete;

    // ===========================================================================
    // DriftDetector Interface
    // ===========================================================================

    absl::Status SetReference(const Distribution& reference) override;

    absl::StatusOr<DriftResult> Compute(
        const std::vector<DataPoint>& current_batch) override;

    DriftType Type() const override { return DriftType::kPrediction; }
    std::string Name() const override { return "PredictionDriftDetector"; }

    absl::StatusOr<std::string> SerializeState() const override;
    absl::Status LoadState(std::string_view state) override;

    double GetThreshold() const override { return config_.threshold; }
    void SetThreshold(double threshold) override { config_.threshold = threshold; }

    // ===========================================================================
    // Output-specific Reference Setting
    // ===========================================================================

    /// @brief Set reference from text outputs
    absl::Status SetReferenceTexts(const std::vector<std::string>& outputs);

    /// @brief Set reference from class labels
    absl::Status SetReferenceClasses(const std::vector<std::string>& classes);

    /// @brief Set reference from numeric outputs
    absl::Status SetReferenceValues(const std::vector<double>& values);

    /// @brief Set reference from probability distributions
    absl::Status SetReferenceProbabilities(
        const std::vector<std::vector<double>>& probabilities,
        const std::vector<std::string>& class_names = {});

    /// @brief Set reference from embeddings directly
    absl::Status SetReferenceEmbeddings(
        const std::vector<std::vector<float>>& embeddings);

    // ===========================================================================
    // Drift Computation
    // ===========================================================================

    /// @brief Compute drift for text outputs
    absl::StatusOr<PredictionDriftResult> ComputeTextDrift(
        const std::vector<std::string>& outputs);

    /// @brief Compute drift for classification outputs
    absl::StatusOr<PredictionDriftResult> ComputeClassDrift(
        const std::vector<std::string>& classes);

    /// @brief Compute drift for regression outputs
    absl::StatusOr<PredictionDriftResult> ComputeValueDrift(
        const std::vector<double>& values);

    /// @brief Compute drift from embeddings directly
    absl::StatusOr<PredictionDriftResult> ComputeEmbeddingDrift(
        const std::vector<std::vector<float>>& embeddings);

    // ===========================================================================
    // Reference Statistics
    // ===========================================================================

    /// @brief Get current reference statistics
    struct ReferenceStats {
        OutputType type;
        size_t sample_count = 0;
        bool is_set = false;

        // Text stats
        double avg_length = 0.0;
        double std_length = 0.0;
        std::vector<float> centroid_embedding;

        // Class stats
        std::unordered_map<std::string, double> class_distribution;
        std::vector<std::string> class_names;

        // Value stats
        double mean = 0.0;
        double std_dev = 0.0;
        double min = 0.0;
        double max = 0.0;
        std::vector<double> histogram_bins;
        std::vector<double> histogram_counts;
    };
    ReferenceStats GetReferenceStats() const;

    /// @brief Check if reference is set
    bool HasReference() const;

    /// @brief Clear reference
    void ClearReference();

    /// @brief Get configuration
    const PredictionDriftConfig& GetConfig() const { return config_; }

private:
    // Statistical computations
    double ComputeKLDivergence(
        const std::unordered_map<std::string, double>& p,
        const std::unordered_map<std::string, double>& q);

    double ComputeJSDivergence(
        const std::unordered_map<std::string, double>& p,
        const std::unordered_map<std::string, double>& q);

    double ComputeKSStatistic(
        const std::vector<double>& reference,
        const std::vector<double>& current);

    double ComputeTTest(
        const std::vector<double>& reference,
        const std::vector<double>& current);

    // Embedding utilities
    std::vector<float> ComputeCentroid(
        const std::vector<std::vector<float>>& embeddings);

    double ComputeCosineSimilarity(
        const std::vector<float>& a,
        const std::vector<float>& b);

    double ComputeMMD(
        const std::vector<std::vector<float>>& reference,
        const std::vector<std::vector<float>>& current);

    // Text analysis
    std::unordered_map<std::string, double> ComputeLengthDistribution(
        const std::vector<std::string>& texts);

    // Class distribution
    std::unordered_map<std::string, double> ComputeClassDistribution(
        const std::vector<std::string>& classes);

    // Histogram computation
    std::pair<std::vector<double>, std::vector<double>> ComputeHistogram(
        const std::vector<double>& values,
        size_t num_bins);

    PredictionDriftConfig config_;

    // Reference data
    struct Reference {
        bool is_set = false;
        OutputType type;

        // For text
        std::vector<float> centroid_embedding;
        std::vector<std::vector<float>> reference_embeddings;
        double avg_length = 0.0;
        double std_length = 0.0;

        // For classification
        std::unordered_map<std::string, double> class_distribution;
        std::vector<std::string> class_names;

        // For regression
        std::vector<double> values;
        double mean = 0.0;
        double std_dev = 0.0;
        double min = 0.0;
        double max = 0.0;
        std::vector<double> histogram_bins;
        std::vector<double> histogram_counts;

        size_t sample_count = 0;
    };
    Reference reference_;

    mutable std::mutex mutex_;
};

/// @brief Factory function
std::unique_ptr<DriftDetector> CreatePredictionDriftDetector(
    PredictionDriftConfig config = {});

/// @brief Convert output type to string
std::string OutputTypeToString(OutputType type);

/// @brief Convert string to output type
OutputType StringToOutputType(const std::string& str);

}  // namespace pyflare::drift
