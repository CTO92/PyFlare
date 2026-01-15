#pragma once

/// @file mmd_detector.h
/// @brief Maximum Mean Discrepancy (MMD) drift detector for embeddings

#include <memory>
#include <string>
#include <vector>

#include "processor/drift/drift_detector.h"

namespace pyflare::drift {

/// @brief Configuration for MMD detector
struct MMDConfig {
    /// MMD threshold for drift detection
    double threshold = 0.1;

    /// RBF kernel bandwidth (sigma)
    /// If 0, will be computed using median heuristic
    double rbf_sigma = 0.0;

    /// Number of permutations for p-value estimation
    size_t num_permutations = 100;

    /// P-value threshold for statistical significance
    double p_value_threshold = 0.05;

    /// Maximum number of samples to use (for performance)
    /// 0 = use all samples
    size_t max_samples = 0;

    /// Random seed for reproducibility (0 = random)
    uint64_t random_seed = 0;
};

/// @brief Maximum Mean Discrepancy (MMD) detector for embedding drift
///
/// MMD is a kernel-based statistic that measures the distance between
/// two probability distributions in a reproducing kernel Hilbert space (RKHS).
///
/// Uses the RBF (Gaussian) kernel: k(x,y) = exp(-||x-y||^2 / (2*sigma^2))
///
/// The unbiased MMD^2 estimator is:
///   MMD^2 = 1/(m*(m-1)) * sum_{i!=j} k(x_i, x_j)
///         + 1/(n*(n-1)) * sum_{i!=j} k(y_i, y_j)
///         - 2/(m*n) * sum_{i,j} k(x_i, y_j)
///
/// Example usage:
/// @code
///   MMDConfig config;
///   config.threshold = 0.1;
///   auto detector = std::make_unique<MMDDriftDetector>(config);
///
///   // Set reference embeddings
///   detector->SetReferenceEmbeddings(training_embeddings);
///
///   // Monitor production embeddings
///   auto result = detector->ComputeFromEmbeddings(production_embeddings);
///   if (result->is_drifted) {
///       // Embedding distribution has shifted
///   }
/// @endcode
class MMDDriftDetector : public DriftDetector {
public:
    explicit MMDDriftDetector(MMDConfig config = {});
    ~MMDDriftDetector() override = default;

    // Disable copy
    MMDDriftDetector(const MMDDriftDetector&) = delete;
    MMDDriftDetector& operator=(const MMDDriftDetector&) = delete;

    /// @brief Set reference distribution from DataPoints
    absl::Status SetReference(const Distribution& reference) override;

    /// @brief Set reference distribution from embeddings directly
    /// @param embeddings Vector of embedding vectors
    absl::Status SetReferenceEmbeddings(
        const std::vector<std::vector<float>>& embeddings);

    /// @brief Compute MMD for current batch against reference
    absl::StatusOr<DriftResult> Compute(
        const std::vector<DataPoint>& current_batch) override;

    /// @brief Compute MMD from embeddings directly
    /// @param embeddings Vector of embedding vectors
    absl::StatusOr<DriftResult> ComputeFromEmbeddings(
        const std::vector<std::vector<float>>& embeddings);

    DriftType Type() const override { return DriftType::kEmbedding; }
    std::string Name() const override { return "MMDDriftDetector"; }

    absl::StatusOr<std::string> SerializeState() const override;
    absl::Status LoadState(std::string_view state) override;

    double GetThreshold() const override { return config_.threshold; }
    void SetThreshold(double threshold) override { config_.threshold = threshold; }

    /// @brief Get the current configuration
    const MMDConfig& GetConfig() const { return config_; }

    /// @brief Compute the reference centroid
    std::vector<float> GetReferenceCentroid() const;

    /// @brief Compute centroid drift (simpler than full MMD)
    /// @param embeddings Current embeddings
    /// @return Cosine distance from reference centroid
    double ComputeCentroidDrift(const std::vector<std::vector<float>>& embeddings);

private:
    /// @brief Compute RBF kernel value between two vectors
    double RBFKernel(const std::vector<float>& x, const std::vector<float>& y) const;

    /// @brief Compute squared L2 distance between two vectors
    double SquaredL2Distance(const std::vector<float>& x,
                             const std::vector<float>& y) const;

    /// @brief Compute MMD^2 unbiased estimator
    double ComputeMMDSquared(const std::vector<std::vector<float>>& X,
                             const std::vector<std::vector<float>>& Y) const;

    /// @brief Estimate sigma using median heuristic
    double MedianHeuristic(const std::vector<std::vector<float>>& X) const;

    /// @brief Compute p-value using permutation test
    double PermutationTest(const std::vector<std::vector<float>>& X,
                           const std::vector<std::vector<float>>& Y,
                           double observed_mmd) const;

    /// @brief Compute centroid of embeddings
    std::vector<float> ComputeCentroid(
        const std::vector<std::vector<float>>& embeddings) const;

    /// @brief Cosine similarity between two vectors
    double CosineSimilarity(const std::vector<float>& a,
                            const std::vector<float>& b) const;

    /// @brief Subsample embeddings if too large
    std::vector<std::vector<float>> Subsample(
        const std::vector<std::vector<float>>& embeddings,
        size_t max_samples) const;

    MMDConfig config_;
    std::vector<std::vector<float>> reference_embeddings_;
    std::vector<float> reference_centroid_;
    double sigma_ = 1.0;  // RBF kernel bandwidth
    mutable std::mt19937 rng_;  // Random number generator
};

/// @brief Factory method to create MMD detector
std::unique_ptr<DriftDetector> CreateMMDDetector(MMDConfig config = {});

}  // namespace pyflare::drift
