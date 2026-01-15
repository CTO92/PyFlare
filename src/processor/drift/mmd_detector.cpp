/// @file mmd_detector.cpp
/// @brief MMD drift detector implementation

#include "processor/drift/mmd_detector.h"

#include <algorithm>
#include <cmath>
#include <numeric>
#include <random>

#include <nlohmann/json.hpp>
#include <spdlog/spdlog.h>

namespace pyflare::drift {

using json = nlohmann::json;

// =============================================================================
// MMDDriftDetector Implementation
// =============================================================================

MMDDriftDetector::MMDDriftDetector(MMDConfig config)
    : config_(std::move(config)) {
    // Initialize random number generator
    if (config_.random_seed != 0) {
        rng_.seed(config_.random_seed);
    } else {
        std::random_device rd;
        rng_.seed(rd());
    }
}

absl::Status MMDDriftDetector::SetReference(const Distribution& reference) {
    // Extract embeddings from DataPoints
    std::vector<std::vector<float>> embeddings;
    embeddings.reserve(reference.Size());

    // Note: This assumes DataPoints have embeddings in the 'embedding' field
    // For feature-based points, we'd need to convert features to embeddings

    spdlog::warn("SetReference from Distribution not fully implemented, "
                 "use SetReferenceEmbeddings instead");

    return absl::OkStatus();
}

absl::Status MMDDriftDetector::SetReferenceEmbeddings(
    const std::vector<std::vector<float>>& embeddings) {

    if (embeddings.empty()) {
        return absl::InvalidArgumentError("Reference embeddings are empty");
    }

    // Subsample if necessary
    if (config_.max_samples > 0 && embeddings.size() > config_.max_samples) {
        reference_embeddings_ = Subsample(embeddings, config_.max_samples);
    } else {
        reference_embeddings_ = embeddings;
    }

    // Compute or use provided sigma
    if (config_.rbf_sigma <= 0) {
        sigma_ = MedianHeuristic(reference_embeddings_);
        spdlog::debug("Computed sigma using median heuristic: {:.4f}", sigma_);
    } else {
        sigma_ = config_.rbf_sigma;
    }

    // Compute reference centroid
    reference_centroid_ = ComputeCentroid(reference_embeddings_);

    spdlog::info("MMD detector: Set reference with {} embeddings, dimension={}, sigma={:.4f}",
                 reference_embeddings_.size(),
                 reference_embeddings_[0].size(),
                 sigma_);

    return absl::OkStatus();
}

absl::StatusOr<DriftResult> MMDDriftDetector::Compute(
    const std::vector<DataPoint>& current_batch) {

    // Extract embeddings from DataPoints
    std::vector<std::vector<float>> embeddings;
    embeddings.reserve(current_batch.size());

    for (const auto& point : current_batch) {
        if (!point.embedding.empty()) {
            embeddings.push_back(point.embedding);
        }
    }

    if (embeddings.empty()) {
        return absl::InvalidArgumentError("No embeddings found in current batch");
    }

    return ComputeFromEmbeddings(embeddings);
}

absl::StatusOr<DriftResult> MMDDriftDetector::ComputeFromEmbeddings(
    const std::vector<std::vector<float>>& embeddings) {

    if (reference_embeddings_.empty()) {
        return absl::FailedPreconditionError("Reference embeddings not set");
    }

    if (embeddings.empty()) {
        return absl::InvalidArgumentError("Current embeddings are empty");
    }

    // Verify dimensions match
    if (embeddings[0].size() != reference_embeddings_[0].size()) {
        return absl::InvalidArgumentError(
            "Embedding dimensions don't match: current=" +
            std::to_string(embeddings[0].size()) + ", reference=" +
            std::to_string(reference_embeddings_[0].size()));
    }

    // Subsample if necessary
    std::vector<std::vector<float>> current;
    if (config_.max_samples > 0 && embeddings.size() > config_.max_samples) {
        current = Subsample(embeddings, config_.max_samples);
    } else {
        current = embeddings;
    }

    auto start_time = std::chrono::steady_clock::now();

    // Compute MMD^2
    double mmd_squared = ComputeMMDSquared(reference_embeddings_, current);
    double mmd = std::sqrt(std::max(0.0, mmd_squared));

    // Compute p-value using permutation test
    double p_value = 1.0;
    if (config_.num_permutations > 0) {
        p_value = PermutationTest(reference_embeddings_, current, mmd_squared);
    }

    // Compute centroid drift as additional metric
    double centroid_drift = ComputeCentroidDrift(embeddings);

    auto end_time = std::chrono::steady_clock::now();
    double compute_time_ms = std::chrono::duration<double, std::milli>(
        end_time - start_time).count();

    DriftResult result;
    result.type = DriftType::kEmbedding;
    result.score = mmd;
    result.threshold = config_.threshold;
    result.is_drifted = mmd > config_.threshold || p_value < config_.p_value_threshold;
    result.detected_at = std::chrono::system_clock::now();

    // Build explanation
    if (result.is_drifted) {
        result.explanation = "Embedding drift detected: MMD=" +
                             std::to_string(mmd) + " (threshold=" +
                             std::to_string(config_.threshold) + "), p-value=" +
                             std::to_string(p_value);
    } else {
        result.explanation = "Embedding distribution is within normal range";
    }

    result.metadata["mmd"] = std::to_string(mmd);
    result.metadata["mmd_squared"] = std::to_string(mmd_squared);
    result.metadata["p_value"] = std::to_string(p_value);
    result.metadata["centroid_drift"] = std::to_string(centroid_drift);
    result.metadata["reference_size"] = std::to_string(reference_embeddings_.size());
    result.metadata["current_size"] = std::to_string(current.size());
    result.metadata["sigma"] = std::to_string(sigma_);
    result.metadata["compute_time_ms"] = std::to_string(compute_time_ms);

    result.feature_scores["mmd"] = mmd;
    result.feature_scores["centroid_drift"] = centroid_drift;
    result.feature_scores["p_value"] = p_value;

    spdlog::debug("MMD computed: score={:.4f}, p_value={:.4f}, drifted={}, time={:.2f}ms",
                  mmd, p_value, result.is_drifted, compute_time_ms);

    return result;
}

double MMDDriftDetector::RBFKernel(const std::vector<float>& x,
                                    const std::vector<float>& y) const {
    double dist_sq = SquaredL2Distance(x, y);
    return std::exp(-dist_sq / (2.0 * sigma_ * sigma_));
}

double MMDDriftDetector::SquaredL2Distance(const std::vector<float>& x,
                                            const std::vector<float>& y) const {
    if (x.size() != y.size()) {
        return 0.0;
    }

    double sum = 0.0;
    for (size_t i = 0; i < x.size(); ++i) {
        double diff = static_cast<double>(x[i]) - static_cast<double>(y[i]);
        sum += diff * diff;
    }
    return sum;
}

double MMDDriftDetector::ComputeMMDSquared(
    const std::vector<std::vector<float>>& X,
    const std::vector<std::vector<float>>& Y) const {

    size_t m = X.size();
    size_t n = Y.size();

    if (m < 2 || n < 2) {
        return 0.0;
    }

    // Compute sum of k(x_i, x_j) for i != j
    double xx_sum = 0.0;
    for (size_t i = 0; i < m; ++i) {
        for (size_t j = i + 1; j < m; ++j) {
            xx_sum += 2.0 * RBFKernel(X[i], X[j]);
        }
    }

    // Compute sum of k(y_i, y_j) for i != j
    double yy_sum = 0.0;
    for (size_t i = 0; i < n; ++i) {
        for (size_t j = i + 1; j < n; ++j) {
            yy_sum += 2.0 * RBFKernel(Y[i], Y[j]);
        }
    }

    // Compute sum of k(x_i, y_j)
    double xy_sum = 0.0;
    for (size_t i = 0; i < m; ++i) {
        for (size_t j = 0; j < n; ++j) {
            xy_sum += RBFKernel(X[i], Y[j]);
        }
    }

    // Unbiased MMD^2 estimator
    double mmd_sq = xx_sum / (static_cast<double>(m) * (m - 1)) +
                    yy_sum / (static_cast<double>(n) * (n - 1)) -
                    2.0 * xy_sum / (static_cast<double>(m) * n);

    return mmd_sq;
}

double MMDDriftDetector::MedianHeuristic(
    const std::vector<std::vector<float>>& X) const {

    if (X.size() < 2) {
        return 1.0;
    }

    // Compute pairwise distances (subsample if too large)
    std::vector<double> distances;
    size_t max_pairs = 1000;  // Limit for performance
    size_t num_samples = std::min(X.size(), static_cast<size_t>(100));

    for (size_t i = 0; i < num_samples && distances.size() < max_pairs; ++i) {
        for (size_t j = i + 1; j < num_samples && distances.size() < max_pairs; ++j) {
            double dist = std::sqrt(SquaredL2Distance(X[i], X[j]));
            distances.push_back(dist);
        }
    }

    if (distances.empty()) {
        return 1.0;
    }

    // Find median
    std::sort(distances.begin(), distances.end());
    double median = distances[distances.size() / 2];

    // Return sigma such that median distance gives kernel value ~0.5
    // k(x,y) = exp(-d^2 / (2*sigma^2)) = 0.5
    // => sigma = d / sqrt(2*ln(2))
    return std::max(median / std::sqrt(2.0 * std::log(2.0)), 0.01);
}

double MMDDriftDetector::PermutationTest(
    const std::vector<std::vector<float>>& X,
    const std::vector<std::vector<float>>& Y,
    double observed_mmd) const {

    // Combine samples
    std::vector<std::vector<float>> combined;
    combined.reserve(X.size() + Y.size());
    combined.insert(combined.end(), X.begin(), X.end());
    combined.insert(combined.end(), Y.begin(), Y.end());

    size_t m = X.size();
    size_t count_greater = 0;

    std::vector<size_t> indices(combined.size());
    std::iota(indices.begin(), indices.end(), 0);

    for (size_t p = 0; p < config_.num_permutations; ++p) {
        // Shuffle indices
        std::shuffle(indices.begin(), indices.end(), rng_);

        // Split into two groups
        std::vector<std::vector<float>> perm_X, perm_Y;
        perm_X.reserve(m);
        perm_Y.reserve(combined.size() - m);

        for (size_t i = 0; i < m; ++i) {
            perm_X.push_back(combined[indices[i]]);
        }
        for (size_t i = m; i < combined.size(); ++i) {
            perm_Y.push_back(combined[indices[i]]);
        }

        // Compute MMD^2 for permuted samples
        double perm_mmd = ComputeMMDSquared(perm_X, perm_Y);

        if (perm_mmd >= observed_mmd) {
            count_greater++;
        }
    }

    // p-value is proportion of permutations with MMD >= observed
    return (static_cast<double>(count_greater) + 1.0) /
           (static_cast<double>(config_.num_permutations) + 1.0);
}

std::vector<float> MMDDriftDetector::ComputeCentroid(
    const std::vector<std::vector<float>>& embeddings) const {

    if (embeddings.empty()) {
        return {};
    }

    size_t dim = embeddings[0].size();
    std::vector<float> centroid(dim, 0.0f);

    for (const auto& emb : embeddings) {
        for (size_t i = 0; i < dim && i < emb.size(); ++i) {
            centroid[i] += emb[i];
        }
    }

    float n = static_cast<float>(embeddings.size());
    for (auto& val : centroid) {
        val /= n;
    }

    return centroid;
}

double MMDDriftDetector::CosineSimilarity(const std::vector<float>& a,
                                           const std::vector<float>& b) const {
    if (a.size() != b.size() || a.empty()) {
        return 0.0;
    }

    double dot = 0.0, norm_a = 0.0, norm_b = 0.0;
    for (size_t i = 0; i < a.size(); ++i) {
        dot += a[i] * b[i];
        norm_a += a[i] * a[i];
        norm_b += b[i] * b[i];
    }

    if (norm_a <= 0.0 || norm_b <= 0.0) {
        return 0.0;
    }

    return dot / (std::sqrt(norm_a) * std::sqrt(norm_b));
}

std::vector<float> MMDDriftDetector::GetReferenceCentroid() const {
    return reference_centroid_;
}

double MMDDriftDetector::ComputeCentroidDrift(
    const std::vector<std::vector<float>>& embeddings) {

    if (embeddings.empty() || reference_centroid_.empty()) {
        return 0.0;
    }

    std::vector<float> current_centroid = ComputeCentroid(embeddings);

    // Return cosine distance (1 - similarity)
    double similarity = CosineSimilarity(reference_centroid_, current_centroid);
    return 1.0 - similarity;
}

std::vector<std::vector<float>> MMDDriftDetector::Subsample(
    const std::vector<std::vector<float>>& embeddings,
    size_t max_samples) const {

    if (embeddings.size() <= max_samples) {
        return embeddings;
    }

    std::vector<size_t> indices(embeddings.size());
    std::iota(indices.begin(), indices.end(), 0);
    std::shuffle(indices.begin(), indices.end(), rng_);

    std::vector<std::vector<float>> subsampled;
    subsampled.reserve(max_samples);

    for (size_t i = 0; i < max_samples; ++i) {
        subsampled.push_back(embeddings[indices[i]]);
    }

    return subsampled;
}

absl::StatusOr<std::string> MMDDriftDetector::SerializeState() const {
    json state;
    state["threshold"] = config_.threshold;
    state["rbf_sigma"] = config_.rbf_sigma;
    state["computed_sigma"] = sigma_;
    state["num_permutations"] = config_.num_permutations;
    state["p_value_threshold"] = config_.p_value_threshold;
    state["reference_size"] = reference_embeddings_.size();

    if (!reference_centroid_.empty()) {
        state["reference_centroid"] = reference_centroid_;
    }

    // Don't serialize full embeddings (too large)
    // Just serialize summary statistics

    return state.dump();
}

absl::Status MMDDriftDetector::LoadState(std::string_view state) {
    try {
        json j = json::parse(state);
        config_.threshold = j.value("threshold", config_.threshold);
        config_.rbf_sigma = j.value("rbf_sigma", config_.rbf_sigma);
        sigma_ = j.value("computed_sigma", sigma_);
        config_.num_permutations = j.value("num_permutations", config_.num_permutations);
        config_.p_value_threshold = j.value("p_value_threshold", config_.p_value_threshold);

        if (j.contains("reference_centroid")) {
            reference_centroid_ = j["reference_centroid"].get<std::vector<float>>();
        }

        return absl::OkStatus();
    } catch (const json::exception& e) {
        return absl::InvalidArgumentError(
            std::string("Failed to parse MMD detector state: ") + e.what());
    }
}

// =============================================================================
// Factory Functions
// =============================================================================

std::unique_ptr<DriftDetector> CreateMMDDetector(MMDConfig config) {
    return std::make_unique<MMDDriftDetector>(std::move(config));
}

}  // namespace pyflare::drift
