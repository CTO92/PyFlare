#include "drift_detector.h"

#include <cmath>
#include <numeric>

#include "common/logging.h"

namespace pyflare::drift {

/// @brief Embedding drift detector using cosine similarity
class EmbeddingDriftDetector : public DriftDetector {
public:
    explicit EmbeddingDriftDetector(double threshold = 0.1)
        : threshold_(threshold) {}

    absl::Status SetReference(const Distribution& reference) override {
        reference_embeddings_.clear();

        for (const auto& sample : reference.Samples()) {
            std::vector<float> embedding(sample.begin(), sample.end());
            reference_embeddings_.push_back(std::move(embedding));
        }

        PYFLARE_LOG_INFO("Set reference with {} embeddings", reference_embeddings_.size());
        return absl::OkStatus();
    }

    absl::StatusOr<DriftResult> Compute(
        const std::vector<DataPoint>& current_batch) override {

        if (reference_embeddings_.empty()) {
            return absl::FailedPreconditionError("Reference distribution not set");
        }

        if (current_batch.empty()) {
            return absl::InvalidArgumentError("Current batch is empty");
        }

        // Compute average cosine similarity between current and reference
        double total_similarity = 0.0;
        size_t count = 0;

        for (const auto& point : current_batch) {
            if (point.embedding.empty()) {
                continue;
            }

            // Find average similarity to reference embeddings
            double point_similarity = 0.0;
            for (const auto& ref : reference_embeddings_) {
                point_similarity += CosineSimilarity(point.embedding, ref);
            }
            point_similarity /= static_cast<double>(reference_embeddings_.size());

            total_similarity += point_similarity;
            ++count;
        }

        if (count == 0) {
            return absl::InvalidArgumentError("No valid embeddings in current batch");
        }

        double avg_similarity = total_similarity / static_cast<double>(count);
        double drift_score = 1.0 - avg_similarity;  // Convert similarity to drift

        DriftResult result;
        result.type = DriftType::kEmbedding;
        result.score = std::clamp(drift_score, 0.0, 1.0);
        result.threshold = threshold_;
        result.is_drifted = result.score > threshold_;
        result.detected_at = std::chrono::system_clock::now();

        if (result.is_drifted) {
            result.explanation = "Embedding drift detected: average cosine similarity "
                               "dropped below threshold";
        } else {
            result.explanation = "Embeddings are within normal range";
        }

        result.metadata["avg_similarity"] = std::to_string(avg_similarity);
        result.metadata["sample_count"] = std::to_string(count);

        return result;
    }

    DriftType Type() const override { return DriftType::kEmbedding; }
    std::string Name() const override { return "EmbeddingDriftDetector"; }

    absl::StatusOr<std::string> SerializeState() const override {
        // Placeholder - would serialize reference embeddings
        return "{}";
    }

    absl::Status LoadState(std::string_view state) override {
        // Placeholder
        return absl::OkStatus();
    }

    double GetThreshold() const override { return threshold_; }
    void SetThreshold(double threshold) override { threshold_ = threshold; }

private:
    static double CosineSimilarity(const std::vector<float>& a, const std::vector<float>& b) {
        if (a.size() != b.size() || a.empty()) {
            return 0.0;
        }

        double dot_product = 0.0;
        double norm_a = 0.0;
        double norm_b = 0.0;

        for (size_t i = 0; i < a.size(); ++i) {
            dot_product += static_cast<double>(a[i]) * static_cast<double>(b[i]);
            norm_a += static_cast<double>(a[i]) * static_cast<double>(a[i]);
            norm_b += static_cast<double>(b[i]) * static_cast<double>(b[i]);
        }

        double denom = std::sqrt(norm_a) * std::sqrt(norm_b);
        if (denom < 1e-10) {
            return 0.0;
        }

        return dot_product / denom;
    }

    double threshold_;
    std::vector<std::vector<float>> reference_embeddings_;
};

std::unique_ptr<DriftDetector> DriftDetectorFactory::CreateEmbeddingDetector(double threshold) {
    return std::make_unique<EmbeddingDriftDetector>(threshold);
}

}  // namespace pyflare::drift
