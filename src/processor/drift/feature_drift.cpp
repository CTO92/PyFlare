#include "drift_detector.h"

#include <algorithm>
#include <cmath>

#include "common/logging.h"

namespace pyflare::drift {

/// @brief Feature drift detector using Kolmogorov-Smirnov test
class FeatureDriftDetector : public DriftDetector {
public:
    explicit FeatureDriftDetector(double threshold = 0.1)
        : threshold_(threshold) {}

    absl::Status SetReference(const Distribution& reference) override {
        reference_ = reference;
        PYFLARE_LOG_INFO("Set reference distribution with {} samples", reference_.Size());
        return absl::OkStatus();
    }

    absl::StatusOr<DriftResult> Compute(
        const std::vector<DataPoint>& current_batch) override {

        if (reference_.Size() == 0) {
            return absl::FailedPreconditionError("Reference distribution not set");
        }

        if (current_batch.empty()) {
            return absl::InvalidArgumentError("Current batch is empty");
        }

        // Build current distribution
        Distribution current;
        current.AddSamples(current_batch);

        if (current.Size() == 0) {
            return absl::InvalidArgumentError("No valid samples in current batch");
        }

        // Perform KS test for each feature
        const auto& ref_samples = reference_.Samples();
        const auto& cur_samples = current.Samples();

        if (ref_samples.empty() || cur_samples.empty()) {
            return absl::InternalError("Empty samples");
        }

        size_t num_features = ref_samples[0].size();
        std::unordered_map<std::string, double> feature_scores;
        double max_ks_stat = 0.0;

        for (size_t f = 0; f < num_features; ++f) {
            // Extract feature values
            std::vector<double> ref_values, cur_values;
            ref_values.reserve(ref_samples.size());
            cur_values.reserve(cur_samples.size());

            for (const auto& sample : ref_samples) {
                if (f < sample.size()) {
                    ref_values.push_back(sample[f]);
                }
            }
            for (const auto& sample : cur_samples) {
                if (f < sample.size()) {
                    cur_values.push_back(sample[f]);
                }
            }

            double ks_stat = KolmogorovSmirnovStatistic(ref_values, cur_values);
            std::string feature_name = "feature_" + std::to_string(f);
            feature_scores[feature_name] = ks_stat;
            max_ks_stat = std::max(max_ks_stat, ks_stat);
        }

        DriftResult result;
        result.type = DriftType::kFeature;
        result.score = max_ks_stat;
        result.threshold = threshold_;
        result.is_drifted = result.score > threshold_;
        result.detected_at = std::chrono::system_clock::now();
        result.feature_scores = std::move(feature_scores);

        if (result.is_drifted) {
            result.explanation = "Feature drift detected: KS statistic exceeds threshold";
        } else {
            result.explanation = "Feature distributions are within normal range";
        }

        result.metadata["reference_size"] = std::to_string(reference_.Size());
        result.metadata["current_size"] = std::to_string(current.Size());
        result.metadata["num_features"] = std::to_string(num_features);

        return result;
    }

    DriftType Type() const override { return DriftType::kFeature; }
    std::string Name() const override { return "FeatureDriftDetector"; }

    absl::StatusOr<std::string> SerializeState() const override {
        // Placeholder
        return "{}";
    }

    absl::Status LoadState(std::string_view state) override {
        // Placeholder
        return absl::OkStatus();
    }

    double GetThreshold() const override { return threshold_; }
    void SetThreshold(double threshold) override { threshold_ = threshold; }

private:
    /// @brief Compute Kolmogorov-Smirnov statistic
    static double KolmogorovSmirnovStatistic(
        std::vector<double> sample1,
        std::vector<double> sample2) {

        if (sample1.empty() || sample2.empty()) {
            return 0.0;
        }

        std::sort(sample1.begin(), sample1.end());
        std::sort(sample2.begin(), sample2.end());

        double n1 = static_cast<double>(sample1.size());
        double n2 = static_cast<double>(sample2.size());

        size_t i = 0, j = 0;
        double d = 0.0;

        while (i < sample1.size() && j < sample2.size()) {
            double cdf1 = static_cast<double>(i + 1) / n1;
            double cdf2 = static_cast<double>(j + 1) / n2;

            if (sample1[i] <= sample2[j]) {
                d = std::max(d, std::abs(cdf1 - static_cast<double>(j) / n2));
                ++i;
            } else {
                d = std::max(d, std::abs(static_cast<double>(i) / n1 - cdf2));
                ++j;
            }
        }

        // Handle remaining elements
        while (i < sample1.size()) {
            double cdf1 = static_cast<double>(i + 1) / n1;
            d = std::max(d, std::abs(cdf1 - 1.0));
            ++i;
        }
        while (j < sample2.size()) {
            double cdf2 = static_cast<double>(j + 1) / n2;
            d = std::max(d, std::abs(1.0 - cdf2));
            ++j;
        }

        return d;
    }

    double threshold_;
    Distribution reference_;
};

std::unique_ptr<DriftDetector> DriftDetectorFactory::CreateFeatureDetector(double threshold) {
    return std::make_unique<FeatureDriftDetector>(threshold);
}

}  // namespace pyflare::drift
