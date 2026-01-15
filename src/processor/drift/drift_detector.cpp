#include "drift_detector.h"

#include <cmath>
#include <numeric>

namespace pyflare::drift {

std::string_view DriftTypeToString(DriftType type) {
    switch (type) {
        case DriftType::kFeature:
            return "feature";
        case DriftType::kEmbedding:
            return "embedding";
        case DriftType::kConcept:
            return "concept";
        case DriftType::kPrediction:
            return "prediction";
        default:
            return "unknown";
    }
}

void Distribution::AddSample(const std::vector<double>& sample) {
    samples_.push_back(sample);
}

void Distribution::AddSamples(const std::vector<DataPoint>& points) {
    for (const auto& point : points) {
        samples_.push_back(point.features);
    }
}

std::vector<double> Distribution::Mean() const {
    if (samples_.empty()) {
        return {};
    }

    size_t num_features = samples_[0].size();
    std::vector<double> mean(num_features, 0.0);

    for (const auto& sample : samples_) {
        for (size_t i = 0; i < num_features && i < sample.size(); ++i) {
            mean[i] += sample[i];
        }
    }

    for (auto& m : mean) {
        m /= static_cast<double>(samples_.size());
    }

    return mean;
}

std::vector<double> Distribution::StdDev() const {
    if (samples_.size() < 2) {
        return std::vector<double>(samples_.empty() ? 0 : samples_[0].size(), 0.0);
    }

    auto mean = Mean();
    size_t num_features = mean.size();
    std::vector<double> variance(num_features, 0.0);

    for (const auto& sample : samples_) {
        for (size_t i = 0; i < num_features && i < sample.size(); ++i) {
            double diff = sample[i] - mean[i];
            variance[i] += diff * diff;
        }
    }

    std::vector<double> stddev(num_features);
    for (size_t i = 0; i < num_features; ++i) {
        stddev[i] = std::sqrt(variance[i] / static_cast<double>(samples_.size() - 1));
    }

    return stddev;
}

void Distribution::Clear() {
    samples_.clear();
}

}  // namespace pyflare::drift
