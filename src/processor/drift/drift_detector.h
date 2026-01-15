#pragma once

/// @file drift_detector.h
/// @brief Base interface for drift detection in PyFlare

#include <chrono>
#include <memory>
#include <string>
#include <unordered_map>
#include <vector>

#include <absl/status/status.h>
#include <absl/status/statusor.h>

namespace pyflare::drift {

/// @brief Types of drift that can be detected
enum class DriftType {
    kFeature,      ///< Input distribution shift
    kEmbedding,    ///< Vector space shift
    kConcept,      ///< Input-output relationship change
    kPrediction    ///< Output distribution shift
};

/// @brief Convert drift type to string
std::string_view DriftTypeToString(DriftType type);

/// @brief Result of drift detection
struct DriftResult {
    DriftType type;
    double score;           ///< Drift score (0.0 - 1.0)
    double threshold;       ///< Threshold for drift detection
    bool is_drifted;        ///< Whether drift exceeds threshold
    std::string explanation;

    std::chrono::system_clock::time_point detected_at;

    /// @brief Per-feature drift scores (for feature drift)
    std::unordered_map<std::string, double> feature_scores;

    /// @brief Additional metadata
    std::unordered_map<std::string, std::string> metadata;
};

/// @brief A single data point for drift analysis
struct DataPoint {
    std::string id;
    std::vector<double> features;
    std::vector<float> embedding;
    std::unordered_map<std::string, std::string> attributes;
};

/// @brief Distribution representation for statistical tests
class Distribution {
public:
    Distribution() = default;

    /// @brief Add a sample to the distribution
    void AddSample(const std::vector<double>& sample);

    /// @brief Add samples from data points
    void AddSamples(const std::vector<DataPoint>& points);

    /// @brief Get the number of samples
    size_t Size() const { return samples_.size(); }

    /// @brief Get all samples
    const std::vector<std::vector<double>>& Samples() const { return samples_; }

    /// @brief Get feature means
    std::vector<double> Mean() const;

    /// @brief Get feature standard deviations
    std::vector<double> StdDev() const;

    /// @brief Clear all samples
    void Clear();

private:
    std::vector<std::vector<double>> samples_;
};

/// @brief Abstract base class for drift detectors
class DriftDetector {
public:
    virtual ~DriftDetector() = default;

    /// @brief Set the reference distribution (e.g., from training data)
    /// @param reference Reference distribution
    virtual absl::Status SetReference(const Distribution& reference) = 0;

    /// @brief Compute drift for a batch of data points
    /// @param current_batch Current data points to compare against reference
    /// @return Drift detection result
    virtual absl::StatusOr<DriftResult> Compute(
        const std::vector<DataPoint>& current_batch) = 0;

    /// @brief Get the drift type this detector handles
    virtual DriftType Type() const = 0;

    /// @brief Get the detector name
    virtual std::string Name() const = 0;

    /// @brief Serialize detector state for persistence
    virtual absl::StatusOr<std::string> SerializeState() const = 0;

    /// @brief Load detector state from serialized data
    virtual absl::Status LoadState(std::string_view state) = 0;

    /// @brief Get the current threshold
    virtual double GetThreshold() const = 0;

    /// @brief Set the detection threshold
    virtual void SetThreshold(double threshold) = 0;
};

/// @brief Factory for creating drift detectors
class DriftDetectorFactory {
public:
    /// @brief Create a feature drift detector
    static std::unique_ptr<DriftDetector> CreateFeatureDetector(double threshold = 0.1);

    /// @brief Create an embedding drift detector
    static std::unique_ptr<DriftDetector> CreateEmbeddingDetector(double threshold = 0.1);
};

}  // namespace pyflare::drift
