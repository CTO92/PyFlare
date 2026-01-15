/// @file psi_detector.cpp
/// @brief PSI drift detector implementation

#include "processor/drift/psi_detector.h"

#include <algorithm>
#include <cmath>
#include <numeric>

#include <nlohmann/json.hpp>
#include <spdlog/spdlog.h>

namespace pyflare::drift {

using json = nlohmann::json;

// =============================================================================
// PSIDriftDetector Implementation
// =============================================================================

PSIDriftDetector::PSIDriftDetector(PSIConfig config)
    : config_(std::move(config)) {}

absl::Status PSIDriftDetector::SetReference(const Distribution& reference) {
    if (reference.Size() == 0) {
        return absl::InvalidArgumentError("Reference distribution is empty");
    }

    reference_ = reference;

    // Pre-compute bin edges and percentages for each feature
    const auto& samples = reference_.Samples();
    if (samples.empty() || samples[0].empty()) {
        return absl::InvalidArgumentError("Reference samples have no features");
    }

    size_t num_features = samples[0].size();
    reference_bin_edges_.resize(num_features);
    reference_bin_percentages_.resize(num_features);

    for (size_t f = 0; f < num_features; ++f) {
        std::vector<double> feature_values;
        feature_values.reserve(samples.size());

        for (const auto& sample : samples) {
            if (f < sample.size()) {
                feature_values.push_back(sample[f]);
            }
        }

        BuildBins(feature_values, reference_bin_edges_[f],
                  reference_bin_percentages_[f]);
    }

    spdlog::info("PSI detector: Set reference with {} samples, {} features",
                 reference_.Size(), num_features);
    return absl::OkStatus();
}

absl::StatusOr<DriftResult> PSIDriftDetector::Compute(
    const std::vector<DataPoint>& current_batch) {

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

    const auto& ref_samples = reference_.Samples();
    const auto& cur_samples = current.Samples();

    if (ref_samples.empty() || cur_samples.empty()) {
        return absl::InternalError("Empty samples");
    }

    size_t num_features = ref_samples[0].size();
    std::unordered_map<std::string, double> feature_scores;
    double max_psi = 0.0;
    double total_psi = 0.0;

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

        double psi = ComputeFeaturePSI(ref_values, cur_values);
        std::string feature_name = "feature_" + std::to_string(f);
        feature_scores[feature_name] = psi;
        max_psi = std::max(max_psi, psi);
        total_psi += psi;
    }

    // Use max PSI as the overall score
    double overall_psi = max_psi;

    DriftResult result;
    result.type = DriftType::kFeature;
    result.score = overall_psi;
    result.threshold = config_.threshold;
    result.is_drifted = overall_psi >= config_.threshold;
    result.detected_at = std::chrono::system_clock::now();
    result.feature_scores = std::move(feature_scores);

    // Build explanation based on PSI value
    if (overall_psi < 0.1) {
        result.explanation = "No significant population shift (PSI < 0.1)";
    } else if (overall_psi < 0.25) {
        result.explanation = "Moderate population shift detected (0.1 <= PSI < 0.25), "
                             "investigation recommended";
    } else {
        result.explanation = "Significant population shift detected (PSI >= 0.25), "
                             "action required";
    }

    result.metadata["reference_size"] = std::to_string(reference_.Size());
    result.metadata["current_size"] = std::to_string(current.Size());
    result.metadata["num_features"] = std::to_string(num_features);
    result.metadata["max_psi"] = std::to_string(max_psi);
    result.metadata["total_psi"] = std::to_string(total_psi);
    result.metadata["num_bins"] = std::to_string(config_.num_bins);

    spdlog::debug("PSI computed: score={:.4f}, threshold={:.4f}, drifted={}",
                  overall_psi, config_.threshold, result.is_drifted);

    return result;
}

double PSIDriftDetector::ComputeFeaturePSI(
    const std::vector<double>& ref_values,
    const std::vector<double>& cur_values) const {

    if (ref_values.empty() || cur_values.empty()) {
        return 0.0;
    }

    // Build bins from reference data
    std::vector<double> bin_edges;
    std::vector<double> ref_percentages;
    BuildBins(ref_values, bin_edges, ref_percentages);

    // Assign current values to same bins
    std::vector<double> cur_percentages = AssignToBins(cur_values, bin_edges);

    // Compute PSI
    double psi = 0.0;
    for (size_t i = 0; i < ref_percentages.size(); ++i) {
        double expected = ref_percentages[i];
        double actual = cur_percentages[i];

        // Add epsilon to avoid log(0)
        expected = std::max(expected, config_.epsilon);
        actual = std::max(actual, config_.epsilon);

        psi += (actual - expected) * std::log(actual / expected);
    }

    return psi;
}

void PSIDriftDetector::BuildBins(
    const std::vector<double>& values,
    std::vector<double>& bin_edges,
    std::vector<double>& bin_percentages) const {

    if (values.empty()) {
        return;
    }

    // Sort values to compute percentiles
    std::vector<double> sorted = values;
    std::sort(sorted.begin(), sorted.end());

    // Create bin edges based on percentiles
    bin_edges.clear();
    bin_edges.push_back(sorted.front() - 1.0);  // Include minimum

    for (size_t i = 1; i < config_.num_bins; ++i) {
        double percentile = static_cast<double>(i) / config_.num_bins;
        size_t idx = static_cast<size_t>(percentile * (sorted.size() - 1));
        bin_edges.push_back(sorted[idx]);
    }

    bin_edges.push_back(sorted.back() + 1.0);  // Include maximum

    // Compute percentages
    bin_percentages = AssignToBins(values, bin_edges);
}

std::vector<double> PSIDriftDetector::AssignToBins(
    const std::vector<double>& values,
    const std::vector<double>& bin_edges) const {

    if (values.empty() || bin_edges.size() < 2) {
        return std::vector<double>(config_.num_bins, 1.0 / config_.num_bins);
    }

    std::vector<size_t> bin_counts(bin_edges.size() - 1, 0);

    for (double val : values) {
        // Find the bin for this value using binary search
        auto it = std::upper_bound(bin_edges.begin(), bin_edges.end(), val);
        size_t bin_idx = static_cast<size_t>(
            std::distance(bin_edges.begin(), it)) - 1;
        bin_idx = std::min(bin_idx, bin_counts.size() - 1);
        bin_counts[bin_idx]++;
    }

    // Convert to percentages
    std::vector<double> percentages(bin_counts.size());
    double n = static_cast<double>(values.size());

    for (size_t i = 0; i < bin_counts.size(); ++i) {
        percentages[i] = static_cast<double>(bin_counts[i]) / n;
    }

    return percentages;
}

absl::StatusOr<std::string> PSIDriftDetector::SerializeState() const {
    json state;
    state["threshold"] = config_.threshold;
    state["num_bins"] = config_.num_bins;
    state["reference_size"] = reference_.Size();

    // Serialize bin edges and percentages
    state["bin_edges"] = reference_bin_edges_;
    state["bin_percentages"] = reference_bin_percentages_;

    return state.dump();
}

absl::Status PSIDriftDetector::LoadState(std::string_view state) {
    try {
        json j = json::parse(state);
        config_.threshold = j.value("threshold", config_.threshold);
        config_.num_bins = j.value("num_bins", config_.num_bins);

        if (j.contains("bin_edges")) {
            reference_bin_edges_ = j["bin_edges"].get<std::vector<std::vector<double>>>();
        }
        if (j.contains("bin_percentages")) {
            reference_bin_percentages_ = j["bin_percentages"].get<std::vector<std::vector<double>>>();
        }

        return absl::OkStatus();
    } catch (const json::exception& e) {
        return absl::InvalidArgumentError(
            std::string("Failed to parse PSI detector state: ") + e.what());
    }
}

// =============================================================================
// ChiSquaredDriftDetector Implementation
// =============================================================================

ChiSquaredDriftDetector::ChiSquaredDriftDetector(Config config)
    : config_(std::move(config)) {}

absl::Status ChiSquaredDriftDetector::SetReference(const Distribution& reference) {
    if (reference.Size() == 0) {
        return absl::InvalidArgumentError("Reference distribution is empty");
    }

    reference_ = reference;
    reference_category_counts_.clear();

    // For categorical features, we would track category frequencies
    // This is a placeholder - real implementation would extract categories
    // from DataPoint attributes

    spdlog::info("Chi-squared detector: Set reference with {} samples",
                 reference_.Size());
    return absl::OkStatus();
}

absl::StatusOr<DriftResult> ChiSquaredDriftDetector::Compute(
    const std::vector<DataPoint>& current_batch) {

    if (reference_.Size() == 0) {
        return absl::FailedPreconditionError("Reference distribution not set");
    }

    if (current_batch.empty()) {
        return absl::InvalidArgumentError("Current batch is empty");
    }

    // Placeholder implementation
    // Real implementation would:
    // 1. Extract categorical features from current batch
    // 2. Compare frequencies against reference
    // 3. Compute chi-squared statistic
    // 4. Calculate p-value

    DriftResult result;
    result.type = DriftType::kFeature;
    result.score = 0.0;  // p-value
    result.threshold = config_.p_value_threshold;
    result.is_drifted = false;
    result.detected_at = std::chrono::system_clock::now();
    result.explanation = "Chi-squared test not yet implemented for categorical features";

    return result;
}

double ChiSquaredDriftDetector::ComputeChiSquared(
    const std::vector<double>& observed,
    const std::vector<double>& expected) const {

    if (observed.size() != expected.size() || observed.empty()) {
        return 0.0;
    }

    double chi_sq = 0.0;
    for (size_t i = 0; i < observed.size(); ++i) {
        if (expected[i] > config_.min_expected_frequency) {
            double diff = observed[i] - expected[i];
            chi_sq += (diff * diff) / expected[i];
        }
    }

    return chi_sq;
}

double ChiSquaredDriftDetector::ChiSquaredPValue(double chi_squared, int df) const {
    // Simplified p-value approximation using Wilson-Hilferty transformation
    // For production, use a proper statistical library

    if (df <= 0 || chi_squared <= 0) {
        return 1.0;
    }

    double k = static_cast<double>(df);
    double z = std::pow(chi_squared / k, 1.0 / 3.0) -
               (1.0 - 2.0 / (9.0 * k));
    z /= std::sqrt(2.0 / (9.0 * k));

    // Normal CDF approximation
    double p = 0.5 * std::erfc(-z / std::sqrt(2.0));
    return 1.0 - p;
}

absl::StatusOr<std::string> ChiSquaredDriftDetector::SerializeState() const {
    json state;
    state["p_value_threshold"] = config_.p_value_threshold;
    state["min_expected_frequency"] = config_.min_expected_frequency;
    state["reference_size"] = reference_.Size();
    return state.dump();
}

absl::Status ChiSquaredDriftDetector::LoadState(std::string_view state) {
    try {
        json j = json::parse(state);
        config_.p_value_threshold = j.value("p_value_threshold", config_.p_value_threshold);
        config_.min_expected_frequency = j.value("min_expected_frequency",
                                                   config_.min_expected_frequency);
        return absl::OkStatus();
    } catch (const json::exception& e) {
        return absl::InvalidArgumentError(
            std::string("Failed to parse Chi-squared detector state: ") + e.what());
    }
}

// =============================================================================
// Factory Functions
// =============================================================================

std::unique_ptr<DriftDetector> CreatePSIDetector(PSIConfig config) {
    return std::make_unique<PSIDriftDetector>(std::move(config));
}

std::unique_ptr<DriftDetector> CreateChiSquaredDetector(
    ChiSquaredDriftDetector::Config config) {
    return std::make_unique<ChiSquaredDriftDetector>(std::move(config));
}

}  // namespace pyflare::drift
