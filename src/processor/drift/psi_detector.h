#pragma once

/// @file psi_detector.h
/// @brief Population Stability Index (PSI) drift detector for PyFlare

#include <memory>
#include <string>
#include <vector>

#include "processor/drift/drift_detector.h"

namespace pyflare::drift {

/// @brief Configuration for PSI detector
struct PSIConfig {
    /// PSI threshold for drift detection (standard: 0.1 = low, 0.25 = significant)
    double threshold = 0.2;

    /// Number of bins for discretization
    size_t num_bins = 10;

    /// Minimum samples per bin to avoid instability
    size_t min_samples_per_bin = 5;

    /// Small constant to avoid log(0)
    double epsilon = 1e-10;
};

/// @brief Population Stability Index (PSI) drift detector
///
/// PSI measures how much a distribution has shifted between two samples.
/// It's commonly used for monitoring model stability in production.
///
/// Formula: PSI = sum((actual_% - expected_%) * ln(actual_% / expected_%))
///
/// Interpretation:
/// - PSI < 0.1: No significant shift
/// - 0.1 <= PSI < 0.25: Moderate shift, investigate
/// - PSI >= 0.25: Significant shift, action required
///
/// Example usage:
/// @code
///   PSIConfig config;
///   config.threshold = 0.2;
///   auto detector = std::make_unique<PSIDriftDetector>(config);
///
///   // Set reference from training data
///   detector->SetReference(training_distribution);
///
///   // Monitor production data
///   auto result = detector->Compute(production_batch);
///   if (result->is_drifted) {
///       // Alert or retrain
///   }
/// @endcode
class PSIDriftDetector : public DriftDetector {
public:
    explicit PSIDriftDetector(PSIConfig config = {});
    ~PSIDriftDetector() override = default;

    // Disable copy
    PSIDriftDetector(const PSIDriftDetector&) = delete;
    PSIDriftDetector& operator=(const PSIDriftDetector&) = delete;

    /// @brief Set reference distribution
    absl::Status SetReference(const Distribution& reference) override;

    /// @brief Compute PSI for current batch against reference
    absl::StatusOr<DriftResult> Compute(
        const std::vector<DataPoint>& current_batch) override;

    DriftType Type() const override { return DriftType::kFeature; }
    std::string Name() const override { return "PSIDriftDetector"; }

    absl::StatusOr<std::string> SerializeState() const override;
    absl::Status LoadState(std::string_view state) override;

    double GetThreshold() const override { return config_.threshold; }
    void SetThreshold(double threshold) override { config_.threshold = threshold; }

    /// @brief Get the current configuration
    const PSIConfig& GetConfig() const { return config_; }

    /// @brief Compute PSI for a single feature
    /// @param ref_values Reference values
    /// @param cur_values Current values
    /// @return PSI score for this feature
    double ComputeFeaturePSI(const std::vector<double>& ref_values,
                             const std::vector<double>& cur_values) const;

private:
    /// @brief Build histogram bins from reference data
    void BuildBins(const std::vector<double>& values,
                   std::vector<double>& bin_edges,
                   std::vector<double>& bin_percentages) const;

    /// @brief Assign values to bins and compute percentages
    std::vector<double> AssignToBins(const std::vector<double>& values,
                                      const std::vector<double>& bin_edges) const;

    PSIConfig config_;
    Distribution reference_;
    std::vector<std::vector<double>> reference_bin_edges_;
    std::vector<std::vector<double>> reference_bin_percentages_;
};

/// @brief Chi-squared drift detector for categorical features
///
/// Uses Pearson's chi-squared test to detect changes in categorical
/// distributions.
class ChiSquaredDriftDetector : public DriftDetector {
public:
    struct Config {
        double p_value_threshold = 0.05;  // Statistical significance level
        size_t min_expected_frequency = 5;  // Minimum expected count per category
    };

    explicit ChiSquaredDriftDetector(Config config = {});
    ~ChiSquaredDriftDetector() override = default;

    absl::Status SetReference(const Distribution& reference) override;
    absl::StatusOr<DriftResult> Compute(
        const std::vector<DataPoint>& current_batch) override;

    DriftType Type() const override { return DriftType::kFeature; }
    std::string Name() const override { return "ChiSquaredDriftDetector"; }

    absl::StatusOr<std::string> SerializeState() const override;
    absl::Status LoadState(std::string_view state) override;

    double GetThreshold() const override { return config_.p_value_threshold; }
    void SetThreshold(double threshold) override { config_.p_value_threshold = threshold; }

private:
    /// @brief Compute chi-squared statistic
    double ComputeChiSquared(const std::vector<double>& observed,
                             const std::vector<double>& expected) const;

    /// @brief Compute p-value from chi-squared statistic and degrees of freedom
    double ChiSquaredPValue(double chi_squared, int df) const;

    Config config_;
    Distribution reference_;
    std::unordered_map<std::string, std::unordered_map<std::string, double>> reference_category_counts_;
};

/// @brief Factory method to create PSI detector
std::unique_ptr<DriftDetector> CreatePSIDetector(PSIConfig config = {});

/// @brief Factory method to create Chi-squared detector
std::unique_ptr<DriftDetector> CreateChiSquaredDetector(
    ChiSquaredDriftDetector::Config config = {});

}  // namespace pyflare::drift
