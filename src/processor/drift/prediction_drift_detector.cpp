/// @file prediction_drift_detector.cpp
/// @brief Implementation of prediction/output drift detection

#include "processor/drift/prediction_drift_detector.h"

#include <algorithm>
#include <cmath>
#include <numeric>
#include <sstream>

#include <nlohmann/json.hpp>
#include <spdlog/spdlog.h>

namespace pyflare::drift {

using json = nlohmann::json;

// =============================================================================
// PredictionDriftDetector Implementation
// =============================================================================

PredictionDriftDetector::PredictionDriftDetector(PredictionDriftConfig config)
    : config_(std::move(config)) {}

PredictionDriftDetector::~PredictionDriftDetector() = default;

absl::Status PredictionDriftDetector::SetReference(const Distribution& reference) {
    std::lock_guard<std::mutex> lock(mutex_);

    reference_.is_set = true;
    reference_.sample_count = reference.sample_count;

    // Store based on distribution type
    if (!reference.numeric_features.empty()) {
        // Use first numeric feature as output values
        auto it = reference.numeric_features.begin();
        reference_.values = it->second;
        reference_.type = OutputType::kRegression;

        // Compute statistics
        if (!reference_.values.empty()) {
            double sum = std::accumulate(reference_.values.begin(),
                                         reference_.values.end(), 0.0);
            reference_.mean = sum / reference_.values.size();

            double sq_sum = 0.0;
            for (double v : reference_.values) {
                sq_sum += (v - reference_.mean) * (v - reference_.mean);
            }
            reference_.std_dev = std::sqrt(sq_sum / reference_.values.size());

            auto [min_it, max_it] = std::minmax_element(
                reference_.values.begin(), reference_.values.end());
            reference_.min = *min_it;
            reference_.max = *max_it;

            // Compute histogram
            auto [bins, counts] = ComputeHistogram(reference_.values, config_.num_bins);
            reference_.histogram_bins = bins;
            reference_.histogram_counts = counts;
        }
    }

    if (!reference.categorical_features.empty()) {
        // Use first categorical feature as class labels
        auto it = reference.categorical_features.begin();
        reference_.class_distribution = ComputeClassDistribution(it->second);
        reference_.type = OutputType::kClassification;

        for (const auto& [cls, _] : reference_.class_distribution) {
            reference_.class_names.push_back(cls);
        }
    }

    return absl::OkStatus();
}

absl::StatusOr<DriftResult> PredictionDriftDetector::Compute(
    const std::vector<DataPoint>& current_batch) {

    if (!HasReference()) {
        return absl::FailedPreconditionError("Reference not set");
    }

    // Extract outputs from data points based on type
    if (reference_.type == OutputType::kClassification) {
        std::vector<std::string> classes;
        for (const auto& dp : current_batch) {
            auto it = dp.attributes.find("output");
            if (it != dp.attributes.end()) {
                classes.push_back(it->second);
            } else {
                auto label_it = dp.attributes.find("class");
                if (label_it != dp.attributes.end()) {
                    classes.push_back(label_it->second);
                }
            }
        }

        auto result = ComputeClassDrift(classes);
        if (!result.ok()) {
            return result.status();
        }

        DriftResult drift_result;
        drift_result.type = DriftType::kPrediction;
        drift_result.score = result->drift_score;
        drift_result.threshold = config_.threshold;
        drift_result.is_drifted = result->drift_detected;
        drift_result.explanation = result->explanation;
        drift_result.detected_at = std::chrono::system_clock::now();
        return drift_result;
    }

    if (reference_.type == OutputType::kRegression) {
        std::vector<double> values;
        for (const auto& dp : current_batch) {
            auto it = dp.attributes.find("output");
            if (it != dp.attributes.end()) {
                try {
                    values.push_back(std::stod(it->second));
                } catch (...) {}
            } else if (!dp.values.empty()) {
                values.push_back(dp.values[0]);
            }
        }

        auto result = ComputeValueDrift(values);
        if (!result.ok()) {
            return result.status();
        }

        DriftResult drift_result;
        drift_result.type = DriftType::kPrediction;
        drift_result.score = result->drift_score;
        drift_result.threshold = config_.threshold;
        drift_result.is_drifted = result->drift_detected;
        drift_result.explanation = result->explanation;
        drift_result.detected_at = std::chrono::system_clock::now();
        return drift_result;
    }

    return absl::UnimplementedError("Output type not supported for batch compute");
}

// =============================================================================
// Reference Setting Methods
// =============================================================================

absl::Status PredictionDriftDetector::SetReferenceTexts(
    const std::vector<std::string>& outputs) {

    if (outputs.empty()) {
        return absl::InvalidArgumentError("Empty outputs");
    }

    std::lock_guard<std::mutex> lock(mutex_);

    reference_.is_set = true;
    reference_.type = OutputType::kText;
    reference_.sample_count = outputs.size();

    // Compute length statistics
    double sum_len = 0.0;
    for (const auto& text : outputs) {
        sum_len += text.length();
    }
    reference_.avg_length = sum_len / outputs.size();

    double sq_sum = 0.0;
    for (const auto& text : outputs) {
        double diff = text.length() - reference_.avg_length;
        sq_sum += diff * diff;
    }
    reference_.std_length = std::sqrt(sq_sum / outputs.size());

    spdlog::info("PredictionDrift: Set text reference with {} samples, "
                 "avg length: {:.1f}", outputs.size(), reference_.avg_length);

    return absl::OkStatus();
}

absl::Status PredictionDriftDetector::SetReferenceClasses(
    const std::vector<std::string>& classes) {

    if (classes.empty()) {
        return absl::InvalidArgumentError("Empty classes");
    }

    std::lock_guard<std::mutex> lock(mutex_);

    reference_.is_set = true;
    reference_.type = OutputType::kClassification;
    reference_.sample_count = classes.size();
    reference_.class_distribution = ComputeClassDistribution(classes);

    reference_.class_names.clear();
    for (const auto& [cls, _] : reference_.class_distribution) {
        reference_.class_names.push_back(cls);
    }

    spdlog::info("PredictionDrift: Set class reference with {} samples, "
                 "{} classes", classes.size(), reference_.class_names.size());

    return absl::OkStatus();
}

absl::Status PredictionDriftDetector::SetReferenceValues(
    const std::vector<double>& values) {

    if (values.empty()) {
        return absl::InvalidArgumentError("Empty values");
    }

    std::lock_guard<std::mutex> lock(mutex_);

    reference_.is_set = true;
    reference_.type = OutputType::kRegression;
    reference_.values = values;
    reference_.sample_count = values.size();

    // Compute statistics
    double sum = std::accumulate(values.begin(), values.end(), 0.0);
    reference_.mean = sum / values.size();

    double sq_sum = 0.0;
    for (double v : values) {
        sq_sum += (v - reference_.mean) * (v - reference_.mean);
    }
    reference_.std_dev = std::sqrt(sq_sum / values.size());

    auto [min_it, max_it] = std::minmax_element(values.begin(), values.end());
    reference_.min = *min_it;
    reference_.max = *max_it;

    // Compute histogram
    auto [bins, counts] = ComputeHistogram(values, config_.num_bins);
    reference_.histogram_bins = bins;
    reference_.histogram_counts = counts;

    spdlog::info("PredictionDrift: Set value reference with {} samples, "
                 "mean: {:.4f}, std: {:.4f}", values.size(),
                 reference_.mean, reference_.std_dev);

    return absl::OkStatus();
}

absl::Status PredictionDriftDetector::SetReferenceProbabilities(
    const std::vector<std::vector<double>>& probabilities,
    const std::vector<std::string>& class_names) {

    if (probabilities.empty()) {
        return absl::InvalidArgumentError("Empty probabilities");
    }

    std::lock_guard<std::mutex> lock(mutex_);

    reference_.is_set = true;
    reference_.type = OutputType::kProbability;
    reference_.sample_count = probabilities.size();

    // Convert to class distribution by taking argmax
    reference_.class_distribution.clear();
    size_t num_classes = probabilities[0].size();

    if (!class_names.empty() && class_names.size() == num_classes) {
        reference_.class_names = class_names;
    } else {
        reference_.class_names.clear();
        for (size_t i = 0; i < num_classes; ++i) {
            reference_.class_names.push_back("class_" + std::to_string(i));
        }
    }

    for (const auto& name : reference_.class_names) {
        reference_.class_distribution[name] = 0.0;
    }

    for (const auto& probs : probabilities) {
        auto max_it = std::max_element(probs.begin(), probs.end());
        size_t max_idx = std::distance(probs.begin(), max_it);
        if (max_idx < reference_.class_names.size()) {
            reference_.class_distribution[reference_.class_names[max_idx]] += 1.0;
        }
    }

    // Normalize
    for (auto& [cls, count] : reference_.class_distribution) {
        count /= probabilities.size();
    }

    return absl::OkStatus();
}

absl::Status PredictionDriftDetector::SetReferenceEmbeddings(
    const std::vector<std::vector<float>>& embeddings) {

    if (embeddings.empty()) {
        return absl::InvalidArgumentError("Empty embeddings");
    }

    std::lock_guard<std::mutex> lock(mutex_);

    reference_.is_set = true;
    reference_.type = OutputType::kEmbedding;
    reference_.sample_count = embeddings.size();
    reference_.reference_embeddings = embeddings;
    reference_.centroid_embedding = ComputeCentroid(embeddings);

    spdlog::info("PredictionDrift: Set embedding reference with {} samples",
                 embeddings.size());

    return absl::OkStatus();
}

// =============================================================================
// Drift Computation Methods
// =============================================================================

absl::StatusOr<PredictionDriftResult> PredictionDriftDetector::ComputeTextDrift(
    const std::vector<std::string>& outputs) {

    std::lock_guard<std::mutex> lock(mutex_);

    if (!reference_.is_set) {
        return absl::FailedPreconditionError("Reference not set");
    }

    if (outputs.size() < config_.min_samples) {
        return absl::InvalidArgumentError(
            "Need at least " + std::to_string(config_.min_samples) + " samples");
    }

    PredictionDriftResult result;
    result.output_type = OutputType::kText;
    result.threshold = config_.threshold;

    // Compute length statistics for current
    double sum_len = 0.0;
    for (const auto& text : outputs) {
        sum_len += text.length();
    }
    double avg_length = sum_len / outputs.size();

    double sq_sum = 0.0;
    for (const auto& text : outputs) {
        double diff = text.length() - avg_length;
        sq_sum += diff * diff;
    }
    double std_length = std::sqrt(sq_sum / outputs.size());

    // Length drift (normalized difference)
    if (reference_.std_length > 0) {
        result.length_drift = std::abs(avg_length - reference_.avg_length) /
                              reference_.std_length;
    } else {
        result.length_drift = std::abs(avg_length - reference_.avg_length) /
                              (reference_.avg_length + 1.0);
    }

    // Overall drift score
    result.drift_score = result.length_drift;

    result.drift_detected = (result.drift_score > config_.threshold);

    if (result.drift_detected) {
        result.explanation = "Text output drift detected: length changed from " +
            std::to_string(reference_.avg_length) + " to " +
            std::to_string(avg_length);
        spdlog::warn("Prediction drift detected in text outputs");
    } else {
        result.explanation = "No significant text output drift";
    }

    return result;
}

absl::StatusOr<PredictionDriftResult> PredictionDriftDetector::ComputeClassDrift(
    const std::vector<std::string>& classes) {

    std::lock_guard<std::mutex> lock(mutex_);

    if (!reference_.is_set) {
        return absl::FailedPreconditionError("Reference not set");
    }

    if (classes.size() < config_.min_samples) {
        return absl::InvalidArgumentError(
            "Need at least " + std::to_string(config_.min_samples) + " samples");
    }

    PredictionDriftResult result;
    result.output_type = OutputType::kClassification;
    result.threshold = config_.threshold;

    // Compute current distribution
    result.class_distribution_current = ComputeClassDistribution(classes);
    result.class_distribution_reference = reference_.class_distribution;

    // Compute JS divergence (symmetric, bounded)
    result.class_distribution_divergence = ComputeJSDivergence(
        reference_.class_distribution, result.class_distribution_current);

    result.drift_score = result.class_distribution_divergence;

    // Find top changed classes
    for (const auto& [cls, ref_prob] : reference_.class_distribution) {
        double curr_prob = 0.0;
        auto it = result.class_distribution_current.find(cls);
        if (it != result.class_distribution_current.end()) {
            curr_prob = it->second;
        }
        double change = std::abs(curr_prob - ref_prob);
        result.top_changed_classes.push_back({cls, change});
    }

    // Check for new classes
    for (const auto& [cls, curr_prob] : result.class_distribution_current) {
        if (reference_.class_distribution.find(cls) ==
            reference_.class_distribution.end()) {
            result.top_changed_classes.push_back({cls, curr_prob});
        }
    }

    // Sort by change magnitude
    std::sort(result.top_changed_classes.begin(),
              result.top_changed_classes.end(),
              [](const auto& a, const auto& b) { return a.second > b.second; });

    // Keep top 5
    if (result.top_changed_classes.size() > 5) {
        result.top_changed_classes.resize(5);
    }

    result.drift_detected = (result.drift_score > config_.threshold);

    if (result.drift_detected) {
        std::string top_cls = result.top_changed_classes.empty() ?
            "unknown" : result.top_changed_classes[0].first;
        result.explanation = "Class distribution drift detected: JS divergence = " +
            std::to_string(result.class_distribution_divergence) +
            ", top changed class: " + top_cls;
        spdlog::warn("Prediction drift detected in class distribution");
    } else {
        result.explanation = "No significant class distribution drift";
    }

    return result;
}

absl::StatusOr<PredictionDriftResult> PredictionDriftDetector::ComputeValueDrift(
    const std::vector<double>& values) {

    std::lock_guard<std::mutex> lock(mutex_);

    if (!reference_.is_set) {
        return absl::FailedPreconditionError("Reference not set");
    }

    if (values.size() < config_.min_samples) {
        return absl::InvalidArgumentError(
            "Need at least " + std::to_string(config_.min_samples) + " samples");
    }

    PredictionDriftResult result;
    result.output_type = OutputType::kRegression;
    result.threshold = config_.threshold;

    // Compute current statistics
    double sum = std::accumulate(values.begin(), values.end(), 0.0);
    result.mean_current = sum / values.size();

    double sq_sum = 0.0;
    for (double v : values) {
        sq_sum += (v - result.mean_current) * (v - result.mean_current);
    }
    result.std_current = std::sqrt(sq_sum / values.size());

    result.mean_reference = reference_.mean;
    result.std_reference = reference_.std_dev;

    // Kolmogorov-Smirnov test
    result.ks_statistic = ComputeKSStatistic(reference_.values, values);

    // Use KS statistic as drift score
    result.drift_score = result.ks_statistic;

    // Approximate p-value using asymptotic distribution
    size_t n = reference_.values.size();
    size_t m = values.size();
    double en = std::sqrt((n * m) / static_cast<double>(n + m));
    double lambda = (en + 0.12 + 0.11 / en) * result.ks_statistic;

    // Approximate Kolmogorov distribution
    result.p_value = 2.0 * std::exp(-2.0 * lambda * lambda);

    result.drift_detected = (result.p_value < config_.p_value_threshold);

    if (result.drift_detected) {
        result.explanation = "Value distribution drift detected: KS = " +
            std::to_string(result.ks_statistic) + ", p-value = " +
            std::to_string(result.p_value) + ", mean changed from " +
            std::to_string(reference_.mean) + " to " +
            std::to_string(result.mean_current);
        spdlog::warn("Prediction drift detected in value distribution");
    } else {
        result.explanation = "No significant value distribution drift";
    }

    return result;
}

absl::StatusOr<PredictionDriftResult> PredictionDriftDetector::ComputeEmbeddingDrift(
    const std::vector<std::vector<float>>& embeddings) {

    std::lock_guard<std::mutex> lock(mutex_);

    if (!reference_.is_set) {
        return absl::FailedPreconditionError("Reference not set");
    }

    if (embeddings.empty()) {
        return absl::InvalidArgumentError("Empty embeddings");
    }

    PredictionDriftResult result;
    result.output_type = OutputType::kEmbedding;
    result.threshold = config_.threshold;

    // Compute centroid of current embeddings
    auto current_centroid = ComputeCentroid(embeddings);

    // Compute cosine similarity between centroids
    double centroid_sim = ComputeCosineSimilarity(
        reference_.centroid_embedding, current_centroid);

    // Convert to drift score (1 - similarity)
    result.embedding_drift = 1.0 - centroid_sim;
    result.drift_score = result.embedding_drift;

    result.drift_detected = (result.drift_score > config_.threshold);

    if (result.drift_detected) {
        result.explanation = "Embedding drift detected: centroid similarity = " +
            std::to_string(centroid_sim);
        spdlog::warn("Prediction drift detected in embeddings");
    } else {
        result.explanation = "No significant embedding drift";
    }

    return result;
}

// =============================================================================
// Reference Statistics
// =============================================================================

PredictionDriftDetector::ReferenceStats
PredictionDriftDetector::GetReferenceStats() const {
    std::lock_guard<std::mutex> lock(mutex_);

    ReferenceStats stats;
    stats.is_set = reference_.is_set;
    stats.type = reference_.type;
    stats.sample_count = reference_.sample_count;
    stats.avg_length = reference_.avg_length;
    stats.std_length = reference_.std_length;
    stats.centroid_embedding = reference_.centroid_embedding;
    stats.class_distribution = reference_.class_distribution;
    stats.class_names = reference_.class_names;
    stats.mean = reference_.mean;
    stats.std_dev = reference_.std_dev;
    stats.min = reference_.min;
    stats.max = reference_.max;
    stats.histogram_bins = reference_.histogram_bins;
    stats.histogram_counts = reference_.histogram_counts;

    return stats;
}

bool PredictionDriftDetector::HasReference() const {
    std::lock_guard<std::mutex> lock(mutex_);
    return reference_.is_set;
}

void PredictionDriftDetector::ClearReference() {
    std::lock_guard<std::mutex> lock(mutex_);
    reference_ = Reference{};
}

// =============================================================================
// Statistical Methods
// =============================================================================

double PredictionDriftDetector::ComputeKLDivergence(
    const std::unordered_map<std::string, double>& p,
    const std::unordered_map<std::string, double>& q) {

    const double epsilon = 1e-10;
    double kl = 0.0;

    for (const auto& [key, p_val] : p) {
        double q_val = epsilon;
        auto it = q.find(key);
        if (it != q.end()) {
            q_val = std::max(it->second, epsilon);
        }
        double p_smooth = std::max(p_val, epsilon);
        kl += p_smooth * std::log(p_smooth / q_val);
    }

    return kl;
}

double PredictionDriftDetector::ComputeJSDivergence(
    const std::unordered_map<std::string, double>& p,
    const std::unordered_map<std::string, double>& q) {

    // M = (P + Q) / 2
    std::unordered_map<std::string, double> m;

    for (const auto& [key, val] : p) {
        m[key] = val / 2.0;
    }
    for (const auto& [key, val] : q) {
        m[key] += val / 2.0;
    }

    // JS = (KL(P||M) + KL(Q||M)) / 2
    double js = (ComputeKLDivergence(p, m) + ComputeKLDivergence(q, m)) / 2.0;

    return js;
}

double PredictionDriftDetector::ComputeKSStatistic(
    const std::vector<double>& reference,
    const std::vector<double>& current) {

    if (reference.empty() || current.empty()) {
        return 0.0;
    }

    // Sort both samples
    std::vector<double> ref_sorted = reference;
    std::vector<double> curr_sorted = current;
    std::sort(ref_sorted.begin(), ref_sorted.end());
    std::sort(curr_sorted.begin(), curr_sorted.end());

    // Compute empirical CDFs and find max difference
    double max_diff = 0.0;

    size_t i = 0, j = 0;
    while (i < ref_sorted.size() && j < curr_sorted.size()) {
        double ref_cdf = static_cast<double>(i + 1) / ref_sorted.size();
        double curr_cdf = static_cast<double>(j + 1) / curr_sorted.size();

        if (ref_sorted[i] <= curr_sorted[j]) {
            max_diff = std::max(max_diff, std::abs(ref_cdf - curr_cdf));
            ++i;
        } else {
            max_diff = std::max(max_diff, std::abs(ref_cdf - curr_cdf));
            ++j;
        }
    }

    return max_diff;
}

double PredictionDriftDetector::ComputeTTest(
    const std::vector<double>& reference,
    const std::vector<double>& current) {

    if (reference.size() < 2 || current.size() < 2) {
        return 0.0;
    }

    // Compute means
    double ref_mean = std::accumulate(reference.begin(), reference.end(), 0.0) /
                      reference.size();
    double curr_mean = std::accumulate(current.begin(), current.end(), 0.0) /
                       current.size();

    // Compute variances
    double ref_var = 0.0, curr_var = 0.0;
    for (double v : reference) {
        ref_var += (v - ref_mean) * (v - ref_mean);
    }
    ref_var /= (reference.size() - 1);

    for (double v : current) {
        curr_var += (v - curr_mean) * (v - curr_mean);
    }
    curr_var /= (current.size() - 1);

    // Welch's t-test
    double se = std::sqrt(ref_var / reference.size() + curr_var / current.size());
    if (se < 1e-10) return 0.0;

    return std::abs(ref_mean - curr_mean) / se;
}

std::vector<float> PredictionDriftDetector::ComputeCentroid(
    const std::vector<std::vector<float>>& embeddings) {

    if (embeddings.empty()) {
        return {};
    }

    size_t dim = embeddings[0].size();
    std::vector<float> centroid(dim, 0.0f);

    for (const auto& emb : embeddings) {
        for (size_t i = 0; i < std::min(dim, emb.size()); ++i) {
            centroid[i] += emb[i];
        }
    }

    for (float& v : centroid) {
        v /= embeddings.size();
    }

    return centroid;
}

double PredictionDriftDetector::ComputeCosineSimilarity(
    const std::vector<float>& a,
    const std::vector<float>& b) {

    if (a.size() != b.size() || a.empty()) {
        return 0.0;
    }

    double dot = 0.0, norm_a = 0.0, norm_b = 0.0;
    for (size_t i = 0; i < a.size(); ++i) {
        dot += a[i] * b[i];
        norm_a += a[i] * a[i];
        norm_b += b[i] * b[i];
    }

    if (norm_a < 1e-10 || norm_b < 1e-10) {
        return 0.0;
    }

    return dot / (std::sqrt(norm_a) * std::sqrt(norm_b));
}

std::unordered_map<std::string, double>
PredictionDriftDetector::ComputeClassDistribution(
    const std::vector<std::string>& classes) {

    std::unordered_map<std::string, double> distribution;

    for (const auto& cls : classes) {
        distribution[cls] += 1.0;
    }

    // Normalize
    for (auto& [cls, count] : distribution) {
        count /= classes.size();
    }

    return distribution;
}

std::pair<std::vector<double>, std::vector<double>>
PredictionDriftDetector::ComputeHistogram(
    const std::vector<double>& values,
    size_t num_bins) {

    if (values.empty() || num_bins == 0) {
        return {{}, {}};
    }

    auto [min_it, max_it] = std::minmax_element(values.begin(), values.end());
    double min_val = *min_it;
    double max_val = *max_it;

    if (max_val - min_val < 1e-10) {
        return {{min_val}, {static_cast<double>(values.size())}};
    }

    double bin_width = (max_val - min_val) / num_bins;

    std::vector<double> bins(num_bins);
    std::vector<double> counts(num_bins, 0.0);

    for (size_t i = 0; i < num_bins; ++i) {
        bins[i] = min_val + (i + 0.5) * bin_width;
    }

    for (double v : values) {
        size_t bin = std::min(
            static_cast<size_t>((v - min_val) / bin_width),
            num_bins - 1);
        counts[bin] += 1.0;
    }

    // Normalize
    for (double& c : counts) {
        c /= values.size();
    }

    return {bins, counts};
}

// =============================================================================
// Serialization
// =============================================================================

absl::StatusOr<std::string> PredictionDriftDetector::SerializeState() const {
    std::lock_guard<std::mutex> lock(mutex_);

    json j;
    j["config"] = {
        {"output_type", static_cast<int>(config_.output_type)},
        {"threshold", config_.threshold},
        {"window_size", config_.window_size},
        {"min_samples", config_.min_samples},
        {"num_bins", config_.num_bins},
        {"p_value_threshold", config_.p_value_threshold}
    };

    j["reference"] = {
        {"is_set", reference_.is_set},
        {"type", static_cast<int>(reference_.type)},
        {"sample_count", reference_.sample_count},
        {"avg_length", reference_.avg_length},
        {"std_length", reference_.std_length},
        {"mean", reference_.mean},
        {"std_dev", reference_.std_dev},
        {"min", reference_.min},
        {"max", reference_.max}
    };

    // Serialize class distribution
    if (!reference_.class_distribution.empty()) {
        j["reference"]["class_distribution"] = reference_.class_distribution;
        j["reference"]["class_names"] = reference_.class_names;
    }

    return j.dump();
}

absl::Status PredictionDriftDetector::LoadState(std::string_view state) {
    std::lock_guard<std::mutex> lock(mutex_);

    try {
        json j = json::parse(state);

        // Load config
        auto& cfg = j["config"];
        config_.output_type = static_cast<OutputType>(cfg["output_type"].get<int>());
        config_.threshold = cfg["threshold"];
        config_.window_size = cfg["window_size"];
        config_.min_samples = cfg["min_samples"];
        config_.num_bins = cfg["num_bins"];
        config_.p_value_threshold = cfg["p_value_threshold"];

        // Load reference
        auto& ref = j["reference"];
        reference_.is_set = ref["is_set"];
        reference_.type = static_cast<OutputType>(ref["type"].get<int>());
        reference_.sample_count = ref["sample_count"];
        reference_.avg_length = ref["avg_length"];
        reference_.std_length = ref["std_length"];
        reference_.mean = ref["mean"];
        reference_.std_dev = ref["std_dev"];
        reference_.min = ref["min"];
        reference_.max = ref["max"];

        if (ref.contains("class_distribution")) {
            reference_.class_distribution =
                ref["class_distribution"].get<std::unordered_map<std::string, double>>();
            reference_.class_names =
                ref["class_names"].get<std::vector<std::string>>();
        }

        return absl::OkStatus();
    } catch (const std::exception& e) {
        return absl::InvalidArgumentError(
            std::string("Failed to parse state: ") + e.what());
    }
}

// =============================================================================
// Factory and Utility Functions
// =============================================================================

std::unique_ptr<DriftDetector> CreatePredictionDriftDetector(
    PredictionDriftConfig config) {
    return std::make_unique<PredictionDriftDetector>(std::move(config));
}

std::string OutputTypeToString(OutputType type) {
    switch (type) {
        case OutputType::kClassification: return "Classification";
        case OutputType::kRegression: return "Regression";
        case OutputType::kText: return "Text";
        case OutputType::kEmbedding: return "Embedding";
        case OutputType::kProbability: return "Probability";
        default: return "Unknown";
    }
}

OutputType StringToOutputType(const std::string& str) {
    if (str == "Classification") return OutputType::kClassification;
    if (str == "Regression") return OutputType::kRegression;
    if (str == "Text") return OutputType::kText;
    if (str == "Embedding") return OutputType::kEmbedding;
    if (str == "Probability") return OutputType::kProbability;
    return OutputType::kText;  // Default
}

}  // namespace pyflare::drift
