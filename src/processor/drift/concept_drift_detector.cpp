/// @file concept_drift_detector.cpp
/// @brief Implementation of concept drift detection algorithms

#include "processor/drift/concept_drift_detector.h"

#include <algorithm>
#include <cmath>
#include <sstream>

#include <nlohmann/json.hpp>
#include <spdlog/spdlog.h>

namespace pyflare::drift {

using json = nlohmann::json;

// =============================================================================
// ConceptDriftDetector Implementation
// =============================================================================

ConceptDriftDetector::ConceptDriftDetector(ConceptDriftConfig config)
    : config_(std::move(config)) {
    Reset();
}

ConceptDriftDetector::~ConceptDriftDetector() = default;

void ConceptDriftDetector::Reset() {
    std::lock_guard<std::mutex> lock(state_mutex_);

    n_samples_ = 0;
    sum_errors_ = 0.0;
    sum_squared_errors_ = 0.0;

    // DDM state
    ddm_p_min_ = std::numeric_limits<double>::max();
    ddm_s_min_ = std::numeric_limits<double>::max();
    ddm_min_index_ = 0;
    ddm_in_warning_ = false;
    ddm_warning_index_ = 0;

    // EDDM state
    eddm_m2_min_ = std::numeric_limits<double>::max();
    eddm_d_min_ = std::numeric_limits<double>::max();
    eddm_last_error_index_ = 0;
    eddm_distances_.clear();

    // ADWIN state
    adwin_buckets_.clear();
    adwin_width_ = 0;
    adwin_total_ = 0.0;
    adwin_variance_ = 0.0;

    // Page-Hinkley state
    ph_sum_ = 0.0;
    ph_mean_ = 0.0;
    ph_min_ = 0.0;
}

absl::Status ConceptDriftDetector::SetReference(const Distribution& reference) {
    // For concept drift, we don't use a reference distribution in the traditional sense
    // Instead, we track the error rate over time
    Reset();
    return absl::OkStatus();
}

absl::StatusOr<DriftResult> ConceptDriftDetector::Compute(
    const std::vector<DataPoint>& current_batch) {
    // Convert data points to error observations
    // Assume each DataPoint has an "error" or "correct" attribute
    std::vector<bool> predictions;
    predictions.reserve(current_batch.size());

    for (const auto& dp : current_batch) {
        // Check for "correct" attribute
        auto it = dp.attributes.find("correct");
        if (it != dp.attributes.end()) {
            predictions.push_back(it->second == "true" || it->second == "1");
        } else {
            // Check for "error" attribute
            auto err_it = dp.attributes.find("error");
            if (err_it != dp.attributes.end()) {
                try {
                    double error = std::stod(err_it->second);
                    predictions.push_back(error == 0.0);
                } catch (...) {
                    predictions.push_back(true);  // Assume correct if can't parse
                }
            } else {
                predictions.push_back(true);  // Default to correct
            }
        }
    }

    auto result = AnalyzeBatch(predictions);
    if (!result.ok()) {
        return result.status();
    }

    // Convert ConceptDriftResult to DriftResult
    DriftResult drift_result;
    drift_result.type = DriftType::kConcept;
    drift_result.score = result->drift_score;
    drift_result.threshold = config_.threshold;
    drift_result.is_drifted = result->drift_detected;
    drift_result.explanation = result->explanation;
    drift_result.detected_at = std::chrono::system_clock::now();

    return drift_result;
}

absl::StatusOr<ConceptDriftResult> ConceptDriftDetector::Update(bool correct) {
    double error = correct ? 0.0 : 1.0;
    return UpdateWithError(error);
}

absl::StatusOr<ConceptDriftResult> ConceptDriftDetector::UpdateWithError(double error) {
    std::lock_guard<std::mutex> lock(state_mutex_);

    switch (config_.method) {
        case ConceptDriftMethod::kDDM:
            return UpdateDDM(error);
        case ConceptDriftMethod::kEDDM:
            return UpdateEDDM(error);
        case ConceptDriftMethod::kADWIN:
            return UpdateADWIN(error);
        case ConceptDriftMethod::kPageHinkley:
            return UpdatePageHinkley(error);
        case ConceptDriftMethod::kEmbeddingDelta:
            return absl::UnimplementedError(
                "EmbeddingDelta requires AnalyzeEmbeddings method");
        default:
            return absl::InvalidArgumentError("Unknown concept drift method");
    }
}

absl::StatusOr<ConceptDriftResult> ConceptDriftDetector::UpdateWithRecord(
    const std::string& input,
    const std::string& output,
    const std::string& expected) {
    // Simple string comparison for correctness
    bool correct = (output == expected);
    return Update(correct);
}

absl::StatusOr<ConceptDriftResult> ConceptDriftDetector::AnalyzeBatch(
    const std::vector<bool>& predictions) {
    ConceptDriftResult final_result;
    final_result.method_used = config_.method;

    for (bool correct : predictions) {
        auto result = Update(correct);
        if (!result.ok()) {
            return result.status();
        }
        final_result = *result;
    }

    return final_result;
}

absl::StatusOr<ConceptDriftResult> ConceptDriftDetector::AnalyzeEmbeddings(
    const std::vector<std::vector<float>>& input_embeddings,
    const std::vector<std::vector<float>>& output_embeddings) {

    if (input_embeddings.size() != output_embeddings.size()) {
        return absl::InvalidArgumentError(
            "Input and output embedding counts must match");
    }

    if (input_embeddings.empty()) {
        return absl::InvalidArgumentError("Empty embedding vectors");
    }

    // Compute relationship between inputs and outputs
    // Track correlation changes over time

    // Simple approach: compute mean cosine similarity between input-output pairs
    // and track if this relationship changes

    auto cosine_similarity = [](const std::vector<float>& a,
                                 const std::vector<float>& b) -> double {
        if (a.size() != b.size() || a.empty()) return 0.0;

        double dot = 0.0, norm_a = 0.0, norm_b = 0.0;
        for (size_t i = 0; i < a.size(); ++i) {
            dot += a[i] * b[i];
            norm_a += a[i] * a[i];
            norm_b += b[i] * b[i];
        }

        if (norm_a < 1e-10 || norm_b < 1e-10) return 0.0;
        return dot / (std::sqrt(norm_a) * std::sqrt(norm_b));
    };

    // Compute similarities and feed to ADWIN
    ConceptDriftResult result;
    result.method_used = ConceptDriftMethod::kEmbeddingDelta;

    for (size_t i = 0; i < input_embeddings.size(); ++i) {
        double sim = cosine_similarity(input_embeddings[i], output_embeddings[i]);
        // Transform similarity to [0,1] range representing "error" (low similarity = high error)
        double error = (1.0 - sim) / 2.0;

        auto update_result = UpdateWithError(error);
        if (!update_result.ok()) {
            return update_result.status();
        }
        result = *update_result;
    }

    return result;
}

// =============================================================================
// DDM (Drift Detection Method) Implementation
// =============================================================================

ConceptDriftResult ConceptDriftDetector::UpdateDDM(double error) {
    n_samples_++;
    sum_errors_ += error;

    // Current error rate and standard deviation
    double p = sum_errors_ / n_samples_;
    double s = std::sqrt(p * (1.0 - p) / n_samples_);

    ConceptDriftResult result = BuildResult(false, false, "");
    result.current_error_rate = p;
    result.samples_analyzed = n_samples_;

    // Need minimum samples before detection
    if (!HasMinimumSamples()) {
        result.explanation = "Collecting samples (min: " +
            std::to_string(config_.min_samples) + ")";
        return result;
    }

    // Update minimum if current is better
    if (p + s < ddm_p_min_ + ddm_s_min_) {
        ddm_p_min_ = p;
        ddm_s_min_ = s;
        ddm_min_index_ = n_samples_;
    }

    result.baseline_error_rate = ddm_p_min_;

    // Calculate drift score as distance from minimum
    double drift_distance = (p + s) - (ddm_p_min_ + ddm_s_min_);
    if (ddm_s_min_ > 0) {
        result.drift_score = drift_distance / ddm_s_min_;
    }

    // Check for warning level
    if (p + s >= ddm_p_min_ + config_.ddm_warning_level * ddm_s_min_) {
        if (!ddm_in_warning_) {
            ddm_in_warning_ = true;
            ddm_warning_index_ = n_samples_;
        }
        result.warning_detected = true;

        // Check for drift level
        if (p + s >= ddm_p_min_ + config_.ddm_drift_level * ddm_s_min_) {
            result.drift_detected = true;
            result.drift_point = ddm_min_index_;
            result.explanation = "DDM detected drift: error rate " +
                std::to_string(p) + " exceeds threshold (" +
                std::to_string(config_.ddm_drift_level) + " std above min " +
                std::to_string(ddm_p_min_) + ")";

            spdlog::warn("DDM concept drift detected at sample {} (drift point: {})",
                        n_samples_, ddm_min_index_);
        } else {
            result.explanation = "DDM warning: error rate increasing";
        }
    } else {
        ddm_in_warning_ = false;
        result.explanation = "No drift detected (error rate: " +
            std::to_string(p) + ")";
    }

    return result;
}

// =============================================================================
// EDDM (Early DDM) Implementation
// =============================================================================

ConceptDriftResult ConceptDriftDetector::UpdateEDDM(double error) {
    n_samples_++;
    sum_errors_ += error;

    ConceptDriftResult result = BuildResult(false, false, "");
    result.current_error_rate = sum_errors_ / n_samples_;
    result.samples_analyzed = n_samples_;

    // EDDM tracks distance between consecutive errors
    if (error > 0.5) {  // Error occurred
        if (eddm_last_error_index_ > 0) {
            size_t distance = n_samples_ - eddm_last_error_index_;
            eddm_distances_.push_back(distance);

            // Keep only recent distances
            while (eddm_distances_.size() > config_.window_size) {
                eddm_distances_.pop_front();
            }
        }
        eddm_last_error_index_ = n_samples_;
    }

    if (!HasMinimumSamples() || eddm_distances_.size() < 10) {
        result.explanation = "Collecting error distance samples";
        return result;
    }

    // Calculate mean and std of error distances
    double sum = 0.0, sum_sq = 0.0;
    for (size_t d : eddm_distances_) {
        sum += d;
        sum_sq += d * d;
    }
    double mean = sum / eddm_distances_.size();
    double variance = (sum_sq / eddm_distances_.size()) - (mean * mean);
    double std_dev = std::sqrt(std::max(0.0, variance));

    double m2 = mean + 2.0 * std_dev;

    // Update minimum
    if (m2 > eddm_m2_min_) {
        eddm_m2_min_ = m2;
        eddm_d_min_ = mean;
    }

    result.baseline_error_rate = 1.0 / eddm_d_min_;  // Convert distance to rate

    // Calculate drift score
    if (eddm_m2_min_ > 0) {
        result.drift_score = 1.0 - (m2 / eddm_m2_min_);
    }

    // Check for warning (90% of best)
    double warning_threshold = 0.9;
    double drift_threshold = 0.5;

    if (m2 < warning_threshold * eddm_m2_min_) {
        result.warning_detected = true;

        if (m2 < drift_threshold * eddm_m2_min_) {
            result.drift_detected = true;
            result.drift_point = n_samples_;
            result.explanation = "EDDM detected drift: error distance decreased significantly";
            spdlog::warn("EDDM concept drift detected at sample {}", n_samples_);
        } else {
            result.explanation = "EDDM warning: error distance decreasing";
        }
    } else {
        result.explanation = "No drift detected";
    }

    return result;
}

// =============================================================================
// ADWIN (Adaptive Windowing) Implementation
// =============================================================================

ConceptDriftResult ConceptDriftDetector::UpdateADWIN(double value) {
    n_samples_++;

    // Add value to buckets
    ADWINBucket new_bucket;
    new_bucket.total = value;
    new_bucket.variance = 0.0;
    new_bucket.count = 1;
    adwin_buckets_.push_back(new_bucket);

    adwin_width_++;
    adwin_total_ += value;

    ConceptDriftResult result = BuildResult(false, false, "");
    result.current_error_rate = adwin_total_ / adwin_width_;
    result.samples_analyzed = n_samples_;

    if (!HasMinimumSamples()) {
        result.explanation = "Collecting samples";
        return result;
    }

    // Compress buckets (merge adjacent buckets with similar values)
    CompressBuckets();

    // Check for drift by finding optimal cut point
    bool drift_found = false;
    size_t cut_point = 0;

    // Try different cut points
    double total_left = 0.0;
    size_t count_left = 0;

    for (size_t i = 0; i < adwin_buckets_.size() - 1 && !drift_found; ++i) {
        total_left += adwin_buckets_[i].total;
        count_left += adwin_buckets_[i].count;

        size_t count_right = adwin_width_ - count_left;
        if (count_left < config_.min_samples / 2 ||
            count_right < config_.min_samples / 2) {
            continue;
        }

        double mean_left = total_left / count_left;
        double mean_right = (adwin_total_ - total_left) / count_right;

        double diff = std::abs(mean_left - mean_right);

        // ADWIN statistical test
        double m = 1.0 / count_left + 1.0 / count_right;
        double epsilon = std::sqrt(0.5 * m * std::log(4.0 / config_.adwin_delta));

        if (diff > epsilon) {
            drift_found = true;
            cut_point = i;

            result.drift_score = diff / epsilon;
            result.statistic = diff;
        }
    }

    if (drift_found) {
        // Remove old buckets before cut point
        while (cut_point > 0 && !adwin_buckets_.empty()) {
            auto& bucket = adwin_buckets_.front();
            adwin_total_ -= bucket.total;
            adwin_width_ -= bucket.count;
            adwin_buckets_.pop_front();
            cut_point--;
        }

        result.drift_detected = true;
        result.drift_point = n_samples_;
        result.change_points.push_back(n_samples_);
        result.explanation = "ADWIN detected drift: significant mean change";

        spdlog::warn("ADWIN concept drift detected at sample {}", n_samples_);
    } else {
        result.explanation = "No drift detected (window mean: " +
            std::to_string(adwin_total_ / adwin_width_) + ")";
    }

    result.baseline_error_rate = adwin_total_ / adwin_width_;

    return result;
}

void ConceptDriftDetector::CompressBuckets() {
    // Merge consecutive buckets when they have similar values
    // This keeps memory bounded while maintaining statistical accuracy

    const size_t max_buckets = 100;

    while (adwin_buckets_.size() > max_buckets) {
        // Find pair with smallest difference to merge
        size_t best_idx = 0;
        double min_diff = std::numeric_limits<double>::max();

        for (size_t i = 0; i < adwin_buckets_.size() - 1; ++i) {
            double mean_i = adwin_buckets_[i].total / adwin_buckets_[i].count;
            double mean_j = adwin_buckets_[i + 1].total / adwin_buckets_[i + 1].count;
            double diff = std::abs(mean_i - mean_j);

            if (diff < min_diff) {
                min_diff = diff;
                best_idx = i;
            }
        }

        // Merge buckets at best_idx and best_idx + 1
        adwin_buckets_[best_idx].total += adwin_buckets_[best_idx + 1].total;
        adwin_buckets_[best_idx].count += adwin_buckets_[best_idx + 1].count;
        adwin_buckets_.erase(adwin_buckets_.begin() + best_idx + 1);
    }
}

// =============================================================================
// Page-Hinkley Implementation
// =============================================================================

ConceptDriftResult ConceptDriftDetector::UpdatePageHinkley(double value) {
    n_samples_++;
    sum_errors_ += value;

    // Update running mean
    double old_mean = ph_mean_;
    ph_mean_ = sum_errors_ / n_samples_;

    // Update cumulative sum with drift allowance
    ph_sum_ += value - old_mean - config_.ph_alpha;

    // Track minimum
    if (ph_sum_ < ph_min_) {
        ph_min_ = ph_sum_;
    }

    ConceptDriftResult result = BuildResult(false, false, "");
    result.current_error_rate = ph_mean_;
    result.samples_analyzed = n_samples_;

    if (!HasMinimumSamples()) {
        result.explanation = "Collecting samples";
        return result;
    }

    // Page-Hinkley test statistic
    double ph_stat = ph_sum_ - ph_min_;
    result.statistic = ph_stat;
    result.drift_score = ph_stat / config_.ph_threshold;

    if (ph_stat > config_.ph_threshold) {
        result.drift_detected = true;
        result.drift_point = n_samples_;
        result.explanation = "Page-Hinkley detected drift: PH=" +
            std::to_string(ph_stat) + " > threshold=" +
            std::to_string(config_.ph_threshold);

        spdlog::warn("Page-Hinkley concept drift detected at sample {}", n_samples_);

        // Reset after detection
        ph_sum_ = 0.0;
        ph_min_ = 0.0;
    } else {
        result.explanation = "No drift detected (PH stat: " +
            std::to_string(ph_stat) + ")";
    }

    return result;
}

// =============================================================================
// Helper Methods
// =============================================================================

bool ConceptDriftDetector::HasMinimumSamples() const {
    return n_samples_ >= config_.min_samples;
}

ConceptDriftResult ConceptDriftDetector::BuildResult(
    bool drift, bool warning, const std::string& explanation) {

    ConceptDriftResult result;
    result.drift_detected = drift;
    result.warning_detected = warning;
    result.explanation = explanation;
    result.method_used = config_.method;
    result.samples_analyzed = n_samples_;

    if (n_samples_ > 0) {
        result.current_error_rate = sum_errors_ / n_samples_;
    }

    return result;
}

ConceptDriftDetector::State ConceptDriftDetector::GetState() const {
    std::lock_guard<std::mutex> lock(state_mutex_);

    State state;
    state.total_samples = n_samples_;

    if (n_samples_ > 0) {
        state.error_rate = sum_errors_ / n_samples_;
        double variance = (sum_squared_errors_ / n_samples_) -
                          (state.error_rate * state.error_rate);
        state.error_rate_std = std::sqrt(std::max(0.0, variance));
    }

    state.in_warning_state = ddm_in_warning_;
    state.warning_start_index = ddm_warning_index_;
    state.min_error_rate = ddm_p_min_;
    state.min_std = ddm_s_min_;
    state.min_error_index = ddm_min_index_;

    state.window_size = adwin_width_;
    if (adwin_width_ > 0) {
        state.window_mean = adwin_total_ / adwin_width_;
    }

    state.cumulative_sum = ph_sum_;
    state.min_cumulative = ph_min_;

    return state;
}

void ConceptDriftDetector::SetMethod(ConceptDriftMethod method) {
    std::lock_guard<std::mutex> lock(state_mutex_);
    config_.method = method;
    // Note: Doesn't reset state, just changes method for future updates
}

absl::StatusOr<std::string> ConceptDriftDetector::SerializeState() const {
    std::lock_guard<std::mutex> lock(state_mutex_);

    json j;
    j["config"] = {
        {"method", static_cast<int>(config_.method)},
        {"window_size", config_.window_size},
        {"threshold", config_.threshold},
        {"min_samples", config_.min_samples},
        {"ddm_warning_level", config_.ddm_warning_level},
        {"ddm_drift_level", config_.ddm_drift_level},
        {"adwin_delta", config_.adwin_delta},
        {"ph_threshold", config_.ph_threshold},
        {"ph_alpha", config_.ph_alpha}
    };

    j["state"] = {
        {"n_samples", n_samples_},
        {"sum_errors", sum_errors_},
        {"sum_squared_errors", sum_squared_errors_},
        {"ddm_p_min", ddm_p_min_},
        {"ddm_s_min", ddm_s_min_},
        {"ddm_min_index", ddm_min_index_},
        {"ddm_in_warning", ddm_in_warning_},
        {"ddm_warning_index", ddm_warning_index_},
        {"ph_sum", ph_sum_},
        {"ph_mean", ph_mean_},
        {"ph_min", ph_min_},
        {"adwin_width", adwin_width_},
        {"adwin_total", adwin_total_}
    };

    return j.dump();
}

absl::Status ConceptDriftDetector::LoadState(std::string_view state) {
    std::lock_guard<std::mutex> lock(state_mutex_);

    try {
        json j = json::parse(state);

        // Load config
        auto& cfg = j["config"];
        config_.method = static_cast<ConceptDriftMethod>(cfg["method"].get<int>());
        config_.window_size = cfg["window_size"];
        config_.threshold = cfg["threshold"];
        config_.min_samples = cfg["min_samples"];
        config_.ddm_warning_level = cfg["ddm_warning_level"];
        config_.ddm_drift_level = cfg["ddm_drift_level"];
        config_.adwin_delta = cfg["adwin_delta"];
        config_.ph_threshold = cfg["ph_threshold"];
        config_.ph_alpha = cfg["ph_alpha"];

        // Load state
        auto& st = j["state"];
        n_samples_ = st["n_samples"];
        sum_errors_ = st["sum_errors"];
        sum_squared_errors_ = st["sum_squared_errors"];
        ddm_p_min_ = st["ddm_p_min"];
        ddm_s_min_ = st["ddm_s_min"];
        ddm_min_index_ = st["ddm_min_index"];
        ddm_in_warning_ = st["ddm_in_warning"];
        ddm_warning_index_ = st["ddm_warning_index"];
        ph_sum_ = st["ph_sum"];
        ph_mean_ = st["ph_mean"];
        ph_min_ = st["ph_min"];
        adwin_width_ = st["adwin_width"];
        adwin_total_ = st["adwin_total"];

        return absl::OkStatus();
    } catch (const std::exception& e) {
        return absl::InvalidArgumentError(
            std::string("Failed to parse state: ") + e.what());
    }
}

// =============================================================================
// Factory and Utility Functions
// =============================================================================

std::unique_ptr<DriftDetector> CreateConceptDriftDetector(
    ConceptDriftConfig config) {
    return std::make_unique<ConceptDriftDetector>(std::move(config));
}

std::string ConceptDriftMethodToString(ConceptDriftMethod method) {
    switch (method) {
        case ConceptDriftMethod::kDDM:
            return "DDM";
        case ConceptDriftMethod::kEDDM:
            return "EDDM";
        case ConceptDriftMethod::kADWIN:
            return "ADWIN";
        case ConceptDriftMethod::kPageHinkley:
            return "PageHinkley";
        case ConceptDriftMethod::kEmbeddingDelta:
            return "EmbeddingDelta";
        default:
            return "Unknown";
    }
}

ConceptDriftMethod StringToConceptDriftMethod(const std::string& str) {
    if (str == "DDM") return ConceptDriftMethod::kDDM;
    if (str == "EDDM") return ConceptDriftMethod::kEDDM;
    if (str == "ADWIN") return ConceptDriftMethod::kADWIN;
    if (str == "PageHinkley") return ConceptDriftMethod::kPageHinkley;
    if (str == "EmbeddingDelta") return ConceptDriftMethod::kEmbeddingDelta;
    return ConceptDriftMethod::kADWIN;  // Default
}

}  // namespace pyflare::drift
