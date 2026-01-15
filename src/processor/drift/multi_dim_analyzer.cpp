/// @file multi_dim_analyzer.cpp
/// @brief Multi-dimensional drift analysis implementation

#include "processor/drift/multi_dim_analyzer.h"

#include <algorithm>
#include <cmath>
#include <numeric>
#include <sstream>

#include <nlohmann/json.hpp>

namespace pyflare::drift {

using json = nlohmann::json;

MultiDimDriftAnalyzer::MultiDimDriftAnalyzer(
    std::shared_ptr<storage::ClickHouseClient> clickhouse,
    std::shared_ptr<storage::QdrantClient> qdrant,
    std::shared_ptr<storage::RedisClient> redis,
    MultiDimAnalyzerConfig config)
    : clickhouse_(std::move(clickhouse)),
      qdrant_(std::move(qdrant)),
      redis_(std::move(redis)),
      config_(std::move(config)) {
}

MultiDimDriftAnalyzer::~MultiDimDriftAnalyzer() = default;

absl::Status MultiDimDriftAnalyzer::Initialize() {
    if (initialized_) {
        return absl::OkStatus();
    }

    // Initialize reference store
    ReferenceStoreConfig ref_config;
    ref_config.vector_dimension = config_.mmd_config.max_samples > 0
        ? config_.mmd_config.max_samples : 1536;
    reference_store_ = std::make_unique<ReferenceStore>(
        qdrant_, redis_, ref_config);

    auto status = reference_store_->Initialize();
    if (!status.ok()) {
        return status;
    }

    initialized_ = true;
    return absl::OkStatus();
}

absl::Status MultiDimDriftAnalyzer::RegisterModel(
    const std::string& model_id,
    const ReferenceData& reference) {
    if (!initialized_) {
        return absl::FailedPreconditionError("Analyzer not initialized");
    }

    if (model_id.empty()) {
        return absl::InvalidArgumentError("Model ID cannot be empty");
    }

    std::lock_guard<std::mutex> lock(detectors_mutex_);

    ModelDetectors detectors;

    // Initialize PSI detector for feature drift
    if (config_.enable_feature_drift && !reference.feature_values.empty()) {
        detectors.psi_detector = std::make_unique<PSIDriftDetector>(config_.psi_config);

        Distribution ref_dist;
        for (const auto& sample : reference.feature_values) {
            ref_dist.AddSample(sample);
        }
        auto status = detectors.psi_detector->SetReference(ref_dist);
        if (!status.ok()) {
            return status;
        }
    }

    // Initialize MMD detector for embedding drift
    if (config_.enable_embedding_drift && !reference.embeddings.empty()) {
        detectors.mmd_detector = std::make_unique<MMDDriftDetector>(config_.mmd_config);

        auto status = detectors.mmd_detector->SetReferenceEmbeddings(reference.embeddings);
        if (!status.ok()) {
            return status;
        }

        // Store embeddings in Qdrant for persistence
        auto store_status = reference_store_->StoreEmbeddings(
            model_id, reference.embeddings);
        if (!store_status.ok()) {
            return store_status;
        }
    }

    // Initialize concept drift detector
    if (config_.enable_concept_drift) {
        detectors.concept_detector = std::make_unique<ConceptDriftDetector>(
            config_.concept_config);

        // Feed historical correctness data if available
        if (!reference.correctness_labels.empty()) {
            for (bool correct : reference.correctness_labels) {
                auto result = detectors.concept_detector->Update(correct);
                // Ignore initial results during reference building
            }
            detectors.concept_detector->Reset();
        }
    }

    // Initialize prediction drift detector
    if (config_.enable_prediction_drift) {
        detectors.prediction_detector = std::make_unique<PredictionDriftDetector>(
            config_.prediction_config);

        if (!reference.text_outputs.empty()) {
            auto status = detectors.prediction_detector->SetReferenceTexts(
                reference.text_outputs);
            if (!status.ok()) {
                return status;
            }
        } else if (!reference.class_labels.empty()) {
            auto status = detectors.prediction_detector->SetReferenceClasses(
                reference.class_labels);
            if (!status.ok()) {
                return status;
            }
        } else if (!reference.numeric_outputs.empty()) {
            auto status = detectors.prediction_detector->SetReferenceValues(
                reference.numeric_outputs);
            if (!status.ok()) {
                return status;
            }
        }
    }

    detectors.is_initialized = true;
    detectors.registered_at = std::chrono::system_clock::now();
    detectors.analysis_count = 0;

    model_detectors_[model_id] = std::move(detectors);

    // Update stats
    {
        std::lock_guard<std::mutex> stats_lock(stats_mutex_);
        stats_.models_registered = model_detectors_.size();
    }

    return absl::OkStatus();
}

absl::Status MultiDimDriftAnalyzer::UpdateReference(
    const std::string& model_id,
    const ReferenceData& reference,
    bool merge) {
    if (!merge) {
        // Full replacement - unregister first
        auto status = UnregisterModel(model_id);
        if (!status.ok() && !absl::IsNotFound(status)) {
            return status;
        }
        return RegisterModel(model_id, reference);
    }

    // For merge, update individual detectors
    std::lock_guard<std::mutex> lock(detectors_mutex_);

    auto it = model_detectors_.find(model_id);
    if (it == model_detectors_.end()) {
        return absl::NotFoundError("Model not registered: " + model_id);
    }

    // Update embedding reference with sliding window
    if (!reference.embeddings.empty() && it->second.mmd_detector) {
        auto status = reference_store_->UpdateEmbeddings(model_id, reference.embeddings);
        if (!status.ok()) {
            return status;
        }

        // Re-fetch and update MMD detector
        auto embeddings = reference_store_->GetEmbeddings(model_id);
        if (embeddings.ok()) {
            auto set_status = it->second.mmd_detector->SetReferenceEmbeddings(*embeddings);
            if (!set_status.ok()) {
                return set_status;
            }
        }
    }

    return absl::OkStatus();
}

bool MultiDimDriftAnalyzer::IsModelRegistered(const std::string& model_id) const {
    std::lock_guard<std::mutex> lock(detectors_mutex_);
    return model_detectors_.find(model_id) != model_detectors_.end();
}

std::vector<std::string> MultiDimDriftAnalyzer::GetRegisteredModels() const {
    std::lock_guard<std::mutex> lock(detectors_mutex_);
    std::vector<std::string> models;
    models.reserve(model_detectors_.size());
    for (const auto& [model_id, _] : model_detectors_) {
        models.push_back(model_id);
    }
    return models;
}

absl::Status MultiDimDriftAnalyzer::UnregisterModel(const std::string& model_id) {
    std::lock_guard<std::mutex> lock(detectors_mutex_);

    auto it = model_detectors_.find(model_id);
    if (it == model_detectors_.end()) {
        return absl::NotFoundError("Model not registered: " + model_id);
    }

    model_detectors_.erase(it);

    // Clean up reference data
    auto status = reference_store_->DeleteReference(model_id);
    if (!status.ok() && !absl::IsNotFound(status)) {
        return status;
    }

    // Update stats
    {
        std::lock_guard<std::mutex> stats_lock(stats_mutex_);
        stats_.models_registered = model_detectors_.size();
    }

    return absl::OkStatus();
}

absl::StatusOr<MultiDimensionalDriftStatus> MultiDimDriftAnalyzer::Analyze(
    const std::string& model_id,
    const std::vector<DataPoint>& current_data) {
    if (!initialized_) {
        return absl::FailedPreconditionError("Analyzer not initialized");
    }

    auto start_time = std::chrono::steady_clock::now();

    // Check if model is registered
    {
        std::lock_guard<std::mutex> lock(detectors_mutex_);
        if (model_detectors_.find(model_id) == model_detectors_.end()) {
            return absl::NotFoundError("Model not registered: " + model_id);
        }
    }

    if (current_data.size() < config_.min_samples) {
        return absl::InvalidArgumentError(
            "Insufficient samples: " + std::to_string(current_data.size()) +
            " < " + std::to_string(config_.min_samples));
    }

    // Try to load cached result
    if (config_.enable_caching) {
        auto cached = LoadCachedResult(model_id);
        if (cached.ok() && cached->has_value()) {
            return *cached->value();
        }
    }

    std::vector<MultiDimensionalDriftStatus::DimensionScore> scores;

    // Compute feature drift
    if (config_.enable_feature_drift) {
        auto result = ComputeFeatureDrift(model_id, current_data);
        if (result.ok()) {
            scores.push_back(*result);
        }
    }

    // Compute embedding drift
    if (config_.enable_embedding_drift) {
        std::vector<std::vector<float>> embeddings;
        embeddings.reserve(current_data.size());
        for (const auto& point : current_data) {
            if (!point.embedding.empty()) {
                embeddings.push_back(point.embedding);
            }
        }
        if (!embeddings.empty()) {
            auto result = ComputeEmbeddingDrift(model_id, embeddings);
            if (result.ok()) {
                scores.push_back(*result);
            }
        }
    }

    // Compute concept drift (requires correctness attribute)
    if (config_.enable_concept_drift) {
        std::vector<bool> correctness;
        for (const auto& point : current_data) {
            auto it = point.attributes.find("correct");
            if (it != point.attributes.end()) {
                correctness.push_back(it->second == "true" || it->second == "1");
            }
        }
        if (!correctness.empty()) {
            auto result = ComputeConceptDrift(model_id, correctness);
            if (result.ok()) {
                scores.push_back(*result);
            }
        }
    }

    // Compute prediction drift (requires output attribute)
    if (config_.enable_prediction_drift) {
        std::vector<std::string> outputs;
        for (const auto& point : current_data) {
            auto it = point.attributes.find("output");
            if (it != point.attributes.end()) {
                outputs.push_back(it->second);
            }
        }
        if (!outputs.empty()) {
            auto result = ComputePredictionDrift(model_id, outputs);
            if (result.ok()) {
                scores.push_back(*result);
            }
        }
    }

    auto end_time = std::chrono::steady_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(
        end_time - start_time);

    auto status = BuildStatus(model_id, scores, current_data.size(), duration);

    // Persist result
    if (config_.persist_results) {
        auto persist_status = PersistResult(status);
        // Log but don't fail on persistence errors
    }

    // Invoke callbacks if drift detected
    if (status.has_any_drift) {
        std::lock_guard<std::mutex> lock(callbacks_mutex_);
        for (const auto& callback : drift_callbacks_) {
            callback(status);
        }
    }

    // Update stats
    {
        std::lock_guard<std::mutex> lock(stats_mutex_);
        stats_.total_analyses++;
        if (status.has_any_drift) {
            stats_.drift_detections++;
        }
        stats_.avg_analysis_time_ms =
            (stats_.avg_analysis_time_ms * (stats_.total_analyses - 1) +
             duration.count()) / stats_.total_analyses;
        stats_.last_analysis = std::chrono::system_clock::now();
    }

    // Update model analysis count
    {
        std::lock_guard<std::mutex> lock(detectors_mutex_);
        auto it = model_detectors_.find(model_id);
        if (it != model_detectors_.end()) {
            it->second.analysis_count++;
        }
    }

    return status;
}

absl::StatusOr<MultiDimensionalDriftStatus> MultiDimDriftAnalyzer::AnalyzeTimeRange(
    const std::string& model_id,
    std::chrono::system_clock::time_point start,
    std::chrono::system_clock::time_point end) {
    // Query data from ClickHouse
    auto data = QueryDataFromClickHouse(model_id, start, end);
    if (!data.ok()) {
        return data.status();
    }

    return Analyze(model_id, *data);
}

absl::StatusOr<MultiDimensionalDriftStatus> MultiDimDriftAnalyzer::AnalyzeTextOutputs(
    const std::string& model_id,
    const std::vector<std::string>& inputs,
    const std::vector<std::string>& outputs,
    const std::vector<std::vector<float>>& embeddings) {
    // Convert to DataPoints
    std::vector<DataPoint> data;
    data.reserve(inputs.size());

    for (size_t i = 0; i < inputs.size(); ++i) {
        DataPoint point;
        point.id = std::to_string(i);
        point.attributes["input"] = i < inputs.size() ? inputs[i] : "";
        point.attributes["output"] = i < outputs.size() ? outputs[i] : "";

        if (i < embeddings.size()) {
            point.embedding = embeddings[i];
        }

        data.push_back(std::move(point));
    }

    return Analyze(model_id, data);
}

absl::StatusOr<MultiDimensionalDriftStatus> MultiDimDriftAnalyzer::AnalyzeClassification(
    const std::string& model_id,
    const std::vector<std::vector<double>>& features,
    const std::vector<std::string>& predictions,
    const std::vector<std::string>& actual) {
    // Convert to DataPoints
    std::vector<DataPoint> data;
    data.reserve(features.size());

    for (size_t i = 0; i < features.size(); ++i) {
        DataPoint point;
        point.id = std::to_string(i);
        point.features = features[i];
        point.attributes["output"] = i < predictions.size() ? predictions[i] : "";

        if (i < actual.size() && i < predictions.size()) {
            point.attributes["correct"] = (predictions[i] == actual[i]) ? "true" : "false";
        }

        data.push_back(std::move(point));
    }

    return Analyze(model_id, data);
}

absl::StatusOr<std::vector<MultiDimensionalDriftStatus>> MultiDimDriftAnalyzer::GetDriftTrend(
    const std::string& model_id,
    std::chrono::system_clock::time_point start,
    std::chrono::system_clock::time_point end,
    std::chrono::minutes interval) {
    std::vector<MultiDimensionalDriftStatus> trend;

    auto current = start;
    while (current < end) {
        auto next = current + interval;
        if (next > end) {
            next = end;
        }

        auto status = AnalyzeTimeRange(model_id, current, next);
        if (status.ok()) {
            trend.push_back(*status);
        }

        current = next;
    }

    return trend;
}

absl::StatusOr<std::vector<MultiDimensionalDriftStatus::Correlation>>
MultiDimDriftAnalyzer::ComputeCorrelations(
    const std::string& model_id,
    size_t history_hours) {
    // Get drift trend for correlation analysis
    auto end = std::chrono::system_clock::now();
    auto start = end - std::chrono::hours(history_hours);

    auto trend = GetDriftTrend(model_id, start, end, std::chrono::hours(1));
    if (!trend.ok()) {
        return trend.status();
    }

    if (trend->size() < 3) {
        return absl::InvalidArgumentError(
            "Insufficient data points for correlation analysis");
    }

    // Extract time series for each dimension
    std::unordered_map<std::string, std::vector<double>> dimension_series;

    for (const auto& status : *trend) {
        for (const auto& score : status.dimension_scores) {
            dimension_series[score.dimension_type].push_back(score.score);
        }
    }

    // Compute pairwise correlations
    std::vector<MultiDimensionalDriftStatus::Correlation> correlations;
    std::vector<std::string> dimensions;
    for (const auto& [dim, _] : dimension_series) {
        dimensions.push_back(dim);
    }

    for (size_t i = 0; i < dimensions.size(); ++i) {
        for (size_t j = i + 1; j < dimensions.size(); ++j) {
            const auto& series_a = dimension_series[dimensions[i]];
            const auto& series_b = dimension_series[dimensions[j]];

            if (series_a.size() != series_b.size() || series_a.size() < 3) {
                continue;
            }

            double r = ComputePearsonCorrelation(series_a, series_b);

            MultiDimensionalDriftStatus::Correlation corr;
            corr.dimension_a = dimensions[i];
            corr.dimension_b = dimensions[j];
            corr.correlation_coefficient = r;
            corr.are_correlated = std::abs(r) >= config_.correlation_threshold;

            if (corr.are_correlated) {
                if (r > 0) {
                    corr.explanation = dimensions[i] + " and " + dimensions[j] +
                        " tend to drift together (r=" + std::to_string(r) + ")";
                } else {
                    corr.explanation = dimensions[i] + " and " + dimensions[j] +
                        " show inverse correlation (r=" + std::to_string(r) + ")";
                }
            } else {
                corr.explanation = "No significant correlation detected";
            }

            correlations.push_back(corr);
        }
    }

    return correlations;
}

void MultiDimDriftAnalyzer::OnDriftDetected(
    std::function<void(const MultiDimensionalDriftStatus&)> callback) {
    std::lock_guard<std::mutex> lock(callbacks_mutex_);
    drift_callbacks_.push_back(std::move(callback));
}

void MultiDimDriftAnalyzer::ClearCallbacks() {
    std::lock_guard<std::mutex> lock(callbacks_mutex_);
    drift_callbacks_.clear();
}

void MultiDimDriftAnalyzer::SetConfig(MultiDimAnalyzerConfig config) {
    config_ = std::move(config);
}

MultiDimDriftAnalyzer::Stats MultiDimDriftAnalyzer::GetStats() const {
    std::lock_guard<std::mutex> lock(stats_mutex_);
    return stats_;
}

// ============================================================================
// Private Implementation
// ============================================================================

absl::StatusOr<MultiDimensionalDriftStatus::DimensionScore>
MultiDimDriftAnalyzer::ComputeFeatureDrift(
    const std::string& model_id,
    const std::vector<DataPoint>& data) {
    std::lock_guard<std::mutex> lock(detectors_mutex_);

    auto it = model_detectors_.find(model_id);
    if (it == model_detectors_.end() || !it->second.psi_detector) {
        return absl::NotFoundError("PSI detector not available for model");
    }

    auto result = it->second.psi_detector->Compute(data);
    if (!result.ok()) {
        return result.status();
    }

    MultiDimensionalDriftStatus::DimensionScore score;
    score.dimension_type = "feature";
    score.score = result->score;
    score.threshold = result->threshold;
    score.is_drifted = result->is_drifted;
    score.explanation = result->explanation;
    score.detector_name = "PSIDriftDetector";
    score.sub_scores = result->feature_scores;

    return score;
}

absl::StatusOr<MultiDimensionalDriftStatus::DimensionScore>
MultiDimDriftAnalyzer::ComputeEmbeddingDrift(
    const std::string& model_id,
    const std::vector<std::vector<float>>& embeddings) {
    std::lock_guard<std::mutex> lock(detectors_mutex_);

    auto it = model_detectors_.find(model_id);
    if (it == model_detectors_.end() || !it->second.mmd_detector) {
        return absl::NotFoundError("MMD detector not available for model");
    }

    auto result = it->second.mmd_detector->ComputeFromEmbeddings(embeddings);
    if (!result.ok()) {
        return result.status();
    }

    MultiDimensionalDriftStatus::DimensionScore score;
    score.dimension_type = "embedding";
    score.score = result->score;
    score.threshold = result->threshold;
    score.is_drifted = result->is_drifted;
    score.explanation = result->explanation;
    score.detector_name = "MMDDriftDetector";

    return score;
}

absl::StatusOr<MultiDimensionalDriftStatus::DimensionScore>
MultiDimDriftAnalyzer::ComputeConceptDrift(
    const std::string& model_id,
    const std::vector<bool>& correctness) {
    std::lock_guard<std::mutex> lock(detectors_mutex_);

    auto it = model_detectors_.find(model_id);
    if (it == model_detectors_.end() || !it->second.concept_detector) {
        return absl::NotFoundError("Concept drift detector not available for model");
    }

    auto result = it->second.concept_detector->AnalyzeBatch(correctness);
    if (!result.ok()) {
        return result.status();
    }

    MultiDimensionalDriftStatus::DimensionScore score;
    score.dimension_type = "concept";
    score.score = result->drift_score;
    score.threshold = it->second.concept_detector->GetThreshold();
    score.is_drifted = result->drift_detected;
    score.explanation = result->explanation;
    score.detector_name = "ConceptDriftDetector";
    score.sub_scores = result->feature_contributions;

    return score;
}

absl::StatusOr<MultiDimensionalDriftStatus::DimensionScore>
MultiDimDriftAnalyzer::ComputePredictionDrift(
    const std::string& model_id,
    const std::vector<std::string>& outputs) {
    std::lock_guard<std::mutex> lock(detectors_mutex_);

    auto it = model_detectors_.find(model_id);
    if (it == model_detectors_.end() || !it->second.prediction_detector) {
        return absl::NotFoundError("Prediction drift detector not available for model");
    }

    // Determine output type from config
    absl::StatusOr<PredictionDriftResult> result;
    switch (config_.prediction_config.output_type) {
        case OutputType::kClassification:
            result = it->second.prediction_detector->ComputeClassDrift(outputs);
            break;
        case OutputType::kText:
        default:
            result = it->second.prediction_detector->ComputeTextDrift(outputs);
            break;
    }

    if (!result.ok()) {
        return result.status();
    }

    MultiDimensionalDriftStatus::DimensionScore score;
    score.dimension_type = "prediction";
    score.score = result->drift_score;
    score.threshold = result->threshold;
    score.is_drifted = result->drift_detected;
    score.explanation = result->explanation;
    score.detector_name = "PredictionDriftDetector";

    return score;
}

double MultiDimDriftAnalyzer::ComputeOverallScore(
    const std::vector<MultiDimensionalDriftStatus::DimensionScore>& scores) {
    if (scores.empty()) {
        return 0.0;
    }

    double weighted_sum = 0.0;
    double weight_sum = 0.0;

    for (const auto& score : scores) {
        double weight = 0.0;
        if (score.dimension_type == "feature") {
            weight = config_.feature_weight;
        } else if (score.dimension_type == "embedding") {
            weight = config_.embedding_weight;
        } else if (score.dimension_type == "concept") {
            weight = config_.concept_weight;
        } else if (score.dimension_type == "prediction") {
            weight = config_.prediction_weight;
        } else {
            weight = 1.0 / scores.size();
        }

        weighted_sum += score.score * weight;
        weight_sum += weight;
    }

    return weight_sum > 0 ? weighted_sum / weight_sum : 0.0;
}

std::string MultiDimDriftAnalyzer::DetermineSeverity(double overall_score) {
    if (overall_score >= config_.critical_threshold) {
        return "critical";
    } else if (overall_score >= config_.high_threshold) {
        return "high";
    } else if (overall_score >= config_.medium_threshold) {
        return "medium";
    } else if (overall_score >= config_.low_threshold) {
        return "low";
    }
    return "none";
}

std::vector<std::string> MultiDimDriftAnalyzer::GenerateLikelyCauses(
    const std::vector<MultiDimensionalDriftStatus::DimensionScore>& scores) {
    std::vector<std::string> causes;

    bool has_feature_drift = false;
    bool has_embedding_drift = false;
    bool has_concept_drift = false;
    bool has_prediction_drift = false;

    for (const auto& score : scores) {
        if (!score.is_drifted) continue;

        if (score.dimension_type == "feature") {
            has_feature_drift = true;
        } else if (score.dimension_type == "embedding") {
            has_embedding_drift = true;
        } else if (score.dimension_type == "concept") {
            has_concept_drift = true;
        } else if (score.dimension_type == "prediction") {
            has_prediction_drift = true;
        }
    }

    // Generate causes based on drift patterns
    if (has_feature_drift && !has_concept_drift) {
        causes.push_back("Input distribution has changed but model behavior is stable - likely new data population");
    }

    if (has_embedding_drift && !has_feature_drift) {
        causes.push_back("Semantic content has shifted - possibly new topics or domains in inputs");
    }

    if (has_concept_drift && !has_feature_drift) {
        causes.push_back("Model behavior changed despite stable inputs - possible model degradation or external dependency change");
    }

    if (has_feature_drift && has_concept_drift) {
        causes.push_back("Both input distribution and model behavior have changed - data shift affecting model performance");
    }

    if (has_prediction_drift && !has_concept_drift) {
        causes.push_back("Output distribution changed without accuracy impact - possibly stylistic or formatting change");
    }

    if (has_prediction_drift && has_concept_drift) {
        causes.push_back("Output quality and distribution both changed - significant model behavior shift");
    }

    if (causes.empty()) {
        causes.push_back("No specific pattern identified - recommend manual investigation");
    }

    return causes;
}

std::vector<std::string> MultiDimDriftAnalyzer::GenerateRecommendations(
    const std::vector<MultiDimensionalDriftStatus::DimensionScore>& scores) {
    std::vector<std::string> recommendations;

    double max_score = 0.0;
    std::string max_dimension;

    for (const auto& score : scores) {
        if (score.score > max_score && score.is_drifted) {
            max_score = score.score;
            max_dimension = score.dimension_type;
        }
    }

    if (max_dimension == "feature") {
        recommendations.push_back("Review input data pipeline for changes in data sources");
        recommendations.push_back("Consider updating reference distribution with recent data");
        recommendations.push_back("Analyze top drifted features for root cause");
    } else if (max_dimension == "embedding") {
        recommendations.push_back("Investigate semantic shift in input content");
        recommendations.push_back("Check for new topics or domains in production data");
        recommendations.push_back("Consider domain adaptation if shift is persistent");
    } else if (max_dimension == "concept") {
        recommendations.push_back("Evaluate model performance on recent data");
        recommendations.push_back("Check for changes in external dependencies (APIs, knowledge bases)");
        recommendations.push_back("Consider model retraining if performance degradation persists");
    } else if (max_dimension == "prediction") {
        recommendations.push_back("Review output quality metrics and examples");
        recommendations.push_back("Check for prompt template changes");
        recommendations.push_back("Analyze output distribution for anomalies");
    }

    if (recommendations.empty()) {
        recommendations.push_back("Continue monitoring - no immediate action required");
    }

    return recommendations;
}

double MultiDimDriftAnalyzer::ComputePearsonCorrelation(
    const std::vector<double>& x,
    const std::vector<double>& y) {
    if (x.size() != y.size() || x.size() < 2) {
        return 0.0;
    }

    size_t n = x.size();

    double mean_x = std::accumulate(x.begin(), x.end(), 0.0) / n;
    double mean_y = std::accumulate(y.begin(), y.end(), 0.0) / n;

    double cov = 0.0;
    double var_x = 0.0;
    double var_y = 0.0;

    for (size_t i = 0; i < n; ++i) {
        double dx = x[i] - mean_x;
        double dy = y[i] - mean_y;
        cov += dx * dy;
        var_x += dx * dx;
        var_y += dy * dy;
    }

    double denom = std::sqrt(var_x * var_y);
    if (denom < 1e-10) {
        return 0.0;
    }

    return cov / denom;
}

absl::Status MultiDimDriftAnalyzer::PersistResult(
    const MultiDimensionalDriftStatus& status) {
    if (!clickhouse_) {
        return absl::OkStatus();
    }

    // Build insert query for drift_alerts table
    std::stringstream ss;
    ss << "INSERT INTO pyflare.drift_alerts (";
    ss << "model_id, timestamp, severity, overall_score, has_drift, ";
    ss << "dimension_scores, correlations, likely_causes, recommendations, samples_analyzed";
    ss << ") VALUES (";

    // Convert dimension scores to JSON
    json dim_scores_json = json::array();
    for (const auto& score : status.dimension_scores) {
        json score_obj;
        score_obj["dimension_type"] = score.dimension_type;
        score_obj["score"] = score.score;
        score_obj["threshold"] = score.threshold;
        score_obj["is_drifted"] = score.is_drifted;
        score_obj["explanation"] = score.explanation;
        score_obj["detector_name"] = score.detector_name;
        dim_scores_json.push_back(score_obj);
    }

    // Convert correlations to JSON
    json corr_json = json::array();
    for (const auto& corr : status.correlations) {
        json corr_obj;
        corr_obj["dimension_a"] = corr.dimension_a;
        corr_obj["dimension_b"] = corr.dimension_b;
        corr_obj["coefficient"] = corr.correlation_coefficient;
        corr_obj["are_correlated"] = corr.are_correlated;
        corr_json.push_back(corr_obj);
    }

    // Format timestamp
    auto time_t = std::chrono::system_clock::to_time_t(status.timestamp);

    ss << "'" << status.model_id << "', ";
    ss << "toDateTime(" << time_t << "), ";
    ss << "'" << status.severity << "', ";
    ss << status.overall_score << ", ";
    ss << (status.has_any_drift ? 1 : 0) << ", ";
    ss << "'" << dim_scores_json.dump() << "', ";
    ss << "'" << corr_json.dump() << "', ";

    // Array of causes
    ss << "[";
    for (size_t i = 0; i < status.likely_causes.size(); ++i) {
        if (i > 0) ss << ", ";
        ss << "'" << status.likely_causes[i] << "'";
    }
    ss << "], ";

    // Array of recommendations
    ss << "[";
    for (size_t i = 0; i < status.recommended_actions.size(); ++i) {
        if (i > 0) ss << ", ";
        ss << "'" << status.recommended_actions[i] << "'";
    }
    ss << "], ";

    ss << status.samples_analyzed;
    ss << ")";

    auto result = clickhouse_->Execute(ss.str());
    if (!result.ok()) {
        return result.status();
    }

    // Also cache in Redis if enabled
    if (config_.enable_caching && redis_) {
        json cache_obj;
        cache_obj["model_id"] = status.model_id;
        cache_obj["severity"] = status.severity;
        cache_obj["overall_score"] = status.overall_score;
        cache_obj["has_any_drift"] = status.has_any_drift;
        cache_obj["dimension_scores"] = dim_scores_json;
        cache_obj["timestamp"] = time_t;

        std::string cache_key = "pyflare:drift:status:" + status.model_id;
        auto cache_status = redis_->Set(cache_key, cache_obj.dump(), config_.cache_ttl);
        // Ignore cache failures
    }

    return absl::OkStatus();
}

absl::StatusOr<std::optional<MultiDimensionalDriftStatus>>
MultiDimDriftAnalyzer::LoadCachedResult(const std::string& model_id) {
    if (!redis_) {
        return std::nullopt;
    }

    std::string cache_key = "pyflare:drift:status:" + model_id;
    auto result = redis_->Get(cache_key);
    if (!result.ok()) {
        return result.status();
    }

    if (!result->has_value()) {
        return std::nullopt;
    }

    // Parse cached result
    try {
        auto j = json::parse(*result->value());

        MultiDimensionalDriftStatus status;
        status.model_id = j["model_id"].get<std::string>();
        status.severity = j["severity"].get<std::string>();
        status.overall_score = j["overall_score"].get<double>();
        status.has_any_drift = j["has_any_drift"].get<bool>();

        auto timestamp = j["timestamp"].get<int64_t>();
        status.timestamp = std::chrono::system_clock::from_time_t(timestamp);

        for (const auto& score_json : j["dimension_scores"]) {
            MultiDimensionalDriftStatus::DimensionScore score;
            score.dimension_type = score_json["dimension_type"].get<std::string>();
            score.score = score_json["score"].get<double>();
            score.threshold = score_json["threshold"].get<double>();
            score.is_drifted = score_json["is_drifted"].get<bool>();
            score.explanation = score_json["explanation"].get<std::string>();
            score.detector_name = score_json["detector_name"].get<std::string>();
            status.dimension_scores.push_back(score);
        }

        return status;
    } catch (const json::exception& e) {
        return absl::InternalError("Failed to parse cached result: " + std::string(e.what()));
    }
}

absl::StatusOr<std::vector<DataPoint>> MultiDimDriftAnalyzer::QueryDataFromClickHouse(
    const std::string& model_id,
    std::chrono::system_clock::time_point start,
    std::chrono::system_clock::time_point end) {
    if (!clickhouse_) {
        return absl::FailedPreconditionError("ClickHouse client not available");
    }

    auto start_time = std::chrono::system_clock::to_time_t(start);
    auto end_time = std::chrono::system_clock::to_time_t(end);

    std::stringstream ss;
    ss << "SELECT trace_id, input_preview, output_preview, ";
    ss << "model_id, status_code ";
    ss << "FROM pyflare.traces ";
    ss << "WHERE model_id = '" << model_id << "' ";
    ss << "AND start_time >= toDateTime(" << start_time << ") ";
    ss << "AND start_time <= toDateTime(" << end_time << ") ";
    ss << "ORDER BY start_time DESC ";
    ss << "LIMIT 10000";

    auto result = clickhouse_->Execute(ss.str());
    if (!result.ok()) {
        return result.status();
    }

    std::vector<DataPoint> data;
    data.reserve(result->rows.size());

    for (const auto& row : result->rows) {
        if (row.size() < 5) continue;

        DataPoint point;
        point.id = row[0];
        point.attributes["input"] = row[1];
        point.attributes["output"] = row[2];
        point.attributes["model_id"] = row[3];
        point.attributes["correct"] = (row[4] == "OK") ? "true" : "false";

        data.push_back(std::move(point));
    }

    return data;
}

MultiDimensionalDriftStatus MultiDimDriftAnalyzer::BuildStatus(
    const std::string& model_id,
    const std::vector<MultiDimensionalDriftStatus::DimensionScore>& scores,
    size_t samples_analyzed,
    std::chrono::milliseconds duration) {
    MultiDimensionalDriftStatus status;
    status.model_id = model_id;
    status.timestamp = std::chrono::system_clock::now();
    status.dimension_scores = scores;
    status.samples_analyzed = samples_analyzed;
    status.analysis_duration = duration;

    // Check if any drift detected
    status.has_any_drift = std::any_of(scores.begin(), scores.end(),
        [](const auto& s) { return s.is_drifted; });

    // Compute overall score
    status.overall_score = ComputeOverallScore(scores);

    // Determine severity
    status.severity = DetermineSeverity(status.overall_score);

    // Generate causes and recommendations
    if (status.has_any_drift) {
        status.likely_causes = GenerateLikelyCauses(scores);
        status.recommended_actions = GenerateRecommendations(scores);
    }

    return status;
}

// ============================================================================
// Factory Function
// ============================================================================

std::unique_ptr<MultiDimDriftAnalyzer> CreateMultiDimAnalyzer(
    std::shared_ptr<storage::ClickHouseClient> clickhouse,
    std::shared_ptr<storage::QdrantClient> qdrant,
    std::shared_ptr<storage::RedisClient> redis,
    MultiDimAnalyzerConfig config) {
    return std::make_unique<MultiDimDriftAnalyzer>(
        std::move(clickhouse),
        std::move(qdrant),
        std::move(redis),
        std::move(config));
}

}  // namespace pyflare::drift
