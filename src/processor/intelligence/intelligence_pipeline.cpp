/// @file intelligence_pipeline.cpp
/// @brief Intelligence pipeline implementation

#include "processor/intelligence/intelligence_pipeline.h"

#include <algorithm>
#include <numeric>
#include <random>
#include <sstream>

#include <nlohmann/json.hpp>

namespace pyflare::intelligence {

using json = nlohmann::json;

IntelligencePipeline::IntelligencePipeline(
    std::shared_ptr<storage::ClickHouseClient> clickhouse,
    std::shared_ptr<storage::QdrantClient> qdrant,
    std::shared_ptr<storage::RedisClient> redis,
    IntelligencePipelineConfig config)
    : clickhouse_(std::move(clickhouse)),
      qdrant_(std::move(qdrant)),
      redis_(std::move(redis)),
      config_(std::move(config)) {}

IntelligencePipeline::~IntelligencePipeline() {
    Stop();
}

absl::Status IntelligencePipeline::Initialize() {
    if (initialized_) {
        return absl::OkStatus();
    }

    // Initialize drift analyzer
    if (config_.enable_drift_detection) {
        drift_analyzer_ = std::make_unique<drift::MultiDimDriftAnalyzer>(
            clickhouse_, qdrant_, config_.drift_config);
        auto status = drift_analyzer_->Initialize();
        if (!status.ok()) {
            return status;
        }
    }

    // Initialize safety detector
    if (config_.enable_safety_checks) {
        injection_detector_ = std::make_unique<eval::PromptInjectionDetector>(
            config_.safety_config);
        auto status = injection_detector_->Initialize();
        if (!status.ok()) {
            return status;
        }
    }

    // Initialize similarity scorer
    if (config_.enable_evaluations) {
        similarity_scorer_ = std::make_unique<eval::SemanticSimilarityScorer>(
            config_.similarity_config);
        auto status = similarity_scorer_->Initialize();
        if (!status.ok()) {
            return status;
        }
    }

    // Initialize RCA service
    if (config_.enable_rca) {
        rca_service_ = std::make_unique<rca::RCAService>(
            clickhouse_, qdrant_, redis_, config_.rca_config);
        auto status = rca_service_->Initialize();
        if (!status.ok()) {
            return status;
        }
    }

    // Initialize alert service
    if (config_.enable_alerting) {
        alert_service_ = std::make_unique<alerting::AlertService>(
            clickhouse_, redis_, config_.alert_config);
        auto status = alert_service_->Initialize();
        if (!status.ok()) {
            return status;
        }
    }

    initialized_ = true;
    return absl::OkStatus();
}

absl::Status IntelligencePipeline::Start() {
    if (!initialized_) {
        return absl::FailedPreconditionError("Pipeline not initialized");
    }

    if (running_.load()) {
        return absl::OkStatus();
    }

    running_.store(true);

    // Start async processing workers
    for (size_t i = 0; i < config_.worker_threads; i++) {
        workers_.emplace_back(&IntelligencePipeline::AsyncProcessingLoop, this);
    }

    // Start alert service
    if (alert_service_) {
        alert_service_->Start();
    }

    return absl::OkStatus();
}

void IntelligencePipeline::Stop() {
    if (!running_.load()) {
        return;
    }

    running_.store(false);
    queue_cv_.notify_all();

    for (auto& worker : workers_) {
        if (worker.joinable()) {
            worker.join();
        }
    }
    workers_.clear();

    if (alert_service_) {
        alert_service_->Stop();
    }
}

IntelligenceResult IntelligencePipeline::Process(
    const eval::InferenceRecord& record) {

    auto start_time = std::chrono::steady_clock::now();

    IntelligenceResult result;
    result.trace_id = record.trace_id;
    result.model_id = record.model_id;
    result.analyzed_at = std::chrono::system_clock::now();

    // Run all analysis phases
    if (config_.enable_drift_detection) {
        RunDriftDetection(record, result);
    }

    if (config_.enable_evaluations) {
        RunEvaluations(record, result);
    }

    if (config_.enable_safety_checks) {
        RunSafetyChecks(record, result);
    }

    if (config_.enable_rca) {
        RunRCAIfNeeded(record, result);
    }

    if (config_.enable_alerting) {
        GenerateAlerts(record, result);
    }

    // Compute overall health score
    result.health_score = ComputeHealthScore(result);

    // Record processing time
    auto end_time = std::chrono::steady_clock::now();
    result.processing_time = std::chrono::duration_cast<std::chrono::milliseconds>(
        end_time - start_time);

    // Update statistics
    {
        std::lock_guard<std::mutex> lock(stats_mutex_);
        stats_.total_processed++;
        stats_.last_processed = std::chrono::system_clock::now();

        if (result.drift.drift_detected) stats_.drift_detections++;
        if (!result.safety.is_safe) stats_.safety_issues++;
        if (!result.evaluation.passed) stats_.evaluation_failures++;
        if (result.rca.rca_triggered) stats_.rca_triggered++;
        stats_.alerts_generated += result.alerts.size();

        // Update processing time stats
        double proc_time = result.processing_time.count();
        processing_times_.push_back(proc_time);
        if (processing_times_.size() > 1000) {
            processing_times_.erase(processing_times_.begin());
        }

        stats_.avg_processing_time_ms =
            (stats_.avg_processing_time_ms * (stats_.total_processed - 1) + proc_time) /
            stats_.total_processed;

        // Calculate p99
        if (processing_times_.size() >= 100) {
            std::vector<double> sorted = processing_times_;
            std::sort(sorted.begin(), sorted.end());
            size_t p99_idx = static_cast<size_t>(sorted.size() * 0.99);
            stats_.p99_processing_time_ms = sorted[p99_idx];
        }
    }

    // Update model health
    {
        std::lock_guard<std::mutex> lock(health_mutex_);
        auto& health = model_health_[record.model_id];
        health.model_id = record.model_id;
        health.health_score = result.health_score;
        health.has_active_drift = result.drift.drift_detected;
        health.active_alerts = result.alerts.size();
        health.recent_safety_issues += result.safety.is_safe ? 0 : 1;
        health.last_analyzed = result.analyzed_at;
    }

    // Persist if enabled
    if (config_.persist_results) {
        PersistResult(result);
    }

    // Notify callbacks
    {
        std::lock_guard<std::mutex> lock(callbacks_mutex_);
        for (const auto& callback : result_callbacks_) {
            callback(result);
        }
    }

    return result;
}

BatchIntelligenceResult IntelligencePipeline::ProcessBatch(
    const std::vector<eval::InferenceRecord>& records) {

    BatchIntelligenceResult batch_result;
    batch_result.batch_start = std::chrono::system_clock::now();

    double total_health = 0.0;
    double total_time = 0.0;

    for (const auto& record : records) {
        auto result = Process(record);
        batch_result.results.push_back(result);
        batch_result.total_processed++;

        if (result.drift.drift_detected) batch_result.drift_detected_count++;
        if (!result.safety.is_safe) batch_result.safety_issues_count++;
        if (!result.evaluation.passed) batch_result.evaluation_failures++;

        total_health += result.health_score;
        total_time += result.processing_time.count();
    }

    batch_result.batch_end = std::chrono::system_clock::now();

    if (!records.empty()) {
        batch_result.avg_health_score = total_health / records.size();
        batch_result.avg_processing_time_ms = total_time / records.size();
    }

    return batch_result;
}

void IntelligencePipeline::ProcessAsync(
    const eval::InferenceRecord& record,
    std::function<void(const IntelligenceResult&)> callback) {

    {
        std::lock_guard<std::mutex> lock(queue_mutex_);
        async_queue_.push({record, std::move(callback)});
    }
    queue_cv_.notify_one();
}

absl::Status IntelligencePipeline::RegisterModel(
    const std::string& model_id,
    const drift::ReferenceData& reference) {

    if (!drift_analyzer_) {
        return absl::FailedPreconditionError("Drift analyzer not enabled");
    }

    return drift_analyzer_->RegisterModel(model_id, reference);
}

absl::Status IntelligencePipeline::UpdateModelReference(
    const std::string& model_id,
    const drift::ReferenceData& reference) {

    if (!drift_analyzer_) {
        return absl::FailedPreconditionError("Drift analyzer not enabled");
    }

    return drift_analyzer_->UpdateReference(model_id, reference);
}

absl::Status IntelligencePipeline::RemoveModel(const std::string& model_id) {
    if (!drift_analyzer_) {
        return absl::FailedPreconditionError("Drift analyzer not enabled");
    }

    return drift_analyzer_->RemoveModel(model_id);
}

std::vector<std::string> IntelligencePipeline::ListModels() const {
    if (!drift_analyzer_) {
        return {};
    }

    return drift_analyzer_->ListModels();
}

absl::Status IntelligencePipeline::AddEvaluator(
    const std::string& name,
    std::unique_ptr<eval::Evaluator> evaluator) {

    std::lock_guard<std::mutex> lock(evaluators_mutex_);

    if (evaluators_.count(name) > 0) {
        return absl::AlreadyExistsError("Evaluator already exists: " + name);
    }

    evaluators_[name] = std::move(evaluator);
    evaluator_enabled_[name] = true;

    return absl::OkStatus();
}

absl::Status IntelligencePipeline::RemoveEvaluator(const std::string& name) {
    std::lock_guard<std::mutex> lock(evaluators_mutex_);

    auto it = evaluators_.find(name);
    if (it == evaluators_.end()) {
        return absl::NotFoundError("Evaluator not found: " + name);
    }

    evaluators_.erase(it);
    evaluator_enabled_.erase(name);

    return absl::OkStatus();
}

std::vector<std::string> IntelligencePipeline::ListEvaluators() const {
    std::lock_guard<std::mutex> lock(evaluators_mutex_);

    std::vector<std::string> names;
    for (const auto& [name, _] : evaluators_) {
        names.push_back(name);
    }

    return names;
}

absl::Status IntelligencePipeline::SetEvaluatorEnabled(
    const std::string& name, bool enabled) {

    std::lock_guard<std::mutex> lock(evaluators_mutex_);

    if (evaluators_.count(name) == 0) {
        return absl::NotFoundError("Evaluator not found: " + name);
    }

    evaluator_enabled_[name] = enabled;
    return absl::OkStatus();
}

absl::Status IntelligencePipeline::AddAlertRule(const alerting::AlertRule& rule) {
    if (!alert_service_) {
        return absl::FailedPreconditionError("Alert service not enabled");
    }

    return alert_service_->AddRule(rule);
}

absl::Status IntelligencePipeline::RemoveAlertRule(const std::string& rule_id) {
    if (!alert_service_) {
        return absl::FailedPreconditionError("Alert service not enabled");
    }

    return alert_service_->RemoveRule(rule_id);
}

std::vector<alerting::AlertRule> IntelligencePipeline::ListAlertRules() const {
    if (!alert_service_) {
        return {};
    }

    return alert_service_->ListRules();
}

absl::StatusOr<rca::RCAReport> IntelligencePipeline::TriggerRCA(
    const std::string& model_id) {

    if (!rca_service_) {
        return absl::FailedPreconditionError("RCA service not enabled");
    }

    return rca_service_->Analyze(model_id);
}

absl::StatusOr<rca::RCAReport> IntelligencePipeline::GetRCAReport(
    const std::string& report_id) {

    if (!rca_service_) {
        return absl::FailedPreconditionError("RCA service not enabled");
    }

    return rca_service_->GetReport(report_id);
}

std::vector<rca::RCAReport> IntelligencePipeline::ListRCAReports(
    const std::string& model_id,
    size_t limit) {

    if (!rca_service_) {
        return {};
    }

    auto result = rca_service_->ListReports(model_id, limit);
    if (!result.ok()) {
        return {};
    }

    return *result;
}

absl::StatusOr<IntelligencePipeline::ModelHealthSummary>
IntelligencePipeline::GetModelHealth(const std::string& model_id) const {

    std::lock_guard<std::mutex> lock(health_mutex_);

    auto it = model_health_.find(model_id);
    if (it == model_health_.end()) {
        return absl::NotFoundError("Model not found: " + model_id);
    }

    return it->second;
}

IntelligencePipeline::SystemHealthSummary
IntelligencePipeline::GetSystemHealth() const {

    std::lock_guard<std::mutex> lock(health_mutex_);

    SystemHealthSummary summary;
    summary.last_update = std::chrono::system_clock::now();

    if (model_health_.empty()) {
        return summary;
    }

    double total_health = 0.0;
    for (const auto& [id, health] : model_health_) {
        total_health += health.health_score;
        if (health.has_active_drift) summary.models_with_drift++;
        summary.total_active_alerts += health.active_alerts;
        summary.models_analyzed++;
    }

    summary.avg_health_score = total_health / model_health_.size();
    summary.overall_health = summary.avg_health_score;

    return summary;
}

void IntelligencePipeline::OnResult(
    std::function<void(const IntelligenceResult&)> callback) {

    std::lock_guard<std::mutex> lock(callbacks_mutex_);
    result_callbacks_.push_back(std::move(callback));
}

void IntelligencePipeline::OnDrift(
    std::function<void(const std::string&, const drift::MultiDimensionalDriftStatus&)> callback) {

    std::lock_guard<std::mutex> lock(callbacks_mutex_);
    drift_callbacks_.push_back(std::move(callback));
}

void IntelligencePipeline::OnSafetyIssue(
    std::function<void(const eval::InjectionDetectionResult&)> callback) {

    std::lock_guard<std::mutex> lock(callbacks_mutex_);
    safety_callbacks_.push_back(std::move(callback));
}

void IntelligencePipeline::ClearCallbacks() {
    std::lock_guard<std::mutex> lock(callbacks_mutex_);
    result_callbacks_.clear();
    drift_callbacks_.clear();
    safety_callbacks_.clear();
}

IntelligencePipeline::Stats IntelligencePipeline::GetStats() const {
    std::lock_guard<std::mutex> lock(stats_mutex_);

    Stats current = stats_;

    // Update component health
    current.drift_service_healthy = drift_analyzer_ != nullptr;
    current.eval_service_healthy = similarity_scorer_ != nullptr;
    current.rca_service_healthy = rca_service_ != nullptr;
    current.alert_service_healthy = alert_service_ != nullptr;

    {
        std::lock_guard<std::mutex> queue_lock(queue_mutex_);
        current.queue_depth = async_queue_.size();
    }

    return current;
}

void IntelligencePipeline::ResetStats() {
    std::lock_guard<std::mutex> lock(stats_mutex_);
    stats_ = Stats{};
    processing_times_.clear();
}

void IntelligencePipeline::SetConfig(IntelligencePipelineConfig config) {
    config_ = std::move(config);

    // Update component configs
    if (drift_analyzer_) {
        drift_analyzer_->SetConfig(config_.drift_config);
    }
    if (injection_detector_) {
        injection_detector_->SetConfig(config_.safety_config);
    }
    if (similarity_scorer_) {
        similarity_scorer_->SetConfig(config_.similarity_config);
    }
    if (rca_service_) {
        rca_service_->SetConfig(config_.rca_config);
    }
    if (alert_service_) {
        alert_service_->SetConfig(config_.alert_config);
    }
}

// Private methods

void IntelligencePipeline::RunDriftDetection(
    const eval::InferenceRecord& record,
    IntelligenceResult& result) {

    if (!drift_analyzer_) return;

    auto drift_result = drift_analyzer_->Analyze(record.model_id, {record});
    if (!drift_result.ok()) return;

    const auto& status = *drift_result;
    result.drift.drift_detected = status.has_drift;
    result.drift.overall_severity = status.overall_severity;

    if (status.feature_drift.has_drift) {
        result.drift.drifted_dimensions.push_back("feature");
    }
    if (status.embedding_drift.has_drift) {
        result.drift.drifted_dimensions.push_back("embedding");
    }
    if (status.concept_drift.has_drift) {
        result.drift.drifted_dimensions.push_back("concept");
    }
    if (status.prediction_drift.has_drift) {
        result.drift.drifted_dimensions.push_back("prediction");
    }

    result.drift.causes = status.potential_causes;

    // Notify drift callbacks
    if (status.has_drift) {
        std::lock_guard<std::mutex> lock(callbacks_mutex_);
        for (const auto& callback : drift_callbacks_) {
            callback(record.model_id, status);
        }
    }
}

void IntelligencePipeline::RunEvaluations(
    const eval::InferenceRecord& record,
    IntelligenceResult& result) {

    double total_score = 0.0;
    size_t eval_count = 0;

    // Run similarity scoring
    if (similarity_scorer_ && record.expected_output.has_value()) {
        auto sim_result = similarity_scorer_->Compare(
            record.output, *record.expected_output);
        if (sim_result.ok()) {
            result.evaluation.evaluator_scores["similarity"] = sim_result->score;
            total_score += sim_result->score;
            eval_count++;

            if (sim_result->score < 0.7) {
                result.evaluation.issues.push_back(
                    "Low similarity to expected output");
                result.evaluation.passed = false;
            }
        }
    }

    // Run custom evaluators
    {
        std::lock_guard<std::mutex> lock(evaluators_mutex_);
        for (const auto& [name, evaluator] : evaluators_) {
            if (!evaluator_enabled_[name]) continue;

            auto eval_result = evaluator->Evaluate(record);
            if (eval_result.ok()) {
                result.evaluation.evaluator_scores[name] = eval_result->score;
                total_score += eval_result->score;
                eval_count++;

                if (!eval_result->passed) {
                    result.evaluation.issues.push_back(
                        name + ": " + eval_result->explanation);
                    result.evaluation.passed = false;
                }
            }
        }
    }

    if (eval_count > 0) {
        result.evaluation.overall_score = total_score / eval_count;
    }
}

void IntelligencePipeline::RunSafetyChecks(
    const eval::InferenceRecord& record,
    IntelligenceResult& result) {

    if (!injection_detector_) return;

    auto detection = injection_detector_->Detect(record.input);
    if (!detection.ok()) return;

    const auto& detection_result = *detection;
    result.safety.is_safe = !detection_result.injection_detected;
    result.safety.risk_score = detection_result.confidence;

    if (detection_result.injection_detected) {
        result.safety.risk_level = eval::InjectionRiskLevelToString(
            detection_result.risk_level);

        for (const auto& issue : detection_result.detected_types) {
            result.safety.detected_issues.push_back(
                eval::InjectionTypeToString(issue));
        }

        // Notify safety callbacks
        std::lock_guard<std::mutex> lock(callbacks_mutex_);
        for (const auto& callback : safety_callbacks_) {
            callback(detection_result);
        }
    }
}

void IntelligencePipeline::RunRCAIfNeeded(
    const eval::InferenceRecord& record,
    IntelligenceResult& result) {

    if (!rca_service_) return;

    bool should_trigger = false;

    // Check if RCA should be triggered
    if (config_.auto_rca_on_drift && result.drift.drift_detected &&
        result.drift.overall_severity >= config_.rca_severity_threshold) {
        should_trigger = true;
    }

    if (config_.auto_rca_on_safety_issues && !result.safety.is_safe &&
        result.safety.risk_score >= config_.rca_severity_threshold) {
        should_trigger = true;
    }

    if (!should_trigger) return;

    // Run quick analysis
    auto rca_result = rca_service_->QuickAnalyze(record.model_id);
    if (!rca_result.ok()) return;

    result.rca.rca_triggered = true;
    result.rca.root_causes = rca_result->root_causes;
    result.rca.recommendations = rca_result->recommendations;
    result.rca.confidence = rca_result->confidence;
}

void IntelligencePipeline::GenerateAlerts(
    const eval::InferenceRecord& record,
    IntelligenceResult& result) {

    if (!alert_service_) return;

    // Convert result to metrics
    auto metrics = ResultToMetrics(result);

    // Ingest metrics (will evaluate rules and generate alerts)
    auto alerts = alert_service_->IngestMetrics(metrics);
    result.alerts = alerts;
}

double IntelligencePipeline::ComputeHealthScore(const IntelligenceResult& result) {
    double score = 1.0;

    // Penalize for drift
    if (result.drift.drift_detected) {
        score -= result.drift.overall_severity * 0.3;
    }

    // Penalize for safety issues
    if (!result.safety.is_safe) {
        score -= result.safety.risk_score * 0.4;
    }

    // Penalize for evaluation failures
    if (!result.evaluation.passed) {
        score -= (1.0 - result.evaluation.overall_score) * 0.3;
    }

    return std::max(0.0, std::min(1.0, score));
}

void IntelligencePipeline::AsyncProcessingLoop() {
    while (running_.load()) {
        AsyncTask task;

        {
            std::unique_lock<std::mutex> lock(queue_mutex_);
            queue_cv_.wait_for(lock, std::chrono::seconds(1), [this] {
                return !async_queue_.empty() || !running_.load();
            });

            if (!running_.load() && async_queue_.empty()) {
                break;
            }

            if (async_queue_.empty()) {
                continue;
            }

            task = std::move(async_queue_.front());
            async_queue_.pop();
        }

        // Process and call callback
        auto result = Process(task.record);
        if (task.callback) {
            task.callback(result);
        }
    }
}

absl::Status IntelligencePipeline::PersistResult(const IntelligenceResult& result) {
    if (!clickhouse_) {
        return absl::FailedPreconditionError("ClickHouse not available");
    }

    auto analyzed_at_sec = std::chrono::duration_cast<std::chrono::seconds>(
        result.analyzed_at.time_since_epoch()).count();

    std::ostringstream query;
    query << "INSERT INTO intelligence_results (trace_id, model_id, analyzed_at, "
          << "health_score, drift_detected, drift_severity, safety_score, "
          << "eval_score, processing_time_ms) VALUES ('"
          << result.trace_id << "', '"
          << result.model_id << "', "
          << "toDateTime(" << analyzed_at_sec << "), "
          << result.health_score << ", "
          << (result.drift.drift_detected ? 1 : 0) << ", "
          << result.drift.overall_severity << ", "
          << (result.safety.is_safe ? 1.0 : result.safety.risk_score) << ", "
          << result.evaluation.overall_score << ", "
          << result.processing_time.count() << ")";

    return clickhouse_->Execute(query.str());
}

std::vector<alerting::MetricValue> IntelligencePipeline::ResultToMetrics(
    const IntelligenceResult& result) {

    std::vector<alerting::MetricValue> metrics;
    auto now = std::chrono::system_clock::now();

    // Health score metric
    alerting::MetricValue health_metric;
    health_metric.name = "model_health_score";
    health_metric.value = result.health_score;
    health_metric.timestamp = now;
    health_metric.labels["model_id"] = result.model_id;
    metrics.push_back(health_metric);

    // Drift severity metric
    if (result.drift.drift_detected) {
        alerting::MetricValue drift_metric;
        drift_metric.name = "drift_severity";
        drift_metric.value = result.drift.overall_severity;
        drift_metric.timestamp = now;
        drift_metric.labels["model_id"] = result.model_id;
        metrics.push_back(drift_metric);
    }

    // Safety score metric
    alerting::MetricValue safety_metric;
    safety_metric.name = "safety_risk_score";
    safety_metric.value = result.safety.risk_score;
    safety_metric.timestamp = now;
    safety_metric.labels["model_id"] = result.model_id;
    metrics.push_back(safety_metric);

    // Evaluation score metric
    alerting::MetricValue eval_metric;
    eval_metric.name = "evaluation_score";
    eval_metric.value = result.evaluation.overall_score;
    eval_metric.timestamp = now;
    eval_metric.labels["model_id"] = result.model_id;
    metrics.push_back(eval_metric);

    return metrics;
}

// Factory function
std::unique_ptr<IntelligencePipeline> CreateIntelligencePipeline(
    std::shared_ptr<storage::ClickHouseClient> clickhouse,
    std::shared_ptr<storage::QdrantClient> qdrant,
    std::shared_ptr<storage::RedisClient> redis,
    IntelligencePipelineConfig config) {

    return std::make_unique<IntelligencePipeline>(
        std::move(clickhouse),
        std::move(qdrant),
        std::move(redis),
        std::move(config));
}

std::string SerializeIntelligenceResult(const IntelligenceResult& result) {
    json j;

    j["trace_id"] = result.trace_id;
    j["model_id"] = result.model_id;
    j["analyzed_at"] = std::chrono::duration_cast<std::chrono::seconds>(
        result.analyzed_at.time_since_epoch()).count();

    // Drift
    j["drift"] = {
        {"drift_detected", result.drift.drift_detected},
        {"overall_severity", result.drift.overall_severity},
        {"drifted_dimensions", result.drift.drifted_dimensions},
        {"causes", result.drift.causes}
    };

    // Evaluation
    j["evaluation"] = {
        {"overall_score", result.evaluation.overall_score},
        {"evaluator_scores", result.evaluation.evaluator_scores},
        {"issues", result.evaluation.issues},
        {"passed", result.evaluation.passed}
    };

    // Safety
    j["safety"] = {
        {"is_safe", result.safety.is_safe},
        {"risk_score", result.safety.risk_score},
        {"detected_issues", result.safety.detected_issues},
        {"risk_level", result.safety.risk_level}
    };

    // RCA
    j["rca"] = {
        {"rca_triggered", result.rca.rca_triggered},
        {"root_causes", result.rca.root_causes},
        {"recommendations", result.rca.recommendations},
        {"confidence", result.rca.confidence}
    };

    j["health_score"] = result.health_score;
    j["processing_time_ms"] = result.processing_time.count();

    return j.dump();
}

absl::StatusOr<IntelligenceResult> DeserializeIntelligenceResult(
    const std::string& json_str) {

    try {
        json j = json::parse(json_str);

        IntelligenceResult result;
        result.trace_id = j.value("trace_id", "");
        result.model_id = j.value("model_id", "");
        result.analyzed_at = std::chrono::system_clock::time_point(
            std::chrono::seconds(j.value("analyzed_at", 0)));

        if (j.contains("drift")) {
            result.drift.drift_detected = j["drift"].value("drift_detected", false);
            result.drift.overall_severity = j["drift"].value("overall_severity", 0.0);
            result.drift.drifted_dimensions = j["drift"].value("drifted_dimensions",
                std::vector<std::string>{});
            result.drift.causes = j["drift"].value("causes", std::vector<std::string>{});
        }

        if (j.contains("evaluation")) {
            result.evaluation.overall_score = j["evaluation"].value("overall_score", 0.0);
            result.evaluation.evaluator_scores = j["evaluation"].value("evaluator_scores",
                std::unordered_map<std::string, double>{});
            result.evaluation.issues = j["evaluation"].value("issues",
                std::vector<std::string>{});
            result.evaluation.passed = j["evaluation"].value("passed", true);
        }

        if (j.contains("safety")) {
            result.safety.is_safe = j["safety"].value("is_safe", true);
            result.safety.risk_score = j["safety"].value("risk_score", 0.0);
            result.safety.detected_issues = j["safety"].value("detected_issues",
                std::vector<std::string>{});
            result.safety.risk_level = j["safety"].value("risk_level", "");
        }

        if (j.contains("rca")) {
            result.rca.rca_triggered = j["rca"].value("rca_triggered", false);
            result.rca.root_causes = j["rca"].value("root_causes",
                std::vector<std::string>{});
            result.rca.recommendations = j["rca"].value("recommendations",
                std::vector<std::string>{});
            result.rca.confidence = j["rca"].value("confidence", 0.0);
        }

        result.health_score = j.value("health_score", 1.0);
        result.processing_time = std::chrono::milliseconds(
            j.value("processing_time_ms", 0));

        return result;
    } catch (const json::exception& e) {
        return absl::InvalidArgumentError(
            std::string("Failed to parse intelligence result JSON: ") + e.what());
    }
}

}  // namespace pyflare::intelligence
