/// @file drift_service.cpp
/// @brief Drift detection orchestration service implementation

#include "processor/drift/drift_service.h"

#include <algorithm>
#include <chrono>
#include <random>
#include <sstream>

#include <nlohmann/json.hpp>

namespace pyflare::drift {

using json = nlohmann::json;

// ============================================================================
// Kafka Implementation Stub
// ============================================================================

class DriftDetectorService::KafkaImpl {
public:
    explicit KafkaImpl(const DriftServiceConfig::KafkaConfig& config)
        : config_(config) {}

    absl::Status Initialize() {
        // TODO: Initialize rdkafka consumer and producer
        // For now, return OK to allow testing without Kafka
        initialized_ = true;
        return absl::OkStatus();
    }

    absl::Status StartConsumer() {
        if (!initialized_) {
            return absl::FailedPreconditionError("Kafka not initialized");
        }
        consuming_ = true;
        return absl::OkStatus();
    }

    absl::Status StopConsumer() {
        consuming_ = false;
        return absl::OkStatus();
    }

    bool IsConnected() const { return initialized_ && connected_; }
    bool IsConsuming() const { return consuming_; }

    absl::StatusOr<std::vector<storage::TraceRecord>> Poll(
        std::chrono::milliseconds timeout) {
        // TODO: Implement actual Kafka polling
        // For now, return empty vector
        std::this_thread::sleep_for(std::min(timeout, std::chrono::milliseconds(100)));
        return std::vector<storage::TraceRecord>{};
    }

    absl::Status Produce(const std::string& topic,
                         const std::string& key,
                         const std::string& value) {
        // TODO: Implement actual Kafka producing
        return absl::OkStatus();
    }

    absl::Status Flush(std::chrono::milliseconds timeout) {
        // TODO: Implement actual flush
        return absl::OkStatus();
    }

private:
    DriftServiceConfig::KafkaConfig config_;
    bool initialized_ = false;
    bool connected_ = false;
    bool consuming_ = false;
};

// ============================================================================
// DriftDetectorService Implementation
// ============================================================================

DriftDetectorService::DriftDetectorService(
    DriftServiceConfig config,
    std::shared_ptr<storage::ClickHouseClient> clickhouse,
    std::shared_ptr<storage::QdrantClient> qdrant,
    std::shared_ptr<storage::RedisClient> redis)
    : config_(std::move(config)),
      clickhouse_(std::move(clickhouse)),
      qdrant_(std::move(qdrant)),
      redis_(std::move(redis)) {
    metrics_.started_at = std::chrono::system_clock::now();
}

DriftDetectorService::~DriftDetectorService() {
    if (running_) {
        Stop();
    }
}

absl::Status DriftDetectorService::Initialize() {
    if (initialized_) {
        return absl::OkStatus();
    }

    // Initialize Kafka
    kafka_impl_ = std::make_unique<KafkaImpl>(config_.kafka);
    auto kafka_status = kafka_impl_->Initialize();
    if (!kafka_status.ok()) {
        return kafka_status;
    }

    // Initialize multi-dimensional analyzer
    analyzer_ = std::make_unique<MultiDimDriftAnalyzer>(
        clickhouse_, qdrant_, redis_, config_.analyzer);

    auto analyzer_status = analyzer_->Initialize();
    if (!analyzer_status.ok()) {
        return analyzer_status;
    }

    // Register drift callback
    analyzer_->OnDriftDetected([this](const MultiDimensionalDriftStatus& status) {
        HandleDriftDetected(status.model_id, status);
    });

    initialized_ = true;
    return absl::OkStatus();
}

absl::Status DriftDetectorService::Start() {
    if (!initialized_) {
        return absl::FailedPreconditionError("Service not initialized");
    }

    if (running_) {
        return absl::AlreadyExistsError("Service already running");
    }

    running_ = true;

    // Start Kafka consumer
    auto kafka_status = kafka_impl_->StartConsumer();
    if (!kafka_status.ok()) {
        running_ = false;
        return kafka_status;
    }

    // Start consumer thread
    consumer_thread_ = std::thread(&DriftDetectorService::ConsumerLoop, this);

    // Start analysis thread if interval-based analysis is enabled
    if (config_.analysis.analysis_interval.count() > 0) {
        analysis_thread_ = std::thread(&DriftDetectorService::AnalysisLoop, this);
    }

    // Start metrics thread
    metrics_thread_ = std::thread(&DriftDetectorService::MetricsLoop, this);

    // Update metrics
    {
        std::lock_guard<std::mutex> lock(metrics_mutex_);
        metrics_.is_running = true;
        metrics_.started_at = std::chrono::system_clock::now();
    }

    return absl::OkStatus();
}

absl::Status DriftDetectorService::Stop() {
    if (!running_) {
        return absl::OkStatus();
    }

    running_ = false;

    // Stop Kafka consumer
    if (kafka_impl_) {
        kafka_impl_->StopConsumer();
    }

    // Wait for threads to finish
    if (consumer_thread_.joinable()) {
        consumer_thread_.join();
    }
    if (analysis_thread_.joinable()) {
        analysis_thread_.join();
    }
    if (metrics_thread_.joinable()) {
        metrics_thread_.join();
    }

    // Join worker threads
    for (auto& thread : worker_threads_) {
        if (thread.joinable()) {
            thread.join();
        }
    }
    worker_threads_.clear();

    // Update metrics
    {
        std::lock_guard<std::mutex> lock(metrics_mutex_);
        metrics_.is_running = false;
    }

    return absl::OkStatus();
}

absl::Status DriftDetectorService::RegisterModel(
    const std::string& model_id,
    const ReferenceData& reference) {
    if (!initialized_) {
        return absl::FailedPreconditionError("Service not initialized");
    }

    auto status = analyzer_->RegisterModel(model_id, reference);
    if (status.ok()) {
        std::lock_guard<std::mutex> lock(metrics_mutex_);
        metrics_.models_monitored = analyzer_->GetRegisteredModels().size();
    }
    return status;
}

absl::Status DriftDetectorService::UpdateModelReference(
    const std::string& model_id,
    const ReferenceData& reference) {
    if (!initialized_) {
        return absl::FailedPreconditionError("Service not initialized");
    }

    return analyzer_->UpdateReference(model_id, reference, true);
}

absl::Status DriftDetectorService::UnregisterModel(const std::string& model_id) {
    if (!initialized_) {
        return absl::FailedPreconditionError("Service not initialized");
    }

    auto status = analyzer_->UnregisterModel(model_id);
    if (status.ok()) {
        std::lock_guard<std::mutex> lock(metrics_mutex_);
        metrics_.models_monitored = analyzer_->GetRegisteredModels().size();
    }
    return status;
}

std::vector<std::string> DriftDetectorService::GetRegisteredModels() const {
    if (!analyzer_) {
        return {};
    }
    return analyzer_->GetRegisteredModels();
}

absl::StatusOr<MultiDimensionalDriftStatus> DriftDetectorService::AnalyzeModel(
    const std::string& model_id) {
    if (!initialized_) {
        return absl::FailedPreconditionError("Service not initialized");
    }

    auto end = std::chrono::system_clock::now();
    auto start = end - config_.analysis.analysis_window;

    return analyzer_->AnalyzeTimeRange(model_id, start, end);
}

absl::StatusOr<std::unordered_map<std::string, MultiDimensionalDriftStatus>>
DriftDetectorService::AnalyzeAllModels() {
    if (!initialized_) {
        return absl::FailedPreconditionError("Service not initialized");
    }

    std::unordered_map<std::string, MultiDimensionalDriftStatus> results;

    for (const auto& model_id : analyzer_->GetRegisteredModels()) {
        auto result = AnalyzeModel(model_id);
        if (result.ok()) {
            results[model_id] = *result;
        }
    }

    return results;
}

absl::StatusOr<std::vector<MultiDimensionalDriftStatus>>
DriftDetectorService::GetDriftTrend(
    const std::string& model_id,
    std::chrono::hours lookback) {
    if (!initialized_) {
        return absl::FailedPreconditionError("Service not initialized");
    }

    auto end = std::chrono::system_clock::now();
    auto start = end - lookback;

    return analyzer_->GetDriftTrend(model_id, start, end, std::chrono::hours(1));
}

void DriftDetectorService::OnAlert(std::function<void(const DriftAlert&)> callback) {
    std::lock_guard<std::mutex> lock(callbacks_mutex_);
    alert_callbacks_.push_back(std::move(callback));
}

void DriftDetectorService::OnAnalysisComplete(
    std::function<void(const std::string&, const MultiDimensionalDriftStatus&)> callback) {
    std::lock_guard<std::mutex> lock(callbacks_mutex_);
    analysis_callbacks_.push_back(std::move(callback));
}

void DriftDetectorService::ClearCallbacks() {
    std::lock_guard<std::mutex> lock(callbacks_mutex_);
    alert_callbacks_.clear();
    analysis_callbacks_.clear();
}

DriftServiceMetrics DriftDetectorService::GetMetrics() const {
    std::lock_guard<std::mutex> lock(metrics_mutex_);
    return metrics_;
}

void DriftDetectorService::ResetMetrics() {
    std::lock_guard<std::mutex> lock(metrics_mutex_);
    auto is_running = metrics_.is_running;
    auto started_at = metrics_.started_at;
    auto models_monitored = metrics_.models_monitored;

    metrics_ = DriftServiceMetrics{};
    metrics_.is_running = is_running;
    metrics_.started_at = started_at;
    metrics_.models_monitored = models_monitored;
}

void DriftDetectorService::SetConfig(DriftServiceConfig config) {
    config_ = std::move(config);
}

DriftDetectorService::HealthStatus DriftDetectorService::GetHealthStatus() const {
    HealthStatus status;

    status.is_healthy = initialized_ && running_;

    if (kafka_impl_) {
        status.kafka_connected = kafka_impl_->IsConnected();
    }

    status.analyzer_ready = analyzer_ != nullptr;

    {
        std::lock_guard<std::mutex> lock(health_mutex_);
        status.last_successful_analysis = last_successful_analysis_;
        status.error_message = last_error_;
    }

    return status;
}

// ============================================================================
// Private Implementation
// ============================================================================

void DriftDetectorService::ConsumerLoop() {
    while (running_) {
        try {
            auto records = kafka_impl_->Poll(std::chrono::milliseconds(100));
            if (!records.ok()) {
                std::lock_guard<std::mutex> lock(metrics_mutex_);
                metrics_.consumer_errors++;
                continue;
            }

            for (const auto& record : *records) {
                ProcessRecord(record);
            }

            // Update metrics
            {
                std::lock_guard<std::mutex> lock(metrics_mutex_);
                metrics_.records_consumed += records->size();
                metrics_.last_activity = std::chrono::system_clock::now();
            }

            // Check if any batches should be processed
            {
                std::lock_guard<std::mutex> lock(batches_mutex_);
                for (auto& [model_id, batch] : model_batches_) {
                    if (ShouldProcessBatch(model_id)) {
                        auto batch_copy = FlushBatch(model_id);
                        ProcessBatch(model_id, batch_copy);
                    }
                }
            }

        } catch (const std::exception& e) {
            std::lock_guard<std::mutex> lock(health_mutex_);
            last_error_ = std::string("Consumer error: ") + e.what();

            std::lock_guard<std::mutex> metrics_lock(metrics_mutex_);
            metrics_.consumer_errors++;
        }
    }
}

void DriftDetectorService::AnalysisLoop() {
    while (running_) {
        // Wait for analysis interval
        std::this_thread::sleep_for(config_.analysis.analysis_interval);

        if (!running_) break;

        try {
            auto results = AnalyzeAllModels();
            if (results.ok()) {
                for (const auto& [model_id, status] : *results) {
                    // Invoke callbacks
                    {
                        std::lock_guard<std::mutex> lock(callbacks_mutex_);
                        for (const auto& callback : analysis_callbacks_) {
                            callback(model_id, status);
                        }
                    }
                }

                std::lock_guard<std::mutex> lock(health_mutex_);
                last_successful_analysis_ = std::chrono::system_clock::now();
            }
        } catch (const std::exception& e) {
            std::lock_guard<std::mutex> lock(health_mutex_);
            last_error_ = std::string("Analysis error: ") + e.what();

            std::lock_guard<std::mutex> metrics_lock(metrics_mutex_);
            metrics_.analysis_errors++;
        }
    }
}

void DriftDetectorService::MetricsLoop() {
    while (running_) {
        std::this_thread::sleep_for(config_.metrics_interval);

        if (!running_) break;

        // Update current batch size metrics
        {
            std::lock_guard<std::mutex> lock(batches_mutex_);
            size_t total_batch_size = 0;
            for (const auto& [_, batch] : model_batches_) {
                total_batch_size += batch.size();
            }

            std::lock_guard<std::mutex> metrics_lock(metrics_mutex_);
            metrics_.current_batch_size = total_batch_size;
        }

        // Update analyzer stats
        if (analyzer_) {
            auto analyzer_stats = analyzer_->GetStats();
            std::lock_guard<std::mutex> lock(metrics_mutex_);
            metrics_.analyses_performed = analyzer_stats.total_analyses;
        }
    }
}

void DriftDetectorService::ProcessRecord(const storage::TraceRecord& record) {
    // Convert trace record to data point
    DataPoint point;
    point.id = record.trace_id;
    point.attributes["input"] = record.input_preview;
    point.attributes["output"] = record.output_preview;
    point.attributes["model_id"] = record.model_id;
    point.attributes["status"] = record.status_code;
    point.attributes["correct"] = (record.status_code == "OK") ? "true" : "false";

    // Extract features if available
    auto it = record.attributes.find("features");
    if (it != record.attributes.end()) {
        // Parse feature vector from JSON
        try {
            auto features = json::parse(it->second);
            if (features.is_array()) {
                for (const auto& f : features) {
                    point.features.push_back(f.get<double>());
                }
            }
        } catch (...) {
            // Ignore parse errors
        }
    }

    // Extract embedding if available
    it = record.attributes.find("embedding");
    if (it != record.attributes.end()) {
        try {
            auto embedding = json::parse(it->second);
            if (embedding.is_array()) {
                for (const auto& e : embedding) {
                    point.embedding.push_back(e.get<float>());
                }
            }
        } catch (...) {
            // Ignore parse errors
        }
    }

    std::string model_id = record.model_id;
    if (model_id.empty()) {
        model_id = "default";
    }

    // Check if we should monitor this model
    if (!config_.analysis.monitored_models.empty()) {
        auto& monitored = config_.analysis.monitored_models;
        if (std::find(monitored.begin(), monitored.end(), model_id) == monitored.end()) {
            return;
        }
    }

    AddToBatch(model_id, std::move(point));

    // Update streaming concept drift if enabled
    if (config_.analysis.enable_streaming_concept_drift) {
        bool correct = record.status_code == "OK";
        UpdateStreamingConceptDrift(model_id, correct);
    }

    // Update metrics
    {
        std::lock_guard<std::mutex> lock(metrics_mutex_);
        metrics_.records_processed++;
    }
}

void DriftDetectorService::ProcessBatch(
    const std::string& model_id,
    std::vector<DataPoint>& batch) {
    if (batch.size() < config_.analysis.min_samples) {
        return;
    }

    auto start_time = std::chrono::steady_clock::now();

    try {
        auto result = analyzer_->Analyze(model_id, batch);
        if (result.ok()) {
            // Invoke analysis callbacks
            {
                std::lock_guard<std::mutex> lock(callbacks_mutex_);
                for (const auto& callback : analysis_callbacks_) {
                    callback(model_id, *result);
                }
            }

            {
                std::lock_guard<std::mutex> lock(health_mutex_);
                last_successful_analysis_ = std::chrono::system_clock::now();
            }
        }
    } catch (const std::exception& e) {
        std::lock_guard<std::mutex> lock(health_mutex_);
        last_error_ = std::string("Batch processing error: ") + e.what();

        std::lock_guard<std::mutex> metrics_lock(metrics_mutex_);
        metrics_.analysis_errors++;
    }

    auto end_time = std::chrono::steady_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(
        end_time - start_time);

    // Update metrics
    {
        std::lock_guard<std::mutex> lock(metrics_mutex_);
        metrics_.batches_processed++;
        metrics_.avg_batch_process_time_ms =
            (metrics_.avg_batch_process_time_ms * (metrics_.batches_processed - 1) +
             duration.count()) / metrics_.batches_processed;
    }
}

void DriftDetectorService::HandleDriftDetected(
    const std::string& model_id,
    const MultiDimensionalDriftStatus& status) {
    if (!ShouldGenerateAlert(model_id, status)) {
        return;
    }

    auto alert = CreateAlert(model_id, status);

    // Update last alert time
    {
        std::lock_guard<std::mutex> lock(alerts_mutex_);
        last_alerts_[model_id] = std::chrono::system_clock::now();
    }

    // Invoke alert callbacks
    {
        std::lock_guard<std::mutex> lock(callbacks_mutex_);
        for (const auto& callback : alert_callbacks_) {
            callback(alert);
        }
    }

    // Publish to Kafka
    if (config_.alert.enable_kafka_alerts) {
        auto publish_status = PublishAlert(alert);
        if (!publish_status.ok()) {
            std::lock_guard<std::mutex> lock(metrics_mutex_);
            metrics_.alert_publish_errors++;
        }
    }

    // Update metrics
    {
        std::lock_guard<std::mutex> lock(metrics_mutex_);
        metrics_.alerts_generated++;
        metrics_.drift_detections++;
        metrics_.detections_by_model[model_id]++;

        for (const auto& score : status.dimension_scores) {
            if (score.is_drifted) {
                metrics_.detections_by_dimension[score.dimension_type]++;
            }
        }
    }
}

DriftAlert DriftDetectorService::CreateAlert(
    const std::string& model_id,
    const MultiDimensionalDriftStatus& status) {
    DriftAlert alert;
    alert.alert_id = GenerateAlertId();
    alert.model_id = model_id;
    alert.timestamp = status.timestamp;
    alert.severity = status.severity;
    alert.overall_score = status.overall_score;

    for (const auto& score : status.dimension_scores) {
        alert.dimension_scores[score.dimension_type] = score.score;
    }

    alert.likely_causes = status.likely_causes;
    alert.recommendations = status.recommended_actions;
    alert.samples_analyzed = status.samples_analyzed;
    alert.time_window = config_.analysis.analysis_window;

    // Build summary
    std::stringstream summary;
    summary << "Drift detected for model " << model_id << " (severity: " << status.severity;
    summary << ", score: " << std::fixed << std::setprecision(2) << status.overall_score << ")";

    std::vector<std::string> drifted_dims;
    for (const auto& score : status.dimension_scores) {
        if (score.is_drifted) {
            drifted_dims.push_back(score.dimension_type);
        }
    }
    if (!drifted_dims.empty()) {
        summary << ". Affected dimensions: ";
        for (size_t i = 0; i < drifted_dims.size(); ++i) {
            if (i > 0) summary << ", ";
            summary << drifted_dims[i];
        }
    }

    alert.summary = summary.str();

    return alert;
}

absl::Status DriftDetectorService::PublishAlert(const DriftAlert& alert) {
    if (!kafka_impl_) {
        return absl::FailedPreconditionError("Kafka not initialized");
    }

    std::string payload = SerializeAlert(alert);

    return kafka_impl_->Produce(
        config_.kafka.alert_topic,
        alert.model_id,
        payload);
}

bool DriftDetectorService::ShouldGenerateAlert(
    const std::string& model_id,
    const MultiDimensionalDriftStatus& status) {
    if (!status.has_any_drift) {
        return false;
    }

    // Check severity threshold
    auto severity_order = [](const std::string& s) {
        if (s == "critical") return 4;
        if (s == "high") return 3;
        if (s == "medium") return 2;
        if (s == "low") return 1;
        return 0;
    };

    if (severity_order(status.severity) < severity_order(config_.alert.min_severity)) {
        return false;
    }

    // Check cooldown
    {
        std::lock_guard<std::mutex> lock(alerts_mutex_);
        auto it = last_alerts_.find(model_id);
        if (it != last_alerts_.end()) {
            auto elapsed = std::chrono::system_clock::now() - it->second;
            if (elapsed < config_.alert.alert_cooldown) {
                return false;
            }
        }
    }

    return true;
}

void DriftDetectorService::AddToBatch(const std::string& model_id, DataPoint&& point) {
    std::lock_guard<std::mutex> lock(batches_mutex_);

    auto& batch = model_batches_[model_id];
    if (batch.empty()) {
        batch_start_times_[model_id] = std::chrono::steady_clock::now();
    }

    batch.push_back(std::move(point));
}

bool DriftDetectorService::ShouldProcessBatch(const std::string& model_id) {
    // Must be called with batches_mutex_ held

    auto it = model_batches_.find(model_id);
    if (it == model_batches_.end()) {
        return false;
    }

    // Check max batch size
    if (it->second.size() >= config_.batch.max_batch_size) {
        return true;
    }

    // Check min batch size
    if (it->second.size() < config_.batch.min_batch_size) {
        // Check timeout
        auto time_it = batch_start_times_.find(model_id);
        if (time_it != batch_start_times_.end()) {
            auto elapsed = std::chrono::steady_clock::now() - time_it->second;
            if (elapsed >= config_.batch.batch_timeout) {
                return true;
            }
        }
        return false;
    }

    return true;
}

std::vector<DataPoint> DriftDetectorService::FlushBatch(const std::string& model_id) {
    // Must be called with batches_mutex_ held

    std::vector<DataPoint> batch;
    auto it = model_batches_.find(model_id);
    if (it != model_batches_.end()) {
        batch = std::move(it->second);
        it->second.clear();
    }

    batch_start_times_.erase(model_id);

    return batch;
}

void DriftDetectorService::UpdateStreamingConceptDrift(
    const std::string& model_id,
    bool correct) {
    // Update the concept drift detector in streaming mode
    // This requires access to the analyzer's internal detectors
    // For now, this is a placeholder - the analyzer handles this internally

    // The analyzer's concept drift detector can be updated through
    // the AnalyzeClassification method or similar
}

std::string DriftDetectorService::GenerateAlertId() {
    static std::random_device rd;
    static std::mt19937 gen(rd());
    static std::uniform_int_distribution<uint64_t> dis;

    std::stringstream ss;
    ss << "drift-alert-" << std::hex << dis(gen);
    return ss.str();
}

std::string DriftDetectorService::SerializeAlert(const DriftAlert& alert) {
    json j;
    j["alert_id"] = alert.alert_id;
    j["model_id"] = alert.model_id;
    j["timestamp"] = std::chrono::system_clock::to_time_t(alert.timestamp);
    j["severity"] = alert.severity;
    j["overall_score"] = alert.overall_score;
    j["dimension_scores"] = alert.dimension_scores;
    j["likely_causes"] = alert.likely_causes;
    j["recommendations"] = alert.recommendations;
    j["summary"] = alert.summary;
    j["samples_analyzed"] = alert.samples_analyzed;
    j["time_window_hours"] = alert.time_window.count();

    return j.dump();
}

// ============================================================================
// Factory Function
// ============================================================================

std::unique_ptr<DriftDetectorService> CreateDriftDetectorService(
    DriftServiceConfig config,
    std::shared_ptr<storage::ClickHouseClient> clickhouse,
    std::shared_ptr<storage::QdrantClient> qdrant,
    std::shared_ptr<storage::RedisClient> redis) {
    return std::make_unique<DriftDetectorService>(
        std::move(config),
        std::move(clickhouse),
        std::move(qdrant),
        std::move(redis));
}

}  // namespace pyflare::drift
