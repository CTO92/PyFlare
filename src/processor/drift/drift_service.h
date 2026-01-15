#pragma once

/// @file drift_service.h
/// @brief Drift detection orchestration service
///
/// Provides a Kafka-based service that:
/// - Consumes trace records from Kafka
/// - Batches records for efficient processing
/// - Runs multi-dimensional drift analysis
/// - Publishes drift alerts to Kafka
/// - Stores results in ClickHouse

#include <atomic>
#include <chrono>
#include <functional>
#include <memory>
#include <mutex>
#include <string>
#include <thread>
#include <unordered_map>
#include <vector>

#include <absl/status/status.h>
#include <absl/status/statusor.h>

#include "processor/drift/multi_dim_analyzer.h"
#include "storage/clickhouse/client.h"
#include "storage/qdrant/client.h"
#include "storage/redis/client.h"

namespace pyflare::drift {

/// @brief Configuration for drift detection service
struct DriftServiceConfig {
    /// Kafka configuration
    struct KafkaConfig {
        std::string bootstrap_servers = "localhost:9092";
        std::string consumer_group = "pyflare-drift-detector";
        std::string input_topic = "pyflare.traces";
        std::string alert_topic = "pyflare.drift-alerts";

        /// Consumer session timeout
        std::chrono::seconds session_timeout = std::chrono::seconds(30);

        /// Maximum poll interval
        std::chrono::seconds max_poll_interval = std::chrono::minutes(5);

        /// Auto offset reset ("earliest", "latest")
        std::string auto_offset_reset = "latest";

        /// Enable auto commit
        bool enable_auto_commit = true;

        /// Auto commit interval
        std::chrono::seconds auto_commit_interval = std::chrono::seconds(5);
    };
    KafkaConfig kafka;

    /// Batch processing configuration
    struct BatchConfig {
        /// Minimum batch size before triggering analysis
        size_t min_batch_size = 100;

        /// Maximum batch size
        size_t max_batch_size = 10000;

        /// Maximum time to wait before processing partial batch
        std::chrono::seconds batch_timeout = std::chrono::seconds(60);

        /// Enable per-model batching (separate batches per model)
        bool per_model_batching = true;
    };
    BatchConfig batch;

    /// Analysis configuration
    struct AnalysisConfig {
        /// Run analysis every N seconds (0 = only on batch threshold)
        std::chrono::seconds analysis_interval = std::chrono::seconds(0);

        /// Analysis time window (lookback)
        std::chrono::hours analysis_window = std::chrono::hours(24);

        /// Minimum samples for analysis
        size_t min_samples = 50;

        /// Enable continuous concept drift monitoring
        bool enable_streaming_concept_drift = true;

        /// Models to monitor (empty = all models)
        std::vector<std::string> monitored_models;
    };
    AnalysisConfig analysis;

    /// Alert configuration
    struct AlertConfig {
        /// Minimum severity to generate alert ("low", "medium", "high", "critical")
        std::string min_severity = "medium";

        /// Cooldown period between alerts for same model
        std::chrono::minutes alert_cooldown = std::chrono::minutes(30);

        /// Enable Kafka alert publishing
        bool enable_kafka_alerts = true;

        /// Enable webhook alerts
        bool enable_webhooks = false;

        /// Webhook URLs
        std::vector<std::string> webhook_urls;
    };
    AlertConfig alert;

    /// Multi-dimensional analyzer configuration
    MultiDimAnalyzerConfig analyzer;

    /// Number of worker threads
    size_t num_workers = 4;

    /// Metrics collection interval
    std::chrono::seconds metrics_interval = std::chrono::seconds(60);
};

/// @brief Drift alert for publishing
struct DriftAlert {
    std::string alert_id;
    std::string model_id;
    std::chrono::system_clock::time_point timestamp;

    std::string severity;  ///< "low", "medium", "high", "critical"
    double overall_score = 0.0;

    /// Drift by dimension
    std::unordered_map<std::string, double> dimension_scores;

    /// Alert details
    std::vector<std::string> likely_causes;
    std::vector<std::string> recommendations;
    std::string summary;

    /// Samples context
    size_t samples_analyzed = 0;
    std::chrono::hours time_window{24};
};

/// @brief Service metrics
struct DriftServiceMetrics {
    // Processing metrics
    size_t records_consumed = 0;
    size_t records_processed = 0;
    size_t batches_processed = 0;
    size_t analyses_performed = 0;
    size_t alerts_generated = 0;

    // Drift metrics
    size_t drift_detections = 0;
    std::unordered_map<std::string, size_t> detections_by_model;
    std::unordered_map<std::string, size_t> detections_by_dimension;

    // Performance metrics
    double avg_batch_process_time_ms = 0.0;
    double avg_analysis_time_ms = 0.0;
    double max_analysis_time_ms = 0.0;

    // Error metrics
    size_t consumer_errors = 0;
    size_t analysis_errors = 0;
    size_t alert_publish_errors = 0;

    // Current state
    bool is_running = false;
    std::chrono::system_clock::time_point started_at;
    std::chrono::system_clock::time_point last_activity;
    size_t current_batch_size = 0;
    size_t models_monitored = 0;
};

/// @brief Drift detection orchestration service
///
/// Consumes traces from Kafka, performs drift analysis, and publishes alerts.
///
/// Example usage:
/// @code
///   DriftServiceConfig config;
///   config.kafka.bootstrap_servers = "kafka:9092";
///   config.kafka.input_topic = "pyflare.traces";
///
///   DriftDetectorService service(config, clickhouse, qdrant, redis);
///   service.Initialize();
///
///   // Register callback for alerts
///   service.OnAlert([](const DriftAlert& alert) {
///       LOG(WARNING) << "Drift detected for " << alert.model_id
///                    << ": " << alert.severity;
///   });
///
///   // Start processing
///   service.Start();
///
///   // ... service runs in background ...
///
///   // Stop when done
///   service.Stop();
/// @endcode
class DriftDetectorService {
public:
    DriftDetectorService(
        DriftServiceConfig config,
        std::shared_ptr<storage::ClickHouseClient> clickhouse,
        std::shared_ptr<storage::QdrantClient> qdrant,
        std::shared_ptr<storage::RedisClient> redis);
    ~DriftDetectorService();

    // Disable copy
    DriftDetectorService(const DriftDetectorService&) = delete;
    DriftDetectorService& operator=(const DriftDetectorService&) = delete;

    /// @brief Initialize the service (create consumers, initialize analyzer)
    absl::Status Initialize();

    /// @brief Start the service (begins consuming and processing)
    absl::Status Start();

    /// @brief Stop the service gracefully
    absl::Status Stop();

    /// @brief Check if service is running
    bool IsRunning() const { return running_.load(); }

    // =========================================================================
    // Model Registration
    // =========================================================================

    /// @brief Register a model with reference data
    absl::Status RegisterModel(
        const std::string& model_id,
        const ReferenceData& reference);

    /// @brief Update reference data for a model
    absl::Status UpdateModelReference(
        const std::string& model_id,
        const ReferenceData& reference);

    /// @brief Unregister a model
    absl::Status UnregisterModel(const std::string& model_id);

    /// @brief Get registered models
    std::vector<std::string> GetRegisteredModels() const;

    // =========================================================================
    // Manual Analysis
    // =========================================================================

    /// @brief Trigger analysis for a specific model
    /// @param model_id Model to analyze
    /// @return Analysis result
    absl::StatusOr<MultiDimensionalDriftStatus> AnalyzeModel(
        const std::string& model_id);

    /// @brief Trigger analysis for all registered models
    /// @return Map of model_id to analysis result
    absl::StatusOr<std::unordered_map<std::string, MultiDimensionalDriftStatus>>
    AnalyzeAllModels();

    /// @brief Get drift trend for a model
    absl::StatusOr<std::vector<MultiDimensionalDriftStatus>> GetDriftTrend(
        const std::string& model_id,
        std::chrono::hours lookback = std::chrono::hours(24));

    // =========================================================================
    // Callbacks
    // =========================================================================

    /// @brief Register callback for drift alerts
    void OnAlert(std::function<void(const DriftAlert&)> callback);

    /// @brief Register callback for analysis completion
    void OnAnalysisComplete(
        std::function<void(const std::string& model_id,
                          const MultiDimensionalDriftStatus&)> callback);

    /// @brief Clear all callbacks
    void ClearCallbacks();

    // =========================================================================
    // Configuration and Metrics
    // =========================================================================

    /// @brief Get current metrics
    DriftServiceMetrics GetMetrics() const;

    /// @brief Reset metrics counters
    void ResetMetrics();

    /// @brief Get configuration
    const DriftServiceConfig& GetConfig() const { return config_; }

    /// @brief Update configuration (requires restart to take effect)
    void SetConfig(DriftServiceConfig config);

    /// @brief Check service health
    struct HealthStatus {
        bool is_healthy = false;
        bool kafka_connected = false;
        bool analyzer_ready = false;
        std::string error_message;
        std::chrono::system_clock::time_point last_successful_analysis;
    };
    HealthStatus GetHealthStatus() const;

private:
    // Worker thread functions
    void ConsumerLoop();
    void AnalysisLoop();
    void MetricsLoop();

    // Record processing
    void ProcessRecord(const storage::TraceRecord& record);
    void ProcessBatch(const std::string& model_id,
                      std::vector<DataPoint>& batch);

    // Alert handling
    void HandleDriftDetected(const std::string& model_id,
                             const MultiDimensionalDriftStatus& status);
    DriftAlert CreateAlert(const std::string& model_id,
                           const MultiDimensionalDriftStatus& status);
    absl::Status PublishAlert(const DriftAlert& alert);
    bool ShouldGenerateAlert(const std::string& model_id,
                             const MultiDimensionalDriftStatus& status);

    // Kafka operations
    absl::Status InitializeKafka();
    absl::Status ConsumeRecords();
    absl::Status PublishToKafka(const std::string& topic,
                                const std::string& key,
                                const std::string& value);

    // Batch management
    void AddToBatch(const std::string& model_id, DataPoint&& point);
    bool ShouldProcessBatch(const std::string& model_id);
    std::vector<DataPoint> FlushBatch(const std::string& model_id);

    // Streaming concept drift
    void UpdateStreamingConceptDrift(const std::string& model_id, bool correct);

    // Utility
    std::string GenerateAlertId();
    std::string SerializeAlert(const DriftAlert& alert);

    // Configuration and state
    DriftServiceConfig config_;
    std::atomic<bool> running_{false};
    std::atomic<bool> initialized_{false};

    // Storage clients
    std::shared_ptr<storage::ClickHouseClient> clickhouse_;
    std::shared_ptr<storage::QdrantClient> qdrant_;
    std::shared_ptr<storage::RedisClient> redis_;

    // Multi-dimensional analyzer
    std::unique_ptr<MultiDimDriftAnalyzer> analyzer_;

    // Batches per model
    std::unordered_map<std::string, std::vector<DataPoint>> model_batches_;
    std::unordered_map<std::string, std::chrono::steady_clock::time_point> batch_start_times_;
    mutable std::mutex batches_mutex_;

    // Alert cooldown tracking
    std::unordered_map<std::string, std::chrono::system_clock::time_point> last_alerts_;
    mutable std::mutex alerts_mutex_;

    // Worker threads
    std::vector<std::thread> worker_threads_;
    std::thread consumer_thread_;
    std::thread analysis_thread_;
    std::thread metrics_thread_;

    // Callbacks
    std::vector<std::function<void(const DriftAlert&)>> alert_callbacks_;
    std::vector<std::function<void(const std::string&, const MultiDimensionalDriftStatus&)>>
        analysis_callbacks_;
    mutable std::mutex callbacks_mutex_;

    // Metrics
    DriftServiceMetrics metrics_;
    mutable std::mutex metrics_mutex_;

    // Health tracking
    std::chrono::system_clock::time_point last_successful_analysis_;
    std::string last_error_;
    mutable std::mutex health_mutex_;

    // Kafka consumer/producer (implementation-specific)
    class KafkaImpl;
    std::unique_ptr<KafkaImpl> kafka_impl_;
};

/// @brief Factory function to create drift detector service
std::unique_ptr<DriftDetectorService> CreateDriftDetectorService(
    DriftServiceConfig config,
    std::shared_ptr<storage::ClickHouseClient> clickhouse,
    std::shared_ptr<storage::QdrantClient> qdrant,
    std::shared_ptr<storage::RedisClient> redis);

}  // namespace pyflare::drift
