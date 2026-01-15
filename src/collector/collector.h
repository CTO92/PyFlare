#pragma once

/// @file collector.h
/// @brief PyFlare OTLP Collector service - main orchestration

#include <atomic>
#include <chrono>
#include <condition_variable>
#include <map>
#include <memory>
#include <mutex>
#include <string>
#include <vector>

#include <absl/status/status.h>
#include <absl/status/statusor.h>
#include <spdlog/spdlog.h>

#include "batcher.h"
#include "kafka_exporter.h"
#include "otlp_receiver.h"
#include "sampler.h"
#include "types.h"

namespace pyflare::collector {

/// Complete collector configuration
struct CollectorConfig {
    /// OTLP receiver settings
    OtlpReceiverConfig receiver;

    /// Batcher settings
    BatcherConfig batcher;

    /// Sampler settings
    SamplerConfig sampler;

    /// Kafka exporter settings
    KafkaExporterConfig kafka;

    /// General settings
    struct General {
        /// Number of worker threads for processing
        size_t worker_threads = 8;

        /// Health check endpoint
        std::string health_endpoint = "0.0.0.0:8081";

        /// Metrics endpoint path
        std::string metrics_path = "/metrics";

        /// Service name for internal telemetry
        std::string service_name = "pyflare-collector";

        /// Enable internal metrics
        bool enable_metrics = true;
    };
    General general;

    /// Enrichment settings
    struct Enrichment {
        /// Add host information to spans
        bool add_host_info = true;

        /// Add process information
        bool add_process_info = true;

        /// Normalize timestamps to UTC
        bool normalize_timestamps = true;

        /// Custom attributes to add to all spans
        std::map<std::string, std::string> custom_attributes;
    };
    Enrichment enrichment;

    /// Create default configuration
    static CollectorConfig Default();

    /// Load configuration from YAML file
    static absl::StatusOr<CollectorConfig> LoadFromFile(const std::string& path);

    /// Load configuration with environment variable overrides
    static absl::StatusOr<CollectorConfig> LoadWithEnv(
        const std::string& path,
        const std::string& env_prefix = "PYFLARE_");
};

/// Collector statistics
struct CollectorStats {
    // Receiver stats
    uint64_t spans_received = 0;
    uint64_t metrics_received = 0;
    uint64_t logs_received = 0;
    uint64_t bytes_received = 0;

    // Processing stats
    uint64_t spans_processed = 0;
    uint64_t spans_sampled = 0;
    uint64_t spans_dropped_sampling = 0;
    uint64_t spans_dropped_queue_full = 0;

    // Export stats
    uint64_t spans_exported = 0;
    uint64_t batches_exported = 0;
    uint64_t export_errors = 0;
    uint64_t bytes_exported = 0;

    // Timing
    std::chrono::steady_clock::time_point start_time;
    std::chrono::steady_clock::time_point last_export_time;
    double avg_processing_latency_ms = 0.0;
    double avg_export_latency_ms = 0.0;

    // Health
    bool is_healthy = false;
    bool is_ready = false;

    // Uptime in seconds
    double UptimeSeconds() const {
        auto now = std::chrono::steady_clock::now();
        return std::chrono::duration<double>(now - start_time).count();
    }
};

/// Main PyFlare Collector class
class Collector {
public:
    /// Create collector with configuration
    explicit Collector(CollectorConfig config);

    /// Destructor
    ~Collector();

    // Non-copyable, non-movable
    Collector(const Collector&) = delete;
    Collector& operator=(const Collector&) = delete;
    Collector(Collector&&) = delete;
    Collector& operator=(Collector&&) = delete;

    /// Start the collector (non-blocking)
    absl::Status Start();

    /// Shutdown the collector gracefully
    absl::Status Shutdown();

    /// Wait for shutdown signal (blocks)
    void WaitForShutdown();

    /// Request shutdown (non-blocking)
    void RequestShutdown();

    /// Check if collector is healthy
    bool IsHealthy() const;

    /// Check if collector is ready to receive traffic
    bool IsReady() const;

    /// Get collector statistics
    CollectorStats GetStats() const;

    /// Get configuration
    const CollectorConfig& GetConfig() const { return config_; }

private:
    /// Process incoming spans from receiver
    void OnSpansReceived(std::vector<Span>&& spans);

    /// Process incoming metrics
    void OnMetricsReceived(std::vector<MetricDataPoint>&& metrics);

    /// Process incoming logs
    void OnLogsReceived(std::vector<LogRecord>&& logs);

    /// Process a batch ready for export
    void OnBatchReady(std::vector<Span>&& batch);

    /// Enrich spans with additional metadata
    void EnrichSpans(std::vector<Span>& spans);

    /// Sample spans
    std::vector<Span> SampleSpans(std::vector<Span>&& spans);

    /// Update internal statistics
    void UpdateStats();

    /// Initialize host information
    void InitHostInfo();

    CollectorConfig config_;

    // Components
    std::unique_ptr<OtlpReceiver> receiver_;
    std::unique_ptr<Sampler> sampler_;
    std::unique_ptr<Batcher> batcher_;
    std::unique_ptr<KafkaExporter> kafka_exporter_;

    // State
    std::atomic<bool> running_{false};
    std::atomic<bool> ready_{false};
    std::atomic<bool> shutdown_requested_{false};

    // Shutdown coordination
    std::mutex shutdown_mutex_;
    std::condition_variable shutdown_cv_;

    // Statistics
    mutable std::mutex stats_mutex_;
    CollectorStats stats_;

    // Host information for enrichment
    std::string hostname_;
    std::string process_id_;

    // Logger
    std::shared_ptr<spdlog::logger> logger_;
};

}  // namespace pyflare::collector
