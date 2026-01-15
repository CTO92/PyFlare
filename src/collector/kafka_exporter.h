#pragma once

/// @file kafka_exporter.h
/// @brief Kafka exporter for sending telemetry to Kafka topics

#include <atomic>
#include <chrono>
#include <memory>
#include <string>
#include <vector>

#include <absl/status/status.h>

#include "types.h"

namespace pyflare::collector {

/// Kafka exporter configuration
struct KafkaExporterConfig {
    /// Kafka broker addresses
    std::vector<std::string> brokers;

    /// Topic names
    struct Topics {
        std::string traces = "pyflare.traces";
        std::string metrics = "pyflare.metrics";
        std::string logs = "pyflare.logs";
    };
    Topics topics;

    /// Producer settings
    struct Producer {
        size_t batch_size = 16384;           ///< Batch size in bytes
        int linger_ms = 5;                   ///< Time to wait for batch
        std::string compression = "lz4";     ///< none, gzip, snappy, lz4, zstd
        std::string acks = "all";            ///< 0, 1, all
        int retries = 3;                     ///< Number of retries
        int retry_backoff_ms = 100;          ///< Backoff between retries
        bool enable_idempotence = true;      ///< Exactly-once semantics
        int max_in_flight_requests = 5;      ///< Max in-flight requests
    };
    Producer producer;

    /// Timeout settings
    struct Timeouts {
        int message_timeout_ms = 30000;      ///< Message delivery timeout
        int socket_timeout_ms = 60000;       ///< Socket timeout
        int metadata_timeout_ms = 10000;     ///< Metadata request timeout
    };
    Timeouts timeouts;

    /// Security settings
    struct Security {
        std::string protocol = "PLAINTEXT";  ///< PLAINTEXT, SSL, SASL_PLAINTEXT, SASL_SSL

        struct Ssl {
            std::string ca_location;
            std::string cert_location;
            std::string key_location;
            std::string key_password;
        };
        Ssl ssl;

        struct Sasl {
            std::string mechanism = "PLAIN";  ///< PLAIN, SCRAM-SHA-256, SCRAM-SHA-512
            std::string username;
            std::string password;
        };
        Sasl sasl;
    };
    Security security;

    /// Serialization format
    enum class SerializationFormat {
        kProtobuf,
        kJson
    };
    SerializationFormat format = SerializationFormat::kProtobuf;
};

/// Statistics for the exporter
struct KafkaExporterStats {
    std::atomic<uint64_t> messages_sent{0};
    std::atomic<uint64_t> messages_failed{0};
    std::atomic<uint64_t> bytes_sent{0};
    std::atomic<uint64_t> batches_sent{0};
    std::atomic<uint64_t> spans_sent{0};
    std::atomic<uint64_t> metrics_sent{0};
    std::atomic<uint64_t> logs_sent{0};
    std::atomic<uint64_t> retries{0};
    std::atomic<double> avg_latency_ms{0.0};
};

/// Kafka exporter for sending telemetry data
class KafkaExporter {
public:
    /// Create exporter with configuration
    explicit KafkaExporter(KafkaExporterConfig config);

    /// Destructor
    ~KafkaExporter();

    // Non-copyable, non-movable
    KafkaExporter(const KafkaExporter&) = delete;
    KafkaExporter& operator=(const KafkaExporter&) = delete;

    /// Start the exporter
    absl::Status Start();

    /// Stop the exporter
    absl::Status Shutdown();

    /// Send a batch of spans
    /// @param spans Spans to send
    /// @return Status
    absl::Status SendTraces(const std::vector<Span>& spans);

    /// Send a batch of metrics
    /// @param metrics Metrics to send
    /// @return Status
    absl::Status SendMetrics(const std::vector<MetricDataPoint>& metrics);

    /// Send a batch of logs
    /// @param logs Log records to send
    /// @return Status
    absl::Status SendLogs(const std::vector<LogRecord>& logs);

    /// Flush pending messages
    /// @param timeout Maximum time to wait
    /// @return Status
    absl::Status Flush(std::chrono::milliseconds timeout = std::chrono::seconds(10));

    /// Check if exporter is running
    bool IsRunning() const { return running_.load(); }

    /// Check if exporter is healthy
    bool IsHealthy() const;

    /// Get statistics
    const KafkaExporterStats& GetStats() const { return stats_; }

    /// Get configuration
    const KafkaExporterConfig& GetConfig() const { return config_; }

private:
    class Impl;
    KafkaExporterConfig config_;
    std::unique_ptr<Impl> impl_;
    KafkaExporterStats stats_;
    std::atomic<bool> running_{false};
};

}  // namespace pyflare::collector
