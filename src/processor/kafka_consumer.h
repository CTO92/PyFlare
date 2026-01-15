#pragma once

/// @file kafka_consumer.h
/// @brief Kafka consumer framework for PyFlare stream processing

#include <atomic>
#include <chrono>
#include <functional>
#include <memory>
#include <string>
#include <thread>
#include <vector>

#include <absl/status/status.h>
#include <absl/status/statusor.h>

#include "collector/types.h"
#include "processor/message_processor.h"

namespace pyflare::processor {

/// @brief Configuration for Kafka consumer
struct KafkaConsumerConfig {
    /// Kafka broker addresses
    std::vector<std::string> bootstrap_servers = {"localhost:9092"};

    /// Consumer group ID
    std::string group_id = "pyflare-processor";

    /// Topics to consume from
    std::vector<std::string> topics = {"pyflare.traces"};

    /// Auto offset reset (earliest, latest)
    std::string auto_offset_reset = "earliest";

    /// Enable auto commit
    bool enable_auto_commit = true;

    /// Auto commit interval (ms)
    int64_t auto_commit_interval_ms = 5000;

    /// Session timeout (ms)
    int64_t session_timeout_ms = 30000;

    /// Heartbeat interval (ms)
    int64_t heartbeat_interval_ms = 10000;

    /// Maximum poll interval (ms)
    int64_t max_poll_interval_ms = 300000;

    /// Maximum records per poll
    int32_t max_poll_records = 500;

    /// Poll timeout (ms)
    int64_t poll_timeout_ms = 1000;

    /// Number of consumer threads
    size_t num_consumer_threads = 1;

    /// Security protocol (plaintext, ssl, sasl_plaintext, sasl_ssl)
    std::string security_protocol = "plaintext";

    /// SASL mechanism (PLAIN, SCRAM-SHA-256, SCRAM-SHA-512)
    std::string sasl_mechanism;

    /// SASL username
    std::string sasl_username;

    /// SASL password
    std::string sasl_password;

    /// SSL CA certificate path
    std::string ssl_ca_location;

    /// SSL certificate path
    std::string ssl_certificate_location;

    /// SSL key path
    std::string ssl_key_location;
};

/// @brief Statistics for Kafka consumer
struct KafkaConsumerStats {
    uint64_t messages_consumed = 0;
    uint64_t messages_processed = 0;
    uint64_t messages_failed = 0;
    uint64_t bytes_consumed = 0;
    uint64_t batches_processed = 0;
    int64_t current_lag = 0;
    double avg_batch_size = 0.0;
    double avg_processing_time_ms = 0.0;
    std::chrono::system_clock::time_point last_message_time;
    std::chrono::system_clock::time_point start_time;
};

/// @brief A consumed Kafka record
struct KafkaRecord {
    std::string topic;
    int32_t partition = 0;
    int64_t offset = 0;
    std::string key;
    std::string value;
    int64_t timestamp_ms = 0;
    std::map<std::string, std::string> headers;
};

/// @brief Callback for processing consumed records
using RecordCallback = std::function<void(std::vector<KafkaRecord>&&)>;

/// @brief Callback for handling errors
using ErrorCallback = std::function<void(const std::string& error)>;

/// @brief Kafka consumer for PyFlare stream processing
///
/// Consumes messages from Kafka topics, deserializes them into spans,
/// and feeds them to a processor pipeline for analysis.
///
/// Example usage:
/// @code
///   KafkaConsumerConfig config;
///   config.bootstrap_servers = {"kafka:9092"};
///   config.topics = {"pyflare.traces"};
///
///   auto consumer = std::make_unique<KafkaConsumer>(config);
///   consumer->SetPipeline(std::move(pipeline));
///   consumer->Start();
/// @endcode
class KafkaConsumer {
public:
    explicit KafkaConsumer(KafkaConsumerConfig config);
    ~KafkaConsumer();

    // Disable copy and move
    KafkaConsumer(const KafkaConsumer&) = delete;
    KafkaConsumer& operator=(const KafkaConsumer&) = delete;
    KafkaConsumer(KafkaConsumer&&) = delete;
    KafkaConsumer& operator=(KafkaConsumer&&) = delete;

    /// @brief Set the processor pipeline
    /// @param pipeline Pipeline to process consumed messages
    void SetPipeline(std::unique_ptr<ProcessorPipeline> pipeline);

    /// @brief Set callback for raw record processing (alternative to pipeline)
    /// @param callback Callback function for records
    void SetRecordCallback(RecordCallback callback);

    /// @brief Set error callback
    /// @param callback Callback function for errors
    void SetErrorCallback(ErrorCallback callback);

    /// @brief Start consuming messages
    /// @return Status indicating success or failure
    absl::Status Start();

    /// @brief Stop consuming messages
    /// @return Status indicating success or failure
    absl::Status Stop();

    /// @brief Check if consumer is running
    /// @return true if running
    bool IsRunning() const;

    /// @brief Check if consumer is healthy
    /// @return true if healthy
    bool IsHealthy() const;

    /// @brief Get consumer statistics
    /// @return Consumer statistics
    KafkaConsumerStats GetStats() const;

    /// @brief Reset statistics
    void ResetStats();

    /// @brief Pause consumption
    void Pause();

    /// @brief Resume consumption
    void Resume();

    /// @brief Commit current offsets manually
    /// @return Status indicating success or failure
    absl::Status CommitOffsets();

    /// @brief Seek to beginning of all partitions
    /// @return Status indicating success or failure
    absl::Status SeekToBeginning();

    /// @brief Seek to end of all partitions
    /// @return Status indicating success or failure
    absl::Status SeekToEnd();

private:
    /// @brief Consumer loop running in thread
    void ConsumerLoop();

    /// @brief Process a batch of records
    /// @param records Records to process
    void ProcessBatch(std::vector<KafkaRecord>&& records);

    /// @brief Deserialize record value to spans
    /// @param record Kafka record
    /// @return Deserialized spans
    absl::StatusOr<std::vector<collector::Span>> DeserializeSpans(
        const KafkaRecord& record);

    /// @brief Update statistics
    void UpdateStats(size_t batch_size, double processing_time_ms,
                     size_t failures = 0);

    KafkaConsumerConfig config_;
    std::unique_ptr<ProcessorPipeline> pipeline_;
    RecordCallback record_callback_;
    ErrorCallback error_callback_;

    std::vector<std::thread> consumer_threads_;
    std::atomic<bool> running_{false};
    std::atomic<bool> paused_{false};
    std::atomic<bool> healthy_{true};

    KafkaConsumerStats stats_;
    mutable std::mutex stats_mutex_;

    // Forward declaration for librdkafka implementation
    class Impl;
    std::unique_ptr<Impl> impl_;
};

/// @brief Factory for creating Kafka consumers
class KafkaConsumerFactory {
public:
    /// @brief Create a Kafka consumer from configuration
    /// @param config Consumer configuration
    /// @return Kafka consumer instance
    static std::unique_ptr<KafkaConsumer> Create(KafkaConsumerConfig config);

    /// @brief Create a Kafka consumer from YAML configuration file
    /// @param config_path Path to YAML configuration file
    /// @return Kafka consumer instance
    static absl::StatusOr<std::unique_ptr<KafkaConsumer>> CreateFromConfig(
        const std::string& config_path);
};

}  // namespace pyflare::processor
