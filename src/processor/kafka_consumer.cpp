/// @file kafka_consumer.cpp
/// @brief Kafka consumer implementation using librdkafka

#include "processor/kafka_consumer.h"

#include <nlohmann/json.hpp>
#include <spdlog/spdlog.h>

#ifdef PYFLARE_HAS_KAFKA
#include <librdkafka/rdkafkacpp.h>
#endif

namespace pyflare::processor {

using json = nlohmann::json;

// =============================================================================
// KafkaConsumer Implementation
// =============================================================================

#ifdef PYFLARE_HAS_KAFKA

/// @brief Internal implementation using librdkafka
class KafkaConsumer::Impl {
public:
    explicit Impl(const KafkaConsumerConfig& config) : config_(config) {}

    ~Impl() {
        if (consumer_) {
            consumer_->close();
        }
    }

    absl::Status Initialize() {
        std::string errstr;

        // Create configuration
        conf_.reset(RdKafka::Conf::create(RdKafka::Conf::CONF_GLOBAL));
        if (!conf_) {
            return absl::InternalError("Failed to create Kafka configuration");
        }

        // Set bootstrap servers
        std::string brokers;
        for (size_t i = 0; i < config_.bootstrap_servers.size(); ++i) {
            if (i > 0) brokers += ",";
            brokers += config_.bootstrap_servers[i];
        }
        if (conf_->set("bootstrap.servers", brokers, errstr) !=
            RdKafka::Conf::CONF_OK) {
            return absl::InvalidArgumentError("Failed to set bootstrap.servers: " + errstr);
        }

        // Set consumer group
        if (conf_->set("group.id", config_.group_id, errstr) !=
            RdKafka::Conf::CONF_OK) {
            return absl::InvalidArgumentError("Failed to set group.id: " + errstr);
        }

        // Set auto offset reset
        if (conf_->set("auto.offset.reset", config_.auto_offset_reset, errstr) !=
            RdKafka::Conf::CONF_OK) {
            return absl::InvalidArgumentError("Failed to set auto.offset.reset: " + errstr);
        }

        // Set auto commit
        if (conf_->set("enable.auto.commit",
                       config_.enable_auto_commit ? "true" : "false",
                       errstr) != RdKafka::Conf::CONF_OK) {
            return absl::InvalidArgumentError("Failed to set enable.auto.commit: " + errstr);
        }

        if (config_.enable_auto_commit) {
            if (conf_->set("auto.commit.interval.ms",
                           std::to_string(config_.auto_commit_interval_ms),
                           errstr) != RdKafka::Conf::CONF_OK) {
                return absl::InvalidArgumentError(
                    "Failed to set auto.commit.interval.ms: " + errstr);
            }
        }

        // Set session timeout
        if (conf_->set("session.timeout.ms",
                       std::to_string(config_.session_timeout_ms),
                       errstr) != RdKafka::Conf::CONF_OK) {
            return absl::InvalidArgumentError("Failed to set session.timeout.ms: " + errstr);
        }

        // Set heartbeat interval
        if (conf_->set("heartbeat.interval.ms",
                       std::to_string(config_.heartbeat_interval_ms),
                       errstr) != RdKafka::Conf::CONF_OK) {
            return absl::InvalidArgumentError(
                "Failed to set heartbeat.interval.ms: " + errstr);
        }

        // Set max poll interval
        if (conf_->set("max.poll.interval.ms",
                       std::to_string(config_.max_poll_interval_ms),
                       errstr) != RdKafka::Conf::CONF_OK) {
            return absl::InvalidArgumentError(
                "Failed to set max.poll.interval.ms: " + errstr);
        }

        // Configure security if specified
        if (!config_.security_protocol.empty() &&
            config_.security_protocol != "plaintext") {
            if (conf_->set("security.protocol", config_.security_protocol, errstr) !=
                RdKafka::Conf::CONF_OK) {
                return absl::InvalidArgumentError(
                    "Failed to set security.protocol: " + errstr);
            }

            // SASL configuration
            if (!config_.sasl_mechanism.empty()) {
                if (conf_->set("sasl.mechanism", config_.sasl_mechanism, errstr) !=
                    RdKafka::Conf::CONF_OK) {
                    return absl::InvalidArgumentError(
                        "Failed to set sasl.mechanism: " + errstr);
                }

                if (!config_.sasl_username.empty()) {
                    if (conf_->set("sasl.username", config_.sasl_username, errstr) !=
                        RdKafka::Conf::CONF_OK) {
                        return absl::InvalidArgumentError(
                            "Failed to set sasl.username: " + errstr);
                    }
                }

                if (!config_.sasl_password.empty()) {
                    if (conf_->set("sasl.password", config_.sasl_password, errstr) !=
                        RdKafka::Conf::CONF_OK) {
                        return absl::InvalidArgumentError(
                            "Failed to set sasl.password: " + errstr);
                    }
                }
            }

            // SSL configuration
            if (!config_.ssl_ca_location.empty()) {
                if (conf_->set("ssl.ca.location", config_.ssl_ca_location, errstr) !=
                    RdKafka::Conf::CONF_OK) {
                    return absl::InvalidArgumentError(
                        "Failed to set ssl.ca.location: " + errstr);
                }
            }

            if (!config_.ssl_certificate_location.empty()) {
                if (conf_->set("ssl.certificate.location",
                               config_.ssl_certificate_location, errstr) !=
                    RdKafka::Conf::CONF_OK) {
                    return absl::InvalidArgumentError(
                        "Failed to set ssl.certificate.location: " + errstr);
                }
            }

            if (!config_.ssl_key_location.empty()) {
                if (conf_->set("ssl.key.location", config_.ssl_key_location, errstr) !=
                    RdKafka::Conf::CONF_OK) {
                    return absl::InvalidArgumentError(
                        "Failed to set ssl.key.location: " + errstr);
                }
            }
        }

        // Create consumer
        consumer_.reset(RdKafka::KafkaConsumer::create(conf_.get(), errstr));
        if (!consumer_) {
            return absl::InternalError("Failed to create Kafka consumer: " + errstr);
        }

        // Subscribe to topics
        RdKafka::ErrorCode err = consumer_->subscribe(config_.topics);
        if (err != RdKafka::ERR_NO_ERROR) {
            return absl::InternalError("Failed to subscribe to topics: " +
                                       RdKafka::err2str(err));
        }

        spdlog::info("Kafka consumer initialized, subscribed to {} topics",
                     config_.topics.size());
        return absl::OkStatus();
    }

    absl::StatusOr<std::vector<KafkaRecord>> Poll(int timeout_ms) {
        std::vector<KafkaRecord> records;
        records.reserve(config_.max_poll_records);

        int count = 0;
        while (count < config_.max_poll_records) {
            std::unique_ptr<RdKafka::Message> msg(
                consumer_->consume(count == 0 ? timeout_ms : 0));

            if (!msg) {
                break;
            }

            switch (msg->err()) {
                case RdKafka::ERR_NO_ERROR: {
                    KafkaRecord record;
                    record.topic = msg->topic_name();
                    record.partition = msg->partition();
                    record.offset = msg->offset();
                    record.timestamp_ms = msg->timestamp().timestamp;

                    if (msg->key()) {
                        record.key = std::string(
                            static_cast<const char*>(msg->key_pointer()),
                            msg->key_len());
                    }

                    if (msg->payload()) {
                        record.value = std::string(
                            static_cast<const char*>(msg->payload()),
                            msg->len());
                    }

                    // Extract headers
                    if (msg->headers()) {
                        std::vector<RdKafka::Headers::Header> hdrs =
                            msg->headers()->get_all();
                        for (const auto& hdr : hdrs) {
                            if (hdr.value()) {
                                record.headers[hdr.key()] = std::string(
                                    static_cast<const char*>(hdr.value()),
                                    hdr.value_size());
                            }
                        }
                    }

                    records.push_back(std::move(record));
                    count++;
                    break;
                }
                case RdKafka::ERR__TIMED_OUT:
                    // No more messages available
                    return records;
                case RdKafka::ERR__PARTITION_EOF:
                    // End of partition, continue polling
                    break;
                default:
                    return absl::InternalError("Kafka consume error: " +
                                               RdKafka::err2str(msg->err()));
            }
        }

        return records;
    }

    absl::Status CommitOffsets() {
        RdKafka::ErrorCode err = consumer_->commitSync();
        if (err != RdKafka::ERR_NO_ERROR) {
            return absl::InternalError("Failed to commit offsets: " +
                                       RdKafka::err2str(err));
        }
        return absl::OkStatus();
    }

    void Close() {
        if (consumer_) {
            consumer_->close();
        }
    }

private:
    KafkaConsumerConfig config_;
    std::unique_ptr<RdKafka::Conf> conf_;
    std::unique_ptr<RdKafka::KafkaConsumer> consumer_;
};

#else  // !PYFLARE_HAS_KAFKA

/// @brief Stub implementation when Kafka is not available
class KafkaConsumer::Impl {
public:
    explicit Impl(const KafkaConsumerConfig&) {}

    absl::Status Initialize() {
        spdlog::warn("Kafka support not compiled in, consumer will not function");
        return absl::UnimplementedError("Kafka support not compiled");
    }

    absl::StatusOr<std::vector<KafkaRecord>> Poll(int) {
        return absl::UnimplementedError("Kafka support not compiled");
    }

    absl::Status CommitOffsets() {
        return absl::UnimplementedError("Kafka support not compiled");
    }

    void Close() {}
};

#endif  // PYFLARE_HAS_KAFKA

// =============================================================================
// KafkaConsumer Public Interface
// =============================================================================

KafkaConsumer::KafkaConsumer(KafkaConsumerConfig config)
    : config_(std::move(config)), impl_(std::make_unique<Impl>(config_)) {
    stats_.start_time = std::chrono::system_clock::now();
}

KafkaConsumer::~KafkaConsumer() {
    if (running_.load()) {
        auto status = Stop();
        if (!status.ok()) {
            spdlog::error("Failed to stop Kafka consumer in destructor: {}",
                          status.message());
        }
    }
}

void KafkaConsumer::SetPipeline(std::unique_ptr<ProcessorPipeline> pipeline) {
    if (running_.load()) {
        spdlog::warn("Cannot set pipeline while consumer is running");
        return;
    }
    pipeline_ = std::move(pipeline);
}

void KafkaConsumer::SetRecordCallback(RecordCallback callback) {
    if (running_.load()) {
        spdlog::warn("Cannot set callback while consumer is running");
        return;
    }
    record_callback_ = std::move(callback);
}

void KafkaConsumer::SetErrorCallback(ErrorCallback callback) {
    error_callback_ = std::move(callback);
}

absl::Status KafkaConsumer::Start() {
    if (running_.load()) {
        return absl::AlreadyExistsError("Consumer already running");
    }

    // Initialize librdkafka consumer
    auto status = impl_->Initialize();
    if (!status.ok()) {
        return status;
    }

    // Start pipeline if configured
    if (pipeline_) {
        status = pipeline_->Start();
        if (!status.ok()) {
            return status;
        }
    }

    running_.store(true);
    healthy_.store(true);

    // Start consumer threads
    for (size_t i = 0; i < config_.num_consumer_threads; ++i) {
        consumer_threads_.emplace_back(&KafkaConsumer::ConsumerLoop, this);
    }

    spdlog::info("Kafka consumer started with {} threads",
                 config_.num_consumer_threads);
    return absl::OkStatus();
}

absl::Status KafkaConsumer::Stop() {
    if (!running_.load()) {
        return absl::OkStatus();
    }

    spdlog::info("Stopping Kafka consumer...");
    running_.store(false);

    // Wait for consumer threads to finish
    for (auto& thread : consumer_threads_) {
        if (thread.joinable()) {
            thread.join();
        }
    }
    consumer_threads_.clear();

    // Stop pipeline
    if (pipeline_) {
        auto status = pipeline_->Stop();
        if (!status.ok()) {
            spdlog::error("Failed to stop pipeline: {}", status.message());
        }
    }

    // Close Kafka consumer
    impl_->Close();

    spdlog::info("Kafka consumer stopped");
    return absl::OkStatus();
}

bool KafkaConsumer::IsRunning() const {
    return running_.load();
}

bool KafkaConsumer::IsHealthy() const {
    return healthy_.load();
}

KafkaConsumerStats KafkaConsumer::GetStats() const {
    std::lock_guard<std::mutex> lock(stats_mutex_);
    return stats_;
}

void KafkaConsumer::ResetStats() {
    std::lock_guard<std::mutex> lock(stats_mutex_);
    stats_ = KafkaConsumerStats{};
    stats_.start_time = std::chrono::system_clock::now();
}

void KafkaConsumer::Pause() {
    paused_.store(true);
    spdlog::info("Kafka consumer paused");
}

void KafkaConsumer::Resume() {
    paused_.store(false);
    spdlog::info("Kafka consumer resumed");
}

absl::Status KafkaConsumer::CommitOffsets() {
    return impl_->CommitOffsets();
}

absl::Status KafkaConsumer::SeekToBeginning() {
    // Implementation would seek to beginning of all assigned partitions
    return absl::UnimplementedError("SeekToBeginning not yet implemented");
}

absl::Status KafkaConsumer::SeekToEnd() {
    // Implementation would seek to end of all assigned partitions
    return absl::UnimplementedError("SeekToEnd not yet implemented");
}

void KafkaConsumer::ConsumerLoop() {
    spdlog::debug("Consumer thread started");

    while (running_.load()) {
        // Check if paused
        if (paused_.load()) {
            std::this_thread::sleep_for(std::chrono::milliseconds(100));
            continue;
        }

        // Poll for messages
        auto records_result = impl_->Poll(static_cast<int>(config_.poll_timeout_ms));
        if (!records_result.ok()) {
            spdlog::error("Poll error: {}", records_result.status().message());
            healthy_.store(false);

            if (error_callback_) {
                error_callback_(std::string(records_result.status().message()));
            }

            std::this_thread::sleep_for(std::chrono::seconds(1));
            continue;
        }

        auto records = std::move(*records_result);
        if (records.empty()) {
            continue;
        }

        // Process the batch
        ProcessBatch(std::move(records));
    }

    spdlog::debug("Consumer thread stopped");
}

void KafkaConsumer::ProcessBatch(std::vector<KafkaRecord>&& records) {
    auto start_time = std::chrono::steady_clock::now();
    size_t batch_size = records.size();
    size_t failures = 0;

    // Update bytes consumed
    size_t bytes = 0;
    for (const auto& record : records) {
        bytes += record.value.size();
    }

    // If record callback is set, use it directly
    if (record_callback_) {
        try {
            record_callback_(std::move(records));
        } catch (const std::exception& e) {
            spdlog::error("Record callback error: {}", e.what());
            failures = batch_size;
        }
    } else if (pipeline_) {
        // Deserialize records to spans and process through pipeline
        std::vector<collector::Span> spans;
        spans.reserve(batch_size);

        for (const auto& record : records) {
            auto span_result = DeserializeSpans(record);
            if (span_result.ok()) {
                for (auto& span : *span_result) {
                    spans.push_back(std::move(span));
                }
            } else {
                spdlog::debug("Failed to deserialize record: {}",
                              span_result.status().message());
                failures++;
            }
        }

        if (!spans.empty()) {
            auto result = pipeline_->Process(std::move(spans));
            if (!result.ok()) {
                spdlog::error("Pipeline processing error: {}",
                              result.status().message());
                failures += batch_size;
            }
        }
    }

    auto end_time = std::chrono::steady_clock::now();
    double processing_time_ms =
        std::chrono::duration<double, std::milli>(end_time - start_time).count();

    // Update statistics
    {
        std::lock_guard<std::mutex> lock(stats_mutex_);
        stats_.messages_consumed += batch_size;
        stats_.messages_processed += (batch_size - failures);
        stats_.messages_failed += failures;
        stats_.bytes_consumed += bytes;
        stats_.batches_processed++;
        stats_.last_message_time = std::chrono::system_clock::now();

        // Update average batch size (exponential moving average)
        const double alpha = 0.1;
        stats_.avg_batch_size =
            alpha * static_cast<double>(batch_size) +
            (1.0 - alpha) * stats_.avg_batch_size;

        // Update average processing time
        stats_.avg_processing_time_ms =
            alpha * processing_time_ms +
            (1.0 - alpha) * stats_.avg_processing_time_ms;
    }

    spdlog::debug("Processed batch of {} records in {:.2f}ms ({} failures)",
                  batch_size, processing_time_ms, failures);
}

absl::StatusOr<std::vector<collector::Span>> KafkaConsumer::DeserializeSpans(
    const KafkaRecord& record) {
    std::vector<collector::Span> spans;

    // Try to parse as JSON (common format from collector)
    try {
        json j = json::parse(record.value);

        // Check if it's a batch of spans
        if (j.contains("spans") && j["spans"].is_array()) {
            for (const auto& span_json : j["spans"]) {
                collector::Span span;
                span.trace_id = span_json.value("trace_id", "");
                span.span_id = span_json.value("span_id", "");
                span.parent_span_id = span_json.value("parent_span_id", "");
                span.name = span_json.value("name", "");
                span.start_time_ns = span_json.value("start_time_ns", 0ULL);
                span.end_time_ns = span_json.value("end_time_ns", 0ULL);

                // Parse span kind
                std::string kind_str = span_json.value("kind", "internal");
                if (kind_str == "server") {
                    span.kind = collector::SpanKind::kServer;
                } else if (kind_str == "client") {
                    span.kind = collector::SpanKind::kClient;
                } else if (kind_str == "producer") {
                    span.kind = collector::SpanKind::kProducer;
                } else if (kind_str == "consumer") {
                    span.kind = collector::SpanKind::kConsumer;
                } else {
                    span.kind = collector::SpanKind::kInternal;
                }

                // Parse status
                if (span_json.contains("status")) {
                    const auto& status_json = span_json["status"];
                    std::string code_str = status_json.value("code", "unset");
                    if (code_str == "ok") {
                        span.status.code = collector::StatusCode::kOk;
                    } else if (code_str == "error") {
                        span.status.code = collector::StatusCode::kError;
                    }
                    span.status.message = status_json.value("message", "");
                }

                // Parse ML attributes
                if (span_json.contains("ml_attributes")) {
                    const auto& ml_json = span_json["ml_attributes"];
                    collector::MLAttributes ml;
                    ml.model_id = ml_json.value("model_id", "");
                    ml.model_version = ml_json.value("model_version", "");
                    ml.model_provider = ml_json.value("model_provider", "");
                    ml.input_preview = ml_json.value("input_preview", "");
                    ml.output_preview = ml_json.value("output_preview", "");
                    ml.cost_micros = ml_json.value("cost_micros", 0ULL);

                    // Parse token usage
                    if (ml_json.contains("token_usage")) {
                        const auto& token_json = ml_json["token_usage"];
                        collector::TokenUsage usage;
                        usage.input_tokens = token_json.value("input_tokens", 0U);
                        usage.output_tokens = token_json.value("output_tokens", 0U);
                        usage.total_tokens = token_json.value("total_tokens", 0U);
                        ml.token_usage = usage;
                    }

                    // Parse inference type
                    std::string type_str = ml_json.value("inference_type", "");
                    if (type_str == "llm") {
                        ml.inference_type = collector::InferenceType::kLlm;
                    } else if (type_str == "embedding") {
                        ml.inference_type = collector::InferenceType::kEmbedding;
                    } else if (type_str == "classification") {
                        ml.inference_type = collector::InferenceType::kClassification;
                    }

                    span.ml_attributes = ml;
                }

                // Parse attributes
                if (span_json.contains("attributes")) {
                    for (auto& [key, val] : span_json["attributes"].items()) {
                        if (val.is_string()) {
                            span.attributes[key] = val.get<std::string>();
                        } else if (val.is_number_integer()) {
                            span.attributes[key] = val.get<int64_t>();
                        } else if (val.is_number_float()) {
                            span.attributes[key] = val.get<double>();
                        } else if (val.is_boolean()) {
                            span.attributes[key] = val.get<bool>();
                        }
                    }
                }

                spans.push_back(std::move(span));
            }
        } else {
            // Single span
            collector::Span span;
            span.trace_id = j.value("trace_id", "");
            span.span_id = j.value("span_id", "");
            span.name = j.value("name", "");
            span.start_time_ns = j.value("start_time_ns", 0ULL);
            span.end_time_ns = j.value("end_time_ns", 0ULL);
            spans.push_back(std::move(span));
        }
    } catch (const json::exception& e) {
        return absl::InvalidArgumentError(
            std::string("Failed to parse JSON: ") + e.what());
    }

    return spans;
}

// =============================================================================
// KafkaConsumerFactory
// =============================================================================

std::unique_ptr<KafkaConsumer> KafkaConsumerFactory::Create(
    KafkaConsumerConfig config) {
    return std::make_unique<KafkaConsumer>(std::move(config));
}

absl::StatusOr<std::unique_ptr<KafkaConsumer>>
KafkaConsumerFactory::CreateFromConfig(const std::string& config_path) {
    // TODO: Implement YAML configuration loading
    return absl::UnimplementedError("CreateFromConfig not yet implemented");
}

}  // namespace pyflare::processor
