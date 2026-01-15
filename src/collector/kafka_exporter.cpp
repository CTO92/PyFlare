/// @file kafka_exporter.cpp
/// @brief Kafka exporter implementation using librdkafka

#include "kafka_exporter.h"

#include <chrono>
#include <sstream>

#include <absl/strings/str_cat.h>
#include <absl/strings/str_join.h>
#include <nlohmann/json.hpp>

#include "src/common/logging.h"

// Conditionally include librdkafka if available
#ifdef PYFLARE_HAS_RDKAFKA
#include <librdkafka/rdkafka.h>
#endif

namespace pyflare::collector {

namespace {

/// Serialize span to JSON format
std::string SerializeSpanToJson(const Span& span) {
    nlohmann::json j;

    j["trace_id"] = span.trace_id;
    j["span_id"] = span.span_id;
    j["parent_span_id"] = span.parent_span_id;
    j["name"] = span.name;
    j["kind"] = static_cast<int>(span.kind);
    j["start_time_ns"] = span.start_time_ns;
    j["end_time_ns"] = span.end_time_ns;
    j["duration_ns"] = span.DurationNs();
    j["status"]["code"] = static_cast<int>(span.status.code);
    j["status"]["message"] = span.status.message;

    // Attributes
    nlohmann::json attrs = nlohmann::json::object();
    for (const auto& [key, value] : span.attributes) {
        std::visit([&attrs, &key](auto&& v) {
            using T = std::decay_t<decltype(v)>;
            if constexpr (std::is_same_v<T, std::string>) {
                attrs[key] = v;
            } else if constexpr (std::is_same_v<T, bool>) {
                attrs[key] = v;
            } else if constexpr (std::is_same_v<T, int64_t>) {
                attrs[key] = v;
            } else if constexpr (std::is_same_v<T, double>) {
                attrs[key] = v;
            } else if constexpr (std::is_same_v<T, std::vector<std::string>>) {
                attrs[key] = v;
            } else {
                attrs[key] = nullptr;
            }
        }, value);
    }
    j["attributes"] = attrs;

    // ML attributes
    if (span.ml_attributes.has_value()) {
        const auto& ml = *span.ml_attributes;
        nlohmann::json ml_j;
        ml_j["model_id"] = ml.model_id;
        ml_j["model_version"] = ml.model_version;
        ml_j["model_provider"] = ml.model_provider;
        ml_j["inference_type"] = static_cast<int>(ml.inference_type);
        ml_j["input_preview"] = ml.input_preview;
        ml_j["output_preview"] = ml.output_preview;
        ml_j["cost_micros"] = ml.cost_micros;

        if (ml.token_usage.has_value()) {
            ml_j["token_usage"]["input_tokens"] = ml.token_usage->input_tokens;
            ml_j["token_usage"]["output_tokens"] = ml.token_usage->output_tokens;
            ml_j["token_usage"]["total_tokens"] = ml.token_usage->total_tokens;
        }

        j["ml_attributes"] = ml_j;
    }

    // Resource
    if (span.resource.has_value()) {
        nlohmann::json res = nlohmann::json::object();
        for (const auto& [key, value] : span.resource->attributes) {
            if (auto* str = std::get_if<std::string>(&value)) {
                res[key] = *str;
            }
        }
        j["resource"] = res;
    }

    return j.dump();
}

/// Serialize batch of spans to JSON
std::string SerializeBatchToJson(const std::vector<Span>& spans) {
    nlohmann::json batch;
    batch["spans"] = nlohmann::json::array();
    batch["timestamp_ns"] = std::chrono::duration_cast<std::chrono::nanoseconds>(
        std::chrono::system_clock::now().time_since_epoch()).count();
    batch["span_count"] = spans.size();

    for (const auto& span : spans) {
        batch["spans"].push_back(nlohmann::json::parse(SerializeSpanToJson(span)));
    }

    return batch.dump();
}

}  // namespace

/// Implementation class (PIMPL pattern)
class KafkaExporter::Impl {
public:
    explicit Impl(KafkaExporterConfig config)
        : config_(std::move(config)),
          logger_(pyflare::GetLogger()) {}

    ~Impl() {
        Shutdown();
    }

    absl::Status Start() {
        if (config_.brokers.empty()) {
            PYFLARE_LOG_WARN("No Kafka brokers configured, exporter will be disabled");
            return absl::OkStatus();
        }

#ifdef PYFLARE_HAS_RDKAFKA
        // Create Kafka configuration
        char errstr[512];
        rd_kafka_conf_t* conf = rd_kafka_conf_new();

        // Set brokers
        std::string brokers = absl::StrJoin(config_.brokers, ",");
        if (rd_kafka_conf_set(conf, "bootstrap.servers", brokers.c_str(),
                              errstr, sizeof(errstr)) != RD_KAFKA_CONF_OK) {
            rd_kafka_conf_destroy(conf);
            return absl::InvalidArgumentError(
                absl::StrCat("Failed to set bootstrap.servers: ", errstr));
        }

        // Set producer configuration
        rd_kafka_conf_set(conf, "batch.size",
                          std::to_string(config_.producer.batch_size).c_str(),
                          nullptr, 0);
        rd_kafka_conf_set(conf, "linger.ms",
                          std::to_string(config_.producer.linger_ms).c_str(),
                          nullptr, 0);
        rd_kafka_conf_set(conf, "compression.type",
                          config_.producer.compression.c_str(),
                          nullptr, 0);
        rd_kafka_conf_set(conf, "acks", config_.producer.acks.c_str(),
                          nullptr, 0);
        rd_kafka_conf_set(conf, "retries",
                          std::to_string(config_.producer.retries).c_str(),
                          nullptr, 0);

        if (config_.producer.enable_idempotence) {
            rd_kafka_conf_set(conf, "enable.idempotence", "true", nullptr, 0);
        }

        // Set timeouts
        rd_kafka_conf_set(conf, "message.timeout.ms",
                          std::to_string(config_.timeouts.message_timeout_ms).c_str(),
                          nullptr, 0);
        rd_kafka_conf_set(conf, "socket.timeout.ms",
                          std::to_string(config_.timeouts.socket_timeout_ms).c_str(),
                          nullptr, 0);

        // Set delivery report callback
        rd_kafka_conf_set_dr_msg_cb(conf, DeliveryCallback);
        rd_kafka_conf_set_opaque(conf, this);

        // Create producer
        producer_ = rd_kafka_new(RD_KAFKA_PRODUCER, conf, errstr, sizeof(errstr));
        if (!producer_) {
            return absl::InternalError(
                absl::StrCat("Failed to create Kafka producer: ", errstr));
        }

        PYFLARE_LOG_INFO("Kafka exporter started, brokers: {}", brokers);
#else
        PYFLARE_LOG_WARN("Kafka support not compiled in, exporter will simulate sends");
#endif

        return absl::OkStatus();
    }

    absl::Status Shutdown() {
#ifdef PYFLARE_HAS_RDKAFKA
        if (producer_) {
            // Wait for outstanding messages
            rd_kafka_flush(producer_, 10000);
            rd_kafka_destroy(producer_);
            producer_ = nullptr;
        }
#endif
        return absl::OkStatus();
    }

    absl::Status SendTraces(const std::vector<Span>& spans) {
        if (spans.empty()) {
            return absl::OkStatus();
        }

        auto start = std::chrono::steady_clock::now();

        std::string payload;
        if (config_.format == KafkaExporterConfig::SerializationFormat::kJson) {
            payload = SerializeBatchToJson(spans);
        } else {
            // Protobuf serialization would go here
            payload = SerializeBatchToJson(spans);  // Fallback to JSON
        }

#ifdef PYFLARE_HAS_RDKAFKA
        if (producer_) {
            // Use first span's trace_id as key for partitioning
            const std::string& key = spans[0].trace_id;

            rd_kafka_resp_err_t err = rd_kafka_producev(
                producer_,
                RD_KAFKA_V_TOPIC(config_.topics.traces.c_str()),
                RD_KAFKA_V_KEY(key.c_str(), key.size()),
                RD_KAFKA_V_VALUE(payload.c_str(), payload.size()),
                RD_KAFKA_V_MSGFLAGS(RD_KAFKA_MSG_F_COPY),
                RD_KAFKA_V_END);

            if (err != RD_KAFKA_RESP_ERR_NO_ERROR) {
                stats_.messages_failed++;
                return absl::InternalError(
                    absl::StrCat("Failed to produce message: ",
                                 rd_kafka_err2str(err)));
            }

            // Poll for delivery reports
            rd_kafka_poll(producer_, 0);
        }
#else
        // Simulate send for testing without Kafka
        PYFLARE_LOG_DEBUG("Would send {} spans ({} bytes) to {}",
                          spans.size(), payload.size(), config_.topics.traces);
#endif

        stats_.messages_sent++;
        stats_.spans_sent += spans.size();
        stats_.bytes_sent += payload.size();
        stats_.batches_sent++;

        auto elapsed = std::chrono::steady_clock::now() - start;
        double latency_ms = std::chrono::duration<double, std::milli>(elapsed).count();

        // Update average latency (exponential moving average)
        double current_avg = stats_.avg_latency_ms.load();
        stats_.avg_latency_ms = current_avg * 0.9 + latency_ms * 0.1;

        return absl::OkStatus();
    }

    absl::Status Flush(std::chrono::milliseconds timeout) {
#ifdef PYFLARE_HAS_RDKAFKA
        if (producer_) {
            rd_kafka_flush(producer_, static_cast<int>(timeout.count()));
        }
#endif
        return absl::OkStatus();
    }

    bool IsHealthy() const {
#ifdef PYFLARE_HAS_RDKAFKA
        return producer_ != nullptr;
#else
        return true;  // Simulation mode always healthy
#endif
    }

#ifdef PYFLARE_HAS_RDKAFKA
    static void DeliveryCallback(rd_kafka_t* rk,
                                  const rd_kafka_message_t* msg,
                                  void* opaque) {
        auto* impl = static_cast<Impl*>(opaque);
        if (msg->err != RD_KAFKA_RESP_ERR_NO_ERROR) {
            impl->stats_.messages_failed++;
            PYFLARE_LOG_ERROR("Message delivery failed: {}",
                              rd_kafka_err2str(msg->err));
        }
    }
#endif

    KafkaExporterConfig config_;
    KafkaExporterStats stats_;
    std::shared_ptr<spdlog::logger> logger_;

#ifdef PYFLARE_HAS_RDKAFKA
    rd_kafka_t* producer_ = nullptr;
#endif
};

// KafkaExporter public interface implementation

KafkaExporter::KafkaExporter(KafkaExporterConfig config)
    : config_(std::move(config)),
      impl_(std::make_unique<Impl>(config_)) {}

KafkaExporter::~KafkaExporter() = default;

absl::Status KafkaExporter::Start() {
    auto status = impl_->Start();
    if (status.ok()) {
        running_ = true;
    }
    return status;
}

absl::Status KafkaExporter::Shutdown() {
    running_ = false;
    return impl_->Shutdown();
}

absl::Status KafkaExporter::SendTraces(const std::vector<Span>& spans) {
    return impl_->SendTraces(spans);
}

absl::Status KafkaExporter::SendMetrics(const std::vector<MetricDataPoint>& metrics) {
    // TODO: Implement metrics serialization and sending
    stats_.metrics_sent += metrics.size();
    return absl::OkStatus();
}

absl::Status KafkaExporter::SendLogs(const std::vector<LogRecord>& logs) {
    // TODO: Implement logs serialization and sending
    stats_.logs_sent += logs.size();
    return absl::OkStatus();
}

absl::Status KafkaExporter::Flush(std::chrono::milliseconds timeout) {
    return impl_->Flush(timeout);
}

bool KafkaExporter::IsHealthy() const {
    return impl_->IsHealthy();
}

}  // namespace pyflare::collector
