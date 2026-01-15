/// @file collector.cpp
/// @brief Main collector orchestration implementation

#include "collector.h"

#include <fstream>
#include <sstream>

#include <absl/strings/str_cat.h>
#include <nlohmann/json.hpp>
#include <yaml-cpp/yaml.h>

#include "src/common/logging.h"

#ifdef _WIN32
#include <windows.h>
#include <process.h>
#define getpid _getpid
#else
#include <unistd.h>
#endif

namespace pyflare::collector {

// ============================================================================
// CollectorConfig implementation
// ============================================================================

CollectorConfig CollectorConfig::Default() {
    CollectorConfig config;

    // Default receiver settings
    config.receiver.grpc.endpoint = "0.0.0.0:4317";
    config.receiver.http.endpoint = "0.0.0.0:4318";

    // Default batcher settings
    config.batcher.max_batch_size = 512;
    config.batcher.max_batch_timeout = std::chrono::milliseconds(5000);
    config.batcher.max_queue_size = 10000;
    config.batcher.num_workers = 4;

    // Default sampler settings
    config.sampler.strategy = SamplerConfig::Strategy::kProbabilistic;
    config.sampler.probability = 1.0;

    // Default Kafka settings - empty brokers means disabled
    config.kafka.topics.traces = "pyflare.traces";
    config.kafka.topics.metrics = "pyflare.metrics";
    config.kafka.topics.logs = "pyflare.logs";

    return config;
}

absl::StatusOr<CollectorConfig> CollectorConfig::LoadFromFile(const std::string& path) {
    try {
        YAML::Node yaml = YAML::LoadFile(path);
        CollectorConfig config = Default();

        // Parse receiver section
        if (yaml["receiver"]) {
            auto recv = yaml["receiver"];
            if (recv["grpc"]) {
                if (recv["grpc"]["endpoint"]) {
                    config.receiver.grpc.endpoint = recv["grpc"]["endpoint"].as<std::string>();
                }
                if (recv["grpc"]["max_recv_msg_size_bytes"]) {
                    config.receiver.grpc.max_recv_msg_size_bytes =
                        recv["grpc"]["max_recv_msg_size_bytes"].as<size_t>();
                }
            }
            if (recv["http"]) {
                if (recv["http"]["endpoint"]) {
                    config.receiver.http.endpoint = recv["http"]["endpoint"].as<std::string>();
                }
            }
        }

        // Parse batcher section
        if (yaml["batcher"]) {
            auto batch = yaml["batcher"];
            if (batch["max_batch_size"]) {
                config.batcher.max_batch_size = batch["max_batch_size"].as<size_t>();
            }
            if (batch["max_batch_timeout_ms"]) {
                config.batcher.max_batch_timeout =
                    std::chrono::milliseconds(batch["max_batch_timeout_ms"].as<int>());
            }
            if (batch["max_queue_size"]) {
                config.batcher.max_queue_size = batch["max_queue_size"].as<size_t>();
            }
            if (batch["num_workers"]) {
                config.batcher.num_workers = batch["num_workers"].as<size_t>();
            }
        }

        // Parse sampler section
        if (yaml["sampler"]) {
            auto samp = yaml["sampler"];
            if (samp["strategy"]) {
                std::string strategy = samp["strategy"].as<std::string>();
                if (strategy == "always_on") {
                    config.sampler.strategy = SamplerConfig::Strategy::kAlwaysOn;
                } else if (strategy == "always_off") {
                    config.sampler.strategy = SamplerConfig::Strategy::kAlwaysOff;
                } else if (strategy == "probabilistic") {
                    config.sampler.strategy = SamplerConfig::Strategy::kProbabilistic;
                } else if (strategy == "rate_limiting") {
                    config.sampler.strategy = SamplerConfig::Strategy::kRateLimiting;
                } else if (strategy == "parent_based") {
                    config.sampler.strategy = SamplerConfig::Strategy::kParentBased;
                }
            }
            if (samp["probability"]) {
                config.sampler.probability = samp["probability"].as<double>();
            }
            if (samp["traces_per_second"]) {
                config.sampler.traces_per_second = samp["traces_per_second"].as<double>();
            }
            if (samp["service_rates"]) {
                for (const auto& sr : samp["service_rates"]) {
                    std::string service = sr.first.as<std::string>();
                    double rate = sr.second.as<double>();
                    config.sampler.service_rates[service] = rate;
                }
            }
        }

        // Parse Kafka section
        if (yaml["kafka"]) {
            auto kafka = yaml["kafka"];
            if (kafka["brokers"]) {
                config.kafka.brokers.clear();
                for (const auto& broker : kafka["brokers"]) {
                    config.kafka.brokers.push_back(broker.as<std::string>());
                }
            }
            if (kafka["topics"]) {
                if (kafka["topics"]["traces"]) {
                    config.kafka.topics.traces = kafka["topics"]["traces"].as<std::string>();
                }
                if (kafka["topics"]["metrics"]) {
                    config.kafka.topics.metrics = kafka["topics"]["metrics"].as<std::string>();
                }
                if (kafka["topics"]["logs"]) {
                    config.kafka.topics.logs = kafka["topics"]["logs"].as<std::string>();
                }
            }
            if (kafka["producer"]) {
                auto prod = kafka["producer"];
                if (prod["batch_size"]) {
                    config.kafka.producer.batch_size = prod["batch_size"].as<size_t>();
                }
                if (prod["linger_ms"]) {
                    config.kafka.producer.linger_ms = prod["linger_ms"].as<int>();
                }
                if (prod["compression"]) {
                    config.kafka.producer.compression = prod["compression"].as<std::string>();
                }
            }
        }

        // Parse general section
        if (yaml["general"]) {
            auto gen = yaml["general"];
            if (gen["worker_threads"]) {
                config.general.worker_threads = gen["worker_threads"].as<size_t>();
            }
            if (gen["health_endpoint"]) {
                config.general.health_endpoint = gen["health_endpoint"].as<std::string>();
            }
            if (gen["service_name"]) {
                config.general.service_name = gen["service_name"].as<std::string>();
            }
        }

        // Parse enrichment section
        if (yaml["enrichment"]) {
            auto enr = yaml["enrichment"];
            if (enr["add_host_info"]) {
                config.enrichment.add_host_info = enr["add_host_info"].as<bool>();
            }
            if (enr["add_process_info"]) {
                config.enrichment.add_process_info = enr["add_process_info"].as<bool>();
            }
            if (enr["custom_attributes"]) {
                for (const auto& attr : enr["custom_attributes"]) {
                    config.enrichment.custom_attributes[attr.first.as<std::string>()] =
                        attr.second.as<std::string>();
                }
            }
        }

        return config;

    } catch (const YAML::Exception& e) {
        return absl::InvalidArgumentError(
            absl::StrCat("Failed to parse config file: ", e.what()));
    } catch (const std::exception& e) {
        return absl::InternalError(
            absl::StrCat("Failed to load config: ", e.what()));
    }
}

absl::StatusOr<CollectorConfig> CollectorConfig::LoadWithEnv(
    const std::string& path,
    const std::string& env_prefix) {

    auto config_or = LoadFromFile(path);
    if (!config_or.ok()) {
        return config_or;
    }

    CollectorConfig config = *config_or;

    // Override with environment variables
    auto get_env = [&env_prefix](const std::string& name) -> std::string {
        std::string env_name = env_prefix + name;
        const char* value = std::getenv(env_name.c_str());
        return value ? std::string(value) : "";
    };

    // GRPC endpoint
    std::string grpc_endpoint = get_env("GRPC_ENDPOINT");
    if (!grpc_endpoint.empty()) {
        config.receiver.grpc.endpoint = grpc_endpoint;
    }

    // HTTP endpoint
    std::string http_endpoint = get_env("HTTP_ENDPOINT");
    if (!http_endpoint.empty()) {
        config.receiver.http.endpoint = http_endpoint;
    }

    // Kafka brokers
    std::string kafka_brokers = get_env("KAFKA_BROKERS");
    if (!kafka_brokers.empty()) {
        config.kafka.brokers.clear();
        std::stringstream ss(kafka_brokers);
        std::string broker;
        while (std::getline(ss, broker, ',')) {
            config.kafka.brokers.push_back(broker);
        }
    }

    // Sample rate
    std::string sample_rate = get_env("SAMPLE_RATE");
    if (!sample_rate.empty()) {
        config.sampler.probability = std::stod(sample_rate);
    }

    return config;
}

// ============================================================================
// Collector implementation
// ============================================================================

Collector::Collector(CollectorConfig config)
    : config_(std::move(config)),
      logger_(pyflare::GetLogger()) {

    stats_.start_time = std::chrono::steady_clock::now();
}

Collector::~Collector() {
    if (running_.load()) {
        Shutdown().IgnoreError();
    }
}

absl::Status Collector::Start() {
    if (running_.exchange(true)) {
        return absl::AlreadyExistsError("Collector already running");
    }

    PYFLARE_LOG_INFO("Starting PyFlare Collector v1.0");

    // Initialize host information
    InitHostInfo();

    // Create sampler
    sampler_ = CreateSampler(config_.sampler);
    PYFLARE_LOG_INFO("Sampler initialized with strategy {}",
                     static_cast<int>(config_.sampler.strategy));

    // Create batcher
    batcher_ = std::make_unique<Batcher>(config_.batcher);
    batcher_->OnBatch([this](std::vector<Span>&& batch) {
        OnBatchReady(std::move(batch));
    });

    auto batcher_status = batcher_->Start();
    if (!batcher_status.ok()) {
        running_ = false;
        return batcher_status;
    }
    PYFLARE_LOG_INFO("Batcher started with {} workers, batch_size={}",
                     config_.batcher.num_workers, config_.batcher.max_batch_size);

    // Create Kafka exporter if brokers configured
    if (!config_.kafka.brokers.empty()) {
        kafka_exporter_ = std::make_unique<KafkaExporter>(config_.kafka);
        auto kafka_status = kafka_exporter_->Start();
        if (!kafka_status.ok()) {
            PYFLARE_LOG_WARN("Failed to start Kafka exporter: {}",
                             kafka_status.message());
            // Continue without Kafka - it's optional
        } else {
            PYFLARE_LOG_INFO("Kafka exporter started, brokers: {}",
                             config_.kafka.brokers[0]);
        }
    } else {
        PYFLARE_LOG_INFO("Kafka exporter disabled (no brokers configured)");
    }

    // Create and start receiver
    receiver_ = std::make_unique<OtlpReceiver>(config_.receiver);

    receiver_->OnSpans([this](std::vector<Span>&& spans) {
        OnSpansReceived(std::move(spans));
    });

    receiver_->OnMetrics([this](std::vector<MetricDataPoint>&& metrics) {
        OnMetricsReceived(std::move(metrics));
    });

    receiver_->OnLogs([this](std::vector<LogRecord>&& logs) {
        OnLogsReceived(std::move(logs));
    });

    auto receiver_status = receiver_->Start();
    if (!receiver_status.ok()) {
        running_ = false;
        return receiver_status;
    }

    PYFLARE_LOG_INFO("OTLP receiver listening on gRPC={}, HTTP={}",
                     config_.receiver.grpc.endpoint,
                     config_.receiver.http.endpoint);

    ready_ = true;

    PYFLARE_LOG_INFO("PyFlare Collector started successfully");
    return absl::OkStatus();
}

absl::Status Collector::Shutdown() {
    if (!running_.exchange(false)) {
        return absl::FailedPreconditionError("Collector not running");
    }

    ready_ = false;
    PYFLARE_LOG_INFO("Shutting down PyFlare Collector...");

    // Stop receiver first (stop accepting new data)
    if (receiver_) {
        auto status = receiver_->Shutdown();
        if (!status.ok()) {
            PYFLARE_LOG_WARN("Error shutting down receiver: {}", status.message());
        }
    }

    // Stop batcher (will flush remaining data)
    if (batcher_) {
        auto status = batcher_->Shutdown();
        if (!status.ok()) {
            PYFLARE_LOG_WARN("Error shutting down batcher: {}", status.message());
        }
    }

    // Stop Kafka exporter last (allow final exports)
    if (kafka_exporter_) {
        kafka_exporter_->Flush(std::chrono::seconds(10)).IgnoreError();
        auto status = kafka_exporter_->Shutdown();
        if (!status.ok()) {
            PYFLARE_LOG_WARN("Error shutting down Kafka exporter: {}", status.message());
        }
    }

    // Notify shutdown waiters
    {
        std::lock_guard<std::mutex> lock(shutdown_mutex_);
        shutdown_requested_ = true;
    }
    shutdown_cv_.notify_all();

    PYFLARE_LOG_INFO("PyFlare Collector shutdown complete. Stats: received={}, "
                     "sampled={}, exported={}",
                     stats_.spans_received,
                     stats_.spans_sampled,
                     stats_.spans_exported);

    return absl::OkStatus();
}

void Collector::WaitForShutdown() {
    std::unique_lock<std::mutex> lock(shutdown_mutex_);
    shutdown_cv_.wait(lock, [this] {
        return shutdown_requested_.load();
    });
}

void Collector::RequestShutdown() {
    shutdown_requested_ = true;
    shutdown_cv_.notify_all();
}

bool Collector::IsHealthy() const {
    if (!running_.load()) {
        return false;
    }

    // Check all components
    if (receiver_ && !receiver_->IsRunning()) {
        return false;
    }

    if (kafka_exporter_ && !kafka_exporter_->IsHealthy()) {
        return false;
    }

    return true;
}

bool Collector::IsReady() const {
    return ready_.load();
}

CollectorStats Collector::GetStats() const {
    std::lock_guard<std::mutex> lock(stats_mutex_);

    CollectorStats result = stats_;
    result.is_healthy = IsHealthy();
    result.is_ready = IsReady();

    // Aggregate stats from components
    if (receiver_) {
        const auto& rs = receiver_->GetStats();
        result.spans_received = rs.spans_received.load();
        result.metrics_received = rs.metrics_received.load();
        result.logs_received = rs.logs_received.load();
        result.bytes_received = rs.bytes_received.load();
    }

    if (batcher_) {
        const auto& bs = batcher_->GetStats();
        result.spans_processed = bs.spans_batched.load();
    }

    if (kafka_exporter_) {
        const auto& ks = kafka_exporter_->GetStats();
        result.spans_exported = ks.spans_sent.load();
        result.batches_exported = ks.batches_sent.load();
        result.export_errors = ks.messages_failed.load();
        result.bytes_exported = ks.bytes_sent.load();
        result.avg_export_latency_ms = ks.avg_latency_ms.load();
    }

    return result;
}

void Collector::OnSpansReceived(std::vector<Span>&& spans) {
    if (spans.empty() || !running_.load()) {
        return;
    }

    auto start = std::chrono::steady_clock::now();

    // Enrich spans
    EnrichSpans(spans);

    // Sample spans
    auto sampled = SampleSpans(std::move(spans));

    // Send to batcher
    if (!sampled.empty()) {
        auto status = batcher_->Add(std::move(sampled));
        if (!status.ok()) {
            std::lock_guard<std::mutex> lock(stats_mutex_);
            stats_.spans_dropped_queue_full += sampled.size();
        }
    }

    // Update latency stats
    auto elapsed = std::chrono::steady_clock::now() - start;
    double latency_ms = std::chrono::duration<double, std::milli>(elapsed).count();

    std::lock_guard<std::mutex> lock(stats_mutex_);
    stats_.avg_processing_latency_ms =
        stats_.avg_processing_latency_ms * 0.9 + latency_ms * 0.1;
}

void Collector::OnMetricsReceived(std::vector<MetricDataPoint>&& metrics) {
    if (metrics.empty() || !running_.load()) {
        return;
    }

    {
        std::lock_guard<std::mutex> lock(stats_mutex_);
        stats_.metrics_received += metrics.size();
    }

    // Forward to Kafka exporter
    if (kafka_exporter_) {
        kafka_exporter_->SendMetrics(metrics).IgnoreError();
    }
}

void Collector::OnLogsReceived(std::vector<LogRecord>&& logs) {
    if (logs.empty() || !running_.load()) {
        return;
    }

    {
        std::lock_guard<std::mutex> lock(stats_mutex_);
        stats_.logs_received += logs.size();
    }

    // Forward to Kafka exporter
    if (kafka_exporter_) {
        kafka_exporter_->SendLogs(logs).IgnoreError();
    }
}

void Collector::OnBatchReady(std::vector<Span>&& batch) {
    if (batch.empty() || !running_.load()) {
        return;
    }

    // Export to Kafka
    if (kafka_exporter_) {
        auto status = kafka_exporter_->SendTraces(batch);
        if (status.ok()) {
            std::lock_guard<std::mutex> lock(stats_mutex_);
            stats_.spans_exported += batch.size();
            stats_.batches_exported++;
            stats_.last_export_time = std::chrono::steady_clock::now();
        } else {
            std::lock_guard<std::mutex> lock(stats_mutex_);
            stats_.export_errors++;
            PYFLARE_LOG_ERROR("Failed to export batch: {}", status.message());
        }
    } else {
        // No exporter configured - just log
        PYFLARE_LOG_DEBUG("Batch ready with {} spans (no exporter configured)",
                          batch.size());
    }
}

void Collector::EnrichSpans(std::vector<Span>& spans) {
    for (auto& span : spans) {
        // Add host information
        if (config_.enrichment.add_host_info && !hostname_.empty()) {
            span.attributes["host.name"] = hostname_;
        }

        // Add process information
        if (config_.enrichment.add_process_info && !process_id_.empty()) {
            span.attributes["process.pid"] = process_id_;
        }

        // Add collector service name
        if (!config_.general.service_name.empty()) {
            span.attributes["collector.service"] = config_.general.service_name;
        }

        // Add custom attributes
        for (const auto& [key, value] : config_.enrichment.custom_attributes) {
            span.attributes[key] = value;
        }

        // Normalize timestamps if needed
        if (config_.enrichment.normalize_timestamps) {
            // Timestamps are already in nanoseconds since epoch (UTC)
            // No conversion needed for standard OTLP format
        }
    }
}

std::vector<Span> Collector::SampleSpans(std::vector<Span>&& spans) {
    if (!sampler_) {
        // No sampler - pass all through
        return std::move(spans);
    }

    std::vector<Span> sampled;
    sampled.reserve(spans.size());

    size_t dropped = 0;

    for (auto& span : spans) {
        auto decision = sampler_->ShouldSample(span);
        if (decision == SamplingDecision::kSample) {
            sampled.push_back(std::move(span));
        } else {
            dropped++;
        }
    }

    // Update stats
    {
        std::lock_guard<std::mutex> lock(stats_mutex_);
        stats_.spans_sampled += sampled.size();
        stats_.spans_dropped_sampling += dropped;
    }

    return sampled;
}

void Collector::UpdateStats() {
    // Called periodically to update statistics
    std::lock_guard<std::mutex> lock(stats_mutex_);

    if (receiver_) {
        const auto& rs = receiver_->GetStats();
        stats_.spans_received = rs.spans_received.load();
    }

    if (kafka_exporter_) {
        const auto& ks = kafka_exporter_->GetStats();
        stats_.spans_exported = ks.spans_sent.load();
    }
}

void Collector::InitHostInfo() {
    // Get hostname
    char hostname[256];
#ifdef _WIN32
    DWORD size = sizeof(hostname);
    if (GetComputerNameA(hostname, &size)) {
        hostname_ = hostname;
    }
#else
    if (gethostname(hostname, sizeof(hostname)) == 0) {
        hostname_ = hostname;
    }
#endif

    // Get process ID
    process_id_ = std::to_string(getpid());

    PYFLARE_LOG_DEBUG("Host info: hostname={}, pid={}", hostname_, process_id_);
}

}  // namespace pyflare::collector
