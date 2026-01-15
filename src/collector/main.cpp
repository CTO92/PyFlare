/// @file main.cpp
/// @brief PyFlare Collector entry point

#include <csignal>
#include <iostream>

#include <CLI/CLI.hpp>

#include "collector.h"
#include "src/common/logging.h"

namespace {

pyflare::collector::Collector* g_collector = nullptr;

void SignalHandler(int signal) {
    PYFLARE_LOG_INFO("Received signal {}, initiating shutdown", signal);
    if (g_collector) {
        g_collector->RequestShutdown();
    }
}

void PrintBanner() {
    std::cout << R"(
  ____        _____ _
 |  _ \ _   _|  ___| | __ _ _ __ ___
 | |_) | | | | |_  | |/ _` | '__/ _ \
 |  __/| |_| |  _| | | (_| | | |  __/
 |_|    \__, |_|   |_|\__,_|_|  \___|
        |___/
  OTLP Collector for AI/ML Observability
)" << std::endl;
}

}  // namespace

int main(int argc, char* argv[]) {
    CLI::App app{"PyFlare Collector - OTLP telemetry collector for AI/ML observability"};

    std::string config_path;
    std::string grpc_endpoint = "0.0.0.0:4317";
    std::string http_endpoint = "0.0.0.0:4318";
    std::string log_level = "info";
    std::vector<std::string> kafka_brokers;
    double sample_rate = 1.0;
    size_t batch_size = 512;
    bool version_flag = false;

    app.add_option("-c,--config", config_path, "Path to YAML configuration file");
    app.add_option("--grpc-endpoint", grpc_endpoint, "gRPC listen endpoint (host:port)");
    app.add_option("--http-endpoint", http_endpoint, "HTTP listen endpoint (host:port)");
    app.add_option("--log-level", log_level, "Log level (trace, debug, info, warn, error)");
    app.add_option("--kafka-brokers", kafka_brokers, "Kafka broker addresses (comma-separated)");
    app.add_option("--sample-rate", sample_rate, "Sampling rate (0.0-1.0)");
    app.add_option("--batch-size", batch_size, "Batch size for export");
    app.add_flag("-v,--version", version_flag, "Print version and exit");

    CLI11_PARSE(app, argc, argv);

    if (version_flag) {
        std::cout << "PyFlare Collector v1.0.0" << std::endl;
        return 0;
    }

    // Initialize logging
    pyflare::LogConfig log_config;
    log_config.name = "pyflare-collector";
    if (log_level == "trace") {
        log_config.level = pyflare::LogLevel::kTrace;
    } else if (log_level == "debug") {
        log_config.level = pyflare::LogLevel::kDebug;
    } else if (log_level == "warn") {
        log_config.level = pyflare::LogLevel::kWarn;
    } else if (log_level == "error") {
        log_config.level = pyflare::LogLevel::kError;
    } else {
        log_config.level = pyflare::LogLevel::kInfo;
    }
    pyflare::InitLogging(log_config);

    PrintBanner();
    PYFLARE_LOG_INFO("PyFlare Collector v1.0.0 starting...");

    // Load or create configuration
    pyflare::collector::CollectorConfig collector_config;

    if (!config_path.empty()) {
        // Load from YAML file with environment variable overrides
        auto config_or = pyflare::collector::CollectorConfig::LoadWithEnv(config_path);
        if (!config_or.ok()) {
            PYFLARE_LOG_ERROR("Failed to load config: {}", config_or.status().message());
            return 1;
        }
        collector_config = *config_or;
        PYFLARE_LOG_INFO("Loaded configuration from {}", config_path);
    } else {
        // Use defaults with CLI overrides
        collector_config = pyflare::collector::CollectorConfig::Default();
    }

    // Apply CLI overrides
    collector_config.receiver.grpc.endpoint = grpc_endpoint;
    collector_config.receiver.http.endpoint = http_endpoint;

    if (!kafka_brokers.empty()) {
        collector_config.kafka.brokers = kafka_brokers;
    }

    if (sample_rate != 1.0) {
        collector_config.sampler.probability = sample_rate;
    }

    if (batch_size != 512) {
        collector_config.batcher.max_batch_size = batch_size;
    }

    // Log configuration summary
    PYFLARE_LOG_INFO("Configuration:");
    PYFLARE_LOG_INFO("  gRPC endpoint: {}", collector_config.receiver.grpc.endpoint);
    PYFLARE_LOG_INFO("  HTTP endpoint: {}", collector_config.receiver.http.endpoint);
    PYFLARE_LOG_INFO("  Kafka brokers: {}",
                     collector_config.kafka.brokers.empty() ? "(none)" :
                     collector_config.kafka.brokers[0] + "...");
    PYFLARE_LOG_INFO("  Sample rate: {:.2f}", collector_config.sampler.probability);
    PYFLARE_LOG_INFO("  Batch size: {}", collector_config.batcher.max_batch_size);

    // Create collector
    pyflare::collector::Collector collector(std::move(collector_config));
    g_collector = &collector;

    // Set up signal handlers
#ifdef _WIN32
    std::signal(SIGINT, SignalHandler);
    std::signal(SIGTERM, SignalHandler);
#else
    struct sigaction sa;
    sa.sa_handler = SignalHandler;
    sigemptyset(&sa.sa_mask);
    sa.sa_flags = 0;
    sigaction(SIGINT, &sa, nullptr);
    sigaction(SIGTERM, &sa, nullptr);
#endif

    // Start collector
    auto status = collector.Start();
    if (!status.ok()) {
        PYFLARE_LOG_ERROR("Failed to start collector: {}", status.message());
        return 1;
    }

    PYFLARE_LOG_INFO("Collector is running. Press Ctrl+C to stop.");

    // Wait for shutdown signal
    collector.WaitForShutdown();

    // Shutdown
    status = collector.Shutdown();
    if (!status.ok()) {
        PYFLARE_LOG_ERROR("Error during shutdown: {}", status.message());
    }

    // Print final stats
    auto stats = collector.GetStats();
    PYFLARE_LOG_INFO("Final Statistics:");
    PYFLARE_LOG_INFO("  Uptime: {:.1f} seconds", stats.UptimeSeconds());
    PYFLARE_LOG_INFO("  Spans received: {}", stats.spans_received);
    PYFLARE_LOG_INFO("  Spans sampled: {}", stats.spans_sampled);
    PYFLARE_LOG_INFO("  Spans exported: {}", stats.spans_exported);
    PYFLARE_LOG_INFO("  Export errors: {}", stats.export_errors);

    PYFLARE_LOG_INFO("Collector stopped successfully");
    pyflare::ShutdownLogging();

    return 0;
}
