#pragma once

/// @file otlp_receiver.h
/// @brief OTLP receiver for accepting trace, metrics, and log data
///
/// SECURITY: This receiver implements multiple security controls:
/// - TLS encryption for gRPC connections
/// - Rate limiting to prevent DoS attacks
/// - CORS origin validation
/// - JSON parsing depth limits

#include <atomic>
#include <chrono>
#include <memory>
#include <string>
#include <vector>

#include <absl/status/status.h>

#include "types.h"

namespace pyflare::collector {

/// OTLP Receiver configuration
struct OtlpReceiverConfig {
    /// gRPC server settings
    struct GrpcConfig {
        std::string endpoint = "0.0.0.0:4317";
        size_t max_recv_msg_size_bytes = 16 * 1024 * 1024;  // 16 MB
        size_t max_concurrent_streams = 100;
        bool enable_reflection = true;

        struct Keepalive {
            std::chrono::seconds time{30};
            std::chrono::seconds timeout{10};
            bool permit_without_stream = true;
        };
        Keepalive keepalive;

        /// TLS configuration for secure connections
        struct TlsConfig {
            bool enabled = false;
            std::string cert_path;      // Path to server certificate
            std::string key_path;       // Path to private key
            std::string ca_cert_path;   // Path to CA cert for client verification
            bool require_client_cert = false;  // Enable mutual TLS
        };
        TlsConfig tls;
    };
    GrpcConfig grpc;

    /// HTTP server settings
    struct HttpConfig {
        std::string endpoint = "0.0.0.0:4318";
        size_t max_request_body_bytes = 16 * 1024 * 1024;  // 16 MB
        std::chrono::seconds read_timeout{30};
        std::chrono::seconds write_timeout{30};

        /// CORS configuration
        struct CorsConfig {
            std::vector<std::string> allowed_origins;  // Empty = deny all, ["*"] = allow all
            std::vector<std::string> allowed_methods = {"POST", "GET", "OPTIONS"};
            std::vector<std::string> allowed_headers = {"Content-Type"};
            bool allow_credentials = false;
        };
        CorsConfig cors;

        /// TLS configuration for HTTPS
        struct TlsConfig {
            bool enabled = false;
            std::string cert_path;
            std::string key_path;
        };
        TlsConfig tls;
    };
    HttpConfig http;

    /// Rate limiting configuration
    struct RateLimitConfig {
        bool enabled = true;
        double requests_per_second = 1000.0;  // Global limit
        double per_ip_requests_per_second = 100.0;  // Per-IP limit
        size_t burst_size = 100;  // Allow burst above rate
    };
    RateLimitConfig rate_limit;

    /// JSON parsing security limits
    struct JsonLimits {
        size_t max_depth = 32;           // Maximum nesting depth
        size_t max_string_length = 1024 * 1024;  // 1 MB per string
        size_t max_array_elements = 10000;
    };
    JsonLimits json_limits;
};

/// Statistics for the receiver
struct OtlpReceiverStats {
    std::atomic<uint64_t> grpc_requests_total{0};
    std::atomic<uint64_t> grpc_requests_failed{0};
    std::atomic<uint64_t> http_requests_total{0};
    std::atomic<uint64_t> http_requests_failed{0};
    std::atomic<uint64_t> spans_received{0};
    std::atomic<uint64_t> metrics_received{0};
    std::atomic<uint64_t> logs_received{0};
    std::atomic<uint64_t> bytes_received{0};
    std::atomic<uint64_t> rate_limited_requests{0};
    std::atomic<uint64_t> cors_rejected_requests{0};
};

/// OTLP Receiver that accepts traces, metrics, and logs over gRPC and HTTP
class OtlpReceiver {
public:
    /// Create receiver with configuration
    explicit OtlpReceiver(OtlpReceiverConfig config);

    /// Destructor
    ~OtlpReceiver();

    // Non-copyable, non-movable
    OtlpReceiver(const OtlpReceiver&) = delete;
    OtlpReceiver& operator=(const OtlpReceiver&) = delete;

    /// Register callback for received spans
    void OnSpans(SpanCallback callback);

    /// Register callback for received metrics
    void OnMetrics(MetricCallback callback);

    /// Register callback for received logs
    void OnLogs(LogCallback callback);

    /// Start the receiver (non-blocking)
    absl::Status Start();

    /// Stop the receiver
    absl::Status Shutdown();

    /// Check if receiver is running
    bool IsRunning() const;

    /// Get statistics
    const OtlpReceiverStats& GetStats() const;

    /// Get configuration
    const OtlpReceiverConfig& GetConfig() const { return config_; }

private:
    class Impl;
    OtlpReceiverConfig config_;
    std::unique_ptr<Impl> impl_;
};

}  // namespace pyflare::collector
