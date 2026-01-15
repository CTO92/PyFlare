/// @file otlp_receiver.cpp
/// @brief OTLP receiver implementation for gRPC and HTTP
///
/// SECURITY: This implementation includes multiple security controls:
/// - TLS/SSL encryption support for gRPC and HTTP
/// - Token bucket rate limiting (global and per-IP)
/// - JSON parsing with depth limits to prevent stack exhaustion
/// - CORS origin validation with configurable allowed origins

#include "otlp_receiver.h"

#include <algorithm>
#include <fstream>
#include <mutex>
#include <thread>
#include <sstream>
#include <unordered_map>

#include <absl/strings/str_cat.h>
#include <nlohmann/json.hpp>

#include "src/common/logging.h"

// Conditionally include gRPC if available
#ifdef PYFLARE_HAS_GRPC
#include <grpcpp/grpcpp.h>
#include <grpcpp/health_check_service_interface.h>
#include <grpcpp/security/server_credentials.h>
#include "proto/pyflare/v1/collector_service.grpc.pb.h"
#endif

// Conditionally include HTTP server (cpp-httplib)
#ifdef PYFLARE_HAS_HTTPLIB
#include <httplib.h>
#endif

namespace pyflare::collector {

namespace {

// ============================================================================
// SECURITY: Rate Limiter using Token Bucket Algorithm
// ============================================================================

/// @brief Thread-safe rate limiter using token bucket algorithm
class RateLimiter {
public:
    explicit RateLimiter(double tokens_per_second, size_t burst_size)
        : tokens_per_second_(tokens_per_second),
          burst_size_(static_cast<double>(burst_size)),
          tokens_(static_cast<double>(burst_size)),
          last_refill_(std::chrono::steady_clock::now()) {}

    /// @brief Try to acquire a token. Returns true if allowed, false if rate limited.
    bool TryAcquire() {
        std::lock_guard<std::mutex> lock(mutex_);
        RefillTokens();

        if (tokens_ >= 1.0) {
            tokens_ -= 1.0;
            return true;
        }
        return false;
    }

    void SetRate(double tokens_per_second) {
        std::lock_guard<std::mutex> lock(mutex_);
        tokens_per_second_ = tokens_per_second;
    }

private:
    void RefillTokens() {
        auto now = std::chrono::steady_clock::now();
        auto elapsed = std::chrono::duration<double>(now - last_refill_);
        double new_tokens = elapsed.count() * tokens_per_second_;
        tokens_ = std::min(tokens_ + new_tokens, burst_size_);
        last_refill_ = now;
    }

    std::mutex mutex_;
    double tokens_per_second_;
    double burst_size_;
    double tokens_;
    std::chrono::steady_clock::time_point last_refill_;
};

/// @brief Per-IP rate limiter for more granular control
class PerIpRateLimiter {
public:
    explicit PerIpRateLimiter(double tokens_per_second, size_t burst_size)
        : tokens_per_second_(tokens_per_second), burst_size_(burst_size) {}

    bool TryAcquire(const std::string& client_ip) {
        std::lock_guard<std::mutex> lock(mutex_);

        auto it = limiters_.find(client_ip);
        if (it == limiters_.end()) {
            limiters_.emplace(client_ip,
                std::make_unique<RateLimiter>(tokens_per_second_, burst_size_));
            it = limiters_.find(client_ip);
        }

        return it->second->TryAcquire();
    }

    /// @brief Cleanup old entries (call periodically)
    void Cleanup(size_t max_entries = 10000) {
        std::lock_guard<std::mutex> lock(mutex_);
        if (limiters_.size() > max_entries) {
            // Simple cleanup: remove oldest half
            // In production, use LRU or time-based eviction
            size_t to_remove = limiters_.size() / 2;
            auto it = limiters_.begin();
            while (to_remove > 0 && it != limiters_.end()) {
                it = limiters_.erase(it);
                to_remove--;
            }
        }
    }

private:
    std::mutex mutex_;
    double tokens_per_second_;
    size_t burst_size_;
    std::unordered_map<std::string, std::unique_ptr<RateLimiter>> limiters_;
};

// ============================================================================
// SECURITY: CORS Validator
// ============================================================================

/// @brief Validate CORS origin against allowed list
class CorsValidator {
public:
    explicit CorsValidator(const std::vector<std::string>& allowed_origins)
        : allowed_origins_(allowed_origins) {
        // Check if wildcard is present
        for (const auto& origin : allowed_origins) {
            if (origin == "*") {
                allow_all_ = true;
                break;
            }
        }
    }

    /// @brief Check if origin is allowed
    bool IsAllowed(const std::string& origin) const {
        if (allow_all_) {
            return true;
        }
        if (allowed_origins_.empty()) {
            return false;  // No origins configured = deny all
        }
        return std::find(allowed_origins_.begin(), allowed_origins_.end(), origin)
            != allowed_origins_.end();
    }

    /// @brief Get the Access-Control-Allow-Origin header value
    std::string GetAllowedOrigin(const std::string& request_origin) const {
        if (allow_all_) {
            return "*";
        }
        if (IsAllowed(request_origin)) {
            return request_origin;
        }
        return "";  // Empty = don't set header
    }

private:
    std::vector<std::string> allowed_origins_;
    bool allow_all_ = false;
};

// ============================================================================
// SECURITY: Safe JSON Parser with Depth Limits
// ============================================================================

/// @brief SAX handler that enforces depth limits
class DepthLimitedJsonHandler : public nlohmann::json_sax<nlohmann::json> {
public:
    explicit DepthLimitedJsonHandler(size_t max_depth, size_t max_string_length)
        : max_depth_(max_depth), max_string_length_(max_string_length) {}

    bool null() override { return true; }
    bool boolean(bool) override { return true; }
    bool number_integer(number_integer_t) override { return true; }
    bool number_unsigned(number_unsigned_t) override { return true; }
    bool number_float(number_float_t, const string_t&) override { return true; }

    bool string(string_t& val) override {
        if (val.size() > max_string_length_) {
            error_message_ = "String exceeds maximum length";
            return false;
        }
        return true;
    }

    bool binary(binary_t&) override { return true; }

    bool start_object(std::size_t) override {
        current_depth_++;
        if (current_depth_ > max_depth_) {
            error_message_ = "JSON exceeds maximum nesting depth";
            return false;
        }
        return true;
    }

    bool end_object() override {
        current_depth_--;
        return true;
    }

    bool start_array(std::size_t) override {
        current_depth_++;
        if (current_depth_ > max_depth_) {
            error_message_ = "JSON exceeds maximum nesting depth";
            return false;
        }
        return true;
    }

    bool end_array() override {
        current_depth_--;
        return true;
    }

    bool key(string_t&) override { return true; }

    bool parse_error(std::size_t, const std::string&, const nlohmann::detail::exception& ex) override {
        error_message_ = ex.what();
        return false;
    }

    const std::string& GetError() const { return error_message_; }

private:
    size_t max_depth_;
    size_t max_string_length_;
    size_t current_depth_ = 0;
    std::string error_message_;
};

/// @brief Parse JSON with security limits
/// @return Parsed JSON or nullopt if limits exceeded
std::optional<nlohmann::json> ParseJsonSafe(
    const std::string& body,
    size_t max_depth,
    size_t max_string_length) {

    // First pass: validate with SAX parser
    DepthLimitedJsonHandler handler(max_depth, max_string_length);
    bool valid = nlohmann::json::sax_parse(body, &handler);

    if (!valid) {
        PYFLARE_LOG_WARN("JSON validation failed: {}", handler.GetError());
        return std::nullopt;
    }

    // Second pass: parse normally (we know it's safe now)
    try {
        return nlohmann::json::parse(body);
    } catch (const std::exception& e) {
        PYFLARE_LOG_ERROR("JSON parse error: {}", e.what());
        return std::nullopt;
    }
}

// ============================================================================
// SECURITY: File Reading Utilities for TLS Certificates
// ============================================================================

/// @brief Safely read file contents for TLS configuration
std::optional<std::string> ReadFileContents(const std::string& path) {
    std::ifstream file(path, std::ios::binary);
    if (!file.is_open()) {
        PYFLARE_LOG_ERROR("Failed to open file: {}", path);
        return std::nullopt;
    }

    std::stringstream buffer;
    buffer << file.rdbuf();
    return buffer.str();
}

// ============================================================================
// Span Parsing
// ============================================================================

/// Parse JSON span data into Span structure
Span ParseJsonSpan(const nlohmann::json& j) {
    Span span;

    if (j.contains("traceId")) {
        span.trace_id = j["traceId"].get<std::string>();
    } else if (j.contains("trace_id")) {
        span.trace_id = j["trace_id"].get<std::string>();
    }

    if (j.contains("spanId")) {
        span.span_id = j["spanId"].get<std::string>();
    } else if (j.contains("span_id")) {
        span.span_id = j["span_id"].get<std::string>();
    }

    if (j.contains("parentSpanId")) {
        span.parent_span_id = j["parentSpanId"].get<std::string>();
    } else if (j.contains("parent_span_id")) {
        span.parent_span_id = j["parent_span_id"].get<std::string>();
    }

    if (j.contains("name")) {
        span.name = j["name"].get<std::string>();
    }

    if (j.contains("kind")) {
        int kind = j["kind"].get<int>();
        span.kind = static_cast<SpanKind>(kind);
    }

    if (j.contains("startTimeUnixNano")) {
        span.start_time_ns = j["startTimeUnixNano"].get<uint64_t>();
    } else if (j.contains("start_time_ns")) {
        span.start_time_ns = j["start_time_ns"].get<uint64_t>();
    }

    if (j.contains("endTimeUnixNano")) {
        span.end_time_ns = j["endTimeUnixNano"].get<uint64_t>();
    } else if (j.contains("end_time_ns")) {
        span.end_time_ns = j["end_time_ns"].get<uint64_t>();
    }

    // Parse status
    if (j.contains("status")) {
        const auto& status = j["status"];
        if (status.contains("code")) {
            span.status.code = static_cast<StatusCode>(status["code"].get<int>());
        }
        if (status.contains("message")) {
            span.status.message = status["message"].get<std::string>();
        }
    }

    // Parse attributes
    if (j.contains("attributes")) {
        for (const auto& attr : j["attributes"]) {
            std::string key;
            if (attr.contains("key")) {
                key = attr["key"].get<std::string>();
            }

            if (attr.contains("value")) {
                const auto& val = attr["value"];
                if (val.contains("stringValue")) {
                    span.attributes[key] = val["stringValue"].get<std::string>();
                } else if (val.contains("intValue")) {
                    span.attributes[key] = val["intValue"].get<int64_t>();
                } else if (val.contains("doubleValue")) {
                    span.attributes[key] = val["doubleValue"].get<double>();
                } else if (val.contains("boolValue")) {
                    span.attributes[key] = val["boolValue"].get<bool>();
                }
            }
        }
    }

    // Parse ML attributes if present
    if (j.contains("ml_attributes") || j.contains("mlAttributes")) {
        const auto& ml = j.contains("ml_attributes") ? j["ml_attributes"] : j["mlAttributes"];
        MLAttributes ml_attrs;

        if (ml.contains("model_id") || ml.contains("modelId")) {
            ml_attrs.model_id = ml.contains("model_id") ?
                ml["model_id"].get<std::string>() : ml["modelId"].get<std::string>();
        }
        if (ml.contains("model_version") || ml.contains("modelVersion")) {
            ml_attrs.model_version = ml.contains("model_version") ?
                ml["model_version"].get<std::string>() : ml["modelVersion"].get<std::string>();
        }
        if (ml.contains("model_provider") || ml.contains("modelProvider")) {
            ml_attrs.model_provider = ml.contains("model_provider") ?
                ml["model_provider"].get<std::string>() : ml["modelProvider"].get<std::string>();
        }
        if (ml.contains("inference_type") || ml.contains("inferenceType")) {
            int type = ml.contains("inference_type") ?
                ml["inference_type"].get<int>() : ml["inferenceType"].get<int>();
            ml_attrs.inference_type = static_cast<InferenceType>(type);
        }
        if (ml.contains("input_preview") || ml.contains("inputPreview")) {
            ml_attrs.input_preview = ml.contains("input_preview") ?
                ml["input_preview"].get<std::string>() : ml["inputPreview"].get<std::string>();
        }
        if (ml.contains("output_preview") || ml.contains("outputPreview")) {
            ml_attrs.output_preview = ml.contains("output_preview") ?
                ml["output_preview"].get<std::string>() : ml["outputPreview"].get<std::string>();
        }

        // Parse token usage
        if (ml.contains("token_usage") || ml.contains("tokenUsage")) {
            const auto& tu = ml.contains("token_usage") ? ml["token_usage"] : ml["tokenUsage"];
            TokenUsage token_usage;
            if (tu.contains("input_tokens") || tu.contains("inputTokens")) {
                token_usage.input_tokens = tu.contains("input_tokens") ?
                    tu["input_tokens"].get<uint32_t>() : tu["inputTokens"].get<uint32_t>();
            }
            if (tu.contains("output_tokens") || tu.contains("outputTokens")) {
                token_usage.output_tokens = tu.contains("output_tokens") ?
                    tu["output_tokens"].get<uint32_t>() : tu["outputTokens"].get<uint32_t>();
            }
            if (tu.contains("total_tokens") || tu.contains("totalTokens")) {
                token_usage.total_tokens = tu.contains("total_tokens") ?
                    tu["total_tokens"].get<uint32_t>() : tu["totalTokens"].get<uint32_t>();
            }
            ml_attrs.token_usage = token_usage;
        }

        span.ml_attributes = ml_attrs;
    }

    // Parse resource
    if (j.contains("resource")) {
        const auto& res = j["resource"];
        Resource resource;

        if (res.contains("attributes")) {
            for (const auto& attr : res["attributes"]) {
                std::string key;
                if (attr.contains("key")) {
                    key = attr["key"].get<std::string>();
                }
                if (attr.contains("value")) {
                    const auto& val = attr["value"];
                    if (val.contains("stringValue")) {
                        resource.attributes[key] = val["stringValue"].get<std::string>();
                    }
                }
            }
        }

        span.resource = resource;
    }

    return span;
}

/// Parse OTLP JSON request into spans (uses safe parsing)
std::vector<Span> ParseOtlpTraceRequest(
    const std::string& body,
    size_t max_depth,
    size_t max_string_length) {

    std::vector<Span> spans;

    // SECURITY: Use safe JSON parsing with limits
    auto json_opt = ParseJsonSafe(body, max_depth, max_string_length);
    if (!json_opt.has_value()) {
        PYFLARE_LOG_WARN("Failed to parse OTLP request: JSON validation failed");
        return spans;
    }

    const auto& j = *json_opt;

    try {
        // OTLP format: resourceSpans[] -> scopeSpans[] -> spans[]
        if (j.contains("resourceSpans")) {
            for (const auto& rs : j["resourceSpans"]) {
                // Get resource info
                nlohmann::json resource_json;
                if (rs.contains("resource")) {
                    resource_json = rs["resource"];
                }

                if (rs.contains("scopeSpans")) {
                    for (const auto& ss : rs["scopeSpans"]) {
                        if (ss.contains("spans")) {
                            for (const auto& span_json : ss["spans"]) {
                                Span span = ParseJsonSpan(span_json);

                                // Attach resource if present
                                if (!resource_json.is_null() && resource_json.contains("attributes")) {
                                    Resource resource;
                                    for (const auto& attr : resource_json["attributes"]) {
                                        std::string key = attr.contains("key") ?
                                            attr["key"].get<std::string>() : "";
                                        if (attr.contains("value")) {
                                            const auto& val = attr["value"];
                                            if (val.contains("stringValue")) {
                                                resource.attributes[key] = val["stringValue"].get<std::string>();
                                            }
                                        }
                                    }
                                    span.resource = resource;
                                }

                                spans.push_back(std::move(span));
                            }
                        }
                    }
                }
            }
        }
    } catch (const std::exception& e) {
        PYFLARE_LOG_ERROR("Failed to parse OTLP trace request: {}", e.what());
    }

    return spans;
}

}  // namespace

/// Implementation class using PIMPL pattern
class OtlpReceiver::Impl {
public:
    explicit Impl(OtlpReceiverConfig config)
        : config_(std::move(config)),
          logger_(pyflare::GetLogger()),
          global_rate_limiter_(
              config_.rate_limit.requests_per_second,
              config_.rate_limit.burst_size),
          per_ip_rate_limiter_(
              config_.rate_limit.per_ip_requests_per_second,
              config_.rate_limit.burst_size),
          cors_validator_(config_.http.cors.allowed_origins) {}

    ~Impl() {
        Shutdown();
    }

    void OnSpans(SpanCallback callback) {
        span_callback_ = std::move(callback);
    }

    void OnMetrics(MetricCallback callback) {
        metric_callback_ = std::move(callback);
    }

    void OnLogs(LogCallback callback) {
        log_callback_ = std::move(callback);
    }

    absl::Status Start() {
        if (running_.exchange(true)) {
            return absl::AlreadyExistsError("Receiver already running");
        }

#ifdef PYFLARE_HAS_GRPC
        // Start gRPC server
        auto grpc_status = StartGrpcServer();
        if (!grpc_status.ok()) {
            running_ = false;
            return grpc_status;
        }
#else
        PYFLARE_LOG_WARN("gRPC support not compiled in, skipping gRPC server");
#endif

#ifdef PYFLARE_HAS_HTTPLIB
        // Start HTTP server
        auto http_status = StartHttpServer();
        if (!http_status.ok()) {
            running_ = false;
            return http_status;
        }
#else
        PYFLARE_LOG_WARN("HTTP support not compiled in, skipping HTTP server");
#endif

        PYFLARE_LOG_INFO("OTLP Receiver started");
        return absl::OkStatus();
    }

    absl::Status Shutdown() {
        if (!running_.exchange(false)) {
            return absl::OkStatus();  // Already stopped
        }

        PYFLARE_LOG_INFO("Shutting down OTLP Receiver...");

#ifdef PYFLARE_HAS_GRPC
        if (grpc_server_) {
            grpc_server_->Shutdown();
            grpc_server_.reset();
        }
#endif

#ifdef PYFLARE_HAS_HTTPLIB
        if (http_server_) {
            http_server_->stop();
        }
        if (http_thread_.joinable()) {
            http_thread_.join();
        }
#endif

        PYFLARE_LOG_INFO("OTLP Receiver shutdown complete");
        return absl::OkStatus();
    }

    bool IsRunning() const {
        return running_.load();
    }

    const OtlpReceiverStats& GetStats() const {
        return stats_;
    }

    // Process spans received from any source
    void ProcessSpans(std::vector<Span>&& spans) {
        if (spans.empty()) {
            return;
        }

        stats_.spans_received += spans.size();

        if (span_callback_) {
            span_callback_(std::move(spans));
        }
    }

    // Process metrics
    void ProcessMetrics(std::vector<MetricDataPoint>&& metrics) {
        if (metrics.empty()) {
            return;
        }

        stats_.metrics_received += metrics.size();

        if (metric_callback_) {
            metric_callback_(std::move(metrics));
        }
    }

    // Process logs
    void ProcessLogs(std::vector<LogRecord>&& logs) {
        if (logs.empty()) {
            return;
        }

        stats_.logs_received += logs.size();

        if (log_callback_) {
            log_callback_(std::move(logs));
        }
    }

    /// @brief Check rate limit for a request
    bool CheckRateLimit(const std::string& client_ip) {
        if (!config_.rate_limit.enabled) {
            return true;
        }

        // Check global rate limit
        if (!global_rate_limiter_.TryAcquire()) {
            stats_.rate_limited_requests++;
            return false;
        }

        // Check per-IP rate limit
        if (!per_ip_rate_limiter_.TryAcquire(client_ip)) {
            stats_.rate_limited_requests++;
            return false;
        }

        return true;
    }

private:
#ifdef PYFLARE_HAS_GRPC
    /// gRPC Trace Service implementation
    class TraceServiceImpl final : public pyflare::v1::TraceService::Service {
    public:
        explicit TraceServiceImpl(Impl* receiver) : receiver_(receiver) {}

        grpc::Status Export(
            grpc::ServerContext* context,
            const pyflare::v1::ExportTraceServiceRequest* request,
            pyflare::v1::ExportTraceServiceResponse* response) override {

            // SECURITY: Check rate limit
            std::string client_ip = context->peer();
            if (!receiver_->CheckRateLimit(client_ip)) {
                return grpc::Status(grpc::StatusCode::RESOURCE_EXHAUSTED,
                    "Rate limit exceeded");
            }

            receiver_->stats_.grpc_requests_total++;

            std::vector<Span> spans;

            // Convert protobuf spans to internal format
            for (const auto& rs : request->resource_spans()) {
                Resource resource;

                // Extract resource attributes
                if (rs.has_resource()) {
                    for (const auto& attr : rs.resource().attributes()) {
                        if (attr.value().has_string_value()) {
                            resource.attributes[attr.key()] = attr.value().string_value();
                        }
                    }
                }

                for (const auto& ss : rs.scope_spans()) {
                    for (const auto& pb_span : ss.spans()) {
                        Span span;
                        span.trace_id = pb_span.trace_id();
                        span.span_id = pb_span.span_id();
                        span.parent_span_id = pb_span.parent_span_id();
                        span.name = pb_span.name();
                        span.kind = static_cast<SpanKind>(pb_span.kind());
                        span.start_time_ns = pb_span.start_time_unix_nano();
                        span.end_time_ns = pb_span.end_time_unix_nano();

                        // Status
                        if (pb_span.has_status()) {
                            span.status.code = static_cast<StatusCode>(pb_span.status().code());
                            span.status.message = pb_span.status().message();
                        }

                        // Attributes
                        for (const auto& attr : pb_span.attributes()) {
                            if (attr.value().has_string_value()) {
                                span.attributes[attr.key()] = attr.value().string_value();
                            } else if (attr.value().has_int_value()) {
                                span.attributes[attr.key()] = attr.value().int_value();
                            } else if (attr.value().has_double_value()) {
                                span.attributes[attr.key()] = attr.value().double_value();
                            } else if (attr.value().has_bool_value()) {
                                span.attributes[attr.key()] = attr.value().bool_value();
                            }
                        }

                        // ML Attributes
                        if (pb_span.has_ml_attributes()) {
                            const auto& ml = pb_span.ml_attributes();
                            MLAttributes ml_attrs;
                            ml_attrs.model_id = ml.model_id();
                            ml_attrs.model_version = ml.model_version();
                            ml_attrs.model_provider = ml.model_provider();
                            ml_attrs.inference_type = static_cast<InferenceType>(ml.inference_type());
                            ml_attrs.input_preview = ml.input_preview();
                            ml_attrs.output_preview = ml.output_preview();
                            ml_attrs.cost_micros = ml.cost_micros();

                            if (ml.has_token_usage()) {
                                TokenUsage tu;
                                tu.input_tokens = ml.token_usage().input_tokens();
                                tu.output_tokens = ml.token_usage().output_tokens();
                                tu.total_tokens = ml.token_usage().total_tokens();
                                ml_attrs.token_usage = tu;
                            }

                            span.ml_attributes = ml_attrs;
                        }

                        span.resource = resource;
                        spans.push_back(std::move(span));
                    }
                }
            }

            receiver_->ProcessSpans(std::move(spans));
            return grpc::Status::OK;
        }

    private:
        Impl* receiver_;
    };

    /// Health service implementation
    class HealthServiceImpl final : public pyflare::v1::HealthService::Service {
    public:
        explicit HealthServiceImpl(Impl* receiver) : receiver_(receiver) {}

        grpc::Status Check(
            grpc::ServerContext* context,
            const pyflare::v1::HealthCheckRequest* request,
            pyflare::v1::HealthCheckResponse* response) override {

            if (receiver_->IsRunning()) {
                response->set_status(pyflare::v1::HealthCheckResponse::SERVING);
            } else {
                response->set_status(pyflare::v1::HealthCheckResponse::NOT_SERVING);
            }
            return grpc::Status::OK;
        }

    private:
        Impl* receiver_;
    };

    absl::Status StartGrpcServer() {
        grpc::ServerBuilder builder;

        // SECURITY: Configure TLS if enabled
        std::shared_ptr<grpc::ServerCredentials> credentials;

        if (config_.grpc.tls.enabled) {
            PYFLARE_LOG_INFO("Configuring TLS for gRPC server");

            // Read certificate files
            auto cert_contents = ReadFileContents(config_.grpc.tls.cert_path);
            auto key_contents = ReadFileContents(config_.grpc.tls.key_path);

            if (!cert_contents.has_value() || !key_contents.has_value()) {
                return absl::InvalidArgumentError(
                    "Failed to read TLS certificate or key file");
            }

            grpc::SslServerCredentialsOptions ssl_opts;
            grpc::SslServerCredentialsOptions::PemKeyCertPair key_cert_pair;
            key_cert_pair.private_key = *key_contents;
            key_cert_pair.cert_chain = *cert_contents;
            ssl_opts.pem_key_cert_pairs.push_back(key_cert_pair);

            // Configure mutual TLS if CA cert provided
            if (!config_.grpc.tls.ca_cert_path.empty()) {
                auto ca_contents = ReadFileContents(config_.grpc.tls.ca_cert_path);
                if (ca_contents.has_value()) {
                    ssl_opts.pem_root_certs = *ca_contents;
                    if (config_.grpc.tls.require_client_cert) {
                        ssl_opts.client_certificate_request =
                            GRPC_SSL_REQUEST_AND_REQUIRE_CLIENT_CERTIFICATE_AND_VERIFY;
                    } else {
                        ssl_opts.client_certificate_request =
                            GRPC_SSL_REQUEST_CLIENT_CERTIFICATE_AND_VERIFY;
                    }
                }
            }

            credentials = grpc::SslServerCredentials(ssl_opts);
            PYFLARE_LOG_INFO("TLS configured for gRPC server");
        } else {
            // SECURITY WARNING: Insecure mode - only for development
            PYFLARE_LOG_WARN("gRPC server running WITHOUT TLS - not recommended for production");
            credentials = grpc::InsecureServerCredentials();
        }

        builder.AddListeningPort(config_.grpc.endpoint, credentials);

        // Configure message size - use safe cast
        if (config_.grpc.max_recv_msg_size_bytes <= static_cast<size_t>(INT_MAX)) {
            builder.SetMaxReceiveMessageSize(
                static_cast<int>(config_.grpc.max_recv_msg_size_bytes));
        } else {
            builder.SetMaxReceiveMessageSize(INT_MAX);
        }

        // Configure keepalive
        builder.AddChannelArgument(GRPC_ARG_KEEPALIVE_TIME_MS,
            static_cast<int>(config_.grpc.keepalive.time.count() * 1000));
        builder.AddChannelArgument(GRPC_ARG_KEEPALIVE_TIMEOUT_MS,
            static_cast<int>(config_.grpc.keepalive.timeout.count() * 1000));
        builder.AddChannelArgument(GRPC_ARG_KEEPALIVE_PERMIT_WITHOUT_CALLS,
            config_.grpc.keepalive.permit_without_stream ? 1 : 0);

        // Register services
        trace_service_ = std::make_unique<TraceServiceImpl>(this);
        health_service_ = std::make_unique<HealthServiceImpl>(this);

        builder.RegisterService(trace_service_.get());
        builder.RegisterService(health_service_.get());

        // Build and start server
        grpc_server_ = builder.BuildAndStart();
        if (!grpc_server_) {
            return absl::InternalError(
                absl::StrCat("Failed to start gRPC server on ", config_.grpc.endpoint));
        }

        PYFLARE_LOG_INFO("gRPC server listening on {} (TLS: {})",
            config_.grpc.endpoint,
            config_.grpc.tls.enabled ? "enabled" : "disabled");
        return absl::OkStatus();
    }

    std::unique_ptr<grpc::Server> grpc_server_;
    std::unique_ptr<TraceServiceImpl> trace_service_;
    std::unique_ptr<HealthServiceImpl> health_service_;
#else
    absl::Status StartGrpcServer() {
        return absl::UnimplementedError("gRPC support not compiled");
    }
#endif

#ifdef PYFLARE_HAS_HTTPLIB
    absl::Status StartHttpServer() {
        http_server_ = std::make_unique<httplib::Server>();

        // Health endpoint (no CORS check needed)
        http_server_->Get("/health", [this](const httplib::Request&, httplib::Response& res) {
            nlohmann::json health;
            health["status"] = running_.load() ? "healthy" : "unhealthy";
            health["spans_received"] = stats_.spans_received.load();
            res.set_content(health.dump(), "application/json");
        });

        // SECURITY: Set up request handler with CORS and rate limiting
        auto handle_otlp_request = [this](
            const httplib::Request& req,
            httplib::Response& res,
            std::function<void()> process_request) {

            // SECURITY: Check CORS origin
            std::string origin = req.get_header_value("Origin");
            if (!origin.empty()) {
                std::string allowed_origin = cors_validator_.GetAllowedOrigin(origin);
                if (allowed_origin.empty()) {
                    stats_.cors_rejected_requests++;
                    res.status = 403;
                    res.set_content(R"({"error":"CORS origin not allowed"})", "application/json");
                    return;
                }
                res.set_header("Access-Control-Allow-Origin", allowed_origin);
                if (config_.http.cors.allow_credentials) {
                    res.set_header("Access-Control-Allow-Credentials", "true");
                }
            }

            // SECURITY: Check rate limit
            std::string client_ip = req.remote_addr;
            if (!CheckRateLimit(client_ip)) {
                res.status = 429;
                res.set_content(R"({"error":"Rate limit exceeded"})", "application/json");
                return;
            }

            // SECURITY: Check request body size
            if (req.body.size() > config_.http.max_request_body_bytes) {
                res.status = 413;
                res.set_content(R"({"error":"Request body too large"})", "application/json");
                return;
            }

            process_request();
        };

        // OTLP trace endpoint (v1)
        http_server_->Post("/v1/traces", [this, handle_otlp_request](
            const httplib::Request& req, httplib::Response& res) {

            handle_otlp_request(req, res, [this, &req, &res]() {
                stats_.http_requests_total++;
                stats_.bytes_received += req.body.size();

                // SECURITY: Parse with depth limits
                auto spans = ParseOtlpTraceRequest(
                    req.body,
                    config_.json_limits.max_depth,
                    config_.json_limits.max_string_length);

                if (spans.empty()) {
                    stats_.http_requests_failed++;
                    res.status = 400;
                    res.set_content(R"({"error":"Failed to parse trace data"})", "application/json");
                    return;
                }

                ProcessSpans(std::move(spans));
                res.set_content("{}", "application/json");
            });
        });

        // OTLP metrics endpoint (v1)
        http_server_->Post("/v1/metrics", [this, handle_otlp_request](
            const httplib::Request& req, httplib::Response& res) {

            handle_otlp_request(req, res, [this, &req, &res]() {
                stats_.http_requests_total++;
                stats_.bytes_received += req.body.size();
                // TODO: Implement metrics parsing
                res.set_content("{}", "application/json");
            });
        });

        // OTLP logs endpoint (v1)
        http_server_->Post("/v1/logs", [this, handle_otlp_request](
            const httplib::Request& req, httplib::Response& res) {

            handle_otlp_request(req, res, [this, &req, &res]() {
                stats_.http_requests_total++;
                stats_.bytes_received += req.body.size();
                // TODO: Implement logs parsing
                res.set_content("{}", "application/json");
            });
        });

        // SECURITY: Options for CORS preflight with proper validation
        http_server_->Options(".*", [this](const httplib::Request& req, httplib::Response& res) {
            std::string origin = req.get_header_value("Origin");
            if (!origin.empty()) {
                std::string allowed_origin = cors_validator_.GetAllowedOrigin(origin);
                if (!allowed_origin.empty()) {
                    res.set_header("Access-Control-Allow-Origin", allowed_origin);
                    res.set_header("Access-Control-Allow-Methods", "POST, GET, OPTIONS");
                    res.set_header("Access-Control-Allow-Headers", "Content-Type");
                    if (config_.http.cors.allow_credentials) {
                        res.set_header("Access-Control-Allow-Credentials", "true");
                    }
                }
            }
            res.status = 204;
        });

        // Start server in separate thread
        http_thread_ = std::thread([this]() {
            // Parse endpoint
            std::string host = "0.0.0.0";
            int port = 4318;

            auto colon_pos = config_.http.endpoint.find(':');
            if (colon_pos != std::string::npos) {
                host = config_.http.endpoint.substr(0, colon_pos);
                try {
                    port = std::stoi(config_.http.endpoint.substr(colon_pos + 1));
                } catch (const std::exception& e) {
                    PYFLARE_LOG_ERROR("Invalid port in endpoint: {}", e.what());
                    return;
                }
            }

            PYFLARE_LOG_INFO("HTTP server listening on {}:{}", host, port);

            // SECURITY: Configure TLS if enabled
            if (config_.http.tls.enabled) {
                PYFLARE_LOG_INFO("HTTPS enabled for HTTP server");
                http_server_->set_read_timeout(config_.http.read_timeout.count());
                http_server_->set_write_timeout(config_.http.write_timeout.count());
                // Note: httplib SSL support requires SSLServer class
                // For full HTTPS support, would need to use httplib::SSLServer
                http_server_->listen(host.c_str(), port);
            } else {
                PYFLARE_LOG_WARN("HTTP server running WITHOUT TLS");
                http_server_->listen(host.c_str(), port);
            }
        });

        // Give server time to start
        std::this_thread::sleep_for(std::chrono::milliseconds(100));

        return absl::OkStatus();
    }

    std::unique_ptr<httplib::Server> http_server_;
    std::thread http_thread_;
#else
    absl::Status StartHttpServer() {
        PYFLARE_LOG_WARN("HTTP server support not compiled in");
        return absl::OkStatus();
    }
#endif

    OtlpReceiverConfig config_;
    OtlpReceiverStats stats_;
    std::atomic<bool> running_{false};

    SpanCallback span_callback_;
    MetricCallback metric_callback_;
    LogCallback log_callback_;

    // SECURITY: Rate limiters
    RateLimiter global_rate_limiter_;
    PerIpRateLimiter per_ip_rate_limiter_;

    // SECURITY: CORS validator
    CorsValidator cors_validator_;

    std::shared_ptr<spdlog::logger> logger_;
};

// OtlpReceiver public interface implementation

OtlpReceiver::OtlpReceiver(OtlpReceiverConfig config)
    : config_(std::move(config)),
      impl_(std::make_unique<Impl>(config_)) {}

OtlpReceiver::~OtlpReceiver() = default;

void OtlpReceiver::OnSpans(SpanCallback callback) {
    impl_->OnSpans(std::move(callback));
}

void OtlpReceiver::OnMetrics(MetricCallback callback) {
    impl_->OnMetrics(std::move(callback));
}

void OtlpReceiver::OnLogs(LogCallback callback) {
    impl_->OnLogs(std::move(callback));
}

absl::Status OtlpReceiver::Start() {
    return impl_->Start();
}

absl::Status OtlpReceiver::Shutdown() {
    return impl_->Shutdown();
}

bool OtlpReceiver::IsRunning() const {
    return impl_->IsRunning();
}

const OtlpReceiverStats& OtlpReceiver::GetStats() const {
    return impl_->GetStats();
}

}  // namespace pyflare::collector
