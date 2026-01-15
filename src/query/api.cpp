#include "api.h"

#include <chrono>
#include <mutex>
#include <regex>
#include <unordered_map>

#include "common/logging.h"

namespace pyflare::query {

// Forward declaration for SQL validation (defined in sql_parser.cpp)
absl::Status IsQuerySafe(const std::string& sql);

// =============================================================================
// SECURITY: Authentication and Authorization
// =============================================================================

/// @brief API Key validation result
struct AuthResult {
    bool authenticated = false;
    std::string client_id;
    std::vector<std::string> permissions;
    std::string error_message;
};

/// @brief Validate API key format (alphanumeric with optional prefix)
static bool IsValidApiKeyFormat(const std::string& key) {
    if (key.empty() || key.size() > 256) {
        return false;
    }
    // Format: prefix_base64characters or just alphanumeric
    static const std::regex api_key_regex("^[a-zA-Z0-9_\\-]+$");
    return std::regex_match(key, api_key_regex);
}

/// @brief Constant-time string comparison to prevent timing attacks
static bool SecureCompare(const std::string& a, const std::string& b) {
    if (a.size() != b.size()) {
        return false;
    }
    volatile int result = 0;
    for (size_t i = 0; i < a.size(); ++i) {
        result |= a[i] ^ b[i];
    }
    return result == 0;
}

// =============================================================================
// SECURITY: Rate Limiting
// =============================================================================

/// @brief Simple token bucket rate limiter
class RateLimiter {
public:
    struct Config {
        size_t max_requests_per_minute = 100;
        size_t max_requests_per_hour = 1000;
        size_t burst_size = 20;
    };

    RateLimiter(Config config) : config_(config) {}

    /// @brief Check if request is allowed for client
    bool Allow(const std::string& client_id) {
        std::lock_guard<std::mutex> lock(mutex_);

        auto now = std::chrono::steady_clock::now();
        auto& bucket = buckets_[client_id];

        // Clean up old entries (every 100 calls)
        if (++cleanup_counter_ >= 100) {
            CleanupOldEntries(now);
            cleanup_counter_ = 0;
        }

        // Refill tokens based on time elapsed
        auto elapsed = std::chrono::duration_cast<std::chrono::milliseconds>(
            now - bucket.last_refill).count();

        double tokens_to_add = (elapsed / 60000.0) * config_.max_requests_per_minute;
        bucket.tokens = std::min(
            static_cast<double>(config_.burst_size),
            bucket.tokens + tokens_to_add);
        bucket.last_refill = now;

        // Check hourly limit
        if (now - bucket.hour_start > std::chrono::hours(1)) {
            bucket.hour_start = now;
            bucket.requests_this_hour = 0;
        }

        if (bucket.requests_this_hour >= config_.max_requests_per_hour) {
            PYFLARE_LOG_WARN("Rate limit exceeded for client {}: hourly limit", client_id);
            return false;
        }

        // Check burst limit
        if (bucket.tokens < 1.0) {
            PYFLARE_LOG_WARN("Rate limit exceeded for client {}: burst limit", client_id);
            return false;
        }

        bucket.tokens -= 1.0;
        bucket.requests_this_hour++;
        return true;
    }

private:
    struct TokenBucket {
        double tokens = 10.0;
        std::chrono::steady_clock::time_point last_refill = std::chrono::steady_clock::now();
        std::chrono::steady_clock::time_point hour_start = std::chrono::steady_clock::now();
        size_t requests_this_hour = 0;
    };

    void CleanupOldEntries(std::chrono::steady_clock::time_point now) {
        auto it = buckets_.begin();
        while (it != buckets_.end()) {
            if (now - it->second.last_refill > std::chrono::hours(24)) {
                it = buckets_.erase(it);
            } else {
                ++it;
            }
        }
    }

    Config config_;
    std::unordered_map<std::string, TokenBucket> buckets_;
    std::mutex mutex_;
    size_t cleanup_counter_ = 0;
};

// =============================================================================
// QueryAPI Implementation
// =============================================================================

class QueryAPI::Impl {
public:
    Impl(QueryAPIConfig config,
         std::shared_ptr<storage::ClickHouseClient> clickhouse,
         std::shared_ptr<storage::QdrantClient> qdrant)
        : config_(std::move(config)),
          clickhouse_(std::move(clickhouse)),
          qdrant_(std::move(qdrant)),
          rate_limiter_(RateLimiter::Config{
              config_.rate_limit_per_minute,
              config_.rate_limit_per_hour,
              config_.rate_limit_burst
          }) {}

    absl::Status Start() {
        PYFLARE_LOG_INFO("Starting Query API on {}:{}",
                        config_.listen_address, config_.port);
        running_ = true;
        // Placeholder - would start HTTP server
        return absl::OkStatus();
    }

    absl::Status Stop() {
        PYFLARE_LOG_INFO("Stopping Query API");
        running_ = false;
        return absl::OkStatus();
    }

    bool IsRunning() const { return running_; }

    /// @brief Authenticate request using API key
    AuthResult Authenticate(const std::string& auth_header) {
        AuthResult result;

        // SECURITY: Check if authentication is required
        if (!config_.require_auth) {
            result.authenticated = true;
            result.client_id = "anonymous";
            return result;
        }

        // Parse Authorization header
        if (auth_header.empty()) {
            result.error_message = "Missing Authorization header";
            return result;
        }

        // Expected format: "Bearer <api_key>" or "ApiKey <api_key>"
        std::string prefix;
        std::string api_key;

        size_t space_pos = auth_header.find(' ');
        if (space_pos == std::string::npos) {
            result.error_message = "Invalid Authorization header format";
            return result;
        }

        prefix = auth_header.substr(0, space_pos);
        api_key = auth_header.substr(space_pos + 1);

        // Validate prefix
        if (prefix != "Bearer" && prefix != "ApiKey") {
            result.error_message = "Invalid Authorization type";
            return result;
        }

        // SECURITY: Validate API key format
        if (!IsValidApiKeyFormat(api_key)) {
            result.error_message = "Invalid API key format";
            return result;
        }

        // SECURITY: Look up API key in configured keys
        auto it = config_.api_keys.find(api_key);
        if (it == config_.api_keys.end()) {
            // SECURITY: Use constant-time comparison for configured master key
            if (!config_.master_api_key.empty() &&
                SecureCompare(api_key, config_.master_api_key)) {
                result.authenticated = true;
                result.client_id = "master";
                result.permissions = {"read", "write", "admin"};
                return result;
            }
            result.error_message = "Invalid API key";
            return result;
        }

        result.authenticated = true;
        result.client_id = it->second.client_id;
        result.permissions = it->second.permissions;
        return result;
    }

    absl::StatusOr<QueryResponse> ExecuteQuery(const QueryRequest& request) {
        if (!running_) {
            return absl::FailedPreconditionError("Server not running");
        }

        // SECURITY: Authenticate request
        auto auth_result = Authenticate(request.auth_header);
        if (!auth_result.authenticated) {
            PYFLARE_LOG_WARN("Authentication failed: {}", auth_result.error_message);
            return absl::UnauthenticatedError(auth_result.error_message);
        }

        // SECURITY: Check rate limits
        if (!rate_limiter_.Allow(auth_result.client_id)) {
            return absl::ResourceExhaustedError("Rate limit exceeded");
        }

        // SECURITY: Validate query before execution
        auto validation_status = IsQuerySafe(request.sql);
        if (!validation_status.ok()) {
            PYFLARE_LOG_WARN("Query validation failed for client {}: {}",
                            auth_result.client_id, validation_status.message());
            return validation_status;
        }

        // SECURITY: Enforce query size limit
        if (request.sql.size() > config_.max_query_size) {
            return absl::InvalidArgumentError("Query exceeds maximum size limit");
        }

        auto start = std::chrono::steady_clock::now();

        // Execute using parameterized query if params provided
        absl::StatusOr<storage::QueryResult> result;

        if (!request.params.empty()) {
            // Convert params and use parameterized execution
            std::vector<storage::QueryParam> storage_params;
            storage_params.reserve(request.params.size());
            for (const auto& p : request.params) {
                storage_params.push_back({p.name, p.value, p.type});
            }
            result = clickhouse_->ExecuteWithParams(request.sql, storage_params);
        } else {
            result = clickhouse_->ExecuteWithParams(request.sql, {});
        }

        if (!result.ok()) {
            // SECURITY: Don't expose internal error details to client
            PYFLARE_LOG_ERROR("Query execution failed for client {}: {}",
                             auth_result.client_id, result.status().message());
            return absl::InternalError("Query execution failed");
        }

        auto end = std::chrono::steady_clock::now();

        QueryResponse response;
        response.columns = result->columns;
        response.rows = result->rows;
        response.total_rows = result->total_rows;
        response.execution_time = std::chrono::duration_cast<std::chrono::milliseconds>(
            end - start);

        // SECURITY: Enforce result row limit
        if (response.rows.size() > config_.max_result_rows) {
            response.rows.resize(config_.max_result_rows);
            PYFLARE_LOG_DEBUG("Query result truncated to {} rows",
                             config_.max_result_rows);
        }

        // Log successful query for audit trail
        PYFLARE_LOG_INFO("Query executed by client {}: {} rows in {}ms",
                        auth_result.client_id, response.rows.size(),
                        response.execution_time.count());

        return response;
    }

    /// @brief Get CORS headers for response
    std::unordered_map<std::string, std::string> GetCorsHeaders(
        const std::string& origin) const {
        std::unordered_map<std::string, std::string> headers;

        // SECURITY: Only allow configured origins
        if (config_.allowed_origins.empty()) {
            // No CORS if not configured
            return headers;
        }

        bool origin_allowed = false;
        for (const auto& allowed : config_.allowed_origins) {
            if (allowed == "*" || allowed == origin) {
                origin_allowed = true;
                break;
            }
        }

        if (origin_allowed) {
            headers["Access-Control-Allow-Origin"] = origin;
            headers["Access-Control-Allow-Methods"] = "GET, POST, OPTIONS";
            headers["Access-Control-Allow-Headers"] =
                "Content-Type, Authorization, X-Request-ID";
            headers["Access-Control-Max-Age"] = "86400";
            headers["Access-Control-Allow-Credentials"] = "true";
        }

        // SECURITY: Add security headers
        headers["X-Content-Type-Options"] = "nosniff";
        headers["X-Frame-Options"] = "DENY";
        headers["X-XSS-Protection"] = "1; mode=block";
        headers["Content-Security-Policy"] = "default-src 'none'";
        headers["Strict-Transport-Security"] = "max-age=31536000; includeSubDomains";

        return headers;
    }

    std::string GetAddress() const {
        return config_.listen_address + ":" + std::to_string(config_.port);
    }

private:
    QueryAPIConfig config_;
    std::shared_ptr<storage::ClickHouseClient> clickhouse_;
    std::shared_ptr<storage::QdrantClient> qdrant_;
    RateLimiter rate_limiter_;
    bool running_ = false;
};

QueryAPI::QueryAPI(
    QueryAPIConfig config,
    std::shared_ptr<storage::ClickHouseClient> clickhouse,
    std::shared_ptr<storage::QdrantClient> qdrant)
    : config_(std::move(config)),
      clickhouse_(std::move(clickhouse)),
      qdrant_(std::move(qdrant)),
      impl_(std::make_unique<Impl>(config_, clickhouse_, qdrant_)) {}

QueryAPI::~QueryAPI() = default;

absl::Status QueryAPI::Start() {
    return impl_->Start();
}

absl::Status QueryAPI::Stop() {
    return impl_->Stop();
}

bool QueryAPI::IsRunning() const {
    return impl_->IsRunning();
}

absl::StatusOr<QueryResponse> QueryAPI::ExecuteQuery(const QueryRequest& request) {
    return impl_->ExecuteQuery(request);
}

std::string QueryAPI::GetAddress() const {
    return impl_->GetAddress();
}

}  // namespace pyflare::query
