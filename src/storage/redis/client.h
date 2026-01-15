#pragma once

/// @file client.h
/// @brief Redis client wrapper for PyFlare caching

#include <chrono>
#include <memory>
#include <optional>
#include <string>
#include <unordered_map>
#include <vector>

#include <absl/status/status.h>
#include <absl/status/statusor.h>

namespace pyflare::storage {

/// @brief Redis client configuration
struct RedisConfig {
    std::string host = "localhost";
    uint16_t port = 6379;
    std::string password;
    int database = 0;

    std::chrono::seconds connection_timeout{5};
    std::chrono::seconds socket_timeout{5};

    bool use_ssl = false;
    std::string ssl_ca_cert_path;

    /// Default TTL for keys (0 = no expiry)
    std::chrono::seconds default_ttl{0};

    /// Connection pool size
    size_t pool_size = 10;
};

/// @brief Redis client for PyFlare caching
///
/// Provides caching functionality for:
/// - Drift reference distributions
/// - Cost aggregations
/// - Budget tracking
/// - Rate limiting
/// - Evaluation result caching
class RedisClient {
public:
    /// @brief Create a client with the given configuration
    explicit RedisClient(RedisConfig config);

    /// @brief Destructor
    ~RedisClient();

    // Non-copyable
    RedisClient(const RedisClient&) = delete;
    RedisClient& operator=(const RedisClient&) = delete;

    // Movable
    RedisClient(RedisClient&&) noexcept;
    RedisClient& operator=(RedisClient&&) noexcept;

    /// @brief Connect to Redis
    absl::Status Connect();

    /// @brief Disconnect from Redis
    absl::Status Disconnect();

    /// @brief Check if connected
    bool IsConnected() const;

    /// @brief Ping Redis server
    absl::Status Ping();

    // ==========================================================================
    // String Operations
    // ==========================================================================

    /// @brief Set a string value
    /// @param key Key name
    /// @param value Value to set
    /// @param ttl Optional TTL (0 = use default, nullopt = no expiry)
    absl::Status Set(const std::string& key, const std::string& value,
                     std::optional<std::chrono::seconds> ttl = std::nullopt);

    /// @brief Get a string value
    /// @param key Key name
    /// @return Value if exists, nullopt if not found
    absl::StatusOr<std::optional<std::string>> Get(const std::string& key);

    /// @brief Delete a key
    /// @param key Key name
    /// @return true if key was deleted, false if key didn't exist
    absl::StatusOr<bool> Delete(const std::string& key);

    /// @brief Check if a key exists
    /// @param key Key name
    absl::StatusOr<bool> Exists(const std::string& key);

    /// @brief Set key expiration
    /// @param key Key name
    /// @param ttl Time to live
    absl::Status Expire(const std::string& key, std::chrono::seconds ttl);

    /// @brief Get remaining TTL for a key
    /// @param key Key name
    /// @return TTL in seconds, -1 if no expiry, -2 if key doesn't exist
    absl::StatusOr<int64_t> TTL(const std::string& key);

    // ==========================================================================
    // Numeric Operations
    // ==========================================================================

    /// @brief Increment a value
    /// @param key Key name
    /// @param delta Amount to increment (default 1)
    /// @return New value
    absl::StatusOr<int64_t> Incr(const std::string& key, int64_t delta = 1);

    /// @brief Increment a float value
    /// @param key Key name
    /// @param delta Amount to increment
    /// @return New value
    absl::StatusOr<double> IncrFloat(const std::string& key, double delta);

    // ==========================================================================
    // Hash Operations
    // ==========================================================================

    /// @brief Set a hash field
    absl::Status HSet(const std::string& key, const std::string& field,
                      const std::string& value);

    /// @brief Get a hash field
    absl::StatusOr<std::optional<std::string>> HGet(
        const std::string& key, const std::string& field);

    /// @brief Get all hash fields
    absl::StatusOr<std::unordered_map<std::string, std::string>> HGetAll(
        const std::string& key);

    /// @brief Delete hash fields
    absl::StatusOr<int64_t> HDel(const std::string& key,
                                  const std::vector<std::string>& fields);

    /// @brief Increment a hash field
    absl::StatusOr<int64_t> HIncr(const std::string& key,
                                   const std::string& field,
                                   int64_t delta = 1);

    // ==========================================================================
    // Set Operations
    // ==========================================================================

    /// @brief Add members to a set
    absl::StatusOr<int64_t> SAdd(const std::string& key,
                                  const std::vector<std::string>& members);

    /// @brief Remove members from a set
    absl::StatusOr<int64_t> SRem(const std::string& key,
                                  const std::vector<std::string>& members);

    /// @brief Get all set members
    absl::StatusOr<std::vector<std::string>> SMembers(const std::string& key);

    /// @brief Check if member exists in set
    absl::StatusOr<bool> SIsMember(const std::string& key,
                                    const std::string& member);

    // ==========================================================================
    // List Operations
    // ==========================================================================

    /// @brief Push values to the left of a list
    absl::StatusOr<int64_t> LPush(const std::string& key,
                                   const std::vector<std::string>& values);

    /// @brief Push values to the right of a list
    absl::StatusOr<int64_t> RPush(const std::string& key,
                                   const std::vector<std::string>& values);

    /// @brief Pop value from the left of a list
    absl::StatusOr<std::optional<std::string>> LPop(const std::string& key);

    /// @brief Get list range
    absl::StatusOr<std::vector<std::string>> LRange(
        const std::string& key, int64_t start, int64_t stop);

    // ==========================================================================
    // PyFlare-Specific Operations
    // ==========================================================================

    /// @brief Store a reference distribution for drift detection
    /// @param model_id Model identifier
    /// @param feature_name Feature name
    /// @param distribution JSON-serialized distribution
    /// @param ttl TTL for the reference (default 24 hours)
    absl::Status StoreDriftReference(
        const std::string& model_id,
        const std::string& feature_name,
        const std::string& distribution,
        std::chrono::seconds ttl = std::chrono::hours(24));

    /// @brief Get a reference distribution for drift detection
    absl::StatusOr<std::optional<std::string>> GetDriftReference(
        const std::string& model_id,
        const std::string& feature_name);

    /// @brief Store an embedding centroid for drift detection
    absl::Status StoreEmbeddingCentroid(
        const std::string& model_id,
        const std::vector<float>& centroid);

    /// @brief Get an embedding centroid for drift detection
    absl::StatusOr<std::optional<std::vector<float>>> GetEmbeddingCentroid(
        const std::string& model_id);

    /// @brief Track cost for a dimension
    /// @param dimension Dimension type (model, user, feature, etc.)
    /// @param dimension_value Dimension value
    /// @param cost_micros Cost to add (in micro-dollars)
    /// @param date Date string (YYYY-MM-DD)
    absl::StatusOr<int64_t> TrackCost(
        const std::string& dimension,
        const std::string& dimension_value,
        int64_t cost_micros,
        const std::string& date);

    /// @brief Get current cost for a dimension
    absl::StatusOr<int64_t> GetCost(
        const std::string& dimension,
        const std::string& dimension_value,
        const std::string& date);

    /// @brief Check and update rate limit
    /// @param key Rate limit key (e.g., IP address)
    /// @param tokens_per_second Token refill rate
    /// @param burst_size Maximum bucket size
    /// @return true if request allowed, false if rate limited
    absl::StatusOr<bool> CheckRateLimit(
        const std::string& key,
        double tokens_per_second,
        double burst_size);

    /// @brief Cache an evaluation result
    /// @param input_hash Hash of the input
    /// @param result JSON-serialized evaluation result
    /// @param ttl TTL for the cache entry
    absl::Status CacheEvaluationResult(
        const std::string& input_hash,
        const std::string& result,
        std::chrono::seconds ttl = std::chrono::hours(24));

    /// @brief Get a cached evaluation result
    absl::StatusOr<std::optional<std::string>> GetCachedEvaluationResult(
        const std::string& input_hash);

    /// @brief Get the configuration
    const RedisConfig& GetConfig() const { return config_; }

private:
    RedisConfig config_;
    class Impl;
    std::unique_ptr<Impl> impl_;
};

}  // namespace pyflare::storage
