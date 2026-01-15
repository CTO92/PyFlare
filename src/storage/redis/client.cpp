/// @file client.cpp
/// @brief Redis client implementation for PyFlare

#include "storage/redis/client.h"

#include <cmath>
#include <sstream>

#include <spdlog/spdlog.h>

#ifdef PYFLARE_HAS_HIREDIS
#include <hiredis/hiredis.h>
#endif

namespace pyflare::storage {

// =============================================================================
// Key Prefix Constants
// =============================================================================

constexpr const char* kDriftRefPrefix = "pyflare:drift:ref:";
constexpr const char* kDriftCentroidPrefix = "pyflare:drift:centroid:";
constexpr const char* kCostPrefix = "pyflare:cost:";
constexpr const char* kBudgetPrefix = "pyflare:budget:";
constexpr const char* kRateLimitPrefix = "pyflare:ratelimit:";
constexpr const char* kEvalCachePrefix = "pyflare:eval:cache:";

// =============================================================================
// RedisClient Implementation
// =============================================================================

#ifdef PYFLARE_HAS_HIREDIS

class RedisClient::Impl {
public:
    explicit Impl(RedisConfig config) : config_(std::move(config)) {}

    ~Impl() {
        Disconnect();
    }

    absl::Status Connect() {
        if (connected_) {
            return absl::OkStatus();
        }

        struct timeval timeout;
        timeout.tv_sec = config_.connection_timeout.count();
        timeout.tv_usec = 0;

        context_ = redisConnectWithTimeout(config_.host.c_str(),
                                           config_.port, timeout);

        if (context_ == nullptr || context_->err) {
            std::string error_msg = context_ ? context_->errstr : "Unknown error";
            if (context_) {
                redisFree(context_);
                context_ = nullptr;
            }
            return absl::UnavailableError("Failed to connect to Redis: " + error_msg);
        }

        // Authenticate if password is set
        if (!config_.password.empty()) {
            redisReply* reply = static_cast<redisReply*>(
                redisCommand(context_, "AUTH %s", config_.password.c_str()));
            if (reply == nullptr || reply->type == REDIS_REPLY_ERROR) {
                std::string error_msg = reply ? reply->str : "Unknown error";
                if (reply) freeReplyObject(reply);
                redisFree(context_);
                context_ = nullptr;
                return absl::PermissionDeniedError("Redis authentication failed: " + error_msg);
            }
            freeReplyObject(reply);
        }

        // Select database
        if (config_.database != 0) {
            redisReply* reply = static_cast<redisReply*>(
                redisCommand(context_, "SELECT %d", config_.database));
            if (reply == nullptr || reply->type == REDIS_REPLY_ERROR) {
                std::string error_msg = reply ? reply->str : "Unknown error";
                if (reply) freeReplyObject(reply);
                redisFree(context_);
                context_ = nullptr;
                return absl::InternalError("Failed to select database: " + error_msg);
            }
            freeReplyObject(reply);
        }

        connected_ = true;
        spdlog::info("Connected to Redis at {}:{}/{}", config_.host,
                     config_.port, config_.database);
        return absl::OkStatus();
    }

    absl::Status Disconnect() {
        if (!connected_ || context_ == nullptr) {
            return absl::OkStatus();
        }

        redisFree(context_);
        context_ = nullptr;
        connected_ = false;
        spdlog::info("Disconnected from Redis");
        return absl::OkStatus();
    }

    bool IsConnected() const {
        return connected_ && context_ != nullptr;
    }

    absl::Status Ping() {
        if (!IsConnected()) {
            return absl::FailedPreconditionError("Not connected to Redis");
        }

        redisReply* reply = static_cast<redisReply*>(redisCommand(context_, "PING"));
        if (reply == nullptr || reply->type == REDIS_REPLY_ERROR) {
            std::string error_msg = reply ? reply->str : "Connection error";
            if (reply) freeReplyObject(reply);
            return absl::InternalError("Ping failed: " + error_msg);
        }
        freeReplyObject(reply);
        return absl::OkStatus();
    }

    absl::Status Set(const std::string& key, const std::string& value,
                     std::optional<std::chrono::seconds> ttl) {
        if (!IsConnected()) {
            return absl::FailedPreconditionError("Not connected to Redis");
        }

        redisReply* reply;
        if (ttl.has_value() && ttl->count() > 0) {
            reply = static_cast<redisReply*>(
                redisCommand(context_, "SET %s %b EX %lld",
                             key.c_str(), value.data(), value.size(),
                             static_cast<long long>(ttl->count())));
        } else if (config_.default_ttl.count() > 0) {
            reply = static_cast<redisReply*>(
                redisCommand(context_, "SET %s %b EX %lld",
                             key.c_str(), value.data(), value.size(),
                             static_cast<long long>(config_.default_ttl.count())));
        } else {
            reply = static_cast<redisReply*>(
                redisCommand(context_, "SET %s %b",
                             key.c_str(), value.data(), value.size()));
        }

        if (reply == nullptr || reply->type == REDIS_REPLY_ERROR) {
            std::string error_msg = reply ? reply->str : "Connection error";
            if (reply) freeReplyObject(reply);
            return absl::InternalError("SET failed: " + error_msg);
        }
        freeReplyObject(reply);
        return absl::OkStatus();
    }

    absl::StatusOr<std::optional<std::string>> Get(const std::string& key) {
        if (!IsConnected()) {
            return absl::FailedPreconditionError("Not connected to Redis");
        }

        redisReply* reply = static_cast<redisReply*>(
            redisCommand(context_, "GET %s", key.c_str()));

        if (reply == nullptr) {
            return absl::InternalError("GET failed: Connection error");
        }

        if (reply->type == REDIS_REPLY_ERROR) {
            std::string error_msg = reply->str;
            freeReplyObject(reply);
            return absl::InternalError("GET failed: " + error_msg);
        }

        if (reply->type == REDIS_REPLY_NIL) {
            freeReplyObject(reply);
            return std::nullopt;
        }

        std::string value(reply->str, reply->len);
        freeReplyObject(reply);
        return value;
    }

    absl::StatusOr<bool> Delete(const std::string& key) {
        if (!IsConnected()) {
            return absl::FailedPreconditionError("Not connected to Redis");
        }

        redisReply* reply = static_cast<redisReply*>(
            redisCommand(context_, "DEL %s", key.c_str()));

        if (reply == nullptr || reply->type == REDIS_REPLY_ERROR) {
            std::string error_msg = reply ? reply->str : "Connection error";
            if (reply) freeReplyObject(reply);
            return absl::InternalError("DEL failed: " + error_msg);
        }

        bool deleted = reply->integer > 0;
        freeReplyObject(reply);
        return deleted;
    }

    absl::StatusOr<bool> Exists(const std::string& key) {
        if (!IsConnected()) {
            return absl::FailedPreconditionError("Not connected to Redis");
        }

        redisReply* reply = static_cast<redisReply*>(
            redisCommand(context_, "EXISTS %s", key.c_str()));

        if (reply == nullptr || reply->type == REDIS_REPLY_ERROR) {
            std::string error_msg = reply ? reply->str : "Connection error";
            if (reply) freeReplyObject(reply);
            return absl::InternalError("EXISTS failed: " + error_msg);
        }

        bool exists = reply->integer > 0;
        freeReplyObject(reply);
        return exists;
    }

    absl::StatusOr<int64_t> Incr(const std::string& key, int64_t delta) {
        if (!IsConnected()) {
            return absl::FailedPreconditionError("Not connected to Redis");
        }

        redisReply* reply = static_cast<redisReply*>(
            redisCommand(context_, "INCRBY %s %lld", key.c_str(),
                         static_cast<long long>(delta)));

        if (reply == nullptr || reply->type == REDIS_REPLY_ERROR) {
            std::string error_msg = reply ? reply->str : "Connection error";
            if (reply) freeReplyObject(reply);
            return absl::InternalError("INCRBY failed: " + error_msg);
        }

        int64_t value = reply->integer;
        freeReplyObject(reply);
        return value;
    }

    absl::StatusOr<double> IncrFloat(const std::string& key, double delta) {
        if (!IsConnected()) {
            return absl::FailedPreconditionError("Not connected to Redis");
        }

        redisReply* reply = static_cast<redisReply*>(
            redisCommand(context_, "INCRBYFLOAT %s %f", key.c_str(), delta));

        if (reply == nullptr || reply->type == REDIS_REPLY_ERROR) {
            std::string error_msg = reply ? reply->str : "Connection error";
            if (reply) freeReplyObject(reply);
            return absl::InternalError("INCRBYFLOAT failed: " + error_msg);
        }

        double value = std::stod(reply->str);
        freeReplyObject(reply);
        return value;
    }

    absl::Status HSet(const std::string& key, const std::string& field,
                      const std::string& value) {
        if (!IsConnected()) {
            return absl::FailedPreconditionError("Not connected to Redis");
        }

        redisReply* reply = static_cast<redisReply*>(
            redisCommand(context_, "HSET %s %s %b",
                         key.c_str(), field.c_str(), value.data(), value.size()));

        if (reply == nullptr || reply->type == REDIS_REPLY_ERROR) {
            std::string error_msg = reply ? reply->str : "Connection error";
            if (reply) freeReplyObject(reply);
            return absl::InternalError("HSET failed: " + error_msg);
        }

        freeReplyObject(reply);
        return absl::OkStatus();
    }

    absl::StatusOr<std::optional<std::string>> HGet(const std::string& key,
                                                     const std::string& field) {
        if (!IsConnected()) {
            return absl::FailedPreconditionError("Not connected to Redis");
        }

        redisReply* reply = static_cast<redisReply*>(
            redisCommand(context_, "HGET %s %s", key.c_str(), field.c_str()));

        if (reply == nullptr) {
            return absl::InternalError("HGET failed: Connection error");
        }

        if (reply->type == REDIS_REPLY_ERROR) {
            std::string error_msg = reply->str;
            freeReplyObject(reply);
            return absl::InternalError("HGET failed: " + error_msg);
        }

        if (reply->type == REDIS_REPLY_NIL) {
            freeReplyObject(reply);
            return std::nullopt;
        }

        std::string value(reply->str, reply->len);
        freeReplyObject(reply);
        return value;
    }

    absl::StatusOr<std::unordered_map<std::string, std::string>> HGetAll(
        const std::string& key) {
        if (!IsConnected()) {
            return absl::FailedPreconditionError("Not connected to Redis");
        }

        redisReply* reply = static_cast<redisReply*>(
            redisCommand(context_, "HGETALL %s", key.c_str()));

        if (reply == nullptr || reply->type == REDIS_REPLY_ERROR) {
            std::string error_msg = reply ? reply->str : "Connection error";
            if (reply) freeReplyObject(reply);
            return absl::InternalError("HGETALL failed: " + error_msg);
        }

        std::unordered_map<std::string, std::string> result;
        if (reply->type == REDIS_REPLY_ARRAY) {
            for (size_t i = 0; i + 1 < reply->elements; i += 2) {
                std::string field(reply->element[i]->str, reply->element[i]->len);
                std::string value(reply->element[i + 1]->str,
                                  reply->element[i + 1]->len);
                result[field] = value;
            }
        }

        freeReplyObject(reply);
        return result;
    }

    absl::StatusOr<int64_t> SAdd(const std::string& key,
                                  const std::vector<std::string>& members) {
        if (!IsConnected()) {
            return absl::FailedPreconditionError("Not connected to Redis");
        }

        if (members.empty()) {
            return 0;
        }

        std::vector<const char*> argv;
        std::vector<size_t> argvlen;
        std::string cmd = "SADD";

        argv.push_back(cmd.c_str());
        argvlen.push_back(cmd.size());
        argv.push_back(key.c_str());
        argvlen.push_back(key.size());

        for (const auto& member : members) {
            argv.push_back(member.c_str());
            argvlen.push_back(member.size());
        }

        redisReply* reply = static_cast<redisReply*>(
            redisCommandArgv(context_, static_cast<int>(argv.size()),
                             argv.data(), argvlen.data()));

        if (reply == nullptr || reply->type == REDIS_REPLY_ERROR) {
            std::string error_msg = reply ? reply->str : "Connection error";
            if (reply) freeReplyObject(reply);
            return absl::InternalError("SADD failed: " + error_msg);
        }

        int64_t added = reply->integer;
        freeReplyObject(reply);
        return added;
    }

    absl::StatusOr<std::vector<std::string>> SMembers(const std::string& key) {
        if (!IsConnected()) {
            return absl::FailedPreconditionError("Not connected to Redis");
        }

        redisReply* reply = static_cast<redisReply*>(
            redisCommand(context_, "SMEMBERS %s", key.c_str()));

        if (reply == nullptr || reply->type == REDIS_REPLY_ERROR) {
            std::string error_msg = reply ? reply->str : "Connection error";
            if (reply) freeReplyObject(reply);
            return absl::InternalError("SMEMBERS failed: " + error_msg);
        }

        std::vector<std::string> result;
        if (reply->type == REDIS_REPLY_ARRAY) {
            result.reserve(reply->elements);
            for (size_t i = 0; i < reply->elements; ++i) {
                result.emplace_back(reply->element[i]->str, reply->element[i]->len);
            }
        }

        freeReplyObject(reply);
        return result;
    }

private:
    RedisConfig config_;
    redisContext* context_ = nullptr;
    bool connected_ = false;
};

#else  // !PYFLARE_HAS_HIREDIS

/// @brief Stub implementation when hiredis is not available
class RedisClient::Impl {
public:
    explicit Impl(RedisConfig config) : config_(std::move(config)) {}

    absl::Status Connect() {
        spdlog::warn("Redis support not compiled in (hiredis not found)");
        connected_ = true;  // Pretend to be connected for testing
        return absl::OkStatus();
    }

    absl::Status Disconnect() {
        connected_ = false;
        return absl::OkStatus();
    }

    bool IsConnected() const { return connected_; }

    absl::Status Ping() { return absl::OkStatus(); }

    absl::Status Set(const std::string& key, const std::string& value,
                     std::optional<std::chrono::seconds>) {
        cache_[key] = value;
        return absl::OkStatus();
    }

    absl::StatusOr<std::optional<std::string>> Get(const std::string& key) {
        auto it = cache_.find(key);
        if (it != cache_.end()) {
            return it->second;
        }
        return std::nullopt;
    }

    absl::StatusOr<bool> Delete(const std::string& key) {
        return cache_.erase(key) > 0;
    }

    absl::StatusOr<bool> Exists(const std::string& key) {
        return cache_.find(key) != cache_.end();
    }

    absl::StatusOr<int64_t> Incr(const std::string& key, int64_t delta) {
        auto it = cache_.find(key);
        int64_t value = delta;
        if (it != cache_.end()) {
            value = std::stoll(it->second) + delta;
        }
        cache_[key] = std::to_string(value);
        return value;
    }

    absl::StatusOr<double> IncrFloat(const std::string& key, double delta) {
        auto it = cache_.find(key);
        double value = delta;
        if (it != cache_.end()) {
            value = std::stod(it->second) + delta;
        }
        cache_[key] = std::to_string(value);
        return value;
    }

    absl::Status HSet(const std::string& key, const std::string& field,
                      const std::string& value) {
        hash_cache_[key][field] = value;
        return absl::OkStatus();
    }

    absl::StatusOr<std::optional<std::string>> HGet(const std::string& key,
                                                     const std::string& field) {
        auto kit = hash_cache_.find(key);
        if (kit == hash_cache_.end()) return std::nullopt;
        auto fit = kit->second.find(field);
        if (fit == kit->second.end()) return std::nullopt;
        return fit->second;
    }

    absl::StatusOr<std::unordered_map<std::string, std::string>> HGetAll(
        const std::string& key) {
        auto it = hash_cache_.find(key);
        if (it != hash_cache_.end()) {
            return it->second;
        }
        return std::unordered_map<std::string, std::string>{};
    }

    absl::StatusOr<int64_t> SAdd(const std::string&,
                                  const std::vector<std::string>& members) {
        return static_cast<int64_t>(members.size());
    }

    absl::StatusOr<std::vector<std::string>> SMembers(const std::string&) {
        return std::vector<std::string>{};
    }

private:
    RedisConfig config_;
    bool connected_ = false;
    std::unordered_map<std::string, std::string> cache_;
    std::unordered_map<std::string, std::unordered_map<std::string, std::string>> hash_cache_;
};

#endif  // PYFLARE_HAS_HIREDIS

// =============================================================================
// RedisClient Public Interface
// =============================================================================

RedisClient::RedisClient(RedisConfig config)
    : config_(std::move(config)), impl_(std::make_unique<Impl>(config_)) {}

RedisClient::~RedisClient() = default;

RedisClient::RedisClient(RedisClient&&) noexcept = default;
RedisClient& RedisClient::operator=(RedisClient&&) noexcept = default;

absl::Status RedisClient::Connect() {
    return impl_->Connect();
}

absl::Status RedisClient::Disconnect() {
    return impl_->Disconnect();
}

bool RedisClient::IsConnected() const {
    return impl_->IsConnected();
}

absl::Status RedisClient::Ping() {
    return impl_->Ping();
}

absl::Status RedisClient::Set(const std::string& key, const std::string& value,
                               std::optional<std::chrono::seconds> ttl) {
    return impl_->Set(key, value, ttl);
}

absl::StatusOr<std::optional<std::string>> RedisClient::Get(const std::string& key) {
    return impl_->Get(key);
}

absl::StatusOr<bool> RedisClient::Delete(const std::string& key) {
    return impl_->Delete(key);
}

absl::StatusOr<bool> RedisClient::Exists(const std::string& key) {
    return impl_->Exists(key);
}

absl::Status RedisClient::Expire(const std::string& key, std::chrono::seconds ttl) {
    // Would need to implement EXPIRE command
    return absl::OkStatus();
}

absl::StatusOr<int64_t> RedisClient::TTL(const std::string& key) {
    // Would need to implement TTL command
    return -1;
}

absl::StatusOr<int64_t> RedisClient::Incr(const std::string& key, int64_t delta) {
    return impl_->Incr(key, delta);
}

absl::StatusOr<double> RedisClient::IncrFloat(const std::string& key, double delta) {
    return impl_->IncrFloat(key, delta);
}

absl::Status RedisClient::HSet(const std::string& key, const std::string& field,
                                const std::string& value) {
    return impl_->HSet(key, field, value);
}

absl::StatusOr<std::optional<std::string>> RedisClient::HGet(
    const std::string& key, const std::string& field) {
    return impl_->HGet(key, field);
}

absl::StatusOr<std::unordered_map<std::string, std::string>> RedisClient::HGetAll(
    const std::string& key) {
    return impl_->HGetAll(key);
}

absl::StatusOr<int64_t> RedisClient::HDel(const std::string& key,
                                           const std::vector<std::string>& fields) {
    // Would need to implement HDEL command
    return 0;
}

absl::StatusOr<int64_t> RedisClient::HIncr(const std::string& key,
                                            const std::string& field,
                                            int64_t delta) {
    // Would need to implement HINCRBY command
    return 0;
}

absl::StatusOr<int64_t> RedisClient::SAdd(const std::string& key,
                                           const std::vector<std::string>& members) {
    return impl_->SAdd(key, members);
}

absl::StatusOr<int64_t> RedisClient::SRem(const std::string& key,
                                           const std::vector<std::string>& members) {
    // Would need to implement SREM command
    return 0;
}

absl::StatusOr<std::vector<std::string>> RedisClient::SMembers(const std::string& key) {
    return impl_->SMembers(key);
}

absl::StatusOr<bool> RedisClient::SIsMember(const std::string& key,
                                             const std::string& member) {
    // Would need to implement SISMEMBER command
    return false;
}

absl::StatusOr<int64_t> RedisClient::LPush(const std::string& key,
                                            const std::vector<std::string>& values) {
    // Would need to implement LPUSH command
    return 0;
}

absl::StatusOr<int64_t> RedisClient::RPush(const std::string& key,
                                            const std::vector<std::string>& values) {
    // Would need to implement RPUSH command
    return 0;
}

absl::StatusOr<std::optional<std::string>> RedisClient::LPop(const std::string& key) {
    // Would need to implement LPOP command
    return std::nullopt;
}

absl::StatusOr<std::vector<std::string>> RedisClient::LRange(
    const std::string& key, int64_t start, int64_t stop) {
    // Would need to implement LRANGE command
    return std::vector<std::string>{};
}

// =============================================================================
// SECURITY: Key Component Validation
// =============================================================================

/// @brief Maximum key component length
static constexpr size_t kMaxKeyComponentLength = 256;

/// @brief SECURITY: Validate a key component to prevent key injection
/// Disallows colons, newlines, and other dangerous characters
static bool IsValidKeyComponent(const std::string& component) {
    if (component.empty() || component.size() > kMaxKeyComponentLength) {
        return false;
    }
    for (unsigned char c : component) {
        // Disallow key separator characters and control characters
        if (c == ':' || c == '\n' || c == '\r' || c == '\0' ||
            c == ' ' || c == '\t' || c < 32) {
            return false;
        }
    }
    return true;
}

/// @brief SECURITY: Build a safe Redis key with validated components
static absl::StatusOr<std::string> BuildSafeKey(
    const std::string& prefix,
    const std::vector<std::string>& components) {

    std::string key = prefix;
    for (size_t i = 0; i < components.size(); ++i) {
        const auto& component = components[i];
        if (!IsValidKeyComponent(component)) {
            spdlog::warn("SECURITY: Invalid key component rejected: length={}, value={}",
                         component.size(),
                         component.size() > 50 ? component.substr(0, 50) + "..." : component);
            return absl::InvalidArgumentError("Invalid key component");
        }
        key += component;
        if (i < components.size() - 1) {
            key += ":";
        }
    }
    return key;
}

// =============================================================================
// PyFlare-Specific Operations
// =============================================================================

absl::Status RedisClient::StoreDriftReference(
    const std::string& model_id,
    const std::string& feature_name,
    const std::string& distribution,
    std::chrono::seconds ttl) {
    // SECURITY: Validate key components to prevent key injection
    auto key_result = BuildSafeKey(kDriftRefPrefix, {model_id, feature_name});
    if (!key_result.ok()) {
        return key_result.status();
    }
    return Set(*key_result, distribution, ttl);
}

absl::StatusOr<std::optional<std::string>> RedisClient::GetDriftReference(
    const std::string& model_id,
    const std::string& feature_name) {
    // SECURITY: Validate key components
    auto key_result = BuildSafeKey(kDriftRefPrefix, {model_id, feature_name});
    if (!key_result.ok()) {
        return key_result.status();
    }
    return Get(*key_result);
}

absl::Status RedisClient::StoreEmbeddingCentroid(
    const std::string& model_id,
    const std::vector<float>& centroid) {
    // SECURITY: Validate key components
    auto key_result = BuildSafeKey(kDriftCentroidPrefix, {model_id});
    if (!key_result.ok()) {
        return key_result.status();
    }
    // Serialize centroid as binary
    std::string value(reinterpret_cast<const char*>(centroid.data()),
                      centroid.size() * sizeof(float));
    return Set(*key_result, value, std::nullopt);
}

absl::StatusOr<std::optional<std::vector<float>>> RedisClient::GetEmbeddingCentroid(
    const std::string& model_id) {
    // SECURITY: Validate key components
    auto key_result = BuildSafeKey(kDriftCentroidPrefix, {model_id});
    if (!key_result.ok()) {
        return key_result.status();
    }
    auto result = Get(*key_result);
    if (!result.ok()) {
        return result.status();
    }
    if (!result->has_value()) {
        return std::nullopt;
    }

    const std::string& data = result->value();
    size_t count = data.size() / sizeof(float);
    std::vector<float> centroid(count);
    std::memcpy(centroid.data(), data.data(), data.size());
    return centroid;
}

absl::StatusOr<int64_t> RedisClient::TrackCost(
    const std::string& dimension,
    const std::string& dimension_value,
    int64_t cost_micros,
    const std::string& date) {
    // SECURITY: Validate key components
    auto key_result = BuildSafeKey(kCostPrefix, {dimension, dimension_value, date});
    if (!key_result.ok()) {
        return key_result.status();
    }
    return Incr(*key_result, cost_micros);
}

absl::StatusOr<int64_t> RedisClient::GetCost(
    const std::string& dimension,
    const std::string& dimension_value,
    const std::string& date) {
    // SECURITY: Validate key components
    auto key_result = BuildSafeKey(kCostPrefix, {dimension, dimension_value, date});
    if (!key_result.ok()) {
        return key_result.status();
    }
    auto result = Get(*key_result);
    if (!result.ok()) {
        return result.status();
    }
    if (!result->has_value()) {
        return 0;
    }
    return std::stoll(result->value());
}

absl::StatusOr<bool> RedisClient::CheckRateLimit(
    const std::string& key,
    double tokens_per_second,
    double burst_size) {
    // Simple token bucket implementation using Redis
    std::string token_key = std::string(kRateLimitPrefix) + key + ":tokens";
    std::string time_key = std::string(kRateLimitPrefix) + key + ":last_refill";

    auto now = std::chrono::duration_cast<std::chrono::milliseconds>(
        std::chrono::system_clock::now().time_since_epoch()).count();

    // Get current tokens and last refill time
    auto tokens_result = Get(token_key);
    auto time_result = Get(time_key);

    if (!tokens_result.ok() || !time_result.ok()) {
        return true;  // Allow on error
    }

    double tokens = burst_size;
    int64_t last_refill = now;

    if (tokens_result->has_value()) {
        tokens = std::stod(tokens_result->value());
    }
    if (time_result->has_value()) {
        last_refill = std::stoll(time_result->value());
    }

    // Refill tokens
    double elapsed_seconds = (now - last_refill) / 1000.0;
    tokens = std::min(burst_size, tokens + elapsed_seconds * tokens_per_second);

    // Try to consume a token
    if (tokens >= 1.0) {
        tokens -= 1.0;
        Set(token_key, std::to_string(tokens), std::chrono::seconds(60));
        Set(time_key, std::to_string(now), std::chrono::seconds(60));
        return true;
    }

    return false;
}

absl::Status RedisClient::CacheEvaluationResult(
    const std::string& input_hash,
    const std::string& result,
    std::chrono::seconds ttl) {
    std::string key = std::string(kEvalCachePrefix) + input_hash;
    return Set(key, result, ttl);
}

absl::StatusOr<std::optional<std::string>> RedisClient::GetCachedEvaluationResult(
    const std::string& input_hash) {
    std::string key = std::string(kEvalCachePrefix) + input_hash;
    return Get(key);
}

}  // namespace pyflare::storage
