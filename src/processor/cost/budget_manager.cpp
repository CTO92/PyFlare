/// @file budget_manager.cpp
/// @brief Budget management implementation
///
/// SECURITY: This implementation uses atomic Redis operations to prevent
/// race conditions in budget checking and spending.

#include "processor/cost/budget_manager.h"

#include <algorithm>
#include <ctime>
#include <regex>

#include <nlohmann/json.hpp>
#include <spdlog/spdlog.h>

namespace pyflare::cost {

using json = nlohmann::json;
using namespace std::chrono;

// =============================================================================
// SECURITY: Input Validation
// =============================================================================

/// @brief Maximum dimension value length
static constexpr size_t kMaxDimensionValueLength = 256;

/// @brief Validate dimension_value format to prevent key injection
static bool IsValidDimensionValue(const std::string& value) {
    if (value.empty() || value.size() > kMaxDimensionValueLength) {
        return false;
    }
    // Only allow alphanumeric, hyphens, underscores, dots
    // Explicitly disallow colons to prevent key injection
    static const std::regex dim_value_regex("^[a-zA-Z0-9_\\-\\.]+$");
    return std::regex_match(value, dim_value_regex);
}

/// @brief Validate spend amount to prevent overflow
static bool IsValidSpendAmount(int64_t amount) {
    // Prevent negative amounts and overflow (max ~10 billion USD in micros)
    return amount >= 0 && amount <= 10000000000000000LL;
}

// =============================================================================
// BudgetManager Implementation
// =============================================================================

BudgetManager::BudgetManager(std::shared_ptr<storage::RedisClient> redis,
                             BudgetManagerConfig config)
    : redis_(std::move(redis)),
      config_(std::move(config)) {}

BudgetManager::~BudgetManager() {
    if (initialized_) {
        auto status = Shutdown();
        if (!status.ok()) {
            spdlog::error("BudgetManager shutdown failed: {}", status.message());
        }
    }
}

absl::Status BudgetManager::Initialize() {
    if (initialized_) {
        return absl::OkStatus();
    }

    // Verify Redis connection
    auto ping = redis_->Ping();
    if (!ping.ok()) {
        spdlog::warn("Redis not available for BudgetManager, using local-only mode");
    }

    initialized_ = true;
    spdlog::info("BudgetManager initialized");
    return absl::OkStatus();
}

absl::Status BudgetManager::Shutdown() {
    if (!initialized_) {
        return absl::OkStatus();
    }

    // Flush any pending counters
    auto status = FlushCounters();
    if (!status.ok()) {
        spdlog::warn("Failed to flush counters on shutdown: {}", status.message());
    }

    initialized_ = false;
    return absl::OkStatus();
}

// =============================================================================
// Budget Configuration
// =============================================================================

absl::Status BudgetManager::CreateBudget(const BudgetConfig& config) {
    json j;
    j["id"] = config.id;
    j["dimension"] = DimensionToString(config.dimension);
    j["dimension_value"] = config.dimension_value;
    j["period"] = PeriodToString(config.period);
    j["soft_limit_micros"] = config.soft_limit_micros;
    j["hard_limit_micros"] = config.hard_limit_micros;
    j["warning_percentage"] = config.warning_percentage;
    j["block_on_exceeded"] = config.block_on_exceeded;
    j["created_at"] = duration_cast<seconds>(
        config.created_at.time_since_epoch()).count();
    j["updated_at"] = duration_cast<seconds>(
        system_clock::now().time_since_epoch()).count();

    auto key = BuildConfigKey(config.dimension, config.dimension_value);
    return redis_->Set(key, j.dump(), hours(24 * 365));  // 1 year TTL
}

absl::StatusOr<BudgetConfig> BudgetManager::GetBudget(
    BudgetDimension dimension,
    const std::string& dimension_value) {

    auto key = BuildConfigKey(dimension, dimension_value);
    auto result = redis_->Get(key);
    if (!result.ok()) {
        return result.status();
    }

    try {
        json j = json::parse(*result);
        BudgetConfig config;
        config.id = j.value("id", "");
        config.dimension = StringToDimension(j.value("dimension", "global"));
        config.dimension_value = j.value("dimension_value", "");
        config.period = [&]() {
            auto p = j.value("period", "daily");
            if (p == "hourly") return BudgetPeriod::kHourly;
            if (p == "weekly") return BudgetPeriod::kWeekly;
            if (p == "monthly") return BudgetPeriod::kMonthly;
            return BudgetPeriod::kDaily;
        }();
        config.soft_limit_micros = j.value("soft_limit_micros", int64_t{0});
        config.hard_limit_micros = j.value("hard_limit_micros", int64_t{0});
        config.warning_percentage = j.value("warning_percentage", 0.8);
        config.block_on_exceeded = j.value("block_on_exceeded", false);

        if (j.contains("created_at")) {
            config.created_at = system_clock::time_point(
                seconds(j["created_at"].get<int64_t>()));
        }
        if (j.contains("updated_at")) {
            config.updated_at = system_clock::time_point(
                seconds(j["updated_at"].get<int64_t>()));
        }

        return config;
    } catch (const json::exception& e) {
        return absl::InvalidArgumentError(
            std::string("Failed to parse budget config: ") + e.what());
    }
}

absl::Status BudgetManager::DeleteBudget(BudgetDimension dimension,
                                          const std::string& dimension_value) {
    auto key = BuildConfigKey(dimension, dimension_value);
    return redis_->Delete(key);
}

absl::StatusOr<std::vector<BudgetConfig>> BudgetManager::ListBudgets(
    BudgetDimension dimension) {
    // Would require Redis SCAN - placeholder
    std::vector<BudgetConfig> budgets;
    return budgets;
}

// =============================================================================
// Budget Operations
// =============================================================================

absl::StatusOr<BudgetCheckResult> BudgetManager::CheckBudget(
    BudgetDimension dimension,
    const std::string& dimension_value,
    int64_t proposed_spend_micros) {

    // SECURITY: Validate inputs
    if (!IsValidDimensionValue(dimension_value)) {
        return absl::InvalidArgumentError("Invalid dimension value format");
    }
    if (!IsValidSpendAmount(proposed_spend_micros)) {
        return absl::InvalidArgumentError("Invalid spend amount");
    }

    BudgetCheckResult result;
    result.allowed = true;

    // Get budget config
    auto config_result = GetBudget(dimension, dimension_value);
    if (!config_result.ok()) {
        // No budget configured - allow by default
        if (absl::IsNotFound(config_result.status())) {
            return result;
        }
        return config_result.status();
    }

    const auto& config = *config_result;

    // Get current status
    auto status_result = GetStatus(dimension, dimension_value);
    if (!status_result.ok() && !absl::IsNotFound(status_result.status())) {
        return status_result.status();
    }

    int64_t current_spend = status_result.ok() ?
        status_result->current_spend_micros : 0;
    int64_t projected_spend = current_spend + proposed_spend_micros;

    // Check against hard limit
    if (config.hard_limit_micros > 0 &&
        projected_spend > config.hard_limit_micros) {
        if (config.block_on_exceeded) {
            result.allowed = false;
            result.blocked_reason = "Hard budget limit exceeded";
        }
        result.remaining_budget_micros = 0;
    } else if (config.hard_limit_micros > 0) {
        result.remaining_budget_micros = config.hard_limit_micros - projected_spend;
    }

    // Check against soft limit (warning)
    if (config.soft_limit_micros > 0) {
        double utilization = static_cast<double>(projected_spend) /
                             config.soft_limit_micros;
        if (utilization >= config.warning_percentage) {
            result.warning = true;
        }
    }

    return result;
}

absl::StatusOr<BudgetCheckResult> BudgetManager::CheckAndRecordSpend(
    BudgetDimension dimension,
    const std::string& dimension_value,
    int64_t spend_micros) {

    // SECURITY: Validate inputs
    if (!IsValidDimensionValue(dimension_value)) {
        return absl::InvalidArgumentError("Invalid dimension value format");
    }
    if (!IsValidSpendAmount(spend_micros)) {
        return absl::InvalidArgumentError("Invalid spend amount");
    }

    if (spend_micros <= 0) {
        BudgetCheckResult result;
        result.allowed = true;
        return result;
    }

    // Get budget config
    auto config_result = GetBudget(dimension, dimension_value);
    BudgetPeriod period = BudgetPeriod::kDaily;
    int64_t hard_limit = 0;
    int64_t soft_limit = 0;
    bool block_on_exceeded = false;
    double warning_pct = config_.default_warning_percentage;

    if (config_result.ok()) {
        period = config_result->period;
        hard_limit = config_result->hard_limit_micros;
        soft_limit = config_result->soft_limit_micros;
        block_on_exceeded = config_result->block_on_exceeded;
        warning_pct = config_result->warning_percentage;
    }

    auto key = BuildSpendKey(dimension, dimension_value, period);

    // SECURITY: Use Redis Lua script for atomic check-and-increment
    // This prevents race conditions between checking and recording spend
    std::string lua_script = R"(
        local key = KEYS[1]
        local spend = tonumber(ARGV[1])
        local hard_limit = tonumber(ARGV[2])
        local block_on_exceeded = ARGV[3] == "true"

        local current = tonumber(redis.call('GET', key) or '0')
        local projected = current + spend

        if hard_limit > 0 and projected > hard_limit and block_on_exceeded then
            return {0, current, projected}  -- blocked
        end

        redis.call('INCRBY', key, spend)
        return {1, current, projected}  -- allowed
    )";

    auto script_result = redis_->EvalScript(
        lua_script,
        {key},
        {std::to_string(spend_micros),
         std::to_string(hard_limit),
         block_on_exceeded ? "true" : "false"});

    BudgetCheckResult result;

    if (!script_result.ok()) {
        // Fall back to non-atomic operation with warning
        spdlog::warn("SECURITY: Atomic budget check failed, falling back to non-atomic: {}",
                     script_result.status().message());

        // Try regular increment
        auto incr_status = redis_->IncrBy(key, spend_micros);
        if (!incr_status.ok()) {
            std::lock_guard<std::mutex> lock(counters_mutex_);
            local_counters_[key].value += spend_micros;
            local_counters_[key].last_sync = steady_clock::now();
        }

        // Do a regular check
        auto check_result = CheckBudget(dimension, dimension_value, 0);
        if (check_result.ok()) {
            result = *check_result;
        }
        result.allowed = true;  // Already recorded, can't undo
        return result;
    }

    // Parse Lua script result
    auto script_values = *script_result;
    bool allowed = script_values.size() > 0 && script_values[0] == 1;
    int64_t current_spend = script_values.size() > 1 ? script_values[1] : 0;
    int64_t projected_spend = script_values.size() > 2 ? script_values[2] : spend_micros;

    result.allowed = allowed;
    if (!allowed) {
        result.blocked_reason = "Hard budget limit exceeded";
    }

    if (hard_limit > 0) {
        result.remaining_budget_micros = std::max(int64_t{0}, hard_limit - projected_spend);
    }

    if (soft_limit > 0) {
        double utilization = static_cast<double>(projected_spend) / soft_limit;
        if (utilization >= warning_pct) {
            result.warning = true;
        }
    }

    // Check and trigger alerts
    if (config_result.ok()) {
        auto status_result = GetStatus(dimension, dimension_value);
        if (status_result.ok()) {
            CheckAndTriggerAlerts(*status_result, *config_result);
        }
    }

    return result;
}

absl::Status BudgetManager::RecordSpend(BudgetDimension dimension,
                                         const std::string& dimension_value,
                                         int64_t spend_micros) {
    // SECURITY: Validate inputs
    if (!IsValidDimensionValue(dimension_value)) {
        return absl::InvalidArgumentError("Invalid dimension value format");
    }
    if (!IsValidSpendAmount(spend_micros)) {
        return absl::InvalidArgumentError("Invalid spend amount");
    }

    if (spend_micros <= 0) {
        return absl::OkStatus();
    }

    // Get budget config to determine period
    auto config_result = GetBudget(dimension, dimension_value);
    BudgetPeriod period = BudgetPeriod::kDaily;  // Default
    if (config_result.ok()) {
        period = config_result->period;
    }

    // Build spend key with current period
    auto key = BuildSpendKey(dimension, dimension_value, period);

    // Increment counter in Redis
    auto status = redis_->IncrBy(key, spend_micros);
    if (!status.ok()) {
        // Fall back to local counter
        std::lock_guard<std::mutex> lock(counters_mutex_);
        local_counters_[key].value += spend_micros;
        local_counters_[key].last_sync = steady_clock::now();
        spdlog::debug("Stored spend locally due to Redis error: {}", key);
    }

    // Check and trigger alerts
    if (config_result.ok()) {
        auto status_result = GetStatus(dimension, dimension_value);
        if (status_result.ok()) {
            CheckAndTriggerAlerts(*status_result, *config_result);
        }
    }

    return absl::OkStatus();
}

absl::Status BudgetManager::RecordSpendMultiple(
    const std::unordered_map<BudgetDimension,
        std::unordered_map<std::string, int64_t>>& spends) {

    for (const auto& [dimension, values] : spends) {
        for (const auto& [dim_value, spend] : values) {
            auto status = RecordSpend(dimension, dim_value, spend);
            if (!status.ok()) {
                spdlog::warn("Failed to record spend for {}/{}: {}",
                             DimensionToString(dimension), dim_value,
                             status.message());
            }
        }
    }
    return absl::OkStatus();
}

absl::StatusOr<BudgetStatus> BudgetManager::GetStatus(
    BudgetDimension dimension,
    const std::string& dimension_value) {

    // Get budget config
    auto config_result = GetBudget(dimension, dimension_value);
    BudgetPeriod period = BudgetPeriod::kDaily;
    int64_t soft_limit = 0;
    int64_t hard_limit = 0;
    bool block_on_exceeded = false;
    double warning_pct = config_.default_warning_percentage;

    if (config_result.ok()) {
        period = config_result->period;
        soft_limit = config_result->soft_limit_micros;
        hard_limit = config_result->hard_limit_micros;
        block_on_exceeded = config_result->block_on_exceeded;
        warning_pct = config_result->warning_percentage;
    }

    // Get current spend from Redis
    auto key = BuildSpendKey(dimension, dimension_value, period);
    int64_t current_spend = 0;

    auto spend_result = redis_->Get(key);
    if (spend_result.ok()) {
        try {
            current_spend = std::stoll(*spend_result);
        } catch (...) {
            current_spend = 0;
        }
    }

    // Add local counter if any
    {
        std::lock_guard<std::mutex> lock(counters_mutex_);
        auto it = local_counters_.find(key);
        if (it != local_counters_.end()) {
            current_spend += it->second.value;
        }
    }

    // Build status
    BudgetStatus status;
    status.budget_id = config_result.ok() ? config_result->id : "";
    status.dimension = dimension;
    status.dimension_value = dimension_value;
    status.period = period;
    status.current_spend_micros = current_spend;
    status.soft_limit_micros = soft_limit;
    status.hard_limit_micros = hard_limit;
    status.last_updated = system_clock::now();

    // Calculate period boundaries
    auto now = system_clock::now();
    status.period_start = GetPeriodStart(period, now);
    status.period_end = status.period_start + PeriodDuration(period);

    // Calculate utilization
    int64_t limit = hard_limit > 0 ? hard_limit : soft_limit;
    if (limit > 0) {
        status.utilization_percentage =
            static_cast<double>(current_spend) / limit * 100.0;
        status.remaining_micros = std::max(int64_t{0}, limit - current_spend);
    }

    // Set flags
    if (soft_limit > 0) {
        status.warning_triggered =
            current_spend >= static_cast<int64_t>(soft_limit * warning_pct);
    }
    if (hard_limit > 0) {
        status.limit_exceeded = current_spend >= hard_limit;
        status.requests_blocked = status.limit_exceeded && block_on_exceeded;
    }

    return status;
}

absl::StatusOr<std::vector<BudgetStatus>> BudgetManager::GetAllStatuses() {
    // Would require scanning all keys - placeholder
    std::vector<BudgetStatus> statuses;
    return statuses;
}

absl::Status BudgetManager::ResetBudget(BudgetDimension dimension,
                                         const std::string& dimension_value) {
    auto config_result = GetBudget(dimension, dimension_value);
    BudgetPeriod period = config_result.ok() ?
        config_result->period : BudgetPeriod::kDaily;

    auto key = BuildSpendKey(dimension, dimension_value, period);
    auto status = redis_->Set(key, "0", PeriodDuration(period));

    // Clear local counter
    {
        std::lock_guard<std::mutex> lock(counters_mutex_);
        local_counters_.erase(key);
    }

    // Trigger reset alert
    if (!alert_callbacks_.empty()) {
        BudgetAlertEvent event;
        event.type = BudgetAlertEvent::AlertType::kReset;
        event.dimension = dimension;
        event.dimension_value = dimension_value;
        event.current_spend_micros = 0;
        event.message = "Budget manually reset";
        event.timestamp = system_clock::now();

        for (const auto& callback : alert_callbacks_) {
            callback(event);
        }
    }

    return status;
}

// =============================================================================
// Alerts and Callbacks
// =============================================================================

void BudgetManager::RegisterAlertCallback(BudgetAlertCallback callback) {
    alert_callbacks_.push_back(std::move(callback));
}

absl::StatusOr<std::vector<BudgetAlertEvent>> BudgetManager::GetRecentAlerts(
    size_t limit) {
    // Would store alerts in Redis list - placeholder
    std::vector<BudgetAlertEvent> alerts;
    return alerts;
}

// =============================================================================
// Forecasting
// =============================================================================

absl::StatusOr<int64_t> BudgetManager::ForecastSpend(
    BudgetDimension dimension,
    const std::string& dimension_value) {

    auto status = GetStatus(dimension, dimension_value);
    if (!status.ok()) {
        return status.status();
    }

    auto now = system_clock::now();
    auto period_start = status->period_start;
    auto period_end = status->period_end;

    // Calculate elapsed and remaining time
    auto elapsed = duration_cast<seconds>(now - period_start);
    auto total_duration = duration_cast<seconds>(period_end - period_start);

    if (elapsed.count() <= 0) {
        return status->current_spend_micros;
    }

    // Project spend at current rate
    double rate_per_second = static_cast<double>(status->current_spend_micros) /
                             elapsed.count();
    int64_t projected = static_cast<int64_t>(rate_per_second * total_duration.count());

    return projected;
}

absl::StatusOr<double> BudgetManager::GetSpendRate(
    BudgetDimension dimension,
    const std::string& dimension_value) {

    auto status = GetStatus(dimension, dimension_value);
    if (!status.ok()) {
        return status.status();
    }

    auto now = system_clock::now();
    auto elapsed = duration_cast<hours>(now - status->period_start);

    if (elapsed.count() <= 0) {
        return 0.0;
    }

    return static_cast<double>(status->current_spend_micros) / elapsed.count();
}

// =============================================================================
// Static Helpers
// =============================================================================

std::string BudgetManager::DimensionToString(BudgetDimension dimension) {
    switch (dimension) {
        case BudgetDimension::kGlobal: return "global";
        case BudgetDimension::kUser: return "user";
        case BudgetDimension::kModel: return "model";
        case BudgetDimension::kFeature: return "feature";
        case BudgetDimension::kTeam: return "team";
        case BudgetDimension::kEnvironment: return "environment";
    }
    return "unknown";
}

BudgetDimension BudgetManager::StringToDimension(const std::string& str) {
    if (str == "user") return BudgetDimension::kUser;
    if (str == "model") return BudgetDimension::kModel;
    if (str == "feature") return BudgetDimension::kFeature;
    if (str == "team") return BudgetDimension::kTeam;
    if (str == "environment") return BudgetDimension::kEnvironment;
    return BudgetDimension::kGlobal;
}

std::string BudgetManager::PeriodToString(BudgetPeriod period) {
    switch (period) {
        case BudgetPeriod::kHourly: return "hourly";
        case BudgetPeriod::kDaily: return "daily";
        case BudgetPeriod::kWeekly: return "weekly";
        case BudgetPeriod::kMonthly: return "monthly";
    }
    return "daily";
}

seconds BudgetManager::PeriodDuration(BudgetPeriod period) {
    switch (period) {
        case BudgetPeriod::kHourly: return hours(1);
        case BudgetPeriod::kDaily: return hours(24);
        case BudgetPeriod::kWeekly: return hours(24 * 7);
        case BudgetPeriod::kMonthly: return hours(24 * 30);  // Approximate
    }
    return hours(24);
}

system_clock::time_point BudgetManager::GetPeriodStart(
    BudgetPeriod period,
    system_clock::time_point timestamp) {

    auto time_t_val = system_clock::to_time_t(timestamp);
    std::tm tm_val;
#ifdef _WIN32
    localtime_s(&tm_val, &time_t_val);
#else
    localtime_r(&time_t_val, &tm_val);
#endif

    switch (period) {
        case BudgetPeriod::kHourly:
            tm_val.tm_min = 0;
            tm_val.tm_sec = 0;
            break;

        case BudgetPeriod::kDaily:
            tm_val.tm_hour = 0;
            tm_val.tm_min = 0;
            tm_val.tm_sec = 0;
            break;

        case BudgetPeriod::kWeekly:
            // Start of week (Sunday)
            tm_val.tm_mday -= tm_val.tm_wday;
            tm_val.tm_hour = 0;
            tm_val.tm_min = 0;
            tm_val.tm_sec = 0;
            break;

        case BudgetPeriod::kMonthly:
            tm_val.tm_mday = 1;
            tm_val.tm_hour = 0;
            tm_val.tm_min = 0;
            tm_val.tm_sec = 0;
            break;
    }

    return system_clock::from_time_t(std::mktime(&tm_val));
}

// =============================================================================
// Private Methods
// =============================================================================

std::string BudgetManager::BuildConfigKey(
    BudgetDimension dimension,
    const std::string& dimension_value) const {
    return config_.key_prefix + ":config:" +
           DimensionToString(dimension) + ":" + dimension_value;
}

std::string BudgetManager::BuildSpendKey(
    BudgetDimension dimension,
    const std::string& dimension_value,
    BudgetPeriod period) const {

    auto period_start = GetPeriodStart(period, system_clock::now());
    auto period_ts = duration_cast<seconds>(
        period_start.time_since_epoch()).count();

    return config_.key_prefix + ":spend:" +
           DimensionToString(dimension) + ":" + dimension_value +
           ":" + PeriodToString(period) + ":" + std::to_string(period_ts);
}

void BudgetManager::CheckAndTriggerAlerts(const BudgetStatus& status,
                                           const BudgetConfig& config) {
    if (alert_callbacks_.empty()) {
        return;
    }

    BudgetAlertEvent event;
    event.dimension = status.dimension;
    event.dimension_value = status.dimension_value;
    event.budget_id = status.budget_id;
    event.current_spend_micros = status.current_spend_micros;
    event.utilization_percentage = status.utilization_percentage;
    event.timestamp = system_clock::now();

    bool should_alert = false;

    // Check hard limit
    if (config.hard_limit_micros > 0 &&
        status.current_spend_micros >= config.hard_limit_micros) {
        event.type = BudgetAlertEvent::AlertType::kHardExceeded;
        event.limit_micros = config.hard_limit_micros;
        event.message = "Hard budget limit exceeded";
        should_alert = true;
    }
    // Check soft limit
    else if (config.soft_limit_micros > 0 &&
             status.current_spend_micros >= config.soft_limit_micros) {
        event.type = BudgetAlertEvent::AlertType::kSoftExceeded;
        event.limit_micros = config.soft_limit_micros;
        event.message = "Soft budget limit exceeded";
        should_alert = true;
    }
    // Check warning threshold
    else if (config.soft_limit_micros > 0 &&
             status.current_spend_micros >=
                 static_cast<int64_t>(config.soft_limit_micros *
                                      config.warning_percentage)) {
        event.type = BudgetAlertEvent::AlertType::kWarning;
        event.limit_micros = config.soft_limit_micros;
        event.message = "Approaching budget limit";
        should_alert = true;
    }

    if (should_alert) {
        for (const auto& callback : alert_callbacks_) {
            callback(event);
        }
    }
}

absl::Status BudgetManager::FlushCounters() {
    std::lock_guard<std::mutex> lock(counters_mutex_);

    for (auto& [key, counter] : local_counters_) {
        if (counter.value > 0) {
            auto status = redis_->IncrBy(key, counter.value);
            if (status.ok()) {
                counter.value = 0;
                counter.last_sync = steady_clock::now();
            }
        }
    }

    return absl::OkStatus();
}

}  // namespace pyflare::cost
