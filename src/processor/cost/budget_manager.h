#pragma once

/// @file budget_manager.h
/// @brief Budget management and cost throttling for PyFlare

#include <chrono>
#include <functional>
#include <memory>
#include <optional>
#include <string>
#include <vector>

#include <absl/status/status.h>
#include <absl/status/statusor.h>

#include "storage/redis/client.h"

namespace pyflare::cost {

/// @brief Time period for budget tracking
enum class BudgetPeriod {
    kHourly,
    kDaily,
    kWeekly,
    kMonthly
};

/// @brief Budget dimension types
enum class BudgetDimension {
    kGlobal,      ///< Total spend across all usage
    kUser,        ///< Per-user budget
    kModel,       ///< Per-model budget
    kFeature,     ///< Per-feature/endpoint budget
    kTeam,        ///< Per-team budget
    kEnvironment  ///< Per-environment (prod/staging/dev)
};

/// @brief Budget configuration
struct BudgetConfig {
    std::string id;
    BudgetDimension dimension = BudgetDimension::kGlobal;
    std::string dimension_value;  ///< e.g., user_id, model_id, etc.
    BudgetPeriod period = BudgetPeriod::kDaily;

    int64_t soft_limit_micros = 0;  ///< Warning threshold (micro-dollars)
    int64_t hard_limit_micros = 0;  ///< Blocking threshold (micro-dollars)

    double warning_percentage = 0.8;  ///< Warn at 80% by default
    bool block_on_exceeded = false;   ///< Block requests when exceeded

    std::chrono::system_clock::time_point created_at;
    std::chrono::system_clock::time_point updated_at;
};

/// @brief Current budget status
struct BudgetStatus {
    std::string budget_id;
    BudgetDimension dimension;
    std::string dimension_value;
    BudgetPeriod period;

    int64_t current_spend_micros = 0;
    int64_t soft_limit_micros = 0;
    int64_t hard_limit_micros = 0;

    double utilization_percentage = 0.0;
    bool warning_triggered = false;
    bool limit_exceeded = false;
    bool requests_blocked = false;

    int64_t remaining_micros = 0;
    std::chrono::system_clock::time_point period_start;
    std::chrono::system_clock::time_point period_end;
    std::chrono::system_clock::time_point last_updated;
};

/// @brief Budget alert event
struct BudgetAlertEvent {
    enum class AlertType {
        kWarning,        ///< Approaching limit
        kSoftExceeded,   ///< Soft limit exceeded
        kHardExceeded,   ///< Hard limit exceeded
        kReset           ///< Budget period reset
    };

    AlertType type;
    std::string budget_id;
    BudgetDimension dimension;
    std::string dimension_value;

    int64_t current_spend_micros = 0;
    int64_t limit_micros = 0;
    double utilization_percentage = 0.0;

    std::string message;
    std::chrono::system_clock::time_point timestamp;
};

/// @brief Callback for budget alerts
using BudgetAlertCallback = std::function<void(const BudgetAlertEvent&)>;

/// @brief Budget check result
struct BudgetCheckResult {
    bool allowed = true;
    bool warning = false;
    std::optional<std::string> blocked_reason;
    std::optional<int64_t> remaining_budget_micros;
    std::optional<std::chrono::seconds> retry_after;  ///< For rate limiting
};

/// @brief Configuration for BudgetManager
struct BudgetManagerConfig {
    /// Redis key prefix for budget data
    std::string key_prefix = "pyflare:budget";

    /// How often to sync local counters to Redis
    std::chrono::seconds sync_interval = std::chrono::seconds(5);

    /// Enable local caching of budget status
    bool enable_local_cache = true;

    /// Cache TTL for budget status
    std::chrono::seconds cache_ttl = std::chrono::seconds(30);

    /// Default soft limit percentage
    double default_warning_percentage = 0.8;
};

/// @brief Budget manager for cost control and alerting
///
/// Provides hierarchical budget management with support for:
/// - Multiple budget dimensions (user, model, feature, team)
/// - Soft and hard limits with customizable alerts
/// - Time-based budget periods (hourly, daily, monthly)
/// - Request blocking when hard limits exceeded
///
/// Example usage:
/// @code
///   BudgetManagerConfig config;
///   auto manager = std::make_unique<BudgetManager>(redis, config);
///
///   // Create a daily budget for a user
///   BudgetConfig budget;
///   budget.dimension = BudgetDimension::kUser;
///   budget.dimension_value = "user123";
///   budget.period = BudgetPeriod::kDaily;
///   budget.soft_limit_micros = 10000000;  // $10
///   budget.hard_limit_micros = 15000000;  // $15
///   budget.block_on_exceeded = true;
///   manager->CreateBudget(budget);
///
///   // Check before processing request
///   auto check = manager->CheckBudget(BudgetDimension::kUser, "user123", cost);
///   if (!check->allowed) {
///       // Block request
///   }
///
///   // Record spend after processing
///   manager->RecordSpend(BudgetDimension::kUser, "user123", actual_cost);
/// @endcode
class BudgetManager {
public:
    BudgetManager(std::shared_ptr<storage::RedisClient> redis,
                  BudgetManagerConfig config = {});
    ~BudgetManager();

    // Disable copy
    BudgetManager(const BudgetManager&) = delete;
    BudgetManager& operator=(const BudgetManager&) = delete;

    /// @brief Initialize the budget manager
    absl::Status Initialize();

    /// @brief Shutdown and flush pending data
    absl::Status Shutdown();

    // =========================================================================
    // Budget Configuration
    // =========================================================================

    /// @brief Create or update a budget
    /// @param config Budget configuration
    absl::Status CreateBudget(const BudgetConfig& config);

    /// @brief Get budget configuration
    /// @param dimension Budget dimension
    /// @param dimension_value Value for the dimension
    absl::StatusOr<BudgetConfig> GetBudget(BudgetDimension dimension,
                                            const std::string& dimension_value);

    /// @brief Delete a budget
    absl::Status DeleteBudget(BudgetDimension dimension,
                              const std::string& dimension_value);

    /// @brief List all budgets for a dimension
    absl::StatusOr<std::vector<BudgetConfig>> ListBudgets(
        BudgetDimension dimension = BudgetDimension::kGlobal);

    // =========================================================================
    // Budget Operations
    // =========================================================================

    /// @brief Check if spend is allowed within budget
    /// @param dimension Budget dimension to check
    /// @param dimension_value Value for the dimension
    /// @param proposed_spend_micros The amount about to be spent
    /// @return Check result with allowed/warning/blocked status
    absl::StatusOr<BudgetCheckResult> CheckBudget(
        BudgetDimension dimension,
        const std::string& dimension_value,
        int64_t proposed_spend_micros = 0);

    /// @brief Record actual spend against a budget
    /// @param dimension Budget dimension
    /// @param dimension_value Value for the dimension
    /// @param spend_micros Amount spent in micro-dollars
    absl::Status RecordSpend(BudgetDimension dimension,
                              const std::string& dimension_value,
                              int64_t spend_micros);

    /// @brief Record spend against multiple dimensions at once
    /// @param spends Map of dimension_value -> spend_micros for each dimension
    absl::Status RecordSpendMultiple(
        const std::unordered_map<BudgetDimension,
            std::unordered_map<std::string, int64_t>>& spends);

    /// @brief Get current budget status
    /// @param dimension Budget dimension
    /// @param dimension_value Value for the dimension
    absl::StatusOr<BudgetStatus> GetStatus(BudgetDimension dimension,
                                            const std::string& dimension_value);

    /// @brief Get aggregated status across all budgets
    absl::StatusOr<std::vector<BudgetStatus>> GetAllStatuses();

    /// @brief Reset budget to zero (for manual reset)
    absl::Status ResetBudget(BudgetDimension dimension,
                              const std::string& dimension_value);

    // =========================================================================
    // Alerts and Callbacks
    // =========================================================================

    /// @brief Register callback for budget alerts
    void RegisterAlertCallback(BudgetAlertCallback callback);

    /// @brief Get recent alerts
    absl::StatusOr<std::vector<BudgetAlertEvent>> GetRecentAlerts(
        size_t limit = 100);

    // =========================================================================
    // Forecasting
    // =========================================================================

    /// @brief Estimate spend for remaining period based on current rate
    /// @param dimension Budget dimension
    /// @param dimension_value Value for the dimension
    /// @return Projected total spend at current rate
    absl::StatusOr<int64_t> ForecastSpend(BudgetDimension dimension,
                                           const std::string& dimension_value);

    /// @brief Get spend rate (micro-dollars per hour)
    /// @param dimension Budget dimension
    /// @param dimension_value Value for the dimension
    absl::StatusOr<double> GetSpendRate(BudgetDimension dimension,
                                         const std::string& dimension_value);

    // =========================================================================
    // Helpers
    // =========================================================================

    /// @brief Convert dimension enum to string
    static std::string DimensionToString(BudgetDimension dimension);

    /// @brief Convert string to dimension enum
    static BudgetDimension StringToDimension(const std::string& str);

    /// @brief Convert period enum to string
    static std::string PeriodToString(BudgetPeriod period);

    /// @brief Get period duration in seconds
    static std::chrono::seconds PeriodDuration(BudgetPeriod period);

    /// @brief Get period start time for a given timestamp
    static std::chrono::system_clock::time_point GetPeriodStart(
        BudgetPeriod period,
        std::chrono::system_clock::time_point timestamp);

private:
    /// @brief Build Redis key for budget config
    std::string BuildConfigKey(BudgetDimension dimension,
                                const std::string& dimension_value) const;

    /// @brief Build Redis key for spend counter
    std::string BuildSpendKey(BudgetDimension dimension,
                               const std::string& dimension_value,
                               BudgetPeriod period) const;

    /// @brief Check and trigger alerts if needed
    void CheckAndTriggerAlerts(const BudgetStatus& status,
                                const BudgetConfig& config);

    /// @brief Flush local counters to Redis
    absl::Status FlushCounters();

    std::shared_ptr<storage::RedisClient> redis_;
    BudgetManagerConfig config_;
    std::vector<BudgetAlertCallback> alert_callbacks_;
    bool initialized_ = false;

    // Local counter cache for batching
    struct LocalCounter {
        int64_t value = 0;
        std::chrono::steady_clock::time_point last_sync;
    };
    std::unordered_map<std::string, LocalCounter> local_counters_;
    mutable std::mutex counters_mutex_;
};

}  // namespace pyflare::cost
