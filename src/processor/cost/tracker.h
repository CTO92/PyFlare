#pragma once

/// @file tracker.h
/// @brief Cost tracking for ML inference in PyFlare

#include <chrono>
#include <functional>
#include <memory>
#include <string>
#include <unordered_map>

#include <absl/status/statusor.h>

namespace pyflare::cost {

/// @brief Cost calculation result
struct CostResult {
    std::string trace_id;
    std::string model_id;

    int64_t input_tokens = 0;
    int64_t output_tokens = 0;
    int64_t total_tokens = 0;

    int64_t input_cost_micros = 0;   ///< Micro-dollars
    int64_t output_cost_micros = 0;
    int64_t total_cost_micros = 0;

    std::string user_id;
    std::string feature_id;
    std::string environment;

    std::chrono::system_clock::time_point timestamp;
};

/// @brief Model pricing configuration
struct ModelPricing {
    std::string model_id;
    std::string provider;
    int64_t input_cost_per_million_tokens;   ///< Micro-dollars
    int64_t output_cost_per_million_tokens;
    std::chrono::system_clock::time_point effective_from;
};

/// @brief Budget alert configuration
struct BudgetAlert {
    std::string dimension;
    std::string value;
    int64_t current_spend_micros;
    int64_t threshold_micros;
    std::chrono::system_clock::time_point triggered_at;
};

/// @brief Trace record for cost calculation
struct TraceRecord {
    std::string trace_id;
    std::string model_id;
    int64_t input_tokens = 0;
    int64_t output_tokens = 0;
    std::string user_id;
    std::string feature_id;
    std::string environment;
    std::chrono::system_clock::time_point timestamp;
};

/// @brief Cost tracker for inference requests
class CostTracker {
public:
    struct Config {
        std::string pricing_config_path;
    };

    explicit CostTracker(Config config = {});
    ~CostTracker();

    /// @brief Calculate cost for a trace
    absl::StatusOr<CostResult> Calculate(const TraceRecord& record);

    /// @brief Update pricing for a model
    absl::Status UpdatePricing(const ModelPricing& pricing);

    /// @brief Get pricing for a model
    absl::StatusOr<ModelPricing> GetPricing(const std::string& model_id) const;

    /// @brief Set a budget alert
    void SetBudgetAlert(
        const std::string& dimension,
        const std::string& value,
        int64_t threshold_micros,
        std::function<void(const BudgetAlert&)> callback);

    /// @brief Get total spend for a dimension
    int64_t GetSpend(const std::string& dimension, const std::string& value) const;

private:
    Config config_;
    class Impl;
    std::unique_ptr<Impl> impl_;
};

}  // namespace pyflare::cost
