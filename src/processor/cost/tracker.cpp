#include "tracker.h"

#include "common/logging.h"

namespace pyflare::cost {

class CostTracker::Impl {
public:
    explicit Impl(Config config) : config_(std::move(config)) {
        // Initialize default pricing for common models
        InitializeDefaultPricing();
    }

    void InitializeDefaultPricing() {
        // GPT-4
        pricing_["gpt-4"] = ModelPricing{
            .model_id = "gpt-4",
            .provider = "openai",
            .input_cost_per_million_tokens = 30000000,   // $30 per million
            .output_cost_per_million_tokens = 60000000,  // $60 per million
            .effective_from = std::chrono::system_clock::now()
        };

        // GPT-3.5-turbo
        pricing_["gpt-3.5-turbo"] = ModelPricing{
            .model_id = "gpt-3.5-turbo",
            .provider = "openai",
            .input_cost_per_million_tokens = 500000,    // $0.50 per million
            .output_cost_per_million_tokens = 1500000,  // $1.50 per million
            .effective_from = std::chrono::system_clock::now()
        };

        // Claude 3 Opus
        pricing_["claude-3-opus"] = ModelPricing{
            .model_id = "claude-3-opus",
            .provider = "anthropic",
            .input_cost_per_million_tokens = 15000000,  // $15 per million
            .output_cost_per_million_tokens = 75000000, // $75 per million
            .effective_from = std::chrono::system_clock::now()
        };
    }

    absl::StatusOr<CostResult> Calculate(const TraceRecord& record) {
        auto it = pricing_.find(record.model_id);
        if (it == pricing_.end()) {
            return absl::NotFoundError(
                "No pricing configuration for model: " + record.model_id);
        }

        const auto& pricing = it->second;

        CostResult result;
        result.trace_id = record.trace_id;
        result.model_id = record.model_id;
        result.input_tokens = record.input_tokens;
        result.output_tokens = record.output_tokens;
        result.total_tokens = record.input_tokens + record.output_tokens;

        // Calculate costs (in micro-dollars)
        result.input_cost_micros = (record.input_tokens *
            pricing.input_cost_per_million_tokens) / 1000000;
        result.output_cost_micros = (record.output_tokens *
            pricing.output_cost_per_million_tokens) / 1000000;
        result.total_cost_micros = result.input_cost_micros + result.output_cost_micros;

        result.user_id = record.user_id;
        result.feature_id = record.feature_id;
        result.environment = record.environment;
        result.timestamp = record.timestamp;

        // Update spend tracking
        UpdateSpend(record.user_id, result.total_cost_micros);
        UpdateSpend(record.feature_id, result.total_cost_micros);

        return result;
    }

    absl::Status UpdatePricing(const ModelPricing& pricing) {
        pricing_[pricing.model_id] = pricing;
        PYFLARE_LOG_INFO("Updated pricing for model: {}", pricing.model_id);
        return absl::OkStatus();
    }

    absl::StatusOr<ModelPricing> GetPricing(const std::string& model_id) const {
        auto it = pricing_.find(model_id);
        if (it == pricing_.end()) {
            return absl::NotFoundError("No pricing for model: " + model_id);
        }
        return it->second;
    }

    void SetBudgetAlert(
        const std::string& dimension,
        const std::string& value,
        int64_t threshold_micros,
        std::function<void(const BudgetAlert&)> callback) {
        // Store alert configuration
        std::string key = dimension + ":" + value;
        budget_alerts_[key] = {threshold_micros, std::move(callback)};
    }

    int64_t GetSpend(const std::string& dimension, const std::string& value) const {
        std::string key = dimension + ":" + value;
        auto it = spend_tracking_.find(key);
        return it != spend_tracking_.end() ? it->second : 0;
    }

private:
    void UpdateSpend(const std::string& key, int64_t amount) {
        spend_tracking_[key] += amount;

        // Check budget alerts
        auto alert_it = budget_alerts_.find(key);
        if (alert_it != budget_alerts_.end()) {
            if (spend_tracking_[key] > alert_it->second.first) {
                BudgetAlert alert{
                    .dimension = "custom",
                    .value = key,
                    .current_spend_micros = spend_tracking_[key],
                    .threshold_micros = alert_it->second.first,
                    .triggered_at = std::chrono::system_clock::now()
                };
                alert_it->second.second(alert);
            }
        }
    }

    Config config_;
    std::unordered_map<std::string, ModelPricing> pricing_;
    std::unordered_map<std::string, int64_t> spend_tracking_;
    std::unordered_map<std::string,
        std::pair<int64_t, std::function<void(const BudgetAlert&)>>> budget_alerts_;
};

CostTracker::CostTracker(Config config)
    : config_(std::move(config)), impl_(std::make_unique<Impl>(config_)) {}

CostTracker::~CostTracker() = default;

absl::StatusOr<CostResult> CostTracker::Calculate(const TraceRecord& record) {
    return impl_->Calculate(record);
}

absl::Status CostTracker::UpdatePricing(const ModelPricing& pricing) {
    return impl_->UpdatePricing(pricing);
}

absl::StatusOr<ModelPricing> CostTracker::GetPricing(const std::string& model_id) const {
    return impl_->GetPricing(model_id);
}

void CostTracker::SetBudgetAlert(
    const std::string& dimension,
    const std::string& value,
    int64_t threshold_micros,
    std::function<void(const BudgetAlert&)> callback) {
    impl_->SetBudgetAlert(dimension, value, threshold_micros, std::move(callback));
}

int64_t CostTracker::GetSpend(const std::string& dimension, const std::string& value) const {
    return impl_->GetSpend(dimension, value);
}

}  // namespace pyflare::cost
