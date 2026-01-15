#pragma once

/// @file costs_handler.h
/// @brief Cost tracking query handlers

#include "query/handlers/handler_base.h"

namespace pyflare::query::handlers {

/// @brief Cost summary data
struct CostSummary {
    std::string period;  ///< "hourly", "daily", "weekly", "monthly"
    std::string start_time;
    std::string end_time;

    int64_t total_cost_micros = 0;
    int64_t input_cost_micros = 0;
    int64_t output_cost_micros = 0;

    int64_t total_tokens = 0;
    int64_t input_tokens = 0;
    int64_t output_tokens = 0;

    int64_t request_count = 0;
    double avg_cost_per_request_micros = 0.0;

    nlohmann::json by_model;
    nlohmann::json by_user;
    nlohmann::json by_feature;
};

/// @brief Get cost summary: GET /api/v1/costs/summary
class CostSummaryHandler : public Handler {
public:
    HttpResponse Handle(const HttpRequest& request,
                        const HandlerContext& context) override;

    std::string GetRoute() const override { return "/api/v1/costs/summary"; }
    std::vector<HttpMethod> GetMethods() const override {
        return {HttpMethod::kGet};
    }
};

/// @brief Get cost breakdown by dimension: GET /api/v1/costs/breakdown
class CostBreakdownHandler : public Handler {
public:
    HttpResponse Handle(const HttpRequest& request,
                        const HandlerContext& context) override;

    std::string GetRoute() const override { return "/api/v1/costs/breakdown"; }
    std::vector<HttpMethod> GetMethods() const override {
        return {HttpMethod::kGet};
    }
};

/// @brief Get cost timeline: GET /api/v1/costs/timeline
class CostTimelineHandler : public Handler {
public:
    HttpResponse Handle(const HttpRequest& request,
                        const HandlerContext& context) override;

    std::string GetRoute() const override { return "/api/v1/costs/timeline"; }
    std::vector<HttpMethod> GetMethods() const override {
        return {HttpMethod::kGet};
    }
};

/// @brief Get budget status: GET /api/v1/costs/budgets
class BudgetStatusHandler : public Handler {
public:
    HttpResponse Handle(const HttpRequest& request,
                        const HandlerContext& context) override;

    std::string GetRoute() const override { return "/api/v1/costs/budgets"; }
    std::vector<HttpMethod> GetMethods() const override {
        return {HttpMethod::kGet};
    }
};

/// @brief Create/update budget: POST /api/v1/costs/budgets
class CreateBudgetHandler : public Handler {
public:
    HttpResponse Handle(const HttpRequest& request,
                        const HandlerContext& context) override;

    std::string GetRoute() const override { return "/api/v1/costs/budgets"; }
    std::vector<HttpMethod> GetMethods() const override {
        return {HttpMethod::kPost};
    }
};

/// @brief Get cost forecast: GET /api/v1/costs/forecast
class CostForecastHandler : public Handler {
public:
    HttpResponse Handle(const HttpRequest& request,
                        const HandlerContext& context) override;

    std::string GetRoute() const override { return "/api/v1/costs/forecast"; }
    std::vector<HttpMethod> GetMethods() const override {
        return {HttpMethod::kGet};
    }
};

/// @brief Register all cost handlers
void RegisterCostsHandlers(HandlerRegistry& registry);

}  // namespace pyflare::query::handlers
