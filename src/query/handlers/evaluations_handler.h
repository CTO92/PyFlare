#pragma once

/// @file evaluations_handler.h
/// @brief Evaluation results query handlers

#include "query/handlers/handler_base.h"

namespace pyflare::query::handlers {

/// @brief Evaluation result data
struct EvaluationData {
    std::string id;
    std::string trace_id;
    std::string model_id;
    std::string evaluator_type;  ///< "hallucination", "toxicity", "rag_quality"

    double score = 0.0;
    std::string verdict;  ///< "pass", "fail", "warn"
    std::string explanation;

    std::string timestamp;
    nlohmann::json metadata;
};

/// @brief List evaluations: GET /api/v1/evaluations
class ListEvaluationsHandler : public Handler {
public:
    HttpResponse Handle(const HttpRequest& request,
                        const HandlerContext& context) override;

    std::string GetRoute() const override { return "/api/v1/evaluations"; }
    std::vector<HttpMethod> GetMethods() const override {
        return {HttpMethod::kGet};
    }
};

/// @brief Get evaluation by ID: GET /api/v1/evaluations/:id
class GetEvaluationHandler : public Handler {
public:
    HttpResponse Handle(const HttpRequest& request,
                        const HandlerContext& context) override;

    std::string GetRoute() const override { return "/api/v1/evaluations/:id"; }
    std::vector<HttpMethod> GetMethods() const override {
        return {HttpMethod::kGet};
    }
};

/// @brief Get evaluation summary: GET /api/v1/evaluations/summary
class EvaluationSummaryHandler : public Handler {
public:
    HttpResponse Handle(const HttpRequest& request,
                        const HandlerContext& context) override;

    std::string GetRoute() const override { return "/api/v1/evaluations/summary"; }
    std::vector<HttpMethod> GetMethods() const override {
        return {HttpMethod::kGet};
    }
};

/// @brief Get evaluations by trace: GET /api/v1/traces/:id/evaluations
class TraceEvaluationsHandler : public Handler {
public:
    HttpResponse Handle(const HttpRequest& request,
                        const HandlerContext& context) override;

    std::string GetRoute() const override { return "/api/v1/traces/:id/evaluations"; }
    std::vector<HttpMethod> GetMethods() const override {
        return {HttpMethod::kGet};
    }
};

/// @brief Trigger manual evaluation: POST /api/v1/evaluations
class TriggerEvaluationHandler : public Handler {
public:
    HttpResponse Handle(const HttpRequest& request,
                        const HandlerContext& context) override;

    std::string GetRoute() const override { return "/api/v1/evaluations"; }
    std::vector<HttpMethod> GetMethods() const override {
        return {HttpMethod::kPost};
    }
};

/// @brief Get evaluation trends: GET /api/v1/evaluations/trends
class EvaluationTrendsHandler : public Handler {
public:
    HttpResponse Handle(const HttpRequest& request,
                        const HandlerContext& context) override;

    std::string GetRoute() const override { return "/api/v1/evaluations/trends"; }
    std::vector<HttpMethod> GetMethods() const override {
        return {HttpMethod::kGet};
    }
};

/// @brief Register all evaluation handlers
void RegisterEvaluationsHandlers(HandlerRegistry& registry);

}  // namespace pyflare::query::handlers
