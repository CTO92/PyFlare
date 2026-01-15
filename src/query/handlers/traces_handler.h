#pragma once

/// @file traces_handler.h
/// @brief Trace query handlers

#include "query/handlers/handler_base.h"

namespace pyflare::query::handlers {

/// @brief Trace data structure
struct TraceData {
    std::string trace_id;
    std::string span_id;
    std::string model_id;
    std::string user_id;
    std::string feature_id;

    int64_t start_time_ns = 0;
    int64_t end_time_ns = 0;
    int64_t latency_ms = 0;

    std::string status;  ///< "ok", "error"
    std::string error_type;
    std::string error_message;

    int64_t input_tokens = 0;
    int64_t output_tokens = 0;
    int64_t total_tokens = 0;
    int64_t cost_micros = 0;

    double eval_score = 0.0;
    double drift_score = 0.0;
    double toxicity_score = 0.0;

    std::string input;
    std::string output;

    nlohmann::json attributes;
};

/// @brief List traces handler: GET /api/v1/traces
class ListTracesHandler : public Handler {
public:
    HttpResponse Handle(const HttpRequest& request,
                        const HandlerContext& context) override;

    std::string GetRoute() const override { return "/api/v1/traces"; }
    std::vector<HttpMethod> GetMethods() const override {
        return {HttpMethod::kGet};
    }
};

/// @brief Get single trace handler: GET /api/v1/traces/:id
class GetTraceHandler : public Handler {
public:
    HttpResponse Handle(const HttpRequest& request,
                        const HandlerContext& context) override;

    std::string GetRoute() const override { return "/api/v1/traces/:id"; }
    std::vector<HttpMethod> GetMethods() const override {
        return {HttpMethod::kGet};
    }
};

/// @brief Search traces handler: POST /api/v1/traces/search
class SearchTracesHandler : public Handler {
public:
    HttpResponse Handle(const HttpRequest& request,
                        const HandlerContext& context) override;

    std::string GetRoute() const override { return "/api/v1/traces/search"; }
    std::vector<HttpMethod> GetMethods() const override {
        return {HttpMethod::kPost};
    }
};

/// @brief Trace statistics handler: GET /api/v1/traces/stats
class TraceStatsHandler : public Handler {
public:
    HttpResponse Handle(const HttpRequest& request,
                        const HandlerContext& context) override;

    std::string GetRoute() const override { return "/api/v1/traces/stats"; }
    std::vector<HttpMethod> GetMethods() const override {
        return {HttpMethod::kGet};
    }
};

/// @brief Trace timeline handler: GET /api/v1/traces/:id/timeline
class TraceTimelineHandler : public Handler {
public:
    HttpResponse Handle(const HttpRequest& request,
                        const HandlerContext& context) override;

    std::string GetRoute() const override { return "/api/v1/traces/:id/timeline"; }
    std::vector<HttpMethod> GetMethods() const override {
        return {HttpMethod::kGet};
    }
};

/// @brief Register all trace handlers
void RegisterTracesHandlers(HandlerRegistry& registry);

}  // namespace pyflare::query::handlers
