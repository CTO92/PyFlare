#pragma once

/// @file drift_handler.h
/// @brief Drift detection query handlers

#include "query/handlers/handler_base.h"

namespace pyflare::query::handlers {

/// @brief Drift alert data
struct DriftAlert {
    std::string id;
    std::string model_id;
    std::string drift_type;  ///< "feature", "embedding", "concept"
    std::string severity;    ///< "low", "medium", "high", "critical"

    double score = 0.0;
    double threshold = 0.0;

    std::string feature_name;  ///< For feature drift
    nlohmann::json affected_slices;

    std::string timestamp;
    std::string resolved_at;
    bool is_resolved = false;

    std::string description;
    nlohmann::json metadata;
};

/// @brief List drift alerts handler: GET /api/v1/drift/alerts
class ListDriftAlertsHandler : public Handler {
public:
    HttpResponse Handle(const HttpRequest& request,
                        const HandlerContext& context) override;

    std::string GetRoute() const override { return "/api/v1/drift/alerts"; }
    std::vector<HttpMethod> GetMethods() const override {
        return {HttpMethod::kGet};
    }
};

/// @brief Get drift status by model: GET /api/v1/drift/models/:id
class GetDriftStatusHandler : public Handler {
public:
    HttpResponse Handle(const HttpRequest& request,
                        const HandlerContext& context) override;

    std::string GetRoute() const override { return "/api/v1/drift/models/:id"; }
    std::vector<HttpMethod> GetMethods() const override {
        return {HttpMethod::kGet};
    }
};

/// @brief Get drift timeline: GET /api/v1/drift/timeline
class DriftTimelineHandler : public Handler {
public:
    HttpResponse Handle(const HttpRequest& request,
                        const HandlerContext& context) override;

    std::string GetRoute() const override { return "/api/v1/drift/timeline"; }
    std::vector<HttpMethod> GetMethods() const override {
        return {HttpMethod::kGet};
    }
};

/// @brief Get drift heatmap data: GET /api/v1/drift/heatmap
class DriftHeatmapHandler : public Handler {
public:
    HttpResponse Handle(const HttpRequest& request,
                        const HandlerContext& context) override;

    std::string GetRoute() const override { return "/api/v1/drift/heatmap"; }
    std::vector<HttpMethod> GetMethods() const override {
        return {HttpMethod::kGet};
    }
};

/// @brief Update drift reference: POST /api/v1/drift/reference
class UpdateDriftReferenceHandler : public Handler {
public:
    HttpResponse Handle(const HttpRequest& request,
                        const HandlerContext& context) override;

    std::string GetRoute() const override { return "/api/v1/drift/reference"; }
    std::vector<HttpMethod> GetMethods() const override {
        return {HttpMethod::kPost};
    }
};

/// @brief Resolve drift alert: POST /api/v1/drift/alerts/:id/resolve
class ResolveDriftAlertHandler : public Handler {
public:
    HttpResponse Handle(const HttpRequest& request,
                        const HandlerContext& context) override;

    std::string GetRoute() const override { return "/api/v1/drift/alerts/:id/resolve"; }
    std::vector<HttpMethod> GetMethods() const override {
        return {HttpMethod::kPost};
    }
};

/// @brief Register all drift handlers
void RegisterDriftHandlers(HandlerRegistry& registry);

}  // namespace pyflare::query::handlers
