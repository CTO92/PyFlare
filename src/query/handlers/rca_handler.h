#pragma once

/// @file rca_handler.h
/// @brief Root Cause Analysis query handlers

#include "query/handlers/handler_base.h"

namespace pyflare::query::handlers {

/// @brief RCA report data
struct RCAReportData {
    std::string id;
    std::string model_id;
    std::string analysis_type;  ///< "failure_analysis", "performance_analysis"

    std::string timestamp;
    std::string status;  ///< "running", "completed", "failed"

    size_t traces_analyzed = 0;
    size_t patterns_found = 0;
    size_t clusters_found = 0;

    nlohmann::json patterns;
    nlohmann::json clusters;
    nlohmann::json problematic_slices;
    nlohmann::json recommendations;
};

/// @brief Run RCA analysis: POST /api/v1/rca/analyze
class RunRCAHandler : public Handler {
public:
    HttpResponse Handle(const HttpRequest& request,
                        const HandlerContext& context) override;

    std::string GetRoute() const override { return "/api/v1/rca/analyze"; }
    std::vector<HttpMethod> GetMethods() const override {
        return {HttpMethod::kPost};
    }
};

/// @brief Get RCA report: GET /api/v1/rca/reports/:id
class GetRCAReportHandler : public Handler {
public:
    HttpResponse Handle(const HttpRequest& request,
                        const HandlerContext& context) override;

    std::string GetRoute() const override { return "/api/v1/rca/reports/:id"; }
    std::vector<HttpMethod> GetMethods() const override {
        return {HttpMethod::kGet};
    }
};

/// @brief List RCA reports: GET /api/v1/rca/reports
class ListRCAReportsHandler : public Handler {
public:
    HttpResponse Handle(const HttpRequest& request,
                        const HandlerContext& context) override;

    std::string GetRoute() const override { return "/api/v1/rca/reports"; }
    std::vector<HttpMethod> GetMethods() const override {
        return {HttpMethod::kGet};
    }
};

/// @brief Get failure patterns: GET /api/v1/rca/patterns
class FailurePatternsHandler : public Handler {
public:
    HttpResponse Handle(const HttpRequest& request,
                        const HandlerContext& context) override;

    std::string GetRoute() const override { return "/api/v1/rca/patterns"; }
    std::vector<HttpMethod> GetMethods() const override {
        return {HttpMethod::kGet};
    }
};

/// @brief Get failure clusters: GET /api/v1/rca/clusters
class FailureClustersHandler : public Handler {
public:
    HttpResponse Handle(const HttpRequest& request,
                        const HandlerContext& context) override;

    std::string GetRoute() const override { return "/api/v1/rca/clusters"; }
    std::vector<HttpMethod> GetMethods() const override {
        return {HttpMethod::kGet};
    }
};

/// @brief Get problematic slices: GET /api/v1/rca/slices
class ProblematicSlicesHandler : public Handler {
public:
    HttpResponse Handle(const HttpRequest& request,
                        const HandlerContext& context) override;

    std::string GetRoute() const override { return "/api/v1/rca/slices"; }
    std::vector<HttpMethod> GetMethods() const override {
        return {HttpMethod::kGet};
    }
};

/// @brief Get slice detail: GET /api/v1/rca/slices/:id
class SliceDetailHandler : public Handler {
public:
    HttpResponse Handle(const HttpRequest& request,
                        const HandlerContext& context) override;

    std::string GetRoute() const override { return "/api/v1/rca/slices/:id"; }
    std::vector<HttpMethod> GetMethods() const override {
        return {HttpMethod::kGet};
    }
};

/// @brief Register all RCA handlers
void RegisterRCAHandlers(HandlerRegistry& registry);

}  // namespace pyflare::query::handlers
