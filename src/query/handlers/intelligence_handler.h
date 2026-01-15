#pragma once

/// @file intelligence_handler.h
/// @brief REST API handler for intelligence operations
///
/// Provides HTTP endpoints for:
/// - Intelligence analysis
/// - Drift detection
/// - RCA operations
/// - Model health monitoring

#include <memory>
#include <string>

#include <absl/status/status.h>
#include <absl/status/statusor.h>

#include "processor/intelligence/intelligence_pipeline.h"

namespace pyflare::query {

/// @brief HTTP request representation
struct HttpRequest {
    std::string method;
    std::string path;
    std::string body;
    std::unordered_map<std::string, std::string> headers;
    std::unordered_map<std::string, std::string> query_params;
    std::unordered_map<std::string, std::string> path_params;
};

/// @brief HTTP response representation
struct HttpResponse {
    int status_code = 200;
    std::string body;
    std::unordered_map<std::string, std::string> headers;

    static HttpResponse Ok(const std::string& body) {
        HttpResponse resp;
        resp.status_code = 200;
        resp.body = body;
        resp.headers["Content-Type"] = "application/json";
        return resp;
    }

    static HttpResponse Created(const std::string& body) {
        HttpResponse resp;
        resp.status_code = 201;
        resp.body = body;
        resp.headers["Content-Type"] = "application/json";
        return resp;
    }

    static HttpResponse BadRequest(const std::string& message) {
        HttpResponse resp;
        resp.status_code = 400;
        resp.body = R"({"error":")" + message + R"("})";
        resp.headers["Content-Type"] = "application/json";
        return resp;
    }

    static HttpResponse NotFound(const std::string& message) {
        HttpResponse resp;
        resp.status_code = 404;
        resp.body = R"({"error":")" + message + R"("})";
        resp.headers["Content-Type"] = "application/json";
        return resp;
    }

    static HttpResponse InternalError(const std::string& message) {
        HttpResponse resp;
        resp.status_code = 500;
        resp.body = R"({"error":")" + message + R"("})";
        resp.headers["Content-Type"] = "application/json";
        return resp;
    }
};

/// @brief Intelligence API handler
///
/// Handles REST API requests for intelligence operations.
///
/// Endpoints:
/// - POST /api/v1/intelligence/analyze - Analyze single inference
/// - POST /api/v1/intelligence/analyze/batch - Batch analysis
/// - GET /api/v1/intelligence/health - System health
/// - GET /api/v1/intelligence/health/:model_id - Model health
/// - GET /api/v1/intelligence/drift/:model_id - Drift status
/// - POST /api/v1/intelligence/rca/:model_id - Trigger RCA
/// - GET /api/v1/intelligence/rca/reports/:report_id - Get RCA report
/// - GET /api/v1/intelligence/stats - Pipeline statistics
///
/// Example:
/// @code
///   IntelligenceHandler handler(pipeline);
///
///   // In HTTP server route handler
///   HttpRequest req;
///   req.method = "POST";
///   req.path = "/api/v1/intelligence/analyze";
///   req.body = R"({"input":"...", "output":"..."})";
///
///   auto resp = handler.Handle(req);
/// @endcode
class IntelligenceHandler {
public:
    explicit IntelligenceHandler(
        std::shared_ptr<intelligence::IntelligencePipeline> pipeline);
    ~IntelligenceHandler();

    // Disable copy
    IntelligenceHandler(const IntelligenceHandler&) = delete;
    IntelligenceHandler& operator=(const IntelligenceHandler&) = delete;

    /// @brief Handle HTTP request
    HttpResponse Handle(const HttpRequest& request);

    /// @brief Get handler name
    std::string Name() const { return "intelligence"; }

    /// @brief Get base path
    std::string BasePath() const { return "/api/v1/intelligence"; }

private:
    // Route handlers
    HttpResponse HandleAnalyze(const HttpRequest& request);
    HttpResponse HandleAnalyzeBatch(const HttpRequest& request);
    HttpResponse HandleSystemHealth(const HttpRequest& request);
    HttpResponse HandleModelHealth(const HttpRequest& request);
    HttpResponse HandleDriftStatus(const HttpRequest& request);
    HttpResponse HandleTriggerRCA(const HttpRequest& request);
    HttpResponse HandleGetRCAReport(const HttpRequest& request);
    HttpResponse HandleListRCAReports(const HttpRequest& request);
    HttpResponse HandleStats(const HttpRequest& request);

    // Model management
    HttpResponse HandleRegisterModel(const HttpRequest& request);
    HttpResponse HandleListModels(const HttpRequest& request);
    HttpResponse HandleRemoveModel(const HttpRequest& request);

    // Evaluator management
    HttpResponse HandleListEvaluators(const HttpRequest& request);
    HttpResponse HandleSetEvaluatorEnabled(const HttpRequest& request);

    // Parse inference record from JSON
    absl::StatusOr<eval::InferenceRecord> ParseInferenceRecord(
        const std::string& json);

    // Parse batch records from JSON
    absl::StatusOr<std::vector<eval::InferenceRecord>> ParseBatchRecords(
        const std::string& json);

    std::shared_ptr<intelligence::IntelligencePipeline> pipeline_;
};

/// @brief Create intelligence handler
std::unique_ptr<IntelligenceHandler> CreateIntelligenceHandler(
    std::shared_ptr<intelligence::IntelligencePipeline> pipeline);

}  // namespace pyflare::query
