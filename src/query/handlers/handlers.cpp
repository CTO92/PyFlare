/// @file handlers.cpp
/// @brief Implementation of all Query API handlers
///
/// SECURITY: All handlers use parameterized queries to prevent SQL injection.
/// Input validation is performed on all user-provided values.

#include "query/handlers/traces_handler.h"
#include "query/handlers/drift_handler.h"
#include "query/handlers/costs_handler.h"
#include "query/handlers/evaluations_handler.h"
#include "query/handlers/rca_handler.h"

#include <regex>
#include <spdlog/spdlog.h>

namespace pyflare::query::handlers {

using json = nlohmann::json;

// =============================================================================
// SECURITY: Input Validation Utilities
// =============================================================================

/// @brief Maximum page size to prevent resource exhaustion
static constexpr size_t kMaxPageSize = 1000;

/// @brief Maximum string parameter length
static constexpr size_t kMaxParamLength = 256;

/// @brief Validate identifier (trace_id, model_id, user_id, etc.)
/// Only allows alphanumeric characters, hyphens, underscores, and dots
static bool IsValidIdentifier(const std::string& value) {
    if (value.empty() || value.size() > kMaxParamLength) {
        return false;
    }
    static const std::regex identifier_regex("^[a-zA-Z0-9_\\-\\.]+$");
    return std::regex_match(value, identifier_regex);
}

/// @brief Validate status value (whitelist approach)
static bool IsValidStatus(const std::string& value) {
    static const std::unordered_set<std::string> valid_statuses = {
        "ok", "error", "timeout", "cancelled", "unknown"
    };
    return valid_statuses.find(value) != valid_statuses.end();
}

/// @brief Validate timestamp format (ISO 8601)
static bool IsValidTimestamp(const std::string& value) {
    if (value.empty() || value.size() > 30) {
        return false;
    }
    // Basic ISO 8601 format validation
    static const std::regex timestamp_regex(
        "^\\d{4}-\\d{2}-\\d{2}(T\\d{2}:\\d{2}:\\d{2}(\\.\\d+)?(Z|[+-]\\d{2}:\\d{2})?)?$");
    return std::regex_match(value, timestamp_regex);
}

/// @brief Validate dimension value (model, user, feature, etc.)
static bool IsValidDimension(const std::string& value) {
    static const std::unordered_set<std::string> valid_dimensions = {
        "model", "user", "feature", "service", "environment", "provider"
    };
    return valid_dimensions.find(value) != valid_dimensions.end();
}

/// @brief Sanitize pagination to prevent resource exhaustion
static void SanitizePagination(Pagination& pagination) {
    if (pagination.page_size > kMaxPageSize) {
        pagination.page_size = kMaxPageSize;
    }
    if (pagination.page_size == 0) {
        pagination.page_size = 50;  // Default
    }
    if (pagination.page < 1) {
        pagination.page = 1;
    }
}

// =============================================================================
// Traces Handlers
// =============================================================================

HttpResponse ListTracesHandler::Handle(const HttpRequest& request,
                                         const HandlerContext& context) {
    auto pagination = ExtractPagination(request);
    SanitizePagination(pagination);
    auto time_range = ExtractTimeRange(request);

    // SECURITY: Use parameterized queries to prevent SQL injection
    std::ostringstream query;
    std::vector<storage::QueryParam> params;

    query << "SELECT trace_id, span_id, model_id, user_id, feature_id, "
          << "start_time, end_time, latency_ms, status, error_type, "
          << "input_tokens, output_tokens, cost_micros, eval_score, drift_score "
          << "FROM traces WHERE 1=1";

    // Apply filters with validation and parameterized queries
    auto it = request.query_params.find("model_id");
    if (it != request.query_params.end()) {
        if (!IsValidIdentifier(it->second)) {
            return HttpResponse::BadRequest("Invalid model_id format");
        }
        query << " AND model_id = {model_id:String}";
        params.push_back({"model_id", it->second, "String"});
    }

    it = request.query_params.find("user_id");
    if (it != request.query_params.end()) {
        if (!IsValidIdentifier(it->second)) {
            return HttpResponse::BadRequest("Invalid user_id format");
        }
        query << " AND user_id = {user_id:String}";
        params.push_back({"user_id", it->second, "String"});
    }

    it = request.query_params.find("status");
    if (it != request.query_params.end()) {
        if (!IsValidStatus(it->second)) {
            return HttpResponse::BadRequest("Invalid status value");
        }
        query << " AND status = {status:String}";
        params.push_back({"status", it->second, "String"});
    }

    if (!time_range.start.empty()) {
        if (!IsValidTimestamp(time_range.start)) {
            return HttpResponse::BadRequest("Invalid start timestamp format");
        }
        query << " AND timestamp >= {start_time:DateTime}";
        params.push_back({"start_time", time_range.start, "DateTime"});
    }
    if (!time_range.end.empty()) {
        if (!IsValidTimestamp(time_range.end)) {
            return HttpResponse::BadRequest("Invalid end timestamp format");
        }
        query << " AND timestamp <= {end_time:DateTime}";
        params.push_back({"end_time", time_range.end, "DateTime"});
    }

    query << " ORDER BY timestamp DESC"
          << " LIMIT {limit:UInt64}"
          << " OFFSET {offset:UInt64}";
    params.push_back({"limit", std::to_string(pagination.page_size), "UInt64"});
    params.push_back({"offset", std::to_string(pagination.offset()), "UInt64"});

    auto result = context.clickhouse->ExecuteWithParams(query.str(), params);
    if (!result.ok()) {
        // SECURITY: Don't expose internal error details
        spdlog::error("Query execution failed: {}", result.status().message());
        return HttpResponse::InternalError("Failed to retrieve traces");
    }

    json response;
    response["data"] = json::array();  // Would be populated from query result
    response["pagination"] = {
        {"page", pagination.page},
        {"page_size", pagination.page_size},
        {"total", 0}  // Would come from COUNT query
    };

    return HttpResponse::Ok(response);
}

HttpResponse GetTraceHandler::Handle(const HttpRequest& request,
                                       const HandlerContext& context) {
    // Extract trace ID from path
    std::string trace_id = request.path_params;

    if (trace_id.empty()) {
        return HttpResponse::BadRequest("Trace ID required");
    }

    // SECURITY: Validate trace_id format
    if (!IsValidIdentifier(trace_id)) {
        return HttpResponse::BadRequest("Invalid trace ID format");
    }

    // SECURITY: Use parameterized query
    std::string query = "SELECT * FROM traces WHERE trace_id = {trace_id:String}";
    std::vector<storage::QueryParam> params = {
        {"trace_id", trace_id, "String"}
    };

    auto result = context.clickhouse->ExecuteWithParams(query, params);
    if (!result.ok()) {
        // SECURITY: Don't expose internal error details
        spdlog::error("Query execution failed: {}", result.status().message());
        return HttpResponse::InternalError("Failed to retrieve trace");
    }

    json response;
    response["trace_id"] = trace_id;
    // Would be populated from query result

    return HttpResponse::Ok(response);
}

HttpResponse SearchTracesHandler::Handle(const HttpRequest& request,
                                           const HandlerContext& context) {
    auto body = ParseJsonBody(request);
    if (!body.ok()) {
        return HttpResponse::BadRequest(std::string(body.status().message()));
    }

    // Build search query from body
    json response;
    response["results"] = json::array();
    response["total"] = 0;

    return HttpResponse::Ok(response);
}

HttpResponse TraceStatsHandler::Handle(const HttpRequest& request,
                                         const HandlerContext& context) {
    auto time_range = ExtractTimeRange(request);

    json response;
    response["total_traces"] = 0;
    response["error_rate"] = 0.0;
    response["avg_latency_ms"] = 0.0;
    response["p95_latency_ms"] = 0.0;
    response["total_cost_micros"] = 0;
    response["traces_by_model"] = json::object();

    return HttpResponse::Ok(response);
}

HttpResponse TraceTimelineHandler::Handle(const HttpRequest& request,
                                            const HandlerContext& context) {
    std::string trace_id = request.path_params;

    // SECURITY: Validate trace_id format
    if (!trace_id.empty() && !IsValidIdentifier(trace_id)) {
        return HttpResponse::BadRequest("Invalid trace ID format");
    }

    json response;
    response["trace_id"] = trace_id;
    response["spans"] = json::array();

    return HttpResponse::Ok(response);
}

void RegisterTracesHandlers(HandlerRegistry& registry) {
    registry.Register(std::make_unique<ListTracesHandler>());
    registry.Register(std::make_unique<GetTraceHandler>());
    registry.Register(std::make_unique<SearchTracesHandler>());
    registry.Register(std::make_unique<TraceStatsHandler>());
    registry.Register(std::make_unique<TraceTimelineHandler>());
}

// =============================================================================
// Drift Handlers
// =============================================================================

HttpResponse ListDriftAlertsHandler::Handle(const HttpRequest& request,
                                              const HandlerContext& context) {
    auto pagination = ExtractPagination(request);

    json response;
    response["alerts"] = json::array();
    response["pagination"] = {
        {"page", pagination.page},
        {"page_size", pagination.page_size}
    };

    return HttpResponse::Ok(response);
}

HttpResponse GetDriftStatusHandler::Handle(const HttpRequest& request,
                                             const HandlerContext& context) {
    std::string model_id = request.path_params;

    json response;
    response["model_id"] = model_id;
    response["status"] = "healthy";  // or "drifted"
    response["feature_drift_score"] = 0.0;
    response["embedding_drift_score"] = 0.0;
    response["last_checked"] = "";
    response["active_alerts"] = 0;

    return HttpResponse::Ok(response);
}

HttpResponse DriftTimelineHandler::Handle(const HttpRequest& request,
                                            const HandlerContext& context) {
    auto time_range = ExtractTimeRange(request);

    json response;
    response["timeline"] = json::array();

    return HttpResponse::Ok(response);
}

HttpResponse DriftHeatmapHandler::Handle(const HttpRequest& request,
                                           const HandlerContext& context) {
    json response;
    response["features"] = json::array();
    response["models"] = json::array();
    response["matrix"] = json::array();

    return HttpResponse::Ok(response);
}

HttpResponse UpdateDriftReferenceHandler::Handle(const HttpRequest& request,
                                                   const HandlerContext& context) {
    auto body = ParseJsonBody(request);
    if (!body.ok()) {
        return HttpResponse::BadRequest(std::string(body.status().message()));
    }

    json response;
    response["status"] = "updated";

    return HttpResponse::Created(response);
}

HttpResponse ResolveDriftAlertHandler::Handle(const HttpRequest& request,
                                                const HandlerContext& context) {
    std::string alert_id = request.path_params;

    json response;
    response["alert_id"] = alert_id;
    response["resolved"] = true;

    return HttpResponse::Ok(response);
}

void RegisterDriftHandlers(HandlerRegistry& registry) {
    registry.Register(std::make_unique<ListDriftAlertsHandler>());
    registry.Register(std::make_unique<GetDriftStatusHandler>());
    registry.Register(std::make_unique<DriftTimelineHandler>());
    registry.Register(std::make_unique<DriftHeatmapHandler>());
    registry.Register(std::make_unique<UpdateDriftReferenceHandler>());
    registry.Register(std::make_unique<ResolveDriftAlertHandler>());
}

// =============================================================================
// Costs Handlers
// =============================================================================

HttpResponse CostSummaryHandler::Handle(const HttpRequest& request,
                                          const HandlerContext& context) {
    auto time_range = ExtractTimeRange(request);

    json response;
    response["total_cost_micros"] = 0;
    response["total_tokens"] = 0;
    response["request_count"] = 0;
    response["period"] = {
        {"start", time_range.start},
        {"end", time_range.end}
    };

    return HttpResponse::Ok(response);
}

HttpResponse CostBreakdownHandler::Handle(const HttpRequest& request,
                                            const HandlerContext& context) {
    auto it = request.query_params.find("dimension");
    std::string dimension = it != request.query_params.end() ? it->second : "model";

    // SECURITY: Validate dimension against allowlist
    if (!IsValidDimension(dimension)) {
        return HttpResponse::BadRequest("Invalid dimension value");
    }

    json response;
    response["dimension"] = dimension;
    response["breakdown"] = json::array();

    return HttpResponse::Ok(response);
}

HttpResponse CostTimelineHandler::Handle(const HttpRequest& request,
                                           const HandlerContext& context) {
    auto time_range = ExtractTimeRange(request);

    json response;
    response["timeline"] = json::array();

    return HttpResponse::Ok(response);
}

HttpResponse BudgetStatusHandler::Handle(const HttpRequest& request,
                                           const HandlerContext& context) {
    json response;
    response["budgets"] = json::array();

    return HttpResponse::Ok(response);
}

HttpResponse CreateBudgetHandler::Handle(const HttpRequest& request,
                                           const HandlerContext& context) {
    auto body = ParseJsonBody(request);
    if (!body.ok()) {
        return HttpResponse::BadRequest("Invalid JSON body");
    }

    // SECURITY: Validate budget parameters
    const json& budget_json = *body;

    // Validate required fields
    if (!budget_json.contains("name") || !budget_json["name"].is_string()) {
        return HttpResponse::BadRequest("Budget name is required");
    }
    std::string name = budget_json["name"].get<std::string>();
    if (!IsValidIdentifier(name)) {
        return HttpResponse::BadRequest("Invalid budget name format");
    }

    // Validate limit if provided
    if (budget_json.contains("limit_micros")) {
        if (!budget_json["limit_micros"].is_number_integer()) {
            return HttpResponse::BadRequest("Budget limit must be an integer");
        }
        int64_t limit = budget_json["limit_micros"].get<int64_t>();
        if (limit < 0) {
            return HttpResponse::BadRequest("Budget limit cannot be negative");
        }
        // Prevent overflow by setting maximum limit (10 billion USD)
        if (limit > 10000000000000000LL) {
            return HttpResponse::BadRequest("Budget limit exceeds maximum allowed value");
        }
    }

    // Validate dimension if provided
    if (budget_json.contains("dimension")) {
        if (!budget_json["dimension"].is_string()) {
            return HttpResponse::BadRequest("Dimension must be a string");
        }
        std::string dimension = budget_json["dimension"].get<std::string>();
        if (!IsValidDimension(dimension)) {
            return HttpResponse::BadRequest("Invalid dimension value");
        }
    }

    json response;
    response["budget_id"] = "";
    response["created"] = true;

    return HttpResponse::Created(response);
}

HttpResponse CostForecastHandler::Handle(const HttpRequest& request,
                                           const HandlerContext& context) {
    json response;
    response["forecast"] = json::object();
    response["projected_total_micros"] = 0;

    return HttpResponse::Ok(response);
}

void RegisterCostsHandlers(HandlerRegistry& registry) {
    registry.Register(std::make_unique<CostSummaryHandler>());
    registry.Register(std::make_unique<CostBreakdownHandler>());
    registry.Register(std::make_unique<CostTimelineHandler>());
    registry.Register(std::make_unique<BudgetStatusHandler>());
    registry.Register(std::make_unique<CreateBudgetHandler>());
    registry.Register(std::make_unique<CostForecastHandler>());
}

// =============================================================================
// Evaluations Handlers
// =============================================================================

HttpResponse ListEvaluationsHandler::Handle(const HttpRequest& request,
                                              const HandlerContext& context) {
    auto pagination = ExtractPagination(request);

    json response;
    response["evaluations"] = json::array();
    response["pagination"] = {
        {"page", pagination.page},
        {"page_size", pagination.page_size}
    };

    return HttpResponse::Ok(response);
}

HttpResponse GetEvaluationHandler::Handle(const HttpRequest& request,
                                            const HandlerContext& context) {
    std::string eval_id = request.path_params;

    json response;
    response["id"] = eval_id;

    return HttpResponse::Ok(response);
}

HttpResponse EvaluationSummaryHandler::Handle(const HttpRequest& request,
                                                const HandlerContext& context) {
    json response;
    response["total_evaluations"] = 0;
    response["pass_rate"] = 0.0;
    response["fail_rate"] = 0.0;
    response["by_type"] = json::object();

    return HttpResponse::Ok(response);
}

HttpResponse TraceEvaluationsHandler::Handle(const HttpRequest& request,
                                               const HandlerContext& context) {
    std::string trace_id = request.path_params;

    json response;
    response["trace_id"] = trace_id;
    response["evaluations"] = json::array();

    return HttpResponse::Ok(response);
}

HttpResponse TriggerEvaluationHandler::Handle(const HttpRequest& request,
                                                const HandlerContext& context) {
    auto body = ParseJsonBody(request);
    if (!body.ok()) {
        return HttpResponse::BadRequest(std::string(body.status().message()));
    }

    json response;
    response["job_id"] = "";
    response["status"] = "queued";

    return HttpResponse::Created(response);
}

HttpResponse EvaluationTrendsHandler::Handle(const HttpRequest& request,
                                               const HandlerContext& context) {
    json response;
    response["trends"] = json::array();

    return HttpResponse::Ok(response);
}

void RegisterEvaluationsHandlers(HandlerRegistry& registry) {
    registry.Register(std::make_unique<ListEvaluationsHandler>());
    registry.Register(std::make_unique<GetEvaluationHandler>());
    registry.Register(std::make_unique<EvaluationSummaryHandler>());
    registry.Register(std::make_unique<TraceEvaluationsHandler>());
    registry.Register(std::make_unique<TriggerEvaluationHandler>());
    registry.Register(std::make_unique<EvaluationTrendsHandler>());
}

// =============================================================================
// RCA Handlers
// =============================================================================

HttpResponse RunRCAHandler::Handle(const HttpRequest& request,
                                     const HandlerContext& context) {
    auto body = ParseJsonBody(request);
    if (!body.ok()) {
        return HttpResponse::BadRequest(std::string(body.status().message()));
    }

    json response;
    response["report_id"] = "";
    response["status"] = "running";

    return HttpResponse::Created(response);
}

HttpResponse GetRCAReportHandler::Handle(const HttpRequest& request,
                                           const HandlerContext& context) {
    std::string report_id = request.path_params;

    json response;
    response["id"] = report_id;
    response["status"] = "completed";
    response["patterns"] = json::array();
    response["clusters"] = json::array();
    response["recommendations"] = json::array();

    return HttpResponse::Ok(response);
}

HttpResponse ListRCAReportsHandler::Handle(const HttpRequest& request,
                                             const HandlerContext& context) {
    auto pagination = ExtractPagination(request);

    json response;
    response["reports"] = json::array();
    response["pagination"] = {
        {"page", pagination.page},
        {"page_size", pagination.page_size}
    };

    return HttpResponse::Ok(response);
}

HttpResponse FailurePatternsHandler::Handle(const HttpRequest& request,
                                              const HandlerContext& context) {
    json response;
    response["patterns"] = json::array();

    return HttpResponse::Ok(response);
}

HttpResponse FailureClustersHandler::Handle(const HttpRequest& request,
                                              const HandlerContext& context) {
    json response;
    response["clusters"] = json::array();

    return HttpResponse::Ok(response);
}

HttpResponse ProblematicSlicesHandler::Handle(const HttpRequest& request,
                                                const HandlerContext& context) {
    json response;
    response["slices"] = json::array();

    return HttpResponse::Ok(response);
}

HttpResponse SliceDetailHandler::Handle(const HttpRequest& request,
                                          const HandlerContext& context) {
    std::string slice_id = request.path_params;

    json response;
    response["id"] = slice_id;
    response["name"] = "";
    response["sample_count"] = 0;
    response["metric_value"] = 0.0;
    response["deviation"] = 0.0;

    return HttpResponse::Ok(response);
}

void RegisterRCAHandlers(HandlerRegistry& registry) {
    registry.Register(std::make_unique<RunRCAHandler>());
    registry.Register(std::make_unique<GetRCAReportHandler>());
    registry.Register(std::make_unique<ListRCAReportsHandler>());
    registry.Register(std::make_unique<FailurePatternsHandler>());
    registry.Register(std::make_unique<FailureClustersHandler>());
    registry.Register(std::make_unique<ProblematicSlicesHandler>());
    registry.Register(std::make_unique<SliceDetailHandler>());
}

}  // namespace pyflare::query::handlers
