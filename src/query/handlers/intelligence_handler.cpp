/// @file intelligence_handler.cpp
/// @brief Intelligence API handler implementation

#include "query/handlers/intelligence_handler.h"

#include <nlohmann/json.hpp>

namespace pyflare::query {

using json = nlohmann::json;

IntelligenceHandler::IntelligenceHandler(
    std::shared_ptr<intelligence::IntelligencePipeline> pipeline)
    : pipeline_(std::move(pipeline)) {}

IntelligenceHandler::~IntelligenceHandler() = default;

HttpResponse IntelligenceHandler::Handle(const HttpRequest& request) {
    std::string path = request.path;

    // Remove base path
    if (path.find(BasePath()) == 0) {
        path = path.substr(BasePath().length());
    }

    // Route requests
    if (request.method == "POST" && path == "/analyze") {
        return HandleAnalyze(request);
    }
    if (request.method == "POST" && path == "/analyze/batch") {
        return HandleAnalyzeBatch(request);
    }
    if (request.method == "GET" && path == "/health") {
        return HandleSystemHealth(request);
    }
    if (request.method == "GET" && path.find("/health/") == 0) {
        return HandleModelHealth(request);
    }
    if (request.method == "GET" && path.find("/drift/") == 0) {
        return HandleDriftStatus(request);
    }
    if (request.method == "POST" && path.find("/rca/") == 0 &&
        path.find("/reports") == std::string::npos) {
        return HandleTriggerRCA(request);
    }
    if (request.method == "GET" && path.find("/rca/reports/") == 0) {
        return HandleGetRCAReport(request);
    }
    if (request.method == "GET" && path.find("/rca/") == 0 &&
        path.find("/reports") != std::string::npos) {
        return HandleListRCAReports(request);
    }
    if (request.method == "GET" && path == "/stats") {
        return HandleStats(request);
    }
    if (request.method == "POST" && path == "/models") {
        return HandleRegisterModel(request);
    }
    if (request.method == "GET" && path == "/models") {
        return HandleListModels(request);
    }
    if (request.method == "DELETE" && path.find("/models/") == 0) {
        return HandleRemoveModel(request);
    }
    if (request.method == "GET" && path == "/evaluators") {
        return HandleListEvaluators(request);
    }
    if (request.method == "PUT" && path.find("/evaluators/") == 0) {
        return HandleSetEvaluatorEnabled(request);
    }

    return HttpResponse::NotFound("Endpoint not found");
}

HttpResponse IntelligenceHandler::HandleAnalyze(const HttpRequest& request) {
    auto record = ParseInferenceRecord(request.body);
    if (!record.ok()) {
        return HttpResponse::BadRequest(std::string(record.status().message()));
    }

    auto result = pipeline_->Process(*record);

    // Serialize result
    std::string result_json = intelligence::SerializeIntelligenceResult(result);

    return HttpResponse::Ok(result_json);
}

HttpResponse IntelligenceHandler::HandleAnalyzeBatch(const HttpRequest& request) {
    auto records = ParseBatchRecords(request.body);
    if (!records.ok()) {
        return HttpResponse::BadRequest(std::string(records.status().message()));
    }

    auto batch_result = pipeline_->ProcessBatch(*records);

    // Serialize batch result
    json j;
    j["total_processed"] = batch_result.total_processed;
    j["drift_detected_count"] = batch_result.drift_detected_count;
    j["safety_issues_count"] = batch_result.safety_issues_count;
    j["evaluation_failures"] = batch_result.evaluation_failures;
    j["avg_health_score"] = batch_result.avg_health_score;
    j["avg_processing_time_ms"] = batch_result.avg_processing_time_ms;

    json results_array = json::array();
    for (const auto& result : batch_result.results) {
        results_array.push_back(
            json::parse(intelligence::SerializeIntelligenceResult(result)));
    }
    j["results"] = results_array;

    return HttpResponse::Ok(j.dump());
}

HttpResponse IntelligenceHandler::HandleSystemHealth(const HttpRequest& request) {
    auto health = pipeline_->GetSystemHealth();

    json j;
    j["overall_health"] = health.overall_health;
    j["models_with_drift"] = health.models_with_drift;
    j["total_active_alerts"] = health.total_active_alerts;
    j["models_analyzed"] = health.models_analyzed;
    j["avg_health_score"] = health.avg_health_score;
    j["last_update"] = std::chrono::duration_cast<std::chrono::seconds>(
        health.last_update.time_since_epoch()).count();

    return HttpResponse::Ok(j.dump());
}

HttpResponse IntelligenceHandler::HandleModelHealth(const HttpRequest& request) {
    // Extract model_id from path
    std::string path = request.path;
    size_t pos = path.rfind('/');
    if (pos == std::string::npos) {
        return HttpResponse::BadRequest("Invalid path");
    }
    std::string model_id = path.substr(pos + 1);

    auto health = pipeline_->GetModelHealth(model_id);
    if (!health.ok()) {
        return HttpResponse::NotFound(std::string(health.status().message()));
    }

    json j;
    j["model_id"] = health->model_id;
    j["health_score"] = health->health_score;
    j["has_active_drift"] = health->has_active_drift;
    j["active_alerts"] = health->active_alerts;
    j["recent_safety_issues"] = health->recent_safety_issues;
    j["avg_evaluation_score"] = health->avg_evaluation_score;
    j["last_analyzed"] = std::chrono::duration_cast<std::chrono::seconds>(
        health->last_analyzed.time_since_epoch()).count();

    return HttpResponse::Ok(j.dump());
}

HttpResponse IntelligenceHandler::HandleDriftStatus(const HttpRequest& request) {
    // Extract model_id from path
    std::string path = request.path;
    size_t pos = path.rfind('/');
    if (pos == std::string::npos) {
        return HttpResponse::BadRequest("Invalid path");
    }
    std::string model_id = path.substr(pos + 1);

    auto health = pipeline_->GetModelHealth(model_id);
    if (!health.ok()) {
        return HttpResponse::NotFound(std::string(health.status().message()));
    }

    json j;
    j["model_id"] = model_id;
    j["has_drift"] = health->has_active_drift;
    j["health_score"] = health->health_score;

    return HttpResponse::Ok(j.dump());
}

HttpResponse IntelligenceHandler::HandleTriggerRCA(const HttpRequest& request) {
    // Extract model_id from path
    std::string path = request.path;
    size_t pos = path.rfind('/');
    if (pos == std::string::npos) {
        return HttpResponse::BadRequest("Invalid path");
    }
    std::string model_id = path.substr(pos + 1);

    auto report = pipeline_->TriggerRCA(model_id);
    if (!report.ok()) {
        return HttpResponse::InternalError(std::string(report.status().message()));
    }

    std::string report_json = rca::SerializeReport(*report);
    return HttpResponse::Created(report_json);
}

HttpResponse IntelligenceHandler::HandleGetRCAReport(const HttpRequest& request) {
    // Extract report_id from path
    std::string path = request.path;
    size_t pos = path.rfind('/');
    if (pos == std::string::npos) {
        return HttpResponse::BadRequest("Invalid path");
    }
    std::string report_id = path.substr(pos + 1);

    auto report = pipeline_->GetRCAReport(report_id);
    if (!report.ok()) {
        return HttpResponse::NotFound(std::string(report.status().message()));
    }

    std::string report_json = rca::SerializeReport(*report);
    return HttpResponse::Ok(report_json);
}

HttpResponse IntelligenceHandler::HandleListRCAReports(const HttpRequest& request) {
    // Extract model_id from path: /rca/{model_id}/reports
    std::string path = request.path;
    size_t rca_pos = path.find("/rca/");
    size_t reports_pos = path.find("/reports");

    if (rca_pos == std::string::npos || reports_pos == std::string::npos) {
        return HttpResponse::BadRequest("Invalid path");
    }

    std::string model_id = path.substr(rca_pos + 5, reports_pos - rca_pos - 5);

    size_t limit = 10;
    if (request.query_params.count("limit") > 0) {
        limit = std::stoul(request.query_params.at("limit"));
    }

    auto reports = pipeline_->ListRCAReports(model_id, limit);

    json j = json::array();
    for (const auto& report : reports) {
        j.push_back(json::parse(rca::SerializeReport(report)));
    }

    return HttpResponse::Ok(j.dump());
}

HttpResponse IntelligenceHandler::HandleStats(const HttpRequest& request) {
    auto stats = pipeline_->GetStats();

    json j;
    j["total_processed"] = stats.total_processed;
    j["drift_detections"] = stats.drift_detections;
    j["safety_issues"] = stats.safety_issues;
    j["evaluation_failures"] = stats.evaluation_failures;
    j["rca_triggered"] = stats.rca_triggered;
    j["alerts_generated"] = stats.alerts_generated;
    j["avg_processing_time_ms"] = stats.avg_processing_time_ms;
    j["p99_processing_time_ms"] = stats.p99_processing_time_ms;
    j["queue_depth"] = stats.queue_depth;

    j["component_health"] = {
        {"drift_service", stats.drift_service_healthy},
        {"eval_service", stats.eval_service_healthy},
        {"rca_service", stats.rca_service_healthy},
        {"alert_service", stats.alert_service_healthy}
    };

    if (stats.last_processed != std::chrono::system_clock::time_point{}) {
        j["last_processed"] = std::chrono::duration_cast<std::chrono::seconds>(
            stats.last_processed.time_since_epoch()).count();
    }

    return HttpResponse::Ok(j.dump());
}

HttpResponse IntelligenceHandler::HandleRegisterModel(const HttpRequest& request) {
    try {
        json j = json::parse(request.body);

        std::string model_id = j.value("model_id", "");
        if (model_id.empty()) {
            return HttpResponse::BadRequest("model_id is required");
        }

        drift::ReferenceData reference;
        // Parse reference data from JSON
        if (j.contains("reference")) {
            // Parse reference features, embeddings, etc.
        }

        auto status = pipeline_->RegisterModel(model_id, reference);
        if (!status.ok()) {
            return HttpResponse::InternalError(std::string(status.message()));
        }

        json response;
        response["model_id"] = model_id;
        response["status"] = "registered";

        return HttpResponse::Created(response.dump());
    } catch (const json::exception& e) {
        return HttpResponse::BadRequest(std::string("Invalid JSON: ") + e.what());
    }
}

HttpResponse IntelligenceHandler::HandleListModels(const HttpRequest& request) {
    auto models = pipeline_->ListModels();

    json j = json::array();
    for (const auto& model_id : models) {
        json model;
        model["model_id"] = model_id;

        auto health = pipeline_->GetModelHealth(model_id);
        if (health.ok()) {
            model["health_score"] = health->health_score;
            model["has_active_drift"] = health->has_active_drift;
        }

        j.push_back(model);
    }

    return HttpResponse::Ok(j.dump());
}

HttpResponse IntelligenceHandler::HandleRemoveModel(const HttpRequest& request) {
    // Extract model_id from path
    std::string path = request.path;
    size_t pos = path.rfind('/');
    if (pos == std::string::npos) {
        return HttpResponse::BadRequest("Invalid path");
    }
    std::string model_id = path.substr(pos + 1);

    auto status = pipeline_->RemoveModel(model_id);
    if (!status.ok()) {
        return HttpResponse::NotFound(std::string(status.message()));
    }

    json response;
    response["model_id"] = model_id;
    response["status"] = "removed";

    return HttpResponse::Ok(response.dump());
}

HttpResponse IntelligenceHandler::HandleListEvaluators(const HttpRequest& request) {
    auto evaluators = pipeline_->ListEvaluators();

    json j = json::array();
    for (const auto& name : evaluators) {
        json eval;
        eval["name"] = name;
        j.push_back(eval);
    }

    return HttpResponse::Ok(j.dump());
}

HttpResponse IntelligenceHandler::HandleSetEvaluatorEnabled(
    const HttpRequest& request) {

    // Extract evaluator name from path
    std::string path = request.path;
    size_t pos = path.rfind('/');
    if (pos == std::string::npos) {
        return HttpResponse::BadRequest("Invalid path");
    }
    std::string name = path.substr(pos + 1);

    try {
        json j = json::parse(request.body);
        bool enabled = j.value("enabled", true);

        auto status = pipeline_->SetEvaluatorEnabled(name, enabled);
        if (!status.ok()) {
            return HttpResponse::NotFound(std::string(status.message()));
        }

        json response;
        response["name"] = name;
        response["enabled"] = enabled;

        return HttpResponse::Ok(response.dump());
    } catch (const json::exception& e) {
        return HttpResponse::BadRequest(std::string("Invalid JSON: ") + e.what());
    }
}

absl::StatusOr<eval::InferenceRecord> IntelligenceHandler::ParseInferenceRecord(
    const std::string& json_str) {

    try {
        json j = json::parse(json_str);

        eval::InferenceRecord record;
        record.trace_id = j.value("trace_id", "");
        record.model_id = j.value("model_id", "");
        record.input = j.value("input", "");
        record.output = j.value("output", "");

        if (j.contains("expected_output")) {
            record.expected_output = j["expected_output"].get<std::string>();
        }

        if (j.contains("metadata")) {
            record.metadata = j["metadata"].get<std::unordered_map<std::string, std::string>>();
        }

        if (j.contains("timestamp")) {
            record.timestamp = std::chrono::system_clock::time_point(
                std::chrono::seconds(j["timestamp"].get<int64_t>()));
        } else {
            record.timestamp = std::chrono::system_clock::now();
        }

        return record;
    } catch (const json::exception& e) {
        return absl::InvalidArgumentError(
            std::string("Failed to parse inference record: ") + e.what());
    }
}

absl::StatusOr<std::vector<eval::InferenceRecord>>
IntelligenceHandler::ParseBatchRecords(const std::string& json_str) {

    try {
        json j = json::parse(json_str);

        if (!j.contains("records") || !j["records"].is_array()) {
            return absl::InvalidArgumentError(
                "Expected 'records' array in request body");
        }

        std::vector<eval::InferenceRecord> records;
        for (const auto& record_json : j["records"]) {
            auto record = ParseInferenceRecord(record_json.dump());
            if (!record.ok()) {
                return record.status();
            }
            records.push_back(*record);
        }

        return records;
    } catch (const json::exception& e) {
        return absl::InvalidArgumentError(
            std::string("Failed to parse batch records: ") + e.what());
    }
}

std::unique_ptr<IntelligenceHandler> CreateIntelligenceHandler(
    std::shared_ptr<intelligence::IntelligencePipeline> pipeline) {

    return std::make_unique<IntelligenceHandler>(std::move(pipeline));
}

}  // namespace pyflare::query
