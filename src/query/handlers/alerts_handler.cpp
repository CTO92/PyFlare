/// @file alerts_handler.cpp
/// @brief Alerts API handler implementation

#include "query/handlers/alerts_handler.h"

#include <nlohmann/json.hpp>

namespace pyflare::query {

using json = nlohmann::json;

AlertsHandler::AlertsHandler(
    std::shared_ptr<alerting::AlertService> alert_service)
    : alert_service_(std::move(alert_service)) {}

AlertsHandler::~AlertsHandler() = default;

HttpResponse AlertsHandler::Handle(const HttpRequest& request) {
    std::string path = request.path;

    // Remove base path
    if (path.find(BasePath()) == 0) {
        path = path.substr(BasePath().length());
    }

    // Handle empty path as list alerts
    if (path.empty() || path == "/") {
        if (request.method == "GET") {
            return HandleListAlerts(request);
        }
        if (request.method == "POST") {
            return HandleFireAlert(request);
        }
    }

    // Stats
    if (path == "/stats" && request.method == "GET") {
        return HandleStats(request);
    }

    // Groups
    if (path == "/groups" && request.method == "GET") {
        return HandleListGroups(request);
    }

    // Rules
    if (path == "/rules") {
        if (request.method == "GET") return HandleListRules(request);
        if (request.method == "POST") return HandleCreateRule(request);
    }
    if (path.find("/rules/") == 0) {
        std::string rule_id = path.substr(7);
        if (path.find("/enabled") != std::string::npos) {
            return HandleSetRuleEnabled(request);
        }
        if (request.method == "GET") return HandleGetRule(request);
        if (request.method == "PUT") return HandleUpdateRule(request);
        if (request.method == "DELETE") return HandleDeleteRule(request);
    }

    // Silences
    if (path == "/silences") {
        if (request.method == "GET") return HandleListSilences(request);
        if (request.method == "POST") return HandleCreateSilence(request);
    }
    if (path.find("/silences/") == 0 && request.method == "DELETE") {
        return HandleDeleteSilence(request);
    }

    // Maintenance
    if (path == "/maintenance") {
        if (request.method == "GET") return HandleListMaintenance(request);
        if (request.method == "POST") return HandleCreateMaintenance(request);
    }
    if (path.find("/maintenance/") == 0 && request.method == "DELETE") {
        return HandleDeleteMaintenance(request);
    }

    // Notification channels
    if (path == "/channels") {
        if (request.method == "GET") return HandleListChannels(request);
        if (request.method == "POST") return HandleCreateChannel(request);
    }
    if (path.find("/channels/") == 0) {
        if (path.find("/test") != std::string::npos) {
            return HandleTestChannel(request);
        }
        if (request.method == "DELETE") {
            return HandleDeleteChannel(request);
        }
    }

    // Alert-specific operations
    if (path.find("/") == 0 && path.length() > 1) {
        // Check for resolve/acknowledge
        if (path.find("/resolve") != std::string::npos) {
            return HandleResolveAlert(request);
        }
        if (path.find("/acknowledge") != std::string::npos) {
            return HandleAcknowledgeAlert(request);
        }
        // Get specific alert
        if (request.method == "GET") {
            return HandleGetAlert(request);
        }
    }

    return HttpResponse::NotFound("Endpoint not found");
}

HttpResponse AlertsHandler::HandleListAlerts(const HttpRequest& request) {
    auto alerts = alert_service_->GetActiveAlerts();

    json j = json::array();
    for (const auto& alert : alerts) {
        j.push_back(json::parse(alerting::SerializeAlertEvent(alert)));
    }

    return HttpResponse::Ok(j.dump());
}

HttpResponse AlertsHandler::HandleGetAlert(const HttpRequest& request) {
    std::string path = request.path;
    size_t pos = path.rfind('/');
    if (pos == std::string::npos) {
        return HttpResponse::BadRequest("Invalid path");
    }
    std::string alert_id = path.substr(pos + 1);

    // Search for alert by ID
    auto alerts = alert_service_->GetActiveAlerts();
    for (const auto& alert : alerts) {
        if (alert.alert_id == alert_id) {
            return HttpResponse::Ok(alerting::SerializeAlertEvent(alert));
        }
    }

    return HttpResponse::NotFound("Alert not found: " + alert_id);
}

HttpResponse AlertsHandler::HandleResolveAlert(const HttpRequest& request) {
    // Extract alert ID/fingerprint from path
    std::string path = request.path;
    size_t resolve_pos = path.find("/resolve");
    std::string fingerprint = path.substr(BasePath().length() + 1,
                                          resolve_pos - BasePath().length() - 1);

    auto status = alert_service_->ResolveAlert(fingerprint);
    if (!status.ok()) {
        return HttpResponse::NotFound(std::string(status.message()));
    }

    json response;
    response["fingerprint"] = fingerprint;
    response["status"] = "resolved";

    return HttpResponse::Ok(response.dump());
}

HttpResponse AlertsHandler::HandleAcknowledgeAlert(const HttpRequest& request) {
    std::string path = request.path;
    size_t ack_pos = path.find("/acknowledge");
    std::string alert_id = path.substr(BasePath().length() + 1,
                                       ack_pos - BasePath().length() - 1);

    std::string acknowledged_by = "api";
    try {
        json j = json::parse(request.body);
        acknowledged_by = j.value("acknowledged_by", "api");
    } catch (...) {}

    auto status = alert_service_->AcknowledgeAlert(alert_id, acknowledged_by);
    if (!status.ok()) {
        return HttpResponse::NotFound(std::string(status.message()));
    }

    json response;
    response["alert_id"] = alert_id;
    response["acknowledged_by"] = acknowledged_by;
    response["status"] = "acknowledged";

    return HttpResponse::Ok(response.dump());
}

HttpResponse AlertsHandler::HandleListGroups(const HttpRequest& request) {
    auto groups = alert_service_->GetAlertGroups();

    json j = json::array();
    for (const auto& group : groups) {
        json g;
        g["group_id"] = group.group_id;
        g["group_key"] = group.group_key;
        g["total_count"] = group.total_count;
        g["max_severity"] = alerting::AlertSeverityToString(group.max_severity);
        g["is_firing"] = group.is_firing;
        g["alert_count"] = group.alerts.size();
        j.push_back(g);
    }

    return HttpResponse::Ok(j.dump());
}

HttpResponse AlertsHandler::HandleFireAlert(const HttpRequest& request) {
    try {
        json j = json::parse(request.body);

        std::string title = j.value("title", "Manual Alert");
        std::string description = j.value("description", "");
        alerting::AlertSeverity severity = alerting::StringToAlertSeverity(
            j.value("severity", "warning"));

        std::unordered_map<std::string, std::string> labels;
        if (j.contains("labels")) {
            labels = j["labels"].get<std::unordered_map<std::string, std::string>>();
        }

        auto result = alert_service_->FireAlert(title, description, severity, labels);
        if (!result.ok()) {
            return HttpResponse::InternalError(std::string(result.status().message()));
        }

        return HttpResponse::Created(alerting::SerializeAlertEvent(*result));
    } catch (const json::exception& e) {
        return HttpResponse::BadRequest(std::string("Invalid JSON: ") + e.what());
    }
}

HttpResponse AlertsHandler::HandleStats(const HttpRequest& request) {
    auto stats = alert_service_->GetStats();

    json j;
    j["metrics_ingested"] = stats.metrics_ingested;
    j["rules_evaluated"] = stats.rules_evaluated;
    j["alerts_generated"] = stats.alerts_generated;
    j["alerts_deduplicated"] = stats.alerts_deduplicated;
    j["alerts_suppressed"] = stats.alerts_suppressed;
    j["notifications_sent"] = stats.notifications_sent;
    j["notifications_failed"] = stats.notifications_failed;
    j["active_alerts"] = stats.active_alerts;
    j["active_rules"] = stats.active_rules;
    j["active_silences"] = stats.active_silences;
    j["avg_eval_time_ms"] = stats.avg_eval_time_ms;
    j["avg_notification_time_ms"] = stats.avg_notification_time_ms;

    return HttpResponse::Ok(j.dump());
}

HttpResponse AlertsHandler::HandleListRules(const HttpRequest& request) {
    auto rules = alert_service_->ListRules();

    json j = json::array();
    for (const auto& rule : rules) {
        json r;
        r["rule_id"] = rule.rule_id;
        r["name"] = rule.name;
        r["description"] = rule.description;
        r["type"] = alerting::RuleTypeToString(rule.type);
        r["severity"] = alerting::AlertSeverityToString(rule.severity);
        r["enabled"] = rule.enabled;
        j.push_back(r);
    }

    return HttpResponse::Ok(j.dump());
}

HttpResponse AlertsHandler::HandleCreateRule(const HttpRequest& request) {
    auto rule = ParseAlertRule(request.body);
    if (!rule.ok()) {
        return HttpResponse::BadRequest(std::string(rule.status().message()));
    }

    auto status = alert_service_->AddRule(*rule);
    if (!status.ok()) {
        return HttpResponse::InternalError(std::string(status.message()));
    }

    json response;
    response["rule_id"] = rule->rule_id;
    response["status"] = "created";

    return HttpResponse::Created(response.dump());
}

HttpResponse AlertsHandler::HandleGetRule(const HttpRequest& request) {
    std::string path = request.path;
    size_t pos = path.rfind('/');
    std::string rule_id = path.substr(pos + 1);

    auto rule = alert_service_->GetRule(rule_id);
    if (!rule.ok()) {
        return HttpResponse::NotFound(std::string(rule.status().message()));
    }

    json j;
    j["rule_id"] = rule->rule_id;
    j["name"] = rule->name;
    j["description"] = rule->description;
    j["type"] = alerting::RuleTypeToString(rule->type);
    j["severity"] = alerting::AlertSeverityToString(rule->severity);
    j["enabled"] = rule->enabled;

    return HttpResponse::Ok(j.dump());
}

HttpResponse AlertsHandler::HandleUpdateRule(const HttpRequest& request) {
    auto rule = ParseAlertRule(request.body);
    if (!rule.ok()) {
        return HttpResponse::BadRequest(std::string(rule.status().message()));
    }

    auto status = alert_service_->UpdateRule(*rule);
    if (!status.ok()) {
        return HttpResponse::NotFound(std::string(status.message()));
    }

    json response;
    response["rule_id"] = rule->rule_id;
    response["status"] = "updated";

    return HttpResponse::Ok(response.dump());
}

HttpResponse AlertsHandler::HandleDeleteRule(const HttpRequest& request) {
    std::string path = request.path;
    size_t pos = path.rfind('/');
    std::string rule_id = path.substr(pos + 1);

    auto status = alert_service_->RemoveRule(rule_id);
    if (!status.ok()) {
        return HttpResponse::NotFound(std::string(status.message()));
    }

    json response;
    response["rule_id"] = rule_id;
    response["status"] = "deleted";

    return HttpResponse::Ok(response.dump());
}

HttpResponse AlertsHandler::HandleSetRuleEnabled(const HttpRequest& request) {
    std::string path = request.path;
    size_t rules_pos = path.find("/rules/");
    size_t enabled_pos = path.find("/enabled");
    std::string rule_id = path.substr(rules_pos + 7,
                                      enabled_pos - rules_pos - 7);

    bool enabled = true;
    try {
        json j = json::parse(request.body);
        enabled = j.value("enabled", true);
    } catch (...) {}

    auto status = alert_service_->SetRuleEnabled(rule_id, enabled);
    if (!status.ok()) {
        return HttpResponse::NotFound(std::string(status.message()));
    }

    json response;
    response["rule_id"] = rule_id;
    response["enabled"] = enabled;

    return HttpResponse::Ok(response.dump());
}

HttpResponse AlertsHandler::HandleListSilences(const HttpRequest& request) {
    auto silences = alert_service_->GetActiveSilences();

    json j = json::array();
    for (const auto& silence : silences) {
        j.push_back(json::parse(alerting::SerializeSilence(silence)));
    }

    return HttpResponse::Ok(j.dump());
}

HttpResponse AlertsHandler::HandleCreateSilence(const HttpRequest& request) {
    auto silence = ParseSilence(request.body);
    if (!silence.ok()) {
        return HttpResponse::BadRequest(std::string(silence.status().message()));
    }

    auto status = alert_service_->AddSilence(*silence);
    if (!status.ok()) {
        return HttpResponse::InternalError(std::string(status.message()));
    }

    json response;
    response["silence_id"] = silence->silence_id;
    response["status"] = "created";

    return HttpResponse::Created(response.dump());
}

HttpResponse AlertsHandler::HandleDeleteSilence(const HttpRequest& request) {
    std::string path = request.path;
    size_t pos = path.rfind('/');
    std::string silence_id = path.substr(pos + 1);

    auto status = alert_service_->RemoveSilence(silence_id);
    if (!status.ok()) {
        return HttpResponse::NotFound(std::string(status.message()));
    }

    json response;
    response["silence_id"] = silence_id;
    response["status"] = "deleted";

    return HttpResponse::Ok(response.dump());
}

HttpResponse AlertsHandler::HandleListMaintenance(const HttpRequest& request) {
    auto windows = alert_service_->GetActiveMaintenanceWindows();

    json j = json::array();
    for (const auto& window : windows) {
        j.push_back(json::parse(alerting::SerializeMaintenanceWindow(window)));
    }

    return HttpResponse::Ok(j.dump());
}

HttpResponse AlertsHandler::HandleCreateMaintenance(const HttpRequest& request) {
    auto window = ParseMaintenanceWindow(request.body);
    if (!window.ok()) {
        return HttpResponse::BadRequest(std::string(window.status().message()));
    }

    auto status = alert_service_->AddMaintenanceWindow(*window);
    if (!status.ok()) {
        return HttpResponse::InternalError(std::string(status.message()));
    }

    json response;
    response["window_id"] = window->window_id;
    response["status"] = "created";

    return HttpResponse::Created(response.dump());
}

HttpResponse AlertsHandler::HandleDeleteMaintenance(const HttpRequest& request) {
    std::string path = request.path;
    size_t pos = path.rfind('/');
    std::string window_id = path.substr(pos + 1);

    auto status = alert_service_->RemoveMaintenanceWindow(window_id);
    if (!status.ok()) {
        return HttpResponse::NotFound(std::string(status.message()));
    }

    json response;
    response["window_id"] = window_id;
    response["status"] = "deleted";

    return HttpResponse::Ok(response.dump());
}

HttpResponse AlertsHandler::HandleListChannels(const HttpRequest& request) {
    // Channel list would come from alert service
    json j = json::array();
    return HttpResponse::Ok(j.dump());
}

HttpResponse AlertsHandler::HandleCreateChannel(const HttpRequest& request) {
    auto config = ParseNotificationConfig(request.body);
    if (!config.ok()) {
        return HttpResponse::BadRequest(std::string(config.status().message()));
    }

    auto status = alert_service_->AddNotificationChannel(*config);
    if (!status.ok()) {
        return HttpResponse::InternalError(std::string(status.message()));
    }

    json response;
    response["name"] = config->name;
    response["status"] = "created";

    return HttpResponse::Created(response.dump());
}

HttpResponse AlertsHandler::HandleDeleteChannel(const HttpRequest& request) {
    std::string path = request.path;
    size_t pos = path.rfind('/');
    std::string name = path.substr(pos + 1);

    auto status = alert_service_->RemoveNotificationChannel(name);
    if (!status.ok()) {
        return HttpResponse::NotFound(std::string(status.message()));
    }

    json response;
    response["name"] = name;
    response["status"] = "deleted";

    return HttpResponse::Ok(response.dump());
}

HttpResponse AlertsHandler::HandleTestChannel(const HttpRequest& request) {
    std::string path = request.path;
    size_t channels_pos = path.find("/channels/");
    size_t test_pos = path.find("/test");
    std::string name = path.substr(channels_pos + 10, test_pos - channels_pos - 10);

    auto result = alert_service_->TestNotificationChannel(name);
    if (!result.ok()) {
        return HttpResponse::NotFound(std::string(result.status().message()));
    }

    json response;
    response["channel"] = name;
    response["success"] = result->success;
    response["error"] = result->error_message;
    response["latency_ms"] = result->latency.count();

    return HttpResponse::Ok(response.dump());
}

absl::StatusOr<alerting::AlertRule> AlertsHandler::ParseAlertRule(
    const std::string& json_str) {

    try {
        json j = json::parse(json_str);

        alerting::AlertRule rule;
        rule.rule_id = j.value("rule_id", "");
        rule.name = j.value("name", "");
        rule.description = j.value("description", "");
        rule.type = alerting::StringToRuleType(j.value("type", "threshold"));
        rule.severity = alerting::StringToAlertSeverity(j.value("severity", "warning"));
        rule.enabled = j.value("enabled", true);

        if (rule.rule_id.empty()) {
            return absl::InvalidArgumentError("rule_id is required");
        }
        if (rule.name.empty()) {
            return absl::InvalidArgumentError("name is required");
        }

        // Parse type-specific config
        if (j.contains("config")) {
            const auto& cfg = j["config"];
            switch (rule.type) {
                case alerting::RuleType::kThreshold: {
                    alerting::ThresholdRuleConfig config;
                    config.metric_name = cfg.value("metric_name", "");
                    config.comparison = alerting::StringToComparisonOp(
                        cfg.value("comparison", "greater_than"));
                    config.threshold = cfg.value("threshold", 0.0);
                    config.consecutive_count = cfg.value("consecutive_count", 1);
                    rule.config = config;
                    break;
                }
                case alerting::RuleType::kAnomaly: {
                    alerting::AnomalyRuleConfig config;
                    config.metric_name = cfg.value("metric_name", "");
                    config.std_dev_threshold = cfg.value("std_dev_threshold", 3.0);
                    rule.config = config;
                    break;
                }
                case alerting::RuleType::kRate: {
                    alerting::RateRuleConfig config;
                    config.metric_name = cfg.value("metric_name", "");
                    config.rate_threshold = cfg.value("rate_threshold", 0.0);
                    rule.config = config;
                    break;
                }
                default:
                    break;
            }
        }

        return rule;
    } catch (const json::exception& e) {
        return absl::InvalidArgumentError(
            std::string("Failed to parse alert rule: ") + e.what());
    }
}

absl::StatusOr<alerting::Silence> AlertsHandler::ParseSilence(
    const std::string& json_str) {

    try {
        json j = json::parse(json_str);

        alerting::Silence silence;
        silence.silence_id = j.value("silence_id", "");
        silence.created_by = j.value("created_by", "api");
        silence.comment = j.value("comment", "");

        if (j.contains("matchers")) {
            silence.matchers = j["matchers"].get<
                std::unordered_map<std::string, std::string>>();
        }

        silence.starts_at = std::chrono::system_clock::now();
        if (j.contains("starts_at")) {
            silence.starts_at = std::chrono::system_clock::time_point(
                std::chrono::seconds(j["starts_at"].get<int64_t>()));
        }

        if (j.contains("ends_at")) {
            silence.ends_at = std::chrono::system_clock::time_point(
                std::chrono::seconds(j["ends_at"].get<int64_t>()));
        } else if (j.contains("duration_hours")) {
            silence.ends_at = silence.starts_at +
                std::chrono::hours(j["duration_hours"].get<int>());
        } else {
            silence.ends_at = silence.starts_at + std::chrono::hours(1);
        }

        return silence;
    } catch (const json::exception& e) {
        return absl::InvalidArgumentError(
            std::string("Failed to parse silence: ") + e.what());
    }
}

absl::StatusOr<alerting::MaintenanceWindow> AlertsHandler::ParseMaintenanceWindow(
    const std::string& json_str) {

    try {
        json j = json::parse(json_str);

        alerting::MaintenanceWindow window;
        window.window_id = j.value("window_id", "");
        window.name = j.value("name", "");
        window.description = j.value("description", "");

        if (j.contains("model_ids")) {
            window.model_ids = j["model_ids"].get<std::vector<std::string>>();
        }
        if (j.contains("rule_ids")) {
            window.rule_ids = j["rule_ids"].get<std::vector<std::string>>();
        }

        window.starts_at = std::chrono::system_clock::now();
        if (j.contains("starts_at")) {
            window.starts_at = std::chrono::system_clock::time_point(
                std::chrono::seconds(j["starts_at"].get<int64_t>()));
        }

        if (j.contains("ends_at")) {
            window.ends_at = std::chrono::system_clock::time_point(
                std::chrono::seconds(j["ends_at"].get<int64_t>()));
        } else if (j.contains("duration_hours")) {
            window.ends_at = window.starts_at +
                std::chrono::hours(j["duration_hours"].get<int>());
        } else {
            window.ends_at = window.starts_at + std::chrono::hours(1);
        }

        window.suppress_alerts = j.value("suppress_alerts", true);
        window.suppress_notifications = j.value("suppress_notifications", true);

        return window;
    } catch (const json::exception& e) {
        return absl::InvalidArgumentError(
            std::string("Failed to parse maintenance window: ") + e.what());
    }
}

absl::StatusOr<alerting::NotificationConfig> AlertsHandler::ParseNotificationConfig(
    const std::string& json_str) {

    try {
        json j = json::parse(json_str);

        alerting::NotificationConfig config;
        config.name = j.value("name", "");
        config.channel = alerting::StringToNotificationChannel(
            j.value("channel", "webhook"));
        config.enabled = j.value("enabled", true);
        config.webhook_url = j.value("webhook_url", "");
        config.api_key = j.value("api_key", "");
        config.routing_key = j.value("routing_key", "");
        config.email_recipients = j.value("email_recipients", "");
        config.min_severity = alerting::StringToAlertSeverity(
            j.value("min_severity", "warning"));
        config.max_per_hour = j.value("max_per_hour", 100);

        if (config.name.empty()) {
            return absl::InvalidArgumentError("name is required");
        }

        return config;
    } catch (const json::exception& e) {
        return absl::InvalidArgumentError(
            std::string("Failed to parse notification config: ") + e.what());
    }
}

std::unique_ptr<AlertsHandler> CreateAlertsHandler(
    std::shared_ptr<alerting::AlertService> alert_service) {

    return std::make_unique<AlertsHandler>(std::move(alert_service));
}

}  // namespace pyflare::query
