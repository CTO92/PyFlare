#pragma once

/// @file alerts_handler.h
/// @brief REST API handler for alerting operations
///
/// Provides HTTP endpoints for:
/// - Alert management (list, acknowledge, resolve)
/// - Alert rules CRUD
/// - Silences management
/// - Maintenance windows
/// - Notification channels

#include <memory>
#include <string>

#include <absl/status/status.h>
#include <absl/status/statusor.h>

#include "processor/alerting/alert_service.h"
#include "query/handlers/intelligence_handler.h"

namespace pyflare::query {

/// @brief Alerts API handler
///
/// Handles REST API requests for alert operations.
///
/// Endpoints:
/// - GET /api/v1/alerts - List active alerts
/// - GET /api/v1/alerts/:alert_id - Get alert details
/// - POST /api/v1/alerts/:alert_id/resolve - Resolve alert
/// - POST /api/v1/alerts/:alert_id/acknowledge - Acknowledge alert
/// - GET /api/v1/alerts/groups - List alert groups
///
/// Rules:
/// - GET /api/v1/alerts/rules - List rules
/// - POST /api/v1/alerts/rules - Create rule
/// - GET /api/v1/alerts/rules/:rule_id - Get rule
/// - PUT /api/v1/alerts/rules/:rule_id - Update rule
/// - DELETE /api/v1/alerts/rules/:rule_id - Delete rule
///
/// Silences:
/// - GET /api/v1/alerts/silences - List silences
/// - POST /api/v1/alerts/silences - Create silence
/// - DELETE /api/v1/alerts/silences/:silence_id - Delete silence
///
/// Maintenance:
/// - GET /api/v1/alerts/maintenance - List maintenance windows
/// - POST /api/v1/alerts/maintenance - Create window
/// - DELETE /api/v1/alerts/maintenance/:window_id - Delete window
///
/// Example:
/// @code
///   AlertsHandler handler(alert_service);
///
///   HttpRequest req;
///   req.method = "GET";
///   req.path = "/api/v1/alerts";
///
///   auto resp = handler.Handle(req);
/// @endcode
class AlertsHandler {
public:
    explicit AlertsHandler(
        std::shared_ptr<alerting::AlertService> alert_service);
    ~AlertsHandler();

    // Disable copy
    AlertsHandler(const AlertsHandler&) = delete;
    AlertsHandler& operator=(const AlertsHandler&) = delete;

    /// @brief Handle HTTP request
    HttpResponse Handle(const HttpRequest& request);

    /// @brief Get handler name
    std::string Name() const { return "alerts"; }

    /// @brief Get base path
    std::string BasePath() const { return "/api/v1/alerts"; }

private:
    // Alert operations
    HttpResponse HandleListAlerts(const HttpRequest& request);
    HttpResponse HandleGetAlert(const HttpRequest& request);
    HttpResponse HandleResolveAlert(const HttpRequest& request);
    HttpResponse HandleAcknowledgeAlert(const HttpRequest& request);
    HttpResponse HandleListGroups(const HttpRequest& request);
    HttpResponse HandleFireAlert(const HttpRequest& request);
    HttpResponse HandleStats(const HttpRequest& request);

    // Rule operations
    HttpResponse HandleListRules(const HttpRequest& request);
    HttpResponse HandleCreateRule(const HttpRequest& request);
    HttpResponse HandleGetRule(const HttpRequest& request);
    HttpResponse HandleUpdateRule(const HttpRequest& request);
    HttpResponse HandleDeleteRule(const HttpRequest& request);
    HttpResponse HandleSetRuleEnabled(const HttpRequest& request);

    // Silence operations
    HttpResponse HandleListSilences(const HttpRequest& request);
    HttpResponse HandleCreateSilence(const HttpRequest& request);
    HttpResponse HandleDeleteSilence(const HttpRequest& request);

    // Maintenance window operations
    HttpResponse HandleListMaintenance(const HttpRequest& request);
    HttpResponse HandleCreateMaintenance(const HttpRequest& request);
    HttpResponse HandleDeleteMaintenance(const HttpRequest& request);

    // Notification channel operations
    HttpResponse HandleListChannels(const HttpRequest& request);
    HttpResponse HandleCreateChannel(const HttpRequest& request);
    HttpResponse HandleDeleteChannel(const HttpRequest& request);
    HttpResponse HandleTestChannel(const HttpRequest& request);

    // Parse helpers
    absl::StatusOr<alerting::AlertRule> ParseAlertRule(const std::string& json);
    absl::StatusOr<alerting::Silence> ParseSilence(const std::string& json);
    absl::StatusOr<alerting::MaintenanceWindow> ParseMaintenanceWindow(
        const std::string& json);
    absl::StatusOr<alerting::NotificationConfig> ParseNotificationConfig(
        const std::string& json);

    std::shared_ptr<alerting::AlertService> alert_service_;
};

/// @brief Create alerts handler
std::unique_ptr<AlertsHandler> CreateAlertsHandler(
    std::shared_ptr<alerting::AlertService> alert_service);

}  // namespace pyflare::query
