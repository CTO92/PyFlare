/// @file alert_service.cpp
/// @brief Alert service implementation

#include "processor/alerting/alert_service.h"

#include <algorithm>
#include <random>
#include <regex>
#include <sstream>

#include <nlohmann/json.hpp>

namespace pyflare::alerting {

using json = nlohmann::json;

AlertService::AlertService(
    std::shared_ptr<storage::ClickHouseClient> clickhouse,
    std::shared_ptr<storage::RedisClient> redis,
    AlertServiceConfig config)
    : clickhouse_(std::move(clickhouse)),
      redis_(std::move(redis)),
      config_(std::move(config)) {}

AlertService::~AlertService() {
    Stop();
}

absl::Status AlertService::Initialize() {
    if (initialized_) {
        return absl::OkStatus();
    }

    // Initialize rules engine
    rules_engine_ = std::make_unique<AlertRulesEngine>();
    auto status = rules_engine_->Initialize();
    if (!status.ok()) {
        return status;
    }

    // Initialize deduplicator
    deduplicator_ = std::make_unique<AlertDeduplicator>(config_.dedup_config);
    status = deduplicator_->Initialize();
    if (!status.ok()) {
        return status;
    }

    // Add notification channels
    for (const auto& channel : config_.notification_channels) {
        notification_channels_[channel.name] = channel;
    }

    initialized_ = true;
    return absl::OkStatus();
}

absl::Status AlertService::Start() {
    if (!initialized_) {
        return absl::FailedPreconditionError("Service not initialized");
    }

    if (running_.load()) {
        return absl::OkStatus();
    }

    running_.store(true);

    // Start evaluation worker
    workers_.emplace_back(&AlertService::EvaluationLoop, this);

    // Start notification worker
    workers_.emplace_back(&AlertService::NotificationLoop, this);

    // Start cleanup worker
    workers_.emplace_back(&AlertService::CleanupLoop, this);

    return absl::OkStatus();
}

void AlertService::Stop() {
    if (!running_.load()) {
        return;
    }

    running_.store(false);
    queue_cv_.notify_all();

    for (auto& worker : workers_) {
        if (worker.joinable()) {
            worker.join();
        }
    }
    workers_.clear();
}

std::vector<AlertEvent> AlertService::IngestMetrics(
    const std::vector<MetricValue>& metrics) {

    if (!initialized_) {
        return {};
    }

    auto start_time = std::chrono::steady_clock::now();

    // Record metrics in rules engine for anomaly/rate detection
    for (const auto& metric : metrics) {
        rules_engine_->RecordMetric(metric);
    }

    // Evaluate rules
    auto alerts = rules_engine_->Evaluate(metrics);

    // Process through deduplicator
    std::vector<AlertEvent> new_alerts;
    for (auto& alert : alerts) {
        auto result = deduplicator_->Process(alert);

        if (result.accepted && !result.is_duplicate && !result.is_suppressed) {
            new_alerts.push_back(alert);

            // Queue for notification
            if (config_.notify_on_new) {
                std::lock_guard<std::mutex> lock(queue_mutex_);
                notification_queue_.push(alert);
                queue_cv_.notify_one();
            }

            // Persist
            if (config_.persist_alerts) {
                PersistAlert(alert);
            }
            if (config_.cache_alerts) {
                CacheAlert(alert);
            }

            // Notify callbacks
            {
                std::lock_guard<std::mutex> lock(callbacks_mutex_);
                for (const auto& callback : alert_callbacks_) {
                    callback(alert);
                }
            }
        }
    }

    // Update stats
    auto end_time = std::chrono::steady_clock::now();
    double eval_time = std::chrono::duration<double, std::milli>(
        end_time - start_time).count();

    {
        std::lock_guard<std::mutex> lock(stats_mutex_);
        stats_.metrics_ingested += metrics.size();
        stats_.rules_evaluated++;
        stats_.alerts_generated += new_alerts.size();
        stats_.last_eval = std::chrono::system_clock::now();

        // Update running average
        stats_.avg_eval_time_ms = (stats_.avg_eval_time_ms *
            (stats_.rules_evaluated - 1) + eval_time) / stats_.rules_evaluated;
    }

    return new_alerts;
}

std::vector<AlertEvent> AlertService::IngestMetric(const MetricValue& metric) {
    return IngestMetrics({metric});
}

absl::StatusOr<AlertEvent> AlertService::FireAlert(
    const std::string& title,
    const std::string& description,
    AlertSeverity severity,
    const std::unordered_map<std::string, std::string>& labels) {

    if (!initialized_) {
        return absl::FailedPreconditionError("Service not initialized");
    }

    // Create alert event
    AlertEvent alert;

    // Generate ID
    static std::random_device rd;
    static std::mt19937 gen(rd());
    static std::uniform_int_distribution<uint64_t> dis;

    std::ostringstream id;
    id << "manual-" << std::hex << dis(gen);
    alert.alert_id = id.str();

    alert.rule_id = "manual";
    alert.rule_name = "Manual Alert";
    alert.severity = severity;
    alert.triggered_at = std::chrono::system_clock::now();
    alert.is_firing = true;
    alert.title = title;
    alert.description = description;
    alert.labels = labels;

    // Generate fingerprint
    std::ostringstream fp;
    fp << "manual:" << title << ":";
    for (const auto& [k, v] : labels) {
        fp << k << "=" << v << ",";
    }
    alert.fingerprint = fp.str();

    // Process through deduplicator
    auto result = deduplicator_->Process(alert);

    if (!result.accepted) {
        return absl::AlreadyExistsError("Alert deduplicated or suppressed");
    }

    // Queue for notification
    if (config_.notify_on_new && !result.is_duplicate) {
        std::lock_guard<std::mutex> lock(queue_mutex_);
        notification_queue_.push(alert);
        queue_cv_.notify_one();
    }

    // Persist and cache
    if (config_.persist_alerts) {
        PersistAlert(alert);
    }
    if (config_.cache_alerts) {
        CacheAlert(alert);
    }

    return alert;
}

absl::Status AlertService::ResolveAlert(const std::string& fingerprint) {
    if (!initialized_) {
        return absl::FailedPreconditionError("Service not initialized");
    }

    auto status = deduplicator_->Resolve(fingerprint);
    if (!status.ok()) {
        return status;
    }

    // Get resolved alert and notify
    auto alerts = deduplicator_->GetAlertsByFingerprint(fingerprint);
    if (!alerts.empty() && config_.notify_on_resolved) {
        std::lock_guard<std::mutex> lock(queue_mutex_);
        notification_queue_.push(alerts[0]);
        queue_cv_.notify_one();

        // Notify callbacks
        {
            std::lock_guard<std::mutex> lock(callbacks_mutex_);
            for (const auto& callback : resolved_callbacks_) {
                callback(alerts[0]);
            }
        }
    }

    return absl::OkStatus();
}

absl::Status AlertService::AcknowledgeAlert(
    const std::string& alert_id,
    const std::string& acknowledged_by) {

    // For now, acknowledgment just adds an annotation
    // Could be extended to pause notifications for this alert

    return absl::OkStatus();
}

absl::Status AlertService::AddRule(const AlertRule& rule) {
    if (!initialized_) {
        return absl::FailedPreconditionError("Service not initialized");
    }

    auto status = rules_engine_->AddRule(rule);
    if (status.ok()) {
        std::lock_guard<std::mutex> lock(stats_mutex_);
        stats_.active_rules++;
    }
    return status;
}

absl::Status AlertService::UpdateRule(const AlertRule& rule) {
    if (!initialized_) {
        return absl::FailedPreconditionError("Service not initialized");
    }
    return rules_engine_->UpdateRule(rule);
}

absl::Status AlertService::RemoveRule(const std::string& rule_id) {
    if (!initialized_) {
        return absl::FailedPreconditionError("Service not initialized");
    }

    auto status = rules_engine_->RemoveRule(rule_id);
    if (status.ok()) {
        std::lock_guard<std::mutex> lock(stats_mutex_);
        stats_.active_rules--;
    }
    return status;
}

absl::StatusOr<AlertRule> AlertService::GetRule(const std::string& rule_id) const {
    if (!initialized_) {
        return absl::FailedPreconditionError("Service not initialized");
    }
    return rules_engine_->GetRule(rule_id);
}

std::vector<AlertRule> AlertService::ListRules() const {
    if (!initialized_) {
        return {};
    }
    return rules_engine_->ListRules();
}

absl::Status AlertService::SetRuleEnabled(const std::string& rule_id, bool enabled) {
    if (!initialized_) {
        return absl::FailedPreconditionError("Service not initialized");
    }
    return rules_engine_->SetRuleEnabled(rule_id, enabled);
}

absl::Status AlertService::AddSilence(const Silence& silence) {
    if (!initialized_) {
        return absl::FailedPreconditionError("Service not initialized");
    }

    auto status = deduplicator_->AddSilence(silence);
    if (status.ok()) {
        std::lock_guard<std::mutex> lock(stats_mutex_);
        stats_.active_silences++;
    }
    return status;
}

absl::Status AlertService::RemoveSilence(const std::string& silence_id) {
    if (!initialized_) {
        return absl::FailedPreconditionError("Service not initialized");
    }

    auto status = deduplicator_->RemoveSilence(silence_id);
    if (status.ok()) {
        std::lock_guard<std::mutex> lock(stats_mutex_);
        stats_.active_silences--;
    }
    return status;
}

std::vector<Silence> AlertService::GetActiveSilences() const {
    if (!initialized_) {
        return {};
    }
    return deduplicator_->GetActiveSilences();
}

absl::Status AlertService::AddMaintenanceWindow(const MaintenanceWindow& window) {
    if (!initialized_) {
        return absl::FailedPreconditionError("Service not initialized");
    }
    return deduplicator_->AddMaintenanceWindow(window);
}

absl::Status AlertService::RemoveMaintenanceWindow(const std::string& window_id) {
    if (!initialized_) {
        return absl::FailedPreconditionError("Service not initialized");
    }
    return deduplicator_->RemoveMaintenanceWindow(window_id);
}

std::vector<MaintenanceWindow> AlertService::GetActiveMaintenanceWindows() const {
    if (!initialized_) {
        return {};
    }
    return deduplicator_->GetActiveMaintenanceWindows();
}

std::vector<AlertEvent> AlertService::GetActiveAlerts() const {
    if (!initialized_) {
        return {};
    }
    return deduplicator_->GetActiveAlerts();
}

std::vector<AlertGroup> AlertService::GetAlertGroups() const {
    if (!initialized_) {
        return {};
    }
    return deduplicator_->GetGroups();
}

std::vector<AlertEvent> AlertService::GetAlertsByRule(
    const std::string& rule_id) const {

    if (!initialized_) {
        return {};
    }
    return deduplicator_->GetAlertsByRule(rule_id);
}

std::vector<AlertEvent> AlertService::GetAlertsInRange(
    std::chrono::system_clock::time_point start,
    std::chrono::system_clock::time_point end) const {

    if (!initialized_) {
        return {};
    }
    return deduplicator_->GetAlertsInRange(start, end);
}

absl::StatusOr<std::vector<AlertEvent>> AlertService::GetAlertHistory(
    size_t limit,
    size_t offset) const {

    if (!initialized_) {
        return absl::FailedPreconditionError("Service not initialized");
    }
    return LoadAlertHistory(limit, offset);
}

absl::Status AlertService::AddNotificationChannel(const NotificationConfig& config) {
    std::lock_guard<std::mutex> lock(channels_mutex_);

    if (notification_channels_.count(config.name) > 0) {
        return absl::AlreadyExistsError(
            "Notification channel already exists: " + config.name);
    }

    notification_channels_[config.name] = config;
    return absl::OkStatus();
}

absl::Status AlertService::RemoveNotificationChannel(const std::string& name) {
    std::lock_guard<std::mutex> lock(channels_mutex_);

    auto it = notification_channels_.find(name);
    if (it == notification_channels_.end()) {
        return absl::NotFoundError(
            "Notification channel not found: " + name);
    }

    notification_channels_.erase(it);
    return absl::OkStatus();
}

absl::StatusOr<NotificationResult> AlertService::TestNotificationChannel(
    const std::string& name) {

    std::lock_guard<std::mutex> lock(channels_mutex_);

    auto it = notification_channels_.find(name);
    if (it == notification_channels_.end()) {
        return absl::NotFoundError(
            "Notification channel not found: " + name);
    }

    // Create test alert
    AlertEvent test_alert;
    test_alert.alert_id = "test-alert";
    test_alert.title = "Test Alert";
    test_alert.description = "This is a test notification from PyFlare";
    test_alert.severity = AlertSeverity::kInfo;
    test_alert.triggered_at = std::chrono::system_clock::now();
    test_alert.is_firing = true;

    // Send notification
    NotificationResult result;
    switch (it->second.channel) {
        case NotificationChannel::kSlack:
            result = SendSlackNotification(test_alert, it->second);
            break;
        case NotificationChannel::kPagerDuty:
            result = SendPagerDutyNotification(test_alert, it->second);
            break;
        case NotificationChannel::kWebhook:
            result = SendWebhookNotification(test_alert, it->second);
            break;
        case NotificationChannel::kEmail:
            result = SendEmailNotification(test_alert, it->second);
            break;
        default:
            result.success = false;
            result.error_message = "Unsupported channel type";
    }

    return result;
}

std::vector<NotificationResult> AlertService::GetNotificationHistory(
    size_t limit) const {

    std::lock_guard<std::mutex> lock(history_mutex_);

    size_t count = std::min(limit, notification_history_.size());
    return std::vector<NotificationResult>(
        notification_history_.end() - count,
        notification_history_.end());
}

void AlertService::OnAlert(std::function<void(const AlertEvent&)> callback) {
    std::lock_guard<std::mutex> lock(callbacks_mutex_);
    alert_callbacks_.push_back(std::move(callback));
}

void AlertService::OnResolved(std::function<void(const AlertEvent&)> callback) {
    std::lock_guard<std::mutex> lock(callbacks_mutex_);
    resolved_callbacks_.push_back(std::move(callback));
}

void AlertService::ClearCallbacks() {
    std::lock_guard<std::mutex> lock(callbacks_mutex_);
    alert_callbacks_.clear();
    resolved_callbacks_.clear();
}

AlertService::Stats AlertService::GetStats() const {
    std::lock_guard<std::mutex> lock(stats_mutex_);
    return stats_;
}

void AlertService::ResetStats() {
    std::lock_guard<std::mutex> lock(stats_mutex_);

    // Keep active counts
    size_t active_alerts = stats_.active_alerts;
    size_t active_rules = stats_.active_rules;
    size_t active_silences = stats_.active_silences;

    stats_ = Stats{};
    stats_.active_alerts = active_alerts;
    stats_.active_rules = active_rules;
    stats_.active_silences = active_silences;
}

void AlertService::SetConfig(AlertServiceConfig config) {
    config_ = std::move(config);

    if (deduplicator_) {
        deduplicator_->SetConfig(config_.dedup_config);
    }
}

// Private methods

void AlertService::EvaluationLoop() {
    while (running_.load()) {
        std::this_thread::sleep_for(config_.eval_interval);

        if (!running_.load()) break;

        // Could poll external metrics sources here
        // For now, metrics are pushed via IngestMetrics()
    }
}

void AlertService::NotificationLoop() {
    while (running_.load()) {
        AlertEvent alert;

        {
            std::unique_lock<std::mutex> lock(queue_mutex_);
            queue_cv_.wait_for(lock, std::chrono::seconds(1), [this] {
                return !notification_queue_.empty() || !running_.load();
            });

            if (!running_.load() && notification_queue_.empty()) {
                break;
            }

            if (notification_queue_.empty()) {
                continue;
            }

            alert = notification_queue_.front();
            notification_queue_.pop();
        }

        // Send notifications
        auto results = SendNotifications(alert);

        // Record results
        {
            std::lock_guard<std::mutex> lock(history_mutex_);
            for (const auto& result : results) {
                notification_history_.push_back(result);
                if (notification_history_.size() > 1000) {
                    notification_history_.erase(notification_history_.begin());
                }
            }
        }

        // Update stats
        {
            std::lock_guard<std::mutex> lock(stats_mutex_);
            for (const auto& result : results) {
                if (result.success) {
                    stats_.notifications_sent++;
                } else {
                    stats_.notifications_failed++;
                }
            }
        }
    }
}

void AlertService::CleanupLoop() {
    while (running_.load()) {
        std::this_thread::sleep_for(std::chrono::minutes(5));

        if (!running_.load()) break;

        if (deduplicator_) {
            deduplicator_->Cleanup();
        }

        // Update active counts
        {
            std::lock_guard<std::mutex> lock(stats_mutex_);
            if (deduplicator_) {
                auto dedup_stats = deduplicator_->GetStats();
                stats_.active_alerts = dedup_stats.active_alerts;
                stats_.active_silences = dedup_stats.active_silences;
            }
        }
    }
}

std::vector<NotificationResult> AlertService::SendNotifications(
    const AlertEvent& alert) {

    std::vector<NotificationResult> results;

    std::lock_guard<std::mutex> lock(channels_mutex_);

    for (const auto& [name, config] : notification_channels_) {
        if (!config.enabled) continue;

        // Check severity filter
        if (static_cast<int>(alert.severity) < static_cast<int>(config.min_severity)) {
            continue;
        }

        NotificationResult result;
        auto start = std::chrono::steady_clock::now();

        switch (config.channel) {
            case NotificationChannel::kSlack:
                result = SendSlackNotification(alert, config);
                break;
            case NotificationChannel::kPagerDuty:
                result = SendPagerDutyNotification(alert, config);
                break;
            case NotificationChannel::kWebhook:
                result = SendWebhookNotification(alert, config);
                break;
            case NotificationChannel::kEmail:
                result = SendEmailNotification(alert, config);
                break;
            default:
                result.success = false;
                result.error_message = "Unsupported channel type";
        }

        auto end = std::chrono::steady_clock::now();
        result.latency = std::chrono::duration_cast<std::chrono::milliseconds>(
            end - start);
        result.channel_name = name;
        result.channel = config.channel;
        result.sent_at = std::chrono::system_clock::now();

        results.push_back(result);
    }

    return results;
}

NotificationResult AlertService::SendSlackNotification(
    const AlertEvent& alert,
    const NotificationConfig& config) {

    NotificationResult result;

    // Build Slack message payload
    json payload;

    std::string color;
    switch (alert.severity) {
        case AlertSeverity::kCritical: color = "#FF0000"; break;
        case AlertSeverity::kError: color = "#FF6600"; break;
        case AlertSeverity::kWarning: color = "#FFCC00"; break;
        default: color = "#36A64F"; break;
    }

    json attachment;
    attachment["color"] = color;
    attachment["title"] = alert.title;
    attachment["text"] = alert.description;
    attachment["footer"] = "PyFlare Alerting";

    json fields = json::array();
    fields.push_back({{"title", "Severity"}, {"value", AlertSeverityToString(alert.severity)}, {"short", true}});
    fields.push_back({{"title", "Rule"}, {"value", alert.rule_name}, {"short", true}});

    if (!alert.model_id.empty()) {
        fields.push_back({{"title", "Model"}, {"value", alert.model_id}, {"short", true}});
    }

    attachment["fields"] = fields;
    payload["attachments"] = json::array({attachment});

    // In a real implementation, this would make an HTTP request
    // For now, simulate success
    result.success = true;

    return result;
}

NotificationResult AlertService::SendPagerDutyNotification(
    const AlertEvent& alert,
    const NotificationConfig& config) {

    NotificationResult result;

    json payload;
    payload["routing_key"] = config.routing_key;
    payload["event_action"] = alert.is_firing ? "trigger" : "resolve";
    payload["dedup_key"] = alert.fingerprint;

    json pd_payload;
    pd_payload["summary"] = alert.title + ": " + alert.description;
    pd_payload["source"] = "pyflare";

    std::string severity;
    switch (alert.severity) {
        case AlertSeverity::kCritical: severity = "critical"; break;
        case AlertSeverity::kError: severity = "error"; break;
        case AlertSeverity::kWarning: severity = "warning"; break;
        default: severity = "info"; break;
    }
    pd_payload["severity"] = severity;

    payload["payload"] = pd_payload;

    // In a real implementation, this would make an HTTP request
    result.success = true;

    return result;
}

NotificationResult AlertService::SendWebhookNotification(
    const AlertEvent& alert,
    const NotificationConfig& config) {

    NotificationResult result;

    // Build webhook payload
    json payload;
    payload["alert_id"] = alert.alert_id;
    payload["title"] = alert.title;
    payload["description"] = alert.description;
    payload["severity"] = AlertSeverityToString(alert.severity);
    payload["is_firing"] = alert.is_firing;
    payload["fingerprint"] = alert.fingerprint;
    payload["rule_id"] = alert.rule_id;
    payload["rule_name"] = alert.rule_name;
    payload["model_id"] = alert.model_id;
    payload["labels"] = alert.labels;
    payload["triggered_at"] = std::chrono::duration_cast<std::chrono::seconds>(
        alert.triggered_at.time_since_epoch()).count();

    // In a real implementation, this would make an HTTP request
    result.success = true;

    return result;
}

NotificationResult AlertService::SendEmailNotification(
    const AlertEvent& alert,
    const NotificationConfig& config) {

    NotificationResult result;

    // Email sending would be implemented with an SMTP library
    // For now, simulate success
    result.success = true;

    return result;
}

std::string AlertService::RenderTemplate(const std::string& tmpl,
                                         const AlertEvent& alert) const {

    std::string result = tmpl;

    // Simple template variable replacement
    auto replace = [&result](const std::string& var, const std::string& value) {
        std::string pattern = "{{" + var + "}}";
        size_t pos;
        while ((pos = result.find(pattern)) != std::string::npos) {
            result.replace(pos, pattern.length(), value);
        }
    };

    replace("title", alert.title);
    replace("description", alert.description);
    replace("severity", AlertSeverityToString(alert.severity));
    replace("rule_id", alert.rule_id);
    replace("rule_name", alert.rule_name);
    replace("model_id", alert.model_id);
    replace("fingerprint", alert.fingerprint);

    return result;
}

absl::Status AlertService::PersistAlert(const AlertEvent& alert) {
    if (!clickhouse_) {
        return absl::FailedPreconditionError("ClickHouse client not available");
    }

    auto triggered_at_sec = std::chrono::duration_cast<std::chrono::seconds>(
        alert.triggered_at.time_since_epoch()).count();

    std::ostringstream query;
    query << "INSERT INTO alerts (alert_id, rule_id, rule_name, severity, "
          << "triggered_at, is_firing, title, description, fingerprint, model_id) VALUES ('"
          << alert.alert_id << "', '"
          << alert.rule_id << "', '"
          << alert.rule_name << "', '"
          << AlertSeverityToString(alert.severity) << "', "
          << "toDateTime(" << triggered_at_sec << "), "
          << (alert.is_firing ? 1 : 0) << ", '"
          << alert.title << "', '"
          << alert.description << "', '"
          << alert.fingerprint << "', '"
          << alert.model_id << "')";

    return clickhouse_->Execute(query.str());
}

absl::Status AlertService::CacheAlert(const AlertEvent& alert) {
    if (!redis_) {
        return absl::FailedPreconditionError("Redis client not available");
    }

    std::string key = "alert:" + alert.alert_id;
    std::string value = SerializeAlertEvent(alert);

    return redis_->Set(key, value,
        std::chrono::duration_cast<std::chrono::seconds>(config_.cache_ttl));
}

absl::StatusOr<std::vector<AlertEvent>> AlertService::LoadAlertHistory(
    size_t limit,
    size_t offset) const {

    if (!clickhouse_) {
        return absl::FailedPreconditionError("ClickHouse client not available");
    }

    std::ostringstream query;
    query << "SELECT * FROM alerts ORDER BY triggered_at DESC "
          << "LIMIT " << limit << " OFFSET " << offset;

    auto result = clickhouse_->Query(query.str());
    if (!result.ok()) {
        return result.status();
    }

    std::vector<AlertEvent> alerts;
    for (const auto& row : *result) {
        AlertEvent alert;
        alert.alert_id = row.at("alert_id");
        alert.rule_id = row.at("rule_id");
        alert.rule_name = row.at("rule_name");
        alert.severity = StringToAlertSeverity(row.at("severity"));
        alert.title = row.at("title");
        alert.description = row.at("description");
        alert.fingerprint = row.at("fingerprint");
        alert.model_id = row.at("model_id");
        alerts.push_back(alert);
    }

    return alerts;
}

// Factory function
std::unique_ptr<AlertService> CreateAlertService(
    std::shared_ptr<storage::ClickHouseClient> clickhouse,
    std::shared_ptr<storage::RedisClient> redis,
    AlertServiceConfig config) {

    return std::make_unique<AlertService>(
        std::move(clickhouse),
        std::move(redis),
        std::move(config));
}

// Utility functions
std::string NotificationChannelToString(NotificationChannel channel) {
    switch (channel) {
        case NotificationChannel::kSlack: return "slack";
        case NotificationChannel::kPagerDuty: return "pagerduty";
        case NotificationChannel::kEmail: return "email";
        case NotificationChannel::kWebhook: return "webhook";
        case NotificationChannel::kOpsgenie: return "opsgenie";
        case NotificationChannel::kMSTeams: return "msteams";
    }
    return "unknown";
}

NotificationChannel StringToNotificationChannel(const std::string& str) {
    if (str == "slack") return NotificationChannel::kSlack;
    if (str == "pagerduty") return NotificationChannel::kPagerDuty;
    if (str == "email") return NotificationChannel::kEmail;
    if (str == "opsgenie") return NotificationChannel::kOpsgenie;
    if (str == "msteams") return NotificationChannel::kMSTeams;
    return NotificationChannel::kWebhook;
}

std::string SerializeAlertEvent(const AlertEvent& alert) {
    json j;
    j["alert_id"] = alert.alert_id;
    j["rule_id"] = alert.rule_id;
    j["rule_name"] = alert.rule_name;
    j["severity"] = AlertSeverityToString(alert.severity);
    j["triggered_at"] = std::chrono::duration_cast<std::chrono::seconds>(
        alert.triggered_at.time_since_epoch()).count();
    j["is_firing"] = alert.is_firing;
    j["is_resolved"] = alert.is_resolved;
    j["title"] = alert.title;
    j["description"] = alert.description;
    j["metric_name"] = alert.metric_name;
    j["metric_value"] = alert.metric_value;
    j["threshold_value"] = alert.threshold_value;
    j["model_id"] = alert.model_id;
    j["labels"] = alert.labels;
    j["annotations"] = alert.annotations;
    j["fingerprint"] = alert.fingerprint;
    return j.dump();
}

absl::StatusOr<AlertEvent> DeserializeAlertEvent(const std::string& json_str) {
    try {
        json j = json::parse(json_str);

        AlertEvent alert;
        alert.alert_id = j.value("alert_id", "");
        alert.rule_id = j.value("rule_id", "");
        alert.rule_name = j.value("rule_name", "");
        alert.severity = StringToAlertSeverity(j.value("severity", "warning"));
        alert.triggered_at = std::chrono::system_clock::time_point(
            std::chrono::seconds(j.value("triggered_at", 0)));
        alert.is_firing = j.value("is_firing", true);
        alert.is_resolved = j.value("is_resolved", false);
        alert.title = j.value("title", "");
        alert.description = j.value("description", "");
        alert.metric_name = j.value("metric_name", "");
        alert.metric_value = j.value("metric_value", 0.0);
        alert.threshold_value = j.value("threshold_value", 0.0);
        alert.model_id = j.value("model_id", "");
        alert.labels = j.value("labels",
            std::unordered_map<std::string, std::string>{});
        alert.annotations = j.value("annotations",
            std::unordered_map<std::string, std::string>{});
        alert.fingerprint = j.value("fingerprint", "");

        return alert;
    } catch (const json::exception& e) {
        return absl::InvalidArgumentError(
            std::string("Failed to parse alert JSON: ") + e.what());
    }
}

}  // namespace pyflare::alerting
