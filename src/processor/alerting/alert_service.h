#pragma once

/// @file alert_service.h
/// @brief Alert service for orchestrating alerting pipeline
///
/// Provides end-to-end alert management:
/// - Metric ingestion and rule evaluation
/// - Alert deduplication and grouping
/// - Multi-channel notifications
/// - Alert history and querying

#include <atomic>
#include <chrono>
#include <condition_variable>
#include <functional>
#include <memory>
#include <mutex>
#include <queue>
#include <string>
#include <thread>
#include <unordered_map>
#include <vector>

#include <absl/status/status.h>
#include <absl/status/statusor.h>

#include "processor/alerting/alert_rules.h"
#include "processor/alerting/deduplicator.h"
#include "storage/clickhouse/client.h"
#include "storage/redis/client.h"

namespace pyflare::alerting {

/// @brief Notification channel types
enum class NotificationChannel {
    kSlack,
    kPagerDuty,
    kEmail,
    kWebhook,
    kOpsgenie,
    kMSTeams
};

/// @brief Notification configuration
struct NotificationConfig {
    NotificationChannel channel = NotificationChannel::kWebhook;
    std::string name;
    bool enabled = true;

    /// Channel-specific settings
    std::string webhook_url;
    std::string api_key;
    std::string routing_key;  // PagerDuty, Opsgenie
    std::string email_recipients;  // Comma-separated

    /// Severity filter (minimum severity to notify)
    AlertSeverity min_severity = AlertSeverity::kWarning;

    /// Rate limiting for notifications
    size_t max_per_hour = 100;

    /// Template customization
    std::string title_template;
    std::string body_template;
};

/// @brief Notification result
struct NotificationResult {
    bool success = false;
    std::string channel_name;
    NotificationChannel channel;
    std::string error_message;
    std::chrono::system_clock::time_point sent_at;
    std::chrono::milliseconds latency{0};
};

/// @brief Alert service configuration
struct AlertServiceConfig {
    /// Rules engine config
    bool enable_rules_engine = true;

    /// Deduplicator config
    DeduplicatorConfig dedup_config;

    /// Evaluation interval
    std::chrono::seconds eval_interval = std::chrono::seconds(30);

    /// Persist alerts to ClickHouse
    bool persist_alerts = true;

    /// Cache alerts in Redis
    bool cache_alerts = true;

    /// Alert cache TTL
    std::chrono::hours cache_ttl = std::chrono::hours(24);

    /// Maximum alerts to retain
    size_t max_alert_history = 10000;

    /// Notification configs
    std::vector<NotificationConfig> notification_channels;

    /// Send notifications on new alerts
    bool notify_on_new = true;

    /// Send notifications on resolved
    bool notify_on_resolved = true;

    /// Background worker threads
    size_t worker_threads = 2;
};

/// @brief Alert service
///
/// Orchestrates the complete alerting pipeline including rule evaluation,
/// deduplication, and notifications.
///
/// Example:
/// @code
///   AlertServiceConfig config;
///   config.dedup_config.dedup_window = std::chrono::minutes(5);
///
///   // Add Slack notification
///   NotificationConfig slack;
///   slack.channel = NotificationChannel::kSlack;
///   slack.webhook_url = "https://hooks.slack.com/...";
///   config.notification_channels.push_back(slack);
///
///   AlertService service(clickhouse, redis, config);
///   service.Initialize();
///   service.Start();
///
///   // Add alert rule
///   auto rule = CreateThresholdRule("high_latency", "p99_latency",
///                                   ComparisonOp::kGreaterThan, 1000);
///   service.AddRule(rule);
///
///   // Ingest metrics (will evaluate rules)
///   std::vector<MetricValue> metrics = GetMetrics();
///   service.IngestMetrics(metrics);
///
///   // Query alerts
///   auto alerts = service.GetActiveAlerts();
/// @endcode
class AlertService {
public:
    AlertService(
        std::shared_ptr<storage::ClickHouseClient> clickhouse,
        std::shared_ptr<storage::RedisClient> redis,
        AlertServiceConfig config = {});
    ~AlertService();

    // Disable copy
    AlertService(const AlertService&) = delete;
    AlertService& operator=(const AlertService&) = delete;

    /// @brief Initialize service
    absl::Status Initialize();

    /// @brief Start background workers
    absl::Status Start();

    /// @brief Stop service
    void Stop();

    /// @brief Check if running
    bool IsRunning() const { return running_.load(); }

    // =========================================================================
    // Metric Ingestion
    // =========================================================================

    /// @brief Ingest metrics and evaluate rules
    std::vector<AlertEvent> IngestMetrics(const std::vector<MetricValue>& metrics);

    /// @brief Ingest single metric
    std::vector<AlertEvent> IngestMetric(const MetricValue& metric);

    // =========================================================================
    // Alert Management
    // =========================================================================

    /// @brief Manually fire an alert
    absl::StatusOr<AlertEvent> FireAlert(
        const std::string& title,
        const std::string& description,
        AlertSeverity severity = AlertSeverity::kWarning,
        const std::unordered_map<std::string, std::string>& labels = {});

    /// @brief Resolve alert
    absl::Status ResolveAlert(const std::string& fingerprint);

    /// @brief Acknowledge alert (suppresses notifications)
    absl::Status AcknowledgeAlert(const std::string& alert_id,
                                  const std::string& acknowledged_by);

    // =========================================================================
    // Rule Management (delegates to AlertRulesEngine)
    // =========================================================================

    /// @brief Add alert rule
    absl::Status AddRule(const AlertRule& rule);

    /// @brief Update rule
    absl::Status UpdateRule(const AlertRule& rule);

    /// @brief Remove rule
    absl::Status RemoveRule(const std::string& rule_id);

    /// @brief Get rule
    absl::StatusOr<AlertRule> GetRule(const std::string& rule_id) const;

    /// @brief List rules
    std::vector<AlertRule> ListRules() const;

    /// @brief Enable/disable rule
    absl::Status SetRuleEnabled(const std::string& rule_id, bool enabled);

    // =========================================================================
    // Silence Management (delegates to AlertDeduplicator)
    // =========================================================================

    /// @brief Add silence
    absl::Status AddSilence(const Silence& silence);

    /// @brief Remove silence
    absl::Status RemoveSilence(const std::string& silence_id);

    /// @brief Get active silences
    std::vector<Silence> GetActiveSilences() const;

    // =========================================================================
    // Maintenance Window Management
    // =========================================================================

    /// @brief Add maintenance window
    absl::Status AddMaintenanceWindow(const MaintenanceWindow& window);

    /// @brief Remove maintenance window
    absl::Status RemoveMaintenanceWindow(const std::string& window_id);

    /// @brief Get active maintenance windows
    std::vector<MaintenanceWindow> GetActiveMaintenanceWindows() const;

    // =========================================================================
    // Alert Querying
    // =========================================================================

    /// @brief Get active alerts
    std::vector<AlertEvent> GetActiveAlerts() const;

    /// @brief Get alert groups
    std::vector<AlertGroup> GetAlertGroups() const;

    /// @brief Get alerts by rule
    std::vector<AlertEvent> GetAlertsByRule(const std::string& rule_id) const;

    /// @brief Get alerts in time range
    std::vector<AlertEvent> GetAlertsInRange(
        std::chrono::system_clock::time_point start,
        std::chrono::system_clock::time_point end) const;

    /// @brief Get alert history
    absl::StatusOr<std::vector<AlertEvent>> GetAlertHistory(
        size_t limit = 100,
        size_t offset = 0) const;

    // =========================================================================
    // Notification Management
    // =========================================================================

    /// @brief Add notification channel
    absl::Status AddNotificationChannel(const NotificationConfig& config);

    /// @brief Remove notification channel
    absl::Status RemoveNotificationChannel(const std::string& name);

    /// @brief Test notification channel
    absl::StatusOr<NotificationResult> TestNotificationChannel(
        const std::string& name);

    /// @brief Get notification history
    std::vector<NotificationResult> GetNotificationHistory(size_t limit = 100) const;

    // =========================================================================
    // Callbacks
    // =========================================================================

    /// @brief Register callback for new alerts
    void OnAlert(std::function<void(const AlertEvent&)> callback);

    /// @brief Register callback for resolved alerts
    void OnResolved(std::function<void(const AlertEvent&)> callback);

    /// @brief Clear callbacks
    void ClearCallbacks();

    // =========================================================================
    // Statistics
    // =========================================================================

    struct Stats {
        // Processing stats
        size_t metrics_ingested = 0;
        size_t rules_evaluated = 0;
        size_t alerts_generated = 0;
        size_t alerts_deduplicated = 0;
        size_t alerts_suppressed = 0;

        // Notification stats
        size_t notifications_sent = 0;
        size_t notifications_failed = 0;

        // Current state
        size_t active_alerts = 0;
        size_t active_rules = 0;
        size_t active_silences = 0;

        // Timing
        double avg_eval_time_ms = 0.0;
        double avg_notification_time_ms = 0.0;
        std::chrono::system_clock::time_point last_eval;
    };
    Stats GetStats() const;

    void ResetStats();

    // =========================================================================
    // Configuration
    // =========================================================================

    /// @brief Update configuration
    void SetConfig(AlertServiceConfig config);

    /// @brief Get configuration
    const AlertServiceConfig& GetConfig() const { return config_; }

private:
    // Background worker
    void EvaluationLoop();
    void NotificationLoop();
    void CleanupLoop();

    // Notification helpers
    std::vector<NotificationResult> SendNotifications(const AlertEvent& alert);
    NotificationResult SendSlackNotification(const AlertEvent& alert,
                                             const NotificationConfig& config);
    NotificationResult SendPagerDutyNotification(const AlertEvent& alert,
                                                  const NotificationConfig& config);
    NotificationResult SendWebhookNotification(const AlertEvent& alert,
                                                const NotificationConfig& config);
    NotificationResult SendEmailNotification(const AlertEvent& alert,
                                              const NotificationConfig& config);

    // Template rendering
    std::string RenderTemplate(const std::string& tmpl,
                               const AlertEvent& alert) const;

    // Persistence helpers
    absl::Status PersistAlert(const AlertEvent& alert);
    absl::Status CacheAlert(const AlertEvent& alert);
    absl::StatusOr<std::vector<AlertEvent>> LoadAlertHistory(
        size_t limit, size_t offset) const;

    // Storage
    std::shared_ptr<storage::ClickHouseClient> clickhouse_;
    std::shared_ptr<storage::RedisClient> redis_;
    AlertServiceConfig config_;

    // Components
    std::unique_ptr<AlertRulesEngine> rules_engine_;
    std::unique_ptr<AlertDeduplicator> deduplicator_;

    // Notification channels
    std::unordered_map<std::string, NotificationConfig> notification_channels_;
    mutable std::mutex channels_mutex_;

    // Notification queue
    std::queue<AlertEvent> notification_queue_;
    std::mutex queue_mutex_;
    std::condition_variable queue_cv_;

    // Notification history
    std::vector<NotificationResult> notification_history_;
    mutable std::mutex history_mutex_;

    // Background workers
    std::vector<std::thread> workers_;
    std::atomic<bool> running_{false};

    // Callbacks
    std::vector<std::function<void(const AlertEvent&)>> alert_callbacks_;
    std::vector<std::function<void(const AlertEvent&)>> resolved_callbacks_;
    mutable std::mutex callbacks_mutex_;

    // Statistics
    Stats stats_;
    mutable std::mutex stats_mutex_;

    bool initialized_ = false;
};

/// @brief Factory function
std::unique_ptr<AlertService> CreateAlertService(
    std::shared_ptr<storage::ClickHouseClient> clickhouse,
    std::shared_ptr<storage::RedisClient> redis,
    AlertServiceConfig config = {});

/// @brief Convert notification channel to string
std::string NotificationChannelToString(NotificationChannel channel);

/// @brief Convert string to notification channel
NotificationChannel StringToNotificationChannel(const std::string& str);

/// @brief Serialize alert event to JSON
std::string SerializeAlertEvent(const AlertEvent& alert);

/// @brief Deserialize alert event from JSON
absl::StatusOr<AlertEvent> DeserializeAlertEvent(const std::string& json);

}  // namespace pyflare::alerting
