#pragma once

/// @file deduplicator.h
/// @brief Alert deduplication and grouping
///
/// Provides intelligent alert management:
/// - Fingerprint-based deduplication
/// - Alert grouping by similarity
/// - Suppression during maintenance windows
/// - Rate limiting for noisy alerts

#include <chrono>
#include <functional>
#include <memory>
#include <mutex>
#include <optional>
#include <string>
#include <unordered_map>
#include <unordered_set>
#include <vector>

#include <absl/status/status.h>
#include <absl/status/statusor.h>

#include "processor/alerting/alert_rules.h"

namespace pyflare::alerting {

/// @brief Alert state in deduplicator
enum class AlertState {
    kFiring,     ///< Alert is currently firing
    kResolved,   ///< Alert has been resolved
    kSuppressed, ///< Alert is suppressed (maintenance, rate limit)
    kSilenced    ///< Alert is silenced by user
};

/// @brief Grouped alert representation
struct AlertGroup {
    std::string group_id;
    std::string group_key;  ///< Grouping criteria

    std::vector<AlertEvent> alerts;
    size_t total_count = 0;

    AlertSeverity max_severity = AlertSeverity::kInfo;

    std::chrono::system_clock::time_point first_occurrence;
    std::chrono::system_clock::time_point last_occurrence;

    bool is_firing = false;
};

/// @brief Silence definition
struct Silence {
    std::string silence_id;
    std::string created_by;
    std::string comment;

    /// Matchers for alerts to silence
    std::unordered_map<std::string, std::string> matchers;

    /// Time bounds
    std::chrono::system_clock::time_point starts_at;
    std::chrono::system_clock::time_point ends_at;

    bool is_active = true;
};

/// @brief Maintenance window definition
struct MaintenanceWindow {
    std::string window_id;
    std::string name;
    std::string description;

    /// Affected entities
    std::vector<std::string> model_ids;
    std::vector<std::string> rule_ids;

    /// Time bounds
    std::chrono::system_clock::time_point starts_at;
    std::chrono::system_clock::time_point ends_at;

    /// Suppress all alerts or just notifications
    bool suppress_alerts = true;
    bool suppress_notifications = true;
};

/// @brief Deduplicator configuration
struct DeduplicatorConfig {
    /// Time window for deduplication (same fingerprint)
    std::chrono::minutes dedup_window = std::chrono::minutes(5);

    /// Time window for grouping related alerts
    std::chrono::minutes group_window = std::chrono::minutes(30);

    /// Maximum alerts per group before summarization
    size_t max_alerts_per_group = 100;

    /// Rate limiting configuration
    struct RateLimitConfig {
        bool enabled = true;
        size_t max_alerts_per_minute = 60;
        size_t max_alerts_per_hour = 500;
    };
    RateLimitConfig rate_limit;

    /// Auto-resolve after no activity
    std::chrono::minutes auto_resolve_after = std::chrono::minutes(15);

    /// Alert TTL in storage
    std::chrono::hours alert_ttl = std::chrono::hours(168);  // 7 days

    /// Grouping configuration
    struct GroupingConfig {
        /// Group by these labels
        std::vector<std::string> group_by = {"model_id", "rule_id"};

        /// Minimum alerts to form a group
        size_t min_group_size = 2;
    };
    GroupingConfig grouping;
};

/// @brief Alert deduplicator and grouping engine
///
/// Manages alert lifecycle including deduplication, grouping,
/// silencing, and maintenance windows.
///
/// Example:
/// @code
///   DeduplicatorConfig config;
///   AlertDeduplicator dedup(config);
///   dedup.Initialize();
///
///   // Process incoming alert
///   AlertEvent alert;
///   alert.fingerprint = "rule1:metric1:service=api";
///   auto result = dedup.Process(alert);
///
///   if (result.is_new) {
///       // Send notification
///   }
///
///   // Create silence
///   Silence silence;
///   silence.matchers["rule_id"] = "high_latency";
///   silence.ends_at = now + std::chrono::hours(2);
///   dedup.AddSilence(silence);
/// @endcode
class AlertDeduplicator {
public:
    explicit AlertDeduplicator(DeduplicatorConfig config = {});
    ~AlertDeduplicator();

    // Disable copy
    AlertDeduplicator(const AlertDeduplicator&) = delete;
    AlertDeduplicator& operator=(const AlertDeduplicator&) = delete;

    /// @brief Initialize deduplicator
    absl::Status Initialize();

    // =========================================================================
    // Alert Processing
    // =========================================================================

    /// @brief Process result
    struct ProcessResult {
        bool accepted = false;       ///< Alert was accepted
        bool is_new = false;         ///< First occurrence of this fingerprint
        bool is_duplicate = false;   ///< Duplicate within window
        bool is_suppressed = false;  ///< Suppressed (maintenance/silence/rate)
        bool is_grouped = false;     ///< Added to existing group

        std::string fingerprint;
        std::string group_id;
        std::string suppression_reason;

        AlertState state = AlertState::kFiring;
    };

    /// @brief Process incoming alert
    ProcessResult Process(const AlertEvent& alert);

    /// @brief Process batch of alerts
    std::vector<ProcessResult> ProcessBatch(const std::vector<AlertEvent>& alerts);

    /// @brief Resolve alert by fingerprint
    absl::Status Resolve(const std::string& fingerprint);

    /// @brief Get current state of alert
    absl::StatusOr<AlertState> GetState(const std::string& fingerprint) const;

    // =========================================================================
    // Alert Retrieval
    // =========================================================================

    /// @brief Get active alerts
    std::vector<AlertEvent> GetActiveAlerts() const;

    /// @brief Get alerts by fingerprint
    std::vector<AlertEvent> GetAlertsByFingerprint(
        const std::string& fingerprint) const;

    /// @brief Get alerts by rule ID
    std::vector<AlertEvent> GetAlertsByRule(const std::string& rule_id) const;

    /// @brief Get alerts in time range
    std::vector<AlertEvent> GetAlertsInRange(
        std::chrono::system_clock::time_point start,
        std::chrono::system_clock::time_point end) const;

    // =========================================================================
    // Alert Groups
    // =========================================================================

    /// @brief Get alert groups
    std::vector<AlertGroup> GetGroups() const;

    /// @brief Get specific group
    absl::StatusOr<AlertGroup> GetGroup(const std::string& group_id) const;

    /// @brief Get active groups
    std::vector<AlertGroup> GetActiveGroups() const;

    // =========================================================================
    // Silences
    // =========================================================================

    /// @brief Add silence
    absl::Status AddSilence(const Silence& silence);

    /// @brief Remove silence
    absl::Status RemoveSilence(const std::string& silence_id);

    /// @brief Get active silences
    std::vector<Silence> GetActiveSilences() const;

    /// @brief Check if alert is silenced
    bool IsSilenced(const AlertEvent& alert) const;

    // =========================================================================
    // Maintenance Windows
    // =========================================================================

    /// @brief Add maintenance window
    absl::Status AddMaintenanceWindow(const MaintenanceWindow& window);

    /// @brief Remove maintenance window
    absl::Status RemoveMaintenanceWindow(const std::string& window_id);

    /// @brief Get active maintenance windows
    std::vector<MaintenanceWindow> GetActiveMaintenanceWindows() const;

    /// @brief Check if in maintenance
    bool IsInMaintenance(const AlertEvent& alert) const;

    // =========================================================================
    // Lifecycle Management
    // =========================================================================

    /// @brief Run periodic cleanup
    void Cleanup();

    /// @brief Clear all alerts
    void ClearAlerts();

    /// @brief Clear all silences
    void ClearSilences();

    // =========================================================================
    // Configuration
    // =========================================================================

    /// @brief Update configuration
    void SetConfig(DeduplicatorConfig config);

    /// @brief Get configuration
    const DeduplicatorConfig& GetConfig() const { return config_; }

    // =========================================================================
    // Statistics
    // =========================================================================

    struct Stats {
        size_t total_processed = 0;
        size_t total_deduplicated = 0;
        size_t total_suppressed = 0;
        size_t total_grouped = 0;
        size_t active_alerts = 0;
        size_t active_groups = 0;
        size_t active_silences = 0;
        size_t rate_limited = 0;
    };
    Stats GetStats() const;

    void ResetStats();

private:
    // Deduplication logic
    bool IsDuplicate(const AlertEvent& alert) const;
    std::string ComputeGroupKey(const AlertEvent& alert) const;
    void UpdateGroup(const std::string& group_key, const AlertEvent& alert);

    // Rate limiting
    bool IsRateLimited();
    void RecordAlertForRateLimit();

    // Silence/maintenance matching
    bool MatchesSilence(const AlertEvent& alert, const Silence& silence) const;
    bool MatchesMaintenance(const AlertEvent& alert,
                           const MaintenanceWindow& window) const;

    // ID generation
    std::string GenerateSilenceId() const;
    std::string GenerateGroupId() const;
    std::string GenerateWindowId() const;

    // State cleanup
    void CleanupExpiredAlerts();
    void CleanupExpiredSilences();
    void CleanupExpiredWindows();
    void AutoResolveStaleAlerts();

    DeduplicatorConfig config_;

    // Alert storage
    struct AlertRecord {
        AlertEvent alert;
        AlertState state = AlertState::kFiring;
        std::chrono::system_clock::time_point first_seen;
        std::chrono::system_clock::time_point last_seen;
        size_t occurrence_count = 0;
    };
    std::unordered_map<std::string, AlertRecord> alerts_;  // keyed by fingerprint

    // Groups
    std::unordered_map<std::string, AlertGroup> groups_;  // keyed by group_key

    // Silences
    std::unordered_map<std::string, Silence> silences_;

    // Maintenance windows
    std::unordered_map<std::string, MaintenanceWindow> maintenance_windows_;

    // Rate limiting state
    struct RateLimitState {
        std::vector<std::chrono::system_clock::time_point> recent_alerts;
        std::chrono::system_clock::time_point last_cleanup;
    };
    RateLimitState rate_limit_state_;

    // Statistics
    Stats stats_;

    mutable std::mutex mutex_;
    bool initialized_ = false;
};

/// @brief Factory function
std::unique_ptr<AlertDeduplicator> CreateAlertDeduplicator(
    DeduplicatorConfig config = {});

/// @brief Convert alert state to string
std::string AlertStateToString(AlertState state);

/// @brief Serialize silence to JSON
std::string SerializeSilence(const Silence& silence);

/// @brief Deserialize silence from JSON
absl::StatusOr<Silence> DeserializeSilence(const std::string& json);

/// @brief Serialize maintenance window to JSON
std::string SerializeMaintenanceWindow(const MaintenanceWindow& window);

/// @brief Deserialize maintenance window from JSON
absl::StatusOr<MaintenanceWindow> DeserializeMaintenanceWindow(
    const std::string& json);

}  // namespace pyflare::alerting
