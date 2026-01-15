/// @file deduplicator.cpp
/// @brief Alert deduplication and grouping implementation

#include "processor/alerting/deduplicator.h"

#include <algorithm>
#include <random>
#include <sstream>

#include <nlohmann/json.hpp>

namespace pyflare::alerting {

using json = nlohmann::json;

AlertDeduplicator::AlertDeduplicator(DeduplicatorConfig config)
    : config_(std::move(config)) {}

AlertDeduplicator::~AlertDeduplicator() = default;

absl::Status AlertDeduplicator::Initialize() {
    if (initialized_) {
        return absl::OkStatus();
    }

    initialized_ = true;
    return absl::OkStatus();
}

AlertDeduplicator::ProcessResult AlertDeduplicator::Process(
    const AlertEvent& alert) {

    std::lock_guard<std::mutex> lock(mutex_);

    ProcessResult result;
    result.fingerprint = alert.fingerprint;

    auto now = std::chrono::system_clock::now();

    // Update stats
    stats_.total_processed++;

    // Check rate limiting first
    if (config_.rate_limit.enabled && IsRateLimited()) {
        result.is_suppressed = true;
        result.suppression_reason = "Rate limited";
        stats_.rate_limited++;
        stats_.total_suppressed++;
        return result;
    }

    // Check maintenance windows
    if (IsInMaintenance(alert)) {
        result.is_suppressed = true;
        result.suppression_reason = "In maintenance window";
        stats_.total_suppressed++;
        return result;
    }

    // Check silences
    if (IsSilenced(alert)) {
        result.is_suppressed = true;
        result.suppression_reason = "Silenced";
        result.state = AlertState::kSilenced;
        stats_.total_suppressed++;
        return result;
    }

    // Check for duplicate
    auto it = alerts_.find(alert.fingerprint);
    if (it != alerts_.end()) {
        auto time_since_last = now - it->second.last_seen;
        if (time_since_last < config_.dedup_window) {
            // Duplicate within window
            result.is_duplicate = true;
            result.accepted = true;
            it->second.last_seen = now;
            it->second.occurrence_count++;
            it->second.alert = alert;  // Update with latest
            stats_.total_deduplicated++;

            // Still update group
            std::string group_key = ComputeGroupKey(alert);
            UpdateGroup(group_key, alert);
            result.group_id = group_key;

            return result;
        } else {
            // Same fingerprint but outside dedup window - treat as new firing
            result.is_new = true;
        }
    } else {
        result.is_new = true;
    }

    // Record alert
    AlertRecord record;
    record.alert = alert;
    record.state = AlertState::kFiring;
    record.first_seen = now;
    record.last_seen = now;
    record.occurrence_count = 1;

    alerts_[alert.fingerprint] = record;
    stats_.active_alerts++;

    // Add to group
    std::string group_key = ComputeGroupKey(alert);
    UpdateGroup(group_key, alert);
    result.group_id = group_key;
    result.is_grouped = groups_.count(group_key) > 0 &&
                        groups_[group_key].alerts.size() > 1;
    if (result.is_grouped) {
        stats_.total_grouped++;
    }

    result.accepted = true;
    result.state = AlertState::kFiring;

    // Record for rate limiting
    RecordAlertForRateLimit();

    return result;
}

std::vector<AlertDeduplicator::ProcessResult> AlertDeduplicator::ProcessBatch(
    const std::vector<AlertEvent>& alerts) {

    std::vector<ProcessResult> results;
    results.reserve(alerts.size());

    for (const auto& alert : alerts) {
        results.push_back(Process(alert));
    }

    return results;
}

absl::Status AlertDeduplicator::Resolve(const std::string& fingerprint) {
    std::lock_guard<std::mutex> lock(mutex_);

    auto it = alerts_.find(fingerprint);
    if (it == alerts_.end()) {
        return absl::NotFoundError("Alert not found: " + fingerprint);
    }

    it->second.state = AlertState::kResolved;
    it->second.alert.is_resolved = true;
    it->second.alert.resolved_at = std::chrono::system_clock::now();
    it->second.alert.is_firing = false;

    stats_.active_alerts--;

    // Update group
    for (auto& [key, group] : groups_) {
        for (auto& alert : group.alerts) {
            if (alert.fingerprint == fingerprint) {
                alert.is_resolved = true;
                alert.is_firing = false;
            }
        }
        // Check if group still has firing alerts
        group.is_firing = std::any_of(group.alerts.begin(), group.alerts.end(),
                                      [](const AlertEvent& a) { return a.is_firing; });
    }

    return absl::OkStatus();
}

absl::StatusOr<AlertState> AlertDeduplicator::GetState(
    const std::string& fingerprint) const {

    std::lock_guard<std::mutex> lock(mutex_);

    auto it = alerts_.find(fingerprint);
    if (it == alerts_.end()) {
        return absl::NotFoundError("Alert not found: " + fingerprint);
    }

    return it->second.state;
}

std::vector<AlertEvent> AlertDeduplicator::GetActiveAlerts() const {
    std::lock_guard<std::mutex> lock(mutex_);

    std::vector<AlertEvent> active;
    for (const auto& [fp, record] : alerts_) {
        if (record.state == AlertState::kFiring) {
            active.push_back(record.alert);
        }
    }

    return active;
}

std::vector<AlertEvent> AlertDeduplicator::GetAlertsByFingerprint(
    const std::string& fingerprint) const {

    std::lock_guard<std::mutex> lock(mutex_);

    std::vector<AlertEvent> result;
    auto it = alerts_.find(fingerprint);
    if (it != alerts_.end()) {
        result.push_back(it->second.alert);
    }

    return result;
}

std::vector<AlertEvent> AlertDeduplicator::GetAlertsByRule(
    const std::string& rule_id) const {

    std::lock_guard<std::mutex> lock(mutex_);

    std::vector<AlertEvent> result;
    for (const auto& [fp, record] : alerts_) {
        if (record.alert.rule_id == rule_id) {
            result.push_back(record.alert);
        }
    }

    return result;
}

std::vector<AlertEvent> AlertDeduplicator::GetAlertsInRange(
    std::chrono::system_clock::time_point start,
    std::chrono::system_clock::time_point end) const {

    std::lock_guard<std::mutex> lock(mutex_);

    std::vector<AlertEvent> result;
    for (const auto& [fp, record] : alerts_) {
        if (record.alert.triggered_at >= start &&
            record.alert.triggered_at <= end) {
            result.push_back(record.alert);
        }
    }

    return result;
}

std::vector<AlertGroup> AlertDeduplicator::GetGroups() const {
    std::lock_guard<std::mutex> lock(mutex_);

    std::vector<AlertGroup> result;
    result.reserve(groups_.size());

    for (const auto& [key, group] : groups_) {
        result.push_back(group);
    }

    return result;
}

absl::StatusOr<AlertGroup> AlertDeduplicator::GetGroup(
    const std::string& group_id) const {

    std::lock_guard<std::mutex> lock(mutex_);

    auto it = groups_.find(group_id);
    if (it == groups_.end()) {
        return absl::NotFoundError("Group not found: " + group_id);
    }

    return it->second;
}

std::vector<AlertGroup> AlertDeduplicator::GetActiveGroups() const {
    std::lock_guard<std::mutex> lock(mutex_);

    std::vector<AlertGroup> result;
    for (const auto& [key, group] : groups_) {
        if (group.is_firing) {
            result.push_back(group);
        }
    }

    return result;
}

absl::Status AlertDeduplicator::AddSilence(const Silence& silence) {
    std::lock_guard<std::mutex> lock(mutex_);

    Silence new_silence = silence;
    if (new_silence.silence_id.empty()) {
        new_silence.silence_id = GenerateSilenceId();
    }

    silences_[new_silence.silence_id] = new_silence;
    stats_.active_silences++;

    return absl::OkStatus();
}

absl::Status AlertDeduplicator::RemoveSilence(const std::string& silence_id) {
    std::lock_guard<std::mutex> lock(mutex_);

    auto it = silences_.find(silence_id);
    if (it == silences_.end()) {
        return absl::NotFoundError("Silence not found: " + silence_id);
    }

    silences_.erase(it);
    stats_.active_silences--;

    return absl::OkStatus();
}

std::vector<Silence> AlertDeduplicator::GetActiveSilences() const {
    std::lock_guard<std::mutex> lock(mutex_);

    auto now = std::chrono::system_clock::now();
    std::vector<Silence> active;

    for (const auto& [id, silence] : silences_) {
        if (silence.is_active &&
            silence.starts_at <= now &&
            silence.ends_at > now) {
            active.push_back(silence);
        }
    }

    return active;
}

bool AlertDeduplicator::IsSilenced(const AlertEvent& alert) const {
    auto now = std::chrono::system_clock::now();

    for (const auto& [id, silence] : silences_) {
        if (!silence.is_active ||
            silence.starts_at > now ||
            silence.ends_at <= now) {
            continue;
        }

        if (MatchesSilence(alert, silence)) {
            return true;
        }
    }

    return false;
}

absl::Status AlertDeduplicator::AddMaintenanceWindow(
    const MaintenanceWindow& window) {

    std::lock_guard<std::mutex> lock(mutex_);

    MaintenanceWindow new_window = window;
    if (new_window.window_id.empty()) {
        new_window.window_id = GenerateWindowId();
    }

    maintenance_windows_[new_window.window_id] = new_window;

    return absl::OkStatus();
}

absl::Status AlertDeduplicator::RemoveMaintenanceWindow(
    const std::string& window_id) {

    std::lock_guard<std::mutex> lock(mutex_);

    auto it = maintenance_windows_.find(window_id);
    if (it == maintenance_windows_.end()) {
        return absl::NotFoundError("Maintenance window not found: " + window_id);
    }

    maintenance_windows_.erase(it);
    return absl::OkStatus();
}

std::vector<MaintenanceWindow> AlertDeduplicator::GetActiveMaintenanceWindows() const {
    std::lock_guard<std::mutex> lock(mutex_);

    auto now = std::chrono::system_clock::now();
    std::vector<MaintenanceWindow> active;

    for (const auto& [id, window] : maintenance_windows_) {
        if (window.starts_at <= now && window.ends_at > now) {
            active.push_back(window);
        }
    }

    return active;
}

bool AlertDeduplicator::IsInMaintenance(const AlertEvent& alert) const {
    auto now = std::chrono::system_clock::now();

    for (const auto& [id, window] : maintenance_windows_) {
        if (window.starts_at > now || window.ends_at <= now) {
            continue;
        }

        if (!window.suppress_alerts) {
            continue;
        }

        if (MatchesMaintenance(alert, window)) {
            return true;
        }
    }

    return false;
}

void AlertDeduplicator::Cleanup() {
    std::lock_guard<std::mutex> lock(mutex_);

    CleanupExpiredAlerts();
    CleanupExpiredSilences();
    CleanupExpiredWindows();
    AutoResolveStaleAlerts();

    // Update active counts
    stats_.active_alerts = 0;
    for (const auto& [fp, record] : alerts_) {
        if (record.state == AlertState::kFiring) {
            stats_.active_alerts++;
        }
    }

    stats_.active_groups = 0;
    for (const auto& [key, group] : groups_) {
        if (group.is_firing) {
            stats_.active_groups++;
        }
    }
}

void AlertDeduplicator::ClearAlerts() {
    std::lock_guard<std::mutex> lock(mutex_);

    alerts_.clear();
    groups_.clear();
    stats_.active_alerts = 0;
    stats_.active_groups = 0;
}

void AlertDeduplicator::ClearSilences() {
    std::lock_guard<std::mutex> lock(mutex_);

    silences_.clear();
    stats_.active_silences = 0;
}

void AlertDeduplicator::SetConfig(DeduplicatorConfig config) {
    std::lock_guard<std::mutex> lock(mutex_);
    config_ = std::move(config);
}

AlertDeduplicator::Stats AlertDeduplicator::GetStats() const {
    std::lock_guard<std::mutex> lock(mutex_);
    return stats_;
}

void AlertDeduplicator::ResetStats() {
    std::lock_guard<std::mutex> lock(mutex_);

    // Keep active counts
    size_t active_alerts = stats_.active_alerts;
    size_t active_groups = stats_.active_groups;
    size_t active_silences = stats_.active_silences;

    stats_ = Stats{};
    stats_.active_alerts = active_alerts;
    stats_.active_groups = active_groups;
    stats_.active_silences = active_silences;
}

// Private methods

bool AlertDeduplicator::IsDuplicate(const AlertEvent& alert) const {
    auto it = alerts_.find(alert.fingerprint);
    if (it == alerts_.end()) {
        return false;
    }

    auto now = std::chrono::system_clock::now();
    auto time_since_last = now - it->second.last_seen;

    return time_since_last < config_.dedup_window;
}

std::string AlertDeduplicator::ComputeGroupKey(const AlertEvent& alert) const {
    std::ostringstream key;

    for (const auto& label : config_.grouping.group_by) {
        if (label == "model_id") {
            key << "model_id=" << alert.model_id << ";";
        } else if (label == "rule_id") {
            key << "rule_id=" << alert.rule_id << ";";
        } else if (label == "severity") {
            key << "severity=" << AlertSeverityToString(alert.severity) << ";";
        } else if (alert.labels.count(label) > 0) {
            key << label << "=" << alert.labels.at(label) << ";";
        }
    }

    return key.str();
}

void AlertDeduplicator::UpdateGroup(const std::string& group_key,
                                    const AlertEvent& alert) {

    auto& group = groups_[group_key];
    auto now = std::chrono::system_clock::now();

    if (group.group_id.empty()) {
        group.group_id = GenerateGroupId();
        group.group_key = group_key;
        group.first_occurrence = now;
        stats_.active_groups++;
    }

    group.last_occurrence = now;
    group.total_count++;

    // Update max severity
    if (static_cast<int>(alert.severity) > static_cast<int>(group.max_severity)) {
        group.max_severity = alert.severity;
    }

    // Add alert if not at limit
    if (group.alerts.size() < config_.max_alerts_per_group) {
        // Check if alert already in group (by fingerprint)
        bool exists = false;
        for (auto& existing : group.alerts) {
            if (existing.fingerprint == alert.fingerprint) {
                existing = alert;  // Update
                exists = true;
                break;
            }
        }
        if (!exists) {
            group.alerts.push_back(alert);
        }
    }

    // Update firing status
    group.is_firing = std::any_of(group.alerts.begin(), group.alerts.end(),
                                  [](const AlertEvent& a) { return a.is_firing; });
}

bool AlertDeduplicator::IsRateLimited() {
    auto now = std::chrono::system_clock::now();

    // Clean up old entries
    auto minute_ago = now - std::chrono::minutes(1);
    auto hour_ago = now - std::chrono::hours(1);

    rate_limit_state_.recent_alerts.erase(
        std::remove_if(rate_limit_state_.recent_alerts.begin(),
                       rate_limit_state_.recent_alerts.end(),
                       [hour_ago](const auto& t) { return t < hour_ago; }),
        rate_limit_state_.recent_alerts.end());

    // Count recent alerts
    size_t last_minute = 0;
    size_t last_hour = rate_limit_state_.recent_alerts.size();

    for (const auto& t : rate_limit_state_.recent_alerts) {
        if (t >= minute_ago) {
            last_minute++;
        }
    }

    return last_minute >= config_.rate_limit.max_alerts_per_minute ||
           last_hour >= config_.rate_limit.max_alerts_per_hour;
}

void AlertDeduplicator::RecordAlertForRateLimit() {
    rate_limit_state_.recent_alerts.push_back(std::chrono::system_clock::now());
}

bool AlertDeduplicator::MatchesSilence(const AlertEvent& alert,
                                       const Silence& silence) const {

    for (const auto& [key, value] : silence.matchers) {
        if (key == "rule_id" && alert.rule_id != value) {
            return false;
        }
        if (key == "model_id" && alert.model_id != value) {
            return false;
        }
        if (key == "metric_name" && alert.metric_name != value) {
            return false;
        }
        if (key == "severity" &&
            AlertSeverityToString(alert.severity) != value) {
            return false;
        }
        if (alert.labels.count(key) > 0 && alert.labels.at(key) != value) {
            return false;
        }
    }

    return true;
}

bool AlertDeduplicator::MatchesMaintenance(const AlertEvent& alert,
                                           const MaintenanceWindow& window) const {

    // Check model_ids
    if (!window.model_ids.empty()) {
        bool model_match = std::find(window.model_ids.begin(),
                                     window.model_ids.end(),
                                     alert.model_id) != window.model_ids.end();
        if (model_match) {
            return true;
        }
    }

    // Check rule_ids
    if (!window.rule_ids.empty()) {
        bool rule_match = std::find(window.rule_ids.begin(),
                                    window.rule_ids.end(),
                                    alert.rule_id) != window.rule_ids.end();
        if (rule_match) {
            return true;
        }
    }

    // If no specific entities, applies to all
    return window.model_ids.empty() && window.rule_ids.empty();
}

std::string AlertDeduplicator::GenerateSilenceId() const {
    static std::random_device rd;
    static std::mt19937 gen(rd());
    static std::uniform_int_distribution<uint64_t> dis;

    std::ostringstream id;
    id << "silence-" << std::hex << dis(gen);
    return id.str();
}

std::string AlertDeduplicator::GenerateGroupId() const {
    static std::random_device rd;
    static std::mt19937 gen(rd());
    static std::uniform_int_distribution<uint64_t> dis;

    std::ostringstream id;
    id << "group-" << std::hex << dis(gen);
    return id.str();
}

std::string AlertDeduplicator::GenerateWindowId() const {
    static std::random_device rd;
    static std::mt19937 gen(rd());
    static std::uniform_int_distribution<uint64_t> dis;

    std::ostringstream id;
    id << "maint-" << std::hex << dis(gen);
    return id.str();
}

void AlertDeduplicator::CleanupExpiredAlerts() {
    auto now = std::chrono::system_clock::now();
    auto cutoff = now - config_.alert_ttl;

    for (auto it = alerts_.begin(); it != alerts_.end();) {
        if (it->second.last_seen < cutoff) {
            it = alerts_.erase(it);
        } else {
            ++it;
        }
    }

    // Cleanup groups with no alerts
    for (auto it = groups_.begin(); it != groups_.end();) {
        it->second.alerts.erase(
            std::remove_if(it->second.alerts.begin(), it->second.alerts.end(),
                           [cutoff](const AlertEvent& a) {
                               return a.triggered_at < cutoff;
                           }),
            it->second.alerts.end());

        if (it->second.alerts.empty()) {
            it = groups_.erase(it);
        } else {
            ++it;
        }
    }
}

void AlertDeduplicator::CleanupExpiredSilences() {
    auto now = std::chrono::system_clock::now();

    for (auto it = silences_.begin(); it != silences_.end();) {
        if (it->second.ends_at <= now) {
            it = silences_.erase(it);
        } else {
            ++it;
        }
    }
}

void AlertDeduplicator::CleanupExpiredWindows() {
    auto now = std::chrono::system_clock::now();

    for (auto it = maintenance_windows_.begin(); it != maintenance_windows_.end();) {
        if (it->second.ends_at <= now) {
            it = maintenance_windows_.erase(it);
        } else {
            ++it;
        }
    }
}

void AlertDeduplicator::AutoResolveStaleAlerts() {
    auto now = std::chrono::system_clock::now();
    auto stale_cutoff = now - config_.auto_resolve_after;

    for (auto& [fp, record] : alerts_) {
        if (record.state == AlertState::kFiring &&
            record.last_seen < stale_cutoff) {
            record.state = AlertState::kResolved;
            record.alert.is_resolved = true;
            record.alert.is_firing = false;
            record.alert.resolved_at = now;
        }
    }
}

// Factory function
std::unique_ptr<AlertDeduplicator> CreateAlertDeduplicator(
    DeduplicatorConfig config) {

    return std::make_unique<AlertDeduplicator>(std::move(config));
}

// Utility functions
std::string AlertStateToString(AlertState state) {
    switch (state) {
        case AlertState::kFiring: return "firing";
        case AlertState::kResolved: return "resolved";
        case AlertState::kSuppressed: return "suppressed";
        case AlertState::kSilenced: return "silenced";
    }
    return "unknown";
}

std::string SerializeSilence(const Silence& silence) {
    json j;
    j["silence_id"] = silence.silence_id;
    j["created_by"] = silence.created_by;
    j["comment"] = silence.comment;
    j["matchers"] = silence.matchers;
    j["starts_at"] = std::chrono::duration_cast<std::chrono::seconds>(
        silence.starts_at.time_since_epoch()).count();
    j["ends_at"] = std::chrono::duration_cast<std::chrono::seconds>(
        silence.ends_at.time_since_epoch()).count();
    j["is_active"] = silence.is_active;
    return j.dump();
}

absl::StatusOr<Silence> DeserializeSilence(const std::string& json_str) {
    try {
        json j = json::parse(json_str);

        Silence silence;
        silence.silence_id = j.value("silence_id", "");
        silence.created_by = j.value("created_by", "");
        silence.comment = j.value("comment", "");
        silence.matchers = j.value("matchers",
            std::unordered_map<std::string, std::string>{});
        silence.starts_at = std::chrono::system_clock::time_point(
            std::chrono::seconds(j.value("starts_at", 0)));
        silence.ends_at = std::chrono::system_clock::time_point(
            std::chrono::seconds(j.value("ends_at", 0)));
        silence.is_active = j.value("is_active", true);

        return silence;
    } catch (const json::exception& e) {
        return absl::InvalidArgumentError(
            std::string("Failed to parse silence JSON: ") + e.what());
    }
}

std::string SerializeMaintenanceWindow(const MaintenanceWindow& window) {
    json j;
    j["window_id"] = window.window_id;
    j["name"] = window.name;
    j["description"] = window.description;
    j["model_ids"] = window.model_ids;
    j["rule_ids"] = window.rule_ids;
    j["starts_at"] = std::chrono::duration_cast<std::chrono::seconds>(
        window.starts_at.time_since_epoch()).count();
    j["ends_at"] = std::chrono::duration_cast<std::chrono::seconds>(
        window.ends_at.time_since_epoch()).count();
    j["suppress_alerts"] = window.suppress_alerts;
    j["suppress_notifications"] = window.suppress_notifications;
    return j.dump();
}

absl::StatusOr<MaintenanceWindow> DeserializeMaintenanceWindow(
    const std::string& json_str) {

    try {
        json j = json::parse(json_str);

        MaintenanceWindow window;
        window.window_id = j.value("window_id", "");
        window.name = j.value("name", "");
        window.description = j.value("description", "");
        window.model_ids = j.value("model_ids", std::vector<std::string>{});
        window.rule_ids = j.value("rule_ids", std::vector<std::string>{});
        window.starts_at = std::chrono::system_clock::time_point(
            std::chrono::seconds(j.value("starts_at", 0)));
        window.ends_at = std::chrono::system_clock::time_point(
            std::chrono::seconds(j.value("ends_at", 0)));
        window.suppress_alerts = j.value("suppress_alerts", true);
        window.suppress_notifications = j.value("suppress_notifications", true);

        return window;
    } catch (const json::exception& e) {
        return absl::InvalidArgumentError(
            std::string("Failed to parse maintenance window JSON: ") + e.what());
    }
}

}  // namespace pyflare::alerting
