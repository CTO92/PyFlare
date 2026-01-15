/// @file alert_rules.cpp
/// @brief Alert rules engine implementation

#include "processor/alerting/alert_rules.h"

#include <algorithm>
#include <cmath>
#include <fstream>
#include <numeric>
#include <random>
#include <sstream>

#include <nlohmann/json.hpp>

namespace pyflare::alerting {

using json = nlohmann::json;

AlertRulesEngine::AlertRulesEngine() = default;
AlertRulesEngine::~AlertRulesEngine() = default;

absl::Status AlertRulesEngine::Initialize() {
    if (initialized_) {
        return absl::OkStatus();
    }

    initialized_ = true;
    return absl::OkStatus();
}

absl::Status AlertRulesEngine::AddRule(const AlertRule& rule) {
    auto validation = ValidateRule(rule);
    if (!validation.ok()) {
        return validation;
    }

    std::lock_guard<std::mutex> lock(rules_mutex_);

    if (rules_.count(rule.rule_id) > 0) {
        return absl::AlreadyExistsError("Rule already exists: " + rule.rule_id);
    }

    AlertRule new_rule = rule;
    new_rule.created_at = std::chrono::system_clock::now();
    new_rule.updated_at = new_rule.created_at;

    rules_[rule.rule_id] = new_rule;

    // Update stats
    {
        std::lock_guard<std::mutex> stats_lock(stats_mutex_);
        stats_.total_rules++;
        if (new_rule.enabled) {
            stats_.enabled_rules++;
        }
        stats_.rules_by_type[new_rule.type]++;
    }

    return absl::OkStatus();
}

absl::Status AlertRulesEngine::UpdateRule(const AlertRule& rule) {
    auto validation = ValidateRule(rule);
    if (!validation.ok()) {
        return validation;
    }

    std::lock_guard<std::mutex> lock(rules_mutex_);

    auto it = rules_.find(rule.rule_id);
    if (it == rules_.end()) {
        return absl::NotFoundError("Rule not found: " + rule.rule_id);
    }

    bool was_enabled = it->second.enabled;
    RuleType old_type = it->second.type;

    AlertRule updated_rule = rule;
    updated_rule.created_at = it->second.created_at;
    updated_rule.updated_at = std::chrono::system_clock::now();

    it->second = updated_rule;

    // Update stats
    {
        std::lock_guard<std::mutex> stats_lock(stats_mutex_);
        if (was_enabled && !rule.enabled) {
            stats_.enabled_rules--;
        } else if (!was_enabled && rule.enabled) {
            stats_.enabled_rules++;
        }
        if (old_type != rule.type) {
            stats_.rules_by_type[old_type]--;
            stats_.rules_by_type[rule.type]++;
        }
    }

    return absl::OkStatus();
}

absl::Status AlertRulesEngine::RemoveRule(const std::string& rule_id) {
    std::lock_guard<std::mutex> lock(rules_mutex_);

    auto it = rules_.find(rule_id);
    if (it == rules_.end()) {
        return absl::NotFoundError("Rule not found: " + rule_id);
    }

    bool was_enabled = it->second.enabled;
    RuleType type = it->second.type;

    rules_.erase(it);
    rule_states_.erase(rule_id);

    // Update stats
    {
        std::lock_guard<std::mutex> stats_lock(stats_mutex_);
        stats_.total_rules--;
        if (was_enabled) {
            stats_.enabled_rules--;
        }
        stats_.rules_by_type[type]--;
    }

    return absl::OkStatus();
}

absl::StatusOr<AlertRule> AlertRulesEngine::GetRule(const std::string& rule_id) const {
    std::lock_guard<std::mutex> lock(rules_mutex_);

    auto it = rules_.find(rule_id);
    if (it == rules_.end()) {
        return absl::NotFoundError("Rule not found: " + rule_id);
    }

    return it->second;
}

std::vector<AlertRule> AlertRulesEngine::ListRules() const {
    std::lock_guard<std::mutex> lock(rules_mutex_);

    std::vector<AlertRule> result;
    result.reserve(rules_.size());

    for (const auto& [id, rule] : rules_) {
        result.push_back(rule);
    }

    return result;
}

std::vector<AlertRule> AlertRulesEngine::ListRulesByType(RuleType type) const {
    std::lock_guard<std::mutex> lock(rules_mutex_);

    std::vector<AlertRule> result;

    for (const auto& [id, rule] : rules_) {
        if (rule.type == type) {
            result.push_back(rule);
        }
    }

    return result;
}

absl::Status AlertRulesEngine::SetRuleEnabled(const std::string& rule_id, bool enabled) {
    std::lock_guard<std::mutex> lock(rules_mutex_);

    auto it = rules_.find(rule_id);
    if (it == rules_.end()) {
        return absl::NotFoundError("Rule not found: " + rule_id);
    }

    bool was_enabled = it->second.enabled;
    it->second.enabled = enabled;
    it->second.updated_at = std::chrono::system_clock::now();

    // Update stats
    {
        std::lock_guard<std::mutex> stats_lock(stats_mutex_);
        if (was_enabled && !enabled) {
            stats_.enabled_rules--;
        } else if (!was_enabled && enabled) {
            stats_.enabled_rules++;
        }
    }

    return absl::OkStatus();
}

std::vector<AlertEvent> AlertRulesEngine::Evaluate(
    const std::vector<MetricValue>& metrics) {

    if (!initialized_) {
        return {};
    }

    std::vector<AlertEvent> alerts;
    std::vector<AlertRule> rules_to_evaluate;

    // Get enabled rules
    {
        std::lock_guard<std::mutex> lock(rules_mutex_);
        for (const auto& [id, rule] : rules_) {
            if (rule.enabled) {
                rules_to_evaluate.push_back(rule);
            }
        }
    }

    // Evaluate each rule
    for (const auto& rule : rules_to_evaluate) {
        RuleEvaluation eval;

        switch (rule.type) {
            case RuleType::kThreshold:
                eval = EvaluateThresholdRule(rule, metrics);
                break;
            case RuleType::kAnomaly:
                eval = EvaluateAnomalyRule(rule, metrics);
                break;
            case RuleType::kRate:
                eval = EvaluateRateRule(rule, metrics);
                break;
            case RuleType::kPattern:
                eval = EvaluatePatternRule(rule, metrics);
                break;
            case RuleType::kComposite:
                eval = EvaluateCompositeRule(rule, metrics);
                break;
        }

        if (eval.fired) {
            // Find the triggering metric
            std::string metric_name;
            if (auto* threshold = std::get_if<ThresholdRuleConfig>(&rule.config)) {
                metric_name = threshold->metric_name;
            } else if (auto* anomaly = std::get_if<AnomalyRuleConfig>(&rule.config)) {
                metric_name = anomaly->metric_name;
            } else if (auto* rate = std::get_if<RateRuleConfig>(&rule.config)) {
                metric_name = rate->metric_name;
            } else if (auto* pattern = std::get_if<PatternRuleConfig>(&rule.config)) {
                metric_name = pattern->metric_name;
            }

            auto metric = FindMetric(metrics, metric_name);
            if (metric) {
                alerts.push_back(CreateAlertEvent(rule, eval, *metric));
            }
        }

        // Update stats
        {
            std::lock_guard<std::mutex> stats_lock(stats_mutex_);
            stats_.evaluations++;
        }
    }

    // Update alert count stats
    {
        std::lock_guard<std::mutex> stats_lock(stats_mutex_);
        stats_.alerts_generated += alerts.size();
        for (const auto& alert : alerts) {
            stats_.alerts_by_severity[alert.severity]++;
        }
    }

    return alerts;
}

absl::StatusOr<RuleEvaluation> AlertRulesEngine::EvaluateRule(
    const std::string& rule_id,
    const std::vector<MetricValue>& metrics) {

    AlertRule rule;
    {
        std::lock_guard<std::mutex> lock(rules_mutex_);
        auto it = rules_.find(rule_id);
        if (it == rules_.end()) {
            return absl::NotFoundError("Rule not found: " + rule_id);
        }
        rule = it->second;
    }

    RuleEvaluation eval;
    switch (rule.type) {
        case RuleType::kThreshold:
            eval = EvaluateThresholdRule(rule, metrics);
            break;
        case RuleType::kAnomaly:
            eval = EvaluateAnomalyRule(rule, metrics);
            break;
        case RuleType::kRate:
            eval = EvaluateRateRule(rule, metrics);
            break;
        case RuleType::kPattern:
            eval = EvaluatePatternRule(rule, metrics);
            break;
        case RuleType::kComposite:
            eval = EvaluateCompositeRule(rule, metrics);
            break;
    }

    return eval;
}

void AlertRulesEngine::RecordMetric(const MetricValue& metric) {
    std::lock_guard<std::mutex> lock(rules_mutex_);

    auto& history = metric_history_[metric.name];
    history.values.push_back(metric);

    // Keep last 24 hours of data
    auto cutoff = std::chrono::system_clock::now() - std::chrono::hours(24);
    history.values.erase(
        std::remove_if(history.values.begin(), history.values.end(),
                       [cutoff](const MetricValue& m) {
                           return m.timestamp < cutoff;
                       }),
        history.values.end());

    // Update statistics
    if (!history.values.empty()) {
        double sum = 0.0;
        for (const auto& v : history.values) {
            sum += v.value;
        }
        history.mean = sum / history.values.size();

        double sq_sum = 0.0;
        for (const auto& v : history.values) {
            sq_sum += (v.value - history.mean) * (v.value - history.mean);
        }
        history.std_dev = std::sqrt(sq_sum / history.values.size());
    }

    history.last_updated = std::chrono::system_clock::now();
}

absl::Status AlertRulesEngine::LoadFromFile(const std::string& path) {
    std::ifstream file(path);
    if (!file.is_open()) {
        return absl::NotFoundError("File not found: " + path);
    }

    std::stringstream buffer;
    buffer << file.rdbuf();

    return LoadFromJson(buffer.str());
}

absl::Status AlertRulesEngine::SaveToFile(const std::string& path) const {
    auto json_result = ExportToJson();
    if (!json_result.ok()) {
        return json_result.status();
    }

    std::ofstream file(path);
    if (!file.is_open()) {
        return absl::PermissionDeniedError("Cannot write to file: " + path);
    }

    file << *json_result;
    return absl::OkStatus();
}

absl::Status AlertRulesEngine::LoadFromJson(const std::string& json_str) {
    try {
        json j = json::parse(json_str);

        if (!j.contains("rules") || !j["rules"].is_array()) {
            return absl::InvalidArgumentError("Invalid rules JSON format");
        }

        for (const auto& rule_json : j["rules"]) {
            AlertRule rule;
            rule.rule_id = rule_json.value("rule_id", "");
            rule.name = rule_json.value("name", "");
            rule.description = rule_json.value("description", "");
            rule.type = StringToRuleType(rule_json.value("type", "threshold"));
            rule.severity = StringToAlertSeverity(rule_json.value("severity", "warning"));
            rule.enabled = rule_json.value("enabled", true);

            // Parse type-specific config
            if (rule_json.contains("config")) {
                const auto& cfg = rule_json["config"];
                switch (rule.type) {
                    case RuleType::kThreshold: {
                        ThresholdRuleConfig config;
                        config.metric_name = cfg.value("metric_name", "");
                        config.comparison = StringToComparisonOp(
                            cfg.value("comparison", "greater_than"));
                        config.threshold = cfg.value("threshold", 0.0);
                        config.consecutive_count = cfg.value("consecutive_count", 1);
                        rule.config = config;
                        break;
                    }
                    case RuleType::kAnomaly: {
                        AnomalyRuleConfig config;
                        config.metric_name = cfg.value("metric_name", "");
                        config.std_dev_threshold = cfg.value("std_dev_threshold", 3.0);
                        config.min_samples = cfg.value("min_samples", 100);
                        rule.config = config;
                        break;
                    }
                    case RuleType::kRate: {
                        RateRuleConfig config;
                        config.metric_name = cfg.value("metric_name", "");
                        config.rate_threshold = cfg.value("rate_threshold", 0.0);
                        config.window = std::chrono::seconds(cfg.value("window_seconds", 300));
                        config.use_percentage = cfg.value("use_percentage", false);
                        rule.config = config;
                        break;
                    }
                    case RuleType::kPattern: {
                        PatternRuleConfig config;
                        config.metric_name = cfg.value("metric_name", "");
                        config.sensitivity = cfg.value("sensitivity", 0.7);
                        rule.config = config;
                        break;
                    }
                    case RuleType::kComposite: {
                        CompositeRuleConfig config;
                        config.rule_ids = cfg.value("rule_ids", std::vector<std::string>{});
                        rule.config = config;
                        break;
                    }
                }
            }

            auto status = AddRule(rule);
            if (!status.ok()) {
                return status;
            }
        }

        return absl::OkStatus();
    } catch (const json::exception& e) {
        return absl::InvalidArgumentError(
            std::string("JSON parse error: ") + e.what());
    }
}

absl::StatusOr<std::string> AlertRulesEngine::ExportToJson() const {
    std::lock_guard<std::mutex> lock(rules_mutex_);

    json j;
    json rules_array = json::array();

    for (const auto& [id, rule] : rules_) {
        json rule_json;
        rule_json["rule_id"] = rule.rule_id;
        rule_json["name"] = rule.name;
        rule_json["description"] = rule.description;
        rule_json["type"] = RuleTypeToString(rule.type);
        rule_json["severity"] = AlertSeverityToString(rule.severity);
        rule_json["enabled"] = rule.enabled;

        // Serialize config based on type
        json cfg;
        if (auto* threshold = std::get_if<ThresholdRuleConfig>(&rule.config)) {
            cfg["metric_name"] = threshold->metric_name;
            cfg["comparison"] = ComparisonOpToString(threshold->comparison);
            cfg["threshold"] = threshold->threshold;
            cfg["consecutive_count"] = threshold->consecutive_count;
        } else if (auto* anomaly = std::get_if<AnomalyRuleConfig>(&rule.config)) {
            cfg["metric_name"] = anomaly->metric_name;
            cfg["std_dev_threshold"] = anomaly->std_dev_threshold;
            cfg["min_samples"] = anomaly->min_samples;
        } else if (auto* rate = std::get_if<RateRuleConfig>(&rule.config)) {
            cfg["metric_name"] = rate->metric_name;
            cfg["rate_threshold"] = rate->rate_threshold;
            cfg["window_seconds"] = rate->window.count();
            cfg["use_percentage"] = rate->use_percentage;
        } else if (auto* pattern = std::get_if<PatternRuleConfig>(&rule.config)) {
            cfg["metric_name"] = pattern->metric_name;
            cfg["sensitivity"] = pattern->sensitivity;
        } else if (auto* composite = std::get_if<CompositeRuleConfig>(&rule.config)) {
            cfg["rule_ids"] = composite->rule_ids;
        }
        rule_json["config"] = cfg;

        rules_array.push_back(rule_json);
    }

    j["rules"] = rules_array;
    return j.dump(2);
}

AlertRulesEngine::Stats AlertRulesEngine::GetStats() const {
    std::lock_guard<std::mutex> lock(stats_mutex_);
    return stats_;
}

void AlertRulesEngine::ResetStats() {
    std::lock_guard<std::mutex> lock(stats_mutex_);

    // Keep rule counts, reset evaluation counts
    stats_.evaluations = 0;
    stats_.alerts_generated = 0;
    stats_.alerts_by_severity.clear();
}

// Private implementation methods

RuleEvaluation AlertRulesEngine::EvaluateThresholdRule(
    const AlertRule& rule,
    const std::vector<MetricValue>& metrics) {

    RuleEvaluation eval;
    eval.rule_id = rule.rule_id;
    eval.evaluated_at = std::chrono::system_clock::now();

    auto* config = std::get_if<ThresholdRuleConfig>(&rule.config);
    if (!config) {
        eval.reason = "Invalid threshold rule configuration";
        return eval;
    }

    auto metric = FindMetric(metrics, config->metric_name);
    if (!metric) {
        eval.reason = "Metric not found: " + config->metric_name;
        return eval;
    }

    eval.current_value = metric->value;
    eval.threshold_value = config->threshold;

    bool threshold_violated = CompareValues(
        metric->value, config->comparison, config->threshold);

    if (threshold_violated) {
        auto& state = rule_states_[rule.rule_id];
        state.consecutive_violations++;

        if (state.consecutive_violations >= config->consecutive_count) {
            eval.fired = true;
            eval.reason = "Threshold exceeded: " +
                          std::to_string(metric->value) + " " +
                          ComparisonOpToString(config->comparison) + " " +
                          std::to_string(config->threshold);
            state.is_firing = true;
        } else {
            eval.pending = true;
            eval.reason = "Threshold violated " +
                          std::to_string(state.consecutive_violations) +
                          "/" + std::to_string(config->consecutive_count) +
                          " times";
        }
    } else {
        rule_states_[rule.rule_id].consecutive_violations = 0;
        rule_states_[rule.rule_id].is_firing = false;
        eval.reason = "Threshold not violated";
    }

    return eval;
}

RuleEvaluation AlertRulesEngine::EvaluateAnomalyRule(
    const AlertRule& rule,
    const std::vector<MetricValue>& metrics) {

    RuleEvaluation eval;
    eval.rule_id = rule.rule_id;
    eval.evaluated_at = std::chrono::system_clock::now();

    auto* config = std::get_if<AnomalyRuleConfig>(&rule.config);
    if (!config) {
        eval.reason = "Invalid anomaly rule configuration";
        return eval;
    }

    auto metric = FindMetric(metrics, config->metric_name);
    if (!metric) {
        eval.reason = "Metric not found: " + config->metric_name;
        return eval;
    }

    auto history_it = metric_history_.find(config->metric_name);
    if (history_it == metric_history_.end() ||
        history_it->second.values.size() < config->min_samples) {
        eval.reason = "Insufficient history for anomaly detection";
        return eval;
    }

    const auto& history = history_it->second;
    eval.current_value = metric->value;

    if (history.std_dev > 0) {
        double z_score = (metric->value - history.mean) / history.std_dev;
        eval.threshold_value = history.mean + (config->std_dev_threshold * history.std_dev);

        bool is_anomaly = false;
        switch (config->direction) {
            case AnomalyRuleConfig::Direction::kBoth:
                is_anomaly = std::abs(z_score) > config->std_dev_threshold;
                break;
            case AnomalyRuleConfig::Direction::kUp:
                is_anomaly = z_score > config->std_dev_threshold;
                break;
            case AnomalyRuleConfig::Direction::kDown:
                is_anomaly = z_score < -config->std_dev_threshold;
                break;
        }

        if (is_anomaly) {
            eval.fired = true;
            std::ostringstream reason;
            reason << "Anomaly detected: value=" << metric->value
                   << " z-score=" << z_score
                   << " (threshold=" << config->std_dev_threshold << ")";
            eval.reason = reason.str();
        } else {
            eval.reason = "Value within normal range";
        }
    } else {
        eval.reason = "Cannot compute anomaly (zero standard deviation)";
    }

    return eval;
}

RuleEvaluation AlertRulesEngine::EvaluateRateRule(
    const AlertRule& rule,
    const std::vector<MetricValue>& metrics) {

    RuleEvaluation eval;
    eval.rule_id = rule.rule_id;
    eval.evaluated_at = std::chrono::system_clock::now();

    auto* config = std::get_if<RateRuleConfig>(&rule.config);
    if (!config) {
        eval.reason = "Invalid rate rule configuration";
        return eval;
    }

    auto history_it = metric_history_.find(config->metric_name);
    if (history_it == metric_history_.end() ||
        history_it->second.values.size() < 2) {
        eval.reason = "Insufficient history for rate calculation";
        return eval;
    }

    const auto& values = history_it->second.values;
    auto now = std::chrono::system_clock::now();
    auto window_start = now - config->window;

    // Find values within window
    std::vector<const MetricValue*> window_values;
    for (const auto& v : values) {
        if (v.timestamp >= window_start) {
            window_values.push_back(&v);
        }
    }

    if (window_values.size() < 2) {
        eval.reason = "Insufficient values in window";
        return eval;
    }

    // Calculate rate
    double first_value = window_values.front()->value;
    double last_value = window_values.back()->value;
    auto time_diff = std::chrono::duration_cast<std::chrono::seconds>(
        window_values.back()->timestamp - window_values.front()->timestamp);

    double rate = 0.0;
    if (time_diff.count() > 0) {
        if (config->use_percentage && first_value != 0) {
            rate = ((last_value - first_value) / first_value) * 100.0 /
                   time_diff.count();
        } else {
            rate = (last_value - first_value) / time_diff.count();
        }
    }

    eval.current_value = rate;
    eval.threshold_value = config->rate_threshold;

    if (CompareValues(rate, config->comparison, config->rate_threshold)) {
        eval.fired = true;
        std::ostringstream reason;
        reason << "Rate threshold exceeded: " << rate
               << (config->use_percentage ? "%/s" : "/s")
               << " " << ComparisonOpToString(config->comparison)
               << " " << config->rate_threshold;
        eval.reason = reason.str();
    } else {
        eval.reason = "Rate within threshold";
    }

    return eval;
}

RuleEvaluation AlertRulesEngine::EvaluatePatternRule(
    const AlertRule& rule,
    const std::vector<MetricValue>& metrics) {

    RuleEvaluation eval;
    eval.rule_id = rule.rule_id;
    eval.evaluated_at = std::chrono::system_clock::now();

    auto* config = std::get_if<PatternRuleConfig>(&rule.config);
    if (!config) {
        eval.reason = "Invalid pattern rule configuration";
        return eval;
    }

    auto history_it = metric_history_.find(config->metric_name);
    if (history_it == metric_history_.end() ||
        history_it->second.values.size() < 10) {
        eval.reason = "Insufficient history for pattern detection";
        return eval;
    }

    const auto& values = history_it->second.values;

    // Simple pattern detection based on recent values
    size_t n = std::min(values.size(), size_t(20));
    std::vector<double> recent;
    for (size_t i = values.size() - n; i < values.size(); i++) {
        recent.push_back(values[i].value);
    }

    bool pattern_detected = false;
    std::string pattern_name;

    switch (config->pattern) {
        case PatternRuleConfig::PatternType::kSpike: {
            // Look for value that goes up then down significantly
            if (recent.size() >= 3) {
                double max_val = *std::max_element(recent.begin(), recent.end());
                double avg = std::accumulate(recent.begin(), recent.end(), 0.0) / recent.size();
                if (max_val > avg * (1 + config->sensitivity)) {
                    pattern_detected = true;
                    pattern_name = "spike";
                }
            }
            break;
        }
        case PatternRuleConfig::PatternType::kDip: {
            if (recent.size() >= 3) {
                double min_val = *std::min_element(recent.begin(), recent.end());
                double avg = std::accumulate(recent.begin(), recent.end(), 0.0) / recent.size();
                if (min_val < avg * (1 - config->sensitivity)) {
                    pattern_detected = true;
                    pattern_name = "dip";
                }
            }
            break;
        }
        case PatternRuleConfig::PatternType::kStep: {
            // Detect level shift
            if (recent.size() >= 6) {
                size_t mid = recent.size() / 2;
                double first_half_avg = std::accumulate(
                    recent.begin(), recent.begin() + mid, 0.0) / mid;
                double second_half_avg = std::accumulate(
                    recent.begin() + mid, recent.end(), 0.0) / (recent.size() - mid);

                double diff = std::abs(second_half_avg - first_half_avg) / first_half_avg;
                if (diff > config->sensitivity) {
                    pattern_detected = true;
                    pattern_name = "step";
                }
            }
            break;
        }
        case PatternRuleConfig::PatternType::kTrend: {
            // Simple linear trend detection
            if (recent.size() >= 5) {
                int increasing = 0;
                int decreasing = 0;
                for (size_t i = 1; i < recent.size(); i++) {
                    if (recent[i] > recent[i-1]) increasing++;
                    else if (recent[i] < recent[i-1]) decreasing++;
                }
                double trend_strength = std::max(increasing, decreasing) /
                                        static_cast<double>(recent.size() - 1);
                if (trend_strength > config->sensitivity) {
                    pattern_detected = true;
                    pattern_name = increasing > decreasing ? "upward_trend" : "downward_trend";
                }
            }
            break;
        }
        case PatternRuleConfig::PatternType::kOscillation: {
            // Detect alternating values
            if (recent.size() >= 4) {
                int changes = 0;
                for (size_t i = 2; i < recent.size(); i++) {
                    if ((recent[i] - recent[i-1]) * (recent[i-1] - recent[i-2]) < 0) {
                        changes++;
                    }
                }
                double oscillation = changes / static_cast<double>(recent.size() - 2);
                if (oscillation > config->sensitivity) {
                    pattern_detected = true;
                    pattern_name = "oscillation";
                }
            }
            break;
        }
    }

    if (pattern_detected) {
        eval.fired = true;
        eval.reason = "Pattern detected: " + pattern_name;
    } else {
        eval.reason = "No pattern detected";
    }

    return eval;
}

RuleEvaluation AlertRulesEngine::EvaluateCompositeRule(
    const AlertRule& rule,
    const std::vector<MetricValue>& metrics) {

    RuleEvaluation eval;
    eval.rule_id = rule.rule_id;
    eval.evaluated_at = std::chrono::system_clock::now();

    auto* config = std::get_if<CompositeRuleConfig>(&rule.config);
    if (!config || config->rule_ids.empty()) {
        eval.reason = "Invalid composite rule configuration";
        return eval;
    }

    std::vector<bool> child_results;

    for (const auto& child_id : config->rule_ids) {
        auto child_eval = EvaluateRule(child_id, metrics);
        if (child_eval.ok()) {
            child_results.push_back(child_eval->fired);
        } else {
            child_results.push_back(false);
        }
    }

    bool result = false;
    if (config->op == CompositeRuleConfig::LogicalOp::kAnd) {
        result = std::all_of(child_results.begin(), child_results.end(),
                             [](bool v) { return v; });
    } else {
        result = std::any_of(child_results.begin(), child_results.end(),
                             [](bool v) { return v; });
    }

    eval.fired = result;
    std::ostringstream reason;
    reason << "Composite rule (" << (config->op == CompositeRuleConfig::LogicalOp::kAnd ? "AND" : "OR")
           << "): ";
    for (size_t i = 0; i < child_results.size(); i++) {
        if (i > 0) reason << ", ";
        reason << config->rule_ids[i] << "=" << (child_results[i] ? "true" : "false");
    }
    eval.reason = reason.str();

    return eval;
}

std::optional<MetricValue> AlertRulesEngine::FindMetric(
    const std::vector<MetricValue>& metrics,
    const std::string& name) const {

    for (const auto& metric : metrics) {
        if (metric.name == name) {
            return metric;
        }
    }
    return std::nullopt;
}

bool AlertRulesEngine::CompareValues(
    double value, ComparisonOp op, double threshold) const {

    switch (op) {
        case ComparisonOp::kGreaterThan:
            return value > threshold;
        case ComparisonOp::kGreaterThanOrEqual:
            return value >= threshold;
        case ComparisonOp::kLessThan:
            return value < threshold;
        case ComparisonOp::kLessThanOrEqual:
            return value <= threshold;
        case ComparisonOp::kEqual:
            return std::abs(value - threshold) < 1e-9;
        case ComparisonOp::kNotEqual:
            return std::abs(value - threshold) >= 1e-9;
    }
    return false;
}

std::string AlertRulesEngine::GenerateAlertId() const {
    static std::random_device rd;
    static std::mt19937 gen(rd());
    static std::uniform_int_distribution<uint64_t> dis;

    std::ostringstream id;
    id << "alert-" << std::hex << dis(gen);
    return id.str();
}

std::string AlertRulesEngine::GenerateFingerprint(
    const AlertRule& rule,
    const MetricValue& metric) const {

    std::ostringstream fp;
    fp << rule.rule_id << ":";
    fp << metric.name << ":";
    for (const auto& [key, val] : metric.labels) {
        fp << key << "=" << val << ",";
    }
    return fp.str();
}

AlertEvent AlertRulesEngine::CreateAlertEvent(
    const AlertRule& rule,
    const RuleEvaluation& eval,
    const MetricValue& metric) {

    AlertEvent event;
    event.alert_id = GenerateAlertId();
    event.rule_id = rule.rule_id;
    event.rule_name = rule.name;
    event.severity = rule.severity;
    event.triggered_at = std::chrono::system_clock::now();
    event.is_firing = true;

    event.title = rule.name;
    event.description = eval.reason;

    event.metric_name = metric.name;
    event.metric_value = eval.current_value;
    event.threshold_value = eval.threshold_value;

    event.labels = metric.labels;
    event.annotations = rule.annotations;

    event.fingerprint = GenerateFingerprint(rule, metric);

    return event;
}

// Factory function
std::unique_ptr<AlertRulesEngine> CreateAlertRulesEngine() {
    return std::make_unique<AlertRulesEngine>();
}

// Utility functions
std::string AlertSeverityToString(AlertSeverity severity) {
    switch (severity) {
        case AlertSeverity::kInfo: return "info";
        case AlertSeverity::kWarning: return "warning";
        case AlertSeverity::kError: return "error";
        case AlertSeverity::kCritical: return "critical";
    }
    return "unknown";
}

AlertSeverity StringToAlertSeverity(const std::string& str) {
    if (str == "info") return AlertSeverity::kInfo;
    if (str == "error") return AlertSeverity::kError;
    if (str == "critical") return AlertSeverity::kCritical;
    return AlertSeverity::kWarning;
}

std::string RuleTypeToString(RuleType type) {
    switch (type) {
        case RuleType::kThreshold: return "threshold";
        case RuleType::kAnomaly: return "anomaly";
        case RuleType::kRate: return "rate";
        case RuleType::kPattern: return "pattern";
        case RuleType::kComposite: return "composite";
    }
    return "unknown";
}

RuleType StringToRuleType(const std::string& str) {
    if (str == "anomaly") return RuleType::kAnomaly;
    if (str == "rate") return RuleType::kRate;
    if (str == "pattern") return RuleType::kPattern;
    if (str == "composite") return RuleType::kComposite;
    return RuleType::kThreshold;
}

std::string ComparisonOpToString(ComparisonOp op) {
    switch (op) {
        case ComparisonOp::kGreaterThan: return "greater_than";
        case ComparisonOp::kGreaterThanOrEqual: return "greater_than_or_equal";
        case ComparisonOp::kLessThan: return "less_than";
        case ComparisonOp::kLessThanOrEqual: return "less_than_or_equal";
        case ComparisonOp::kEqual: return "equal";
        case ComparisonOp::kNotEqual: return "not_equal";
    }
    return "unknown";
}

ComparisonOp StringToComparisonOp(const std::string& str) {
    if (str == "greater_than_or_equal") return ComparisonOp::kGreaterThanOrEqual;
    if (str == "less_than") return ComparisonOp::kLessThan;
    if (str == "less_than_or_equal") return ComparisonOp::kLessThanOrEqual;
    if (str == "equal") return ComparisonOp::kEqual;
    if (str == "not_equal") return ComparisonOp::kNotEqual;
    return ComparisonOp::kGreaterThan;
}

absl::Status ValidateRule(const AlertRule& rule) {
    if (rule.rule_id.empty()) {
        return absl::InvalidArgumentError("Rule ID is required");
    }
    if (rule.name.empty()) {
        return absl::InvalidArgumentError("Rule name is required");
    }

    // Validate type-specific config
    switch (rule.type) {
        case RuleType::kThreshold: {
            auto* config = std::get_if<ThresholdRuleConfig>(&rule.config);
            if (!config || config->metric_name.empty()) {
                return absl::InvalidArgumentError(
                    "Threshold rule requires metric_name");
            }
            break;
        }
        case RuleType::kAnomaly: {
            auto* config = std::get_if<AnomalyRuleConfig>(&rule.config);
            if (!config || config->metric_name.empty()) {
                return absl::InvalidArgumentError(
                    "Anomaly rule requires metric_name");
            }
            break;
        }
        case RuleType::kRate: {
            auto* config = std::get_if<RateRuleConfig>(&rule.config);
            if (!config || config->metric_name.empty()) {
                return absl::InvalidArgumentError(
                    "Rate rule requires metric_name");
            }
            break;
        }
        case RuleType::kPattern: {
            auto* config = std::get_if<PatternRuleConfig>(&rule.config);
            if (!config || config->metric_name.empty()) {
                return absl::InvalidArgumentError(
                    "Pattern rule requires metric_name");
            }
            break;
        }
        case RuleType::kComposite: {
            auto* config = std::get_if<CompositeRuleConfig>(&rule.config);
            if (!config || config->rule_ids.empty()) {
                return absl::InvalidArgumentError(
                    "Composite rule requires child rule_ids");
            }
            break;
        }
    }

    return absl::OkStatus();
}

AlertRule CreateThresholdRule(
    const std::string& name,
    const std::string& metric_name,
    ComparisonOp comparison,
    double threshold,
    AlertSeverity severity) {

    AlertRule rule;
    rule.rule_id = "rule-" + name;
    rule.name = name;
    rule.type = RuleType::kThreshold;
    rule.severity = severity;

    ThresholdRuleConfig config;
    config.metric_name = metric_name;
    config.comparison = comparison;
    config.threshold = threshold;
    rule.config = config;

    return rule;
}

AlertRule CreateAnomalyRule(
    const std::string& name,
    const std::string& metric_name,
    double std_dev_threshold,
    AlertSeverity severity) {

    AlertRule rule;
    rule.rule_id = "rule-" + name;
    rule.name = name;
    rule.type = RuleType::kAnomaly;
    rule.severity = severity;

    AnomalyRuleConfig config;
    config.metric_name = metric_name;
    config.std_dev_threshold = std_dev_threshold;
    rule.config = config;

    return rule;
}

AlertRule CreateRateRule(
    const std::string& name,
    const std::string& metric_name,
    double rate_threshold,
    std::chrono::seconds window,
    AlertSeverity severity) {

    AlertRule rule;
    rule.rule_id = "rule-" + name;
    rule.name = name;
    rule.type = RuleType::kRate;
    rule.severity = severity;

    RateRuleConfig config;
    config.metric_name = metric_name;
    config.rate_threshold = rate_threshold;
    config.window = window;
    rule.config = config;

    return rule;
}

}  // namespace pyflare::alerting
