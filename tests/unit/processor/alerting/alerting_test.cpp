/// @file alerting_test.cpp
/// @brief Unit tests for alerting system

#include <gtest/gtest.h>
#include <gmock/gmock.h>

#include "processor/alerting/alert_rules.h"
#include "processor/alerting/deduplicator.h"

namespace pyflare::alerting {
namespace {

using ::testing::_;
using ::testing::Return;

// ============================================================================
// Alert Rules Engine Tests
// ============================================================================

class AlertRulesEngineTest : public ::testing::Test {
protected:
    void SetUp() override {
        engine_ = CreateAlertRulesEngine();
        ASSERT_TRUE(engine_->Initialize().ok());
    }

    std::unique_ptr<AlertRulesEngine> engine_;
};

TEST_F(AlertRulesEngineTest, AddRuleSucceeds) {
    auto rule = CreateThresholdRule(
        "high_latency",
        "p99_latency",
        ComparisonOp::kGreaterThan,
        1000.0,
        AlertSeverity::kWarning);

    auto status = engine_->AddRule(rule);
    EXPECT_TRUE(status.ok());
}

TEST_F(AlertRulesEngineTest, AddDuplicateRuleFails) {
    auto rule = CreateThresholdRule(
        "high_latency",
        "p99_latency",
        ComparisonOp::kGreaterThan,
        1000.0);

    EXPECT_TRUE(engine_->AddRule(rule).ok());
    EXPECT_FALSE(engine_->AddRule(rule).ok());
}

TEST_F(AlertRulesEngineTest, GetRuleReturnsAdded) {
    auto rule = CreateThresholdRule(
        "test_rule",
        "metric",
        ComparisonOp::kGreaterThan,
        100.0);

    engine_->AddRule(rule);
    auto retrieved = engine_->GetRule(rule.rule_id);

    ASSERT_TRUE(retrieved.ok());
    EXPECT_EQ(retrieved->name, rule.name);
}

TEST_F(AlertRulesEngineTest, GetNonExistentRuleFails) {
    auto result = engine_->GetRule("non_existent");
    EXPECT_FALSE(result.ok());
}

TEST_F(AlertRulesEngineTest, RemoveRuleSucceeds) {
    auto rule = CreateThresholdRule(
        "test_rule",
        "metric",
        ComparisonOp::kGreaterThan,
        100.0);

    engine_->AddRule(rule);
    auto status = engine_->RemoveRule(rule.rule_id);
    EXPECT_TRUE(status.ok());

    EXPECT_FALSE(engine_->GetRule(rule.rule_id).ok());
}

TEST_F(AlertRulesEngineTest, ListRulesReturnsAll) {
    engine_->AddRule(CreateThresholdRule("rule1", "m1", ComparisonOp::kGreaterThan, 1.0));
    engine_->AddRule(CreateThresholdRule("rule2", "m2", ComparisonOp::kLessThan, 2.0));
    engine_->AddRule(CreateAnomalyRule("rule3", "m3"));

    auto rules = engine_->ListRules();
    EXPECT_EQ(rules.size(), 3);
}

TEST_F(AlertRulesEngineTest, EvaluateThresholdRuleFires) {
    auto rule = CreateThresholdRule(
        "high_error_rate",
        "error_rate",
        ComparisonOp::kGreaterThan,
        0.1,
        AlertSeverity::kError);

    engine_->AddRule(rule);

    MetricValue metric;
    metric.name = "error_rate";
    metric.value = 0.15;  // Above threshold
    metric.timestamp = std::chrono::system_clock::now();

    auto alerts = engine_->Evaluate({metric});
    EXPECT_EQ(alerts.size(), 1);
    EXPECT_EQ(alerts[0].severity, AlertSeverity::kError);
}

TEST_F(AlertRulesEngineTest, EvaluateThresholdRuleDoesNotFire) {
    auto rule = CreateThresholdRule(
        "high_error_rate",
        "error_rate",
        ComparisonOp::kGreaterThan,
        0.1);

    engine_->AddRule(rule);

    MetricValue metric;
    metric.name = "error_rate";
    metric.value = 0.05;  // Below threshold
    metric.timestamp = std::chrono::system_clock::now();

    auto alerts = engine_->Evaluate({metric});
    EXPECT_EQ(alerts.size(), 0);
}

TEST_F(AlertRulesEngineTest, DisabledRuleDoesNotFire) {
    auto rule = CreateThresholdRule(
        "test_rule",
        "metric",
        ComparisonOp::kGreaterThan,
        0.0);  // Always fires if enabled

    engine_->AddRule(rule);
    engine_->SetRuleEnabled(rule.rule_id, false);

    MetricValue metric;
    metric.name = "metric";
    metric.value = 100.0;
    metric.timestamp = std::chrono::system_clock::now();

    auto alerts = engine_->Evaluate({metric});
    EXPECT_EQ(alerts.size(), 0);
}

TEST_F(AlertRulesEngineTest, StatsAreUpdated) {
    auto rule = CreateThresholdRule(
        "test_rule",
        "metric",
        ComparisonOp::kGreaterThan,
        50.0);

    engine_->AddRule(rule);

    MetricValue metric;
    metric.name = "metric";
    metric.value = 100.0;
    metric.timestamp = std::chrono::system_clock::now();

    engine_->Evaluate({metric});
    engine_->Evaluate({metric});

    auto stats = engine_->GetStats();
    EXPECT_EQ(stats.total_rules, 1);
    EXPECT_EQ(stats.evaluations, 2);
    EXPECT_EQ(stats.alerts_generated, 2);
}

// Comparison operator tests
TEST_F(AlertRulesEngineTest, AllComparisonOperatorsWork) {
    std::vector<std::pair<ComparisonOp, double>> test_cases = {
        {ComparisonOp::kGreaterThan, 51.0},
        {ComparisonOp::kGreaterThanOrEqual, 50.0},
        {ComparisonOp::kLessThan, 49.0},
        {ComparisonOp::kLessThanOrEqual, 50.0},
        {ComparisonOp::kEqual, 50.0},
        {ComparisonOp::kNotEqual, 51.0},
    };

    for (size_t i = 0; i < test_cases.size(); i++) {
        auto [op, value] = test_cases[i];
        std::string name = "rule_" + std::to_string(i);

        auto rule = CreateThresholdRule(name, "metric", op, 50.0);
        engine_->AddRule(rule);

        MetricValue metric;
        metric.name = "metric";
        metric.value = value;
        metric.timestamp = std::chrono::system_clock::now();

        auto result = engine_->EvaluateRule(rule.rule_id, {metric});
        ASSERT_TRUE(result.ok()) << "Failed for operator " << ComparisonOpToString(op);
        EXPECT_TRUE(result->fired) << "Rule should fire for operator " << ComparisonOpToString(op);
    }
}

// ============================================================================
// Alert Deduplicator Tests
// ============================================================================

class AlertDeduplicatorTest : public ::testing::Test {
protected:
    void SetUp() override {
        config_.dedup_window = std::chrono::minutes(5);
        config_.rate_limit.enabled = false;  // Disable for tests

        dedup_ = CreateAlertDeduplicator(config_);
        ASSERT_TRUE(dedup_->Initialize().ok());
    }

    AlertEvent CreateAlert(const std::string& fingerprint,
                           AlertSeverity severity = AlertSeverity::kWarning) {
        AlertEvent alert;
        alert.alert_id = "alert-" + fingerprint;
        alert.rule_id = "rule-" + fingerprint;
        alert.rule_name = "Test Rule";
        alert.fingerprint = fingerprint;
        alert.severity = severity;
        alert.triggered_at = std::chrono::system_clock::now();
        alert.is_firing = true;
        alert.title = "Test Alert";
        alert.description = "Test description";
        return alert;
    }

    DeduplicatorConfig config_;
    std::unique_ptr<AlertDeduplicator> dedup_;
};

TEST_F(AlertDeduplicatorTest, FirstAlertIsNew) {
    auto alert = CreateAlert("test-fp-1");
    auto result = dedup_->Process(alert);

    EXPECT_TRUE(result.accepted);
    EXPECT_TRUE(result.is_new);
    EXPECT_FALSE(result.is_duplicate);
    EXPECT_FALSE(result.is_suppressed);
}

TEST_F(AlertDeduplicatorTest, DuplicateAlertIsDetected) {
    auto alert = CreateAlert("test-fp-1");

    dedup_->Process(alert);
    auto result = dedup_->Process(alert);

    EXPECT_TRUE(result.accepted);
    EXPECT_FALSE(result.is_new);
    EXPECT_TRUE(result.is_duplicate);
}

TEST_F(AlertDeduplicatorTest, DifferentFingerprintsAreNotDuplicates) {
    auto alert1 = CreateAlert("fp-1");
    auto alert2 = CreateAlert("fp-2");

    auto result1 = dedup_->Process(alert1);
    auto result2 = dedup_->Process(alert2);

    EXPECT_TRUE(result1.is_new);
    EXPECT_TRUE(result2.is_new);
}

TEST_F(AlertDeduplicatorTest, GetActiveAlertsReturnsAll) {
    dedup_->Process(CreateAlert("fp-1"));
    dedup_->Process(CreateAlert("fp-2"));
    dedup_->Process(CreateAlert("fp-3"));

    auto active = dedup_->GetActiveAlerts();
    EXPECT_EQ(active.size(), 3);
}

TEST_F(AlertDeduplicatorTest, ResolveAlertWorks) {
    auto alert = CreateAlert("fp-1");
    dedup_->Process(alert);

    auto status = dedup_->Resolve("fp-1");
    EXPECT_TRUE(status.ok());

    auto state = dedup_->GetState("fp-1");
    ASSERT_TRUE(state.ok());
    EXPECT_EQ(*state, AlertState::kResolved);
}

TEST_F(AlertDeduplicatorTest, SilenceBlocksAlerts) {
    Silence silence;
    silence.matchers["rule_id"] = "rule-fp-1";
    silence.starts_at = std::chrono::system_clock::now() - std::chrono::minutes(1);
    silence.ends_at = std::chrono::system_clock::now() + std::chrono::hours(1);

    dedup_->AddSilence(silence);

    auto alert = CreateAlert("fp-1");
    auto result = dedup_->Process(alert);

    EXPECT_TRUE(result.is_suppressed);
    EXPECT_EQ(result.state, AlertState::kSilenced);
}

TEST_F(AlertDeduplicatorTest, MaintenanceWindowSuppresses) {
    MaintenanceWindow window;
    window.rule_ids = {"rule-fp-1"};
    window.starts_at = std::chrono::system_clock::now() - std::chrono::minutes(1);
    window.ends_at = std::chrono::system_clock::now() + std::chrono::hours(1);
    window.suppress_alerts = true;

    dedup_->AddMaintenanceWindow(window);

    auto alert = CreateAlert("fp-1");
    auto result = dedup_->Process(alert);

    EXPECT_TRUE(result.is_suppressed);
}

TEST_F(AlertDeduplicatorTest, AlertGroupsAreCreated) {
    // Process alerts that should group together
    for (int i = 0; i < 5; i++) {
        auto alert = CreateAlert("fp-" + std::to_string(i));
        alert.rule_id = "common-rule";  // Same rule
        alert.model_id = "common-model";  // Same model
        dedup_->Process(alert);
    }

    auto groups = dedup_->GetGroups();
    EXPECT_GE(groups.size(), 1);
}

TEST_F(AlertDeduplicatorTest, CleanupRemovesExpiredSilences) {
    Silence silence;
    silence.matchers["rule_id"] = "test";
    silence.starts_at = std::chrono::system_clock::now() - std::chrono::hours(2);
    silence.ends_at = std::chrono::system_clock::now() - std::chrono::hours(1);  // Already expired

    dedup_->AddSilence(silence);
    dedup_->Cleanup();

    auto silences = dedup_->GetActiveSilences();
    EXPECT_EQ(silences.size(), 0);
}

TEST_F(AlertDeduplicatorTest, StatsAreTracked) {
    dedup_->Process(CreateAlert("fp-1"));
    dedup_->Process(CreateAlert("fp-1"));  // Duplicate
    dedup_->Process(CreateAlert("fp-2"));

    auto stats = dedup_->GetStats();
    EXPECT_EQ(stats.total_processed, 3);
    EXPECT_EQ(stats.total_deduplicated, 1);
    EXPECT_EQ(stats.active_alerts, 2);
}

// ============================================================================
// Utility Function Tests
// ============================================================================

TEST(AlertingSeverityTest, SeverityConversion) {
    EXPECT_EQ(AlertSeverityToString(AlertSeverity::kInfo), "info");
    EXPECT_EQ(AlertSeverityToString(AlertSeverity::kWarning), "warning");
    EXPECT_EQ(AlertSeverityToString(AlertSeverity::kError), "error");
    EXPECT_EQ(AlertSeverityToString(AlertSeverity::kCritical), "critical");

    EXPECT_EQ(StringToAlertSeverity("info"), AlertSeverity::kInfo);
    EXPECT_EQ(StringToAlertSeverity("warning"), AlertSeverity::kWarning);
    EXPECT_EQ(StringToAlertSeverity("error"), AlertSeverity::kError);
    EXPECT_EQ(StringToAlertSeverity("critical"), AlertSeverity::kCritical);
}

TEST(AlertingRuleTypeTest, RuleTypeConversion) {
    EXPECT_EQ(RuleTypeToString(RuleType::kThreshold), "threshold");
    EXPECT_EQ(RuleTypeToString(RuleType::kAnomaly), "anomaly");
    EXPECT_EQ(RuleTypeToString(RuleType::kRate), "rate");
    EXPECT_EQ(RuleTypeToString(RuleType::kPattern), "pattern");
    EXPECT_EQ(RuleTypeToString(RuleType::kComposite), "composite");

    EXPECT_EQ(StringToRuleType("threshold"), RuleType::kThreshold);
    EXPECT_EQ(StringToRuleType("anomaly"), RuleType::kAnomaly);
    EXPECT_EQ(StringToRuleType("rate"), RuleType::kRate);
}

TEST(AlertingSerializationTest, AlertEventSerialization) {
    AlertEvent alert;
    alert.alert_id = "test-alert";
    alert.rule_id = "test-rule";
    alert.rule_name = "Test Rule";
    alert.severity = AlertSeverity::kWarning;
    alert.triggered_at = std::chrono::system_clock::now();
    alert.is_firing = true;
    alert.title = "Test Title";
    alert.description = "Test Description";
    alert.fingerprint = "test-fp";

    std::string json = SerializeAlertEvent(alert);
    EXPECT_FALSE(json.empty());

    auto deserialized = DeserializeAlertEvent(json);
    ASSERT_TRUE(deserialized.ok());

    EXPECT_EQ(deserialized->alert_id, alert.alert_id);
    EXPECT_EQ(deserialized->rule_id, alert.rule_id);
    EXPECT_EQ(deserialized->severity, alert.severity);
    EXPECT_EQ(deserialized->title, alert.title);
}

TEST(AlertingSilenceSerializationTest, SilenceSerialization) {
    Silence silence;
    silence.silence_id = "test-silence";
    silence.created_by = "test-user";
    silence.comment = "Test comment";
    silence.matchers["rule_id"] = "test-rule";
    silence.starts_at = std::chrono::system_clock::now();
    silence.ends_at = std::chrono::system_clock::now() + std::chrono::hours(1);

    std::string json = SerializeSilence(silence);
    EXPECT_FALSE(json.empty());

    auto deserialized = DeserializeSilence(json);
    ASSERT_TRUE(deserialized.ok());

    EXPECT_EQ(deserialized->silence_id, silence.silence_id);
    EXPECT_EQ(deserialized->created_by, silence.created_by);
    EXPECT_EQ(deserialized->matchers, silence.matchers);
}

}  // namespace
}  // namespace pyflare::alerting
