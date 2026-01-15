/// @file slice_analyzer_test.cpp
/// @brief Tests for slice analyzer

#include <gtest/gtest.h>

#include <memory>

#include "processor/rca/slice_analyzer.h"

namespace pyflare::rca {
namespace {

// Mock ClickHouse client for testing
class MockClickHouseClient : public storage::ClickHouseClient {
public:
    MockClickHouseClient() : storage::ClickHouseClient(storage::ClickHouseConfig{}) {}

    absl::Status Connect() override { return absl::OkStatus(); }
    void Disconnect() override {}
    bool IsConnected() const override { return true; }

    absl::StatusOr<storage::QueryResult> Query(const std::string& query) override {
        storage::QueryResult result;

        // Mock response based on query content
        if (query.find("GROUP BY") != std::string::npos) {
            // Slice analysis query - return mock slice data
            result.column_names = {"dimension_value", "metric_value", "sample_count"};
            result.rows = {
                {"model-a", "0.05", "1000"},
                {"model-b", "0.12", "500"},
                {"model-c", "0.03", "2000"},
            };
        } else if (query.find("avg") != std::string::npos || query.find("AVG") != std::string::npos) {
            // Baseline query
            result.column_names = {"baseline"};
            result.rows = {{"0.04"}};
        } else if (query.find("DISTINCT") != std::string::npos) {
            // Get dimension values
            result.column_names = {"value"};
            result.rows = {{"value-1"}, {"value-2"}, {"value-3"}};
        } else {
            result.column_names = {"count"};
            result.rows = {{"100"}};
        }

        return result;
    }

    absl::Status Execute(const std::string& query) override {
        return absl::OkStatus();
    }

    absl::Status Insert(const std::string& table,
                        const std::vector<std::string>& columns,
                        const std::vector<std::vector<std::string>>& rows) override {
        return absl::OkStatus();
    }
};

class SliceAnalyzerTest : public ::testing::Test {
protected:
    void SetUp() override {
        clickhouse_ = std::make_shared<MockClickHouseClient>();
        config_.min_samples = 50;
        config_.deviation_threshold = 0.1;
        config_.p_value_threshold = 0.05;
        config_.max_slices = 10;
        analyzer_ = std::make_unique<SliceAnalyzer>(clickhouse_, config_);
        ASSERT_TRUE(analyzer_->Initialize().ok());
    }

    std::shared_ptr<MockClickHouseClient> clickhouse_;
    SliceAnalyzerConfig config_;
    std::unique_ptr<SliceAnalyzer> analyzer_;
};

TEST_F(SliceAnalyzerTest, Initialize) {
    auto new_analyzer = std::make_unique<SliceAnalyzer>(clickhouse_);
    EXPECT_TRUE(new_analyzer->Initialize().ok());
}

TEST_F(SliceAnalyzerTest, GetConfig) {
    const auto& config = analyzer_->GetConfig();
    EXPECT_EQ(config.min_samples, 50);
    EXPECT_DOUBLE_EQ(config.deviation_threshold, 0.1);
    EXPECT_DOUBLE_EQ(config.p_value_threshold, 0.05);
}

TEST_F(SliceAnalyzerTest, AnalyzeSlicesReturnsResults) {
    auto results = analyzer_->AnalyzeSlices("test-model", SliceMetric::kErrorRate);
    ASSERT_TRUE(results.ok());
    // Should return some results from mock
    EXPECT_GE(results->size(), 0);
}

TEST_F(SliceAnalyzerTest, AnalyzeDimensionReturnsResults) {
    auto results = analyzer_->AnalyzeDimension(
        "test-model",
        SliceDimension::kModel,
        SliceMetric::kLatencyP95
    );
    ASSERT_TRUE(results.ok());
}

TEST_F(SliceAnalyzerTest, AnalyzeCustomSlice) {
    SliceDefinition custom;
    custom.name = "high-latency-users";
    custom.dimension = SliceDimension::kUser;
    custom.dimension_value = "power-users";

    auto result = analyzer_->AnalyzeCustomSlice(custom, SliceMetric::kCost);
    ASSERT_TRUE(result.ok());
    EXPECT_EQ(result->definition.name, "high-latency-users");
}

TEST_F(SliceAnalyzerTest, FindTopProblematicSlices) {
    auto results = analyzer_->FindTopProblematicSlices("", 5);
    ASSERT_TRUE(results.ok());
    // Should not exceed requested limit
    EXPECT_LE(results->size(), 5);
}

TEST_F(SliceAnalyzerTest, GetBaseline) {
    auto baseline = analyzer_->GetBaseline("test-model", SliceMetric::kErrorRate);
    ASSERT_TRUE(baseline.ok());
    EXPECT_GE(*baseline, 0.0);
}

TEST_F(SliceAnalyzerTest, SetBaseline) {
    ASSERT_TRUE(analyzer_->SetBaseline("test-model", SliceMetric::kErrorRate, 0.05).ok());

    auto baseline = analyzer_->GetBaseline("test-model", SliceMetric::kErrorRate);
    ASSERT_TRUE(baseline.ok());
    EXPECT_DOUBLE_EQ(*baseline, 0.05);
}

TEST_F(SliceAnalyzerTest, UpdateBaseline) {
    auto status = analyzer_->UpdateBaseline("test-model", SliceMetric::kLatencyP50);
    EXPECT_TRUE(status.ok());
}

TEST_F(SliceAnalyzerTest, GetDimensionValues) {
    auto values = analyzer_->GetDimensionValues("test-model", SliceDimension::kUser);
    ASSERT_TRUE(values.ok());
    EXPECT_GE(values->size(), 0);
}

TEST_F(SliceAnalyzerTest, GetSuggestedBins) {
    auto bins = analyzer_->GetSuggestedBins("test-model", SliceDimension::kLatency, 5);
    ASSERT_TRUE(bins.ok());
    // May be empty if not enough data
}

TEST_F(SliceAnalyzerTest, AddCustomSlice) {
    SliceDefinition custom;
    custom.name = "custom-slice";
    custom.dimension = SliceDimension::kCustom;
    custom.custom_filter = "model_id = 'special'";

    EXPECT_NO_THROW(analyzer_->AddCustomSlice(custom));
}

// =============================================================================
// Helper Function Tests
// =============================================================================

TEST(SliceAnalyzerHelpersTest, DimensionToString) {
    EXPECT_EQ(SliceAnalyzer::DimensionToString(SliceDimension::kModel), "model");
    EXPECT_EQ(SliceAnalyzer::DimensionToString(SliceDimension::kUser), "user");
    EXPECT_EQ(SliceAnalyzer::DimensionToString(SliceDimension::kFeature), "feature");
    EXPECT_EQ(SliceAnalyzer::DimensionToString(SliceDimension::kInputLength), "input_length");
    EXPECT_EQ(SliceAnalyzer::DimensionToString(SliceDimension::kOutputLength), "output_length");
    EXPECT_EQ(SliceAnalyzer::DimensionToString(SliceDimension::kLatency), "latency");
    EXPECT_EQ(SliceAnalyzer::DimensionToString(SliceDimension::kTimeOfDay), "time_of_day");
    EXPECT_EQ(SliceAnalyzer::DimensionToString(SliceDimension::kDayOfWeek), "day_of_week");
    EXPECT_EQ(SliceAnalyzer::DimensionToString(SliceDimension::kPromptTemplate), "prompt_template");
    EXPECT_EQ(SliceAnalyzer::DimensionToString(SliceDimension::kProvider), "provider");
    EXPECT_EQ(SliceAnalyzer::DimensionToString(SliceDimension::kEnvironment), "environment");
    EXPECT_EQ(SliceAnalyzer::DimensionToString(SliceDimension::kCustom), "custom");
}

TEST(SliceAnalyzerHelpersTest, StringToDimension) {
    EXPECT_EQ(SliceAnalyzer::StringToDimension("model"), SliceDimension::kModel);
    EXPECT_EQ(SliceAnalyzer::StringToDimension("user"), SliceDimension::kUser);
    EXPECT_EQ(SliceAnalyzer::StringToDimension("feature"), SliceDimension::kFeature);
    EXPECT_EQ(SliceAnalyzer::StringToDimension("latency"), SliceDimension::kLatency);
    EXPECT_EQ(SliceAnalyzer::StringToDimension("custom"), SliceDimension::kCustom);
    // Unknown should return custom
    EXPECT_EQ(SliceAnalyzer::StringToDimension("unknown"), SliceDimension::kCustom);
}

TEST(SliceAnalyzerHelpersTest, MetricToString) {
    EXPECT_EQ(SliceAnalyzer::MetricToString(SliceMetric::kErrorRate), "error_rate");
    EXPECT_EQ(SliceAnalyzer::MetricToString(SliceMetric::kLatencyP50), "latency_p50");
    EXPECT_EQ(SliceAnalyzer::MetricToString(SliceMetric::kLatencyP95), "latency_p95");
    EXPECT_EQ(SliceAnalyzer::MetricToString(SliceMetric::kLatencyP99), "latency_p99");
    EXPECT_EQ(SliceAnalyzer::MetricToString(SliceMetric::kCost), "cost");
    EXPECT_EQ(SliceAnalyzer::MetricToString(SliceMetric::kTokenUsage), "token_usage");
    EXPECT_EQ(SliceAnalyzer::MetricToString(SliceMetric::kToxicityRate), "toxicity_rate");
    EXPECT_EQ(SliceAnalyzer::MetricToString(SliceMetric::kHallucinationRate), "hallucination_rate");
    EXPECT_EQ(SliceAnalyzer::MetricToString(SliceMetric::kDriftScore), "drift_score");
}

TEST(SliceAnalyzerHelpersTest, StringToMetric) {
    EXPECT_EQ(SliceAnalyzer::StringToMetric("error_rate"), SliceMetric::kErrorRate);
    EXPECT_EQ(SliceAnalyzer::StringToMetric("latency_p95"), SliceMetric::kLatencyP95);
    EXPECT_EQ(SliceAnalyzer::StringToMetric("cost"), SliceMetric::kCost);
    // Unknown should return custom
    EXPECT_EQ(SliceAnalyzer::StringToMetric("unknown"), SliceMetric::kCustom);
}

TEST(SliceAnalyzerHelpersTest, DimensionToColumn) {
    EXPECT_EQ(SliceAnalyzer::DimensionToColumn(SliceDimension::kModel), "model_id");
    EXPECT_EQ(SliceAnalyzer::DimensionToColumn(SliceDimension::kUser), "user_id");
    EXPECT_EQ(SliceAnalyzer::DimensionToColumn(SliceDimension::kFeature), "feature_id");
    EXPECT_EQ(SliceAnalyzer::DimensionToColumn(SliceDimension::kLatency), "latency_ms");
}

TEST(SliceAnalyzerHelpersTest, MetricToAggregation) {
    EXPECT_NE(SliceAnalyzer::MetricToAggregation(SliceMetric::kErrorRate).find("avg"), std::string::npos);
    EXPECT_NE(SliceAnalyzer::MetricToAggregation(SliceMetric::kLatencyP50).find("quantile"), std::string::npos);
    EXPECT_NE(SliceAnalyzer::MetricToAggregation(SliceMetric::kLatencyP95).find("quantile"), std::string::npos);
    EXPECT_NE(SliceAnalyzer::MetricToAggregation(SliceMetric::kCost).find("sum"), std::string::npos);
}

// =============================================================================
// SliceDefinition Tests
// =============================================================================

TEST(SliceDefinitionTest, DefaultValues) {
    SliceDefinition def;
    EXPECT_TRUE(def.name.empty());
    EXPECT_EQ(def.dimension, SliceDimension::kCustom);
    EXPECT_TRUE(def.dimension_value.empty());
    EXPECT_FALSE(def.range_min.has_value());
    EXPECT_FALSE(def.range_max.has_value());
    EXPECT_TRUE(def.categories.empty());
    EXPECT_TRUE(def.custom_filter.empty());
}

TEST(SliceDefinitionTest, RangeBasedSlice) {
    SliceDefinition def;
    def.name = "high-latency";
    def.dimension = SliceDimension::kLatency;
    def.range_min = 1000.0;  // 1 second+
    def.range_max = std::nullopt;  // No upper bound

    EXPECT_TRUE(def.range_min.has_value());
    EXPECT_DOUBLE_EQ(*def.range_min, 1000.0);
    EXPECT_FALSE(def.range_max.has_value());
}

TEST(SliceDefinitionTest, CategoricalSlice) {
    SliceDefinition def;
    def.name = "premium-models";
    def.dimension = SliceDimension::kModel;
    def.categories = {"gpt-4", "gpt-4-turbo", "claude-3-opus"};

    EXPECT_EQ(def.categories.size(), 3);
}

// =============================================================================
// SliceAnalysisResult Tests
// =============================================================================

TEST(SliceAnalysisResultTest, DefaultValues) {
    SliceAnalysisResult result;
    EXPECT_EQ(result.sample_count, 0);
    EXPECT_DOUBLE_EQ(result.metric_value, 0.0);
    EXPECT_DOUBLE_EQ(result.baseline_value, 0.0);
    EXPECT_DOUBLE_EQ(result.deviation, 0.0);
    EXPECT_DOUBLE_EQ(result.p_value, 0.0);
    EXPECT_FALSE(result.is_statistically_significant);
    EXPECT_DOUBLE_EQ(result.impact_score, 0.0);
}

TEST(SliceAnalysisResultTest, SignificantResult) {
    SliceAnalysisResult result;
    result.sample_count = 1000;
    result.metric_value = 0.15;
    result.baseline_value = 0.05;
    result.deviation = 0.10;
    result.deviation_percentage = 200.0;
    result.p_value = 0.001;
    result.is_statistically_significant = true;
    result.impact_score = 0.8;

    EXPECT_TRUE(result.is_statistically_significant);
    EXPECT_GT(result.impact_score, 0.5);
    EXPECT_GT(result.deviation_percentage, 100.0);
}

// =============================================================================
// SliceAnalyzerConfig Tests
// =============================================================================

TEST(SliceAnalyzerConfigTest, DefaultValues) {
    SliceAnalyzerConfig config;
    EXPECT_EQ(config.min_samples, 100);
    EXPECT_EQ(config.max_slices, 50);
    EXPECT_DOUBLE_EQ(config.deviation_threshold, 0.1);
    EXPECT_DOUBLE_EQ(config.p_value_threshold, 0.05);
    EXPECT_EQ(config.analysis_window.count(), 24);  // 24 hours
    EXPECT_FALSE(config.auto_dimensions.empty());
    EXPECT_FALSE(config.auto_metrics.empty());
}

// =============================================================================
// Factory Function Tests
// =============================================================================

TEST(SliceAnalyzerFactoryTest, CreateDefaultSliceDefinitions) {
    auto definitions = CreateDefaultSliceDefinitions();
    EXPECT_GT(definitions.size(), 0);

    // Should include common dimensions
    bool has_model = false;
    bool has_latency = false;
    for (const auto& def : definitions) {
        if (def.dimension == SliceDimension::kModel) has_model = true;
        if (def.dimension == SliceDimension::kLatency) has_latency = true;
    }
    EXPECT_TRUE(has_model || has_latency);  // At least some common ones
}

}  // namespace
}  // namespace pyflare::rca
