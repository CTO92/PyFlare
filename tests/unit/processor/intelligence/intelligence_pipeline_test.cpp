/// @file intelligence_pipeline_test.cpp
/// @brief Unit tests for intelligence pipeline

#include <gtest/gtest.h>
#include <gmock/gmock.h>

#include "processor/intelligence/intelligence_pipeline.h"

namespace pyflare::intelligence {
namespace {

using ::testing::_;
using ::testing::Return;

// Mock storage clients
class MockClickHouseClient : public storage::ClickHouseClient {
public:
    MOCK_METHOD(absl::Status, Execute, (const std::string& query), (override));
    MOCK_METHOD(absl::StatusOr<std::vector<std::unordered_map<std::string, std::string>>>,
                Query, (const std::string& query), (override));
};

class MockQdrantClient : public storage::QdrantClient {
public:
    MOCK_METHOD(absl::Status, Upsert, (const std::string& collection,
                                       const std::vector<std::pair<std::string, std::vector<float>>>& vectors), (override));
    MOCK_METHOD(absl::StatusOr<std::vector<std::pair<std::string, float>>>,
                Search, (const std::string& collection,
                        const std::vector<float>& query,
                        size_t limit), (override));
};

class MockRedisClient : public storage::RedisClient {
public:
    MOCK_METHOD(absl::Status, Set, (const std::string& key,
                                    const std::string& value,
                                    std::chrono::seconds ttl), (override));
    MOCK_METHOD(absl::StatusOr<std::optional<std::string>>,
                Get, (const std::string& key), (override));
};

class IntelligencePipelineTest : public ::testing::Test {
protected:
    void SetUp() override {
        clickhouse_ = std::make_shared<MockClickHouseClient>();
        qdrant_ = std::make_shared<MockQdrantClient>();
        redis_ = std::make_shared<MockRedisClient>();

        config_.enable_drift_detection = false;  // Disable for basic tests
        config_.enable_evaluations = false;
        config_.enable_safety_checks = false;
        config_.enable_rca = false;
        config_.enable_alerting = false;
    }

    std::shared_ptr<MockClickHouseClient> clickhouse_;
    std::shared_ptr<MockQdrantClient> qdrant_;
    std::shared_ptr<MockRedisClient> redis_;
    IntelligencePipelineConfig config_;
};

TEST_F(IntelligencePipelineTest, InitializesSuccessfully) {
    auto pipeline = CreateIntelligencePipeline(clickhouse_, qdrant_, redis_, config_);
    ASSERT_NE(pipeline, nullptr);

    auto status = pipeline->Initialize();
    EXPECT_TRUE(status.ok());
}

TEST_F(IntelligencePipelineTest, ProcessReturnsResult) {
    auto pipeline = CreateIntelligencePipeline(clickhouse_, qdrant_, redis_, config_);
    ASSERT_TRUE(pipeline->Initialize().ok());

    eval::InferenceRecord record;
    record.trace_id = "test-trace-1";
    record.model_id = "test-model";
    record.input = "What is 2+2?";
    record.output = "4";

    auto result = pipeline->Process(record);

    EXPECT_EQ(result.trace_id, "test-trace-1");
    EXPECT_EQ(result.model_id, "test-model");
    EXPECT_GE(result.health_score, 0.0);
    EXPECT_LE(result.health_score, 1.0);
}

TEST_F(IntelligencePipelineTest, ProcessBatchReturnsResults) {
    auto pipeline = CreateIntelligencePipeline(clickhouse_, qdrant_, redis_, config_);
    ASSERT_TRUE(pipeline->Initialize().ok());

    std::vector<eval::InferenceRecord> records;
    for (int i = 0; i < 5; i++) {
        eval::InferenceRecord record;
        record.trace_id = "trace-" + std::to_string(i);
        record.model_id = "test-model";
        record.input = "Input " + std::to_string(i);
        record.output = "Output " + std::to_string(i);
        records.push_back(record);
    }

    auto batch_result = pipeline->ProcessBatch(records);

    EXPECT_EQ(batch_result.total_processed, 5);
    EXPECT_EQ(batch_result.results.size(), 5);
}

TEST_F(IntelligencePipelineTest, StatsAreUpdated) {
    auto pipeline = CreateIntelligencePipeline(clickhouse_, qdrant_, redis_, config_);
    ASSERT_TRUE(pipeline->Initialize().ok());

    eval::InferenceRecord record;
    record.trace_id = "test-trace";
    record.model_id = "test-model";
    record.input = "Test input";
    record.output = "Test output";

    pipeline->Process(record);
    pipeline->Process(record);
    pipeline->Process(record);

    auto stats = pipeline->GetStats();
    EXPECT_EQ(stats.total_processed, 3);
}

TEST_F(IntelligencePipelineTest, ResetStatsClearsCounters) {
    auto pipeline = CreateIntelligencePipeline(clickhouse_, qdrant_, redis_, config_);
    ASSERT_TRUE(pipeline->Initialize().ok());

    eval::InferenceRecord record;
    record.trace_id = "test-trace";
    record.model_id = "test-model";
    record.input = "Test input";
    record.output = "Test output";

    pipeline->Process(record);
    pipeline->ResetStats();

    auto stats = pipeline->GetStats();
    EXPECT_EQ(stats.total_processed, 0);
}

TEST_F(IntelligencePipelineTest, CallbacksAreCalled) {
    auto pipeline = CreateIntelligencePipeline(clickhouse_, qdrant_, redis_, config_);
    ASSERT_TRUE(pipeline->Initialize().ok());

    int callback_count = 0;
    pipeline->OnResult([&callback_count](const IntelligenceResult&) {
        callback_count++;
    });

    eval::InferenceRecord record;
    record.trace_id = "test-trace";
    record.model_id = "test-model";
    record.input = "Test input";
    record.output = "Test output";

    pipeline->Process(record);
    pipeline->Process(record);

    EXPECT_EQ(callback_count, 2);
}

TEST_F(IntelligencePipelineTest, ClearCallbacksRemovesAll) {
    auto pipeline = CreateIntelligencePipeline(clickhouse_, qdrant_, redis_, config_);
    ASSERT_TRUE(pipeline->Initialize().ok());

    int callback_count = 0;
    pipeline->OnResult([&callback_count](const IntelligenceResult&) {
        callback_count++;
    });

    pipeline->ClearCallbacks();

    eval::InferenceRecord record;
    record.trace_id = "test-trace";
    record.model_id = "test-model";
    record.input = "Test input";
    record.output = "Test output";

    pipeline->Process(record);

    EXPECT_EQ(callback_count, 0);
}

TEST_F(IntelligencePipelineTest, GetSystemHealthReturnsDefault) {
    auto pipeline = CreateIntelligencePipeline(clickhouse_, qdrant_, redis_, config_);
    ASSERT_TRUE(pipeline->Initialize().ok());

    auto health = pipeline->GetSystemHealth();
    EXPECT_EQ(health.models_analyzed, 0);
    EXPECT_EQ(health.total_active_alerts, 0);
}

TEST_F(IntelligencePipelineTest, ListModelsReturnsEmpty) {
    auto pipeline = CreateIntelligencePipeline(clickhouse_, qdrant_, redis_, config_);
    ASSERT_TRUE(pipeline->Initialize().ok());

    auto models = pipeline->ListModels();
    EXPECT_TRUE(models.empty());
}

TEST_F(IntelligencePipelineTest, ListEvaluatorsReturnsEmpty) {
    auto pipeline = CreateIntelligencePipeline(clickhouse_, qdrant_, redis_, config_);
    ASSERT_TRUE(pipeline->Initialize().ok());

    auto evaluators = pipeline->ListEvaluators();
    EXPECT_TRUE(evaluators.empty());
}

// Serialization tests
TEST(IntelligenceResultSerializationTest, SerializeAndDeserialize) {
    IntelligenceResult original;
    original.trace_id = "test-trace-123";
    original.model_id = "test-model";
    original.health_score = 0.85;
    original.drift.drift_detected = true;
    original.drift.overall_severity = 0.6;
    original.drift.drifted_dimensions = {"feature", "embedding"};
    original.evaluation.overall_score = 0.9;
    original.evaluation.passed = true;
    original.safety.is_safe = true;
    original.safety.risk_score = 0.1;

    std::string json = SerializeIntelligenceResult(original);
    EXPECT_FALSE(json.empty());

    auto deserialized = DeserializeIntelligenceResult(json);
    ASSERT_TRUE(deserialized.ok());

    EXPECT_EQ(deserialized->trace_id, original.trace_id);
    EXPECT_EQ(deserialized->model_id, original.model_id);
    EXPECT_DOUBLE_EQ(deserialized->health_score, original.health_score);
    EXPECT_EQ(deserialized->drift.drift_detected, original.drift.drift_detected);
    EXPECT_DOUBLE_EQ(deserialized->drift.overall_severity, original.drift.overall_severity);
    EXPECT_EQ(deserialized->drift.drifted_dimensions, original.drift.drifted_dimensions);
}

TEST(IntelligenceResultSerializationTest, DeserializeInvalidJsonFails) {
    auto result = DeserializeIntelligenceResult("not valid json");
    EXPECT_FALSE(result.ok());
}

}  // namespace
}  // namespace pyflare::intelligence
