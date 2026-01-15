/// @file cost_tracker_test.cpp
/// @brief Tests for cost tracking

#include <gtest/gtest.h>

#include "processor/cost/tracker.h"

namespace pyflare::cost {
namespace {

TEST(CostTrackerTest, CalculateGPT4Cost) {
    CostTracker tracker;

    TraceRecord record{
        .trace_id = "test-trace",
        .model_id = "gpt-4",
        .input_tokens = 1000,
        .output_tokens = 500,
        .timestamp = std::chrono::system_clock::now()
    };

    auto result = tracker.Calculate(record);
    ASSERT_TRUE(result.ok());

    EXPECT_EQ(result->trace_id, "test-trace");
    EXPECT_EQ(result->model_id, "gpt-4");
    EXPECT_EQ(result->input_tokens, 1000);
    EXPECT_EQ(result->output_tokens, 500);
    EXPECT_EQ(result->total_tokens, 1500);

    // GPT-4 pricing: $30/M input, $60/M output
    // 1000 * 30 / 1000 = 30 micro-dollars input
    // 500 * 60 / 1000 = 30 micro-dollars output
    EXPECT_EQ(result->input_cost_micros, 30);
    EXPECT_EQ(result->output_cost_micros, 30);
    EXPECT_EQ(result->total_cost_micros, 60);
}

TEST(CostTrackerTest, UnknownModel) {
    CostTracker tracker;

    TraceRecord record{
        .trace_id = "test-trace",
        .model_id = "unknown-model",
        .input_tokens = 100,
        .output_tokens = 50,
        .timestamp = std::chrono::system_clock::now()
    };

    auto result = tracker.Calculate(record);
    EXPECT_FALSE(result.ok());
}

TEST(CostTrackerTest, UpdatePricing) {
    CostTracker tracker;

    ModelPricing new_pricing{
        .model_id = "custom-model",
        .provider = "custom",
        .input_cost_per_million_tokens = 1000000,   // $1/M
        .output_cost_per_million_tokens = 2000000,  // $2/M
        .effective_from = std::chrono::system_clock::now()
    };

    ASSERT_TRUE(tracker.UpdatePricing(new_pricing).ok());

    TraceRecord record{
        .trace_id = "test",
        .model_id = "custom-model",
        .input_tokens = 1000000,
        .output_tokens = 500000,
        .timestamp = std::chrono::system_clock::now()
    };

    auto result = tracker.Calculate(record);
    ASSERT_TRUE(result.ok());

    EXPECT_EQ(result->input_cost_micros, 1000000);   // $1
    EXPECT_EQ(result->output_cost_micros, 1000000);  // $1 (half the tokens)
}

TEST(CostTrackerTest, GetPricing) {
    CostTracker tracker;

    auto result = tracker.GetPricing("gpt-4");
    ASSERT_TRUE(result.ok());
    EXPECT_EQ(result->provider, "openai");

    auto missing = tracker.GetPricing("nonexistent");
    EXPECT_FALSE(missing.ok());
}

TEST(CostTrackerTest, SpendTracking) {
    CostTracker tracker;

    TraceRecord record1{
        .model_id = "gpt-3.5-turbo",
        .input_tokens = 1000,
        .output_tokens = 500,
        .user_id = "user-123",
        .timestamp = std::chrono::system_clock::now()
    };

    TraceRecord record2{
        .model_id = "gpt-3.5-turbo",
        .input_tokens = 2000,
        .output_tokens = 1000,
        .user_id = "user-123",
        .timestamp = std::chrono::system_clock::now()
    };

    ASSERT_TRUE(tracker.Calculate(record1).ok());
    ASSERT_TRUE(tracker.Calculate(record2).ok());

    // Spend should be tracked
    int64_t spend = tracker.GetSpend("user", "user-123");
    EXPECT_GT(spend, 0);
}

TEST(CostTrackerTest, BudgetAlert) {
    CostTracker tracker;

    bool alert_triggered = false;
    int64_t alert_spend = 0;

    tracker.SetBudgetAlert(
        "user",
        "test-user",
        100,  // Low threshold
        [&](const BudgetAlert& alert) {
            alert_triggered = true;
            alert_spend = alert.current_spend_micros;
        }
    );

    // Make expensive request
    TraceRecord record{
        .model_id = "gpt-4",
        .input_tokens = 10000,
        .output_tokens = 5000,
        .user_id = "test-user",
        .timestamp = std::chrono::system_clock::now()
    };

    ASSERT_TRUE(tracker.Calculate(record).ok());

    // Alert should have been triggered
    EXPECT_TRUE(alert_triggered);
    EXPECT_GT(alert_spend, 100);
}

}  // namespace
}  // namespace pyflare::cost
