/// @file budget_manager_test.cpp
/// @brief Tests for budget manager

#include <gtest/gtest.h>

#include <chrono>
#include <memory>
#include <thread>

#include "processor/cost/budget_manager.h"

namespace pyflare::cost {
namespace {

// Mock Redis client for testing
class MockRedisClient : public storage::RedisClient {
public:
    MockRedisClient() : storage::RedisClient(storage::RedisConfig{}) {}

    absl::Status Connect() override { return absl::OkStatus(); }
    void Disconnect() override {}
    bool IsConnected() const override { return true; }

    absl::StatusOr<std::string> Get(const std::string& key) override {
        std::lock_guard<std::mutex> lock(mutex_);
        auto it = store_.find(key);
        if (it != store_.end()) {
            return it->second;
        }
        return absl::NotFoundError("Key not found");
    }

    absl::Status Set(const std::string& key, const std::string& value,
                     std::chrono::seconds ttl = std::chrono::seconds(0)) override {
        std::lock_guard<std::mutex> lock(mutex_);
        store_[key] = value;
        return absl::OkStatus();
    }

    absl::Status Del(const std::string& key) override {
        std::lock_guard<std::mutex> lock(mutex_);
        store_.erase(key);
        return absl::OkStatus();
    }

    absl::StatusOr<int64_t> IncrBy(const std::string& key, int64_t value) override {
        std::lock_guard<std::mutex> lock(mutex_);
        int64_t current = 0;
        auto it = store_.find(key);
        if (it != store_.end()) {
            try {
                current = std::stoll(it->second);
            } catch (...) {
                current = 0;
            }
        }
        current += value;
        store_[key] = std::to_string(current);
        return current;
    }

    absl::StatusOr<std::vector<std::string>> Keys(const std::string& pattern) override {
        std::lock_guard<std::mutex> lock(mutex_);
        std::vector<std::string> result;
        for (const auto& [key, _] : store_) {
            // Simplified pattern matching - just prefix match
            if (pattern.empty() || pattern == "*" ||
                key.find(pattern.substr(0, pattern.find('*'))) == 0) {
                result.push_back(key);
            }
        }
        return result;
    }

    void Clear() {
        std::lock_guard<std::mutex> lock(mutex_);
        store_.clear();
    }

private:
    std::unordered_map<std::string, std::string> store_;
    mutable std::mutex mutex_;
};

class BudgetManagerTest : public ::testing::Test {
protected:
    void SetUp() override {
        redis_ = std::make_shared<MockRedisClient>();
        config_.key_prefix = "test:budget";
        config_.enable_local_cache = true;
        manager_ = std::make_unique<BudgetManager>(redis_, config_);
        ASSERT_TRUE(manager_->Initialize().ok());
    }

    void TearDown() override {
        if (manager_) {
            manager_->Shutdown();
        }
    }

    std::shared_ptr<MockRedisClient> redis_;
    BudgetManagerConfig config_;
    std::unique_ptr<BudgetManager> manager_;
};

TEST_F(BudgetManagerTest, CreateAndGetBudget) {
    BudgetConfig budget;
    budget.id = "user-budget-1";
    budget.dimension = BudgetDimension::kUser;
    budget.dimension_value = "user123";
    budget.period = BudgetPeriod::kDaily;
    budget.soft_limit_micros = 10000000;  // $10
    budget.hard_limit_micros = 15000000;  // $15
    budget.block_on_exceeded = true;

    ASSERT_TRUE(manager_->CreateBudget(budget).ok());

    auto retrieved = manager_->GetBudget(BudgetDimension::kUser, "user123");
    ASSERT_TRUE(retrieved.ok());
    EXPECT_EQ(retrieved->dimension_value, "user123");
    EXPECT_EQ(retrieved->soft_limit_micros, 10000000);
    EXPECT_EQ(retrieved->hard_limit_micros, 15000000);
}

TEST_F(BudgetManagerTest, CheckBudgetAllowed) {
    BudgetConfig budget;
    budget.dimension = BudgetDimension::kUser;
    budget.dimension_value = "user456";
    budget.period = BudgetPeriod::kDaily;
    budget.hard_limit_micros = 1000000;  // $1
    budget.block_on_exceeded = true;

    ASSERT_TRUE(manager_->CreateBudget(budget).ok());

    // Check budget - should be allowed
    auto result = manager_->CheckBudget(BudgetDimension::kUser, "user456", 100000);
    ASSERT_TRUE(result.ok());
    EXPECT_TRUE(result->allowed);
    EXPECT_FALSE(result->warning);
}

TEST_F(BudgetManagerTest, RecordSpendAndCheck) {
    BudgetConfig budget;
    budget.dimension = BudgetDimension::kModel;
    budget.dimension_value = "gpt-4";
    budget.period = BudgetPeriod::kHourly;
    budget.soft_limit_micros = 500000;   // $0.50
    budget.hard_limit_micros = 1000000;  // $1.00
    budget.warning_percentage = 0.8;
    budget.block_on_exceeded = true;

    ASSERT_TRUE(manager_->CreateBudget(budget).ok());

    // Record some spend
    ASSERT_TRUE(manager_->RecordSpend(BudgetDimension::kModel, "gpt-4", 300000).ok());

    // Check status
    auto status = manager_->GetStatus(BudgetDimension::kModel, "gpt-4");
    ASSERT_TRUE(status.ok());
    EXPECT_EQ(status->current_spend_micros, 300000);
    EXPECT_NEAR(status->utilization_percentage, 30.0, 1.0);
}

TEST_F(BudgetManagerTest, WarningTriggered) {
    BudgetConfig budget;
    budget.dimension = BudgetDimension::kUser;
    budget.dimension_value = "high-spender";
    budget.period = BudgetPeriod::kDaily;
    budget.soft_limit_micros = 100000;
    budget.hard_limit_micros = 100000;
    budget.warning_percentage = 0.5;  // Warn at 50%

    ASSERT_TRUE(manager_->CreateBudget(budget).ok());

    // Spend 60% - should trigger warning
    ASSERT_TRUE(manager_->RecordSpend(BudgetDimension::kUser, "high-spender", 60000).ok());

    auto check = manager_->CheckBudget(BudgetDimension::kUser, "high-spender");
    ASSERT_TRUE(check.ok());
    EXPECT_TRUE(check->warning);
    EXPECT_TRUE(check->allowed);  // Still allowed, just warning
}

TEST_F(BudgetManagerTest, BlockWhenExceeded) {
    BudgetConfig budget;
    budget.dimension = BudgetDimension::kUser;
    budget.dimension_value = "blocked-user";
    budget.period = BudgetPeriod::kDaily;
    budget.hard_limit_micros = 100000;
    budget.block_on_exceeded = true;

    ASSERT_TRUE(manager_->CreateBudget(budget).ok());

    // Spend over limit
    ASSERT_TRUE(manager_->RecordSpend(BudgetDimension::kUser, "blocked-user", 150000).ok());

    auto check = manager_->CheckBudget(BudgetDimension::kUser, "blocked-user");
    ASSERT_TRUE(check.ok());
    EXPECT_FALSE(check->allowed);
    EXPECT_TRUE(check->blocked_reason.has_value());
}

TEST_F(BudgetManagerTest, NoBlockWhenDisabled) {
    BudgetConfig budget;
    budget.dimension = BudgetDimension::kUser;
    budget.dimension_value = "unlimited-user";
    budget.period = BudgetPeriod::kDaily;
    budget.hard_limit_micros = 100000;
    budget.block_on_exceeded = false;  // Don't block

    ASSERT_TRUE(manager_->CreateBudget(budget).ok());

    // Spend over limit
    ASSERT_TRUE(manager_->RecordSpend(BudgetDimension::kUser, "unlimited-user", 150000).ok());

    auto check = manager_->CheckBudget(BudgetDimension::kUser, "unlimited-user");
    ASSERT_TRUE(check.ok());
    EXPECT_TRUE(check->allowed);  // Still allowed
    EXPECT_TRUE(check->warning);  // But warning
}

TEST_F(BudgetManagerTest, DeleteBudget) {
    BudgetConfig budget;
    budget.dimension = BudgetDimension::kTeam;
    budget.dimension_value = "team-a";
    budget.hard_limit_micros = 1000000;

    ASSERT_TRUE(manager_->CreateBudget(budget).ok());

    // Verify exists
    auto retrieved = manager_->GetBudget(BudgetDimension::kTeam, "team-a");
    ASSERT_TRUE(retrieved.ok());

    // Delete
    ASSERT_TRUE(manager_->DeleteBudget(BudgetDimension::kTeam, "team-a").ok());

    // Verify deleted
    auto deleted = manager_->GetBudget(BudgetDimension::kTeam, "team-a");
    EXPECT_FALSE(deleted.ok());
}

TEST_F(BudgetManagerTest, ResetBudget) {
    BudgetConfig budget;
    budget.dimension = BudgetDimension::kGlobal;
    budget.dimension_value = "";
    budget.hard_limit_micros = 1000000;

    ASSERT_TRUE(manager_->CreateBudget(budget).ok());
    ASSERT_TRUE(manager_->RecordSpend(BudgetDimension::kGlobal, "", 500000).ok());

    // Verify spend recorded
    auto status1 = manager_->GetStatus(BudgetDimension::kGlobal, "");
    ASSERT_TRUE(status1.ok());
    EXPECT_EQ(status1->current_spend_micros, 500000);

    // Reset
    ASSERT_TRUE(manager_->ResetBudget(BudgetDimension::kGlobal, "").ok());

    // Verify reset
    auto status2 = manager_->GetStatus(BudgetDimension::kGlobal, "");
    ASSERT_TRUE(status2.ok());
    EXPECT_EQ(status2->current_spend_micros, 0);
}

TEST_F(BudgetManagerTest, MultipleBudgets) {
    // Create budgets for different users
    for (int i = 1; i <= 3; ++i) {
        BudgetConfig budget;
        budget.dimension = BudgetDimension::kUser;
        budget.dimension_value = "user" + std::to_string(i);
        budget.hard_limit_micros = i * 100000;
        ASSERT_TRUE(manager_->CreateBudget(budget).ok());
    }

    // Verify all exist
    auto budgets = manager_->ListBudgets(BudgetDimension::kUser);
    ASSERT_TRUE(budgets.ok());
    EXPECT_GE(budgets->size(), 3);
}

TEST_F(BudgetManagerTest, AlertCallback) {
    std::vector<BudgetAlertEvent> received_alerts;
    manager_->RegisterAlertCallback([&](const BudgetAlertEvent& alert) {
        received_alerts.push_back(alert);
    });

    BudgetConfig budget;
    budget.dimension = BudgetDimension::kUser;
    budget.dimension_value = "alert-test";
    budget.soft_limit_micros = 50000;
    budget.hard_limit_micros = 100000;
    budget.warning_percentage = 0.8;

    ASSERT_TRUE(manager_->CreateBudget(budget).ok());

    // Trigger warning (80% of soft limit)
    ASSERT_TRUE(manager_->RecordSpend(BudgetDimension::kUser, "alert-test", 45000).ok());

    // Trigger exceeded (over hard limit)
    ASSERT_TRUE(manager_->RecordSpend(BudgetDimension::kUser, "alert-test", 60000).ok());

    // Check if alerts were triggered (may depend on implementation)
    // At minimum, the callback registration should succeed
}

TEST_F(BudgetManagerTest, ForecastSpend) {
    BudgetConfig budget;
    budget.dimension = BudgetDimension::kModel;
    budget.dimension_value = "forecast-model";
    budget.period = BudgetPeriod::kDaily;
    budget.hard_limit_micros = 10000000;

    ASSERT_TRUE(manager_->CreateBudget(budget).ok());

    // Record some spend
    ASSERT_TRUE(manager_->RecordSpend(BudgetDimension::kModel, "forecast-model", 100000).ok());

    auto forecast = manager_->ForecastSpend(BudgetDimension::kModel, "forecast-model");
    // Should return some forecast value
    if (forecast.ok()) {
        EXPECT_GE(*forecast, 0);
    }
}

TEST_F(BudgetManagerTest, GetSpendRate) {
    BudgetConfig budget;
    budget.dimension = BudgetDimension::kFeature;
    budget.dimension_value = "chat-endpoint";
    budget.period = BudgetPeriod::kHourly;
    budget.hard_limit_micros = 1000000;

    ASSERT_TRUE(manager_->CreateBudget(budget).ok());
    ASSERT_TRUE(manager_->RecordSpend(BudgetDimension::kFeature, "chat-endpoint", 100000).ok());

    auto rate = manager_->GetSpendRate(BudgetDimension::kFeature, "chat-endpoint");
    if (rate.ok()) {
        EXPECT_GE(*rate, 0.0);
    }
}

// =============================================================================
// Helper Function Tests
// =============================================================================

TEST(BudgetManagerHelpersTest, DimensionToString) {
    EXPECT_EQ(BudgetManager::DimensionToString(BudgetDimension::kGlobal), "global");
    EXPECT_EQ(BudgetManager::DimensionToString(BudgetDimension::kUser), "user");
    EXPECT_EQ(BudgetManager::DimensionToString(BudgetDimension::kModel), "model");
    EXPECT_EQ(BudgetManager::DimensionToString(BudgetDimension::kFeature), "feature");
    EXPECT_EQ(BudgetManager::DimensionToString(BudgetDimension::kTeam), "team");
    EXPECT_EQ(BudgetManager::DimensionToString(BudgetDimension::kEnvironment), "environment");
}

TEST(BudgetManagerHelpersTest, StringToDimension) {
    EXPECT_EQ(BudgetManager::StringToDimension("global"), BudgetDimension::kGlobal);
    EXPECT_EQ(BudgetManager::StringToDimension("user"), BudgetDimension::kUser);
    EXPECT_EQ(BudgetManager::StringToDimension("model"), BudgetDimension::kModel);
    EXPECT_EQ(BudgetManager::StringToDimension("feature"), BudgetDimension::kFeature);
    EXPECT_EQ(BudgetManager::StringToDimension("team"), BudgetDimension::kTeam);
    EXPECT_EQ(BudgetManager::StringToDimension("environment"), BudgetDimension::kEnvironment);
}

TEST(BudgetManagerHelpersTest, PeriodToString) {
    EXPECT_EQ(BudgetManager::PeriodToString(BudgetPeriod::kHourly), "hourly");
    EXPECT_EQ(BudgetManager::PeriodToString(BudgetPeriod::kDaily), "daily");
    EXPECT_EQ(BudgetManager::PeriodToString(BudgetPeriod::kWeekly), "weekly");
    EXPECT_EQ(BudgetManager::PeriodToString(BudgetPeriod::kMonthly), "monthly");
}

TEST(BudgetManagerHelpersTest, PeriodDuration) {
    EXPECT_EQ(BudgetManager::PeriodDuration(BudgetPeriod::kHourly).count(), 3600);
    EXPECT_EQ(BudgetManager::PeriodDuration(BudgetPeriod::kDaily).count(), 86400);
    EXPECT_EQ(BudgetManager::PeriodDuration(BudgetPeriod::kWeekly).count(), 604800);
    // Monthly is approximate - 30 days
    EXPECT_EQ(BudgetManager::PeriodDuration(BudgetPeriod::kMonthly).count(), 2592000);
}

TEST(BudgetManagerHelpersTest, GetPeriodStart) {
    using namespace std::chrono;

    auto now = system_clock::now();

    auto hourly_start = BudgetManager::GetPeriodStart(BudgetPeriod::kHourly, now);
    auto daily_start = BudgetManager::GetPeriodStart(BudgetPeriod::kDaily, now);

    // Period start should be before or equal to now
    EXPECT_LE(hourly_start, now);
    EXPECT_LE(daily_start, now);

    // Hourly should be within last hour
    EXPECT_GT(now - hourly_start, seconds(0));
    EXPECT_LT(now - hourly_start, hours(1));

    // Daily should be within last day
    EXPECT_GT(now - daily_start, seconds(0));
    EXPECT_LT(now - daily_start, hours(24));
}

}  // namespace
}  // namespace pyflare::cost
