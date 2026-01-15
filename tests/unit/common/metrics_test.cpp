/// @file metrics_test.cpp
/// @brief Tests for PyFlare metrics collection

#include <gtest/gtest.h>

#include "common/metrics.h"

namespace pyflare {
namespace {

class MetricsTest : public ::testing::Test {
protected:
    void SetUp() override {
        MetricsRegistry::Instance().Reset();
    }
};

TEST_F(MetricsTest, CounterIncrement) {
    auto& counter = MetricsRegistry::Instance().GetCounter("test_counter");

    EXPECT_EQ(counter.Value(), 0);

    counter.Increment();
    EXPECT_EQ(counter.Value(), 1);

    counter.Add(5);
    EXPECT_EQ(counter.Value(), 6);
}

TEST_F(MetricsTest, CounterNegativeIgnored) {
    auto& counter = MetricsRegistry::Instance().GetCounter("test_counter");

    counter.Add(10);
    counter.Add(-5);  // Should be ignored

    EXPECT_EQ(counter.Value(), 10);
}

TEST_F(MetricsTest, GaugeOperations) {
    auto& gauge = MetricsRegistry::Instance().GetGauge("test_gauge");

    gauge.Set(100.0);
    EXPECT_EQ(gauge.Value(), 100.0);

    gauge.Increment(10.0);
    EXPECT_EQ(gauge.Value(), 110.0);

    gauge.Decrement(30.0);
    EXPECT_EQ(gauge.Value(), 80.0);
}

TEST_F(MetricsTest, HistogramObservations) {
    auto& histogram = MetricsRegistry::Instance().GetHistogram("test_histogram");

    histogram.Observe(0.005);
    histogram.Observe(0.050);
    histogram.Observe(1.000);

    EXPECT_EQ(histogram.Count(), 3);
    EXPECT_NEAR(histogram.Sum(), 1.055, 0.001);
}

TEST_F(MetricsTest, HistogramBuckets) {
    std::vector<double> buckets = {0.01, 0.1, 1.0, 10.0};
    Histogram histogram("test", buckets);

    histogram.Observe(0.005);  // bucket 0.01
    histogram.Observe(0.05);   // bucket 0.1
    histogram.Observe(0.5);    // bucket 1.0
    histogram.Observe(5.0);    // bucket 10.0
    histogram.Observe(50.0);   // +Inf bucket

    auto bucket_counts = histogram.Buckets();
    ASSERT_EQ(bucket_counts.size(), 5);  // 4 buckets + Inf

    // Cumulative counts
    EXPECT_EQ(bucket_counts[0].second, 1);  // <= 0.01
    EXPECT_EQ(bucket_counts[1].second, 2);  // <= 0.1
    EXPECT_EQ(bucket_counts[2].second, 3);  // <= 1.0
    EXPECT_EQ(bucket_counts[3].second, 4);  // <= 10.0
    EXPECT_EQ(bucket_counts[4].second, 5);  // +Inf
}

TEST_F(MetricsTest, ScopedTimer) {
    auto& histogram = MetricsRegistry::Instance().GetHistogram("timer_test");

    {
        ScopedTimer timer(histogram);
        // Simulate some work
    }

    EXPECT_EQ(histogram.Count(), 1);
    EXPECT_GT(histogram.Sum(), 0);
}

TEST_F(MetricsTest, RegistrySingleton) {
    auto& counter1 = MetricsRegistry::Instance().GetCounter("same_counter");
    auto& counter2 = MetricsRegistry::Instance().GetCounter("same_counter");

    counter1.Increment();
    EXPECT_EQ(counter2.Value(), 1);

    // They should be the same object
    EXPECT_EQ(&counter1, &counter2);
}

TEST_F(MetricsTest, ExportText) {
    auto& counter = MetricsRegistry::Instance().GetCounter("my_counter", "A test counter");
    counter.Add(42);

    auto& gauge = MetricsRegistry::Instance().GetGauge("my_gauge", "A test gauge");
    gauge.Set(3.14);

    std::string output = MetricsRegistry::Instance().ExportText();

    EXPECT_NE(output.find("my_counter"), std::string::npos);
    EXPECT_NE(output.find("42"), std::string::npos);
    EXPECT_NE(output.find("my_gauge"), std::string::npos);
}

}  // namespace
}  // namespace pyflare
