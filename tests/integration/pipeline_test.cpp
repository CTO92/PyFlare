/// @file pipeline_test.cpp
/// @brief Integration tests for the message processing pipeline

#include <gtest/gtest.h>

#include <chrono>
#include <memory>
#include <thread>
#include <vector>

#include "processor/pipeline.h"
#include "processor/message_processor.h"

namespace pyflare::processor {
namespace {

// =============================================================================
// Test Processors
// =============================================================================

/// @brief Test processor that counts processed spans
class CountingProcessor : public MessageProcessor {
public:
    absl::StatusOr<std::vector<Span>> Process(std::vector<Span>&& spans) override {
        processed_count_ += spans.size();
        return std::move(spans);
    }

    std::string Name() const override { return "CountingProcessor"; }
    bool IsHealthy() const override { return true; }
    absl::Status Flush() override { return absl::OkStatus(); }

    size_t GetProcessedCount() const { return processed_count_; }

private:
    std::atomic<size_t> processed_count_{0};
};

/// @brief Test processor that filters spans by status
class FilteringProcessor : public MessageProcessor {
public:
    explicit FilteringProcessor(const std::string& status_to_keep)
        : status_to_keep_(status_to_keep) {}

    absl::StatusOr<std::vector<Span>> Process(std::vector<Span>&& spans) override {
        std::vector<Span> filtered;
        for (auto& span : spans) {
            if (span.status == status_to_keep_) {
                filtered.push_back(std::move(span));
            }
        }
        return filtered;
    }

    std::string Name() const override { return "FilteringProcessor"; }
    bool IsHealthy() const override { return true; }
    absl::Status Flush() override { return absl::OkStatus(); }

private:
    std::string status_to_keep_;
};

/// @brief Test processor that adds metadata to spans
class EnrichmentProcessor : public MessageProcessor {
public:
    absl::StatusOr<std::vector<Span>> Process(std::vector<Span>&& spans) override {
        for (auto& span : spans) {
            span.attributes["enriched"] = "true";
            span.attributes["processor_name"] = "EnrichmentProcessor";
        }
        return std::move(spans);
    }

    std::string Name() const override { return "EnrichmentProcessor"; }
    bool IsHealthy() const override { return true; }
    absl::Status Flush() override { return absl::OkStatus(); }
};

/// @brief Test processor that sometimes fails
class UnreliableProcessor : public MessageProcessor {
public:
    explicit UnreliableProcessor(double failure_rate = 0.5)
        : failure_rate_(failure_rate) {}

    absl::StatusOr<std::vector<Span>> Process(std::vector<Span>&& spans) override {
        call_count_++;
        if (static_cast<double>(call_count_ % 10) / 10.0 < failure_rate_) {
            return absl::InternalError("Random failure");
        }
        return std::move(spans);
    }

    std::string Name() const override { return "UnreliableProcessor"; }
    bool IsHealthy() const override { return true; }
    absl::Status Flush() override { return absl::OkStatus(); }

private:
    double failure_rate_;
    std::atomic<size_t> call_count_{0};
};

/// @brief Test processor that introduces latency
class SlowProcessor : public MessageProcessor {
public:
    explicit SlowProcessor(std::chrono::milliseconds delay)
        : delay_(delay) {}

    absl::StatusOr<std::vector<Span>> Process(std::vector<Span>&& spans) override {
        std::this_thread::sleep_for(delay_);
        return std::move(spans);
    }

    std::string Name() const override { return "SlowProcessor"; }
    bool IsHealthy() const override { return true; }
    absl::Status Flush() override { return absl::OkStatus(); }

private:
    std::chrono::milliseconds delay_;
};

// =============================================================================
// Helper Functions
// =============================================================================

std::vector<Span> CreateTestSpans(size_t count, const std::string& status = "ok") {
    std::vector<Span> spans;
    spans.reserve(count);

    auto now = std::chrono::system_clock::now();
    for (size_t i = 0; i < count; ++i) {
        Span span;
        span.trace_id = "trace-" + std::to_string(i);
        span.span_id = "span-" + std::to_string(i);
        span.model_id = "test-model";
        span.status = status;
        span.start_time = now;
        span.end_time = now + std::chrono::milliseconds(100);
        span.latency_ms = 100;
        span.input_tokens = 50;
        span.output_tokens = 100;
        spans.push_back(std::move(span));
    }

    return spans;
}

// =============================================================================
// Pipeline Tests
// =============================================================================

class PipelineTest : public ::testing::Test {
protected:
    void SetUp() override {
        PipelineConfig config;
        config.name = "test-pipeline";
        config.max_batch_size = 100;
        config.flush_interval = std::chrono::seconds(1);
        pipeline_ = std::make_unique<Pipeline>(config);
    }

    void TearDown() override {
        if (pipeline_) {
            pipeline_->Stop();
        }
    }

    std::unique_ptr<Pipeline> pipeline_;
};

TEST_F(PipelineTest, StartAndStop) {
    ASSERT_TRUE(pipeline_->Start().ok());
    EXPECT_TRUE(pipeline_->IsRunning());

    pipeline_->Stop();
    EXPECT_FALSE(pipeline_->IsRunning());
}

TEST_F(PipelineTest, AddProcessor) {
    auto processor = std::make_shared<CountingProcessor>();
    EXPECT_TRUE(pipeline_->AddProcessor(processor).ok());
}

TEST_F(PipelineTest, ProcessSingleBatch) {
    auto counter = std::make_shared<CountingProcessor>();
    ASSERT_TRUE(pipeline_->AddProcessor(counter).ok());
    ASSERT_TRUE(pipeline_->Start().ok());

    auto spans = CreateTestSpans(10);
    auto result = pipeline_->Process(std::move(spans));
    ASSERT_TRUE(result.ok());
    EXPECT_EQ(result->size(), 10);
    EXPECT_EQ(counter->GetProcessedCount(), 10);
}

TEST_F(PipelineTest, ProcessMultipleBatches) {
    auto counter = std::make_shared<CountingProcessor>();
    ASSERT_TRUE(pipeline_->AddProcessor(counter).ok());
    ASSERT_TRUE(pipeline_->Start().ok());

    for (int i = 0; i < 5; ++i) {
        auto spans = CreateTestSpans(20);
        auto result = pipeline_->Process(std::move(spans));
        ASSERT_TRUE(result.ok());
    }

    EXPECT_EQ(counter->GetProcessedCount(), 100);
}

TEST_F(PipelineTest, ChainedProcessors) {
    auto enricher = std::make_shared<EnrichmentProcessor>();
    auto counter = std::make_shared<CountingProcessor>();

    ASSERT_TRUE(pipeline_->AddProcessor(enricher).ok());
    ASSERT_TRUE(pipeline_->AddProcessor(counter).ok());
    ASSERT_TRUE(pipeline_->Start().ok());

    auto spans = CreateTestSpans(5);
    auto result = pipeline_->Process(std::move(spans));
    ASSERT_TRUE(result.ok());
    EXPECT_EQ(result->size(), 5);

    // Verify enrichment happened
    for (const auto& span : *result) {
        EXPECT_EQ(span.attributes.at("enriched"), "true");
    }
}

TEST_F(PipelineTest, FilteringReducesOutput) {
    auto filter = std::make_shared<FilteringProcessor>("error");
    auto counter = std::make_shared<CountingProcessor>();

    ASSERT_TRUE(pipeline_->AddProcessor(filter).ok());
    ASSERT_TRUE(pipeline_->AddProcessor(counter).ok());
    ASSERT_TRUE(pipeline_->Start().ok());

    // Create mixed status spans
    std::vector<Span> spans;
    for (int i = 0; i < 10; ++i) {
        auto batch = CreateTestSpans(1, i % 2 == 0 ? "ok" : "error");
        spans.insert(spans.end(), batch.begin(), batch.end());
    }

    auto result = pipeline_->Process(std::move(spans));
    ASSERT_TRUE(result.ok());
    EXPECT_EQ(result->size(), 5);  // Only error spans
    EXPECT_EQ(counter->GetProcessedCount(), 5);
}

TEST_F(PipelineTest, EmptyBatch) {
    auto counter = std::make_shared<CountingProcessor>();
    ASSERT_TRUE(pipeline_->AddProcessor(counter).ok());
    ASSERT_TRUE(pipeline_->Start().ok());

    std::vector<Span> empty;
    auto result = pipeline_->Process(std::move(empty));
    ASSERT_TRUE(result.ok());
    EXPECT_TRUE(result->empty());
    EXPECT_EQ(counter->GetProcessedCount(), 0);
}

TEST_F(PipelineTest, ProcessorFailureHandled) {
    auto unreliable = std::make_shared<UnreliableProcessor>(1.0);  // Always fails
    ASSERT_TRUE(pipeline_->AddProcessor(unreliable).ok());
    ASSERT_TRUE(pipeline_->Start().ok());

    auto spans = CreateTestSpans(5);
    auto result = pipeline_->Process(std::move(spans));
    EXPECT_FALSE(result.ok());
}

TEST_F(PipelineTest, GetMetrics) {
    auto counter = std::make_shared<CountingProcessor>();
    ASSERT_TRUE(pipeline_->AddProcessor(counter).ok());
    ASSERT_TRUE(pipeline_->Start().ok());

    for (int i = 0; i < 3; ++i) {
        auto spans = CreateTestSpans(10);
        pipeline_->Process(std::move(spans));
    }

    auto metrics = pipeline_->GetMetrics();
    EXPECT_GE(metrics.total_spans_processed, 30);
    EXPECT_GE(metrics.total_batches_processed, 3);
}

TEST_F(PipelineTest, FlushProcessors) {
    auto counter = std::make_shared<CountingProcessor>();
    ASSERT_TRUE(pipeline_->AddProcessor(counter).ok());
    ASSERT_TRUE(pipeline_->Start().ok());

    EXPECT_TRUE(pipeline_->Flush().ok());
}

// =============================================================================
// Concurrent Processing Tests
// =============================================================================

class ConcurrentPipelineTest : public ::testing::Test {
protected:
    void SetUp() override {
        PipelineConfig config;
        config.name = "concurrent-pipeline";
        config.max_batch_size = 50;
        config.num_workers = 4;
        pipeline_ = std::make_unique<Pipeline>(config);
    }

    void TearDown() override {
        if (pipeline_) {
            pipeline_->Stop();
        }
    }

    std::unique_ptr<Pipeline> pipeline_;
};

TEST_F(ConcurrentPipelineTest, ParallelProcessing) {
    auto counter = std::make_shared<CountingProcessor>();
    ASSERT_TRUE(pipeline_->AddProcessor(counter).ok());
    ASSERT_TRUE(pipeline_->Start().ok());

    std::vector<std::thread> threads;
    const int num_threads = 4;
    const int spans_per_thread = 100;

    for (int t = 0; t < num_threads; ++t) {
        threads.emplace_back([this, &counter, spans_per_thread]() {
            for (int i = 0; i < 10; ++i) {
                auto spans = CreateTestSpans(spans_per_thread / 10);
                auto result = pipeline_->Process(std::move(spans));
                EXPECT_TRUE(result.ok());
            }
        });
    }

    for (auto& t : threads) {
        t.join();
    }

    EXPECT_EQ(counter->GetProcessedCount(), num_threads * spans_per_thread);
}

TEST_F(ConcurrentPipelineTest, SlowProcessorDoesNotBlock) {
    auto slow = std::make_shared<SlowProcessor>(std::chrono::milliseconds(10));
    auto counter = std::make_shared<CountingProcessor>();

    ASSERT_TRUE(pipeline_->AddProcessor(slow).ok());
    ASSERT_TRUE(pipeline_->AddProcessor(counter).ok());
    ASSERT_TRUE(pipeline_->Start().ok());

    auto start = std::chrono::steady_clock::now();

    // Process multiple batches
    for (int i = 0; i < 5; ++i) {
        auto spans = CreateTestSpans(10);
        auto result = pipeline_->Process(std::move(spans));
        EXPECT_TRUE(result.ok());
    }

    auto elapsed = std::chrono::steady_clock::now() - start;
    auto ms = std::chrono::duration_cast<std::chrono::milliseconds>(elapsed).count();

    // Should complete in reasonable time even with slow processor
    EXPECT_LT(ms, 1000);  // Less than 1 second
    EXPECT_EQ(counter->GetProcessedCount(), 50);
}

// =============================================================================
// Pipeline Configuration Tests
// =============================================================================

TEST(PipelineConfigTest, DefaultValues) {
    PipelineConfig config;
    EXPECT_EQ(config.name, "default");
    EXPECT_GT(config.max_batch_size, 0);
    EXPECT_GT(config.flush_interval.count(), 0);
}

TEST(PipelineConfigTest, CustomValues) {
    PipelineConfig config;
    config.name = "custom-pipeline";
    config.max_batch_size = 500;
    config.flush_interval = std::chrono::seconds(30);
    config.num_workers = 8;

    EXPECT_EQ(config.name, "custom-pipeline");
    EXPECT_EQ(config.max_batch_size, 500);
    EXPECT_EQ(config.num_workers, 8);
}

// =============================================================================
// Pipeline Metrics Tests
// =============================================================================

TEST(PipelineMetricsTest, DefaultValues) {
    PipelineMetrics metrics;
    EXPECT_EQ(metrics.total_spans_processed, 0);
    EXPECT_EQ(metrics.total_batches_processed, 0);
    EXPECT_EQ(metrics.failed_spans, 0);
    EXPECT_DOUBLE_EQ(metrics.avg_batch_latency_ms, 0.0);
}

// =============================================================================
// End-to-End Integration Test
// =============================================================================

TEST(EndToEndPipelineTest, FullPipelineFlow) {
    // Create pipeline with multiple processors
    PipelineConfig config;
    config.name = "e2e-test";
    config.max_batch_size = 100;

    auto pipeline = std::make_unique<Pipeline>(config);

    // Add processing chain
    auto enricher = std::make_shared<EnrichmentProcessor>();
    auto counter = std::make_shared<CountingProcessor>();

    ASSERT_TRUE(pipeline->AddProcessor(enricher).ok());
    ASSERT_TRUE(pipeline->AddProcessor(counter).ok());

    // Start pipeline
    ASSERT_TRUE(pipeline->Start().ok());
    EXPECT_TRUE(pipeline->IsRunning());

    // Process spans
    auto spans = CreateTestSpans(50);
    auto result = pipeline->Process(std::move(spans));
    ASSERT_TRUE(result.ok());
    EXPECT_EQ(result->size(), 50);

    // Verify processing happened
    EXPECT_EQ(counter->GetProcessedCount(), 50);
    for (const auto& span : *result) {
        EXPECT_EQ(span.attributes.at("enriched"), "true");
    }

    // Flush and stop
    ASSERT_TRUE(pipeline->Flush().ok());
    pipeline->Stop();
    EXPECT_FALSE(pipeline->IsRunning());

    // Verify metrics
    auto metrics = pipeline->GetMetrics();
    EXPECT_EQ(metrics.total_spans_processed, 50);
    EXPECT_GE(metrics.total_batches_processed, 1);
}

}  // namespace
}  // namespace pyflare::processor
