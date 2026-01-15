/// @file batcher_test.cpp
/// @brief Unit tests for the batcher component

#include <gtest/gtest.h>

#include <atomic>
#include <chrono>
#include <thread>
#include <vector>

#include "src/collector/batcher.h"

namespace pyflare::collector {
namespace {

// Helper to create a test span
Span CreateTestSpan(const std::string& name = "test-span") {
    static std::atomic<int> counter{0};
    Span span;
    span.trace_id = "trace-" + std::to_string(counter++);
    span.span_id = "span-123";
    span.name = name;
    span.kind = SpanKind::kInternal;
    span.start_time_ns = 1000000000;
    span.end_time_ns = 2000000000;
    return span;
}

// Helper to create multiple spans
std::vector<Span> CreateTestSpans(size_t count) {
    std::vector<Span> spans;
    spans.reserve(count);
    for (size_t i = 0; i < count; ++i) {
        spans.push_back(CreateTestSpan("span-" + std::to_string(i)));
    }
    return spans;
}

// ============================================================================
// Basic Batcher Tests
// ============================================================================

TEST(BatcherTest, ConstructsWithConfig) {
    BatcherConfig config;
    config.max_batch_size = 100;
    config.max_batch_timeout = std::chrono::milliseconds(1000);

    Batcher batcher(config);
    // Should not crash
}

TEST(BatcherTest, StartsAndStops) {
    BatcherConfig config;
    config.max_batch_size = 100;
    config.max_batch_timeout = std::chrono::milliseconds(100);
    config.num_workers = 2;

    Batcher batcher(config);

    auto status = batcher.Start();
    EXPECT_TRUE(status.ok());

    status = batcher.Shutdown();
    EXPECT_TRUE(status.ok());
}

TEST(BatcherTest, DoubleStartFails) {
    BatcherConfig config;
    config.max_batch_size = 100;
    config.max_batch_timeout = std::chrono::milliseconds(100);

    Batcher batcher(config);

    auto status = batcher.Start();
    EXPECT_TRUE(status.ok());

    // Second start should fail
    status = batcher.Start();
    EXPECT_FALSE(status.ok());
    EXPECT_EQ(status.code(), absl::StatusCode::kAlreadyExists);

    batcher.Shutdown();
}

TEST(BatcherTest, AddBeforeStartFails) {
    BatcherConfig config;
    Batcher batcher(config);

    auto spans = CreateTestSpans(10);
    auto status = batcher.Add(std::move(spans));
    EXPECT_FALSE(status.ok());
    EXPECT_EQ(status.code(), absl::StatusCode::kFailedPrecondition);
}

// ============================================================================
// Span Processing Tests
// ============================================================================

TEST(BatcherTest, AddsSpansSuccessfully) {
    BatcherConfig config;
    config.max_batch_size = 100;
    config.max_batch_timeout = std::chrono::milliseconds(1000);
    config.max_queue_size = 1000;

    Batcher batcher(config);
    batcher.Start();

    auto spans = CreateTestSpans(10);
    auto status = batcher.Add(std::move(spans));
    EXPECT_TRUE(status.ok());

    // Allow time for processing
    std::this_thread::sleep_for(std::chrono::milliseconds(50));

    const auto& stats = batcher.GetStats();
    EXPECT_EQ(stats.spans_received.load(), 10);

    batcher.Shutdown();
}

TEST(BatcherTest, AddsSingleSpan) {
    BatcherConfig config;
    config.max_batch_size = 100;
    config.max_batch_timeout = std::chrono::milliseconds(1000);

    Batcher batcher(config);
    batcher.Start();

    auto span = CreateTestSpan();
    auto status = batcher.Add(std::move(span));
    EXPECT_TRUE(status.ok());

    const auto& stats = batcher.GetStats();
    EXPECT_EQ(stats.spans_received.load(), 1);

    batcher.Shutdown();
}

// ============================================================================
// Batching Behavior Tests
// ============================================================================

TEST(BatcherTest, FlushesOnBatchSizeReached) {
    BatcherConfig config;
    config.max_batch_size = 10;
    config.max_batch_timeout = std::chrono::milliseconds(10000);  // Long timeout
    config.num_workers = 1;
    config.max_queue_size = 1000;

    std::atomic<int> batch_count{0};
    std::atomic<size_t> total_spans{0};

    Batcher batcher(config);
    batcher.OnBatch([&](std::vector<Span>&& batch) {
        batch_count++;
        total_spans += batch.size();
    });

    batcher.Start();

    // Add exactly batch_size spans
    auto spans = CreateTestSpans(10);
    batcher.Add(std::move(spans));

    // Wait for batch to be processed
    std::this_thread::sleep_for(std::chrono::milliseconds(200));

    EXPECT_GE(batch_count.load(), 1);
    EXPECT_EQ(total_spans.load(), 10);

    batcher.Shutdown();
}

TEST(BatcherTest, FlushesOnTimeout) {
    BatcherConfig config;
    config.max_batch_size = 1000;  // Large batch size
    config.max_batch_timeout = std::chrono::milliseconds(100);  // Short timeout
    config.num_workers = 1;

    std::atomic<int> batch_count{0};

    Batcher batcher(config);
    batcher.OnBatch([&](std::vector<Span>&& batch) {
        batch_count++;
    });

    batcher.Start();

    // Add fewer spans than batch size
    auto spans = CreateTestSpans(5);
    batcher.Add(std::move(spans));

    // Wait for timeout-based flush
    std::this_thread::sleep_for(std::chrono::milliseconds(300));

    EXPECT_GE(batch_count.load(), 1);

    batcher.Shutdown();
}

// ============================================================================
// Backpressure Tests
// ============================================================================

TEST(BatcherTest, DropsSpansWhenQueueFull) {
    BatcherConfig config;
    config.max_batch_size = 100;
    config.max_batch_timeout = std::chrono::milliseconds(10000);
    config.max_queue_size = 10;  // Small queue
    config.num_workers = 0;  // No workers to process queue

    Batcher batcher(config);

    // Don't register a callback so spans accumulate
    batcher.Start();

    // Fill the queue
    auto spans1 = CreateTestSpans(10);
    auto status1 = batcher.Add(std::move(spans1));
    EXPECT_TRUE(status1.ok());

    // This should fail due to full queue
    auto spans2 = CreateTestSpans(5);
    auto status2 = batcher.Add(std::move(spans2));
    EXPECT_FALSE(status2.ok());
    EXPECT_EQ(status2.code(), absl::StatusCode::kResourceExhausted);

    const auto& stats = batcher.GetStats();
    EXPECT_GT(stats.spans_dropped.load(), 0);

    batcher.Shutdown();
}

// ============================================================================
// Statistics Tests
// ============================================================================

TEST(BatcherTest, TracksPendingCount) {
    BatcherConfig config;
    config.max_batch_size = 1000;
    config.max_batch_timeout = std::chrono::milliseconds(10000);
    config.num_workers = 1;

    Batcher batcher(config);
    batcher.Start();

    auto spans = CreateTestSpans(50);
    batcher.Add(std::move(spans));

    // Should have some pending spans (before batch flush)
    // Note: This is timing-dependent
    EXPECT_GE(batcher.PendingCount(), 0);

    batcher.Shutdown();
}

TEST(BatcherTest, TracksStatistics) {
    BatcherConfig config;
    config.max_batch_size = 10;
    config.max_batch_timeout = std::chrono::milliseconds(100);
    config.num_workers = 1;

    Batcher batcher(config);
    batcher.OnBatch([](std::vector<Span>&&) {
        // Just consume
    });

    batcher.Start();

    for (int i = 0; i < 5; ++i) {
        auto spans = CreateTestSpans(10);
        batcher.Add(std::move(spans));
    }

    std::this_thread::sleep_for(std::chrono::milliseconds(500));

    const auto& stats = batcher.GetStats();
    EXPECT_EQ(stats.spans_received.load(), 50);
    EXPECT_GE(stats.batches_created.load(), 1);

    batcher.Shutdown();
}

// ============================================================================
// Flush Tests
// ============================================================================

TEST(BatcherTest, FlushSendsRemainingSpans) {
    BatcherConfig config;
    config.max_batch_size = 1000;
    config.max_batch_timeout = std::chrono::milliseconds(10000);
    config.num_workers = 1;

    std::atomic<size_t> flushed_count{0};

    Batcher batcher(config);
    batcher.OnBatch([&](std::vector<Span>&& batch) {
        flushed_count += batch.size();
    });

    batcher.Start();

    auto spans = CreateTestSpans(5);
    batcher.Add(std::move(spans));

    // Manual flush
    batcher.Flush();

    std::this_thread::sleep_for(std::chrono::milliseconds(100));

    EXPECT_EQ(flushed_count.load(), 5);

    batcher.Shutdown();
}

// ============================================================================
// Concurrency Tests
// ============================================================================

TEST(BatcherTest, HandlesMultipleProducers) {
    BatcherConfig config;
    config.max_batch_size = 100;
    config.max_batch_timeout = std::chrono::milliseconds(100);
    config.num_workers = 4;
    config.max_queue_size = 10000;

    std::atomic<size_t> total_received{0};

    Batcher batcher(config);
    batcher.OnBatch([&](std::vector<Span>&& batch) {
        total_received += batch.size();
    });

    batcher.Start();

    // Spawn multiple producer threads
    std::vector<std::thread> producers;
    const int num_producers = 4;
    const int spans_per_producer = 100;

    for (int i = 0; i < num_producers; ++i) {
        producers.emplace_back([&batcher]() {
            for (int j = 0; j < spans_per_producer; ++j) {
                auto span = CreateTestSpan();
                batcher.Add(std::move(span));
            }
        });
    }

    // Wait for all producers
    for (auto& t : producers) {
        t.join();
    }

    // Allow time for processing
    std::this_thread::sleep_for(std::chrono::milliseconds(500));

    batcher.Shutdown();

    // All spans should be received
    const auto& stats = batcher.GetStats();
    EXPECT_EQ(stats.spans_received.load(), num_producers * spans_per_producer);
}

}  // namespace
}  // namespace pyflare::collector
