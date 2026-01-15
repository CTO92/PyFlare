#pragma once

/// @file batcher.h
/// @brief Batching logic for aggregating spans before export

#include <atomic>
#include <chrono>
#include <condition_variable>
#include <functional>
#include <mutex>
#include <queue>
#include <thread>
#include <vector>

#include <absl/status/status.h>

#include "types.h"

namespace pyflare::collector {

/// Batcher configuration
struct BatcherConfig {
    /// Maximum number of spans per batch
    size_t max_batch_size = 1000;

    /// Maximum time to wait before flushing a batch
    std::chrono::milliseconds max_batch_timeout{100};

    /// Maximum number of spans queued (backpressure)
    size_t max_queue_size = 100000;

    /// Number of concurrent batch processors
    size_t num_workers = 4;
};

/// Statistics for the batcher
struct BatcherStats {
    std::atomic<uint64_t> spans_received{0};
    std::atomic<uint64_t> spans_batched{0};
    std::atomic<uint64_t> spans_dropped{0};
    std::atomic<uint64_t> batches_created{0};
    std::atomic<uint64_t> batches_flushed{0};
    std::atomic<uint64_t> queue_high_watermark{0};
};

/// Callback for completed batches
using BatchCallback = std::function<void(std::vector<Span>&&)>;

/// Batcher that aggregates spans into batches for efficient export
class Batcher {
public:
    /// Create batcher with configuration
    explicit Batcher(BatcherConfig config);

    /// Destructor
    ~Batcher();

    // Non-copyable, non-movable
    Batcher(const Batcher&) = delete;
    Batcher& operator=(const Batcher&) = delete;

    /// Register callback for completed batches
    void OnBatch(BatchCallback callback);

    /// Add spans to be batched
    /// @param spans Spans to add
    /// @return Status (may fail if queue is full)
    absl::Status Add(std::vector<Span>&& spans);

    /// Add a single span
    /// @param span Span to add
    /// @return Status
    absl::Status Add(Span&& span);

    /// Start the batcher (begins timer thread)
    absl::Status Start();

    /// Stop the batcher and flush pending spans
    absl::Status Shutdown();

    /// Flush current batch immediately
    void Flush();

    /// Check if batcher is running
    bool IsRunning() const { return running_.load(); }

    /// Get number of pending spans
    size_t PendingCount() const;

    /// Get statistics
    const BatcherStats& GetStats() const { return stats_; }

    /// Get configuration
    const BatcherConfig& GetConfig() const { return config_; }

private:
    /// Internal batch structure
    struct Batch {
        std::vector<Span> spans;
        std::chrono::steady_clock::time_point created_at;

        Batch() : created_at(std::chrono::steady_clock::now()) {}

        void Reset() {
            spans.clear();
            created_at = std::chrono::steady_clock::now();
        }
    };

    /// Timer thread function
    void TimerLoop();

    /// Worker thread function
    void WorkerLoop();

    /// Check if current batch should be flushed
    bool ShouldFlush() const;

    /// Flush the current batch to the callback
    void FlushBatch();

    /// Move spans from queue to current batch
    void ProcessQueue();

    BatcherConfig config_;
    BatchCallback callback_;
    BatcherStats stats_;

    // Current batch being built
    Batch current_batch_;
    mutable std::mutex batch_mutex_;

    // Input queue for incoming spans
    std::queue<Span> input_queue_;
    mutable std::mutex queue_mutex_;
    std::condition_variable queue_cv_;

    // Threads
    std::thread timer_thread_;
    std::vector<std::thread> worker_threads_;
    std::atomic<bool> running_{false};
    std::atomic<bool> shutdown_requested_{false};
};

}  // namespace pyflare::collector
