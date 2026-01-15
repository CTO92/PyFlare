/// @file batcher.cpp
/// @brief Batching logic for aggregating spans before export

#include "batcher.h"

#include <algorithm>

#include "src/common/logging.h"

namespace pyflare::collector {

Batcher::Batcher(BatcherConfig config)
    : config_(std::move(config)) {}

Batcher::~Batcher() {
    if (running_.load()) {
        Shutdown();
    }
}

void Batcher::OnBatch(BatchCallback callback) {
    callback_ = std::move(callback);
}

absl::Status Batcher::Add(std::vector<Span>&& spans) {
    if (!running_.load()) {
        return absl::FailedPreconditionError("Batcher not running");
    }

    stats_.spans_received += spans.size();

    {
        std::lock_guard<std::mutex> lock(queue_mutex_);

        // Check queue size for backpressure
        size_t current_size = input_queue_.size();
        if (current_size >= config_.max_queue_size) {
            stats_.spans_dropped += spans.size();
            return absl::ResourceExhaustedError("Queue full, dropping spans");
        }

        // Update high watermark
        size_t new_size = current_size + spans.size();
        size_t current_watermark = stats_.queue_high_watermark.load();
        if (new_size > current_watermark) {
            stats_.queue_high_watermark = new_size;
        }

        // Add spans to queue
        for (auto& span : spans) {
            input_queue_.push(std::move(span));
        }
    }

    // Notify workers
    queue_cv_.notify_one();

    return absl::OkStatus();
}

absl::Status Batcher::Add(Span&& span) {
    std::vector<Span> spans;
    spans.push_back(std::move(span));
    return Add(std::move(spans));
}

absl::Status Batcher::Start() {
    if (running_.exchange(true)) {
        return absl::AlreadyExistsError("Batcher already running");
    }

    shutdown_requested_ = false;
    current_batch_.Reset();

    // Start timer thread
    timer_thread_ = std::thread(&Batcher::TimerLoop, this);

    // Start worker threads
    for (size_t i = 0; i < config_.num_workers; ++i) {
        worker_threads_.emplace_back(&Batcher::WorkerLoop, this);
    }

    PYFLARE_LOG_INFO("Batcher started with {} workers, batch_size={}, timeout={}ms",
                     config_.num_workers, config_.max_batch_size,
                     config_.max_batch_timeout.count());

    return absl::OkStatus();
}

absl::Status Batcher::Shutdown() {
    if (!running_.exchange(false)) {
        return absl::FailedPreconditionError("Batcher not running");
    }

    PYFLARE_LOG_INFO("Shutting down batcher...");

    shutdown_requested_ = true;

    // Wake up all waiting threads
    queue_cv_.notify_all();

    // Wait for timer thread
    if (timer_thread_.joinable()) {
        timer_thread_.join();
    }

    // Wait for worker threads
    for (auto& worker : worker_threads_) {
        if (worker.joinable()) {
            worker.join();
        }
    }
    worker_threads_.clear();

    // Flush any remaining spans
    Flush();

    PYFLARE_LOG_INFO("Batcher shutdown complete. Stats: received={}, batched={}, dropped={}",
                     stats_.spans_received.load(),
                     stats_.spans_batched.load(),
                     stats_.spans_dropped.load());

    return absl::OkStatus();
}

void Batcher::Flush() {
    std::lock_guard<std::mutex> lock(batch_mutex_);
    FlushBatch();
}

size_t Batcher::PendingCount() const {
    std::lock_guard<std::mutex> queue_lock(queue_mutex_);
    std::lock_guard<std::mutex> batch_lock(batch_mutex_);
    return input_queue_.size() + current_batch_.spans.size();
}

void Batcher::TimerLoop() {
    while (!shutdown_requested_.load()) {
        // Sleep for a fraction of the timeout to check regularly
        std::this_thread::sleep_for(config_.max_batch_timeout / 10);

        if (shutdown_requested_.load()) {
            break;
        }

        // Check if batch should be flushed due to timeout
        std::lock_guard<std::mutex> lock(batch_mutex_);
        if (ShouldFlush()) {
            FlushBatch();
        }
    }
}

void Batcher::WorkerLoop() {
    while (!shutdown_requested_.load()) {
        // Wait for spans in queue
        std::unique_lock<std::mutex> lock(queue_mutex_);
        queue_cv_.wait_for(lock, std::chrono::milliseconds(100), [this] {
            return !input_queue_.empty() || shutdown_requested_.load();
        });

        if (shutdown_requested_.load() && input_queue_.empty()) {
            break;
        }

        // Process spans from queue
        ProcessQueue();
    }
}

bool Batcher::ShouldFlush() const {
    // Already locked by caller
    if (current_batch_.spans.empty()) {
        return false;
    }

    // Check batch size
    if (current_batch_.spans.size() >= config_.max_batch_size) {
        return true;
    }

    // Check timeout
    auto elapsed = std::chrono::steady_clock::now() - current_batch_.created_at;
    return elapsed >= config_.max_batch_timeout;
}

void Batcher::FlushBatch() {
    // Already locked by caller
    if (current_batch_.spans.empty()) {
        return;
    }

    stats_.spans_batched += current_batch_.spans.size();
    stats_.batches_created++;

    if (callback_) {
        // Move batch to callback
        std::vector<Span> batch_to_send;
        batch_to_send.swap(current_batch_.spans);

        stats_.batches_flushed++;

        // Call callback outside lock would be better, but we're already holding it
        callback_(std::move(batch_to_send));
    }

    current_batch_.Reset();
}

void Batcher::ProcessQueue() {
    // Called with queue_mutex_ held
    if (input_queue_.empty()) {
        return;
    }

    // Move spans from queue to current batch
    std::vector<Span> spans_to_process;

    // Take spans from queue (up to max batch size)
    size_t to_take = std::min(input_queue_.size(),
                               config_.max_batch_size - current_batch_.spans.size());

    // Limit to prevent holding lock too long
    to_take = std::min(to_take, static_cast<size_t>(1000));

    for (size_t i = 0; i < to_take && !input_queue_.empty(); ++i) {
        spans_to_process.push_back(std::move(input_queue_.front()));
        input_queue_.pop();
    }

    // Release queue lock before acquiring batch lock
    queue_mutex_.unlock();

    // Add to current batch
    {
        std::lock_guard<std::mutex> batch_lock(batch_mutex_);

        for (auto& span : spans_to_process) {
            current_batch_.spans.push_back(std::move(span));
        }

        // Check if batch should be flushed
        if (ShouldFlush()) {
            FlushBatch();
        }
    }

    // Re-acquire queue lock (caller expects it held)
    queue_mutex_.lock();
}

}  // namespace pyflare::collector
