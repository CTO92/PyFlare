#pragma once

/// @file message_processor.h
/// @brief Base interface for stream processors in PyFlare Phase 2

#include <chrono>
#include <memory>
#include <string>
#include <vector>

#include <absl/status/status.h>
#include <absl/status/statusor.h>

#include "collector/types.h"

namespace pyflare::processor {

/// @brief Statistics for a processor
struct ProcessorStats {
    uint64_t records_processed = 0;
    uint64_t records_failed = 0;
    uint64_t batches_processed = 0;
    double avg_processing_time_ms = 0.0;
    double p99_processing_time_ms = 0.0;
    std::chrono::system_clock::time_point last_processed;
};

/// @brief Configuration for processor behavior
struct ProcessorConfig {
    /// Whether the processor is enabled
    bool enabled = true;

    /// Maximum batch size to process at once
    size_t max_batch_size = 1000;

    /// Timeout for processing a single batch (milliseconds)
    int64_t processing_timeout_ms = 30000;

    /// Whether to continue on individual record failures
    bool continue_on_error = true;

    /// Sample rate for this processor (0.0 - 1.0)
    double sample_rate = 1.0;
};

/// @brief Abstract base class for message processors
///
/// MessageProcessor defines the interface for all stream processing components
/// in PyFlare. Processors form a pipeline where each stage receives spans,
/// processes them (enriching, filtering, or analyzing), and passes them on.
///
/// Example usage:
/// @code
///   auto drift_detector = std::make_unique<DriftProcessor>(config);
///   auto cost_tracker = std::make_unique<CostProcessor>(config);
///   pipeline.AddProcessor(std::move(drift_detector));
///   pipeline.AddProcessor(std::move(cost_tracker));
///   pipeline.Start();
/// @endcode
class MessageProcessor {
public:
    virtual ~MessageProcessor() = default;

    // Disable copy and move
    MessageProcessor(const MessageProcessor&) = delete;
    MessageProcessor& operator=(const MessageProcessor&) = delete;
    MessageProcessor(MessageProcessor&&) = delete;
    MessageProcessor& operator=(MessageProcessor&&) = delete;

    /// @brief Process a batch of spans
    /// @param spans Input spans from the previous stage (moved in)
    /// @return Processed spans (may be modified, filtered, or unchanged)
    /// @note Implementations may modify spans in-place, filter them, or
    ///       generate side effects (e.g., alerts, metrics) without modification
    virtual absl::StatusOr<std::vector<collector::Span>> Process(
        std::vector<collector::Span>&& spans) = 0;

    /// @brief Get the processor name for logging and metrics
    /// @return Human-readable processor name
    virtual std::string Name() const = 0;

    /// @brief Check if the processor is healthy and ready to process
    /// @return true if healthy, false otherwise
    virtual bool IsHealthy() const = 0;

    /// @brief Check if the processor is ready to accept requests
    /// @return true if ready, false if still initializing
    virtual bool IsReady() const = 0;

    /// @brief Flush any buffered data
    /// @return Status indicating success or failure
    /// @note Called during graceful shutdown or when immediate output is needed
    virtual absl::Status Flush() = 0;

    /// @brief Get current processor statistics
    /// @return Statistics structure with processing metrics
    virtual ProcessorStats GetStats() const = 0;

    /// @brief Reset statistics counters
    virtual void ResetStats() = 0;

    /// @brief Initialize the processor
    /// @return Status indicating success or failure
    /// @note Called once before any Process() calls
    virtual absl::Status Initialize() = 0;

    /// @brief Shutdown the processor
    /// @return Status indicating success or failure
    /// @note Called once after all Process() calls complete
    virtual absl::Status Shutdown() = 0;

protected:
    MessageProcessor() = default;
};

/// @brief Base implementation with common functionality
///
/// Provides default implementations for statistics tracking, health checks,
/// and lifecycle management. Derived classes should override Process() and Name().
class BaseProcessor : public MessageProcessor {
public:
    explicit BaseProcessor(ProcessorConfig config = {});
    ~BaseProcessor() override = default;

    bool IsHealthy() const override;
    bool IsReady() const override;
    absl::Status Flush() override;
    ProcessorStats GetStats() const override;
    void ResetStats() override;
    absl::Status Initialize() override;
    absl::Status Shutdown() override;

protected:
    /// @brief Update statistics after processing a batch
    /// @param batch_size Number of records processed
    /// @param processing_time_ms Time taken to process the batch
    /// @param failures Number of failed records
    void UpdateStats(size_t batch_size, double processing_time_ms, size_t failures = 0);

    /// @brief Check if we should process this batch based on sample rate
    /// @return true if batch should be processed
    bool ShouldProcess() const;

    ProcessorConfig config_;
    ProcessorStats stats_;
    mutable std::mutex stats_mutex_;

    std::atomic<bool> initialized_{false};
    std::atomic<bool> healthy_{true};
    std::atomic<bool> ready_{false};
    std::atomic<bool> shutdown_{false};
};

/// @brief Result of pipeline processing
struct PipelineResult {
    std::vector<collector::Span> spans;
    size_t total_processed = 0;
    size_t total_failed = 0;
    std::vector<std::string> processor_errors;
    double total_processing_time_ms = 0.0;
};

/// @brief Pipeline of processors that execute in sequence
///
/// Orchestrates multiple MessageProcessors in a chain. Each processor
/// receives the output of the previous one. The pipeline handles:
/// - Sequential execution of processors
/// - Error propagation and recovery
/// - Statistics aggregation
/// - Graceful startup and shutdown
class ProcessorPipeline {
public:
    struct Config {
        /// Stop pipeline on first processor error
        bool stop_on_error = false;

        /// Maximum time to wait for pipeline shutdown (ms)
        int64_t shutdown_timeout_ms = 30000;
    };

    explicit ProcessorPipeline(Config config = {});
    ~ProcessorPipeline();

    // Disable copy and move
    ProcessorPipeline(const ProcessorPipeline&) = delete;
    ProcessorPipeline& operator=(const ProcessorPipeline&) = delete;
    ProcessorPipeline(ProcessorPipeline&&) = delete;
    ProcessorPipeline& operator=(ProcessorPipeline&&) = delete;

    /// @brief Add a processor to the end of the pipeline
    /// @param processor Processor to add (ownership transferred)
    void AddProcessor(std::unique_ptr<MessageProcessor> processor);

    /// @brief Process spans through all processors in sequence
    /// @param spans Input spans
    /// @return Pipeline result with processed spans and statistics
    absl::StatusOr<PipelineResult> Process(std::vector<collector::Span>&& spans);

    /// @brief Initialize all processors in the pipeline
    /// @return Status indicating success or failure
    absl::Status Start();

    /// @brief Shutdown all processors (flushes and stops)
    /// @return Status indicating success or failure
    absl::Status Stop();

    /// @brief Check if all processors are healthy
    /// @return true if all healthy, false otherwise
    bool IsHealthy() const;

    /// @brief Check if pipeline is ready to process
    /// @return true if ready, false otherwise
    bool IsReady() const;

    /// @brief Get the number of processors in the pipeline
    size_t ProcessorCount() const;

    /// @brief Get processor by index
    /// @param index Processor index
    /// @return Pointer to processor or nullptr if index out of bounds
    MessageProcessor* GetProcessor(size_t index);

    /// @brief Get aggregated statistics from all processors
    std::vector<std::pair<std::string, ProcessorStats>> GetAllStats() const;

private:
    Config config_;
    std::vector<std::unique_ptr<MessageProcessor>> processors_;
    mutable std::mutex pipeline_mutex_;
    std::atomic<bool> running_{false};
};

}  // namespace pyflare::processor
