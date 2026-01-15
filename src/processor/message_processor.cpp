/// @file message_processor.cpp
/// @brief Implementation of base message processor and pipeline

#include "processor/message_processor.h"

#include <random>

#include <spdlog/spdlog.h>

namespace pyflare::processor {

// =============================================================================
// BaseProcessor Implementation
// =============================================================================

BaseProcessor::BaseProcessor(ProcessorConfig config)
    : config_(std::move(config)) {}

bool BaseProcessor::IsHealthy() const {
    return healthy_.load();
}

bool BaseProcessor::IsReady() const {
    return ready_.load() && !shutdown_.load();
}

absl::Status BaseProcessor::Flush() {
    // Default implementation does nothing
    return absl::OkStatus();
}

ProcessorStats BaseProcessor::GetStats() const {
    std::lock_guard<std::mutex> lock(stats_mutex_);
    return stats_;
}

void BaseProcessor::ResetStats() {
    std::lock_guard<std::mutex> lock(stats_mutex_);
    stats_ = ProcessorStats{};
}

absl::Status BaseProcessor::Initialize() {
    if (initialized_.load()) {
        return absl::AlreadyExistsError("Processor already initialized");
    }

    initialized_.store(true);
    ready_.store(true);
    spdlog::info("Processor {} initialized", Name());
    return absl::OkStatus();
}

absl::Status BaseProcessor::Shutdown() {
    if (shutdown_.load()) {
        return absl::OkStatus();
    }

    shutdown_.store(true);
    ready_.store(false);

    auto status = Flush();
    if (!status.ok()) {
        spdlog::warn("Processor {} flush failed during shutdown: {}",
                     Name(), status.message());
    }

    spdlog::info("Processor {} shutdown complete", Name());
    return absl::OkStatus();
}

void BaseProcessor::UpdateStats(size_t batch_size, double processing_time_ms,
                                 size_t failures) {
    std::lock_guard<std::mutex> lock(stats_mutex_);
    stats_.records_processed += batch_size;
    stats_.records_failed += failures;
    stats_.batches_processed++;
    stats_.last_processed = std::chrono::system_clock::now();

    // Update average processing time (exponential moving average)
    const double alpha = 0.1;
    stats_.avg_processing_time_ms =
        alpha * processing_time_ms + (1.0 - alpha) * stats_.avg_processing_time_ms;

    // Update p99 (simplified approximation)
    if (processing_time_ms > stats_.p99_processing_time_ms) {
        stats_.p99_processing_time_ms =
            0.99 * processing_time_ms + 0.01 * stats_.p99_processing_time_ms;
    } else {
        stats_.p99_processing_time_ms =
            0.01 * processing_time_ms + 0.99 * stats_.p99_processing_time_ms;
    }
}

bool BaseProcessor::ShouldProcess() const {
    if (config_.sample_rate >= 1.0) {
        return true;
    }
    if (config_.sample_rate <= 0.0) {
        return false;
    }

    // Thread-local random generator for efficiency
    thread_local std::mt19937 gen(std::random_device{}());
    thread_local std::uniform_real_distribution<double> dist(0.0, 1.0);

    return dist(gen) < config_.sample_rate;
}

// =============================================================================
// ProcessorPipeline Implementation
// =============================================================================

ProcessorPipeline::ProcessorPipeline(Config config)
    : config_(std::move(config)) {}

ProcessorPipeline::~ProcessorPipeline() {
    if (running_.load()) {
        auto status = Stop();
        if (!status.ok()) {
            spdlog::error("Pipeline stop failed in destructor: {}",
                          status.message());
        }
    }
}

void ProcessorPipeline::AddProcessor(std::unique_ptr<MessageProcessor> processor) {
    if (running_.load()) {
        spdlog::warn("Cannot add processor while pipeline is running");
        return;
    }

    std::lock_guard<std::mutex> lock(pipeline_mutex_);
    spdlog::info("Adding processor: {}", processor->Name());
    processors_.push_back(std::move(processor));
}

absl::StatusOr<PipelineResult> ProcessorPipeline::Process(
    std::vector<collector::Span>&& spans) {
    if (!running_.load()) {
        return absl::FailedPreconditionError("Pipeline is not running");
    }

    PipelineResult result;
    result.spans = std::move(spans);
    result.total_processed = result.spans.size();

    auto pipeline_start = std::chrono::steady_clock::now();

    std::lock_guard<std::mutex> lock(pipeline_mutex_);
    for (auto& processor : processors_) {
        if (result.spans.empty()) {
            break;
        }

        auto proc_start = std::chrono::steady_clock::now();
        auto processed = processor->Process(std::move(result.spans));
        auto proc_end = std::chrono::steady_clock::now();

        double proc_time_ms =
            std::chrono::duration<double, std::milli>(proc_end - proc_start).count();

        if (!processed.ok()) {
            std::string error_msg = std::string(processor->Name()) + ": " +
                                    std::string(processed.status().message());
            result.processor_errors.push_back(error_msg);
            spdlog::error("Processor {} failed: {}", processor->Name(),
                          processed.status().message());

            if (config_.stop_on_error) {
                return processed.status();
            }

            // On error with continue_on_error, keep the previous spans
            result.total_failed += result.spans.size();
            continue;
        }

        result.spans = std::move(*processed);

        spdlog::debug("Processor {} processed {} spans in {:.2f}ms",
                      processor->Name(), result.spans.size(), proc_time_ms);
    }

    auto pipeline_end = std::chrono::steady_clock::now();
    result.total_processing_time_ms =
        std::chrono::duration<double, std::milli>(pipeline_end - pipeline_start)
            .count();

    return result;
}

absl::Status ProcessorPipeline::Start() {
    if (running_.load()) {
        return absl::AlreadyExistsError("Pipeline already running");
    }

    std::lock_guard<std::mutex> lock(pipeline_mutex_);

    spdlog::info("Starting pipeline with {} processors", processors_.size());

    for (auto& processor : processors_) {
        auto status = processor->Initialize();
        if (!status.ok()) {
            spdlog::error("Failed to initialize processor {}: {}",
                          processor->Name(), status.message());
            // Shutdown already initialized processors
            for (auto& p : processors_) {
                if (p.get() == processor.get()) break;
                p->Shutdown();
            }
            return status;
        }
    }

    running_.store(true);
    spdlog::info("Pipeline started successfully");
    return absl::OkStatus();
}

absl::Status ProcessorPipeline::Stop() {
    if (!running_.load()) {
        return absl::OkStatus();
    }

    spdlog::info("Stopping pipeline...");
    running_.store(false);

    std::lock_guard<std::mutex> lock(pipeline_mutex_);

    absl::Status overall_status = absl::OkStatus();

    // Shutdown in reverse order
    for (auto it = processors_.rbegin(); it != processors_.rend(); ++it) {
        auto status = (*it)->Shutdown();
        if (!status.ok()) {
            spdlog::error("Failed to shutdown processor {}: {}",
                          (*it)->Name(), status.message());
            if (overall_status.ok()) {
                overall_status = status;
            }
        }
    }

    spdlog::info("Pipeline stopped");
    return overall_status;
}

bool ProcessorPipeline::IsHealthy() const {
    if (!running_.load()) {
        return false;
    }

    std::lock_guard<std::mutex> lock(pipeline_mutex_);
    for (const auto& processor : processors_) {
        if (!processor->IsHealthy()) {
            return false;
        }
    }
    return true;
}

bool ProcessorPipeline::IsReady() const {
    if (!running_.load()) {
        return false;
    }

    std::lock_guard<std::mutex> lock(pipeline_mutex_);
    for (const auto& processor : processors_) {
        if (!processor->IsReady()) {
            return false;
        }
    }
    return true;
}

size_t ProcessorPipeline::ProcessorCount() const {
    std::lock_guard<std::mutex> lock(pipeline_mutex_);
    return processors_.size();
}

MessageProcessor* ProcessorPipeline::GetProcessor(size_t index) {
    std::lock_guard<std::mutex> lock(pipeline_mutex_);
    if (index >= processors_.size()) {
        return nullptr;
    }
    return processors_[index].get();
}

std::vector<std::pair<std::string, ProcessorStats>>
ProcessorPipeline::GetAllStats() const {
    std::lock_guard<std::mutex> lock(pipeline_mutex_);

    std::vector<std::pair<std::string, ProcessorStats>> all_stats;
    all_stats.reserve(processors_.size());

    for (const auto& processor : processors_) {
        all_stats.emplace_back(processor->Name(), processor->GetStats());
    }

    return all_stats;
}

}  // namespace pyflare::processor
