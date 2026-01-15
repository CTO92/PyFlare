/// @file sampler.cpp
/// @brief Sampling strategy implementations

#include "sampler.h"

#include <algorithm>
#include <cmath>
#include <limits>

#include "src/common/logging.h"

namespace pyflare::collector {

namespace {

// Maximum value for 64-bit unsigned integer
constexpr uint64_t kMaxUint64 = std::numeric_limits<uint64_t>::max();

// FNV-1a hash constants
constexpr uint64_t kFnvPrime = 0x100000001b3;
constexpr uint64_t kFnvOffset = 0xcbf29ce484222325;

}  // namespace

// ============================================================================
// ProbabilisticSampler
// ============================================================================

ProbabilisticSampler::ProbabilisticSampler(double probability)
    : probability_(std::clamp(probability, 0.0, 1.0)) {
    UpdateThreshold();
}

void ProbabilisticSampler::SetProbability(double probability) {
    probability_ = std::clamp(probability, 0.0, 1.0);
    UpdateThreshold();
}

void ProbabilisticSampler::UpdateThreshold() {
    double prob = probability_.load();
    if (prob >= 1.0) {
        threshold_ = kMaxUint64;
    } else if (prob <= 0.0) {
        threshold_ = 0;
    } else {
        threshold_ = static_cast<uint64_t>(prob * static_cast<double>(kMaxUint64));
    }
}

uint64_t ProbabilisticSampler::HashTraceId(const std::string& trace_id) {
    // Use FNV-1a hash for consistent hashing
    uint64_t hash = kFnvOffset;
    for (char c : trace_id) {
        hash ^= static_cast<uint64_t>(static_cast<unsigned char>(c));
        hash *= kFnvPrime;
    }
    return hash;
}

SamplingDecision ProbabilisticSampler::ShouldSample(const Span& span) {
    if (probability_.load() >= 1.0) {
        return SamplingDecision::kSample;
    }
    if (probability_.load() <= 0.0) {
        return SamplingDecision::kDrop;
    }

    // Use trace ID hash for consistent sampling
    // This ensures all spans with the same trace ID get the same decision
    uint64_t hash = HashTraceId(span.trace_id);
    return hash < threshold_ ? SamplingDecision::kSample : SamplingDecision::kDrop;
}

// ============================================================================
// RateLimitingSampler
// ============================================================================

RateLimitingSampler::RateLimitingSampler(double traces_per_second)
    : max_tokens_per_second_(std::max(traces_per_second, 0.0)),
      tokens_(traces_per_second),
      last_refill_(std::chrono::steady_clock::now()) {}

void RateLimitingSampler::SetRate(double traces_per_second) {
    std::lock_guard<std::mutex> lock(mutex_);
    max_tokens_per_second_ = std::max(traces_per_second, 0.0);
}

double RateLimitingSampler::GetTokens() const {
    std::lock_guard<std::mutex> lock(mutex_);
    return tokens_.load();
}

double RateLimitingSampler::GetProbability() const {
    uint64_t total = total_requests_.load();
    if (total == 0) {
        return 1.0;
    }
    return static_cast<double>(sampled_requests_.load()) / static_cast<double>(total);
}

void RateLimitingSampler::RefillTokens() {
    auto now = std::chrono::steady_clock::now();
    auto elapsed = std::chrono::duration<double>(now - last_refill_);

    double new_tokens = elapsed.count() * max_tokens_per_second_.load();
    double current = tokens_.load();
    double max = max_tokens_per_second_.load();

    // Cap tokens at max rate
    tokens_ = std::min(current + new_tokens, max);
    last_refill_ = now;
}

SamplingDecision RateLimitingSampler::ShouldSample(const Span& span) {
    std::lock_guard<std::mutex> lock(mutex_);

    total_requests_++;
    RefillTokens();

    double current_tokens = tokens_.load();
    if (current_tokens >= 1.0) {
        tokens_ = current_tokens - 1.0;
        sampled_requests_++;
        return SamplingDecision::kSample;
    }

    return SamplingDecision::kDrop;
}

// ============================================================================
// ParentBasedSampler
// ============================================================================

ParentBasedSampler::ParentBasedSampler(std::unique_ptr<Sampler> root_sampler)
    : root_sampler_(std::move(root_sampler)) {
    if (!root_sampler_) {
        root_sampler_ = std::make_unique<AlwaysOnSampler>();
    }
}

SamplingDecision ParentBasedSampler::ShouldSample(const Span& span) {
    // If this is a root span (no parent), use root sampler
    if (span.parent_span_id.empty()) {
        return root_sampler_->ShouldSample(span);
    }

    // Check trace state for sampling decision from parent
    // The trace state might contain "sampled=true" or similar
    if (!span.trace_state.empty()) {
        // Look for sampling flag in trace state
        // Format: key1=value1,key2=value2,...
        if (span.trace_state.find("sampled=true") != std::string::npos ||
            span.trace_state.find("sampled=1") != std::string::npos) {
            return SamplingDecision::kSample;
        }
        if (span.trace_state.find("sampled=false") != std::string::npos ||
            span.trace_state.find("sampled=0") != std::string::npos) {
            return SamplingDecision::kDrop;
        }
    }

    // Default: follow root sampler decision for consistency
    return root_sampler_->ShouldSample(span);
}

double ParentBasedSampler::GetProbability() const {
    return root_sampler_->GetProbability();
}

// ============================================================================
// CompositeSampler
// ============================================================================

CompositeSampler::CompositeSampler(std::unique_ptr<Sampler> default_sampler)
    : default_sampler_(std::move(default_sampler)) {
    if (!default_sampler_) {
        default_sampler_ = std::make_unique<AlwaysOnSampler>();
    }
}

void CompositeSampler::AddServiceSampler(const std::string& service_name,
                                          std::unique_ptr<Sampler> sampler) {
    std::lock_guard<std::mutex> lock(mutex_);
    service_samplers_[service_name] = std::move(sampler);
}

SamplingDecision CompositeSampler::ShouldSample(const Span& span) {
    std::lock_guard<std::mutex> lock(mutex_);

    // Try to find service-specific sampler
    std::string service_name;
    if (span.resource.has_value()) {
        auto it = span.resource->attributes.find("service.name");
        if (it != span.resource->attributes.end()) {
            if (auto* str = std::get_if<std::string>(&it->second)) {
                service_name = *str;
            }
        }
    }

    if (!service_name.empty()) {
        auto sampler_it = service_samplers_.find(service_name);
        if (sampler_it != service_samplers_.end()) {
            return sampler_it->second->ShouldSample(span);
        }
    }

    // Fall back to default sampler
    return default_sampler_->ShouldSample(span);
}

double CompositeSampler::GetProbability() const {
    return default_sampler_->GetProbability();
}

// ============================================================================
// Factory Function
// ============================================================================

std::unique_ptr<Sampler> CreateSampler(const SamplerConfig& config) {
    std::unique_ptr<Sampler> base_sampler;

    switch (config.strategy) {
        case SamplerConfig::Strategy::kAlwaysOn:
            base_sampler = std::make_unique<AlwaysOnSampler>();
            break;

        case SamplerConfig::Strategy::kAlwaysOff:
            base_sampler = std::make_unique<AlwaysOffSampler>();
            break;

        case SamplerConfig::Strategy::kProbabilistic:
            base_sampler = std::make_unique<ProbabilisticSampler>(config.probability);
            break;

        case SamplerConfig::Strategy::kRateLimiting:
            base_sampler = std::make_unique<RateLimitingSampler>(config.traces_per_second);
            break;

        case SamplerConfig::Strategy::kParentBased:
            base_sampler = std::make_unique<ParentBasedSampler>(
                std::make_unique<ProbabilisticSampler>(config.probability));
            break;

        default:
            base_sampler = std::make_unique<ProbabilisticSampler>(config.probability);
            break;
    }

    // If there are service-specific overrides, wrap in composite sampler
    if (!config.service_rates.empty()) {
        auto composite = std::make_unique<CompositeSampler>(std::move(base_sampler));

        for (const auto& [service, rate] : config.service_rates) {
            composite->AddServiceSampler(
                service,
                std::make_unique<ProbabilisticSampler>(rate));
        }

        return composite;
    }

    return base_sampler;
}

}  // namespace pyflare::collector
