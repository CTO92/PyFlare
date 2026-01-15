#pragma once

/// @file sampler.h
/// @brief Sampling strategies for telemetry data

#include <atomic>
#include <chrono>
#include <memory>
#include <mutex>
#include <random>
#include <string>
#include <unordered_map>

#include "types.h"

namespace pyflare::collector {

/// Sampling decision
enum class SamplingDecision {
    kDrop,    ///< Drop the span
    kSample,  ///< Sample the span
    kRecord   ///< Record but don't sample (for root span propagation)
};

/// Base sampler interface
class Sampler {
public:
    virtual ~Sampler() = default;

    /// Determine if a span should be sampled
    /// @param span The span to evaluate
    /// @return Sampling decision
    virtual SamplingDecision ShouldSample(const Span& span) = 0;

    /// Get the sampler name
    virtual std::string Name() const = 0;

    /// Get current sampling probability (for metrics)
    virtual double GetProbability() const = 0;
};

/// Sampler configuration
struct SamplerConfig {
    enum class Strategy {
        kAlwaysOn,        ///< Always sample
        kAlwaysOff,       ///< Never sample
        kProbabilistic,   ///< Probabilistic sampling
        kRateLimiting,    ///< Rate-limiting sampler
        kParentBased      ///< Based on parent decision
    };

    Strategy strategy = Strategy::kProbabilistic;
    double probability = 1.0;              ///< For probabilistic
    double traces_per_second = 10000.0;    ///< For rate limiting

    /// Per-service overrides
    std::unordered_map<std::string, double> service_rates;
};

/// Always sample
class AlwaysOnSampler : public Sampler {
public:
    SamplingDecision ShouldSample(const Span& span) override {
        return SamplingDecision::kSample;
    }
    std::string Name() const override { return "always_on"; }
    double GetProbability() const override { return 1.0; }
};

/// Never sample
class AlwaysOffSampler : public Sampler {
public:
    SamplingDecision ShouldSample(const Span& span) override {
        return SamplingDecision::kDrop;
    }
    std::string Name() const override { return "always_off"; }
    double GetProbability() const override { return 0.0; }
};

/// Probabilistic sampler using trace ID hashing for consistency
class ProbabilisticSampler : public Sampler {
public:
    /// Create sampler with given probability
    /// @param probability Sampling probability (0.0 to 1.0)
    explicit ProbabilisticSampler(double probability);

    SamplingDecision ShouldSample(const Span& span) override;
    std::string Name() const override { return "probabilistic"; }
    double GetProbability() const override { return probability_; }

    /// Update probability dynamically
    void SetProbability(double probability);

private:
    std::atomic<double> probability_;
    uint64_t threshold_;

    void UpdateThreshold();
    static uint64_t HashTraceId(const std::string& trace_id);
};

/// Rate-limiting sampler using token bucket
class RateLimitingSampler : public Sampler {
public:
    /// Create sampler with given rate limit
    /// @param traces_per_second Maximum traces per second
    explicit RateLimitingSampler(double traces_per_second);

    SamplingDecision ShouldSample(const Span& span) override;
    std::string Name() const override { return "rate_limiting"; }
    double GetProbability() const override;

    /// Update rate limit dynamically
    void SetRate(double traces_per_second);

    /// Get current token count
    double GetTokens() const;

private:
    void RefillTokens();

    std::atomic<double> max_tokens_per_second_;
    std::atomic<double> tokens_;
    std::chrono::steady_clock::time_point last_refill_;
    mutable std::mutex mutex_;

    // Statistics
    std::atomic<uint64_t> total_requests_{0};
    std::atomic<uint64_t> sampled_requests_{0};
};

/// Parent-based sampler that respects parent sampling decision
class ParentBasedSampler : public Sampler {
public:
    /// Create parent-based sampler with fallback for root spans
    /// @param root_sampler Sampler to use for root spans
    explicit ParentBasedSampler(std::unique_ptr<Sampler> root_sampler);

    SamplingDecision ShouldSample(const Span& span) override;
    std::string Name() const override { return "parent_based"; }
    double GetProbability() const override;

private:
    std::unique_ptr<Sampler> root_sampler_;
};

/// Composite sampler with per-service overrides
class CompositeSampler : public Sampler {
public:
    /// Create composite sampler with default and per-service overrides
    /// @param default_sampler Default sampler for unspecified services
    explicit CompositeSampler(std::unique_ptr<Sampler> default_sampler);

    /// Add service-specific sampler
    void AddServiceSampler(const std::string& service_name,
                           std::unique_ptr<Sampler> sampler);

    SamplingDecision ShouldSample(const Span& span) override;
    std::string Name() const override { return "composite"; }
    double GetProbability() const override;

private:
    std::unique_ptr<Sampler> default_sampler_;
    std::unordered_map<std::string, std::unique_ptr<Sampler>> service_samplers_;
    mutable std::mutex mutex_;
};

/// Factory function to create sampler from configuration
std::unique_ptr<Sampler> CreateSampler(const SamplerConfig& config);

}  // namespace pyflare::collector
