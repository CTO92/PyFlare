/// @file sampler_test.cpp
/// @brief Unit tests for sampling strategies

#include <gtest/gtest.h>

#include "src/collector/sampler.h"

namespace pyflare::collector {
namespace {

// Helper to create a test span
Span CreateTestSpan(const std::string& trace_id, const std::string& name = "test-span") {
    Span span;
    span.trace_id = trace_id;
    span.span_id = "span-123";
    span.name = name;
    span.kind = SpanKind::kInternal;
    span.start_time_ns = 1000000000;
    span.end_time_ns = 2000000000;
    return span;
}

// ============================================================================
// AlwaysOnSampler Tests
// ============================================================================

TEST(AlwaysOnSamplerTest, SamplesAllSpans) {
    AlwaysOnSampler sampler;

    for (int i = 0; i < 100; ++i) {
        auto span = CreateTestSpan("trace-" + std::to_string(i));
        EXPECT_EQ(sampler.ShouldSample(span), SamplingDecision::kSample);
    }
}

TEST(AlwaysOnSamplerTest, ReturnsFullProbability) {
    AlwaysOnSampler sampler;
    EXPECT_DOUBLE_EQ(sampler.GetProbability(), 1.0);
}

// ============================================================================
// AlwaysOffSampler Tests
// ============================================================================

TEST(AlwaysOffSamplerTest, DropsAllSpans) {
    AlwaysOffSampler sampler;

    for (int i = 0; i < 100; ++i) {
        auto span = CreateTestSpan("trace-" + std::to_string(i));
        EXPECT_EQ(sampler.ShouldSample(span), SamplingDecision::kDrop);
    }
}

TEST(AlwaysOffSamplerTest, ReturnsZeroProbability) {
    AlwaysOffSampler sampler;
    EXPECT_DOUBLE_EQ(sampler.GetProbability(), 0.0);
}

// ============================================================================
// ProbabilisticSampler Tests
// ============================================================================

TEST(ProbabilisticSamplerTest, FullProbabilitySamplesAll) {
    ProbabilisticSampler sampler(1.0);

    for (int i = 0; i < 100; ++i) {
        auto span = CreateTestSpan("trace-" + std::to_string(i));
        EXPECT_EQ(sampler.ShouldSample(span), SamplingDecision::kSample);
    }
}

TEST(ProbabilisticSamplerTest, ZeroProbabilityDropsAll) {
    ProbabilisticSampler sampler(0.0);

    for (int i = 0; i < 100; ++i) {
        auto span = CreateTestSpan("trace-" + std::to_string(i));
        EXPECT_EQ(sampler.ShouldSample(span), SamplingDecision::kDrop);
    }
}

TEST(ProbabilisticSamplerTest, ConsistentSamplingForSameTraceId) {
    ProbabilisticSampler sampler(0.5);

    auto span1 = CreateTestSpan("consistent-trace-id");
    auto span2 = CreateTestSpan("consistent-trace-id");

    // Same trace ID should give same decision
    EXPECT_EQ(sampler.ShouldSample(span1), sampler.ShouldSample(span2));
}

TEST(ProbabilisticSamplerTest, ApproximateSampleRate) {
    ProbabilisticSampler sampler(0.5);

    int sampled = 0;
    const int total = 10000;

    for (int i = 0; i < total; ++i) {
        auto span = CreateTestSpan("trace-" + std::to_string(i));
        if (sampler.ShouldSample(span) == SamplingDecision::kSample) {
            ++sampled;
        }
    }

    // Should be approximately 50% with some tolerance
    double rate = static_cast<double>(sampled) / total;
    EXPECT_GT(rate, 0.4);
    EXPECT_LT(rate, 0.6);
}

TEST(ProbabilisticSamplerTest, ProbabilityClampedToValidRange) {
    // Test that probability is clamped to [0, 1]
    ProbabilisticSampler high_sampler(2.0);
    EXPECT_DOUBLE_EQ(high_sampler.GetProbability(), 1.0);

    ProbabilisticSampler negative_sampler(-1.0);
    EXPECT_DOUBLE_EQ(negative_sampler.GetProbability(), 0.0);
}

TEST(ProbabilisticSamplerTest, SetProbabilityUpdatesThreshold) {
    ProbabilisticSampler sampler(1.0);

    // Initially samples everything
    auto span = CreateTestSpan("test-trace");
    EXPECT_EQ(sampler.ShouldSample(span), SamplingDecision::kSample);

    // Change to 0% sampling
    sampler.SetProbability(0.0);
    EXPECT_EQ(sampler.ShouldSample(span), SamplingDecision::kDrop);
}

// ============================================================================
// RateLimitingSampler Tests
// ============================================================================

TEST(RateLimitingSamplerTest, InitialTokensAvailable) {
    RateLimitingSampler sampler(10.0);  // 10 traces per second

    // Should be able to sample immediately up to the rate
    for (int i = 0; i < 10; ++i) {
        auto span = CreateTestSpan("trace-" + std::to_string(i));
        EXPECT_EQ(sampler.ShouldSample(span), SamplingDecision::kSample);
    }
}

TEST(RateLimitingSamplerTest, TokensDeplete) {
    RateLimitingSampler sampler(5.0);  // 5 traces per second

    // Use up all tokens
    for (int i = 0; i < 5; ++i) {
        auto span = CreateTestSpan("trace-" + std::to_string(i));
        sampler.ShouldSample(span);
    }

    // Should drop next span (tokens depleted)
    auto extra_span = CreateTestSpan("extra-trace");
    EXPECT_EQ(sampler.ShouldSample(extra_span), SamplingDecision::kDrop);
}

TEST(RateLimitingSamplerTest, GetTokensReturnsCurrentValue) {
    RateLimitingSampler sampler(10.0);

    // Initially should have ~10 tokens
    EXPECT_GT(sampler.GetTokens(), 9.0);
    EXPECT_LE(sampler.GetTokens(), 10.0);

    // Use some tokens
    for (int i = 0; i < 5; ++i) {
        auto span = CreateTestSpan("trace-" + std::to_string(i));
        sampler.ShouldSample(span);
    }

    // Should have ~5 tokens remaining
    EXPECT_GT(sampler.GetTokens(), 4.0);
    EXPECT_LE(sampler.GetTokens(), 5.5);
}

// ============================================================================
// ParentBasedSampler Tests
// ============================================================================

TEST(ParentBasedSamplerTest, RootSpanUsesRootSampler) {
    auto root_sampler = std::make_unique<ProbabilisticSampler>(1.0);
    ParentBasedSampler sampler(std::move(root_sampler));

    // Root span (no parent)
    auto span = CreateTestSpan("root-trace");
    span.parent_span_id = "";  // No parent

    EXPECT_EQ(sampler.ShouldSample(span), SamplingDecision::kSample);
}

TEST(ParentBasedSamplerTest, ChildSpanRespectsTraceState) {
    auto root_sampler = std::make_unique<AlwaysOffSampler>();
    ParentBasedSampler sampler(std::move(root_sampler));

    // Child span with sampled=true in trace state
    auto span = CreateTestSpan("child-trace");
    span.parent_span_id = "parent-123";
    span.trace_state = "sampled=true";

    EXPECT_EQ(sampler.ShouldSample(span), SamplingDecision::kSample);
}

TEST(ParentBasedSamplerTest, ChildSpanRespectsTraceStateDropped) {
    auto root_sampler = std::make_unique<AlwaysOnSampler>();
    ParentBasedSampler sampler(std::move(root_sampler));

    // Child span with sampled=false in trace state
    auto span = CreateTestSpan("child-trace");
    span.parent_span_id = "parent-123";
    span.trace_state = "sampled=false";

    EXPECT_EQ(sampler.ShouldSample(span), SamplingDecision::kDrop);
}

TEST(ParentBasedSamplerTest, DefaultsToRootSamplerWithoutTraceState) {
    auto root_sampler = std::make_unique<AlwaysOnSampler>();
    ParentBasedSampler sampler(std::move(root_sampler));

    // Child span without sampling info in trace state
    auto span = CreateTestSpan("child-trace");
    span.parent_span_id = "parent-123";
    span.trace_state = "other=value";

    // Should fall back to root sampler
    EXPECT_EQ(sampler.ShouldSample(span), SamplingDecision::kSample);
}

// ============================================================================
// CompositeSampler Tests
// ============================================================================

TEST(CompositeSamplerTest, UsesDefaultSamplerForUnknownService) {
    auto default_sampler = std::make_unique<AlwaysOnSampler>();
    CompositeSampler sampler(std::move(default_sampler));

    auto span = CreateTestSpan("unknown-trace");
    EXPECT_EQ(sampler.ShouldSample(span), SamplingDecision::kSample);
}

TEST(CompositeSamplerTest, UsesServiceSpecificSampler) {
    auto default_sampler = std::make_unique<AlwaysOffSampler>();
    CompositeSampler sampler(std::move(default_sampler));

    // Add service-specific sampler
    sampler.AddServiceSampler("my-service", std::make_unique<AlwaysOnSampler>());

    // Create span with service name in resource
    auto span = CreateTestSpan("service-trace");
    Resource resource;
    resource.attributes["service.name"] = std::string("my-service");
    span.resource = resource;

    EXPECT_EQ(sampler.ShouldSample(span), SamplingDecision::kSample);
}

TEST(CompositeSamplerTest, FallsBackToDefaultForOtherService) {
    auto default_sampler = std::make_unique<AlwaysOffSampler>();
    CompositeSampler sampler(std::move(default_sampler));

    sampler.AddServiceSampler("service-a", std::make_unique<AlwaysOnSampler>());

    // Create span with different service name
    auto span = CreateTestSpan("other-trace");
    Resource resource;
    resource.attributes["service.name"] = std::string("service-b");
    span.resource = resource;

    // Should use default sampler (AlwaysOff)
    EXPECT_EQ(sampler.ShouldSample(span), SamplingDecision::kDrop);
}

// ============================================================================
// Factory Function Tests
// ============================================================================

TEST(CreateSamplerTest, CreatesAlwaysOnSampler) {
    SamplerConfig config;
    config.strategy = SamplerConfig::Strategy::kAlwaysOn;

    auto sampler = CreateSampler(config);
    EXPECT_NE(sampler, nullptr);
    EXPECT_DOUBLE_EQ(sampler->GetProbability(), 1.0);
}

TEST(CreateSamplerTest, CreatesAlwaysOffSampler) {
    SamplerConfig config;
    config.strategy = SamplerConfig::Strategy::kAlwaysOff;

    auto sampler = CreateSampler(config);
    EXPECT_NE(sampler, nullptr);
    EXPECT_DOUBLE_EQ(sampler->GetProbability(), 0.0);
}

TEST(CreateSamplerTest, CreatesProbabilisticSampler) {
    SamplerConfig config;
    config.strategy = SamplerConfig::Strategy::kProbabilistic;
    config.probability = 0.75;

    auto sampler = CreateSampler(config);
    EXPECT_NE(sampler, nullptr);
    EXPECT_DOUBLE_EQ(sampler->GetProbability(), 0.75);
}

TEST(CreateSamplerTest, CreatesCompositeSamplerWithServiceRates) {
    SamplerConfig config;
    config.strategy = SamplerConfig::Strategy::kProbabilistic;
    config.probability = 0.5;
    config.service_rates["high-priority"] = 1.0;
    config.service_rates["low-priority"] = 0.1;

    auto sampler = CreateSampler(config);
    EXPECT_NE(sampler, nullptr);
}

}  // namespace
}  // namespace pyflare::collector
