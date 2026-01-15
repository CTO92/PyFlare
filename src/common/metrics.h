#pragma once

/// @file metrics.h
/// @brief PyFlare internal metrics collection for self-monitoring

#include <atomic>
#include <chrono>
#include <functional>
#include <memory>
#include <mutex>
#include <string>
#include <unordered_map>

namespace pyflare {

/// @brief A simple counter metric
class Counter {
public:
    explicit Counter(std::string name, std::string description = "");

    /// @brief Increment the counter by 1
    void Increment();

    /// @brief Increment the counter by a specific amount
    /// @param delta Amount to add (must be non-negative)
    void Add(int64_t delta);

    /// @brief Get the current value
    int64_t Value() const;

    /// @brief Get the metric name
    const std::string& Name() const { return name_; }

    /// @brief Get the metric description
    const std::string& Description() const { return description_; }

private:
    std::string name_;
    std::string description_;
    std::atomic<int64_t> value_{0};
};

/// @brief A gauge metric that can go up and down
class Gauge {
public:
    explicit Gauge(std::string name, std::string description = "");

    /// @brief Set the gauge value
    void Set(double value);

    /// @brief Increment the gauge
    void Increment(double delta = 1.0);

    /// @brief Decrement the gauge
    void Decrement(double delta = 1.0);

    /// @brief Get the current value
    double Value() const;

    const std::string& Name() const { return name_; }
    const std::string& Description() const { return description_; }

private:
    std::string name_;
    std::string description_;
    std::atomic<double> value_{0.0};
};

/// @brief A histogram for measuring value distributions
class Histogram {
public:
    /// @brief Create a histogram with default buckets
    explicit Histogram(std::string name, std::string description = "");

    /// @brief Create a histogram with custom buckets
    Histogram(std::string name, std::vector<double> buckets, std::string description = "");

    /// @brief Record a value
    void Observe(double value);

    /// @brief Get the count of observations
    int64_t Count() const;

    /// @brief Get the sum of all observations
    double Sum() const;

    /// @brief Get bucket counts
    std::vector<std::pair<double, int64_t>> Buckets() const;

    const std::string& Name() const { return name_; }
    const std::string& Description() const { return description_; }

private:
    std::string name_;
    std::string description_;
    std::vector<double> bucket_bounds_;
    std::vector<std::atomic<int64_t>> bucket_counts_;
    std::atomic<int64_t> count_{0};
    std::atomic<double> sum_{0.0};
    mutable std::mutex mutex_;
};

/// @brief RAII timer for measuring duration
class ScopedTimer {
public:
    explicit ScopedTimer(Histogram& histogram);
    ~ScopedTimer();

    // Non-copyable
    ScopedTimer(const ScopedTimer&) = delete;
    ScopedTimer& operator=(const ScopedTimer&) = delete;

private:
    Histogram& histogram_;
    std::chrono::steady_clock::time_point start_;
};

/// @brief Metrics registry for managing all metrics
class MetricsRegistry {
public:
    /// @brief Get the global registry instance
    static MetricsRegistry& Instance();

    /// @brief Register or get an existing counter
    Counter& GetCounter(const std::string& name, const std::string& description = "");

    /// @brief Register or get an existing gauge
    Gauge& GetGauge(const std::string& name, const std::string& description = "");

    /// @brief Register or get an existing histogram
    Histogram& GetHistogram(const std::string& name, const std::string& description = "");

    /// @brief Export all metrics in a format suitable for logging/monitoring
    std::string ExportText() const;

    /// @brief Reset all metrics (primarily for testing)
    void Reset();

private:
    MetricsRegistry() = default;

    mutable std::mutex mutex_;
    std::unordered_map<std::string, std::unique_ptr<Counter>> counters_;
    std::unordered_map<std::string, std::unique_ptr<Gauge>> gauges_;
    std::unordered_map<std::string, std::unique_ptr<Histogram>> histograms_;
};

// Convenience macros for metrics

#define PYFLARE_COUNTER(name) \
    ::pyflare::MetricsRegistry::Instance().GetCounter(name)

#define PYFLARE_GAUGE(name) \
    ::pyflare::MetricsRegistry::Instance().GetGauge(name)

#define PYFLARE_HISTOGRAM(name) \
    ::pyflare::MetricsRegistry::Instance().GetHistogram(name)

#define PYFLARE_TIMER(histogram) \
    ::pyflare::ScopedTimer _timer_##__LINE__(histogram)

}  // namespace pyflare
