#include "metrics.h"

#include <algorithm>
#include <sstream>

namespace pyflare {

// Default histogram buckets (latency in seconds)
static const std::vector<double> kDefaultBuckets = {
    0.001, 0.005, 0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1.0, 2.5, 5.0, 10.0
};

// Counter implementation
Counter::Counter(std::string name, std::string description)
    : name_(std::move(name)), description_(std::move(description)) {}

void Counter::Increment() {
    value_.fetch_add(1, std::memory_order_relaxed);
}

void Counter::Add(int64_t delta) {
    if (delta >= 0) {
        value_.fetch_add(delta, std::memory_order_relaxed);
    }
}

int64_t Counter::Value() const {
    return value_.load(std::memory_order_relaxed);
}

// Gauge implementation
Gauge::Gauge(std::string name, std::string description)
    : name_(std::move(name)), description_(std::move(description)) {}

void Gauge::Set(double value) {
    value_.store(value, std::memory_order_relaxed);
}

void Gauge::Increment(double delta) {
    double current = value_.load(std::memory_order_relaxed);
    while (!value_.compare_exchange_weak(current, current + delta,
                                          std::memory_order_relaxed)) {
        // Retry on failure
    }
}

void Gauge::Decrement(double delta) {
    Increment(-delta);
}

double Gauge::Value() const {
    return value_.load(std::memory_order_relaxed);
}

// Histogram implementation
Histogram::Histogram(std::string name, std::string description)
    : Histogram(std::move(name), kDefaultBuckets, std::move(description)) {}

Histogram::Histogram(std::string name, std::vector<double> buckets, std::string description)
    : name_(std::move(name)),
      description_(std::move(description)),
      bucket_bounds_(std::move(buckets)) {
    std::sort(bucket_bounds_.begin(), bucket_bounds_.end());
    bucket_counts_.resize(bucket_bounds_.size() + 1);  // +1 for +Inf bucket
    for (auto& count : bucket_counts_) {
        count.store(0, std::memory_order_relaxed);
    }
}

void Histogram::Observe(double value) {
    // Update sum and count
    count_.fetch_add(1, std::memory_order_relaxed);

    double current_sum = sum_.load(std::memory_order_relaxed);
    while (!sum_.compare_exchange_weak(current_sum, current_sum + value,
                                        std::memory_order_relaxed)) {
        // Retry on failure
    }

    // Find the bucket and increment
    auto it = std::upper_bound(bucket_bounds_.begin(), bucket_bounds_.end(), value);
    size_t bucket_idx = std::distance(bucket_bounds_.begin(), it);
    bucket_counts_[bucket_idx].fetch_add(1, std::memory_order_relaxed);
}

int64_t Histogram::Count() const {
    return count_.load(std::memory_order_relaxed);
}

double Histogram::Sum() const {
    return sum_.load(std::memory_order_relaxed);
}

std::vector<std::pair<double, int64_t>> Histogram::Buckets() const {
    std::vector<std::pair<double, int64_t>> result;
    result.reserve(bucket_bounds_.size() + 1);

    int64_t cumulative = 0;
    for (size_t i = 0; i < bucket_bounds_.size(); ++i) {
        cumulative += bucket_counts_[i].load(std::memory_order_relaxed);
        result.emplace_back(bucket_bounds_[i], cumulative);
    }
    // +Inf bucket
    cumulative += bucket_counts_.back().load(std::memory_order_relaxed);
    result.emplace_back(std::numeric_limits<double>::infinity(), cumulative);

    return result;
}

// ScopedTimer implementation
ScopedTimer::ScopedTimer(Histogram& histogram)
    : histogram_(histogram), start_(std::chrono::steady_clock::now()) {}

ScopedTimer::~ScopedTimer() {
    auto end = std::chrono::steady_clock::now();
    std::chrono::duration<double> duration = end - start_;
    histogram_.Observe(duration.count());
}

// MetricsRegistry implementation
MetricsRegistry& MetricsRegistry::Instance() {
    static MetricsRegistry instance;
    return instance;
}

Counter& MetricsRegistry::GetCounter(const std::string& name, const std::string& description) {
    std::lock_guard<std::mutex> lock(mutex_);
    auto it = counters_.find(name);
    if (it == counters_.end()) {
        auto [new_it, _] = counters_.emplace(name, std::make_unique<Counter>(name, description));
        return *new_it->second;
    }
    return *it->second;
}

Gauge& MetricsRegistry::GetGauge(const std::string& name, const std::string& description) {
    std::lock_guard<std::mutex> lock(mutex_);
    auto it = gauges_.find(name);
    if (it == gauges_.end()) {
        auto [new_it, _] = gauges_.emplace(name, std::make_unique<Gauge>(name, description));
        return *new_it->second;
    }
    return *it->second;
}

Histogram& MetricsRegistry::GetHistogram(const std::string& name, const std::string& description) {
    std::lock_guard<std::mutex> lock(mutex_);
    auto it = histograms_.find(name);
    if (it == histograms_.end()) {
        auto [new_it, _] = histograms_.emplace(name, std::make_unique<Histogram>(name, description));
        return *new_it->second;
    }
    return *it->second;
}

std::string MetricsRegistry::ExportText() const {
    std::lock_guard<std::mutex> lock(mutex_);
    std::ostringstream oss;

    // Export counters
    for (const auto& [name, counter] : counters_) {
        oss << "# HELP " << name << " " << counter->Description() << "\n";
        oss << "# TYPE " << name << " counter\n";
        oss << name << " " << counter->Value() << "\n";
    }

    // Export gauges
    for (const auto& [name, gauge] : gauges_) {
        oss << "# HELP " << name << " " << gauge->Description() << "\n";
        oss << "# TYPE " << name << " gauge\n";
        oss << name << " " << gauge->Value() << "\n";
    }

    // Export histograms
    for (const auto& [name, histogram] : histograms_) {
        oss << "# HELP " << name << " " << histogram->Description() << "\n";
        oss << "# TYPE " << name << " histogram\n";
        for (const auto& [bound, count] : histogram->Buckets()) {
            oss << name << "_bucket{le=\"" << bound << "\"} " << count << "\n";
        }
        oss << name << "_sum " << histogram->Sum() << "\n";
        oss << name << "_count " << histogram->Count() << "\n";
    }

    return oss.str();
}

void MetricsRegistry::Reset() {
    std::lock_guard<std::mutex> lock(mutex_);
    counters_.clear();
    gauges_.clear();
    histograms_.clear();
}

}  // namespace pyflare
