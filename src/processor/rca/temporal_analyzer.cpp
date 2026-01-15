/// @file temporal_analyzer.cpp
/// @brief Temporal correlation analysis implementation

#include "processor/rca/temporal_analyzer.h"

#include <algorithm>
#include <cmath>
#include <numeric>
#include <sstream>

namespace pyflare::rca {

TemporalAnalyzer::TemporalAnalyzer(
    std::shared_ptr<storage::ClickHouseClient> clickhouse,
    TemporalAnalyzerConfig config)
    : clickhouse_(std::move(clickhouse)),
      config_(std::move(config)) {}

TemporalAnalyzer::~TemporalAnalyzer() = default;

absl::Status TemporalAnalyzer::Initialize() {
    if (initialized_) {
        return absl::OkStatus();
    }

    initialized_ = true;
    return absl::OkStatus();
}

absl::StatusOr<std::vector<TemporalCorrelation>> TemporalAnalyzer::FindCorrelations(
    const std::string& model_id) {
    if (!initialized_) {
        return absl::FailedPreconditionError("Analyzer not initialized");
    }

    auto start_time = std::chrono::steady_clock::now();

    auto end = std::chrono::system_clock::now();
    auto start = end - config_.analysis_window;

    std::vector<TemporalCorrelation> correlations;

    // Get events for each type
    std::unordered_map<std::string, std::vector<double>> time_series;

    for (const auto& event_type : config_.event_types) {
        auto events = QueryEvents(model_id, event_type, start, end);
        if (events.ok() && !events->empty()) {
            time_series[event_type] = ComputeEventCounts(
                *events, start, end, config_.resolution);
        }
    }

    // Compute pairwise correlations
    for (size_t i = 0; i < config_.event_types.size(); ++i) {
        for (size_t j = i + 1; j < config_.event_types.size(); ++j) {
            const auto& type_a = config_.event_types[i];
            const auto& type_b = config_.event_types[j];

            auto it_a = time_series.find(type_a);
            auto it_b = time_series.find(type_b);

            if (it_a == time_series.end() || it_b == time_series.end()) {
                continue;
            }

            const auto& series_a = it_a->second;
            const auto& series_b = it_b->second;

            if (series_a.size() < config_.min_events ||
                series_b.size() < config_.min_events) {
                continue;
            }

            // Find best lag
            size_t max_lag_steps = config_.max_lag.count() * 60 / config_.resolution.count();
            auto [best_lag, correlation] = FindBestLag(series_a, series_b, max_lag_steps);

            double p_value = ComputePValue(correlation, series_a.size());

            auto corr = BuildCorrelation(
                type_a, type_b, correlation, best_lag * config_.resolution.count(), p_value);

            correlations.push_back(corr);
        }
    }

    // Sort by absolute correlation strength
    std::sort(correlations.begin(), correlations.end(),
        [](const TemporalCorrelation& a, const TemporalCorrelation& b) {
            return std::abs(a.correlation) > std::abs(b.correlation);
        });

    auto end_time = std::chrono::steady_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(
        end_time - start_time);

    // Update statistics
    {
        std::lock_guard<std::mutex> lock(stats_mutex_);
        stats_.correlations_computed += correlations.size();
        stats_.avg_analysis_time_ms = (stats_.avg_analysis_time_ms *
            stats_.correlations_computed + duration.count()) /
            (stats_.correlations_computed + 1);
    }

    return correlations;
}

absl::StatusOr<TemporalCorrelation> TemporalAnalyzer::FindCorrelation(
    const std::string& model_id,
    const std::string& event_type_a,
    const std::string& event_type_b) {
    if (!initialized_) {
        return absl::FailedPreconditionError("Analyzer not initialized");
    }

    auto end = std::chrono::system_clock::now();
    auto start = end - config_.analysis_window;

    auto events_a = QueryEvents(model_id, event_type_a, start, end);
    if (!events_a.ok()) return events_a.status();

    auto events_b = QueryEvents(model_id, event_type_b, start, end);
    if (!events_b.ok()) return events_b.status();

    if (events_a->size() < config_.min_events || events_b->size() < config_.min_events) {
        return absl::InvalidArgumentError("Insufficient events for correlation analysis");
    }

    auto series_a = ComputeEventCounts(*events_a, start, end, config_.resolution);
    auto series_b = ComputeEventCounts(*events_b, start, end, config_.resolution);

    size_t max_lag_steps = config_.max_lag.count() * 60 / config_.resolution.count();
    auto [best_lag, correlation] = FindBestLag(series_a, series_b, max_lag_steps);

    double p_value = ComputePValue(correlation, series_a.size());

    return BuildCorrelation(
        event_type_a, event_type_b, correlation,
        best_lag * config_.resolution.count(), p_value);
}

absl::StatusOr<std::vector<TemporalCorrelation>> TemporalAnalyzer::FindPrecedingEvents(
    const std::string& model_id,
    const std::string& target_event,
    size_t max_results) {
    auto all_correlations = FindCorrelations(model_id);
    if (!all_correlations.ok()) {
        return all_correlations.status();
    }

    std::vector<TemporalCorrelation> preceding;

    for (const auto& corr : *all_correlations) {
        bool involves_target = (corr.event_type_a == target_event ||
                               corr.event_type_b == target_event);
        if (!involves_target || !corr.is_significant) continue;

        // Check if something precedes the target
        if ((corr.event_type_b == target_event &&
             corr.direction == TemporalCorrelation::Direction::kALeadsB) ||
            (corr.event_type_a == target_event &&
             corr.direction == TemporalCorrelation::Direction::kBLeadsA)) {
            preceding.push_back(corr);
        }
    }

    // Sort by correlation strength
    std::sort(preceding.begin(), preceding.end(),
        [](const TemporalCorrelation& a, const TemporalCorrelation& b) {
            return std::abs(a.correlation) > std::abs(b.correlation);
        });

    if (preceding.size() > max_results) {
        preceding.resize(max_results);
    }

    return preceding;
}

absl::StatusOr<std::vector<TemporalPattern>> TemporalAnalyzer::DetectPatterns(
    const std::string& model_id) {
    if (!initialized_) {
        return absl::FailedPreconditionError("Analyzer not initialized");
    }

    std::vector<TemporalPattern> patterns;

    auto end = std::chrono::system_clock::now();
    auto start = end - config_.analysis_window;

    for (const auto& event_type : config_.event_types) {
        auto events = QueryEvents(model_id, event_type, start, end);
        if (!events.ok() || events->empty()) continue;

        auto series = ComputeEventCounts(*events, start, end, config_.resolution);

        // Detect spikes
        auto spike = DetectSpike(series, start, config_.resolution);
        if (spike.has_value()) {
            spike->event_types.push_back(event_type);
            patterns.push_back(*spike);
        }

        // Detect periodicity
        auto periodicity = DetectPeriodicity(series, start, config_.resolution);
        if (periodicity.has_value()) {
            periodicity->event_types.push_back(event_type);
            patterns.push_back(*periodicity);
        }

        // Detect trends
        auto trend = DetectTrend(model_id, event_type);
        if (trend.ok() && trend->confidence > 0.5) {
            patterns.push_back(*trend);
        }
    }

    // Update statistics
    {
        std::lock_guard<std::mutex> lock(stats_mutex_);
        stats_.patterns_detected += patterns.size();
    }

    return patterns;
}

absl::StatusOr<std::vector<TemporalPattern>> TemporalAnalyzer::DetectAnomalies(
    const std::string& model_id) {
    if (!initialized_) {
        return absl::FailedPreconditionError("Analyzer not initialized");
    }

    std::vector<TemporalPattern> anomalies;

    auto end = std::chrono::system_clock::now();
    auto start = end - config_.analysis_window;

    for (const auto& event_type : config_.event_types) {
        auto events = QueryEvents(model_id, event_type, start, end);
        if (!events.ok() || events->empty()) continue;

        auto series = ComputeEventCounts(*events, start, end, config_.resolution);

        auto anomaly_indices = DetectAnomalyPoints(series, config_.anomaly_sensitivity);

        for (size_t idx : anomaly_indices) {
            TemporalPattern anomaly;
            anomaly.pattern_id = model_id + "_" + event_type + "_anomaly_" + std::to_string(idx);
            anomaly.pattern_type = "anomaly";
            anomaly.event_types.push_back(event_type);
            anomaly.start_time = start + std::chrono::minutes(idx * config_.resolution.count());
            anomaly.end_time = anomaly.start_time + config_.resolution;
            anomaly.anomaly_score = series[idx];
            anomaly.confidence = 0.8;
            anomaly.description = "Anomalous spike in " + event_type + " events";

            anomalies.push_back(anomaly);
        }
    }

    // Update statistics
    {
        std::lock_guard<std::mutex> lock(stats_mutex_);
        stats_.anomalies_detected += anomalies.size();
    }

    return anomalies;
}

absl::StatusOr<TemporalPattern> TemporalAnalyzer::DetectTrend(
    const std::string& model_id,
    const std::string& event_type) {
    auto end = std::chrono::system_clock::now();
    auto start = end - config_.analysis_window;

    auto events = QueryEvents(model_id, event_type, start, end);
    if (!events.ok()) return events.status();

    auto series = ComputeEventCounts(*events, start, end, config_.resolution);

    if (series.size() < 10) {
        return absl::InvalidArgumentError("Insufficient data for trend detection");
    }

    double slope = ComputeTrendSlope(series);

    TemporalPattern pattern;
    pattern.pattern_id = model_id + "_" + event_type + "_trend";
    pattern.pattern_type = "trend";
    pattern.event_types.push_back(event_type);
    pattern.start_time = start;
    pattern.end_time = end;
    pattern.trend_slope = slope;

    // Confidence based on how pronounced the trend is
    pattern.confidence = std::min(1.0, std::abs(slope) * 10);

    if (slope > 0.01) {
        pattern.description = "Increasing trend in " + event_type + " events";
    } else if (slope < -0.01) {
        pattern.description = "Decreasing trend in " + event_type + " events";
    } else {
        pattern.description = "Stable " + event_type + " event rate";
        pattern.confidence = 0.3;
    }

    return pattern;
}

absl::Status TemporalAnalyzer::AddEvent(const TemporalEvent& event) {
    std::lock_guard<std::mutex> lock(buffer_mutex_);
    event_buffer_.push_back(event);

    // Update statistics
    {
        std::lock_guard<std::mutex> stats_lock(stats_mutex_);
        stats_.events_processed++;
    }

    return absl::OkStatus();
}

absl::StatusOr<std::vector<TemporalEvent>> TemporalAnalyzer::QueryEvents(
    const std::string& model_id,
    const std::string& event_type,
    std::optional<std::chrono::system_clock::time_point> start,
    std::optional<std::chrono::system_clock::time_point> end) {
    // First check in-memory buffer
    std::vector<TemporalEvent> results;

    {
        std::lock_guard<std::mutex> lock(buffer_mutex_);
        for (const auto& event : event_buffer_) {
            bool model_match = model_id.empty() || event.model_id == model_id;
            bool type_match = event_type.empty() || event.event_type == event_type;
            bool time_match = true;

            if (start.has_value() && event.timestamp < *start) time_match = false;
            if (end.has_value() && event.timestamp > *end) time_match = false;

            if (model_match && type_match && time_match) {
                results.push_back(event);
            }
        }
    }

    // If ClickHouse is available, also query from storage
    if (clickhouse_ && clickhouse_->IsConnected()) {
        auto now = std::chrono::system_clock::now();
        auto query_start = start.value_or(now - config_.analysis_window);
        auto query_end = end.value_or(now);

        std::string query = BuildEventQuery(model_id, event_type, query_start, query_end);
        auto query_result = clickhouse_->Execute(query);

        if (query_result.ok()) {
            for (const auto& row : query_result->rows) {
                if (row.size() >= 5) {
                    TemporalEvent event;
                    event.event_id = row[0];
                    event.event_type = row[1];
                    event.model_id = row[2];
                    // Parse timestamp from row[3]
                    event.severity = std::stod(row[4]);
                    results.push_back(event);
                }
            }
        }
    }

    // Sort by timestamp
    std::sort(results.begin(), results.end(),
        [](const TemporalEvent& a, const TemporalEvent& b) {
            return a.timestamp < b.timestamp;
        });

    return results;
}

absl::StatusOr<std::vector<std::pair<std::chrono::system_clock::time_point, double>>>
TemporalAnalyzer::GetTimeSeries(
    const std::string& model_id,
    const std::string& event_type,
    std::chrono::minutes resolution) {
    auto end = std::chrono::system_clock::now();
    auto start = end - config_.analysis_window;

    auto events = QueryEvents(model_id, event_type, start, end);
    if (!events.ok()) return events.status();

    auto counts = ComputeEventCounts(*events, start, end, resolution);

    std::vector<std::pair<std::chrono::system_clock::time_point, double>> series;
    auto current = start;

    for (double count : counts) {
        series.emplace_back(current, count);
        current += resolution;
    }

    return series;
}

absl::StatusOr<std::vector<std::pair<int, double>>> TemporalAnalyzer::ComputeCrossCorrelation(
    const std::vector<double>& series_a,
    const std::vector<double>& series_b,
    size_t max_lag) {
    if (series_a.size() != series_b.size()) {
        return absl::InvalidArgumentError("Series must have same length");
    }

    std::vector<std::pair<int, double>> correlations;

    for (int lag = -static_cast<int>(max_lag); lag <= static_cast<int>(max_lag); ++lag) {
        std::vector<double> shifted_a, shifted_b;

        if (lag >= 0) {
            for (size_t i = lag; i < series_a.size(); ++i) {
                shifted_a.push_back(series_a[i - lag]);
                shifted_b.push_back(series_b[i]);
            }
        } else {
            for (size_t i = -lag; i < series_a.size(); ++i) {
                shifted_a.push_back(series_a[i]);
                shifted_b.push_back(series_b[i + lag]);
            }
        }

        if (shifted_a.size() >= 3) {
            double corr = ComputePearsonCorrelation(shifted_a, shifted_b);
            correlations.emplace_back(lag, corr);
        }
    }

    return correlations;
}

void TemporalAnalyzer::SetConfig(TemporalAnalyzerConfig config) {
    config_ = std::move(config);
}

TemporalAnalyzer::Stats TemporalAnalyzer::GetStats() const {
    std::lock_guard<std::mutex> lock(stats_mutex_);
    return stats_;
}

void TemporalAnalyzer::ResetStats() {
    std::lock_guard<std::mutex> lock(stats_mutex_);
    stats_ = Stats{};
}

// ============================================================================
// Private Implementation
// ============================================================================

std::vector<double> TemporalAnalyzer::ComputeEventCounts(
    const std::vector<TemporalEvent>& events,
    std::chrono::system_clock::time_point start,
    std::chrono::system_clock::time_point end,
    std::chrono::minutes resolution) {
    auto duration = std::chrono::duration_cast<std::chrono::minutes>(end - start);
    size_t num_buckets = duration.count() / resolution.count() + 1;

    std::vector<double> counts(num_buckets, 0.0);

    for (const auto& event : events) {
        auto offset = std::chrono::duration_cast<std::chrono::minutes>(
            event.timestamp - start);
        size_t bucket = offset.count() / resolution.count();

        if (bucket < counts.size()) {
            counts[bucket] += 1.0;
        }
    }

    return counts;
}

double TemporalAnalyzer::ComputePearsonCorrelation(
    const std::vector<double>& x,
    const std::vector<double>& y) {
    if (x.size() != y.size() || x.size() < 2) {
        return 0.0;
    }

    size_t n = x.size();
    double mean_x = std::accumulate(x.begin(), x.end(), 0.0) / n;
    double mean_y = std::accumulate(y.begin(), y.end(), 0.0) / n;

    double cov = 0.0, var_x = 0.0, var_y = 0.0;

    for (size_t i = 0; i < n; ++i) {
        double dx = x[i] - mean_x;
        double dy = y[i] - mean_y;
        cov += dx * dy;
        var_x += dx * dx;
        var_y += dy * dy;
    }

    double denom = std::sqrt(var_x * var_y);
    if (denom < 1e-10) return 0.0;

    return cov / denom;
}

double TemporalAnalyzer::ComputePValue(double correlation, size_t n) {
    if (n < 3) return 1.0;

    // t-statistic for correlation
    double t = correlation * std::sqrt((n - 2) / (1 - correlation * correlation));

    // Approximate p-value using normal distribution for large n
    // For proper implementation, use t-distribution
    double z = std::abs(t);
    double p = std::erfc(z / std::sqrt(2.0));

    return p;
}

std::pair<int, double> TemporalAnalyzer::FindBestLag(
    const std::vector<double>& series_a,
    const std::vector<double>& series_b,
    size_t max_lag) {
    int best_lag = 0;
    double best_corr = 0.0;

    auto cross_corr = ComputeCrossCorrelation(series_a, series_b, max_lag);
    if (!cross_corr.ok()) {
        return {0, 0.0};
    }

    for (const auto& [lag, corr] : *cross_corr) {
        if (std::abs(corr) > std::abs(best_corr)) {
            best_corr = corr;
            best_lag = lag;
        }
    }

    return {best_lag, best_corr};
}

std::optional<TemporalPattern> TemporalAnalyzer::DetectSpike(
    const std::vector<double>& series,
    std::chrono::system_clock::time_point start,
    std::chrono::minutes resolution) {
    if (series.size() < 5) return std::nullopt;

    // Compute mean and std
    double mean = std::accumulate(series.begin(), series.end(), 0.0) / series.size();
    double sq_sum = 0.0;
    for (double v : series) {
        sq_sum += (v - mean) * (v - mean);
    }
    double std_dev = std::sqrt(sq_sum / series.size());

    if (std_dev < 1e-10) return std::nullopt;

    // Find spikes (> 3 std devs)
    double threshold = mean + 3 * std_dev;

    for (size_t i = 0; i < series.size(); ++i) {
        if (series[i] > threshold) {
            TemporalPattern pattern;
            pattern.pattern_type = "spike";
            pattern.start_time = start + std::chrono::minutes(i * resolution.count());
            pattern.end_time = pattern.start_time + resolution;
            pattern.anomaly_score = (series[i] - mean) / std_dev;
            pattern.confidence = 0.9;
            pattern.description = "Event spike detected";
            return pattern;
        }
    }

    return std::nullopt;
}

std::optional<TemporalPattern> TemporalAnalyzer::DetectPeriodicity(
    const std::vector<double>& series,
    std::chrono::system_clock::time_point start,
    std::chrono::minutes resolution) {
    if (series.size() < 20) return std::nullopt;

    // Simple autocorrelation-based periodicity detection
    std::vector<double> autocorr;

    for (size_t lag = 1; lag < series.size() / 2; ++lag) {
        std::vector<double> s1(series.begin(), series.end() - lag);
        std::vector<double> s2(series.begin() + lag, series.end());
        autocorr.push_back(ComputePearsonCorrelation(s1, s2));
    }

    // Find peaks in autocorrelation
    for (size_t i = 1; i < autocorr.size() - 1; ++i) {
        if (autocorr[i] > 0.5 &&
            autocorr[i] > autocorr[i-1] &&
            autocorr[i] > autocorr[i+1]) {
            TemporalPattern pattern;
            pattern.pattern_type = "periodicity";
            pattern.start_time = start;
            pattern.end_time = start + config_.analysis_window;
            pattern.frequency = 1.0 / ((i + 1) * resolution.count());
            pattern.confidence = autocorr[i];
            pattern.description = "Periodic pattern with period ~" +
                std::to_string((i + 1) * resolution.count()) + " minutes";
            return pattern;
        }
    }

    return std::nullopt;
}

double TemporalAnalyzer::ComputeTrendSlope(const std::vector<double>& series) {
    if (series.size() < 2) return 0.0;

    // Simple linear regression
    size_t n = series.size();
    double sum_x = 0, sum_y = 0, sum_xy = 0, sum_xx = 0;

    for (size_t i = 0; i < n; ++i) {
        sum_x += i;
        sum_y += series[i];
        sum_xy += i * series[i];
        sum_xx += i * i;
    }

    double denom = n * sum_xx - sum_x * sum_x;
    if (std::abs(denom) < 1e-10) return 0.0;

    return (n * sum_xy - sum_x * sum_y) / denom;
}

std::vector<size_t> TemporalAnalyzer::DetectAnomalyPoints(
    const std::vector<double>& series,
    double sensitivity) {
    std::vector<size_t> anomalies;

    if (series.size() < 5) return anomalies;

    // Compute mean and std
    double mean = std::accumulate(series.begin(), series.end(), 0.0) / series.size();
    double sq_sum = 0.0;
    for (double v : series) {
        sq_sum += (v - mean) * (v - mean);
    }
    double std_dev = std::sqrt(sq_sum / series.size());

    if (std_dev < 1e-10) return anomalies;

    // Threshold based on sensitivity
    double threshold = mean + (4 - 2 * sensitivity) * std_dev;

    for (size_t i = 0; i < series.size(); ++i) {
        if (series[i] > threshold) {
            anomalies.push_back(i);
        }
    }

    return anomalies;
}

TemporalCorrelation TemporalAnalyzer::BuildCorrelation(
    const std::string& event_type_a,
    const std::string& event_type_b,
    double correlation,
    int lag_minutes,
    double p_value) {
    TemporalCorrelation corr;
    corr.event_type_a = event_type_a;
    corr.event_type_b = event_type_b;
    corr.correlation = correlation;
    corr.lag = std::chrono::minutes(std::abs(lag_minutes));
    corr.p_value = p_value;
    corr.is_significant = p_value < config_.p_value_threshold &&
                          std::abs(correlation) >= config_.min_correlation;

    // Determine direction
    if (!corr.is_significant) {
        corr.direction = TemporalCorrelation::Direction::kNone;
    } else if (std::abs(lag_minutes) < 5) {
        corr.direction = TemporalCorrelation::Direction::kConcurrent;
    } else if (lag_minutes > 0) {
        corr.direction = TemporalCorrelation::Direction::kALeadsB;
    } else {
        corr.direction = TemporalCorrelation::Direction::kBLeadsA;
    }

    // Interpret strength
    corr.strength = InterpretCorrelation(correlation);

    // Build explanation
    std::stringstream ss;
    if (corr.is_significant) {
        ss << event_type_a << " and " << event_type_b << " show ";
        ss << corr.strength << " correlation (r=" << std::fixed << std::setprecision(2)
           << correlation << ")";
        if (corr.direction == TemporalCorrelation::Direction::kALeadsB) {
            ss << ". " << event_type_a << " typically precedes " << event_type_b
               << " by ~" << lag_minutes << " minutes.";
        } else if (corr.direction == TemporalCorrelation::Direction::kBLeadsA) {
            ss << ". " << event_type_b << " typically precedes " << event_type_a
               << " by ~" << std::abs(lag_minutes) << " minutes.";
        } else {
            ss << ". Events tend to occur together.";
        }
    } else {
        ss << "No significant correlation between " << event_type_a
           << " and " << event_type_b;
    }
    corr.explanation = ss.str();

    return corr;
}

std::string TemporalAnalyzer::BuildEventQuery(
    const std::string& model_id,
    const std::string& event_type,
    std::chrono::system_clock::time_point start,
    std::chrono::system_clock::time_point end) {
    auto start_time = std::chrono::system_clock::to_time_t(start);
    auto end_time = std::chrono::system_clock::to_time_t(end);

    std::stringstream ss;
    ss << "SELECT event_id, event_type, model_id, timestamp, severity ";
    ss << "FROM pyflare.events ";
    ss << "WHERE timestamp >= toDateTime(" << start_time << ") ";
    ss << "AND timestamp <= toDateTime(" << end_time << ") ";

    if (!model_id.empty()) {
        ss << "AND model_id = '" << model_id << "' ";
    }
    if (!event_type.empty()) {
        ss << "AND event_type = '" << event_type << "' ";
    }

    ss << "ORDER BY timestamp";

    return ss.str();
}

// ============================================================================
// Utility Functions
// ============================================================================

std::unique_ptr<TemporalAnalyzer> CreateTemporalAnalyzer(
    std::shared_ptr<storage::ClickHouseClient> clickhouse,
    TemporalAnalyzerConfig config) {
    auto analyzer = std::make_unique<TemporalAnalyzer>(
        std::move(clickhouse), std::move(config));
    analyzer->Initialize();
    return analyzer;
}

std::string DirectionToString(TemporalCorrelation::Direction direction) {
    switch (direction) {
        case TemporalCorrelation::Direction::kNone: return "none";
        case TemporalCorrelation::Direction::kALeadsB: return "a_leads_b";
        case TemporalCorrelation::Direction::kBLeadsA: return "b_leads_a";
        case TemporalCorrelation::Direction::kConcurrent: return "concurrent";
    }
    return "unknown";
}

std::string InterpretCorrelation(double correlation) {
    double abs_corr = std::abs(correlation);
    if (abs_corr >= 0.8) return "strong";
    if (abs_corr >= 0.5) return "moderate";
    if (abs_corr >= 0.3) return "weak";
    return "none";
}

}  // namespace pyflare::rca
