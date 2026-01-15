#pragma once

/// @file client.h
/// @brief ClickHouse client wrapper for PyFlare

#include <chrono>
#include <memory>
#include <optional>
#include <string>
#include <vector>

#include <absl/status/status.h>
#include <absl/status/statusor.h>

namespace pyflare::storage {

/// @brief Query parameter for parameterized queries
struct QueryParam {
    std::string name;
    std::string value;
    std::string type;  // "String", "Int64", "Float64", etc.
};

/// @brief Result of a query execution
struct QueryResult {
    std::vector<std::string> columns;
    std::vector<std::string> column_types;
    std::vector<std::vector<std::string>> rows;
    size_t total_rows = 0;
    std::chrono::milliseconds execution_time{0};
};

/// @brief Trace record for storage
struct TraceRecord {
    std::string trace_id;
    std::string span_id;
    std::optional<std::string> parent_span_id;

    std::chrono::system_clock::time_point start_time;
    std::chrono::system_clock::time_point end_time;
    uint64_t duration_ns = 0;

    std::string service_name;
    std::string operation_name;
    std::string span_kind;
    std::string status_code;
    std::optional<std::string> status_message;

    std::string model_id;
    std::string model_version;
    std::string inference_type;

    std::string input_preview;
    std::string output_preview;

    std::optional<uint32_t> input_tokens;
    std::optional<uint32_t> output_tokens;
    std::optional<uint64_t> cost_micros;

    std::unordered_map<std::string, std::string> attributes;
    std::unordered_map<std::string, std::string> resource;
};

/// @brief Metric record for storage
struct MetricRecord {
    std::string metric_name;
    std::string service_name;
    std::string model_id;
    std::string model_version;
    std::string environment;

    std::chrono::system_clock::time_point timestamp;

    std::string value_type;  // "gauge", "counter", "histogram"
    double value = 0.0;

    std::unordered_map<std::string, std::string> attributes;
};

/// @brief ClickHouse client configuration
struct ClickHouseConfig {
    std::string host = "localhost";
    uint16_t port = 9000;
    std::string database = "pyflare";
    std::string user = "default";
    std::string password;

    size_t max_connections = 10;
    std::chrono::seconds connection_timeout{30};
    std::chrono::seconds query_timeout{60};

    bool use_compression = true;
};

/// @brief ClickHouse client for PyFlare
class ClickHouseClient {
public:
    /// @brief Create a client with the given configuration
    explicit ClickHouseClient(ClickHouseConfig config);

    /// @brief Destructor
    ~ClickHouseClient();

    // Non-copyable
    ClickHouseClient(const ClickHouseClient&) = delete;
    ClickHouseClient& operator=(const ClickHouseClient&) = delete;

    // Movable
    ClickHouseClient(ClickHouseClient&&) noexcept;
    ClickHouseClient& operator=(ClickHouseClient&&) noexcept;

    /// @brief Connect to ClickHouse
    absl::Status Connect();

    /// @brief Disconnect from ClickHouse
    absl::Status Disconnect();

    /// @brief Check if connected
    bool IsConnected() const;

    /// @brief Execute a query and return results
    absl::StatusOr<QueryResult> Execute(const std::string& sql);

    /// @brief Execute a parameterized query
    absl::StatusOr<QueryResult> ExecuteWithParams(
        const std::string& sql,
        const std::vector<QueryParam>& params);

    /// @brief Insert trace records
    absl::Status InsertTraces(const std::vector<TraceRecord>& traces);

    /// @brief Insert metric records
    absl::Status InsertMetrics(const std::vector<MetricRecord>& metrics);

    /// @brief Run schema migrations
    absl::Status RunMigrations();

    /// @brief Get the configuration
    const ClickHouseConfig& GetConfig() const { return config_; }

private:
    ClickHouseConfig config_;
    class Impl;
    std::unique_ptr<Impl> impl_;
};

}  // namespace pyflare::storage
