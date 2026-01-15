#pragma once

/// @file api.h
/// @brief PyFlare Query API server

#include <chrono>
#include <functional>
#include <memory>
#include <optional>
#include <string>
#include <vector>

#include <absl/status/status.h>
#include <absl/status/statusor.h>

#include "storage/clickhouse/client.h"
#include "storage/qdrant/client.h"

namespace pyflare::query {

/// @brief Query parameter
struct QueryParam {
    std::string name;
    std::string value;
    std::string type;
};

/// @brief Query request
struct QueryRequest {
    std::string sql;
    std::vector<QueryParam> params;
    std::optional<size_t> limit;
    std::optional<size_t> offset;
    std::optional<std::string> format;  ///< "json", "csv", "arrow"
};

/// @brief Query response
struct QueryResponse {
    std::vector<std::string> columns;
    std::vector<std::vector<std::string>> rows;
    size_t total_rows = 0;
    std::chrono::milliseconds execution_time{0};
};

/// @brief Query API configuration
struct QueryAPIConfig {
    std::string listen_address = "0.0.0.0";
    uint16_t port = 8080;
    size_t max_query_size = 1024 * 1024;  ///< 1 MB
    std::chrono::seconds query_timeout{30};
    size_t max_result_rows = 10000;
    size_t max_connections = 100;
};

/// @brief Query API server
class QueryAPI {
public:
    /// @brief Create the API with given configuration and storage clients
    QueryAPI(
        QueryAPIConfig config,
        std::shared_ptr<storage::ClickHouseClient> clickhouse,
        std::shared_ptr<storage::QdrantClient> qdrant);

    ~QueryAPI();

    // Non-copyable
    QueryAPI(const QueryAPI&) = delete;
    QueryAPI& operator=(const QueryAPI&) = delete;

    /// @brief Start the API server
    absl::Status Start();

    /// @brief Stop the API server
    absl::Status Stop();

    /// @brief Check if the server is running
    bool IsRunning() const;

    /// @brief Execute a query directly
    absl::StatusOr<QueryResponse> ExecuteQuery(const QueryRequest& request);

    /// @brief Get the server address
    std::string GetAddress() const;

private:
    QueryAPIConfig config_;
    std::shared_ptr<storage::ClickHouseClient> clickhouse_;
    std::shared_ptr<storage::QdrantClient> qdrant_;
    class Impl;
    std::unique_ptr<Impl> impl_;
};

}  // namespace pyflare::query
