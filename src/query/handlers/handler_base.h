#pragma once

/// @file handler_base.h
/// @brief Base handler interface for Query API

#include <memory>
#include <string>
#include <unordered_map>
#include <vector>

#include <absl/status/statusor.h>
#include <nlohmann/json.hpp>

#include "storage/clickhouse/client.h"
#include "storage/qdrant/client.h"
#include "storage/redis/client.h"

namespace pyflare::query::handlers {

/// @brief HTTP method enum
enum class HttpMethod {
    kGet,
    kPost,
    kPut,
    kDelete,
    kPatch
};

/// @brief HTTP request
struct HttpRequest {
    HttpMethod method = HttpMethod::kGet;
    std::string path;
    std::unordered_map<std::string, std::string> query_params;
    std::unordered_map<std::string, std::string> headers;
    std::string body;
    std::string path_params;  ///< Extracted path parameters like :id
};

/// @brief HTTP response
struct HttpResponse {
    int status_code = 200;
    std::unordered_map<std::string, std::string> headers;
    std::string body;
    std::string content_type = "application/json";

    static HttpResponse Ok(const nlohmann::json& body) {
        HttpResponse resp;
        resp.status_code = 200;
        resp.body = body.dump();
        return resp;
    }

    static HttpResponse Created(const nlohmann::json& body) {
        HttpResponse resp;
        resp.status_code = 201;
        resp.body = body.dump();
        return resp;
    }

    static HttpResponse BadRequest(const std::string& message) {
        HttpResponse resp;
        resp.status_code = 400;
        resp.body = nlohmann::json{{"error", message}}.dump();
        return resp;
    }

    static HttpResponse NotFound(const std::string& message = "Not found") {
        HttpResponse resp;
        resp.status_code = 404;
        resp.body = nlohmann::json{{"error", message}}.dump();
        return resp;
    }

    static HttpResponse InternalError(const std::string& message) {
        HttpResponse resp;
        resp.status_code = 500;
        resp.body = nlohmann::json{{"error", message}}.dump();
        return resp;
    }
};

/// @brief Pagination parameters
struct PaginationParams {
    size_t page = 1;
    size_t page_size = 50;
    size_t offset() const { return (page - 1) * page_size; }
};

/// @brief Time range filter
struct TimeRangeFilter {
    std::string start;  ///< ISO 8601 format
    std::string end;
    std::string interval;  ///< "1h", "1d", etc.
};

/// @brief Handler context with shared resources
struct HandlerContext {
    std::shared_ptr<storage::ClickHouseClient> clickhouse;
    std::shared_ptr<storage::QdrantClient> qdrant;
    std::shared_ptr<storage::RedisClient> redis;
};

/// @brief Base handler interface
class Handler {
public:
    virtual ~Handler() = default;

    /// @brief Handle the request
    virtual HttpResponse Handle(const HttpRequest& request,
                                 const HandlerContext& context) = 0;

    /// @brief Get the route pattern (e.g., "/api/v1/traces/:id")
    virtual std::string GetRoute() const = 0;

    /// @brief Get supported HTTP methods
    virtual std::vector<HttpMethod> GetMethods() const = 0;

protected:
    /// @brief Extract pagination from query params
    PaginationParams ExtractPagination(const HttpRequest& request) {
        PaginationParams params;

        auto it = request.query_params.find("page");
        if (it != request.query_params.end()) {
            try {
                params.page = std::max(size_t{1}, std::stoull(it->second));
            } catch (...) {}
        }

        it = request.query_params.find("page_size");
        if (it != request.query_params.end()) {
            try {
                params.page_size = std::min(size_t{1000},
                    std::max(size_t{1}, std::stoull(it->second)));
            } catch (...) {}
        }

        return params;
    }

    /// @brief Extract time range from query params
    TimeRangeFilter ExtractTimeRange(const HttpRequest& request) {
        TimeRangeFilter filter;

        auto it = request.query_params.find("start");
        if (it != request.query_params.end()) {
            filter.start = it->second;
        }

        it = request.query_params.find("end");
        if (it != request.query_params.end()) {
            filter.end = it->second;
        }

        it = request.query_params.find("interval");
        if (it != request.query_params.end()) {
            filter.interval = it->second;
        }

        return filter;
    }

    /// @brief Parse JSON body safely
    absl::StatusOr<nlohmann::json> ParseJsonBody(const HttpRequest& request) {
        try {
            return nlohmann::json::parse(request.body);
        } catch (const nlohmann::json::exception& e) {
            return absl::InvalidArgumentError(
                std::string("Invalid JSON: ") + e.what());
        }
    }
};

/// @brief Register all handlers with the router
class HandlerRegistry {
public:
    void Register(std::unique_ptr<Handler> handler) {
        handlers_.push_back(std::move(handler));
    }

    const std::vector<std::unique_ptr<Handler>>& GetHandlers() const {
        return handlers_;
    }

private:
    std::vector<std::unique_ptr<Handler>> handlers_;
};

}  // namespace pyflare::query::handlers
