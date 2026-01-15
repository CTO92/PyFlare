/// @file client.cpp
/// @brief Qdrant vector database client implementation for PyFlare

#include "storage/qdrant/client.h"

#include <sstream>

#include <nlohmann/json.hpp>
#include <spdlog/spdlog.h>

#ifdef PYFLARE_HAS_HTTPLIB
#include <httplib.h>
#endif

namespace pyflare::storage {

using json = nlohmann::json;

// =============================================================================
// Helper Functions
// =============================================================================

std::string DistanceMetricToString(DistanceMetric metric) {
    switch (metric) {
        case DistanceMetric::kCosine: return "Cosine";
        case DistanceMetric::kEuclidean: return "Euclid";
        case DistanceMetric::kDotProduct: return "Dot";
        default: return "Cosine";
    }
}

// =============================================================================
// QdrantClient Implementation
// =============================================================================

#ifdef PYFLARE_HAS_HTTPLIB

class QdrantClient::Impl {
public:
    explicit Impl(QdrantConfig config) : config_(std::move(config)) {}

    ~Impl() {
        Disconnect();
    }

    absl::Status Connect() {
        if (connected_) {
            return absl::OkStatus();
        }

        try {
            std::string scheme = config_.use_tls ? "https" : "http";
            std::string host = config_.host + ":" + std::to_string(config_.http_port);

            if (config_.use_tls) {
                https_client_ = std::make_unique<httplib::SSLClient>(
                    config_.host, config_.http_port);
                https_client_->set_connection_timeout(config_.connection_timeout);
            } else {
                http_client_ = std::make_unique<httplib::Client>(
                    config_.host, config_.http_port);
                http_client_->set_connection_timeout(config_.connection_timeout);
            }

            // Set API key if provided
            if (!config_.api_key.empty()) {
                httplib::Headers headers = {
                    {"api-key", config_.api_key}
                };
                if (http_client_) {
                    http_client_->set_default_headers(headers);
                }
                if (https_client_) {
                    https_client_->set_default_headers(headers);
                }
            }

            // Test connection by getting collections
            auto result = Get("/collections");
            if (!result.ok()) {
                return result.status();
            }

            connected_ = true;
            spdlog::info("Connected to Qdrant at {}:{}", config_.host, config_.http_port);
            return absl::OkStatus();

        } catch (const std::exception& e) {
            return absl::UnavailableError(
                std::string("Failed to connect to Qdrant: ") + e.what());
        }
    }

    absl::Status Disconnect() {
        if (!connected_) {
            return absl::OkStatus();
        }

        http_client_.reset();
        https_client_.reset();
        connected_ = false;
        spdlog::info("Disconnected from Qdrant");
        return absl::OkStatus();
    }

    bool IsConnected() const {
        return connected_;
    }

    absl::Status CreateCollection(const std::string& name, size_t vector_size,
                                   DistanceMetric metric) {
        if (!IsConnected()) {
            return absl::FailedPreconditionError("Not connected to Qdrant");
        }

        json body = {
            {"vectors", {
                {"size", vector_size},
                {"distance", DistanceMetricToString(metric)}
            }}
        };

        auto result = Put("/collections/" + name, body.dump());
        if (!result.ok()) {
            return result.status();
        }

        spdlog::info("Created collection '{}' with vector size {} and {} distance",
                     name, vector_size, DistanceMetricToString(metric));
        return absl::OkStatus();
    }

    absl::Status DeleteCollection(const std::string& name) {
        if (!IsConnected()) {
            return absl::FailedPreconditionError("Not connected to Qdrant");
        }

        auto result = Delete("/collections/" + name);
        if (!result.ok()) {
            return result.status();
        }

        spdlog::info("Deleted collection '{}'", name);
        return absl::OkStatus();
    }

    absl::StatusOr<bool> CollectionExists(const std::string& name) {
        if (!IsConnected()) {
            return absl::FailedPreconditionError("Not connected to Qdrant");
        }

        auto result = Get("/collections/" + name);
        if (!result.ok()) {
            // 404 means collection doesn't exist
            if (absl::IsNotFound(result.status())) {
                return false;
            }
            return result.status();
        }

        return true;
    }

    absl::Status Upsert(const std::string& collection,
                         const std::vector<VectorPoint>& points) {
        if (!IsConnected()) {
            return absl::FailedPreconditionError("Not connected to Qdrant");
        }

        if (points.empty()) {
            return absl::OkStatus();
        }

        // Build points array
        json points_array = json::array();
        for (const auto& point : points) {
            json point_json = {
                {"id", point.id},
                {"vector", point.vector}
            };

            if (!point.payload.empty()) {
                json payload;
                for (const auto& [key, value] : point.payload) {
                    payload[key] = value;
                }
                point_json["payload"] = payload;
            }

            points_array.push_back(point_json);
        }

        json body = {{"points", points_array}};

        auto result = Put("/collections/" + collection + "/points", body.dump());
        if (!result.ok()) {
            return result.status();
        }

        spdlog::debug("Upserted {} points to collection '{}'", points.size(), collection);
        return absl::OkStatus();
    }

    absl::StatusOr<std::vector<SearchResult>> Search(
        const std::string& collection,
        const std::vector<float>& query_vector,
        size_t limit,
        const std::optional<VectorFilter>& filter) {

        if (!IsConnected()) {
            return absl::FailedPreconditionError("Not connected to Qdrant");
        }

        json body = {
            {"vector", query_vector},
            {"limit", limit},
            {"with_payload", true}
        };

        // Add filter if provided
        if (filter.has_value()) {
            json filter_json;
            const auto& f = filter.value();

            if (f.op == "eq") {
                filter_json = {
                    {"must", {{
                        {"key", f.field},
                        {"match", {{"value", f.value}}}
                    }}}
                };
            } else if (f.op == "in") {
                // Parse comma-separated values
                std::vector<std::string> values;
                std::stringstream ss(f.value);
                std::string item;
                while (std::getline(ss, item, ',')) {
                    values.push_back(item);
                }
                filter_json = {
                    {"must", {{
                        {"key", f.field},
                        {"match", {{"any", values}}}
                    }}}
                };
            }

            if (!filter_json.empty()) {
                body["filter"] = filter_json;
            }
        }

        auto result = Post("/collections/" + collection + "/points/search", body.dump());
        if (!result.ok()) {
            return result.status();
        }

        // Parse response
        std::vector<SearchResult> results;
        try {
            json response = json::parse(*result);
            if (response.contains("result") && response["result"].is_array()) {
                for (const auto& item : response["result"]) {
                    SearchResult sr;
                    sr.id = item.value("id", "");
                    sr.score = item.value("score", 0.0f);

                    if (item.contains("payload") && item["payload"].is_object()) {
                        for (auto& [key, value] : item["payload"].items()) {
                            if (value.is_string()) {
                                sr.payload[key] = value.get<std::string>();
                            } else {
                                sr.payload[key] = value.dump();
                            }
                        }
                    }

                    results.push_back(std::move(sr));
                }
            }
        } catch (const json::exception& e) {
            return absl::InternalError(
                std::string("Failed to parse search response: ") + e.what());
        }

        spdlog::debug("Search in '{}' returned {} results", collection, results.size());
        return results;
    }

    absl::StatusOr<std::vector<std::vector<SearchResult>>> BatchSearch(
        const std::string& collection,
        const std::vector<std::vector<float>>& query_vectors,
        size_t limit) {

        if (!IsConnected()) {
            return absl::FailedPreconditionError("Not connected to Qdrant");
        }

        // Build batch search request
        json searches = json::array();
        for (const auto& query_vector : query_vectors) {
            searches.push_back({
                {"vector", query_vector},
                {"limit", limit},
                {"with_payload", true}
            });
        }

        json body = {{"searches", searches}};

        auto result = Post("/collections/" + collection + "/points/search/batch", body.dump());
        if (!result.ok()) {
            return result.status();
        }

        // Parse response
        std::vector<std::vector<SearchResult>> all_results;
        try {
            json response = json::parse(*result);
            if (response.contains("result") && response["result"].is_array()) {
                for (const auto& batch : response["result"]) {
                    std::vector<SearchResult> batch_results;
                    if (batch.is_array()) {
                        for (const auto& item : batch) {
                            SearchResult sr;
                            sr.id = item.value("id", "");
                            sr.score = item.value("score", 0.0f);

                            if (item.contains("payload") && item["payload"].is_object()) {
                                for (auto& [key, value] : item["payload"].items()) {
                                    if (value.is_string()) {
                                        sr.payload[key] = value.get<std::string>();
                                    } else {
                                        sr.payload[key] = value.dump();
                                    }
                                }
                            }
                            batch_results.push_back(std::move(sr));
                        }
                    }
                    all_results.push_back(std::move(batch_results));
                }
            }
        } catch (const json::exception& e) {
            return absl::InternalError(
                std::string("Failed to parse batch search response: ") + e.what());
        }

        spdlog::debug("Batch search in '{}' processed {} queries",
                      collection, query_vectors.size());
        return all_results;
    }

    absl::Status DeletePoints(const std::string& collection,
                               const std::vector<std::string>& ids) {
        if (!IsConnected()) {
            return absl::FailedPreconditionError("Not connected to Qdrant");
        }

        if (ids.empty()) {
            return absl::OkStatus();
        }

        json body = {
            {"points", ids}
        };

        auto result = Post("/collections/" + collection + "/points/delete", body.dump());
        if (!result.ok()) {
            return result.status();
        }

        spdlog::debug("Deleted {} points from collection '{}'", ids.size(), collection);
        return absl::OkStatus();
    }

private:
    absl::StatusOr<std::string> Get(const std::string& path) {
        httplib::Result res;
        if (https_client_) {
            res = https_client_->Get(path);
        } else if (http_client_) {
            res = http_client_->Get(path);
        } else {
            return absl::FailedPreconditionError("No HTTP client available");
        }

        return HandleResponse(res);
    }

    absl::StatusOr<std::string> Post(const std::string& path, const std::string& body) {
        httplib::Result res;
        if (https_client_) {
            res = https_client_->Post(path, body, "application/json");
        } else if (http_client_) {
            res = http_client_->Post(path, body, "application/json");
        } else {
            return absl::FailedPreconditionError("No HTTP client available");
        }

        return HandleResponse(res);
    }

    absl::StatusOr<std::string> Put(const std::string& path, const std::string& body) {
        httplib::Result res;
        if (https_client_) {
            res = https_client_->Put(path, body, "application/json");
        } else if (http_client_) {
            res = http_client_->Put(path, body, "application/json");
        } else {
            return absl::FailedPreconditionError("No HTTP client available");
        }

        return HandleResponse(res);
    }

    absl::StatusOr<std::string> Delete(const std::string& path) {
        httplib::Result res;
        if (https_client_) {
            res = https_client_->Delete(path);
        } else if (http_client_) {
            res = http_client_->Delete(path);
        } else {
            return absl::FailedPreconditionError("No HTTP client available");
        }

        return HandleResponse(res);
    }

    absl::StatusOr<std::string> HandleResponse(const httplib::Result& res) {
        if (!res) {
            return absl::UnavailableError("Request failed: " +
                                          httplib::to_string(res.error()));
        }

        if (res->status == 404) {
            return absl::NotFoundError("Resource not found");
        }

        if (res->status >= 400) {
            return absl::InternalError("HTTP error " + std::to_string(res->status) +
                                       ": " + res->body);
        }

        return res->body;
    }

    QdrantConfig config_;
    std::unique_ptr<httplib::Client> http_client_;
    std::unique_ptr<httplib::SSLClient> https_client_;
    bool connected_ = false;
};

#else  // !PYFLARE_HAS_HTTPLIB

/// @brief Stub implementation when HTTP library is not available
class QdrantClient::Impl {
public:
    explicit Impl(QdrantConfig config) : config_(std::move(config)) {}

    absl::Status Connect() {
        spdlog::warn("Qdrant HTTP support not compiled in");
        connected_ = true;  // Pretend to be connected for testing
        return absl::OkStatus();
    }

    absl::Status Disconnect() {
        connected_ = false;
        return absl::OkStatus();
    }

    bool IsConnected() const { return connected_; }

    absl::Status CreateCollection(const std::string& name, size_t vector_size,
                                   DistanceMetric) {
        spdlog::debug("Mock creating collection '{}' with size {}", name, vector_size);
        return absl::OkStatus();
    }

    absl::Status DeleteCollection(const std::string&) {
        return absl::OkStatus();
    }

    absl::StatusOr<bool> CollectionExists(const std::string&) {
        return false;
    }

    absl::Status Upsert(const std::string& collection,
                         const std::vector<VectorPoint>& points) {
        spdlog::debug("Mock upserting {} points to '{}'", points.size(), collection);
        return absl::OkStatus();
    }

    absl::StatusOr<std::vector<SearchResult>> Search(
        const std::string&, const std::vector<float>&, size_t,
        const std::optional<VectorFilter>&) {
        return std::vector<SearchResult>{};
    }

    absl::StatusOr<std::vector<std::vector<SearchResult>>> BatchSearch(
        const std::string&,
        const std::vector<std::vector<float>>& query_vectors,
        size_t) {
        return std::vector<std::vector<SearchResult>>(query_vectors.size());
    }

    absl::Status DeletePoints(const std::string&, const std::vector<std::string>&) {
        return absl::OkStatus();
    }

private:
    QdrantConfig config_;
    bool connected_ = false;
};

#endif  // PYFLARE_HAS_HTTPLIB

// =============================================================================
// QdrantClient Public Interface
// =============================================================================

QdrantClient::QdrantClient(QdrantConfig config)
    : config_(std::move(config)), impl_(std::make_unique<Impl>(config_)) {}

QdrantClient::~QdrantClient() = default;

QdrantClient::QdrantClient(QdrantClient&&) noexcept = default;
QdrantClient& QdrantClient::operator=(QdrantClient&&) noexcept = default;

absl::Status QdrantClient::Connect() {
    return impl_->Connect();
}

absl::Status QdrantClient::Disconnect() {
    return impl_->Disconnect();
}

bool QdrantClient::IsConnected() const {
    return impl_->IsConnected();
}

absl::Status QdrantClient::CreateCollection(
    const std::string& name,
    size_t vector_size,
    DistanceMetric metric) {
    return impl_->CreateCollection(name, vector_size, metric);
}

absl::Status QdrantClient::DeleteCollection(const std::string& name) {
    return impl_->DeleteCollection(name);
}

absl::StatusOr<bool> QdrantClient::CollectionExists(const std::string& name) {
    return impl_->CollectionExists(name);
}

absl::Status QdrantClient::Upsert(
    const std::string& collection,
    const std::vector<VectorPoint>& points) {
    return impl_->Upsert(collection, points);
}

absl::StatusOr<std::vector<SearchResult>> QdrantClient::Search(
    const std::string& collection,
    const std::vector<float>& query_vector,
    size_t limit,
    const std::optional<VectorFilter>& filter) {
    return impl_->Search(collection, query_vector, limit, filter);
}

absl::StatusOr<std::vector<std::vector<SearchResult>>> QdrantClient::BatchSearch(
    const std::string& collection,
    const std::vector<std::vector<float>>& query_vectors,
    size_t limit) {
    return impl_->BatchSearch(collection, query_vectors, limit);
}

absl::Status QdrantClient::Delete(
    const std::string& collection,
    const std::vector<std::string>& ids) {
    return impl_->DeletePoints(collection, ids);
}

}  // namespace pyflare::storage
