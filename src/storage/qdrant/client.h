#pragma once

/// @file client.h
/// @brief Qdrant vector database client wrapper for PyFlare

#include <memory>
#include <optional>
#include <string>
#include <unordered_map>
#include <vector>

#include <absl/status/status.h>
#include <absl/status/statusor.h>

namespace pyflare::storage {

/// @brief Distance metric for vector similarity
enum class DistanceMetric {
    kCosine,
    kEuclidean,
    kDotProduct
};

/// @brief A point with vector and payload
struct VectorPoint {
    std::string id;
    std::vector<float> vector;
    std::unordered_map<std::string, std::string> payload;
};

/// @brief Search result from vector query
struct SearchResult {
    std::string id;
    float score;
    std::unordered_map<std::string, std::string> payload;
};

/// @brief Filter for vector search
struct VectorFilter {
    std::string field;
    std::string op;  // "eq", "ne", "gt", "gte", "lt", "lte", "in"
    std::string value;
};

/// @brief Qdrant client configuration
struct QdrantConfig {
    std::string host = "localhost";
    uint16_t grpc_port = 6334;
    uint16_t http_port = 6333;
    std::string api_key;
    bool use_tls = false;

    std::chrono::seconds connection_timeout{30};
};

/// @brief Qdrant client for PyFlare
class QdrantClient {
public:
    /// @brief Create a client with the given configuration
    explicit QdrantClient(QdrantConfig config);

    /// @brief Destructor
    ~QdrantClient();

    // Non-copyable
    QdrantClient(const QdrantClient&) = delete;
    QdrantClient& operator=(const QdrantClient&) = delete;

    // Movable
    QdrantClient(QdrantClient&&) noexcept;
    QdrantClient& operator=(QdrantClient&&) noexcept;

    /// @brief Connect to Qdrant
    absl::Status Connect();

    /// @brief Disconnect from Qdrant
    absl::Status Disconnect();

    /// @brief Check if connected
    bool IsConnected() const;

    /// @brief Create a new collection
    /// @param name Collection name
    /// @param vector_size Dimensionality of vectors
    /// @param metric Distance metric to use
    absl::Status CreateCollection(
        const std::string& name,
        size_t vector_size,
        DistanceMetric metric = DistanceMetric::kCosine);

    /// @brief Delete a collection
    absl::Status DeleteCollection(const std::string& name);

    /// @brief Check if a collection exists
    absl::StatusOr<bool> CollectionExists(const std::string& name);

    /// @brief Upsert vectors into a collection
    absl::Status Upsert(
        const std::string& collection,
        const std::vector<VectorPoint>& points);

    /// @brief Search for similar vectors
    /// @param collection Collection name
    /// @param query_vector Query vector
    /// @param limit Maximum number of results
    /// @param filter Optional filter to apply
    absl::StatusOr<std::vector<SearchResult>> Search(
        const std::string& collection,
        const std::vector<float>& query_vector,
        size_t limit,
        const std::optional<VectorFilter>& filter = std::nullopt);

    /// @brief Batch search for multiple query vectors
    absl::StatusOr<std::vector<std::vector<SearchResult>>> BatchSearch(
        const std::string& collection,
        const std::vector<std::vector<float>>& query_vectors,
        size_t limit);

    /// @brief Delete points by IDs
    absl::Status Delete(
        const std::string& collection,
        const std::vector<std::string>& ids);

    /// @brief Get the configuration
    const QdrantConfig& GetConfig() const { return config_; }

private:
    QdrantConfig config_;
    class Impl;
    std::unique_ptr<Impl> impl_;
};

}  // namespace pyflare::storage
