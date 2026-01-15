#pragma once

/// @file reference_store.h
/// @brief Reference distribution storage for drift detection

#include <chrono>
#include <memory>
#include <optional>
#include <string>
#include <vector>

#include <absl/status/status.h>
#include <absl/status/statusor.h>

#include "storage/qdrant/client.h"
#include "storage/redis/client.h"

namespace pyflare::drift {

/// @brief Configuration for reference store
struct ReferenceStoreConfig {
    /// Qdrant collection name for embeddings
    std::string embeddings_collection = "pyflare_reference_embeddings";

    /// Vector dimension (must match model output)
    size_t vector_dimension = 1536;

    /// Maximum number of reference embeddings to store per model
    size_t max_embeddings_per_model = 10000;

    /// TTL for cached references in Redis
    std::chrono::seconds cache_ttl = std::chrono::hours(24);

    /// Whether to use sliding window updates
    bool use_sliding_window = true;

    /// Sliding window size (number of recent embeddings to keep)
    size_t sliding_window_size = 5000;
};

/// @brief Metadata for a reference distribution
struct ReferenceMetadata {
    std::string model_id;
    std::string feature_name;  // Empty for embeddings
    std::string reference_type;  // "embedding", "feature", "categorical"

    std::chrono::system_clock::time_point created_at;
    std::chrono::system_clock::time_point updated_at;

    size_t sample_count = 0;
    std::string data_source;  // "training", "production", "validation"

    std::vector<float> centroid;  // For embeddings
    double mean = 0.0;            // For features
    double std_dev = 0.0;
};

/// @brief Reference store for drift detection baselines
///
/// Stores and manages reference distributions for drift detection:
/// - Embedding vectors in Qdrant for semantic similarity
/// - Distribution statistics in Redis for fast access
/// - Feature distributions for PSI/KS tests
///
/// Example usage:
/// @code
///   ReferenceStoreConfig config;
///   auto store = std::make_unique<ReferenceStore>(qdrant, redis, config);
///
///   // Store training embeddings as reference
///   store->StoreEmbeddings("gpt-4", training_embeddings);
///
///   // Get reference for drift detection
///   auto ref = store->GetEmbeddings("gpt-4");
/// @endcode
class ReferenceStore {
public:
    ReferenceStore(std::shared_ptr<storage::QdrantClient> qdrant,
                   std::shared_ptr<storage::RedisClient> redis,
                   ReferenceStoreConfig config = {});
    ~ReferenceStore();

    // Disable copy
    ReferenceStore(const ReferenceStore&) = delete;
    ReferenceStore& operator=(const ReferenceStore&) = delete;

    /// @brief Initialize the store (create collections if needed)
    absl::Status Initialize();

    // =========================================================================
    // Embedding Reference Operations
    // =========================================================================

    /// @brief Store reference embeddings for a model
    /// @param model_id Model identifier
    /// @param embeddings Reference embedding vectors
    /// @param metadata Optional metadata about the reference
    absl::Status StoreEmbeddings(
        const std::string& model_id,
        const std::vector<std::vector<float>>& embeddings,
        const std::optional<ReferenceMetadata>& metadata = std::nullopt);

    /// @brief Get reference embeddings for a model
    /// @param model_id Model identifier
    /// @param limit Maximum number of embeddings to return (0 = all)
    absl::StatusOr<std::vector<std::vector<float>>> GetEmbeddings(
        const std::string& model_id,
        size_t limit = 0);

    /// @brief Update reference with new embeddings (sliding window)
    /// @param model_id Model identifier
    /// @param new_embeddings New embedding vectors to add
    absl::Status UpdateEmbeddings(
        const std::string& model_id,
        const std::vector<std::vector<float>>& new_embeddings);

    /// @brief Get the reference centroid for a model
    /// @param model_id Model identifier
    absl::StatusOr<std::vector<float>> GetCentroid(const std::string& model_id);

    /// @brief Store/update the reference centroid
    /// @param model_id Model identifier
    /// @param centroid Centroid vector
    absl::Status StoreCentroid(const std::string& model_id,
                                const std::vector<float>& centroid);

    // =========================================================================
    // Feature Reference Operations
    // =========================================================================

    /// @brief Store reference feature distribution
    /// @param model_id Model identifier
    /// @param feature_name Feature name
    /// @param values Feature values
    absl::Status StoreFeatureDistribution(
        const std::string& model_id,
        const std::string& feature_name,
        const std::vector<double>& values);

    /// @brief Get reference feature distribution
    /// @param model_id Model identifier
    /// @param feature_name Feature name
    absl::StatusOr<std::vector<double>> GetFeatureDistribution(
        const std::string& model_id,
        const std::string& feature_name);

    /// @brief Store bin edges and percentages for PSI
    absl::Status StorePSIReference(
        const std::string& model_id,
        const std::string& feature_name,
        const std::vector<double>& bin_edges,
        const std::vector<double>& bin_percentages);

    /// @brief Get PSI reference
    absl::StatusOr<std::pair<std::vector<double>, std::vector<double>>>
    GetPSIReference(const std::string& model_id, const std::string& feature_name);

    // =========================================================================
    // Metadata Operations
    // =========================================================================

    /// @brief Get metadata for a reference
    absl::StatusOr<ReferenceMetadata> GetMetadata(
        const std::string& model_id,
        const std::string& feature_name = "");

    /// @brief Check if reference exists
    absl::StatusOr<bool> HasReference(
        const std::string& model_id,
        const std::string& feature_name = "");

    /// @brief Delete reference for a model
    absl::Status DeleteReference(
        const std::string& model_id,
        const std::string& feature_name = "");

    /// @brief List all models with references
    absl::StatusOr<std::vector<std::string>> ListModels();

    /// @brief Get configuration
    const ReferenceStoreConfig& GetConfig() const { return config_; }

private:
    /// @brief Build Qdrant point ID from model and index
    std::string BuildPointId(const std::string& model_id, size_t index) const;

    /// @brief Build Redis key for model reference
    std::string BuildRedisKey(const std::string& model_id,
                               const std::string& suffix) const;

    std::shared_ptr<storage::QdrantClient> qdrant_;
    std::shared_ptr<storage::RedisClient> redis_;
    ReferenceStoreConfig config_;
    bool initialized_ = false;
};

}  // namespace pyflare::drift
