/// @file reference_store.cpp
/// @brief Reference distribution storage implementation

#include "processor/drift/reference_store.h"

#include <algorithm>
#include <numeric>
#include <sstream>

#include <nlohmann/json.hpp>
#include <spdlog/spdlog.h>

namespace pyflare::drift {

using json = nlohmann::json;

// =============================================================================
// ReferenceStore Implementation
// =============================================================================

ReferenceStore::ReferenceStore(std::shared_ptr<storage::QdrantClient> qdrant,
                               std::shared_ptr<storage::RedisClient> redis,
                               ReferenceStoreConfig config)
    : qdrant_(std::move(qdrant)),
      redis_(std::move(redis)),
      config_(std::move(config)) {}

ReferenceStore::~ReferenceStore() = default;

absl::Status ReferenceStore::Initialize() {
    if (initialized_) {
        return absl::OkStatus();
    }

    // Create Qdrant collection for embeddings if it doesn't exist
    auto status = qdrant_->CreateCollection(
        config_.embeddings_collection,
        config_.vector_dimension);

    if (!status.ok() && !absl::IsAlreadyExists(status)) {
        return status;
    }

    initialized_ = true;
    spdlog::info("ReferenceStore initialized with collection: {}",
                 config_.embeddings_collection);
    return absl::OkStatus();
}

// =============================================================================
// Embedding Reference Operations
// =============================================================================

absl::Status ReferenceStore::StoreEmbeddings(
    const std::string& model_id,
    const std::vector<std::vector<float>>& embeddings,
    const std::optional<ReferenceMetadata>& metadata) {

    if (!initialized_) {
        return absl::FailedPreconditionError("ReferenceStore not initialized");
    }

    if (embeddings.empty()) {
        return absl::InvalidArgumentError("Embeddings vector is empty");
    }

    // Validate embedding dimensions
    for (const auto& emb : embeddings) {
        if (emb.size() != config_.vector_dimension) {
            return absl::InvalidArgumentError(
                "Embedding dimension mismatch: expected " +
                std::to_string(config_.vector_dimension) +
                ", got " + std::to_string(emb.size()));
        }
    }

    // Limit embeddings if needed
    size_t num_to_store = std::min(embeddings.size(),
                                    config_.max_embeddings_per_model);

    // Build points for Qdrant
    std::vector<storage::QdrantClient::Point> points;
    points.reserve(num_to_store);

    for (size_t i = 0; i < num_to_store; ++i) {
        storage::QdrantClient::Point point;
        point.id = BuildPointId(model_id, i);
        point.vector = embeddings[i];
        point.payload["model_id"] = model_id;
        point.payload["index"] = static_cast<int64_t>(i);
        point.payload["timestamp"] = std::chrono::duration_cast<std::chrono::seconds>(
            std::chrono::system_clock::now().time_since_epoch()).count();

        if (metadata) {
            point.payload["data_source"] = metadata->data_source;
            point.payload["reference_type"] = metadata->reference_type;
        }

        points.push_back(std::move(point));
    }

    // Upsert to Qdrant
    auto status = qdrant_->Upsert(config_.embeddings_collection, points);
    if (!status.ok()) {
        return status;
    }

    // Compute and store centroid
    std::vector<float> centroid(config_.vector_dimension, 0.0f);
    for (size_t i = 0; i < num_to_store; ++i) {
        for (size_t j = 0; j < config_.vector_dimension; ++j) {
            centroid[j] += embeddings[i][j];
        }
    }
    for (size_t j = 0; j < config_.vector_dimension; ++j) {
        centroid[j] /= static_cast<float>(num_to_store);
    }

    status = StoreCentroid(model_id, centroid);
    if (!status.ok()) {
        spdlog::warn("Failed to store centroid: {}", status.message());
    }

    // Store metadata in Redis
    ReferenceMetadata meta = metadata.value_or(ReferenceMetadata{});
    meta.model_id = model_id;
    meta.reference_type = "embedding";
    meta.sample_count = num_to_store;
    meta.centroid = centroid;
    meta.updated_at = std::chrono::system_clock::now();
    if (meta.created_at == std::chrono::system_clock::time_point{}) {
        meta.created_at = meta.updated_at;
    }

    json meta_json;
    meta_json["model_id"] = meta.model_id;
    meta_json["feature_name"] = meta.feature_name;
    meta_json["reference_type"] = meta.reference_type;
    meta_json["sample_count"] = meta.sample_count;
    meta_json["data_source"] = meta.data_source;
    meta_json["created_at"] = std::chrono::duration_cast<std::chrono::seconds>(
        meta.created_at.time_since_epoch()).count();
    meta_json["updated_at"] = std::chrono::duration_cast<std::chrono::seconds>(
        meta.updated_at.time_since_epoch()).count();

    auto redis_key = BuildRedisKey(model_id, "metadata");
    status = redis_->Set(redis_key, meta_json.dump(), config_.cache_ttl);
    if (!status.ok()) {
        spdlog::warn("Failed to store metadata in Redis: {}", status.message());
    }

    spdlog::info("Stored {} reference embeddings for model {}", num_to_store, model_id);
    return absl::OkStatus();
}

absl::StatusOr<std::vector<std::vector<float>>> ReferenceStore::GetEmbeddings(
    const std::string& model_id,
    size_t limit) {

    if (!initialized_) {
        return absl::FailedPreconditionError("ReferenceStore not initialized");
    }

    // Search for all embeddings with this model_id
    storage::QdrantClient::SearchRequest request;
    request.vector.resize(config_.vector_dimension, 0.0f);  // Dummy vector
    request.limit = limit > 0 ? limit : config_.max_embeddings_per_model;
    request.with_vector = true;
    request.filter["model_id"] = model_id;

    // Use scroll instead of search to get all vectors
    auto result = qdrant_->Scroll(config_.embeddings_collection, model_id, limit);
    if (!result.ok()) {
        return result.status();
    }

    std::vector<std::vector<float>> embeddings;
    for (const auto& point : *result) {
        embeddings.push_back(point.vector);
    }

    return embeddings;
}

absl::Status ReferenceStore::UpdateEmbeddings(
    const std::string& model_id,
    const std::vector<std::vector<float>>& new_embeddings) {

    if (!config_.use_sliding_window) {
        // Simply replace all embeddings
        return StoreEmbeddings(model_id, new_embeddings, std::nullopt);
    }

    // Get existing embeddings
    auto existing_result = GetEmbeddings(model_id, 0);
    if (!existing_result.ok()) {
        // If no existing, just store new ones
        if (absl::IsNotFound(existing_result.status())) {
            return StoreEmbeddings(model_id, new_embeddings, std::nullopt);
        }
        return existing_result.status();
    }

    auto existing = std::move(*existing_result);

    // Combine existing and new, keeping sliding window size
    std::vector<std::vector<float>> combined;
    combined.reserve(existing.size() + new_embeddings.size());

    // Add new embeddings first (most recent)
    combined.insert(combined.end(), new_embeddings.begin(), new_embeddings.end());

    // Add existing up to window size
    size_t remaining = config_.sliding_window_size > combined.size()
                       ? config_.sliding_window_size - combined.size()
                       : 0;
    if (remaining > 0 && !existing.empty()) {
        size_t to_keep = std::min(remaining, existing.size());
        combined.insert(combined.end(), existing.begin(), existing.begin() + to_keep);
    }

    // Limit to window size
    if (combined.size() > config_.sliding_window_size) {
        combined.resize(config_.sliding_window_size);
    }

    // Delete old and store new
    auto delete_status = DeleteReference(model_id, "");
    if (!delete_status.ok() && !absl::IsNotFound(delete_status)) {
        spdlog::warn("Failed to delete old embeddings: {}", delete_status.message());
    }

    return StoreEmbeddings(model_id, combined, std::nullopt);
}

absl::StatusOr<std::vector<float>> ReferenceStore::GetCentroid(
    const std::string& model_id) {

    // Try Redis cache first
    auto redis_key = BuildRedisKey(model_id, "centroid");
    auto cached = redis_->Get(redis_key);
    if (cached.ok()) {
        try {
            json j = json::parse(*cached);
            return j.get<std::vector<float>>();
        } catch (const json::exception& e) {
            spdlog::warn("Failed to parse cached centroid: {}", e.what());
        }
    }

    // Compute from embeddings if not cached
    auto embeddings = GetEmbeddings(model_id, 0);
    if (!embeddings.ok()) {
        return embeddings.status();
    }

    if (embeddings->empty()) {
        return absl::NotFoundError("No embeddings found for model: " + model_id);
    }

    // Compute centroid
    std::vector<float> centroid(config_.vector_dimension, 0.0f);
    for (const auto& emb : *embeddings) {
        for (size_t j = 0; j < config_.vector_dimension && j < emb.size(); ++j) {
            centroid[j] += emb[j];
        }
    }
    float n = static_cast<float>(embeddings->size());
    for (size_t j = 0; j < config_.vector_dimension; ++j) {
        centroid[j] /= n;
    }

    // Cache it
    auto status = StoreCentroid(model_id, centroid);
    if (!status.ok()) {
        spdlog::warn("Failed to cache centroid: {}", status.message());
    }

    return centroid;
}

absl::Status ReferenceStore::StoreCentroid(const std::string& model_id,
                                            const std::vector<float>& centroid) {
    auto redis_key = BuildRedisKey(model_id, "centroid");
    json j = centroid;
    return redis_->Set(redis_key, j.dump(), config_.cache_ttl);
}

// =============================================================================
// Feature Reference Operations
// =============================================================================

absl::Status ReferenceStore::StoreFeatureDistribution(
    const std::string& model_id,
    const std::string& feature_name,
    const std::vector<double>& values) {

    if (values.empty()) {
        return absl::InvalidArgumentError("Feature values vector is empty");
    }

    // Compute statistics
    double sum = std::accumulate(values.begin(), values.end(), 0.0);
    double mean = sum / values.size();

    double sq_sum = 0.0;
    for (double v : values) {
        sq_sum += (v - mean) * (v - mean);
    }
    double std_dev = std::sqrt(sq_sum / values.size());

    // Store distribution in Redis
    json dist;
    dist["values"] = values;
    dist["mean"] = mean;
    dist["std_dev"] = std_dev;
    dist["sample_count"] = values.size();
    dist["min"] = *std::min_element(values.begin(), values.end());
    dist["max"] = *std::max_element(values.begin(), values.end());

    auto redis_key = BuildRedisKey(model_id, "feature:" + feature_name);
    auto status = redis_->Set(redis_key, dist.dump(), config_.cache_ttl);
    if (!status.ok()) {
        return status;
    }

    // Update metadata
    ReferenceMetadata meta;
    meta.model_id = model_id;
    meta.feature_name = feature_name;
    meta.reference_type = "feature";
    meta.sample_count = values.size();
    meta.mean = mean;
    meta.std_dev = std_dev;
    meta.updated_at = std::chrono::system_clock::now();

    json meta_json;
    meta_json["model_id"] = meta.model_id;
    meta_json["feature_name"] = meta.feature_name;
    meta_json["reference_type"] = meta.reference_type;
    meta_json["sample_count"] = meta.sample_count;
    meta_json["mean"] = meta.mean;
    meta_json["std_dev"] = meta.std_dev;

    auto meta_key = BuildRedisKey(model_id, "metadata:" + feature_name);
    return redis_->Set(meta_key, meta_json.dump(), config_.cache_ttl);
}

absl::StatusOr<std::vector<double>> ReferenceStore::GetFeatureDistribution(
    const std::string& model_id,
    const std::string& feature_name) {

    auto redis_key = BuildRedisKey(model_id, "feature:" + feature_name);
    auto result = redis_->Get(redis_key);
    if (!result.ok()) {
        return result.status();
    }

    try {
        json j = json::parse(*result);
        return j["values"].get<std::vector<double>>();
    } catch (const json::exception& e) {
        return absl::InvalidArgumentError(
            std::string("Failed to parse feature distribution: ") + e.what());
    }
}

absl::Status ReferenceStore::StorePSIReference(
    const std::string& model_id,
    const std::string& feature_name,
    const std::vector<double>& bin_edges,
    const std::vector<double>& bin_percentages) {

    if (bin_edges.size() < 2 || bin_percentages.empty()) {
        return absl::InvalidArgumentError("Invalid bin data");
    }

    json psi_ref;
    psi_ref["bin_edges"] = bin_edges;
    psi_ref["bin_percentages"] = bin_percentages;
    psi_ref["num_bins"] = bin_percentages.size();

    auto redis_key = BuildRedisKey(model_id, "psi:" + feature_name);
    return redis_->Set(redis_key, psi_ref.dump(), config_.cache_ttl);
}

absl::StatusOr<std::pair<std::vector<double>, std::vector<double>>>
ReferenceStore::GetPSIReference(const std::string& model_id,
                                 const std::string& feature_name) {

    auto redis_key = BuildRedisKey(model_id, "psi:" + feature_name);
    auto result = redis_->Get(redis_key);
    if (!result.ok()) {
        return result.status();
    }

    try {
        json j = json::parse(*result);
        auto edges = j["bin_edges"].get<std::vector<double>>();
        auto percentages = j["bin_percentages"].get<std::vector<double>>();
        return std::make_pair(std::move(edges), std::move(percentages));
    } catch (const json::exception& e) {
        return absl::InvalidArgumentError(
            std::string("Failed to parse PSI reference: ") + e.what());
    }
}

// =============================================================================
// Metadata Operations
// =============================================================================

absl::StatusOr<ReferenceMetadata> ReferenceStore::GetMetadata(
    const std::string& model_id,
    const std::string& feature_name) {

    std::string redis_key;
    if (feature_name.empty()) {
        redis_key = BuildRedisKey(model_id, "metadata");
    } else {
        redis_key = BuildRedisKey(model_id, "metadata:" + feature_name);
    }

    auto result = redis_->Get(redis_key);
    if (!result.ok()) {
        return result.status();
    }

    try {
        json j = json::parse(*result);
        ReferenceMetadata meta;
        meta.model_id = j.value("model_id", "");
        meta.feature_name = j.value("feature_name", "");
        meta.reference_type = j.value("reference_type", "");
        meta.sample_count = j.value("sample_count", size_t{0});
        meta.data_source = j.value("data_source", "");
        meta.mean = j.value("mean", 0.0);
        meta.std_dev = j.value("std_dev", 0.0);

        if (j.contains("created_at")) {
            meta.created_at = std::chrono::system_clock::time_point(
                std::chrono::seconds(j["created_at"].get<int64_t>()));
        }
        if (j.contains("updated_at")) {
            meta.updated_at = std::chrono::system_clock::time_point(
                std::chrono::seconds(j["updated_at"].get<int64_t>()));
        }

        return meta;
    } catch (const json::exception& e) {
        return absl::InvalidArgumentError(
            std::string("Failed to parse metadata: ") + e.what());
    }
}

absl::StatusOr<bool> ReferenceStore::HasReference(
    const std::string& model_id,
    const std::string& feature_name) {

    auto meta = GetMetadata(model_id, feature_name);
    if (meta.ok()) {
        return true;
    }
    if (absl::IsNotFound(meta.status())) {
        return false;
    }
    return meta.status();
}

absl::Status ReferenceStore::DeleteReference(
    const std::string& model_id,
    const std::string& feature_name) {

    if (feature_name.empty()) {
        // Delete all embeddings for this model from Qdrant
        auto status = qdrant_->DeleteByFilter(
            config_.embeddings_collection,
            {{"model_id", model_id}});
        if (!status.ok() && !absl::IsNotFound(status)) {
            return status;
        }

        // Delete metadata from Redis
        auto meta_key = BuildRedisKey(model_id, "metadata");
        redis_->Delete(meta_key);

        auto centroid_key = BuildRedisKey(model_id, "centroid");
        redis_->Delete(centroid_key);

        spdlog::info("Deleted embedding reference for model: {}", model_id);
    } else {
        // Delete specific feature reference
        auto feature_key = BuildRedisKey(model_id, "feature:" + feature_name);
        auto status = redis_->Delete(feature_key);
        if (!status.ok()) {
            return status;
        }

        auto psi_key = BuildRedisKey(model_id, "psi:" + feature_name);
        redis_->Delete(psi_key);

        auto meta_key = BuildRedisKey(model_id, "metadata:" + feature_name);
        redis_->Delete(meta_key);

        spdlog::info("Deleted feature reference {} for model: {}",
                     feature_name, model_id);
    }

    return absl::OkStatus();
}

absl::StatusOr<std::vector<std::string>> ReferenceStore::ListModels() {
    // This would require scanning Redis keys or maintaining an index
    // For now, return empty - would need proper implementation with Redis SCAN
    std::vector<std::string> models;

    // Could use Redis SCAN with pattern "pyflare:drift:*:metadata"
    // and extract model IDs, but that requires additional Redis client support

    return models;
}

// =============================================================================
// Private Helpers
// =============================================================================

std::string ReferenceStore::BuildPointId(const std::string& model_id,
                                          size_t index) const {
    return model_id + "_" + std::to_string(index);
}

std::string ReferenceStore::BuildRedisKey(const std::string& model_id,
                                           const std::string& suffix) const {
    return "pyflare:drift:" + model_id + ":" + suffix;
}

}  // namespace pyflare::drift
