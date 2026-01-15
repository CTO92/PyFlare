#pragma once

/// @file failure_cluster.h
/// @brief Failure clustering for root cause analysis

#include <chrono>
#include <memory>
#include <string>
#include <unordered_map>
#include <vector>

#include <absl/status/statusor.h>

#include "processor/rca/analyzer.h"

namespace pyflare::rca {

/// @brief Clustering algorithm types
enum class ClusteringMethod {
    kTextSimilarity,    ///< Cluster by error message similarity
    kEmbedding,         ///< Cluster by embedding similarity
    kAttributeBased,    ///< Cluster by common attributes
    kTemporal,          ///< Cluster by time proximity
    kHierarchical,      ///< Hierarchical clustering
    kDBSCAN             ///< Density-based clustering
};

/// @brief A cluster of related failures
struct FailureCluster {
    std::string id;
    std::string name;  ///< Human-readable cluster name

    // Cluster members
    std::vector<std::string> trace_ids;
    std::vector<FailureRecord> failures;
    size_t size = 0;

    // Cluster characteristics
    std::string representative_error;  ///< Most common/central error
    std::unordered_map<std::string, std::string> common_attributes;
    std::vector<std::string> common_keywords;

    // Timing
    std::chrono::system_clock::time_point first_occurrence;
    std::chrono::system_clock::time_point last_occurrence;
    double occurrence_rate = 0.0;  ///< Failures per hour

    // Quality metrics
    double cohesion = 0.0;      ///< How similar cluster members are
    double separation = 0.0;    ///< How different from other clusters
    double silhouette = 0.0;    ///< Silhouette score

    // Impact
    size_t affected_users = 0;
    int64_t cost_impact_micros = 0;
    double severity = 0.0;

    // Centroid (for embedding-based clustering)
    std::vector<float> centroid;

    // Labels
    std::vector<std::string> tags;
    std::string category;
};

/// @brief Configuration for failure clustering
struct FailureClusterConfig {
    /// Clustering method to use
    ClusteringMethod method = ClusteringMethod::kTextSimilarity;

    /// Minimum cluster size
    size_t min_cluster_size = 3;

    /// Maximum number of clusters
    size_t max_clusters = 20;

    /// Similarity threshold for text-based clustering
    double similarity_threshold = 0.7;

    /// DBSCAN epsilon (distance threshold)
    double dbscan_epsilon = 0.3;

    /// DBSCAN min_samples
    size_t dbscan_min_samples = 3;

    /// Time window for temporal clustering
    std::chrono::minutes temporal_window = std::chrono::minutes(5);

    /// Whether to merge similar clusters
    bool merge_similar_clusters = true;

    /// Merge threshold (similarity for merging)
    double merge_threshold = 0.9;

    /// Whether to generate cluster names automatically
    bool auto_name_clusters = true;

    /// Embedding model for embedding-based clustering
    std::string embedding_model = "text-embedding-3-small";

    /// API key for embedding service
    std::string api_key;
};

/// @brief Clustering quality metrics
struct ClusteringMetrics {
    size_t num_clusters = 0;
    size_t num_unclustered = 0;
    double avg_cluster_size = 0.0;
    double avg_cohesion = 0.0;
    double avg_separation = 0.0;
    double silhouette_score = 0.0;
    double calinski_harabasz = 0.0;
    double davies_bouldin = 0.0;
};

/// @brief Failure clustering for grouping similar failures
///
/// Groups failures into clusters to:
/// - Identify distinct failure modes
/// - Find common root causes
/// - Prioritize by cluster size/impact
/// - Track cluster evolution over time
///
/// Example usage:
/// @code
///   FailureClusterConfig config;
///   config.method = ClusteringMethod::kTextSimilarity;
///   auto clusterer = std::make_unique<FailureClusterer>(config);
///
///   // Cluster recent failures
///   auto clusters = clusterer->ClusterFailures(failures);
///
///   // Sort by impact
///   for (const auto& cluster : *clusters) {
///       std::cout << cluster.name << " (" << cluster.size << " failures)\n";
///       std::cout << "  Representative: " << cluster.representative_error << "\n";
///   }
/// @endcode
class FailureClusterer {
public:
    explicit FailureClusterer(FailureClusterConfig config = {});
    ~FailureClusterer();

    // Disable copy
    FailureClusterer(const FailureClusterer&) = delete;
    FailureClusterer& operator=(const FailureClusterer&) = delete;

    /// @brief Initialize the clusterer
    absl::Status Initialize();

    // =========================================================================
    // Clustering Operations
    // =========================================================================

    /// @brief Cluster a set of failures
    /// @param failures Failures to cluster
    absl::StatusOr<std::vector<FailureCluster>> ClusterFailures(
        const std::vector<FailureRecord>& failures);

    /// @brief Cluster failures using a specific method
    absl::StatusOr<std::vector<FailureCluster>> ClusterFailures(
        const std::vector<FailureRecord>& failures,
        ClusteringMethod method);

    /// @brief Assign a new failure to existing clusters
    /// @param failure New failure to assign
    /// @param clusters Existing clusters
    /// @return Cluster ID or empty if no match
    absl::StatusOr<std::string> AssignToCluster(
        const FailureRecord& failure,
        const std::vector<FailureCluster>& clusters);

    /// @brief Merge similar clusters
    std::vector<FailureCluster> MergeClusters(
        const std::vector<FailureCluster>& clusters);

    // =========================================================================
    // Cluster Analysis
    // =========================================================================

    /// @brief Calculate clustering quality metrics
    ClusteringMetrics CalculateMetrics(
        const std::vector<FailureCluster>& clusters);

    /// @brief Find the most representative error in a cluster
    std::string FindRepresentativeError(const FailureCluster& cluster);

    /// @brief Extract common keywords from cluster
    std::vector<std::string> ExtractKeywords(const FailureCluster& cluster);

    /// @brief Generate a human-readable name for a cluster
    std::string GenerateClusterName(const FailureCluster& cluster);

    /// @brief Calculate cluster severity based on impact
    double CalculateSeverity(const FailureCluster& cluster);

    // =========================================================================
    // Distance/Similarity Calculations
    // =========================================================================

    /// @brief Calculate similarity between two error messages
    double CalculateTextSimilarity(const std::string& text1,
                                    const std::string& text2);

    /// @brief Calculate similarity between failure records
    double CalculateFailureSimilarity(const FailureRecord& f1,
                                       const FailureRecord& f2);

    /// @brief Calculate cluster similarity
    double CalculateClusterSimilarity(const FailureCluster& c1,
                                       const FailureCluster& c2);

    // =========================================================================
    // Configuration
    // =========================================================================

    /// @brief Get configuration
    const FailureClusterConfig& GetConfig() const { return config_; }

    /// @brief Set clustering method
    void SetMethod(ClusteringMethod method) { config_.method = method; }

    // =========================================================================
    // Helpers
    // =========================================================================

    /// @brief Convert method to string
    static std::string MethodToString(ClusteringMethod method);

    /// @brief Convert string to method
    static ClusteringMethod StringToMethod(const std::string& str);

private:
    /// @brief Text-based clustering
    std::vector<FailureCluster> ClusterByText(
        const std::vector<FailureRecord>& failures);

    /// @brief Attribute-based clustering
    std::vector<FailureCluster> ClusterByAttributes(
        const std::vector<FailureRecord>& failures);

    /// @brief Temporal clustering
    std::vector<FailureCluster> ClusterByTime(
        const std::vector<FailureRecord>& failures);

    /// @brief DBSCAN clustering on embeddings
    std::vector<FailureCluster> ClusterDBSCAN(
        const std::vector<FailureRecord>& failures);

    /// @brief Calculate cluster centroid
    std::vector<float> CalculateCentroid(
        const std::vector<std::vector<float>>& embeddings);

    /// @brief Get embedding for error message
    absl::StatusOr<std::vector<float>> GetEmbedding(const std::string& text);

    /// @brief Tokenize text for similarity calculation
    std::vector<std::string> Tokenize(const std::string& text);

    /// @brief Calculate Jaccard similarity
    double JaccardSimilarity(const std::vector<std::string>& tokens1,
                              const std::vector<std::string>& tokens2);

    FailureClusterConfig config_;

    class Impl;
    std::unique_ptr<Impl> impl_;
};

/// @brief Create a failure clusterer with default configuration
std::unique_ptr<FailureClusterer> CreateFailureClusterer(
    FailureClusterConfig config = {});

}  // namespace pyflare::rca
