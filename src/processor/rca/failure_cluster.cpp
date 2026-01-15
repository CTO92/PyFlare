/// @file failure_cluster.cpp
/// @brief Failure clustering implementation

#include "processor/rca/failure_cluster.h"

#include <algorithm>
#include <cctype>
#include <cmath>
#include <numeric>
#include <set>
#include <sstream>

#include <spdlog/spdlog.h>

namespace pyflare::rca {

// =============================================================================
// Implementation Class
// =============================================================================

class FailureClusterer::Impl {
public:
    // Embedding cache
    std::unordered_map<std::string, std::vector<float>> embedding_cache_;
    std::mutex cache_mutex_;
};

// =============================================================================
// FailureClusterer Implementation
// =============================================================================

FailureClusterer::FailureClusterer(FailureClusterConfig config)
    : config_(std::move(config)),
      impl_(std::make_unique<Impl>()) {}

FailureClusterer::~FailureClusterer() = default;

absl::Status FailureClusterer::Initialize() {
    spdlog::info("FailureClusterer initialized with method: {}",
                 MethodToString(config_.method));
    return absl::OkStatus();
}

absl::StatusOr<std::vector<FailureCluster>> FailureClusterer::ClusterFailures(
    const std::vector<FailureRecord>& failures) {
    return ClusterFailures(failures, config_.method);
}

absl::StatusOr<std::vector<FailureCluster>> FailureClusterer::ClusterFailures(
    const std::vector<FailureRecord>& failures,
    ClusteringMethod method) {

    if (failures.empty()) {
        return std::vector<FailureCluster>{};
    }

    std::vector<FailureCluster> clusters;

    switch (method) {
        case ClusteringMethod::kTextSimilarity:
            clusters = ClusterByText(failures);
            break;
        case ClusteringMethod::kAttributeBased:
            clusters = ClusterByAttributes(failures);
            break;
        case ClusteringMethod::kTemporal:
            clusters = ClusterByTime(failures);
            break;
        case ClusteringMethod::kDBSCAN:
            clusters = ClusterDBSCAN(failures);
            break;
        default:
            clusters = ClusterByText(failures);
    }

    // Merge similar clusters if enabled
    if (config_.merge_similar_clusters) {
        clusters = MergeClusters(clusters);
    }

    // Enrich clusters
    for (auto& cluster : clusters) {
        cluster.representative_error = FindRepresentativeError(cluster);
        cluster.common_keywords = ExtractKeywords(cluster);
        cluster.severity = CalculateSeverity(cluster);

        if (config_.auto_name_clusters) {
            cluster.name = GenerateClusterName(cluster);
        }
    }

    // Sort by size/severity
    std::sort(clusters.begin(), clusters.end(),
              [](const auto& a, const auto& b) {
                  return a.severity > b.severity;
              });

    // Limit to max clusters
    if (clusters.size() > config_.max_clusters) {
        clusters.resize(config_.max_clusters);
    }

    return clusters;
}

absl::StatusOr<std::string> FailureClusterer::AssignToCluster(
    const FailureRecord& failure,
    const std::vector<FailureCluster>& clusters) {

    double best_similarity = 0.0;
    std::string best_cluster_id;

    for (const auto& cluster : clusters) {
        // Compare to representative error
        double similarity = CalculateTextSimilarity(
            failure.error_message, cluster.representative_error);

        if (similarity > best_similarity &&
            similarity >= config_.similarity_threshold) {
            best_similarity = similarity;
            best_cluster_id = cluster.id;
        }
    }

    return best_cluster_id;
}

std::vector<FailureCluster> FailureClusterer::MergeClusters(
    const std::vector<FailureCluster>& clusters) {

    if (clusters.size() < 2) {
        return clusters;
    }

    std::vector<FailureCluster> merged;
    std::vector<bool> absorbed(clusters.size(), false);

    for (size_t i = 0; i < clusters.size(); ++i) {
        if (absorbed[i]) continue;

        FailureCluster combined = clusters[i];

        for (size_t j = i + 1; j < clusters.size(); ++j) {
            if (absorbed[j]) continue;

            double similarity = CalculateClusterSimilarity(combined, clusters[j]);
            if (similarity >= config_.merge_threshold) {
                // Merge cluster j into combined
                for (const auto& tid : clusters[j].trace_ids) {
                    combined.trace_ids.push_back(tid);
                }
                for (const auto& failure : clusters[j].failures) {
                    combined.failures.push_back(failure);
                }
                combined.size += clusters[j].size;
                absorbed[j] = true;
            }
        }

        merged.push_back(std::move(combined));
    }

    return merged;
}

ClusteringMetrics FailureClusterer::CalculateMetrics(
    const std::vector<FailureCluster>& clusters) {

    ClusteringMetrics metrics;
    metrics.num_clusters = clusters.size();

    if (clusters.empty()) {
        return metrics;
    }

    double total_size = 0;
    double total_cohesion = 0;
    double total_separation = 0;

    for (const auto& cluster : clusters) {
        total_size += cluster.size;
        total_cohesion += cluster.cohesion;
        total_separation += cluster.separation;
    }

    metrics.avg_cluster_size = total_size / clusters.size();
    metrics.avg_cohesion = total_cohesion / clusters.size();
    metrics.avg_separation = total_separation / clusters.size();

    // Calculate silhouette score (simplified)
    if (metrics.avg_cohesion + metrics.avg_separation > 0) {
        metrics.silhouette_score =
            (metrics.avg_separation - metrics.avg_cohesion) /
            std::max(metrics.avg_separation, metrics.avg_cohesion);
    }

    return metrics;
}

std::string FailureClusterer::FindRepresentativeError(const FailureCluster& cluster) {
    if (cluster.failures.empty()) {
        return "";
    }

    // Find most common error message
    std::unordered_map<std::string, size_t> counts;
    for (const auto& failure : cluster.failures) {
        counts[failure.error_message]++;
    }

    std::string most_common;
    size_t max_count = 0;
    for (const auto& [msg, count] : counts) {
        if (count > max_count) {
            max_count = count;
            most_common = msg;
        }
    }

    return most_common;
}

std::vector<std::string> FailureClusterer::ExtractKeywords(
    const FailureCluster& cluster) {

    std::unordered_map<std::string, size_t> word_counts;

    for (const auto& failure : cluster.failures) {
        auto tokens = Tokenize(failure.error_message);
        for (const auto& token : tokens) {
            word_counts[token]++;
        }
    }

    // Sort by frequency
    std::vector<std::pair<std::string, size_t>> sorted(
        word_counts.begin(), word_counts.end());
    std::sort(sorted.begin(), sorted.end(),
              [](const auto& a, const auto& b) {
                  return a.second > b.second;
              });

    // Return top keywords
    std::vector<std::string> keywords;
    for (size_t i = 0; i < std::min(size_t{10}, sorted.size()); ++i) {
        keywords.push_back(sorted[i].first);
    }

    return keywords;
}

std::string FailureClusterer::GenerateClusterName(const FailureCluster& cluster) {
    if (cluster.common_keywords.empty()) {
        return "Cluster " + cluster.id;
    }

    std::ostringstream name;
    name << cluster.common_keywords[0];
    if (cluster.common_keywords.size() > 1) {
        name << " - " << cluster.common_keywords[1];
    }
    name << " (" << cluster.size << " failures)";

    return name.str();
}

double FailureClusterer::CalculateSeverity(const FailureCluster& cluster) {
    // Combine size, recency, and impact
    double size_factor = std::min(1.0, cluster.size / 50.0);

    auto now = std::chrono::system_clock::now();
    auto age = std::chrono::duration_cast<std::chrono::hours>(
        now - cluster.last_occurrence);
    double recency_factor = std::exp(-age.count() / 24.0);  // Decay over 24 hours

    double user_factor = std::min(1.0, cluster.affected_users / 20.0);

    return (size_factor * 0.4 + recency_factor * 0.3 + user_factor * 0.3);
}

double FailureClusterer::CalculateTextSimilarity(
    const std::string& text1,
    const std::string& text2) {

    auto tokens1 = Tokenize(text1);
    auto tokens2 = Tokenize(text2);

    return JaccardSimilarity(tokens1, tokens2);
}

double FailureClusterer::CalculateFailureSimilarity(
    const FailureRecord& f1,
    const FailureRecord& f2) {

    double text_sim = CalculateTextSimilarity(f1.error_message, f2.error_message);

    // Also consider type match
    double type_match = (f1.failure_type == f2.failure_type) ? 1.0 : 0.0;

    // And model match
    double model_match = (f1.model_id == f2.model_id) ? 1.0 : 0.0;

    return text_sim * 0.6 + type_match * 0.25 + model_match * 0.15;
}

double FailureClusterer::CalculateClusterSimilarity(
    const FailureCluster& c1,
    const FailureCluster& c2) {

    // Compare representative errors
    double text_sim = CalculateTextSimilarity(
        c1.representative_error, c2.representative_error);

    // Compare keywords
    std::set<std::string> kw1(c1.common_keywords.begin(), c1.common_keywords.end());
    std::set<std::string> kw2(c2.common_keywords.begin(), c2.common_keywords.end());

    std::vector<std::string> common;
    std::set_intersection(kw1.begin(), kw1.end(),
                          kw2.begin(), kw2.end(),
                          std::back_inserter(common));

    std::set<std::string> all;
    all.insert(kw1.begin(), kw1.end());
    all.insert(kw2.begin(), kw2.end());

    double keyword_sim = all.empty() ? 0.0 :
        static_cast<double>(common.size()) / all.size();

    return text_sim * 0.7 + keyword_sim * 0.3;
}

// =============================================================================
// Static Methods
// =============================================================================

std::string FailureClusterer::MethodToString(ClusteringMethod method) {
    switch (method) {
        case ClusteringMethod::kTextSimilarity: return "text_similarity";
        case ClusteringMethod::kEmbedding: return "embedding";
        case ClusteringMethod::kAttributeBased: return "attribute_based";
        case ClusteringMethod::kTemporal: return "temporal";
        case ClusteringMethod::kHierarchical: return "hierarchical";
        case ClusteringMethod::kDBSCAN: return "dbscan";
    }
    return "unknown";
}

ClusteringMethod FailureClusterer::StringToMethod(const std::string& str) {
    if (str == "text_similarity") return ClusteringMethod::kTextSimilarity;
    if (str == "embedding") return ClusteringMethod::kEmbedding;
    if (str == "attribute_based") return ClusteringMethod::kAttributeBased;
    if (str == "temporal") return ClusteringMethod::kTemporal;
    if (str == "hierarchical") return ClusteringMethod::kHierarchical;
    if (str == "dbscan") return ClusteringMethod::kDBSCAN;
    return ClusteringMethod::kTextSimilarity;
}

// =============================================================================
// Private Methods
// =============================================================================

std::vector<FailureCluster> FailureClusterer::ClusterByText(
    const std::vector<FailureRecord>& failures) {

    std::vector<FailureCluster> clusters;
    std::vector<bool> assigned(failures.size(), false);

    for (size_t i = 0; i < failures.size(); ++i) {
        if (assigned[i]) continue;

        FailureCluster cluster;
        cluster.id = "cluster_" + std::to_string(clusters.size());
        cluster.failures.push_back(failures[i]);
        cluster.trace_ids.push_back(failures[i].trace_id);
        cluster.first_occurrence = failures[i].timestamp;
        cluster.last_occurrence = failures[i].timestamp;
        assigned[i] = true;

        // Find similar failures
        for (size_t j = i + 1; j < failures.size(); ++j) {
            if (assigned[j]) continue;

            double similarity = CalculateTextSimilarity(
                failures[i].error_message, failures[j].error_message);

            if (similarity >= config_.similarity_threshold) {
                cluster.failures.push_back(failures[j]);
                cluster.trace_ids.push_back(failures[j].trace_id);

                if (failures[j].timestamp < cluster.first_occurrence) {
                    cluster.first_occurrence = failures[j].timestamp;
                }
                if (failures[j].timestamp > cluster.last_occurrence) {
                    cluster.last_occurrence = failures[j].timestamp;
                }

                assigned[j] = true;
            }
        }

        cluster.size = cluster.failures.size();
        if (cluster.size >= config_.min_cluster_size) {
            clusters.push_back(std::move(cluster));
        }
    }

    return clusters;
}

std::vector<FailureCluster> FailureClusterer::ClusterByAttributes(
    const std::vector<FailureRecord>& failures) {

    std::vector<FailureCluster> clusters;

    // Group by failure type
    std::unordered_map<std::string, std::vector<FailureRecord>> by_type;
    for (const auto& f : failures) {
        by_type[f.failure_type].push_back(f);
    }

    for (auto& [type, group] : by_type) {
        if (group.size() >= config_.min_cluster_size) {
            FailureCluster cluster;
            cluster.id = "cluster_" + type;
            cluster.category = type;
            cluster.failures = std::move(group);

            for (const auto& f : cluster.failures) {
                cluster.trace_ids.push_back(f.trace_id);
            }
            cluster.size = cluster.failures.size();

            if (!cluster.failures.empty()) {
                cluster.first_occurrence = cluster.failures.front().timestamp;
                cluster.last_occurrence = cluster.failures.back().timestamp;
            }

            clusters.push_back(std::move(cluster));
        }
    }

    return clusters;
}

std::vector<FailureCluster> FailureClusterer::ClusterByTime(
    const std::vector<FailureRecord>& failures) {

    std::vector<FailureCluster> clusters;

    if (failures.empty()) {
        return clusters;
    }

    // Sort by timestamp
    std::vector<FailureRecord> sorted = failures;
    std::sort(sorted.begin(), sorted.end(),
              [](const auto& a, const auto& b) {
                  return a.timestamp < b.timestamp;
              });

    // Group by temporal proximity
    FailureCluster current;
    current.id = "cluster_0";
    current.failures.push_back(sorted[0]);
    current.trace_ids.push_back(sorted[0].trace_id);
    current.first_occurrence = sorted[0].timestamp;
    current.last_occurrence = sorted[0].timestamp;

    for (size_t i = 1; i < sorted.size(); ++i) {
        auto gap = sorted[i].timestamp - current.last_occurrence;

        if (gap <= config_.temporal_window) {
            // Add to current cluster
            current.failures.push_back(sorted[i]);
            current.trace_ids.push_back(sorted[i].trace_id);
            current.last_occurrence = sorted[i].timestamp;
        } else {
            // Start new cluster
            current.size = current.failures.size();
            if (current.size >= config_.min_cluster_size) {
                clusters.push_back(std::move(current));
            }

            current = FailureCluster{};
            current.id = "cluster_" + std::to_string(clusters.size());
            current.failures.push_back(sorted[i]);
            current.trace_ids.push_back(sorted[i].trace_id);
            current.first_occurrence = sorted[i].timestamp;
            current.last_occurrence = sorted[i].timestamp;
        }
    }

    // Add last cluster
    current.size = current.failures.size();
    if (current.size >= config_.min_cluster_size) {
        clusters.push_back(std::move(current));
    }

    return clusters;
}

std::vector<FailureCluster> FailureClusterer::ClusterDBSCAN(
    const std::vector<FailureRecord>& failures) {

    // Placeholder - would implement DBSCAN on embeddings
    // For now, fall back to text similarity
    return ClusterByText(failures);
}

std::vector<float> FailureClusterer::CalculateCentroid(
    const std::vector<std::vector<float>>& embeddings) {

    if (embeddings.empty()) {
        return {};
    }

    size_t dim = embeddings[0].size();
    std::vector<float> centroid(dim, 0.0f);

    for (const auto& emb : embeddings) {
        for (size_t i = 0; i < dim && i < emb.size(); ++i) {
            centroid[i] += emb[i];
        }
    }

    float n = static_cast<float>(embeddings.size());
    for (size_t i = 0; i < dim; ++i) {
        centroid[i] /= n;
    }

    return centroid;
}

absl::StatusOr<std::vector<float>> FailureClusterer::GetEmbedding(
    const std::string& text) {

    // Check cache
    {
        std::lock_guard<std::mutex> lock(impl_->cache_mutex_);
        auto it = impl_->embedding_cache_.find(text);
        if (it != impl_->embedding_cache_.end()) {
            return it->second;
        }
    }

    // Would call embedding API here
    return absl::UnavailableError("Embedding API not implemented");
}

std::vector<std::string> FailureClusterer::Tokenize(const std::string& text) {
    std::vector<std::string> tokens;
    std::string current;

    for (char c : text) {
        if (std::isalnum(static_cast<unsigned char>(c))) {
            current += std::tolower(static_cast<unsigned char>(c));
        } else if (!current.empty()) {
            if (current.length() >= 3) {  // Skip short words
                tokens.push_back(current);
            }
            current.clear();
        }
    }

    if (!current.empty() && current.length() >= 3) {
        tokens.push_back(current);
    }

    return tokens;
}

double FailureClusterer::JaccardSimilarity(
    const std::vector<std::string>& tokens1,
    const std::vector<std::string>& tokens2) {

    if (tokens1.empty() && tokens2.empty()) {
        return 1.0;
    }
    if (tokens1.empty() || tokens2.empty()) {
        return 0.0;
    }

    std::set<std::string> set1(tokens1.begin(), tokens1.end());
    std::set<std::string> set2(tokens2.begin(), tokens2.end());

    std::vector<std::string> intersection;
    std::set_intersection(set1.begin(), set1.end(),
                          set2.begin(), set2.end(),
                          std::back_inserter(intersection));

    std::set<std::string> union_set;
    union_set.insert(set1.begin(), set1.end());
    union_set.insert(set2.begin(), set2.end());

    return static_cast<double>(intersection.size()) / union_set.size();
}

// =============================================================================
// Factory Function
// =============================================================================

std::unique_ptr<FailureClusterer> CreateFailureClusterer(
    FailureClusterConfig config) {
    return std::make_unique<FailureClusterer>(std::move(config));
}

}  // namespace pyflare::rca
