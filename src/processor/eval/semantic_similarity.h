#pragma once

/// @file semantic_similarity.h
/// @brief Semantic similarity computation using embeddings
///
/// Provides embedding-based semantic similarity scoring for:
/// - Comparing model outputs to references
/// - Finding similar traces
/// - Quality assessment
/// - Detecting semantic drift

#include <chrono>
#include <memory>
#include <mutex>
#include <optional>
#include <string>
#include <unordered_map>
#include <vector>

#include <absl/status/status.h>
#include <absl/status/statusor.h>

#include "processor/eval/evaluator.h"

namespace pyflare::eval {

/// @brief Similarity metric types
enum class SimilarityMetric {
    kCosine,           ///< Cosine similarity (default)
    kDotProduct,       ///< Dot product
    kEuclidean,        ///< Euclidean distance (normalized)
    kManhattan,        ///< Manhattan distance (normalized)
    kJaccardEmbedding  ///< Jaccard-like for embeddings
};

/// @brief Configuration for semantic similarity
struct SemanticSimilarityConfig {
    /// Embedding model to use (OpenAI-compatible API)
    std::string embedding_model = "text-embedding-3-small";

    /// Embedding dimension
    size_t embedding_dimension = 1536;

    /// Similarity metric
    SimilarityMetric metric = SimilarityMetric::kCosine;

    /// API endpoint
    std::string api_endpoint = "https://api.openai.com/v1/embeddings";

    /// API key (or use environment variable OPENAI_API_KEY)
    std::string api_key;

    /// Request timeout
    std::chrono::seconds timeout = std::chrono::seconds(30);

    /// Maximum retries on failure
    size_t max_retries = 3;

    /// Batch size for embedding requests
    size_t batch_size = 100;

    /// Enable caching of embeddings
    bool enable_cache = true;

    /// Cache TTL
    std::chrono::hours cache_ttl = std::chrono::hours(24);

    /// Threshold for "similar" texts
    double similarity_threshold = 0.8;

    /// Threshold for "different" texts
    double difference_threshold = 0.5;
};

/// @brief Result of similarity comparison
struct SimilarityResult {
    double score = 0.0;  ///< 0.0 - 1.0 (higher = more similar)

    SimilarityMetric metric_used;

    /// Interpretation of score
    enum class Interpretation {
        kHighlySimilar,   ///< score >= 0.9
        kSimilar,         ///< score >= 0.8
        kSomewhatSimilar, ///< score >= 0.6
        kDifferent,       ///< score >= 0.3
        kVeryDifferent    ///< score < 0.3
    };
    Interpretation interpretation;

    /// Additional details
    std::string explanation;

    /// If comparing to multiple references, best match info
    std::optional<size_t> best_match_index;
    std::optional<double> best_match_score;

    /// All scores if multiple references
    std::vector<double> all_scores;
};

/// @brief Semantic similarity scorer using embeddings
///
/// Computes semantic similarity between texts using embedding vectors.
///
/// Supports:
/// - Single text comparison
/// - Comparison to multiple references
/// - Batch similarity computation
/// - Embedding caching for efficiency
///
/// Example:
/// @code
///   SemanticSimilarityConfig config;
///   config.embedding_model = "text-embedding-3-small";
///   config.api_key = "sk-...";
///   auto scorer = std::make_unique<SemanticSimilarityScorer>(config);
///   scorer->Initialize();
///
///   // Compare two texts
///   auto result = scorer->Compare(
///       "Paris is the capital of France",
///       "The capital city of France is Paris");
///   // result.score ~= 0.95
///
///   // Compare to multiple references
///   auto result = scorer->CompareToReferences(
///       "What is 2+2?",
///       {"The answer is 4", "2+2=4", "Four"});
///   // Returns best matching reference
/// @endcode
class SemanticSimilarityScorer : public Evaluator {
public:
    explicit SemanticSimilarityScorer(SemanticSimilarityConfig config = {});
    ~SemanticSimilarityScorer() override;

    // Disable copy
    SemanticSimilarityScorer(const SemanticSimilarityScorer&) = delete;
    SemanticSimilarityScorer& operator=(const SemanticSimilarityScorer&) = delete;

    /// @brief Initialize scorer (verify API access)
    absl::Status Initialize();

    // =========================================================================
    // Evaluator Interface
    // =========================================================================

    absl::StatusOr<EvalResult> Evaluate(const InferenceRecord& record) override;
    absl::StatusOr<std::vector<EvalResult>> EvaluateBatch(
        const std::vector<InferenceRecord>& records) override;
    std::string Type() const override { return "SemanticSimilarity"; }

    // =========================================================================
    // Similarity API
    // =========================================================================

    /// @brief Compare two texts
    /// @param text_a First text
    /// @param text_b Second text
    absl::StatusOr<SimilarityResult> Compare(
        const std::string& text_a,
        const std::string& text_b);

    /// @brief Compare text to multiple references
    /// @param text Text to compare
    /// @param references Reference texts
    /// @return Result with best match info
    absl::StatusOr<SimilarityResult> CompareToReferences(
        const std::string& text,
        const std::vector<std::string>& references);

    /// @brief Compute pairwise similarities
    /// @param texts Texts to compare
    /// @return Matrix of similarity scores (row i, col j = similarity(i, j))
    absl::StatusOr<std::vector<std::vector<double>>> ComputePairwiseSimilarity(
        const std::vector<std::string>& texts);

    /// @brief Find most similar texts in a corpus
    /// @param query Query text
    /// @param corpus Corpus to search
    /// @param top_k Number of results to return
    /// @return Indices and scores of most similar texts
    absl::StatusOr<std::vector<std::pair<size_t, double>>> FindMostSimilar(
        const std::string& query,
        const std::vector<std::string>& corpus,
        size_t top_k = 5);

    // =========================================================================
    // Embedding API
    // =========================================================================

    /// @brief Get embedding for text
    /// @param text Text to embed
    absl::StatusOr<std::vector<float>> GetEmbedding(const std::string& text);

    /// @brief Get embeddings for multiple texts
    /// @param texts Texts to embed
    absl::StatusOr<std::vector<std::vector<float>>> GetEmbeddings(
        const std::vector<std::string>& texts);

    /// @brief Compute similarity from pre-computed embeddings
    /// @param embedding_a First embedding
    /// @param embedding_b Second embedding
    double ComputeSimilarity(
        const std::vector<float>& embedding_a,
        const std::vector<float>& embedding_b);

    // =========================================================================
    // Configuration
    // =========================================================================

    /// @brief Update configuration
    void SetConfig(SemanticSimilarityConfig config);

    /// @brief Get configuration
    const SemanticSimilarityConfig& GetConfig() const { return config_; }

    /// @brief Clear embedding cache
    void ClearCache();

    // =========================================================================
    // Statistics
    // =========================================================================

    /// @brief Get statistics
    struct Stats {
        size_t total_comparisons = 0;
        size_t embedding_requests = 0;
        size_t cache_hits = 0;
        size_t api_errors = 0;
        double avg_embedding_time_ms = 0.0;
        double avg_comparison_time_ms = 0.0;
    };
    Stats GetStats() const;

    /// @brief Reset statistics
    void ResetStats();

private:
    // Similarity computations
    double ComputeCosine(const std::vector<float>& a, const std::vector<float>& b);
    double ComputeDotProduct(const std::vector<float>& a, const std::vector<float>& b);
    double ComputeEuclidean(const std::vector<float>& a, const std::vector<float>& b);
    double ComputeManhattan(const std::vector<float>& a, const std::vector<float>& b);
    double ComputeJaccard(const std::vector<float>& a, const std::vector<float>& b);

    // Score interpretation
    SimilarityResult::Interpretation InterpretScore(double score);

    // Cache management
    std::string ComputeCacheKey(const std::string& text);
    std::optional<std::vector<float>> GetCachedEmbedding(const std::string& key);
    void CacheEmbedding(const std::string& key, const std::vector<float>& embedding);

    // API calls
    absl::StatusOr<std::vector<std::vector<float>>> CallEmbeddingAPI(
        const std::vector<std::string>& texts);

    // Convert to eval result
    EvalResult ToEvalResult(const SimilarityResult& result);

    SemanticSimilarityConfig config_;

    // Embedding cache
    mutable std::mutex cache_mutex_;
    std::unordered_map<std::string, std::vector<float>> embedding_cache_;

    // Statistics
    Stats stats_;
    mutable std::mutex stats_mutex_;

    bool initialized_ = false;
};

/// @brief Create semantic similarity scorer
std::unique_ptr<Evaluator> CreateSemanticSimilarityScorer(
    SemanticSimilarityConfig config = {});

/// @brief Convert metric to string
std::string SimilarityMetricToString(SimilarityMetric metric);

/// @brief Convert string to metric
SimilarityMetric StringToSimilarityMetric(const std::string& str);

}  // namespace pyflare::eval
