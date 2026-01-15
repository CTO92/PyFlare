/// @file semantic_similarity.cpp
/// @brief Semantic similarity computation implementation

#include "processor/eval/semantic_similarity.h"

#include <algorithm>
#include <cmath>
#include <functional>
#include <numeric>
#include <sstream>

#include <nlohmann/json.hpp>

namespace pyflare::eval {

using json = nlohmann::json;

namespace {

// Simple FNV-1a hash for cache keys
uint64_t HashString(const std::string& str) {
    uint64_t hash = 14695981039346656037ULL;
    for (char c : str) {
        hash ^= static_cast<unsigned char>(c);
        hash *= 1099511628211ULL;
    }
    return hash;
}

}  // namespace

SemanticSimilarityScorer::SemanticSimilarityScorer(SemanticSimilarityConfig config)
    : config_(std::move(config)) {}

SemanticSimilarityScorer::~SemanticSimilarityScorer() = default;

absl::Status SemanticSimilarityScorer::Initialize() {
    if (initialized_) {
        return absl::OkStatus();
    }

    // Check for API key
    if (config_.api_key.empty()) {
        const char* env_key = std::getenv("OPENAI_API_KEY");
        if (env_key != nullptr) {
            config_.api_key = env_key;
        }
    }

    if (config_.api_key.empty()) {
        return absl::FailedPreconditionError(
            "API key not provided and OPENAI_API_KEY environment variable not set");
    }

    // Test API connection with a simple embedding request
    // (In production, you might want to skip this for faster startup)

    initialized_ = true;
    return absl::OkStatus();
}

absl::StatusOr<EvalResult> SemanticSimilarityScorer::Evaluate(
    const InferenceRecord& record) {
    if (!record.expected_output.has_value()) {
        EvalResult result;
        result.evaluator_type = "SemanticSimilarity";
        result.score = 0.0;
        result.verdict = "skip";
        result.explanation = "No expected output to compare against";
        return result;
    }

    auto similarity = Compare(record.output, *record.expected_output);
    if (!similarity.ok()) {
        return similarity.status();
    }

    return ToEvalResult(*similarity);
}

absl::StatusOr<std::vector<EvalResult>> SemanticSimilarityScorer::EvaluateBatch(
    const std::vector<InferenceRecord>& records) {
    // Collect all texts to embed
    std::vector<std::string> texts;
    std::vector<std::pair<size_t, size_t>> comparison_pairs;  // (output_idx, expected_idx)

    for (size_t i = 0; i < records.size(); ++i) {
        if (!records[i].expected_output.has_value()) {
            continue;
        }
        size_t output_idx = texts.size();
        texts.push_back(records[i].output);
        size_t expected_idx = texts.size();
        texts.push_back(*records[i].expected_output);
        comparison_pairs.emplace_back(output_idx, expected_idx);
    }

    // Get all embeddings in batch
    auto embeddings = GetEmbeddings(texts);
    if (!embeddings.ok()) {
        return embeddings.status();
    }

    // Compute similarities
    std::vector<EvalResult> results;
    results.reserve(records.size());

    size_t pair_idx = 0;
    for (size_t i = 0; i < records.size(); ++i) {
        if (!records[i].expected_output.has_value()) {
            EvalResult result;
            result.evaluator_type = "SemanticSimilarity";
            result.score = 0.0;
            result.verdict = "skip";
            result.explanation = "No expected output to compare against";
            results.push_back(result);
            continue;
        }

        auto [output_idx, expected_idx] = comparison_pairs[pair_idx++];
        double score = ComputeSimilarity(
            (*embeddings)[output_idx],
            (*embeddings)[expected_idx]);

        SimilarityResult sim_result;
        sim_result.score = score;
        sim_result.metric_used = config_.metric;
        sim_result.interpretation = InterpretScore(score);

        results.push_back(ToEvalResult(sim_result));
    }

    return results;
}

absl::StatusOr<SimilarityResult> SemanticSimilarityScorer::Compare(
    const std::string& text_a,
    const std::string& text_b) {
    auto start_time = std::chrono::steady_clock::now();

    // Get embeddings
    auto embedding_a = GetEmbedding(text_a);
    if (!embedding_a.ok()) {
        return embedding_a.status();
    }

    auto embedding_b = GetEmbedding(text_b);
    if (!embedding_b.ok()) {
        return embedding_b.status();
    }

    // Compute similarity
    double score = ComputeSimilarity(*embedding_a, *embedding_b);

    SimilarityResult result;
    result.score = score;
    result.metric_used = config_.metric;
    result.interpretation = InterpretScore(score);

    // Build explanation
    std::stringstream ss;
    ss << "Semantic similarity: " << std::fixed << std::setprecision(2) << score;
    switch (result.interpretation) {
        case SimilarityResult::Interpretation::kHighlySimilar:
            ss << " (highly similar)";
            break;
        case SimilarityResult::Interpretation::kSimilar:
            ss << " (similar)";
            break;
        case SimilarityResult::Interpretation::kSomewhatSimilar:
            ss << " (somewhat similar)";
            break;
        case SimilarityResult::Interpretation::kDifferent:
            ss << " (different)";
            break;
        case SimilarityResult::Interpretation::kVeryDifferent:
            ss << " (very different)";
            break;
    }
    result.explanation = ss.str();

    auto end_time = std::chrono::steady_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::microseconds>(
        end_time - start_time);

    // Update statistics
    {
        std::lock_guard<std::mutex> lock(stats_mutex_);
        stats_.total_comparisons++;
        stats_.avg_comparison_time_ms = (stats_.avg_comparison_time_ms *
            (stats_.total_comparisons - 1) + duration.count() / 1000.0) /
            stats_.total_comparisons;
    }

    return result;
}

absl::StatusOr<SimilarityResult> SemanticSimilarityScorer::CompareToReferences(
    const std::string& text,
    const std::vector<std::string>& references) {
    if (references.empty()) {
        return absl::InvalidArgumentError("References list cannot be empty");
    }

    // Get embedding for text
    auto text_embedding = GetEmbedding(text);
    if (!text_embedding.ok()) {
        return text_embedding.status();
    }

    // Get embeddings for references
    auto ref_embeddings = GetEmbeddings(references);
    if (!ref_embeddings.ok()) {
        return ref_embeddings.status();
    }

    // Compute similarities
    SimilarityResult result;
    result.metric_used = config_.metric;
    result.all_scores.reserve(references.size());

    double best_score = -1.0;
    size_t best_idx = 0;

    for (size_t i = 0; i < ref_embeddings->size(); ++i) {
        double score = ComputeSimilarity(*text_embedding, (*ref_embeddings)[i]);
        result.all_scores.push_back(score);

        if (score > best_score) {
            best_score = score;
            best_idx = i;
        }
    }

    result.score = best_score;
    result.best_match_index = best_idx;
    result.best_match_score = best_score;
    result.interpretation = InterpretScore(best_score);

    // Build explanation
    std::stringstream ss;
    ss << "Best match similarity: " << std::fixed << std::setprecision(2) << best_score;
    ss << " (reference #" << best_idx << ")";
    result.explanation = ss.str();

    return result;
}

absl::StatusOr<std::vector<std::vector<double>>>
SemanticSimilarityScorer::ComputePairwiseSimilarity(
    const std::vector<std::string>& texts) {
    if (texts.empty()) {
        return std::vector<std::vector<double>>();
    }

    // Get all embeddings
    auto embeddings = GetEmbeddings(texts);
    if (!embeddings.ok()) {
        return embeddings.status();
    }

    size_t n = texts.size();
    std::vector<std::vector<double>> matrix(n, std::vector<double>(n, 0.0));

    for (size_t i = 0; i < n; ++i) {
        matrix[i][i] = 1.0;  // Self-similarity is 1
        for (size_t j = i + 1; j < n; ++j) {
            double score = ComputeSimilarity((*embeddings)[i], (*embeddings)[j]);
            matrix[i][j] = score;
            matrix[j][i] = score;  // Symmetric
        }
    }

    return matrix;
}

absl::StatusOr<std::vector<std::pair<size_t, double>>>
SemanticSimilarityScorer::FindMostSimilar(
    const std::string& query,
    const std::vector<std::string>& corpus,
    size_t top_k) {
    if (corpus.empty()) {
        return std::vector<std::pair<size_t, double>>();
    }

    // Get query embedding
    auto query_embedding = GetEmbedding(query);
    if (!query_embedding.ok()) {
        return query_embedding.status();
    }

    // Get corpus embeddings
    auto corpus_embeddings = GetEmbeddings(corpus);
    if (!corpus_embeddings.ok()) {
        return corpus_embeddings.status();
    }

    // Compute similarities
    std::vector<std::pair<size_t, double>> scores;
    scores.reserve(corpus.size());

    for (size_t i = 0; i < corpus_embeddings->size(); ++i) {
        double score = ComputeSimilarity(*query_embedding, (*corpus_embeddings)[i]);
        scores.emplace_back(i, score);
    }

    // Sort by score descending
    std::sort(scores.begin(), scores.end(),
        [](const auto& a, const auto& b) { return a.second > b.second; });

    // Take top k
    if (scores.size() > top_k) {
        scores.resize(top_k);
    }

    return scores;
}

absl::StatusOr<std::vector<float>> SemanticSimilarityScorer::GetEmbedding(
    const std::string& text) {
    // Check cache
    if (config_.enable_cache) {
        std::string cache_key = ComputeCacheKey(text);
        auto cached = GetCachedEmbedding(cache_key);
        if (cached.has_value()) {
            std::lock_guard<std::mutex> lock(stats_mutex_);
            stats_.cache_hits++;
            return *cached;
        }
    }

    // Call API for single text
    auto result = CallEmbeddingAPI({text});
    if (!result.ok()) {
        return result.status();
    }

    if (result->empty()) {
        return absl::InternalError("Empty embedding result");
    }

    // Cache result
    if (config_.enable_cache) {
        std::string cache_key = ComputeCacheKey(text);
        CacheEmbedding(cache_key, (*result)[0]);
    }

    return (*result)[0];
}

absl::StatusOr<std::vector<std::vector<float>>> SemanticSimilarityScorer::GetEmbeddings(
    const std::vector<std::string>& texts) {
    if (texts.empty()) {
        return std::vector<std::vector<float>>();
    }

    std::vector<std::vector<float>> embeddings(texts.size());
    std::vector<size_t> uncached_indices;
    std::vector<std::string> uncached_texts;

    // Check cache for each text
    if (config_.enable_cache) {
        for (size_t i = 0; i < texts.size(); ++i) {
            std::string cache_key = ComputeCacheKey(texts[i]);
            auto cached = GetCachedEmbedding(cache_key);
            if (cached.has_value()) {
                embeddings[i] = *cached;
                std::lock_guard<std::mutex> lock(stats_mutex_);
                stats_.cache_hits++;
            } else {
                uncached_indices.push_back(i);
                uncached_texts.push_back(texts[i]);
            }
        }
    } else {
        for (size_t i = 0; i < texts.size(); ++i) {
            uncached_indices.push_back(i);
            uncached_texts.push_back(texts[i]);
        }
    }

    // Fetch uncached embeddings in batches
    if (!uncached_texts.empty()) {
        for (size_t start = 0; start < uncached_texts.size();
             start += config_.batch_size) {
            size_t end = std::min(start + config_.batch_size, uncached_texts.size());
            std::vector<std::string> batch(
                uncached_texts.begin() + start,
                uncached_texts.begin() + end);

            auto batch_result = CallEmbeddingAPI(batch);
            if (!batch_result.ok()) {
                return batch_result.status();
            }

            for (size_t i = 0; i < batch_result->size(); ++i) {
                size_t idx = uncached_indices[start + i];
                embeddings[idx] = (*batch_result)[i];

                // Cache result
                if (config_.enable_cache) {
                    std::string cache_key = ComputeCacheKey(texts[idx]);
                    CacheEmbedding(cache_key, (*batch_result)[i]);
                }
            }
        }
    }

    return embeddings;
}

double SemanticSimilarityScorer::ComputeSimilarity(
    const std::vector<float>& embedding_a,
    const std::vector<float>& embedding_b) {
    switch (config_.metric) {
        case SimilarityMetric::kCosine:
            return ComputeCosine(embedding_a, embedding_b);
        case SimilarityMetric::kDotProduct:
            return ComputeDotProduct(embedding_a, embedding_b);
        case SimilarityMetric::kEuclidean:
            return ComputeEuclidean(embedding_a, embedding_b);
        case SimilarityMetric::kManhattan:
            return ComputeManhattan(embedding_a, embedding_b);
        case SimilarityMetric::kJaccardEmbedding:
            return ComputeJaccard(embedding_a, embedding_b);
    }
    return ComputeCosine(embedding_a, embedding_b);
}

void SemanticSimilarityScorer::SetConfig(SemanticSimilarityConfig config) {
    config_ = std::move(config);
}

void SemanticSimilarityScorer::ClearCache() {
    std::lock_guard<std::mutex> lock(cache_mutex_);
    embedding_cache_.clear();
}

SemanticSimilarityScorer::Stats SemanticSimilarityScorer::GetStats() const {
    std::lock_guard<std::mutex> lock(stats_mutex_);
    return stats_;
}

void SemanticSimilarityScorer::ResetStats() {
    std::lock_guard<std::mutex> lock(stats_mutex_);
    stats_ = Stats{};
}

// ============================================================================
// Private Implementation
// ============================================================================

double SemanticSimilarityScorer::ComputeCosine(
    const std::vector<float>& a,
    const std::vector<float>& b) {
    if (a.size() != b.size() || a.empty()) {
        return 0.0;
    }

    double dot = 0.0, norm_a = 0.0, norm_b = 0.0;
    for (size_t i = 0; i < a.size(); ++i) {
        dot += static_cast<double>(a[i]) * b[i];
        norm_a += static_cast<double>(a[i]) * a[i];
        norm_b += static_cast<double>(b[i]) * b[i];
    }

    double denom = std::sqrt(norm_a) * std::sqrt(norm_b);
    if (denom < 1e-10) {
        return 0.0;
    }

    return dot / denom;
}

double SemanticSimilarityScorer::ComputeDotProduct(
    const std::vector<float>& a,
    const std::vector<float>& b) {
    if (a.size() != b.size()) {
        return 0.0;
    }

    double dot = 0.0;
    for (size_t i = 0; i < a.size(); ++i) {
        dot += static_cast<double>(a[i]) * b[i];
    }

    return dot;
}

double SemanticSimilarityScorer::ComputeEuclidean(
    const std::vector<float>& a,
    const std::vector<float>& b) {
    if (a.size() != b.size()) {
        return 0.0;
    }

    double sum_sq = 0.0;
    for (size_t i = 0; i < a.size(); ++i) {
        double diff = static_cast<double>(a[i]) - b[i];
        sum_sq += diff * diff;
    }

    double distance = std::sqrt(sum_sq);
    // Normalize to [0, 1] where 1 = identical
    // Using 1 / (1 + distance) as a simple normalization
    return 1.0 / (1.0 + distance);
}

double SemanticSimilarityScorer::ComputeManhattan(
    const std::vector<float>& a,
    const std::vector<float>& b) {
    if (a.size() != b.size()) {
        return 0.0;
    }

    double sum = 0.0;
    for (size_t i = 0; i < a.size(); ++i) {
        sum += std::abs(static_cast<double>(a[i]) - b[i]);
    }

    // Normalize similar to euclidean
    return 1.0 / (1.0 + sum);
}

double SemanticSimilarityScorer::ComputeJaccard(
    const std::vector<float>& a,
    const std::vector<float>& b) {
    if (a.size() != b.size() || a.empty()) {
        return 0.0;
    }

    // Jaccard-like for continuous values:
    // sum(min(a_i, b_i)) / sum(max(a_i, b_i))
    double min_sum = 0.0, max_sum = 0.0;
    for (size_t i = 0; i < a.size(); ++i) {
        // Only consider positive values for this metric
        double ai = std::max(0.0, static_cast<double>(a[i]));
        double bi = std::max(0.0, static_cast<double>(b[i]));
        min_sum += std::min(ai, bi);
        max_sum += std::max(ai, bi);
    }

    if (max_sum < 1e-10) {
        return 1.0;  // Both zero vectors
    }

    return min_sum / max_sum;
}

SimilarityResult::Interpretation SemanticSimilarityScorer::InterpretScore(
    double score) {
    if (score >= 0.9) {
        return SimilarityResult::Interpretation::kHighlySimilar;
    } else if (score >= 0.8) {
        return SimilarityResult::Interpretation::kSimilar;
    } else if (score >= 0.6) {
        return SimilarityResult::Interpretation::kSomewhatSimilar;
    } else if (score >= 0.3) {
        return SimilarityResult::Interpretation::kDifferent;
    }
    return SimilarityResult::Interpretation::kVeryDifferent;
}

std::string SemanticSimilarityScorer::ComputeCacheKey(const std::string& text) {
    uint64_t hash = HashString(text);
    std::stringstream ss;
    ss << config_.embedding_model << "_" << std::hex << hash;
    return ss.str();
}

std::optional<std::vector<float>> SemanticSimilarityScorer::GetCachedEmbedding(
    const std::string& key) {
    std::lock_guard<std::mutex> lock(cache_mutex_);
    auto it = embedding_cache_.find(key);
    if (it != embedding_cache_.end()) {
        return it->second;
    }
    return std::nullopt;
}

void SemanticSimilarityScorer::CacheEmbedding(
    const std::string& key,
    const std::vector<float>& embedding) {
    std::lock_guard<std::mutex> lock(cache_mutex_);
    embedding_cache_[key] = embedding;
}

absl::StatusOr<std::vector<std::vector<float>>>
SemanticSimilarityScorer::CallEmbeddingAPI(
    const std::vector<std::string>& texts) {
    if (!initialized_) {
        return absl::FailedPreconditionError("Scorer not initialized");
    }

    auto start_time = std::chrono::steady_clock::now();

    // Build request
    json request_body;
    request_body["model"] = config_.embedding_model;
    request_body["input"] = texts;

    // TODO: Implement actual HTTP call using libcurl or similar
    // For now, return placeholder embeddings for testing

    // Placeholder implementation - generate random embeddings
    // In production, this would call the OpenAI API
    std::vector<std::vector<float>> embeddings;
    embeddings.reserve(texts.size());

    for (const auto& text : texts) {
        std::vector<float> embedding(config_.embedding_dimension);
        // Simple hash-based pseudo-embedding for testing
        uint64_t hash = HashString(text);
        std::mt19937 gen(hash);
        std::normal_distribution<float> dist(0.0f, 1.0f);

        float norm = 0.0f;
        for (size_t i = 0; i < config_.embedding_dimension; ++i) {
            embedding[i] = dist(gen);
            norm += embedding[i] * embedding[i];
        }

        // Normalize to unit length
        norm = std::sqrt(norm);
        if (norm > 0) {
            for (auto& v : embedding) {
                v /= norm;
            }
        }

        embeddings.push_back(std::move(embedding));
    }

    auto end_time = std::chrono::steady_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::microseconds>(
        end_time - start_time);

    // Update statistics
    {
        std::lock_guard<std::mutex> lock(stats_mutex_);
        stats_.embedding_requests++;
        stats_.avg_embedding_time_ms = (stats_.avg_embedding_time_ms *
            (stats_.embedding_requests - 1) + duration.count() / 1000.0) /
            stats_.embedding_requests;
    }

    return embeddings;
}

EvalResult SemanticSimilarityScorer::ToEvalResult(const SimilarityResult& result) {
    EvalResult eval;
    eval.evaluator_type = "SemanticSimilarity";
    eval.score = result.score;

    if (result.score >= config_.similarity_threshold) {
        eval.verdict = "pass";
    } else if (result.score >= config_.difference_threshold) {
        eval.verdict = "warn";
    } else {
        eval.verdict = "fail";
    }

    eval.explanation = result.explanation;

    eval.metadata["metric"] = SimilarityMetricToString(result.metric_used);
    eval.metadata["score"] = std::to_string(result.score);

    if (result.best_match_index.has_value()) {
        eval.metadata["best_match_index"] = std::to_string(*result.best_match_index);
    }

    return eval;
}

// ============================================================================
// Utility Functions
// ============================================================================

std::unique_ptr<Evaluator> CreateSemanticSimilarityScorer(
    SemanticSimilarityConfig config) {
    auto scorer = std::make_unique<SemanticSimilarityScorer>(std::move(config));
    scorer->Initialize();
    return scorer;
}

std::string SimilarityMetricToString(SimilarityMetric metric) {
    switch (metric) {
        case SimilarityMetric::kCosine: return "cosine";
        case SimilarityMetric::kDotProduct: return "dot_product";
        case SimilarityMetric::kEuclidean: return "euclidean";
        case SimilarityMetric::kManhattan: return "manhattan";
        case SimilarityMetric::kJaccardEmbedding: return "jaccard";
    }
    return "unknown";
}

SimilarityMetric StringToSimilarityMetric(const std::string& str) {
    if (str == "cosine") return SimilarityMetric::kCosine;
    if (str == "dot_product") return SimilarityMetric::kDotProduct;
    if (str == "euclidean") return SimilarityMetric::kEuclidean;
    if (str == "manhattan") return SimilarityMetric::kManhattan;
    if (str == "jaccard") return SimilarityMetric::kJaccardEmbedding;
    return SimilarityMetric::kCosine;
}

}  // namespace pyflare::eval
