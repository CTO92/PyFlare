/// @file rag_evaluator.cpp
/// @brief RAG quality evaluator implementation

#include "processor/eval/rag_evaluator.h"

#include <algorithm>
#include <cctype>
#include <cmath>
#include <numeric>
#include <regex>
#include <set>
#include <sstream>

#include <nlohmann/json.hpp>
#include <spdlog/spdlog.h>

namespace pyflare::eval {

using json = nlohmann::json;

// =============================================================================
// Implementation Class
// =============================================================================

class RAGEvaluator::Impl {
public:
    Impl(const RAGEvaluatorConfig& config) : config_(config) {}

    // Embedding cache
    std::unordered_map<std::string, std::vector<float>> embedding_cache_;
    std::mutex cache_mutex_;

    const RAGEvaluatorConfig& config_;
};

// =============================================================================
// RAGEvaluator Implementation
// =============================================================================

RAGEvaluator::RAGEvaluator(RAGEvaluatorConfig config)
    : config_(std::move(config)),
      impl_(std::make_unique<Impl>(config_)) {}

RAGEvaluator::~RAGEvaluator() = default;

absl::Status RAGEvaluator::Initialize() {
    if (config_.use_llm_evaluation && config_.api_key.empty()) {
        const char* env_key = std::getenv("OPENAI_API_KEY");
        if (env_key) {
            config_.api_key = env_key;
        } else {
            spdlog::warn("No API key for RAGEvaluator, using heuristics only");
            config_.use_llm_evaluation = false;
            config_.use_embedding_similarity = false;
        }
    }

    spdlog::info("RAGEvaluator initialized (LLM={}, embeddings={})",
                 config_.use_llm_evaluation, config_.use_embedding_similarity);
    return absl::OkStatus();
}

absl::StatusOr<EvalResult> RAGEvaluator::Evaluate(const InferenceRecord& record) {
    auto metrics = EvaluateRAG(record);
    if (!metrics.ok()) {
        return metrics.status();
    }

    EvalResult result;
    result.evaluator_type = "RAGEvaluator";
    result.score = metrics->overall_score;

    // Determine verdict based on thresholds
    if (metrics->faithfulness >= config_.faithfulness_threshold &&
        metrics->context_relevance >= config_.context_relevance_threshold) {
        result.verdict = "pass";
    } else if (metrics->faithfulness < 0.3 || metrics->has_hallucination) {
        result.verdict = "fail";
    } else {
        result.verdict = "warn";
    }

    // Build explanation
    std::ostringstream explanation;
    explanation << "Context Relevance: " << std::fixed << std::setprecision(2)
                << metrics->context_relevance * 100 << "%, "
                << "Faithfulness: " << metrics->faithfulness * 100 << "%, "
                << "Answer Relevance: " << metrics->answer_relevance * 100 << "%";

    if (!metrics->issues.empty()) {
        explanation << ". Issues: ";
        for (size_t i = 0; i < metrics->issues.size() && i < 3; ++i) {
            if (i > 0) explanation << "; ";
            explanation << metrics->issues[i];
        }
    }

    result.explanation = explanation.str();

    // Add metadata
    result.metadata["context_relevance"] =
        std::to_string(metrics->context_relevance);
    result.metadata["faithfulness"] = std::to_string(metrics->faithfulness);
    result.metadata["answer_relevance"] =
        std::to_string(metrics->answer_relevance);
    result.metadata["groundedness"] = std::to_string(metrics->groundedness);
    result.metadata["has_hallucination"] =
        metrics->has_hallucination ? "true" : "false";

    return result;
}

absl::StatusOr<std::vector<EvalResult>> RAGEvaluator::EvaluateBatch(
    const std::vector<InferenceRecord>& records) {

    std::vector<EvalResult> results;
    results.reserve(records.size());

    for (const auto& record : records) {
        auto result = Evaluate(record);
        if (result.ok()) {
            results.push_back(std::move(*result));
        } else {
            EvalResult err;
            err.evaluator_type = "RAGEvaluator";
            err.score = 0.0;
            err.verdict = "error";
            err.explanation = std::string(result.status().message());
            results.push_back(std::move(err));
        }
    }

    return results;
}

absl::StatusOr<RAGMetrics> RAGEvaluator::EvaluateRAG(const InferenceRecord& record) {
    RAGMetrics metrics;

    if (!record.retrieved_contexts || record.retrieved_contexts->empty()) {
        return absl::InvalidArgumentError(
            "RAG evaluation requires retrieved_contexts");
    }

    const auto& contexts = *record.retrieved_contexts;

    // Evaluate context relevance
    auto context_rel = EvaluateContextRelevance(record.input, contexts);
    if (context_rel.ok()) {
        metrics.context_relevance = *context_rel;
    } else {
        metrics.issues.push_back("Failed to evaluate context relevance");
    }

    // Evaluate faithfulness
    auto faithfulness = EvaluateFaithfulness(record.output, contexts);
    if (faithfulness.ok()) {
        metrics.faithfulness = *faithfulness;
    } else {
        metrics.issues.push_back("Failed to evaluate faithfulness");
    }

    // Evaluate answer relevance
    auto answer_rel = EvaluateAnswerRelevance(record.input, record.output);
    if (answer_rel.ok()) {
        metrics.answer_relevance = *answer_rel;
    } else {
        metrics.issues.push_back("Failed to evaluate answer relevance");
    }

    // Evaluate groundedness
    auto groundedness = EvaluateGroundedness(record.output, contexts);
    if (groundedness.ok()) {
        metrics.groundedness = *groundedness;
    } else {
        metrics.issues.push_back("Failed to evaluate groundedness");
    }

    // Check for hallucination
    if (metrics.faithfulness < 0.5 || metrics.groundedness < 0.5) {
        metrics.has_hallucination = true;
        metrics.issues.push_back("Possible hallucination detected");
    }

    // Check for irrelevant context
    if (metrics.context_relevance < 0.3) {
        metrics.has_irrelevant_context = true;
        metrics.issues.push_back("Retrieved context may not be relevant");
    }

    // Calculate token efficiency
    size_t total_tokens = 0;
    for (const auto& ctx : contexts) {
        total_tokens += ctx.size() / 4;  // Rough estimate
    }
    metrics.total_context_tokens = total_tokens;
    metrics.used_context_tokens = record.output.size() / 4;
    if (total_tokens > 0) {
        metrics.token_efficiency = static_cast<double>(metrics.used_context_tokens) /
                                   total_tokens;
    }

    // Calculate overall score
    metrics.overall_score = CalculateOverallScore(metrics);

    return metrics;
}

absl::StatusOr<double> RAGEvaluator::EvaluateContextRelevance(
    const std::string& query,
    const std::vector<std::string>& contexts) {

    if (contexts.empty()) {
        return 0.0;
    }

    // Calculate relevance of each context to the query
    std::vector<double> relevance_scores;
    relevance_scores.reserve(contexts.size());

    for (const auto& context : contexts) {
        double score;
        if (config_.use_embedding_similarity) {
            auto sim = CalculateSimilarity(query, context);
            if (sim.ok()) {
                score = *sim;
            } else {
                score = CalculateKeywordOverlap(query, context);
            }
        } else {
            score = CalculateKeywordOverlap(query, context);
        }
        relevance_scores.push_back(score);
    }

    // Return average relevance, weighted by position (earlier contexts weighted higher)
    double weighted_sum = 0.0;
    double weight_sum = 0.0;
    for (size_t i = 0; i < relevance_scores.size(); ++i) {
        double weight = 1.0 / (i + 1);  // Position-based weighting
        weighted_sum += relevance_scores[i] * weight;
        weight_sum += weight;
    }

    return weight_sum > 0 ? weighted_sum / weight_sum : 0.0;
}

absl::StatusOr<double> RAGEvaluator::EvaluateFaithfulness(
    const std::string& answer,
    const std::vector<std::string>& contexts) {

    // Extract claims from the answer
    auto claims = ExtractClaims(answer);
    if (claims.empty()) {
        return 1.0;  // No claims = fully faithful
    }

    // Check each claim against contexts
    size_t supported_claims = 0;
    for (const auto& claim : claims) {
        if (IsClaimSupported(claim, contexts)) {
            supported_claims++;
        }
    }

    return static_cast<double>(supported_claims) / claims.size();
}

absl::StatusOr<double> RAGEvaluator::EvaluateAnswerRelevance(
    const std::string& query,
    const std::string& answer) {

    if (answer.empty()) {
        return 0.0;
    }

    // Use similarity between query and answer
    if (config_.use_embedding_similarity) {
        auto sim = CalculateSimilarity(query, answer);
        if (sim.ok()) {
            return *sim;
        }
    }

    // Fall back to keyword overlap
    return CalculateKeywordOverlap(query, answer);
}

absl::StatusOr<double> RAGEvaluator::EvaluateGroundedness(
    const std::string& answer,
    const std::vector<std::string>& contexts) {

    // Combine all contexts
    std::string combined_context;
    for (const auto& ctx : contexts) {
        combined_context += ctx + " ";
    }

    // Extract claims and check grounding
    auto claims = ExtractClaims(answer);
    if (claims.empty()) {
        return 1.0;
    }

    size_t grounded_claims = 0;
    for (const auto& claim : claims) {
        // A claim is grounded if there's significant overlap with context
        double overlap;
        if (config_.use_embedding_similarity) {
            auto sim = CalculateSimilarity(claim, combined_context);
            overlap = sim.ok() ? *sim : CalculateKeywordOverlap(claim, combined_context);
        } else {
            overlap = CalculateKeywordOverlap(claim, combined_context);
        }

        if (overlap > 0.3) {  // Threshold for "grounded"
            grounded_claims++;
        }
    }

    return static_cast<double>(grounded_claims) / claims.size();
}

double RAGEvaluator::CalculateOverallScore(const RAGMetrics& metrics) const {
    const auto& w = config_.weights;

    double score = w.context_relevance * metrics.context_relevance +
                   w.faithfulness * metrics.faithfulness +
                   w.answer_relevance * metrics.answer_relevance +
                   w.groundedness * metrics.groundedness;

    // Normalize by total weights
    double total_weight = w.context_relevance + w.faithfulness +
                          w.answer_relevance + w.groundedness;

    return total_weight > 0 ? score / total_weight : 0.0;
}

// =============================================================================
// Private Methods
// =============================================================================

std::vector<std::string> RAGEvaluator::ExtractClaims(const std::string& text) {
    std::vector<std::string> claims;

    // Split by sentence-ending punctuation
    std::regex sentence_regex(R"([^.!?]+[.!?])");
    std::sregex_iterator begin(text.begin(), text.end(), sentence_regex);
    std::sregex_iterator end;

    for (auto it = begin; it != end; ++it) {
        std::string sentence = it->str();

        // Trim whitespace
        sentence.erase(0, sentence.find_first_not_of(" \t\n\r"));
        sentence.erase(sentence.find_last_not_of(" \t\n\r") + 1);

        // Skip very short sentences
        if (sentence.length() > 10) {
            claims.push_back(sentence);
        }
    }

    // If no sentences found, treat whole text as one claim
    if (claims.empty() && text.length() > 10) {
        claims.push_back(text);
    }

    return claims;
}

bool RAGEvaluator::IsClaimSupported(
    const std::string& claim,
    const std::vector<std::string>& contexts) {

    for (const auto& context : contexts) {
        double overlap;
        if (config_.use_embedding_similarity) {
            auto sim = CalculateSimilarity(claim, context);
            overlap = sim.ok() ? *sim : CalculateKeywordOverlap(claim, context);
        } else {
            overlap = CalculateKeywordOverlap(claim, context);
        }

        if (overlap > 0.4) {  // Threshold for "supported"
            return true;
        }
    }

    return false;
}

absl::StatusOr<double> RAGEvaluator::CalculateSimilarity(
    const std::string& text1,
    const std::string& text2) {

    auto emb1 = GetEmbedding(text1);
    if (!emb1.ok()) {
        return emb1.status();
    }

    auto emb2 = GetEmbedding(text2);
    if (!emb2.ok()) {
        return emb2.status();
    }

    return CosineSimilarity(*emb1, *emb2);
}

double RAGEvaluator::CalculateKeywordOverlap(
    const std::string& text1,
    const std::string& text2) {

    // Tokenize and normalize
    auto tokenize = [](const std::string& text) -> std::set<std::string> {
        std::set<std::string> tokens;
        std::istringstream iss(text);
        std::string word;

        while (iss >> word) {
            // Normalize: lowercase and remove punctuation
            std::string normalized;
            for (char c : word) {
                if (std::isalnum(static_cast<unsigned char>(c))) {
                    normalized += std::tolower(static_cast<unsigned char>(c));
                }
            }
            if (normalized.length() >= 3) {  // Skip very short words
                tokens.insert(normalized);
            }
        }
        return tokens;
    };

    auto tokens1 = tokenize(text1);
    auto tokens2 = tokenize(text2);

    if (tokens1.empty() || tokens2.empty()) {
        return 0.0;
    }

    // Calculate Jaccard similarity
    std::set<std::string> intersection;
    std::set_intersection(tokens1.begin(), tokens1.end(),
                          tokens2.begin(), tokens2.end(),
                          std::inserter(intersection, intersection.begin()));

    std::set<std::string> union_set;
    std::set_union(tokens1.begin(), tokens1.end(),
                   tokens2.begin(), tokens2.end(),
                   std::inserter(union_set, union_set.begin()));

    return static_cast<double>(intersection.size()) / union_set.size();
}

absl::StatusOr<std::vector<float>> RAGEvaluator::GetEmbedding(
    const std::string& text) {

    // Check cache
    {
        std::lock_guard<std::mutex> lock(impl_->cache_mutex_);
        auto it = impl_->embedding_cache_.find(text);
        if (it != impl_->embedding_cache_.end()) {
            return it->second;
        }
    }

    // For now, return error - actual implementation would call embedding API
    // This is a placeholder until HTTP client is integrated
    return absl::UnavailableError("Embedding API not implemented");
}

double RAGEvaluator::CosineSimilarity(const std::vector<float>& a,
                                       const std::vector<float>& b) {
    if (a.size() != b.size() || a.empty()) {
        return 0.0;
    }

    double dot = 0.0, norm_a = 0.0, norm_b = 0.0;
    for (size_t i = 0; i < a.size(); ++i) {
        dot += a[i] * b[i];
        norm_a += a[i] * a[i];
        norm_b += b[i] * b[i];
    }

    double denom = std::sqrt(norm_a) * std::sqrt(norm_b);
    return denom > 0 ? dot / denom : 0.0;
}

absl::StatusOr<double> RAGEvaluator::LLMEvaluate(
    const std::string& prompt,
    const std::string& criteria) {
    // Placeholder for LLM-based evaluation
    return absl::UnavailableError("LLM evaluation not implemented");
}

// =============================================================================
// Factory Function
// =============================================================================

std::unique_ptr<Evaluator> CreateRAGEvaluator(RAGEvaluatorConfig config) {
    return std::make_unique<RAGEvaluator>(std::move(config));
}

}  // namespace pyflare::eval
