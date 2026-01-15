#pragma once

/// @file rag_evaluator.h
/// @brief RAG (Retrieval-Augmented Generation) quality evaluator

#include <memory>
#include <optional>
#include <string>
#include <vector>

#include <absl/status/statusor.h>

#include "processor/eval/evaluator.h"

namespace pyflare::eval {

/// @brief RAG evaluation metrics
struct RAGMetrics {
    // Retrieval quality
    double context_relevance = 0.0;    ///< How relevant retrieved docs are to query
    double context_precision = 0.0;    ///< Precision of retrieved contexts
    double context_recall = 0.0;       ///< Recall of retrieved contexts (if ground truth available)

    // Generation quality
    double faithfulness = 0.0;         ///< How faithful the answer is to context
    double answer_relevance = 0.0;     ///< How relevant answer is to the question
    double groundedness = 0.0;         ///< How grounded the answer is in the context

    // Combined scores
    double overall_score = 0.0;        ///< Weighted combination of metrics

    // Token efficiency
    size_t total_context_tokens = 0;   ///< Tokens in retrieved context
    size_t used_context_tokens = 0;    ///< Tokens actually used in answer
    double token_efficiency = 0.0;     ///< Ratio of used to total

    // Issue tracking
    bool has_hallucination = false;
    bool has_missing_attribution = false;
    bool has_irrelevant_context = false;
    std::vector<std::string> issues;
};

/// @brief Configuration for RAG evaluator
struct RAGEvaluatorConfig {
    /// Minimum context relevance threshold
    double context_relevance_threshold = 0.5;

    /// Minimum faithfulness threshold
    double faithfulness_threshold = 0.7;

    /// Use embedding similarity for relevance (vs keyword overlap)
    bool use_embedding_similarity = true;

    /// Embedding model for similarity calculation
    std::string embedding_model = "text-embedding-3-small";

    /// Weights for overall score calculation
    struct Weights {
        double context_relevance = 0.2;
        double faithfulness = 0.4;
        double answer_relevance = 0.3;
        double groundedness = 0.1;
    } weights;

    /// Whether to use LLM for evaluation (vs heuristics only)
    bool use_llm_evaluation = true;

    /// LLM model for evaluation
    std::string llm_model = "gpt-4o-mini";

    /// API key for LLM/embedding calls
    std::string api_key;
};

/// @brief RAG quality evaluator
///
/// Evaluates RAG systems on multiple dimensions:
/// - Context Relevance: Are the retrieved docs relevant to the query?
/// - Faithfulness: Does the answer stick to the retrieved context?
/// - Answer Relevance: Does the answer address the user's question?
/// - Groundedness: Is every claim in the answer grounded in the context?
///
/// Supports both LLM-based evaluation and embedding/heuristic-based methods.
///
/// Example usage:
/// @code
///   RAGEvaluatorConfig config;
///   config.use_llm_evaluation = true;
///   auto evaluator = std::make_unique<RAGEvaluator>(config);
///
///   InferenceRecord record;
///   record.input = "What is quantum computing?";
///   record.output = "Quantum computing uses qubits to perform calculations...";
///   record.retrieved_contexts = {
///       "Quantum computers use quantum bits or qubits...",
///       "Classical computers use binary digits..."
///   };
///
///   auto metrics = evaluator->EvaluateRAG(record);
///   // metrics.faithfulness, metrics.context_relevance, etc.
/// @endcode
class RAGEvaluator : public Evaluator {
public:
    explicit RAGEvaluator(RAGEvaluatorConfig config = {});
    ~RAGEvaluator() override;

    // Disable copy
    RAGEvaluator(const RAGEvaluator&) = delete;
    RAGEvaluator& operator=(const RAGEvaluator&) = delete;

    /// @brief Initialize the evaluator
    absl::Status Initialize();

    /// @brief Evaluate a single inference
    absl::StatusOr<EvalResult> Evaluate(const InferenceRecord& record) override;

    /// @brief Batch evaluation
    absl::StatusOr<std::vector<EvalResult>> EvaluateBatch(
        const std::vector<InferenceRecord>& records) override;

    std::string Type() const override { return "RAGEvaluator"; }

    // =========================================================================
    // RAG-Specific Evaluation
    // =========================================================================

    /// @brief Full RAG evaluation with all metrics
    /// @param record Inference record with retrieved_contexts
    absl::StatusOr<RAGMetrics> EvaluateRAG(const InferenceRecord& record);

    /// @brief Evaluate context relevance to the query
    /// @param query User query/question
    /// @param contexts Retrieved contexts
    absl::StatusOr<double> EvaluateContextRelevance(
        const std::string& query,
        const std::vector<std::string>& contexts);

    /// @brief Evaluate faithfulness of answer to context
    /// @param answer Generated answer
    /// @param contexts Retrieved contexts
    absl::StatusOr<double> EvaluateFaithfulness(
        const std::string& answer,
        const std::vector<std::string>& contexts);

    /// @brief Evaluate answer relevance to the query
    /// @param query User query
    /// @param answer Generated answer
    absl::StatusOr<double> EvaluateAnswerRelevance(
        const std::string& query,
        const std::string& answer);

    /// @brief Evaluate groundedness of claims in context
    /// @param answer Generated answer
    /// @param contexts Retrieved contexts
    absl::StatusOr<double> EvaluateGroundedness(
        const std::string& answer,
        const std::vector<std::string>& contexts);

    /// @brief Calculate overall RAG score from metrics
    double CalculateOverallScore(const RAGMetrics& metrics) const;

    /// @brief Get configuration
    const RAGEvaluatorConfig& GetConfig() const { return config_; }

private:
    /// @brief Extract sentences/claims from text
    std::vector<std::string> ExtractClaims(const std::string& text);

    /// @brief Check if claim is supported by contexts
    bool IsClaimSupported(const std::string& claim,
                          const std::vector<std::string>& contexts);

    /// @brief Calculate semantic similarity using embeddings
    absl::StatusOr<double> CalculateSimilarity(
        const std::string& text1,
        const std::string& text2);

    /// @brief Calculate keyword overlap (fallback)
    double CalculateKeywordOverlap(
        const std::string& text1,
        const std::string& text2);

    /// @brief Get embedding for text
    absl::StatusOr<std::vector<float>> GetEmbedding(const std::string& text);

    /// @brief Cosine similarity between vectors
    double CosineSimilarity(const std::vector<float>& a,
                            const std::vector<float>& b);

    /// @brief Use LLM for evaluation
    absl::StatusOr<double> LLMEvaluate(
        const std::string& prompt,
        const std::string& criteria);

    RAGEvaluatorConfig config_;

    class Impl;
    std::unique_ptr<Impl> impl_;
};

/// @brief Create a RAG evaluator with default configuration
std::unique_ptr<Evaluator> CreateRAGEvaluator(RAGEvaluatorConfig config = {});

}  // namespace pyflare::eval
