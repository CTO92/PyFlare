#pragma once

/// @file llm_judge.h
/// @brief LLM-as-Judge evaluator for hallucination and quality assessment

#include <chrono>
#include <functional>
#include <memory>
#include <string>
#include <vector>

#include <absl/status/status.h>
#include <absl/status/statusor.h>

#include "processor/eval/evaluator.h"

namespace pyflare::eval {

/// @brief Configuration for LLM Judge
struct LLMJudgeConfig {
    /// LLM API endpoint (OpenAI-compatible)
    std::string api_endpoint = "https://api.openai.com/v1/chat/completions";

    /// API key (or use environment variable)
    std::string api_key;

    /// Model to use for judging
    std::string judge_model = "gpt-4o-mini";

    /// Maximum tokens for judge response
    size_t max_tokens = 1024;

    /// Temperature for judge (lower = more consistent)
    double temperature = 0.0;

    /// Request timeout
    std::chrono::seconds timeout = std::chrono::seconds(30);

    /// Maximum retries on failure
    size_t max_retries = 3;

    /// Batch size for parallel evaluation
    size_t batch_size = 10;

    /// Whether to include explanation in results
    bool include_explanation = true;

    /// Cache judge results
    bool enable_cache = true;

    /// Cache TTL
    std::chrono::hours cache_ttl = std::chrono::hours(24);
};

/// @brief Verdict from LLM judge
struct JudgeVerdict {
    enum class Result {
        kPass,      ///< Output is acceptable
        kFail,      ///< Output has issues
        kUnsure,    ///< Judge is uncertain
        kError      ///< Evaluation error
    };

    Result result = Result::kUnsure;
    double score = 0.0;             ///< 0.0 - 1.0 confidence
    std::string explanation;
    std::string raw_response;

    // Specific issue types detected
    bool has_hallucination = false;
    bool has_factual_error = false;
    bool has_contradiction = false;
    bool has_unsupported_claim = false;
};

/// @brief Prompt template for LLM judge
struct JudgePromptTemplate {
    std::string system_prompt;
    std::string user_prompt_template;  ///< Use {input}, {output}, {context} placeholders
    std::string response_format;       ///< Expected response format description
};

/// @brief LLM-as-Judge evaluator for quality assessment
///
/// Uses an LLM to evaluate outputs for:
/// - Hallucination detection (claims not in context)
/// - Factual accuracy
/// - Contradiction detection
/// - Quality and relevance
///
/// Supports various evaluation modes:
/// - Pairwise comparison (A vs B)
/// - Reference-based (output vs expected)
/// - Context-grounded (output vs retrieved docs)
///
/// Example usage:
/// @code
///   LLMJudgeConfig config;
///   config.api_key = "sk-...";
///   config.judge_model = "gpt-4o-mini";
///   auto judge = std::make_unique<LLMJudgeEvaluator>(config);
///
///   InferenceRecord record;
///   record.input = "What is the capital of France?";
///   record.output = "The capital of France is Paris, founded in 250 BC.";
///   record.retrieved_contexts = {"Paris is the capital of France."};
///
///   auto result = judge->Evaluate(record);
///   // result.score ~= 0.5 (correct answer but hallucinated founding date)
/// @endcode
class LLMJudgeEvaluator : public Evaluator {
public:
    explicit LLMJudgeEvaluator(LLMJudgeConfig config = {});
    ~LLMJudgeEvaluator() override;

    // Disable copy
    LLMJudgeEvaluator(const LLMJudgeEvaluator&) = delete;
    LLMJudgeEvaluator& operator=(const LLMJudgeEvaluator&) = delete;

    /// @brief Initialize the evaluator (verify API access)
    absl::Status Initialize();

    /// @brief Evaluate a single inference
    absl::StatusOr<EvalResult> Evaluate(const InferenceRecord& record) override;

    /// @brief Batch evaluation with parallel processing
    absl::StatusOr<std::vector<EvalResult>> EvaluateBatch(
        const std::vector<InferenceRecord>& records) override;

    std::string Type() const override { return "LLMJudge"; }

    // =========================================================================
    // Specialized Evaluation Methods
    // =========================================================================

    /// @brief Evaluate specifically for hallucination
    /// @param record Inference record to evaluate
    /// @return Detailed verdict with hallucination analysis
    absl::StatusOr<JudgeVerdict> EvaluateHallucination(
        const InferenceRecord& record);

    /// @brief Evaluate output against expected/reference
    /// @param record Record with expected_output set
    absl::StatusOr<JudgeVerdict> EvaluateAgainstReference(
        const InferenceRecord& record);

    /// @brief Pairwise comparison of two outputs
    /// @param input Original input/question
    /// @param output_a First output to compare
    /// @param output_b Second output to compare
    /// @param criteria Evaluation criteria
    /// @return Verdict indicating which output is better
    absl::StatusOr<JudgeVerdict> ComparePairwise(
        const std::string& input,
        const std::string& output_a,
        const std::string& output_b,
        const std::string& criteria = "");

    /// @brief Evaluate with custom prompt template
    /// @param record Inference record
    /// @param prompt_template Custom prompt template
    absl::StatusOr<JudgeVerdict> EvaluateWithTemplate(
        const InferenceRecord& record,
        const JudgePromptTemplate& prompt_template);

    // =========================================================================
    // Configuration
    // =========================================================================

    /// @brief Set custom hallucination prompt
    void SetHallucinationPrompt(const JudgePromptTemplate& prompt);

    /// @brief Set custom quality prompt
    void SetQualityPrompt(const JudgePromptTemplate& prompt);

    /// @brief Get configuration
    const LLMJudgeConfig& GetConfig() const { return config_; }

    /// @brief Get evaluation statistics
    struct Stats {
        size_t total_evaluations = 0;
        size_t pass_count = 0;
        size_t fail_count = 0;
        size_t error_count = 0;
        size_t cache_hits = 0;
        double avg_latency_ms = 0.0;
        double avg_score = 0.0;
    };
    Stats GetStats() const { return stats_; }

private:
    /// @brief Call the LLM API
    absl::StatusOr<std::string> CallLLM(
        const std::string& system_prompt,
        const std::string& user_prompt);

    /// @brief Parse LLM response to extract verdict
    JudgeVerdict ParseVerdict(const std::string& response);

    /// @brief Build hallucination detection prompt
    std::string BuildHallucinationPrompt(const InferenceRecord& record);

    /// @brief Build reference comparison prompt
    std::string BuildReferencePrompt(const InferenceRecord& record);

    /// @brief Get cache key for a record
    std::string GetCacheKey(const InferenceRecord& record,
                            const std::string& eval_type);

    LLMJudgeConfig config_;
    JudgePromptTemplate hallucination_prompt_;
    JudgePromptTemplate quality_prompt_;
    Stats stats_;

    class Impl;
    std::unique_ptr<Impl> impl_;
};

/// @brief Create default hallucination detection prompt
JudgePromptTemplate CreateHallucinationPromptTemplate();

/// @brief Create default quality evaluation prompt
JudgePromptTemplate CreateQualityPromptTemplate();

/// @brief Create default RAG grounding prompt
JudgePromptTemplate CreateRAGGroundingPromptTemplate();

}  // namespace pyflare::eval
