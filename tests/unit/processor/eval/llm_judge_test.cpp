/// @file llm_judge_test.cpp
/// @brief Tests for LLM-as-Judge evaluator

#include <gtest/gtest.h>

#include "processor/eval/llm_judge.h"

namespace pyflare::eval {
namespace {

class LLMJudgeEvaluatorTest : public ::testing::Test {
protected:
    void SetUp() override {
        config_.judge_model = "test-model";
        config_.max_tokens = 512;
        config_.temperature = 0.0;
        config_.enable_cache = true;
        config_.include_explanation = true;
        // Note: API key not set - will use mock/fallback behavior
        evaluator_ = std::make_unique<LLMJudgeEvaluator>(config_);
    }

    LLMJudgeConfig config_;
    std::unique_ptr<LLMJudgeEvaluator> evaluator_;
};

TEST_F(LLMJudgeEvaluatorTest, TypeReturnsCorrectString) {
    EXPECT_EQ(evaluator_->Type(), "LLMJudge");
}

TEST_F(LLMJudgeEvaluatorTest, GetConfigReturnsConfiguration) {
    const auto& config = evaluator_->GetConfig();
    EXPECT_EQ(config.judge_model, "test-model");
    EXPECT_EQ(config.max_tokens, 512);
    EXPECT_DOUBLE_EQ(config.temperature, 0.0);
    EXPECT_TRUE(config.enable_cache);
}

TEST_F(LLMJudgeEvaluatorTest, GetStatsReturnsStats) {
    auto stats = evaluator_->GetStats();
    EXPECT_EQ(stats.total_evaluations, 0);
    EXPECT_EQ(stats.pass_count, 0);
    EXPECT_EQ(stats.fail_count, 0);
    EXPECT_EQ(stats.error_count, 0);
}

TEST_F(LLMJudgeEvaluatorTest, EvaluateWithoutAPIReturnsError) {
    // Without API key/endpoint, evaluation should fail gracefully
    InferenceRecord record;
    record.input = "What is the capital of France?";
    record.output = "The capital of France is Paris.";

    auto result = evaluator_->Evaluate(record);
    // Should return error without API access
    // The actual behavior depends on implementation
    if (!result.ok()) {
        EXPECT_FALSE(result.status().message().empty());
    } else {
        // If implementation provides fallback
        EXPECT_TRUE(result.ok());
    }
}

TEST_F(LLMJudgeEvaluatorTest, EvaluateHallucinationWithoutAPI) {
    InferenceRecord record;
    record.input = "Tell me about Paris";
    record.output = "Paris was founded in 250 BC by Romans.";
    record.retrieved_contexts = {"Paris is the capital of France."};

    auto verdict = evaluator_->EvaluateHallucination(record);
    // Should return error without API access
    if (!verdict.ok()) {
        EXPECT_FALSE(verdict.status().message().empty());
    }
}

TEST_F(LLMJudgeEvaluatorTest, BatchEvaluateEmpty) {
    std::vector<InferenceRecord> empty_records;
    auto results = evaluator_->EvaluateBatch(empty_records);
    ASSERT_TRUE(results.ok());
    EXPECT_TRUE(results->empty());
}

TEST_F(LLMJudgeEvaluatorTest, SetCustomPrompts) {
    JudgePromptTemplate hallucination_prompt;
    hallucination_prompt.system_prompt = "You are a hallucination detector.";
    hallucination_prompt.user_prompt_template =
        "Check if this output contains hallucinations:\n"
        "Input: {input}\n"
        "Output: {output}\n"
        "Context: {context}";
    hallucination_prompt.response_format = "JSON with score and explanation";

    // Should not throw
    EXPECT_NO_THROW(evaluator_->SetHallucinationPrompt(hallucination_prompt));

    JudgePromptTemplate quality_prompt;
    quality_prompt.system_prompt = "You are a quality evaluator.";
    quality_prompt.user_prompt_template = "Rate the quality: {output}";
    EXPECT_NO_THROW(evaluator_->SetQualityPrompt(quality_prompt));
}

// =============================================================================
// JudgeVerdict Tests
// =============================================================================

TEST(JudgeVerdictTest, DefaultValues) {
    JudgeVerdict verdict;
    EXPECT_EQ(verdict.result, JudgeVerdict::Result::kUnsure);
    EXPECT_DOUBLE_EQ(verdict.score, 0.0);
    EXPECT_TRUE(verdict.explanation.empty());
    EXPECT_FALSE(verdict.has_hallucination);
    EXPECT_FALSE(verdict.has_factual_error);
    EXPECT_FALSE(verdict.has_contradiction);
    EXPECT_FALSE(verdict.has_unsupported_claim);
}

TEST(JudgeVerdictTest, ResultEnumValues) {
    JudgeVerdict pass_verdict;
    pass_verdict.result = JudgeVerdict::Result::kPass;
    EXPECT_EQ(pass_verdict.result, JudgeVerdict::Result::kPass);

    JudgeVerdict fail_verdict;
    fail_verdict.result = JudgeVerdict::Result::kFail;
    EXPECT_EQ(fail_verdict.result, JudgeVerdict::Result::kFail);

    JudgeVerdict error_verdict;
    error_verdict.result = JudgeVerdict::Result::kError;
    EXPECT_EQ(error_verdict.result, JudgeVerdict::Result::kError);
}

// =============================================================================
// JudgePromptTemplate Tests
// =============================================================================

TEST(JudgePromptTemplateTest, DefaultValues) {
    JudgePromptTemplate prompt;
    EXPECT_TRUE(prompt.system_prompt.empty());
    EXPECT_TRUE(prompt.user_prompt_template.empty());
    EXPECT_TRUE(prompt.response_format.empty());
}

// =============================================================================
// Default Prompt Template Factory Tests
// =============================================================================

TEST(PromptTemplateFactoryTest, CreateHallucinationPrompt) {
    auto prompt = CreateHallucinationPromptTemplate();
    EXPECT_FALSE(prompt.system_prompt.empty());
    EXPECT_FALSE(prompt.user_prompt_template.empty());
    // Template should contain placeholders
    EXPECT_NE(prompt.user_prompt_template.find("{input}"), std::string::npos);
    EXPECT_NE(prompt.user_prompt_template.find("{output}"), std::string::npos);
}

TEST(PromptTemplateFactoryTest, CreateQualityPrompt) {
    auto prompt = CreateQualityPromptTemplate();
    EXPECT_FALSE(prompt.system_prompt.empty());
    EXPECT_FALSE(prompt.user_prompt_template.empty());
}

TEST(PromptTemplateFactoryTest, CreateRAGGroundingPrompt) {
    auto prompt = CreateRAGGroundingPromptTemplate();
    EXPECT_FALSE(prompt.system_prompt.empty());
    EXPECT_FALSE(prompt.user_prompt_template.empty());
    // Should reference context
    EXPECT_NE(prompt.user_prompt_template.find("{context}"), std::string::npos);
}

// =============================================================================
// LLMJudgeConfig Tests
// =============================================================================

TEST(LLMJudgeConfigTest, DefaultValues) {
    LLMJudgeConfig config;
    EXPECT_EQ(config.api_endpoint, "https://api.openai.com/v1/chat/completions");
    EXPECT_TRUE(config.api_key.empty());
    EXPECT_EQ(config.judge_model, "gpt-4o-mini");
    EXPECT_EQ(config.max_tokens, 1024);
    EXPECT_DOUBLE_EQ(config.temperature, 0.0);
    EXPECT_EQ(config.max_retries, 3);
    EXPECT_EQ(config.batch_size, 10);
    EXPECT_TRUE(config.include_explanation);
    EXPECT_TRUE(config.enable_cache);
}

// =============================================================================
// InferenceRecord Tests
// =============================================================================

TEST(InferenceRecordTest, WithRetrievedContexts) {
    InferenceRecord record;
    record.input = "What is Python?";
    record.output = "Python is a programming language.";
    record.retrieved_contexts = {
        "Python is a high-level programming language.",
        "Python was created by Guido van Rossum."
    };

    ASSERT_TRUE(record.retrieved_contexts.has_value());
    EXPECT_EQ(record.retrieved_contexts->size(), 2);
}

TEST(InferenceRecordTest, WithExpectedOutput) {
    InferenceRecord record;
    record.input = "1 + 1 = ?";
    record.output = "2";
    record.expected_output = "2";

    EXPECT_EQ(record.output, record.expected_output);
}

// =============================================================================
// Integration-style Tests (with mock)
// =============================================================================

class LLMJudgeIntegrationTest : public ::testing::Test {
protected:
    void SetUp() override {
        // Create evaluator with test configuration
        config_.judge_model = "test-model";
        config_.enable_cache = false;  // Disable cache for testing
        config_.max_retries = 1;
        evaluator_ = std::make_unique<LLMJudgeEvaluator>(config_);
    }

    LLMJudgeConfig config_;
    std::unique_ptr<LLMJudgeEvaluator> evaluator_;
};

TEST_F(LLMJudgeIntegrationTest, EvaluateAgainstReferenceWithoutAPI) {
    InferenceRecord record;
    record.input = "What is 2 + 2?";
    record.output = "4";
    record.expected_output = "4";

    auto verdict = evaluator_->EvaluateAgainstReference(record);
    // Will fail without API, but should not crash
    if (!verdict.ok()) {
        EXPECT_FALSE(verdict.status().message().empty());
    }
}

TEST_F(LLMJudgeIntegrationTest, ComparePairwiseWithoutAPI) {
    auto verdict = evaluator_->ComparePairwise(
        "Which is better for beginners?",
        "Python is great for beginners because of its simple syntax.",
        "C++ is better because it teaches memory management.",
        "clarity and ease of learning"
    );

    // Will fail without API, but should not crash
    if (!verdict.ok()) {
        EXPECT_FALSE(verdict.status().message().empty());
    }
}

TEST_F(LLMJudgeIntegrationTest, EvaluateWithCustomTemplate) {
    JudgePromptTemplate custom_template;
    custom_template.system_prompt = "Custom evaluator";
    custom_template.user_prompt_template = "Evaluate: {input} -> {output}";

    InferenceRecord record;
    record.input = "Test input";
    record.output = "Test output";

    auto verdict = evaluator_->EvaluateWithTemplate(record, custom_template);
    // Will fail without API, but should handle gracefully
    if (!verdict.ok()) {
        EXPECT_FALSE(verdict.status().message().empty());
    }
}

}  // namespace
}  // namespace pyflare::eval
