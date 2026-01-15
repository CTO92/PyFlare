#pragma once

/// @file token_extractor.h
/// @brief Token extraction from various LLM provider response formats

#include <memory>
#include <optional>
#include <string>
#include <unordered_map>
#include <vector>

#include <absl/status/statusor.h>

namespace pyflare::cost {

/// @brief Token usage information extracted from LLM responses
struct TokenUsage {
    int64_t input_tokens = 0;       ///< Prompt/input tokens
    int64_t output_tokens = 0;      ///< Completion/output tokens
    int64_t total_tokens = 0;       ///< Total tokens used

    // Extended metrics (provider-specific)
    int64_t cached_tokens = 0;      ///< Tokens from cache (reduced cost)
    int64_t reasoning_tokens = 0;   ///< Reasoning/thinking tokens (o1, etc.)
    int64_t audio_tokens = 0;       ///< Audio processing tokens
    int64_t image_tokens = 0;       ///< Image processing tokens

    std::string model_id;
    std::string provider;

    // Raw data for debugging
    std::unordered_map<std::string, std::string> raw_usage;
};

/// @brief Provider types for token extraction
enum class LLMProvider {
    kUnknown,
    kOpenAI,
    kAnthropic,
    kAzureOpenAI,
    kGoogleVertex,
    kAWS_Bedrock,
    kCohere,
    kHuggingFace,
    kCustom
};

/// @brief Token extractor interface
class TokenExtractorInterface {
public:
    virtual ~TokenExtractorInterface() = default;

    /// @brief Extract token usage from response body
    /// @param response_body JSON response from LLM provider
    /// @return Extracted token usage
    virtual absl::StatusOr<TokenUsage> Extract(
        const std::string& response_body) const = 0;

    /// @brief Get the provider this extractor handles
    virtual LLMProvider Provider() const = 0;

    /// @brief Get provider name as string
    virtual std::string ProviderName() const = 0;
};

/// @brief OpenAI token extractor
///
/// Handles responses from:
/// - OpenAI Chat Completions API
/// - OpenAI Completions API (legacy)
/// - OpenAI Embeddings API
///
/// Example response format:
/// @code
///   {
///     "usage": {
///       "prompt_tokens": 100,
///       "completion_tokens": 50,
///       "total_tokens": 150
///     }
///   }
/// @endcode
class OpenAITokenExtractor : public TokenExtractorInterface {
public:
    absl::StatusOr<TokenUsage> Extract(
        const std::string& response_body) const override;

    LLMProvider Provider() const override { return LLMProvider::kOpenAI; }
    std::string ProviderName() const override { return "OpenAI"; }
};

/// @brief Anthropic token extractor
///
/// Handles responses from Claude API:
/// @code
///   {
///     "usage": {
///       "input_tokens": 100,
///       "output_tokens": 50
///     }
///   }
/// @endcode
class AnthropicTokenExtractor : public TokenExtractorInterface {
public:
    absl::StatusOr<TokenUsage> Extract(
        const std::string& response_body) const override;

    LLMProvider Provider() const override { return LLMProvider::kAnthropic; }
    std::string ProviderName() const override { return "Anthropic"; }
};

/// @brief Azure OpenAI token extractor
class AzureOpenAITokenExtractor : public TokenExtractorInterface {
public:
    absl::StatusOr<TokenUsage> Extract(
        const std::string& response_body) const override;

    LLMProvider Provider() const override { return LLMProvider::kAzureOpenAI; }
    std::string ProviderName() const override { return "AzureOpenAI"; }
};

/// @brief Google Vertex AI token extractor
///
/// Handles responses from:
/// - PaLM API
/// - Gemini API
class GoogleVertexTokenExtractor : public TokenExtractorInterface {
public:
    absl::StatusOr<TokenUsage> Extract(
        const std::string& response_body) const override;

    LLMProvider Provider() const override { return LLMProvider::kGoogleVertex; }
    std::string ProviderName() const override { return "GoogleVertex"; }
};

/// @brief AWS Bedrock token extractor
///
/// Handles responses from various Bedrock models:
/// - Claude (Anthropic)
/// - Titan (AWS)
/// - Llama (Meta)
class AWSBedrockTokenExtractor : public TokenExtractorInterface {
public:
    absl::StatusOr<TokenUsage> Extract(
        const std::string& response_body) const override;

    LLMProvider Provider() const override { return LLMProvider::kAWS_Bedrock; }
    std::string ProviderName() const override { return "AWS_Bedrock"; }
};

/// @brief Cohere token extractor
class CohereTokenExtractor : public TokenExtractorInterface {
public:
    absl::StatusOr<TokenUsage> Extract(
        const std::string& response_body) const override;

    LLMProvider Provider() const override { return LLMProvider::kCohere; }
    std::string ProviderName() const override { return "Cohere"; }
};

/// @brief Main token extractor with auto-detection and manual extraction
///
/// Provides:
/// - Automatic provider detection from response format
/// - Provider-specific extraction
/// - Fallback extraction from common attributes
/// - Header-based extraction for streaming responses
///
/// Example usage:
/// @code
///   TokenExtractor extractor;
///
///   // Auto-detect provider and extract
///   auto usage = extractor.ExtractAuto(response_body);
///
///   // Or specify provider explicitly
///   auto usage = extractor.Extract(LLMProvider::kOpenAI, response_body);
///
///   // Extract from response headers (for streaming)
///   auto usage = extractor.ExtractFromHeaders(headers);
/// @endcode
class TokenExtractor {
public:
    TokenExtractor();
    ~TokenExtractor();

    /// @brief Auto-detect provider and extract tokens
    /// @param response_body JSON response body
    /// @param model_hint Optional model name hint for detection
    absl::StatusOr<TokenUsage> ExtractAuto(
        const std::string& response_body,
        const std::string& model_hint = "") const;

    /// @brief Extract tokens using specific provider
    /// @param provider LLM provider
    /// @param response_body JSON response body
    absl::StatusOr<TokenUsage> Extract(
        LLMProvider provider,
        const std::string& response_body) const;

    /// @brief Extract tokens from HTTP headers (streaming responses)
    /// @param headers Response headers
    absl::StatusOr<TokenUsage> ExtractFromHeaders(
        const std::unordered_map<std::string, std::string>& headers) const;

    /// @brief Extract tokens from OpenTelemetry span attributes
    /// @param attributes Span attributes
    absl::StatusOr<TokenUsage> ExtractFromSpanAttributes(
        const std::unordered_map<std::string, std::string>& attributes) const;

    /// @brief Detect provider from response format
    /// @param response_body JSON response body
    /// @param model_hint Optional model name hint
    LLMProvider DetectProvider(const std::string& response_body,
                                const std::string& model_hint = "") const;

    /// @brief Detect provider from model name
    /// @param model_name Model identifier
    static LLMProvider DetectProviderFromModel(const std::string& model_name);

    /// @brief Convert provider enum to string
    static std::string ProviderToString(LLMProvider provider);

    /// @brief Convert string to provider enum
    static LLMProvider StringToProvider(const std::string& str);

    /// @brief Register a custom token extractor
    void RegisterExtractor(std::unique_ptr<TokenExtractorInterface> extractor);

private:
    /// @brief Try generic extraction from common fields
    absl::StatusOr<TokenUsage> ExtractGeneric(
        const std::string& response_body) const;

    std::unordered_map<LLMProvider, std::unique_ptr<TokenExtractorInterface>>
        extractors_;
};

/// @brief Estimate token count from text (approximate)
///
/// Uses simple heuristics:
/// - ~4 characters per token for English
/// - ~1.3 tokens per word for English
///
/// For accurate counts, use the actual tokenizer for the model.
///
/// @param text Input text
/// @param method Estimation method: "chars" or "words"
/// @return Estimated token count
int64_t EstimateTokenCount(const std::string& text,
                           const std::string& method = "chars");

}  // namespace pyflare::cost
