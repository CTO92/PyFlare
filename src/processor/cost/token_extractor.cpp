/// @file token_extractor.cpp
/// @brief Token extraction implementation

#include "processor/cost/token_extractor.h"

#include <algorithm>
#include <cctype>
#include <regex>

#include <nlohmann/json.hpp>
#include <spdlog/spdlog.h>

namespace pyflare::cost {

using json = nlohmann::json;

// =============================================================================
// OpenAI Token Extractor
// =============================================================================

absl::StatusOr<TokenUsage> OpenAITokenExtractor::Extract(
    const std::string& response_body) const {

    try {
        json j = json::parse(response_body);
        TokenUsage usage;
        usage.provider = "OpenAI";

        // Extract model if present
        if (j.contains("model")) {
            usage.model_id = j["model"].get<std::string>();
        }

        // Standard usage object
        if (j.contains("usage") && j["usage"].is_object()) {
            const auto& u = j["usage"];

            usage.input_tokens = u.value("prompt_tokens", int64_t{0});
            usage.output_tokens = u.value("completion_tokens", int64_t{0});
            usage.total_tokens = u.value("total_tokens", int64_t{0});

            // Extended metrics for newer models
            if (u.contains("prompt_tokens_details")) {
                const auto& details = u["prompt_tokens_details"];
                usage.cached_tokens = details.value("cached_tokens", int64_t{0});
                usage.audio_tokens = details.value("audio_tokens", int64_t{0});
            }

            if (u.contains("completion_tokens_details")) {
                const auto& details = u["completion_tokens_details"];
                usage.reasoning_tokens = details.value("reasoning_tokens", int64_t{0});
                usage.audio_tokens += details.value("audio_tokens", int64_t{0});
            }

            // Store raw usage
            for (auto& [key, value] : u.items()) {
                if (value.is_number()) {
                    usage.raw_usage[key] = std::to_string(value.get<int64_t>());
                } else if (value.is_string()) {
                    usage.raw_usage[key] = value.get<std::string>();
                }
            }

            return usage;
        }

        // Embedding response format
        if (j.contains("data") && j["data"].is_array()) {
            // Embeddings don't always include token counts in the same place
            if (j.contains("usage")) {
                const auto& u = j["usage"];
                usage.input_tokens = u.value("prompt_tokens", int64_t{0});
                usage.total_tokens = u.value("total_tokens",
                    usage.input_tokens);
                return usage;
            }
        }

        return absl::NotFoundError("No usage information found in OpenAI response");

    } catch (const json::exception& e) {
        return absl::InvalidArgumentError(
            std::string("Failed to parse OpenAI response: ") + e.what());
    }
}

// =============================================================================
// Anthropic Token Extractor
// =============================================================================

absl::StatusOr<TokenUsage> AnthropicTokenExtractor::Extract(
    const std::string& response_body) const {

    try {
        json j = json::parse(response_body);
        TokenUsage usage;
        usage.provider = "Anthropic";

        // Extract model if present
        if (j.contains("model")) {
            usage.model_id = j["model"].get<std::string>();
        }

        // Anthropic usage format
        if (j.contains("usage") && j["usage"].is_object()) {
            const auto& u = j["usage"];

            usage.input_tokens = u.value("input_tokens", int64_t{0});
            usage.output_tokens = u.value("output_tokens", int64_t{0});
            usage.total_tokens = usage.input_tokens + usage.output_tokens;

            // Cache metrics
            if (u.contains("cache_creation_input_tokens")) {
                usage.cached_tokens = u["cache_creation_input_tokens"].get<int64_t>();
            }
            if (u.contains("cache_read_input_tokens")) {
                usage.cached_tokens += u["cache_read_input_tokens"].get<int64_t>();
            }

            return usage;
        }

        return absl::NotFoundError("No usage information found in Anthropic response");

    } catch (const json::exception& e) {
        return absl::InvalidArgumentError(
            std::string("Failed to parse Anthropic response: ") + e.what());
    }
}

// =============================================================================
// Azure OpenAI Token Extractor
// =============================================================================

absl::StatusOr<TokenUsage> AzureOpenAITokenExtractor::Extract(
    const std::string& response_body) const {
    // Azure OpenAI uses same format as OpenAI
    OpenAITokenExtractor openai_extractor;
    auto result = openai_extractor.Extract(response_body);
    if (result.ok()) {
        result->provider = "AzureOpenAI";
    }
    return result;
}

// =============================================================================
// Google Vertex Token Extractor
// =============================================================================

absl::StatusOr<TokenUsage> GoogleVertexTokenExtractor::Extract(
    const std::string& response_body) const {

    try {
        json j = json::parse(response_body);
        TokenUsage usage;
        usage.provider = "GoogleVertex";

        // Gemini format
        if (j.contains("usageMetadata")) {
            const auto& u = j["usageMetadata"];
            usage.input_tokens = u.value("promptTokenCount", int64_t{0});
            usage.output_tokens = u.value("candidatesTokenCount", int64_t{0});
            usage.total_tokens = u.value("totalTokenCount",
                usage.input_tokens + usage.output_tokens);
            return usage;
        }

        // PaLM format
        if (j.contains("metadata") && j["metadata"].contains("tokenMetadata")) {
            const auto& tm = j["metadata"]["tokenMetadata"];
            usage.input_tokens = tm.value("inputTokenCount", int64_t{0});
            usage.output_tokens = tm.value("outputTokenCount", int64_t{0});
            usage.total_tokens = usage.input_tokens + usage.output_tokens;
            return usage;
        }

        return absl::NotFoundError(
            "No usage information found in Google Vertex response");

    } catch (const json::exception& e) {
        return absl::InvalidArgumentError(
            std::string("Failed to parse Google Vertex response: ") + e.what());
    }
}

// =============================================================================
// AWS Bedrock Token Extractor
// =============================================================================

absl::StatusOr<TokenUsage> AWSBedrockTokenExtractor::Extract(
    const std::string& response_body) const {

    try {
        json j = json::parse(response_body);
        TokenUsage usage;
        usage.provider = "AWS_Bedrock";

        // Bedrock wraps various model responses
        // Claude on Bedrock
        if (j.contains("usage")) {
            const auto& u = j["usage"];
            usage.input_tokens = u.value("input_tokens", int64_t{0});
            usage.output_tokens = u.value("output_tokens", int64_t{0});
            usage.total_tokens = usage.input_tokens + usage.output_tokens;
            return usage;
        }

        // Titan format
        if (j.contains("inputTextTokenCount")) {
            usage.input_tokens = j["inputTextTokenCount"].get<int64_t>();
            if (j.contains("results") && j["results"].is_array() &&
                !j["results"].empty()) {
                usage.output_tokens = j["results"][0].value(
                    "tokenCount", int64_t{0});
            }
            usage.total_tokens = usage.input_tokens + usage.output_tokens;
            return usage;
        }

        // Llama format (via Bedrock)
        if (j.contains("prompt_token_count")) {
            usage.input_tokens = j["prompt_token_count"].get<int64_t>();
            usage.output_tokens = j.value("generation_token_count", int64_t{0});
            usage.total_tokens = usage.input_tokens + usage.output_tokens;
            return usage;
        }

        return absl::NotFoundError(
            "No usage information found in AWS Bedrock response");

    } catch (const json::exception& e) {
        return absl::InvalidArgumentError(
            std::string("Failed to parse AWS Bedrock response: ") + e.what());
    }
}

// =============================================================================
// Cohere Token Extractor
// =============================================================================

absl::StatusOr<TokenUsage> CohereTokenExtractor::Extract(
    const std::string& response_body) const {

    try {
        json j = json::parse(response_body);
        TokenUsage usage;
        usage.provider = "Cohere";

        // Cohere chat format
        if (j.contains("meta") && j["meta"].contains("billed_units")) {
            const auto& bu = j["meta"]["billed_units"];
            usage.input_tokens = bu.value("input_tokens", int64_t{0});
            usage.output_tokens = bu.value("output_tokens", int64_t{0});
            usage.total_tokens = usage.input_tokens + usage.output_tokens;
            return usage;
        }

        // Alternative format
        if (j.contains("token_count")) {
            const auto& tc = j["token_count"];
            usage.input_tokens = tc.value("prompt_tokens", int64_t{0});
            usage.output_tokens = tc.value("response_tokens", int64_t{0});
            usage.total_tokens = tc.value("total_tokens",
                usage.input_tokens + usage.output_tokens);
            return usage;
        }

        return absl::NotFoundError(
            "No usage information found in Cohere response");

    } catch (const json::exception& e) {
        return absl::InvalidArgumentError(
            std::string("Failed to parse Cohere response: ") + e.what());
    }
}

// =============================================================================
// Main TokenExtractor
// =============================================================================

TokenExtractor::TokenExtractor() {
    // Register built-in extractors
    extractors_[LLMProvider::kOpenAI] =
        std::make_unique<OpenAITokenExtractor>();
    extractors_[LLMProvider::kAnthropic] =
        std::make_unique<AnthropicTokenExtractor>();
    extractors_[LLMProvider::kAzureOpenAI] =
        std::make_unique<AzureOpenAITokenExtractor>();
    extractors_[LLMProvider::kGoogleVertex] =
        std::make_unique<GoogleVertexTokenExtractor>();
    extractors_[LLMProvider::kAWS_Bedrock] =
        std::make_unique<AWSBedrockTokenExtractor>();
    extractors_[LLMProvider::kCohere] =
        std::make_unique<CohereTokenExtractor>();
}

TokenExtractor::~TokenExtractor() = default;

absl::StatusOr<TokenUsage> TokenExtractor::ExtractAuto(
    const std::string& response_body,
    const std::string& model_hint) const {

    auto provider = DetectProvider(response_body, model_hint);

    if (provider != LLMProvider::kUnknown) {
        auto result = Extract(provider, response_body);
        if (result.ok()) {
            return result;
        }
        spdlog::debug("Provider-specific extraction failed, trying generic");
    }

    // Fall back to generic extraction
    return ExtractGeneric(response_body);
}

absl::StatusOr<TokenUsage> TokenExtractor::Extract(
    LLMProvider provider,
    const std::string& response_body) const {

    auto it = extractors_.find(provider);
    if (it == extractors_.end()) {
        return absl::InvalidArgumentError(
            "No extractor for provider: " + ProviderToString(provider));
    }

    return it->second->Extract(response_body);
}

absl::StatusOr<TokenUsage> TokenExtractor::ExtractFromHeaders(
    const std::unordered_map<std::string, std::string>& headers) const {

    TokenUsage usage;

    // OpenAI-style headers
    auto find_header = [&](const std::string& key) -> std::optional<int64_t> {
        auto it = headers.find(key);
        if (it != headers.end()) {
            try {
                return std::stoll(it->second);
            } catch (...) {}
        }
        // Try lowercase
        std::string lower_key = key;
        std::transform(lower_key.begin(), lower_key.end(),
                       lower_key.begin(), ::tolower);
        it = headers.find(lower_key);
        if (it != headers.end()) {
            try {
                return std::stoll(it->second);
            } catch (...) {}
        }
        return std::nullopt;
    };

    // OpenAI streaming response headers
    if (auto val = find_header("x-request-id")) {
        // Just checking if this is an OpenAI response
        usage.provider = "OpenAI";
    }

    if (auto val = find_header("x-ratelimit-remaining-tokens")) {
        // This gives us rate limit info, not actual usage
    }

    // Anthropic headers
    if (auto val = find_header("anthropic-ratelimit-input-tokens")) {
        usage.provider = "Anthropic";
    }

    // If we found any token counts in headers
    bool found_any = false;

    if (auto val = find_header("x-prompt-tokens")) {
        usage.input_tokens = *val;
        found_any = true;
    }
    if (auto val = find_header("x-completion-tokens")) {
        usage.output_tokens = *val;
        found_any = true;
    }
    if (auto val = find_header("x-total-tokens")) {
        usage.total_tokens = *val;
        found_any = true;
    }

    if (!found_any) {
        return absl::NotFoundError("No token usage found in headers");
    }

    if (usage.total_tokens == 0) {
        usage.total_tokens = usage.input_tokens + usage.output_tokens;
    }

    return usage;
}

absl::StatusOr<TokenUsage> TokenExtractor::ExtractFromSpanAttributes(
    const std::unordered_map<std::string, std::string>& attributes) const {

    TokenUsage usage;

    // OpenTelemetry semantic conventions for LLM
    auto get_attr = [&](const std::string& key) -> std::optional<int64_t> {
        auto it = attributes.find(key);
        if (it != attributes.end()) {
            try {
                return std::stoll(it->second);
            } catch (...) {}
        }
        return std::nullopt;
    };

    // LLM semantic conventions
    if (auto val = get_attr("llm.usage.prompt_tokens")) {
        usage.input_tokens = *val;
    } else if (auto val = get_attr("gen_ai.usage.input_tokens")) {
        usage.input_tokens = *val;
    }

    if (auto val = get_attr("llm.usage.completion_tokens")) {
        usage.output_tokens = *val;
    } else if (auto val = get_attr("gen_ai.usage.output_tokens")) {
        usage.output_tokens = *val;
    }

    if (auto val = get_attr("llm.usage.total_tokens")) {
        usage.total_tokens = *val;
    } else if (auto val = get_attr("gen_ai.usage.total_tokens")) {
        usage.total_tokens = *val;
    }

    // Model info
    auto model_it = attributes.find("llm.model");
    if (model_it == attributes.end()) {
        model_it = attributes.find("gen_ai.request.model");
    }
    if (model_it != attributes.end()) {
        usage.model_id = model_it->second;
        usage.provider = ProviderToString(DetectProviderFromModel(usage.model_id));
    }

    // Provider info
    auto provider_it = attributes.find("llm.vendor");
    if (provider_it == attributes.end()) {
        provider_it = attributes.find("gen_ai.system");
    }
    if (provider_it != attributes.end()) {
        usage.provider = provider_it->second;
    }

    if (usage.total_tokens == 0) {
        usage.total_tokens = usage.input_tokens + usage.output_tokens;
    }

    if (usage.input_tokens == 0 && usage.output_tokens == 0) {
        return absl::NotFoundError("No token usage found in span attributes");
    }

    return usage;
}

LLMProvider TokenExtractor::DetectProvider(
    const std::string& response_body,
    const std::string& model_hint) const {

    // Try model hint first
    if (!model_hint.empty()) {
        auto provider = DetectProviderFromModel(model_hint);
        if (provider != LLMProvider::kUnknown) {
            return provider;
        }
    }

    try {
        json j = json::parse(response_body);

        // Check for model field
        if (j.contains("model")) {
            auto model = j["model"].get<std::string>();
            auto provider = DetectProviderFromModel(model);
            if (provider != LLMProvider::kUnknown) {
                return provider;
            }
        }

        // Anthropic specific fields
        if (j.contains("stop_reason") ||
            (j.contains("usage") && j["usage"].contains("input_tokens"))) {
            return LLMProvider::kAnthropic;
        }

        // Google Vertex specific
        if (j.contains("usageMetadata") || j.contains("candidates")) {
            return LLMProvider::kGoogleVertex;
        }

        // Cohere specific
        if (j.contains("meta") && j["meta"].contains("billed_units")) {
            return LLMProvider::kCohere;
        }

        // OpenAI format (most common fallback)
        if (j.contains("usage") && j["usage"].contains("prompt_tokens")) {
            return LLMProvider::kOpenAI;
        }

        // AWS Bedrock Claude
        if (j.contains("completion") || j.contains("inputTextTokenCount")) {
            return LLMProvider::kAWS_Bedrock;
        }

    } catch (const json::exception&) {
        // Not valid JSON
    }

    return LLMProvider::kUnknown;
}

LLMProvider TokenExtractor::DetectProviderFromModel(const std::string& model_name) {
    std::string lower = model_name;
    std::transform(lower.begin(), lower.end(), lower.begin(), ::tolower);

    // OpenAI models
    if (lower.find("gpt-") != std::string::npos ||
        lower.find("o1") != std::string::npos ||
        lower.find("davinci") != std::string::npos ||
        lower.find("curie") != std::string::npos ||
        lower.find("babbage") != std::string::npos ||
        lower.find("ada") != std::string::npos ||
        lower.find("text-embedding") != std::string::npos) {
        return LLMProvider::kOpenAI;
    }

    // Anthropic models
    if (lower.find("claude") != std::string::npos) {
        return LLMProvider::kAnthropic;
    }

    // Google models
    if (lower.find("gemini") != std::string::npos ||
        lower.find("palm") != std::string::npos ||
        lower.find("bison") != std::string::npos) {
        return LLMProvider::kGoogleVertex;
    }

    // AWS Bedrock model IDs
    if (lower.find("anthropic.claude") != std::string::npos ||
        lower.find("amazon.titan") != std::string::npos ||
        lower.find("meta.llama") != std::string::npos) {
        return LLMProvider::kAWS_Bedrock;
    }

    // Cohere models
    if (lower.find("command") != std::string::npos ||
        lower.find("cohere") != std::string::npos) {
        return LLMProvider::kCohere;
    }

    return LLMProvider::kUnknown;
}

std::string TokenExtractor::ProviderToString(LLMProvider provider) {
    switch (provider) {
        case LLMProvider::kOpenAI: return "OpenAI";
        case LLMProvider::kAnthropic: return "Anthropic";
        case LLMProvider::kAzureOpenAI: return "AzureOpenAI";
        case LLMProvider::kGoogleVertex: return "GoogleVertex";
        case LLMProvider::kAWS_Bedrock: return "AWS_Bedrock";
        case LLMProvider::kCohere: return "Cohere";
        case LLMProvider::kHuggingFace: return "HuggingFace";
        case LLMProvider::kCustom: return "Custom";
        case LLMProvider::kUnknown:
        default: return "Unknown";
    }
}

LLMProvider TokenExtractor::StringToProvider(const std::string& str) {
    std::string lower = str;
    std::transform(lower.begin(), lower.end(), lower.begin(), ::tolower);

    if (lower == "openai") return LLMProvider::kOpenAI;
    if (lower == "anthropic") return LLMProvider::kAnthropic;
    if (lower == "azureopenai" || lower == "azure") return LLMProvider::kAzureOpenAI;
    if (lower == "googlevertex" || lower == "google" || lower == "vertex")
        return LLMProvider::kGoogleVertex;
    if (lower == "aws_bedrock" || lower == "bedrock")
        return LLMProvider::kAWS_Bedrock;
    if (lower == "cohere") return LLMProvider::kCohere;
    if (lower == "huggingface") return LLMProvider::kHuggingFace;

    return LLMProvider::kUnknown;
}

void TokenExtractor::RegisterExtractor(
    std::unique_ptr<TokenExtractorInterface> extractor) {
    if (extractor) {
        extractors_[extractor->Provider()] = std::move(extractor);
    }
}

absl::StatusOr<TokenUsage> TokenExtractor::ExtractGeneric(
    const std::string& response_body) const {

    try {
        json j = json::parse(response_body);
        TokenUsage usage;

        // Try common field names
        auto extract_number = [&](const json& obj, const std::vector<std::string>& keys)
            -> std::optional<int64_t> {
            for (const auto& key : keys) {
                if (obj.contains(key) && obj[key].is_number()) {
                    return obj[key].get<int64_t>();
                }
            }
            return std::nullopt;
        };

        // Look for usage object at various paths
        std::vector<const json*> usage_objects;
        if (j.contains("usage")) {
            usage_objects.push_back(&j["usage"]);
        }
        if (j.contains("meta") && j["meta"].contains("usage")) {
            usage_objects.push_back(&j["meta"]["usage"]);
        }
        if (j.contains("metadata") && j["metadata"].contains("usage")) {
            usage_objects.push_back(&j["metadata"]["usage"]);
        }
        usage_objects.push_back(&j);  // Also check root

        for (const json* obj : usage_objects) {
            // Input tokens
            auto input = extract_number(*obj, {
                "prompt_tokens", "input_tokens", "promptTokenCount",
                "inputTokenCount", "prompt_token_count"
            });
            if (input) {
                usage.input_tokens = *input;
            }

            // Output tokens
            auto output = extract_number(*obj, {
                "completion_tokens", "output_tokens", "completionTokenCount",
                "outputTokenCount", "generation_token_count", "candidatesTokenCount"
            });
            if (output) {
                usage.output_tokens = *output;
            }

            // Total tokens
            auto total = extract_number(*obj, {
                "total_tokens", "totalTokenCount", "totalTokens"
            });
            if (total) {
                usage.total_tokens = *total;
            }

            if (usage.input_tokens > 0 || usage.output_tokens > 0) {
                break;  // Found something
            }
        }

        if (usage.total_tokens == 0) {
            usage.total_tokens = usage.input_tokens + usage.output_tokens;
        }

        if (usage.input_tokens == 0 && usage.output_tokens == 0 &&
            usage.total_tokens == 0) {
            return absl::NotFoundError("No token usage found in response");
        }

        usage.provider = "Unknown";
        return usage;

    } catch (const json::exception& e) {
        return absl::InvalidArgumentError(
            std::string("Failed to parse response: ") + e.what());
    }
}

// =============================================================================
// Utility Functions
// =============================================================================

int64_t EstimateTokenCount(const std::string& text, const std::string& method) {
    if (text.empty()) {
        return 0;
    }

    if (method == "words") {
        // Count words (space-separated)
        int64_t word_count = 0;
        bool in_word = false;

        for (char c : text) {
            if (std::isspace(static_cast<unsigned char>(c))) {
                in_word = false;
            } else if (!in_word) {
                in_word = true;
                ++word_count;
            }
        }

        // Approximate: 1.3 tokens per word for English
        return static_cast<int64_t>(word_count * 1.3);
    }

    // Default: character-based estimation
    // Approximate: 4 characters per token for English
    return static_cast<int64_t>(text.size() / 4.0 + 0.5);
}

}  // namespace pyflare::cost
