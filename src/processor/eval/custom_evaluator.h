#pragma once

/// @file custom_evaluator.h
/// @brief Custom evaluator framework for user-defined evaluation logic
///
/// Supports:
/// - Rule-based evaluators (regex, conditions)
/// - Script-based evaluators (Python, JavaScript)
/// - LLM-based evaluators (custom prompts)
/// - External API evaluators (webhooks)

#include <chrono>
#include <functional>
#include <memory>
#include <mutex>
#include <string>
#include <unordered_map>
#include <variant>
#include <vector>

#include <absl/status/status.h>
#include <absl/status/statusor.h>

#include "processor/eval/evaluator.h"

namespace pyflare::eval {

/// @brief Types of custom evaluators
enum class CustomEvaluatorType {
    kRule,      ///< Rule-based (regex, conditions)
    kScript,    ///< Script-based (Python, JavaScript)
    kLLM,       ///< LLM-based (custom prompt)
    kExternal   ///< External API (webhook)
};

/// @brief Rule configuration for rule-based evaluators
struct RuleConfig {
    /// Condition expression (supports basic logic)
    /// Variables: input, output, input_len, output_len
    std::string condition_expression;

    /// Regex patterns to match against output
    std::vector<std::string> regex_patterns;

    /// Required keywords in output
    std::vector<std::string> required_keywords;

    /// Forbidden keywords in output
    std::vector<std::string> forbidden_keywords;

    /// Length constraints
    std::optional<size_t> min_length;
    std::optional<size_t> max_length;

    /// Score mapping based on matches
    double pass_score = 1.0;
    double fail_score = 0.0;
};

/// @brief Script configuration for script-based evaluators
struct ScriptConfig {
    /// Script language
    enum class Language {
        kPython,
        kJavaScript
    };
    Language language = Language::kPython;

    /// Script content
    std::string script;

    /// Script file path (alternative to inline script)
    std::string script_path;

    /// Timeout for script execution
    std::chrono::seconds timeout = std::chrono::seconds(30);

    /// Environment variables
    std::unordered_map<std::string, std::string> environment;
};

/// @brief LLM configuration for LLM-based evaluators
struct LLMEvaluatorConfig {
    /// LLM API endpoint
    std::string api_endpoint = "https://api.openai.com/v1/chat/completions";

    /// API key
    std::string api_key;

    /// Model to use
    std::string model = "gpt-4o-mini";

    /// System prompt for evaluation
    std::string system_prompt;

    /// User prompt template
    /// Placeholders: {input}, {output}, {expected}, {context}
    std::string user_prompt_template;

    /// Temperature (0 = deterministic)
    double temperature = 0.0;

    /// Maximum tokens
    size_t max_tokens = 256;

    /// Expected response format
    std::string response_format;  ///< "json" or "text"

    /// Score extraction regex
    std::string score_extraction_pattern = R"(score[:\s]*(\d+(?:\.\d+)?))";
};

/// @brief External API configuration for webhook-based evaluators
struct ExternalConfig {
    /// Webhook URL
    std::string url;

    /// HTTP method
    std::string method = "POST";

    /// Authentication
    struct Auth {
        enum class Type {
            kNone,
            kApiKey,
            kBearer,
            kBasic
        };
        Type type = Type::kNone;
        std::string token;
        std::string header_name = "Authorization";
    };
    Auth auth;

    /// Request timeout
    std::chrono::seconds timeout = std::chrono::seconds(30);

    /// Retry configuration
    size_t max_retries = 3;

    /// Response parsing
    std::string score_json_path = "$.score";
    std::string verdict_json_path = "$.verdict";
    std::string explanation_json_path = "$.explanation";
};

/// @brief Custom evaluator definition
struct CustomEvaluatorDefinition {
    std::string name;
    std::string description;
    std::string version = "1.0.0";

    CustomEvaluatorType type = CustomEvaluatorType::kRule;

    /// Type-specific configuration
    std::variant<RuleConfig, ScriptConfig, LLMEvaluatorConfig, ExternalConfig> config;

    /// Metadata
    std::unordered_map<std::string, std::string> metadata;

    /// Tags for categorization
    std::vector<std::string> tags;

    /// Enable/disable flag
    bool enabled = true;

    /// Creation timestamp
    std::chrono::system_clock::time_point created_at;

    /// Last update timestamp
    std::chrono::system_clock::time_point updated_at;
};

/// @brief Registry for custom evaluators
///
/// Manages custom evaluator definitions and instantiation.
///
/// Example:
/// @code
///   CustomEvaluatorRegistry registry;
///
///   // Register a rule-based evaluator
///   CustomEvaluatorDefinition def;
///   def.name = "length_check";
///   def.type = CustomEvaluatorType::kRule;
///   RuleConfig rule;
///   rule.min_length = 10;
///   rule.max_length = 1000;
///   def.config = rule;
///   registry.Register(def);
///
///   // Create evaluator instance
///   auto evaluator = registry.Create("length_check");
///   auto result = evaluator->Evaluate(record);
/// @endcode
class CustomEvaluatorRegistry {
public:
    CustomEvaluatorRegistry();
    ~CustomEvaluatorRegistry();

    // Disable copy
    CustomEvaluatorRegistry(const CustomEvaluatorRegistry&) = delete;
    CustomEvaluatorRegistry& operator=(const CustomEvaluatorRegistry&) = delete;

    /// @brief Register custom evaluator definition
    absl::Status Register(const CustomEvaluatorDefinition& definition);

    /// @brief Update existing evaluator definition
    absl::Status Update(const CustomEvaluatorDefinition& definition);

    /// @brief Unregister evaluator
    absl::Status Unregister(const std::string& name);

    /// @brief Create evaluator instance
    absl::StatusOr<std::unique_ptr<Evaluator>> Create(const std::string& name);

    /// @brief Check if evaluator exists
    bool Exists(const std::string& name) const;

    /// @brief List registered evaluators
    std::vector<std::string> ListEvaluators() const;

    /// @brief List evaluators by tag
    std::vector<std::string> ListByTag(const std::string& tag) const;

    /// @brief Get evaluator definition
    absl::StatusOr<CustomEvaluatorDefinition> GetDefinition(
        const std::string& name) const;

    /// @brief Load definitions from YAML file
    absl::Status LoadFromFile(const std::string& path);

    /// @brief Save definitions to YAML file
    absl::Status SaveToFile(const std::string& path) const;

    /// @brief Load definitions from JSON string
    absl::Status LoadFromJson(const std::string& json);

    /// @brief Export definitions to JSON string
    absl::StatusOr<std::string> ExportToJson() const;

    /// @brief Get registry statistics
    struct Stats {
        size_t total_evaluators = 0;
        size_t enabled_evaluators = 0;
        std::unordered_map<CustomEvaluatorType, size_t> by_type;
    };
    Stats GetStats() const;

private:
    std::unordered_map<std::string, CustomEvaluatorDefinition> definitions_;
    mutable std::mutex mutex_;
};

/// @brief Custom evaluator base implementation
class CustomEvaluator : public Evaluator {
public:
    explicit CustomEvaluator(CustomEvaluatorDefinition definition);
    ~CustomEvaluator() override;

    // Disable copy
    CustomEvaluator(const CustomEvaluator&) = delete;
    CustomEvaluator& operator=(const CustomEvaluator&) = delete;

    /// @brief Initialize evaluator
    absl::Status Initialize();

    absl::StatusOr<EvalResult> Evaluate(const InferenceRecord& record) override;
    absl::StatusOr<std::vector<EvalResult>> EvaluateBatch(
        const std::vector<InferenceRecord>& records) override;
    std::string Type() const override { return "Custom:" + definition_.name; }

    /// @brief Get definition
    const CustomEvaluatorDefinition& GetDefinition() const { return definition_; }

    /// @brief Get statistics
    struct Stats {
        size_t evaluations = 0;
        size_t passes = 0;
        size_t fails = 0;
        size_t errors = 0;
        double avg_score = 0.0;
        double avg_latency_ms = 0.0;
    };
    Stats GetStats() const;

private:
    // Type-specific evaluation
    EvalResult EvaluateRule(const InferenceRecord& record);
    EvalResult EvaluateScript(const InferenceRecord& record);
    EvalResult EvaluateLLM(const InferenceRecord& record);
    EvalResult EvaluateExternal(const InferenceRecord& record);

    // Rule evaluation helpers
    bool EvaluateCondition(const std::string& expression,
                          const InferenceRecord& record);
    bool MatchesRegex(const std::string& text, const std::string& pattern);
    bool ContainsKeyword(const std::string& text, const std::string& keyword);

    // Script execution helpers
    absl::StatusOr<EvalResult> ExecutePython(const InferenceRecord& record);
    absl::StatusOr<EvalResult> ExecuteJavaScript(const InferenceRecord& record);

    // LLM evaluation helpers
    absl::StatusOr<EvalResult> CallLLMEvaluator(const InferenceRecord& record);
    std::string BuildLLMPrompt(const InferenceRecord& record);
    EvalResult ParseLLMResponse(const std::string& response);

    // External API helpers
    absl::StatusOr<EvalResult> CallExternalAPI(const InferenceRecord& record);

    CustomEvaluatorDefinition definition_;
    Stats stats_;
    mutable std::mutex stats_mutex_;
    bool initialized_ = false;

    // Compiled regexes for rule-based
    std::vector<std::regex> compiled_patterns_;
};

/// @brief Factory function to create custom evaluator registry
std::unique_ptr<CustomEvaluatorRegistry> CreateCustomEvaluatorRegistry();

/// @brief Convert evaluator type to string
std::string CustomEvaluatorTypeToString(CustomEvaluatorType type);

/// @brief Convert string to evaluator type
CustomEvaluatorType StringToCustomEvaluatorType(const std::string& str);

/// @brief Validate evaluator definition
absl::Status ValidateDefinition(const CustomEvaluatorDefinition& definition);

/// @brief Create a simple rule-based evaluator definition
CustomEvaluatorDefinition CreateLengthCheckEvaluator(
    const std::string& name,
    size_t min_length,
    size_t max_length);

/// @brief Create a keyword-based evaluator definition
CustomEvaluatorDefinition CreateKeywordEvaluator(
    const std::string& name,
    const std::vector<std::string>& required_keywords,
    const std::vector<std::string>& forbidden_keywords = {});

/// @brief Create an LLM-based evaluator definition
CustomEvaluatorDefinition CreateLLMEvaluator(
    const std::string& name,
    const std::string& system_prompt,
    const std::string& user_prompt_template);

}  // namespace pyflare::eval
