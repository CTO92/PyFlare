#pragma once

/// @file counterfactual.h
/// @brief Counterfactual explanation generation for model behavior analysis
///
/// Generates "what if" explanations that answer:
/// "What input change would produce a different output?"

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

namespace pyflare::rca {

/// @brief A single change in a counterfactual
struct CounterfactualChange {
    enum class ChangeType {
        kAddition,     ///< Added content
        kDeletion,     ///< Removed content
        kReplacement,  ///< Replaced content
        kReordering    ///< Reordered content
    };

    ChangeType type = ChangeType::kReplacement;

    /// Position in the original text
    size_t start_position = 0;
    size_t end_position = 0;

    /// Original content (for replacement/deletion)
    std::string original_content;

    /// New content (for addition/replacement)
    std::string new_content;

    /// Importance score of this change (0.0 - 1.0)
    double importance = 0.0;

    /// Explanation of why this change matters
    std::string explanation;
};

/// @brief A counterfactual explanation
struct Counterfactual {
    /// The modified input that produces different output
    std::string modified_input;

    /// The original input
    std::string original_input;

    /// The original output
    std::string original_output;

    /// The expected output after modification
    std::string expected_output;

    /// List of changes made
    std::vector<CounterfactualChange> changes;

    /// Confidence in the counterfactual (0.0 - 1.0)
    double confidence = 0.0;

    /// Human-readable explanation
    std::string explanation;

    /// Semantic similarity between original and modified input
    double input_similarity = 0.0;

    /// Edit distance (number of edits)
    size_t edit_distance = 0;

    /// Category of the counterfactual
    std::string category;  // "minimal", "diverse", "contrastive"

    /// Whether this was validated by running through model
    bool validated = false;

    /// Validation result (if validated)
    std::optional<std::string> validation_output;
};

/// @brief Configuration for counterfactual generation
struct CounterfactualConfig {
    /// Generation method
    enum class Method {
        kTextPerturbation,   ///< Simple text modifications
        kFeatureAttribution, ///< SHAP/LIME-like attribution
        kGradientBased,      ///< Gradient-guided search
        kLLMBased            ///< Use LLM to generate counterfactuals
    };
    Method method = Method::kLLMBased;

    /// Maximum number of edits to try
    size_t max_edits = 5;

    /// Minimum similarity to original (0.0 - 1.0)
    double min_similarity = 0.7;

    /// Maximum similarity (for diversity)
    double max_similarity = 0.99;

    /// Number of candidates to generate
    size_t num_candidates = 10;

    /// Whether to validate counterfactuals
    bool validate_counterfactuals = false;

    /// LLM configuration (for LLM-based method)
    struct LLMConfig {
        std::string api_endpoint = "https://api.openai.com/v1/chat/completions";
        std::string api_key;
        std::string model = "gpt-4o-mini";
        double temperature = 0.7;
        size_t max_tokens = 1024;
    };
    LLMConfig llm;

    /// Text perturbation configuration
    struct PerturbationConfig {
        bool enable_word_replacement = true;
        bool enable_word_deletion = true;
        bool enable_word_insertion = true;
        bool enable_sentence_reordering = true;
        size_t max_word_changes = 3;
    };
    PerturbationConfig perturbation;

    /// Timeout for generation
    std::chrono::seconds timeout = std::chrono::seconds(60);

    /// Enable caching
    bool enable_cache = true;

    /// Cache TTL
    std::chrono::hours cache_ttl = std::chrono::hours(24);
};

/// @brief Counterfactual explanation generator
///
/// Generates counterfactual explanations that help understand
/// model behavior by showing what changes would lead to different outputs.
///
/// Useful for:
/// - Debugging unexpected model behavior
/// - Understanding decision boundaries
/// - Improving prompts
/// - Identifying sensitive input features
///
/// Example:
/// @code
///   CounterfactualConfig config;
///   config.method = CounterfactualConfig::Method::kLLMBased;
///   auto generator = std::make_unique<CounterfactualGenerator>(config);
///   generator->Initialize();
///
///   eval::InferenceRecord record;
///   record.input = "What is the capital of France?";
///   record.output = "The capital of France is Berlin.";  // Wrong!
///
///   auto cf = generator->Generate(record, "Correct answer about Paris");
///   // cf.modified_input might be: "What is the capital of Germany?"
///   // Showing that the model confuses France and Germany
/// @endcode
class CounterfactualGenerator {
public:
    explicit CounterfactualGenerator(CounterfactualConfig config = {});
    ~CounterfactualGenerator();

    // Disable copy
    CounterfactualGenerator(const CounterfactualGenerator&) = delete;
    CounterfactualGenerator& operator=(const CounterfactualGenerator&) = delete;

    /// @brief Initialize generator
    absl::Status Initialize();

    // =========================================================================
    // Generation API
    // =========================================================================

    /// @brief Generate counterfactual for single inference
    /// @param record Original inference record
    /// @param target_outcome Desired outcome description
    absl::StatusOr<Counterfactual> Generate(
        const eval::InferenceRecord& record,
        const std::string& target_outcome);

    /// @brief Generate multiple counterfactuals
    /// @param record Original inference record
    /// @param target_outcome Desired outcome
    /// @param count Number of counterfactuals to generate
    absl::StatusOr<std::vector<Counterfactual>> GenerateMultiple(
        const eval::InferenceRecord& record,
        const std::string& target_outcome,
        size_t count = 3);

    /// @brief Generate diverse counterfactuals (different change types)
    /// @param record Original inference record
    /// @param target_outcome Desired outcome
    absl::StatusOr<std::vector<Counterfactual>> GenerateDiverse(
        const eval::InferenceRecord& record,
        const std::string& target_outcome);

    /// @brief Generate minimal counterfactual (smallest change)
    /// @param record Original inference record
    /// @param target_outcome Desired outcome
    absl::StatusOr<Counterfactual> GenerateMinimal(
        const eval::InferenceRecord& record,
        const std::string& target_outcome);

    // =========================================================================
    // Analysis API
    // =========================================================================

    /// @brief Analyze which parts of input are most influential
    /// @param record Inference record to analyze
    /// @return Map of text spans to importance scores
    absl::StatusOr<std::vector<std::pair<std::string, double>>> AnalyzeInfluence(
        const eval::InferenceRecord& record);

    /// @brief Find critical words/phrases in input
    /// @param record Inference record to analyze
    absl::StatusOr<std::vector<std::string>> FindCriticalTokens(
        const eval::InferenceRecord& record);

    // =========================================================================
    // Validation API
    // =========================================================================

    /// @brief Validate a counterfactual by running through model
    /// @param counterfactual Counterfactual to validate
    /// @param model_callback Function to get model output for input
    absl::StatusOr<bool> Validate(
        Counterfactual& counterfactual,
        std::function<std::string(const std::string&)> model_callback);

    // =========================================================================
    // Configuration
    // =========================================================================

    /// @brief Update configuration
    void SetConfig(CounterfactualConfig config);

    /// @brief Get configuration
    const CounterfactualConfig& GetConfig() const { return config_; }

    /// @brief Clear cache
    void ClearCache();

    // =========================================================================
    // Statistics
    // =========================================================================

    /// @brief Get statistics
    struct Stats {
        size_t total_generations = 0;
        size_t successful_generations = 0;
        size_t cache_hits = 0;
        size_t validations = 0;
        size_t successful_validations = 0;
        double avg_generation_time_ms = 0.0;
        double avg_edit_distance = 0.0;
        double avg_confidence = 0.0;
    };
    Stats GetStats() const;

    /// @brief Reset statistics
    void ResetStats();

private:
    // Generation methods
    absl::StatusOr<Counterfactual> GenerateWithLLM(
        const eval::InferenceRecord& record,
        const std::string& target_outcome);

    absl::StatusOr<Counterfactual> GenerateWithPerturbation(
        const eval::InferenceRecord& record,
        const std::string& target_outcome);

    absl::StatusOr<Counterfactual> GenerateWithAttribution(
        const eval::InferenceRecord& record,
        const std::string& target_outcome);

    // LLM helpers
    std::string BuildLLMPrompt(
        const eval::InferenceRecord& record,
        const std::string& target_outcome,
        const std::string& category);

    absl::StatusOr<Counterfactual> ParseLLMResponse(
        const std::string& response,
        const eval::InferenceRecord& record);

    absl::StatusOr<std::string> CallLLM(const std::string& prompt);

    // Text perturbation helpers
    std::vector<std::string> GeneratePerturbations(const std::string& text);
    std::vector<std::string> TokenizeText(const std::string& text);
    std::string JoinTokens(const std::vector<std::string>& tokens);

    // Similarity and distance
    double ComputeSimilarity(const std::string& a, const std::string& b);
    size_t ComputeEditDistance(const std::string& a, const std::string& b);
    std::vector<CounterfactualChange> ExtractChanges(
        const std::string& original,
        const std::string& modified);

    // Cache helpers
    std::string ComputeCacheKey(
        const eval::InferenceRecord& record,
        const std::string& target_outcome);

    CounterfactualConfig config_;

    // Cache
    mutable std::mutex cache_mutex_;
    std::unordered_map<std::string, Counterfactual> cache_;

    // Statistics
    Stats stats_;
    mutable std::mutex stats_mutex_;

    bool initialized_ = false;
};

/// @brief Factory function to create counterfactual generator
std::unique_ptr<CounterfactualGenerator> CreateCounterfactualGenerator(
    CounterfactualConfig config = {});

/// @brief Convert change type to string
std::string ChangeTypeToString(CounterfactualChange::ChangeType type);

/// @brief Convert method to string
std::string CounterfactualMethodToString(CounterfactualConfig::Method method);

}  // namespace pyflare::rca
