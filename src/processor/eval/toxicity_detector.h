#pragma once

/// @file toxicity_detector.h
/// @brief Toxicity and harmful content detection for LLM outputs

#include <memory>
#include <string>
#include <unordered_map>
#include <vector>

#include <absl/status/statusor.h>

#include "processor/eval/evaluator.h"

namespace pyflare::eval {

/// @brief Toxicity categories
enum class ToxicityCategory {
    kNone,
    kHateSpeech,
    kHarassment,
    kViolence,
    kSexualContent,
    kSelfHarm,
    kDangerous,
    kPII,
    kProfanity,
    kMisinformation,
    kOther
};

/// @brief Toxicity detection result
struct ToxicityResult {
    bool is_toxic = false;
    ToxicityCategory primary_category = ToxicityCategory::kNone;
    double toxicity_score = 0.0;  ///< 0.0 (safe) to 1.0 (highly toxic)

    // Per-category scores
    std::unordered_map<ToxicityCategory, double> category_scores;

    // Specific flags
    bool contains_hate_speech = false;
    bool contains_harassment = false;
    bool contains_violence = false;
    bool contains_sexual_content = false;
    bool contains_self_harm = false;
    bool contains_dangerous_content = false;
    bool contains_pii = false;
    bool contains_profanity = false;

    // Details
    std::vector<std::string> flagged_phrases;
    std::string explanation;
    std::unordered_map<std::string, std::string> metadata;
};

/// @brief PII types for detection
enum class PIIType {
    kEmail,
    kPhone,
    kSSN,
    kCreditCard,
    kAddress,
    kName,
    kDateOfBirth,
    kIPAddress,
    kAPIKey,
    kPassword,
    kOther
};

/// @brief PII detection result
struct PIIDetectionResult {
    bool has_pii = false;
    std::vector<std::pair<PIIType, std::string>> detected_pii;
    size_t total_pii_count = 0;
    std::string scrubbed_text;  ///< Text with PII redacted
};

/// @brief Configuration for toxicity detector
struct ToxicityDetectorConfig {
    /// Toxicity threshold for flagging (0.0-1.0)
    double toxicity_threshold = 0.5;

    /// Whether to detect PII
    bool detect_pii = true;

    /// Whether to scrub detected PII
    bool scrub_pii = false;

    /// Use ML model for detection (vs word lists only)
    bool use_ml_model = false;

    /// ML model for toxicity classification
    std::string model_name = "HuggingFace/toxic-bert";

    /// Use OpenAI moderation API
    bool use_openai_moderation = true;

    /// API key for moderation services
    std::string api_key;

    /// Categories to check
    std::vector<ToxicityCategory> enabled_categories = {
        ToxicityCategory::kHateSpeech,
        ToxicityCategory::kHarassment,
        ToxicityCategory::kViolence,
        ToxicityCategory::kSexualContent,
        ToxicityCategory::kSelfHarm,
        ToxicityCategory::kDangerous
    };

    /// Custom word lists for detection
    std::unordered_map<ToxicityCategory, std::vector<std::string>> word_lists;
};

/// @brief Toxicity and harmful content detector
///
/// Detects various forms of harmful content in LLM outputs:
/// - Hate speech and discrimination
/// - Harassment and bullying
/// - Violence and threats
/// - Sexual content
/// - Self-harm content
/// - Dangerous instructions
/// - PII (personally identifiable information)
///
/// Supports multiple detection methods:
/// - OpenAI Moderation API
/// - Word/phrase lists (built-in and custom)
/// - ML-based classification (via external models)
///
/// Example usage:
/// @code
///   ToxicityDetectorConfig config;
///   config.toxicity_threshold = 0.3;  // Stricter threshold
///   auto detector = std::make_unique<ToxicityDetector>(config);
///
///   InferenceRecord record;
///   record.output = "Some text to check for toxicity...";
///
///   auto result = detector->Evaluate(record);
///   if (result->verdict == "fail") {
///       // Handle toxic content
///   }
/// @endcode
class ToxicityDetector : public Evaluator {
public:
    explicit ToxicityDetector(ToxicityDetectorConfig config = {});
    ~ToxicityDetector() override;

    // Disable copy
    ToxicityDetector(const ToxicityDetector&) = delete;
    ToxicityDetector& operator=(const ToxicityDetector&) = delete;

    /// @brief Initialize the detector
    absl::Status Initialize();

    /// @brief Evaluate a single inference
    absl::StatusOr<EvalResult> Evaluate(const InferenceRecord& record) override;

    /// @brief Batch evaluation
    absl::StatusOr<std::vector<EvalResult>> EvaluateBatch(
        const std::vector<InferenceRecord>& records) override;

    std::string Type() const override { return "ToxicityDetector"; }

    // =========================================================================
    // Toxicity Detection
    // =========================================================================

    /// @brief Detect toxicity in text
    /// @param text Text to analyze
    absl::StatusOr<ToxicityResult> DetectToxicity(const std::string& text);

    /// @brief Detect toxicity in both input and output
    /// @param record Inference record
    absl::StatusOr<ToxicityResult> DetectToxicity(const InferenceRecord& record);

    /// @brief Check if text exceeds toxicity threshold
    /// @param text Text to check
    absl::StatusOr<bool> IsToxic(const std::string& text);

    // =========================================================================
    // PII Detection
    // =========================================================================

    /// @brief Detect PII in text
    /// @param text Text to analyze
    PIIDetectionResult DetectPII(const std::string& text);

    /// @brief Scrub PII from text
    /// @param text Text to scrub
    /// @return Text with PII redacted
    std::string ScrubPII(const std::string& text);

    // =========================================================================
    // Configuration
    // =========================================================================

    /// @brief Add custom word list for a category
    void AddWordList(ToxicityCategory category,
                     const std::vector<std::string>& words);

    /// @brief Set toxicity threshold
    void SetThreshold(double threshold) { config_.toxicity_threshold = threshold; }

    /// @brief Get configuration
    const ToxicityDetectorConfig& GetConfig() const { return config_; }

    // =========================================================================
    // Helpers
    // =========================================================================

    /// @brief Convert category to string
    static std::string CategoryToString(ToxicityCategory category);

    /// @brief Convert string to category
    static ToxicityCategory StringToCategory(const std::string& str);

    /// @brief Convert PII type to string
    static std::string PIITypeToString(PIIType type);

private:
    /// @brief Check text against word lists
    ToxicityResult CheckWordLists(const std::string& text);

    /// @brief Use OpenAI moderation API
    absl::StatusOr<ToxicityResult> CheckOpenAIModeration(const std::string& text);

    /// @brief Detect PII using regex patterns
    PIIDetectionResult DetectPIIWithRegex(const std::string& text);

    /// @brief Initialize default word lists
    void InitializeDefaultWordLists();

    ToxicityDetectorConfig config_;

    // Built-in word lists (loaded on init)
    std::unordered_map<ToxicityCategory, std::vector<std::string>> default_word_lists_;

    class Impl;
    std::unique_ptr<Impl> impl_;
};

/// @brief Create a toxicity detector with default configuration
std::unique_ptr<Evaluator> CreateToxicityDetector(ToxicityDetectorConfig config = {});

}  // namespace pyflare::eval
