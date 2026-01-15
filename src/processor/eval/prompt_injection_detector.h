#pragma once

/// @file prompt_injection_detector.h
/// @brief Prompt injection attack detection
///
/// Detects various prompt injection attack patterns:
/// - Direct injection attempts
/// - Jailbreak patterns
/// - Role playing exploits
/// - Encoded payloads
/// - Context manipulation

#include <chrono>
#include <memory>
#include <mutex>
#include <regex>
#include <string>
#include <unordered_set>
#include <vector>

#include <absl/status/status.h>
#include <absl/status/statusor.h>

#include "processor/eval/evaluator.h"

namespace pyflare::eval {

/// @brief Types of injection attacks
enum class InjectionType {
    kNone,              ///< No injection detected
    kDirectInjection,   ///< Direct prompt override
    kJailbreak,         ///< Jailbreak attempt
    kRolePlay,          ///< Role-playing exploit
    kEncodedPayload,    ///< Base64/hex encoded attack
    kContextLeak,       ///< Context/system prompt extraction
    kDelimiterManip,    ///< Delimiter manipulation
    kRecursivePrompt,   ///< Recursive prompt injection
    kIndirectInjection, ///< Indirect injection via data
    kUnknown            ///< Unknown pattern detected
};

/// @brief Risk level of detected injection
enum class InjectionRiskLevel {
    kNone,     ///< No risk detected
    kLow,      ///< Low risk - suspicious but likely benign
    kMedium,   ///< Medium risk - potential attack
    kHigh,     ///< High risk - likely malicious
    kCritical  ///< Critical - definite attack pattern
};

/// @brief Result of injection detection
struct InjectionDetectionResult {
    bool injection_detected = false;
    InjectionType injection_type = InjectionType::kNone;
    InjectionRiskLevel risk_level = InjectionRiskLevel::kNone;

    double confidence = 0.0;  ///< 0.0 - 1.0

    /// Matched patterns
    std::vector<std::string> matched_patterns;

    /// Location in text where injection was found
    size_t injection_start = 0;
    size_t injection_length = 0;

    /// Explanation of detection
    std::string explanation;

    /// Recommended action
    std::string recommendation;

    /// Additional details
    std::unordered_map<std::string, std::string> metadata;
};

/// @brief Configuration for prompt injection detection
struct PromptInjectionConfig {
    /// Enable pattern-based detection
    bool enable_pattern_detection = true;

    /// Enable ML-based detection (requires model)
    bool enable_ml_detection = false;

    /// Enable semantic analysis
    bool enable_semantic_analysis = false;

    /// Confidence threshold for detection
    double confidence_threshold = 0.7;

    /// Risk level threshold for alerting
    InjectionRiskLevel alert_threshold = InjectionRiskLevel::kMedium;

    /// Enable encoded content detection (Base64, hex)
    bool detect_encoded_content = true;

    /// Enable delimiter manipulation detection
    bool detect_delimiter_manipulation = true;

    /// Custom patterns to detect (regex)
    std::vector<std::string> custom_patterns;

    /// Phrases to always block
    std::vector<std::string> blocklist;

    /// Phrases to always allow (overrides detection)
    std::vector<std::string> allowlist;

    /// Maximum input length to analyze (0 = unlimited)
    size_t max_input_length = 0;

    /// Enable caching of detection results
    bool enable_caching = true;

    /// Cache TTL
    std::chrono::seconds cache_ttl = std::chrono::hours(1);
};

/// @brief Prompt injection attack detector
///
/// Multi-layer detection approach:
/// - Pattern matching against known attack signatures
/// - ML-based classification (optional)
/// - Semantic analysis for subtle attacks
///
/// Example:
/// @code
///   PromptInjectionConfig config;
///   config.confidence_threshold = 0.8;
///   auto detector = std::make_unique<PromptInjectionDetector>(config);
///   detector->Initialize();
///
///   // Check user input
///   auto result = detector->Detect(user_input);
///   if (result->injection_detected) {
///       if (result->risk_level >= InjectionRiskLevel::kHigh) {
///           // Block the request
///           LOG(WARNING) << "Blocked injection attempt: " << result->explanation;
///       }
///   }
/// @endcode
class PromptInjectionDetector : public Evaluator {
public:
    explicit PromptInjectionDetector(PromptInjectionConfig config = {});
    ~PromptInjectionDetector() override;

    // Disable copy
    PromptInjectionDetector(const PromptInjectionDetector&) = delete;
    PromptInjectionDetector& operator=(const PromptInjectionDetector&) = delete;

    /// @brief Initialize detector (load models, compile patterns)
    absl::Status Initialize();

    // =========================================================================
    // Evaluator Interface
    // =========================================================================

    absl::StatusOr<EvalResult> Evaluate(const InferenceRecord& record) override;
    absl::StatusOr<std::vector<EvalResult>> EvaluateBatch(
        const std::vector<InferenceRecord>& records) override;
    std::string Type() const override { return "PromptInjection"; }

    // =========================================================================
    // Detection API
    // =========================================================================

    /// @brief Detect injection in text
    /// @param text Text to analyze
    /// @return Detection result
    absl::StatusOr<InjectionDetectionResult> Detect(const std::string& text);

    /// @brief Detect injection with context
    /// @param text Text to analyze
    /// @param system_prompt System prompt for context
    /// @param previous_messages Previous conversation messages
    absl::StatusOr<InjectionDetectionResult> DetectWithContext(
        const std::string& text,
        const std::string& system_prompt = "",
        const std::vector<std::string>& previous_messages = {});

    /// @brief Batch detection
    /// @param texts Texts to analyze
    absl::StatusOr<std::vector<InjectionDetectionResult>> DetectBatch(
        const std::vector<std::string>& texts);

    /// @brief Check if text is safe (quick check)
    /// @param text Text to check
    /// @return true if likely safe, false if potentially dangerous
    bool IsSafe(const std::string& text);

    // =========================================================================
    // Pattern Management
    // =========================================================================

    /// @brief Add custom detection pattern
    /// @param pattern Regex pattern
    /// @param type Injection type this pattern detects
    /// @param risk_level Risk level for matches
    void AddPattern(const std::string& pattern,
                    InjectionType type,
                    InjectionRiskLevel risk_level = InjectionRiskLevel::kMedium);

    /// @brief Add phrase to blocklist
    void AddToBlocklist(const std::string& phrase);

    /// @brief Add phrase to allowlist
    void AddToAllowlist(const std::string& phrase);

    /// @brief Clear custom patterns
    void ClearCustomPatterns();

    // =========================================================================
    // Configuration
    // =========================================================================

    /// @brief Update configuration
    void SetConfig(PromptInjectionConfig config);

    /// @brief Get configuration
    const PromptInjectionConfig& GetConfig() const { return config_; }

    // =========================================================================
    // Statistics
    // =========================================================================

    /// @brief Get detection statistics
    struct Stats {
        size_t total_checks = 0;
        size_t detections = 0;
        size_t blocked = 0;
        std::unordered_map<InjectionType, size_t> detections_by_type;
        std::unordered_map<InjectionRiskLevel, size_t> detections_by_risk;
        double avg_check_time_ms = 0.0;
    };
    Stats GetStats() const;

    /// @brief Reset statistics
    void ResetStats();

private:
    // Pattern-based detection
    InjectionDetectionResult DetectPatterns(const std::string& text);

    // ML-based detection (if enabled)
    InjectionDetectionResult DetectWithML(const std::string& text);

    // Semantic analysis (if enabled)
    InjectionDetectionResult DetectSemantic(const std::string& text);

    // Check for encoded content
    bool CheckEncodedContent(const std::string& text,
                             std::string& decoded,
                             std::string& encoding_type);

    // Check for delimiter manipulation
    bool CheckDelimiterManipulation(const std::string& text,
                                    std::vector<std::string>& matched);

    // Check blocklist/allowlist
    bool IsInBlocklist(const std::string& text);
    bool IsInAllowlist(const std::string& text);

    // Normalize text for comparison
    std::string NormalizeText(const std::string& text);

    // Compute hash for caching
    std::string ComputeHash(const std::string& text);

    // Merge detection results from multiple detectors
    InjectionDetectionResult MergeResults(
        const std::vector<InjectionDetectionResult>& results);

    // Convert to evaluation result
    EvalResult ToEvalResult(const InjectionDetectionResult& detection);

    PromptInjectionConfig config_;

    // Compiled patterns
    struct CompiledPattern {
        std::regex regex;
        InjectionType type;
        InjectionRiskLevel risk_level;
        std::string description;
    };
    std::vector<CompiledPattern> compiled_patterns_;

    // Blocklist and allowlist (lowercase for comparison)
    std::unordered_set<std::string> blocklist_;
    std::unordered_set<std::string> allowlist_;

    // Statistics
    Stats stats_;
    mutable std::mutex stats_mutex_;

    // Cache
    mutable std::mutex cache_mutex_;
    std::unordered_map<std::string, InjectionDetectionResult> cache_;

    bool initialized_ = false;
};

/// @brief Create default prompt injection detector
std::unique_ptr<Evaluator> CreatePromptInjectionDetector(
    PromptInjectionConfig config = {});

/// @brief Convert injection type to string
std::string InjectionTypeToString(InjectionType type);

/// @brief Convert risk level to string
std::string RiskLevelToString(InjectionRiskLevel level);

/// @brief Get default injection patterns
std::vector<std::tuple<std::string, InjectionType, InjectionRiskLevel>>
GetDefaultInjectionPatterns();

}  // namespace pyflare::eval
