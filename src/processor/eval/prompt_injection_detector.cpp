/// @file prompt_injection_detector.cpp
/// @brief Prompt injection attack detection implementation

#include "processor/eval/prompt_injection_detector.h"

#include <algorithm>
#include <cctype>
#include <chrono>
#include <cstring>
#include <functional>
#include <sstream>

namespace pyflare::eval {

namespace {

// Base64 decoding table
static const int kBase64DecodeTable[128] = {
    -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1,
    -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1,
    -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, 62, -1, -1, -1, 63,
    52, 53, 54, 55, 56, 57, 58, 59, 60, 61, -1, -1, -1, -1, -1, -1,
    -1,  0,  1,  2,  3,  4,  5,  6,  7,  8,  9, 10, 11, 12, 13, 14,
    15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, -1, -1, -1, -1, -1,
    -1, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40,
    41, 42, 43, 44, 45, 46, 47, 48, 49, 50, 51, -1, -1, -1, -1, -1
};

bool IsBase64(const std::string& str) {
    if (str.length() < 4 || str.length() % 4 != 0) {
        return false;
    }
    for (size_t i = 0; i < str.length(); ++i) {
        char c = str[i];
        if (c == '=') {
            // Padding should only be at the end
            for (size_t j = i; j < str.length(); ++j) {
                if (str[j] != '=') return false;
            }
            break;
        }
        if (c < 0 || c >= 128 || kBase64DecodeTable[static_cast<unsigned char>(c)] == -1) {
            return false;
        }
    }
    return true;
}

std::string DecodeBase64(const std::string& input) {
    std::string output;
    int val = 0, valb = -8;
    for (unsigned char c : input) {
        if (c == '=') break;
        if (c < 0 || c >= 128) return "";
        int d = kBase64DecodeTable[c];
        if (d == -1) return "";
        val = (val << 6) + d;
        valb += 6;
        if (valb >= 0) {
            output.push_back(static_cast<char>((val >> valb) & 0xFF));
            valb -= 8;
        }
    }
    return output;
}

bool IsHexString(const std::string& str) {
    if (str.length() < 4 || str.length() % 2 != 0) {
        return false;
    }
    for (char c : str) {
        if (!std::isxdigit(static_cast<unsigned char>(c))) {
            return false;
        }
    }
    return true;
}

std::string DecodeHex(const std::string& input) {
    std::string output;
    for (size_t i = 0; i + 1 < input.length(); i += 2) {
        int high = std::isdigit(input[i]) ? input[i] - '0' : std::tolower(input[i]) - 'a' + 10;
        int low = std::isdigit(input[i+1]) ? input[i+1] - '0' : std::tolower(input[i+1]) - 'a' + 10;
        output.push_back(static_cast<char>((high << 4) | low));
    }
    return output;
}

uint64_t HashString(const std::string& str) {
    // Simple FNV-1a hash
    uint64_t hash = 14695981039346656037ULL;
    for (char c : str) {
        hash ^= static_cast<unsigned char>(c);
        hash *= 1099511628211ULL;
    }
    return hash;
}

}  // namespace

PromptInjectionDetector::PromptInjectionDetector(PromptInjectionConfig config)
    : config_(std::move(config)) {}

PromptInjectionDetector::~PromptInjectionDetector() = default;

absl::Status PromptInjectionDetector::Initialize() {
    if (initialized_) {
        return absl::OkStatus();
    }

    // Compile default patterns
    auto default_patterns = GetDefaultInjectionPatterns();
    for (const auto& [pattern, type, risk] : default_patterns) {
        try {
            CompiledPattern cp;
            cp.regex = std::regex(pattern, std::regex::icase | std::regex::optimize);
            cp.type = type;
            cp.risk_level = risk;
            cp.description = pattern;
            compiled_patterns_.push_back(std::move(cp));
        } catch (const std::regex_error& e) {
            // Skip invalid patterns
            continue;
        }
    }

    // Compile custom patterns
    for (const auto& pattern : config_.custom_patterns) {
        try {
            CompiledPattern cp;
            cp.regex = std::regex(pattern, std::regex::icase | std::regex::optimize);
            cp.type = InjectionType::kUnknown;
            cp.risk_level = InjectionRiskLevel::kMedium;
            cp.description = pattern;
            compiled_patterns_.push_back(std::move(cp));
        } catch (const std::regex_error& e) {
            // Skip invalid patterns
            continue;
        }
    }

    // Initialize blocklist
    for (const auto& phrase : config_.blocklist) {
        blocklist_.insert(NormalizeText(phrase));
    }

    // Initialize allowlist
    for (const auto& phrase : config_.allowlist) {
        allowlist_.insert(NormalizeText(phrase));
    }

    initialized_ = true;
    return absl::OkStatus();
}

absl::StatusOr<EvalResult> PromptInjectionDetector::Evaluate(
    const InferenceRecord& record) {
    auto result = Detect(record.input);
    if (!result.ok()) {
        return result.status();
    }
    return ToEvalResult(*result);
}

absl::StatusOr<std::vector<EvalResult>> PromptInjectionDetector::EvaluateBatch(
    const std::vector<InferenceRecord>& records) {
    std::vector<std::string> inputs;
    inputs.reserve(records.size());
    for (const auto& record : records) {
        inputs.push_back(record.input);
    }

    auto results = DetectBatch(inputs);
    if (!results.ok()) {
        return results.status();
    }

    std::vector<EvalResult> eval_results;
    eval_results.reserve(results->size());
    for (const auto& detection : *results) {
        eval_results.push_back(ToEvalResult(detection));
    }

    return eval_results;
}

absl::StatusOr<InjectionDetectionResult> PromptInjectionDetector::Detect(
    const std::string& text) {
    if (!initialized_) {
        return absl::FailedPreconditionError("Detector not initialized");
    }

    auto start_time = std::chrono::steady_clock::now();

    // Check allowlist first
    if (IsInAllowlist(text)) {
        InjectionDetectionResult result;
        result.explanation = "Text is in allowlist";
        return result;
    }

    // Check blocklist
    if (IsInBlocklist(text)) {
        InjectionDetectionResult result;
        result.injection_detected = true;
        result.injection_type = InjectionType::kDirectInjection;
        result.risk_level = InjectionRiskLevel::kCritical;
        result.confidence = 1.0;
        result.explanation = "Text matches blocklist entry";
        result.recommendation = "Block this input";

        {
            std::lock_guard<std::mutex> lock(stats_mutex_);
            stats_.total_checks++;
            stats_.detections++;
            stats_.blocked++;
            stats_.detections_by_type[InjectionType::kDirectInjection]++;
            stats_.detections_by_risk[InjectionRiskLevel::kCritical]++;
        }

        return result;
    }

    // Check cache
    std::string cache_key;
    if (config_.enable_caching) {
        cache_key = ComputeHash(text);
        std::lock_guard<std::mutex> lock(cache_mutex_);
        auto it = cache_.find(cache_key);
        if (it != cache_.end()) {
            return it->second;
        }
    }

    // Truncate if needed
    std::string input_text = text;
    if (config_.max_input_length > 0 && text.length() > config_.max_input_length) {
        input_text = text.substr(0, config_.max_input_length);
    }

    std::vector<InjectionDetectionResult> results;

    // Pattern-based detection
    if (config_.enable_pattern_detection) {
        results.push_back(DetectPatterns(input_text));
    }

    // ML-based detection
    if (config_.enable_ml_detection) {
        results.push_back(DetectWithML(input_text));
    }

    // Semantic analysis
    if (config_.enable_semantic_analysis) {
        results.push_back(DetectSemantic(input_text));
    }

    // Merge results
    auto final_result = MergeResults(results);

    // Update statistics
    auto end_time = std::chrono::steady_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::microseconds>(
        end_time - start_time);

    {
        std::lock_guard<std::mutex> lock(stats_mutex_);
        stats_.total_checks++;
        if (final_result.injection_detected) {
            stats_.detections++;
            stats_.detections_by_type[final_result.injection_type]++;
            stats_.detections_by_risk[final_result.risk_level]++;
            if (final_result.risk_level >= config_.alert_threshold) {
                stats_.blocked++;
            }
        }
        stats_.avg_check_time_ms = (stats_.avg_check_time_ms * (stats_.total_checks - 1) +
                                    duration.count() / 1000.0) / stats_.total_checks;
    }

    // Cache result
    if (config_.enable_caching && !cache_key.empty()) {
        std::lock_guard<std::mutex> lock(cache_mutex_);
        cache_[cache_key] = final_result;
    }

    return final_result;
}

absl::StatusOr<InjectionDetectionResult> PromptInjectionDetector::DetectWithContext(
    const std::string& text,
    const std::string& system_prompt,
    const std::vector<std::string>& previous_messages) {
    // For now, just check the text itself
    // More sophisticated context analysis can be added
    return Detect(text);
}

absl::StatusOr<std::vector<InjectionDetectionResult>>
PromptInjectionDetector::DetectBatch(const std::vector<std::string>& texts) {
    std::vector<InjectionDetectionResult> results;
    results.reserve(texts.size());

    for (const auto& text : texts) {
        auto result = Detect(text);
        if (result.ok()) {
            results.push_back(*result);
        } else {
            InjectionDetectionResult error_result;
            error_result.metadata["error"] = std::string(result.status().message());
            results.push_back(error_result);
        }
    }

    return results;
}

bool PromptInjectionDetector::IsSafe(const std::string& text) {
    auto result = Detect(text);
    if (!result.ok()) {
        return false;  // Error = unsafe
    }
    return !result->injection_detected ||
           result->risk_level < config_.alert_threshold;
}

void PromptInjectionDetector::AddPattern(
    const std::string& pattern,
    InjectionType type,
    InjectionRiskLevel risk_level) {
    try {
        CompiledPattern cp;
        cp.regex = std::regex(pattern, std::regex::icase | std::regex::optimize);
        cp.type = type;
        cp.risk_level = risk_level;
        cp.description = pattern;
        compiled_patterns_.push_back(std::move(cp));
    } catch (const std::regex_error& e) {
        // Invalid pattern, ignore
    }
}

void PromptInjectionDetector::AddToBlocklist(const std::string& phrase) {
    blocklist_.insert(NormalizeText(phrase));
}

void PromptInjectionDetector::AddToAllowlist(const std::string& phrase) {
    allowlist_.insert(NormalizeText(phrase));
}

void PromptInjectionDetector::ClearCustomPatterns() {
    compiled_patterns_.clear();
    // Re-add default patterns
    auto default_patterns = GetDefaultInjectionPatterns();
    for (const auto& [pattern, type, risk] : default_patterns) {
        try {
            CompiledPattern cp;
            cp.regex = std::regex(pattern, std::regex::icase | std::regex::optimize);
            cp.type = type;
            cp.risk_level = risk;
            cp.description = pattern;
            compiled_patterns_.push_back(std::move(cp));
        } catch (const std::regex_error& e) {
            continue;
        }
    }
}

void PromptInjectionDetector::SetConfig(PromptInjectionConfig config) {
    config_ = std::move(config);
}

PromptInjectionDetector::Stats PromptInjectionDetector::GetStats() const {
    std::lock_guard<std::mutex> lock(stats_mutex_);
    return stats_;
}

void PromptInjectionDetector::ResetStats() {
    std::lock_guard<std::mutex> lock(stats_mutex_);
    stats_ = Stats{};
}

// ============================================================================
// Private Implementation
// ============================================================================

InjectionDetectionResult PromptInjectionDetector::DetectPatterns(
    const std::string& text) {
    InjectionDetectionResult result;

    InjectionRiskLevel max_risk = InjectionRiskLevel::kNone;

    for (const auto& pattern : compiled_patterns_) {
        std::smatch match;
        if (std::regex_search(text, match, pattern.regex)) {
            result.injection_detected = true;

            if (pattern.risk_level > max_risk) {
                max_risk = pattern.risk_level;
                result.injection_type = pattern.type;
                result.risk_level = pattern.risk_level;
                result.injection_start = match.position();
                result.injection_length = match.length();
            }

            result.matched_patterns.push_back(pattern.description);
        }
    }

    // Check for encoded content
    if (config_.detect_encoded_content) {
        std::string decoded;
        std::string encoding_type;
        if (CheckEncodedContent(text, decoded, encoding_type)) {
            // Recursively check decoded content
            auto decoded_result = DetectPatterns(decoded);
            if (decoded_result.injection_detected) {
                result.injection_detected = true;
                result.injection_type = InjectionType::kEncodedPayload;
                if (decoded_result.risk_level > result.risk_level) {
                    result.risk_level = decoded_result.risk_level;
                }
                result.metadata["encoding"] = encoding_type;
                result.metadata["decoded_content"] = decoded.substr(0, 100);
            }
        }
    }

    // Check for delimiter manipulation
    if (config_.detect_delimiter_manipulation) {
        std::vector<std::string> matched;
        if (CheckDelimiterManipulation(text, matched)) {
            result.injection_detected = true;
            if (result.injection_type == InjectionType::kNone) {
                result.injection_type = InjectionType::kDelimiterManip;
            }
            if (result.risk_level < InjectionRiskLevel::kMedium) {
                result.risk_level = InjectionRiskLevel::kMedium;
            }
            for (const auto& m : matched) {
                result.matched_patterns.push_back("delimiter: " + m);
            }
        }
    }

    if (result.injection_detected) {
        // Calculate confidence based on number of matches and severity
        size_t num_matches = result.matched_patterns.size();
        double base_confidence = 0.5;
        double pattern_boost = std::min(0.4, num_matches * 0.1);
        double risk_boost = static_cast<int>(result.risk_level) * 0.1;
        result.confidence = std::min(1.0, base_confidence + pattern_boost + risk_boost);

        // Build explanation
        std::stringstream ss;
        ss << "Detected " << InjectionTypeToString(result.injection_type);
        ss << " attack (risk: " << RiskLevelToString(result.risk_level) << ")";
        ss << ". Matched " << num_matches << " pattern(s).";
        result.explanation = ss.str();

        // Build recommendation
        switch (result.risk_level) {
            case InjectionRiskLevel::kCritical:
            case InjectionRiskLevel::kHigh:
                result.recommendation = "Block this input and log the attempt";
                break;
            case InjectionRiskLevel::kMedium:
                result.recommendation = "Review input carefully before processing";
                break;
            case InjectionRiskLevel::kLow:
                result.recommendation = "Monitor for patterns; likely benign";
                break;
            default:
                break;
        }
    }

    return result;
}

InjectionDetectionResult PromptInjectionDetector::DetectWithML(
    const std::string& text) {
    // Placeholder for ML-based detection
    // Would use a fine-tuned classifier model
    InjectionDetectionResult result;
    result.metadata["ml_detection"] = "not_implemented";
    return result;
}

InjectionDetectionResult PromptInjectionDetector::DetectSemantic(
    const std::string& text) {
    // Placeholder for semantic analysis
    // Would use embeddings to detect semantic intent
    InjectionDetectionResult result;
    result.metadata["semantic_detection"] = "not_implemented";
    return result;
}

bool PromptInjectionDetector::CheckEncodedContent(
    const std::string& text,
    std::string& decoded,
    std::string& encoding_type) {
    // Look for potential Base64 strings
    std::regex base64_pattern(R"([A-Za-z0-9+/]{20,}={0,2})");
    std::smatch match;
    std::string::const_iterator search_start(text.cbegin());

    while (std::regex_search(search_start, text.cend(), match, base64_pattern)) {
        std::string candidate = match[0];
        if (IsBase64(candidate)) {
            std::string dec = DecodeBase64(candidate);
            // Check if decoded content looks like text
            bool looks_like_text = true;
            for (char c : dec) {
                if (c < 32 && c != '\n' && c != '\r' && c != '\t') {
                    looks_like_text = false;
                    break;
                }
            }
            if (looks_like_text && dec.length() > 10) {
                decoded = dec;
                encoding_type = "base64";
                return true;
            }
        }
        search_start = match.suffix().first;
    }

    // Look for potential hex strings
    std::regex hex_pattern(R"([0-9A-Fa-f]{40,})");
    search_start = text.cbegin();

    while (std::regex_search(search_start, text.cend(), match, hex_pattern)) {
        std::string candidate = match[0];
        if (IsHexString(candidate)) {
            std::string dec = DecodeHex(candidate);
            bool looks_like_text = true;
            for (char c : dec) {
                if (c < 32 && c != '\n' && c != '\r' && c != '\t') {
                    looks_like_text = false;
                    break;
                }
            }
            if (looks_like_text && dec.length() > 10) {
                decoded = dec;
                encoding_type = "hex";
                return true;
            }
        }
        search_start = match.suffix().first;
    }

    return false;
}

bool PromptInjectionDetector::CheckDelimiterManipulation(
    const std::string& text,
    std::vector<std::string>& matched) {
    bool found = false;

    // Check for common delimiter patterns
    std::vector<std::pair<std::regex, std::string>> delimiter_patterns = {
        {std::regex(R"(\[\[|]]|\{\{|\}\}|<<<|>>>)", std::regex::icase), "bracket_delimiter"},
        {std::regex(R"(---+|===+|\*\*\*+)", std::regex::icase), "horizontal_delimiter"},
        {std::regex(R"(```.*```)", std::regex::icase), "code_block"},
        {std::regex(R"(<\|.*\|>)", std::regex::icase), "special_delimiter"},
        {std::regex(R"(\\n\\n\\n|\\r\\n\\r\\n)", std::regex::icase), "whitespace_delimiter"}
    };

    for (const auto& [pattern, name] : delimiter_patterns) {
        if (std::regex_search(text, pattern)) {
            matched.push_back(name);
            found = true;
        }
    }

    return found;
}

bool PromptInjectionDetector::IsInBlocklist(const std::string& text) {
    std::string normalized = NormalizeText(text);
    for (const auto& blocked : blocklist_) {
        if (normalized.find(blocked) != std::string::npos) {
            return true;
        }
    }
    return false;
}

bool PromptInjectionDetector::IsInAllowlist(const std::string& text) {
    std::string normalized = NormalizeText(text);
    return allowlist_.find(normalized) != allowlist_.end();
}

std::string PromptInjectionDetector::NormalizeText(const std::string& text) {
    std::string result;
    result.reserve(text.length());
    for (char c : text) {
        if (std::isalnum(static_cast<unsigned char>(c))) {
            result += std::tolower(static_cast<unsigned char>(c));
        } else if (!result.empty() && result.back() != ' ') {
            result += ' ';
        }
    }
    // Trim trailing space
    while (!result.empty() && result.back() == ' ') {
        result.pop_back();
    }
    return result;
}

std::string PromptInjectionDetector::ComputeHash(const std::string& text) {
    uint64_t hash = HashString(text);
    std::stringstream ss;
    ss << std::hex << hash;
    return ss.str();
}

InjectionDetectionResult PromptInjectionDetector::MergeResults(
    const std::vector<InjectionDetectionResult>& results) {
    InjectionDetectionResult merged;

    for (const auto& result : results) {
        if (result.injection_detected) {
            merged.injection_detected = true;

            if (result.risk_level > merged.risk_level) {
                merged.risk_level = result.risk_level;
                merged.injection_type = result.injection_type;
                merged.injection_start = result.injection_start;
                merged.injection_length = result.injection_length;
            }

            // Average confidence
            if (merged.confidence == 0.0) {
                merged.confidence = result.confidence;
            } else {
                merged.confidence = (merged.confidence + result.confidence) / 2.0;
            }

            // Merge matched patterns
            for (const auto& pattern : result.matched_patterns) {
                merged.matched_patterns.push_back(pattern);
            }

            // Merge metadata
            for (const auto& [key, value] : result.metadata) {
                merged.metadata[key] = value;
            }
        }
    }

    // Build merged explanation if detected
    if (merged.injection_detected) {
        std::stringstream ss;
        ss << "Detected " << InjectionTypeToString(merged.injection_type);
        ss << " (risk: " << RiskLevelToString(merged.risk_level);
        ss << ", confidence: " << std::fixed << std::setprecision(2) << merged.confidence << ")";
        merged.explanation = ss.str();

        switch (merged.risk_level) {
            case InjectionRiskLevel::kCritical:
            case InjectionRiskLevel::kHigh:
                merged.recommendation = "Block this input immediately";
                break;
            case InjectionRiskLevel::kMedium:
                merged.recommendation = "Review and sanitize input before processing";
                break;
            case InjectionRiskLevel::kLow:
                merged.recommendation = "Log and monitor; consider allowing";
                break;
            default:
                break;
        }
    }

    return merged;
}

EvalResult PromptInjectionDetector::ToEvalResult(
    const InjectionDetectionResult& detection) {
    EvalResult result;
    result.evaluator_type = "PromptInjection";

    if (detection.injection_detected) {
        result.score = 1.0 - detection.confidence;  // Lower score = worse (injection found)
        result.verdict = detection.risk_level >= InjectionRiskLevel::kHigh ? "fail" : "warn";
    } else {
        result.score = 1.0;
        result.verdict = "pass";
    }

    result.explanation = detection.explanation;

    result.metadata["injection_type"] = InjectionTypeToString(detection.injection_type);
    result.metadata["risk_level"] = RiskLevelToString(detection.risk_level);
    result.metadata["confidence"] = std::to_string(detection.confidence);
    result.metadata["recommendation"] = detection.recommendation;

    return result;
}

// ============================================================================
// Utility Functions
// ============================================================================

std::unique_ptr<Evaluator> CreatePromptInjectionDetector(
    PromptInjectionConfig config) {
    auto detector = std::make_unique<PromptInjectionDetector>(std::move(config));
    detector->Initialize();
    return detector;
}

std::string InjectionTypeToString(InjectionType type) {
    switch (type) {
        case InjectionType::kNone: return "none";
        case InjectionType::kDirectInjection: return "direct_injection";
        case InjectionType::kJailbreak: return "jailbreak";
        case InjectionType::kRolePlay: return "role_play";
        case InjectionType::kEncodedPayload: return "encoded_payload";
        case InjectionType::kContextLeak: return "context_leak";
        case InjectionType::kDelimiterManip: return "delimiter_manipulation";
        case InjectionType::kRecursivePrompt: return "recursive_prompt";
        case InjectionType::kIndirectInjection: return "indirect_injection";
        case InjectionType::kUnknown: return "unknown";
    }
    return "unknown";
}

std::string RiskLevelToString(InjectionRiskLevel level) {
    switch (level) {
        case InjectionRiskLevel::kNone: return "none";
        case InjectionRiskLevel::kLow: return "low";
        case InjectionRiskLevel::kMedium: return "medium";
        case InjectionRiskLevel::kHigh: return "high";
        case InjectionRiskLevel::kCritical: return "critical";
    }
    return "unknown";
}

std::vector<std::tuple<std::string, InjectionType, InjectionRiskLevel>>
GetDefaultInjectionPatterns() {
    return {
        // Direct injection patterns
        {R"(ignore\s+(all\s+)?(previous|above|prior)\s+(instructions?|prompts?|text))",
         InjectionType::kDirectInjection, InjectionRiskLevel::kHigh},
        {R"(disregard\s+(all\s+)?(previous|above|prior)\s+(instructions?|prompts?))",
         InjectionType::kDirectInjection, InjectionRiskLevel::kHigh},
        {R"(forget\s+(all\s+)?(previous|above|prior)\s+(instructions?|prompts?))",
         InjectionType::kDirectInjection, InjectionRiskLevel::kHigh},
        {R"(new\s+instructions?\s*:)",
         InjectionType::kDirectInjection, InjectionRiskLevel::kMedium},
        {R"(override\s+(system|previous)\s+(prompt|instructions?))",
         InjectionType::kDirectInjection, InjectionRiskLevel::kHigh},

        // Jailbreak patterns
        {R"(DAN\s+mode|do\s+anything\s+now)",
         InjectionType::kJailbreak, InjectionRiskLevel::kCritical},
        {R"(developer\s+mode|god\s+mode)",
         InjectionType::kJailbreak, InjectionRiskLevel::kCritical},
        {R"(jailbreak|jailbroken)",
         InjectionType::kJailbreak, InjectionRiskLevel::kCritical},
        {R"(bypass\s+(safety|content|filter|restriction))",
         InjectionType::kJailbreak, InjectionRiskLevel::kHigh},
        {R"(pretend\s+you\s+(are|have)\s+no\s+(rules|restrictions|limits))",
         InjectionType::kJailbreak, InjectionRiskLevel::kHigh},
        {R"(act\s+as\s+if\s+you\s+(have\s+no|don't\s+have)\s+(rules|restrictions))",
         InjectionType::kJailbreak, InjectionRiskLevel::kHigh},

        // Role play exploits
        {R"(pretend\s+(to\s+be|you\s+are)\s+a\s+(different|new|other)\s+AI)",
         InjectionType::kRolePlay, InjectionRiskLevel::kMedium},
        {R"(you\s+are\s+now\s+(a|an)\s+)",
         InjectionType::kRolePlay, InjectionRiskLevel::kLow},
        {R"(roleplay\s+as\s+)",
         InjectionType::kRolePlay, InjectionRiskLevel::kLow},
        {R"(simulate\s+being\s+)",
         InjectionType::kRolePlay, InjectionRiskLevel::kLow},

        // Context leak attempts
        {R"(what\s+(is|are)\s+(your|the)\s+(system\s+)?(prompt|instructions?))",
         InjectionType::kContextLeak, InjectionRiskLevel::kMedium},
        {R"(reveal\s+(your|the)\s+(system\s+)?(prompt|instructions?))",
         InjectionType::kContextLeak, InjectionRiskLevel::kMedium},
        {R"(show\s+(me\s+)?(your|the)\s+(hidden|secret|original)\s+(prompt|instructions?))",
         InjectionType::kContextLeak, InjectionRiskLevel::kMedium},
        {R"(print\s+(your|the)\s+(system\s+)?(prompt|instructions?))",
         InjectionType::kContextLeak, InjectionRiskLevel::kMedium},
        {R"(repeat\s+(your|the)\s+(initial|original)\s+(prompt|instructions?))",
         InjectionType::kContextLeak, InjectionRiskLevel::kMedium},

        // Recursive prompt injection
        {R"(\[\[SYSTEM\]\]|\[\[USER\]\]|\[\[ASSISTANT\]\])",
         InjectionType::kRecursivePrompt, InjectionRiskLevel::kHigh},
        {R"(<\|im_start\|>|<\|im_end\|>)",
         InjectionType::kRecursivePrompt, InjectionRiskLevel::kHigh},
        {R"(###\s*(System|User|Assistant)\s*:)",
         InjectionType::kRecursivePrompt, InjectionRiskLevel::kMedium},
        {R"(\[INST\]|\[/INST\])",
         InjectionType::kRecursivePrompt, InjectionRiskLevel::kMedium},

        // Delimiter manipulation
        {R"(\n{5,}|\\n{5,})",
         InjectionType::kDelimiterManip, InjectionRiskLevel::kLow},
        {R"(-{10,}|={10,}|\*{10,})",
         InjectionType::kDelimiterManip, InjectionRiskLevel::kLow},

        // Indirect injection via data
        {R"(when\s+you\s+(read|see|process)\s+this)",
         InjectionType::kIndirectInjection, InjectionRiskLevel::kMedium},
        {R"(if\s+you\s+are\s+(an?\s+)?(AI|LLM|language\s+model|assistant))",
         InjectionType::kIndirectInjection, InjectionRiskLevel::kMedium},
        {R"(attention\s+(AI|assistant|model))",
         InjectionType::kIndirectInjection, InjectionRiskLevel::kMedium}
    };
}

}  // namespace pyflare::eval
