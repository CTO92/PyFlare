/// @file counterfactual.cpp
/// @brief Counterfactual explanation generation implementation

#include "processor/rca/counterfactual.h"

#include <algorithm>
#include <cctype>
#include <chrono>
#include <random>
#include <regex>
#include <sstream>

#include <nlohmann/json.hpp>

namespace pyflare::rca {

using json = nlohmann::json;

namespace {

// Simple hash for cache keys
uint64_t HashString(const std::string& str) {
    uint64_t hash = 14695981039346656037ULL;
    for (char c : str) {
        hash ^= static_cast<unsigned char>(c);
        hash *= 1099511628211ULL;
    }
    return hash;
}

// Levenshtein distance for edit distance computation
size_t LevenshteinDistance(const std::string& s1, const std::string& s2) {
    size_t m = s1.length();
    size_t n = s2.length();

    std::vector<std::vector<size_t>> dp(m + 1, std::vector<size_t>(n + 1));

    for (size_t i = 0; i <= m; ++i) dp[i][0] = i;
    for (size_t j = 0; j <= n; ++j) dp[0][j] = j;

    for (size_t i = 1; i <= m; ++i) {
        for (size_t j = 1; j <= n; ++j) {
            if (s1[i - 1] == s2[j - 1]) {
                dp[i][j] = dp[i - 1][j - 1];
            } else {
                dp[i][j] = 1 + std::min({dp[i - 1][j], dp[i][j - 1], dp[i - 1][j - 1]});
            }
        }
    }

    return dp[m][n];
}

}  // namespace

CounterfactualGenerator::CounterfactualGenerator(CounterfactualConfig config)
    : config_(std::move(config)) {}

CounterfactualGenerator::~CounterfactualGenerator() = default;

absl::Status CounterfactualGenerator::Initialize() {
    if (initialized_) {
        return absl::OkStatus();
    }

    // Check for API key if using LLM method
    if (config_.method == CounterfactualConfig::Method::kLLMBased) {
        if (config_.llm.api_key.empty()) {
            const char* env_key = std::getenv("OPENAI_API_KEY");
            if (env_key != nullptr) {
                config_.llm.api_key = env_key;
            }
        }

        if (config_.llm.api_key.empty()) {
            return absl::FailedPreconditionError(
                "LLM API key not provided for LLM-based counterfactual generation");
        }
    }

    initialized_ = true;
    return absl::OkStatus();
}

absl::StatusOr<Counterfactual> CounterfactualGenerator::Generate(
    const eval::InferenceRecord& record,
    const std::string& target_outcome) {
    if (!initialized_) {
        return absl::FailedPreconditionError("Generator not initialized");
    }

    auto start_time = std::chrono::steady_clock::now();

    // Check cache
    if (config_.enable_cache) {
        std::string cache_key = ComputeCacheKey(record, target_outcome);
        std::lock_guard<std::mutex> lock(cache_mutex_);
        auto it = cache_.find(cache_key);
        if (it != cache_.end()) {
            std::lock_guard<std::mutex> stats_lock(stats_mutex_);
            stats_.cache_hits++;
            return it->second;
        }
    }

    absl::StatusOr<Counterfactual> result;

    switch (config_.method) {
        case CounterfactualConfig::Method::kLLMBased:
            result = GenerateWithLLM(record, target_outcome);
            break;
        case CounterfactualConfig::Method::kTextPerturbation:
            result = GenerateWithPerturbation(record, target_outcome);
            break;
        case CounterfactualConfig::Method::kFeatureAttribution:
            result = GenerateWithAttribution(record, target_outcome);
            break;
        case CounterfactualConfig::Method::kGradientBased:
            // Not implemented - requires model access
            return absl::UnimplementedError(
                "Gradient-based counterfactual generation not implemented");
    }

    auto end_time = std::chrono::steady_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(
        end_time - start_time);

    // Update statistics
    {
        std::lock_guard<std::mutex> lock(stats_mutex_);
        stats_.total_generations++;
        if (result.ok()) {
            stats_.successful_generations++;
            stats_.avg_edit_distance = (stats_.avg_edit_distance *
                (stats_.successful_generations - 1) + result->edit_distance) /
                stats_.successful_generations;
            stats_.avg_confidence = (stats_.avg_confidence *
                (stats_.successful_generations - 1) + result->confidence) /
                stats_.successful_generations;
        }
        stats_.avg_generation_time_ms = (stats_.avg_generation_time_ms *
            (stats_.total_generations - 1) + duration.count()) /
            stats_.total_generations;
    }

    // Cache result
    if (result.ok() && config_.enable_cache) {
        std::string cache_key = ComputeCacheKey(record, target_outcome);
        std::lock_guard<std::mutex> lock(cache_mutex_);
        cache_[cache_key] = *result;
    }

    return result;
}

absl::StatusOr<std::vector<Counterfactual>> CounterfactualGenerator::GenerateMultiple(
    const eval::InferenceRecord& record,
    const std::string& target_outcome,
    size_t count) {
    std::vector<Counterfactual> results;
    results.reserve(count);

    for (size_t i = 0; i < count; ++i) {
        auto cf = Generate(record, target_outcome);
        if (cf.ok()) {
            results.push_back(*cf);
        }
    }

    if (results.empty()) {
        return absl::InternalError("Failed to generate any counterfactuals");
    }

    return results;
}

absl::StatusOr<std::vector<Counterfactual>> CounterfactualGenerator::GenerateDiverse(
    const eval::InferenceRecord& record,
    const std::string& target_outcome) {
    std::vector<Counterfactual> results;

    // Generate with different prompts for diversity
    std::vector<std::string> categories = {"minimal", "diverse", "contrastive"};

    for (const auto& category : categories) {
        std::string prompt = BuildLLMPrompt(record, target_outcome, category);
        auto response = CallLLM(prompt);
        if (response.ok()) {
            auto cf = ParseLLMResponse(*response, record);
            if (cf.ok()) {
                cf->category = category;
                results.push_back(*cf);
            }
        }
    }

    if (results.empty()) {
        return absl::InternalError("Failed to generate diverse counterfactuals");
    }

    return results;
}

absl::StatusOr<Counterfactual> CounterfactualGenerator::GenerateMinimal(
    const eval::InferenceRecord& record,
    const std::string& target_outcome) {
    auto counterfactuals = GenerateMultiple(record, target_outcome, 5);
    if (!counterfactuals.ok()) {
        return counterfactuals.status();
    }

    // Find the one with smallest edit distance
    auto& cfs = *counterfactuals;
    auto min_it = std::min_element(cfs.begin(), cfs.end(),
        [](const Counterfactual& a, const Counterfactual& b) {
            return a.edit_distance < b.edit_distance;
        });

    if (min_it == cfs.end()) {
        return absl::InternalError("No counterfactuals generated");
    }

    min_it->category = "minimal";
    return *min_it;
}

absl::StatusOr<std::vector<std::pair<std::string, double>>>
CounterfactualGenerator::AnalyzeInfluence(const eval::InferenceRecord& record) {
    std::vector<std::pair<std::string, double>> influences;

    // Tokenize input
    auto tokens = TokenizeText(record.input);

    // For each token, measure influence by deletion
    for (size_t i = 0; i < tokens.size(); ++i) {
        // Create input with token removed
        std::vector<std::string> modified_tokens = tokens;
        std::string removed_token = modified_tokens[i];
        modified_tokens.erase(modified_tokens.begin() + i);
        std::string modified_input = JoinTokens(modified_tokens);

        // Compute similarity as proxy for influence
        double similarity = ComputeSimilarity(record.input, modified_input);
        double influence = 1.0 - similarity;

        influences.emplace_back(removed_token, influence);
    }

    // Sort by influence descending
    std::sort(influences.begin(), influences.end(),
        [](const auto& a, const auto& b) { return a.second > b.second; });

    return influences;
}

absl::StatusOr<std::vector<std::string>> CounterfactualGenerator::FindCriticalTokens(
    const eval::InferenceRecord& record) {
    auto influences = AnalyzeInfluence(record);
    if (!influences.ok()) {
        return influences.status();
    }

    std::vector<std::string> critical;
    for (const auto& [token, influence] : *influences) {
        if (influence > 0.1) {  // Threshold for "critical"
            critical.push_back(token);
        }
        if (critical.size() >= 10) {
            break;
        }
    }

    return critical;
}

absl::StatusOr<bool> CounterfactualGenerator::Validate(
    Counterfactual& counterfactual,
    std::function<std::string(const std::string&)> model_callback) {
    if (!model_callback) {
        return absl::InvalidArgumentError("Model callback is required");
    }

    std::string actual_output = model_callback(counterfactual.modified_input);
    counterfactual.validated = true;
    counterfactual.validation_output = actual_output;

    // Check if output is closer to expected
    double original_distance = ComputeEditDistance(
        counterfactual.original_output, counterfactual.expected_output);
    double new_distance = ComputeEditDistance(
        actual_output, counterfactual.expected_output);

    bool success = new_distance < original_distance;

    {
        std::lock_guard<std::mutex> lock(stats_mutex_);
        stats_.validations++;
        if (success) {
            stats_.successful_validations++;
        }
    }

    return success;
}

void CounterfactualGenerator::SetConfig(CounterfactualConfig config) {
    config_ = std::move(config);
}

void CounterfactualGenerator::ClearCache() {
    std::lock_guard<std::mutex> lock(cache_mutex_);
    cache_.clear();
}

CounterfactualGenerator::Stats CounterfactualGenerator::GetStats() const {
    std::lock_guard<std::mutex> lock(stats_mutex_);
    return stats_;
}

void CounterfactualGenerator::ResetStats() {
    std::lock_guard<std::mutex> lock(stats_mutex_);
    stats_ = Stats{};
}

// ============================================================================
// Private Implementation
// ============================================================================

absl::StatusOr<Counterfactual> CounterfactualGenerator::GenerateWithLLM(
    const eval::InferenceRecord& record,
    const std::string& target_outcome) {
    std::string prompt = BuildLLMPrompt(record, target_outcome, "standard");
    auto response = CallLLM(prompt);
    if (!response.ok()) {
        return response.status();
    }

    return ParseLLMResponse(*response, record);
}

absl::StatusOr<Counterfactual> CounterfactualGenerator::GenerateWithPerturbation(
    const eval::InferenceRecord& record,
    const std::string& target_outcome) {
    auto perturbations = GeneratePerturbations(record.input);

    if (perturbations.empty()) {
        return absl::InternalError("No perturbations generated");
    }

    // Select the most different one that stays within similarity bounds
    Counterfactual best;
    best.confidence = 0.0;

    for (const auto& perturbed : perturbations) {
        double similarity = ComputeSimilarity(record.input, perturbed);

        if (similarity >= config_.min_similarity && similarity <= config_.max_similarity) {
            Counterfactual cf;
            cf.original_input = record.input;
            cf.original_output = record.output;
            cf.modified_input = perturbed;
            cf.expected_output = target_outcome;
            cf.input_similarity = similarity;
            cf.edit_distance = ComputeEditDistance(record.input, perturbed);
            cf.changes = ExtractChanges(record.input, perturbed);
            cf.confidence = 1.0 - similarity;  // More different = higher confidence
            cf.category = "perturbation";

            if (cf.confidence > best.confidence) {
                best = cf;
            }
        }
    }

    if (best.confidence == 0.0) {
        return absl::NotFoundError("No valid perturbation found");
    }

    best.explanation = "Generated by text perturbation with " +
        std::to_string(best.changes.size()) + " change(s)";

    return best;
}

absl::StatusOr<Counterfactual> CounterfactualGenerator::GenerateWithAttribution(
    const eval::InferenceRecord& record,
    const std::string& target_outcome) {
    // Find critical tokens
    auto critical = FindCriticalTokens(record);
    if (!critical.ok()) {
        return critical.status();
    }

    if (critical->empty()) {
        return absl::NotFoundError("No critical tokens found");
    }

    // Modify the most critical token
    auto tokens = TokenizeText(record.input);
    std::string most_critical = (*critical)[0];

    // Find and replace the critical token
    std::string modified_input = record.input;
    size_t pos = modified_input.find(most_critical);
    if (pos != std::string::npos) {
        modified_input.replace(pos, most_critical.length(), "[MODIFIED]");
    }

    Counterfactual cf;
    cf.original_input = record.input;
    cf.original_output = record.output;
    cf.modified_input = modified_input;
    cf.expected_output = target_outcome;
    cf.input_similarity = ComputeSimilarity(record.input, modified_input);
    cf.edit_distance = ComputeEditDistance(record.input, modified_input);
    cf.changes = ExtractChanges(record.input, modified_input);
    cf.confidence = 0.7;
    cf.category = "attribution";
    cf.explanation = "Modified critical token: " + most_critical;

    return cf;
}

std::string CounterfactualGenerator::BuildLLMPrompt(
    const eval::InferenceRecord& record,
    const std::string& target_outcome,
    const std::string& category) {
    std::stringstream ss;

    ss << "You are an AI assistant that generates counterfactual explanations.\n\n";
    ss << "Given an input-output pair from an AI model, generate a modified input ";
    ss << "that would likely produce a different, specified outcome.\n\n";

    ss << "Original Input:\n" << record.input << "\n\n";
    ss << "Original Output:\n" << record.output << "\n\n";
    ss << "Desired Outcome:\n" << target_outcome << "\n\n";

    if (category == "minimal") {
        ss << "Generate the SMALLEST possible change to the input that would ";
        ss << "achieve the desired outcome. Change as few words as possible.\n\n";
    } else if (category == "diverse") {
        ss << "Generate a CREATIVE and DIFFERENT modification to the input. ";
        ss << "Try an unconventional approach.\n\n";
    } else if (category == "contrastive") {
        ss << "Generate a modification that clearly CONTRASTS with the original. ";
        ss << "Make the difference obvious and instructive.\n\n";
    }

    ss << "Respond in JSON format:\n";
    ss << "{\n";
    ss << "  \"modified_input\": \"the modified input text\",\n";
    ss << "  \"expected_output\": \"what output you expect\",\n";
    ss << "  \"changes\": [\n";
    ss << "    {\n";
    ss << "      \"type\": \"replacement|deletion|addition\",\n";
    ss << "      \"original\": \"original text\",\n";
    ss << "      \"new\": \"new text\",\n";
    ss << "      \"explanation\": \"why this change matters\"\n";
    ss << "    }\n";
    ss << "  ],\n";
    ss << "  \"explanation\": \"overall explanation of the counterfactual\",\n";
    ss << "  \"confidence\": 0.0-1.0\n";
    ss << "}\n";

    return ss.str();
}

absl::StatusOr<Counterfactual> CounterfactualGenerator::ParseLLMResponse(
    const std::string& response,
    const eval::InferenceRecord& record) {
    try {
        // Find JSON in response
        size_t start = response.find('{');
        size_t end = response.rfind('}');
        if (start == std::string::npos || end == std::string::npos) {
            return absl::InvalidArgumentError("No JSON found in LLM response");
        }

        std::string json_str = response.substr(start, end - start + 1);
        auto j = json::parse(json_str);

        Counterfactual cf;
        cf.original_input = record.input;
        cf.original_output = record.output;
        cf.modified_input = j.value("modified_input", "");
        cf.expected_output = j.value("expected_output", "");
        cf.explanation = j.value("explanation", "");
        cf.confidence = j.value("confidence", 0.5);

        // Parse changes
        if (j.contains("changes") && j["changes"].is_array()) {
            for (const auto& change_json : j["changes"]) {
                CounterfactualChange change;
                std::string type_str = change_json.value("type", "replacement");
                if (type_str == "deletion") {
                    change.type = CounterfactualChange::ChangeType::kDeletion;
                } else if (type_str == "addition") {
                    change.type = CounterfactualChange::ChangeType::kAddition;
                } else {
                    change.type = CounterfactualChange::ChangeType::kReplacement;
                }
                change.original_content = change_json.value("original", "");
                change.new_content = change_json.value("new", "");
                change.explanation = change_json.value("explanation", "");
                cf.changes.push_back(change);
            }
        }

        // Compute metrics
        cf.input_similarity = ComputeSimilarity(record.input, cf.modified_input);
        cf.edit_distance = ComputeEditDistance(record.input, cf.modified_input);

        return cf;
    } catch (const json::exception& e) {
        return absl::InvalidArgumentError(
            "Failed to parse LLM response: " + std::string(e.what()));
    }
}

absl::StatusOr<std::string> CounterfactualGenerator::CallLLM(
    const std::string& prompt) {
    // Placeholder implementation
    // In production, would call the actual LLM API

    // For now, return a mock response
    json mock_response;
    mock_response["modified_input"] = "Modified version of the input";
    mock_response["expected_output"] = "Expected different output";
    mock_response["changes"] = json::array();
    mock_response["explanation"] = "This is a placeholder counterfactual";
    mock_response["confidence"] = 0.7;

    return mock_response.dump();
}

std::vector<std::string> CounterfactualGenerator::GeneratePerturbations(
    const std::string& text) {
    std::vector<std::string> perturbations;
    auto tokens = TokenizeText(text);

    if (tokens.empty()) {
        return perturbations;
    }

    std::random_device rd;
    std::mt19937 gen(rd());

    // Word deletion
    if (config_.perturbation.enable_word_deletion && tokens.size() > 1) {
        for (size_t i = 0; i < std::min(tokens.size(), config_.perturbation.max_word_changes); ++i) {
            std::uniform_int_distribution<> dis(0, tokens.size() - 1);
            size_t idx = dis(gen);
            auto modified = tokens;
            modified.erase(modified.begin() + idx);
            perturbations.push_back(JoinTokens(modified));
        }
    }

    // Word replacement (with placeholder)
    if (config_.perturbation.enable_word_replacement) {
        for (size_t i = 0; i < std::min(tokens.size(), config_.perturbation.max_word_changes); ++i) {
            std::uniform_int_distribution<> dis(0, tokens.size() - 1);
            size_t idx = dis(gen);
            auto modified = tokens;
            modified[idx] = "[REPLACED]";
            perturbations.push_back(JoinTokens(modified));
        }
    }

    // Word insertion
    if (config_.perturbation.enable_word_insertion) {
        for (size_t i = 0; i < config_.perturbation.max_word_changes; ++i) {
            std::uniform_int_distribution<> dis(0, tokens.size());
            size_t idx = dis(gen);
            auto modified = tokens;
            modified.insert(modified.begin() + idx, "[INSERTED]");
            perturbations.push_back(JoinTokens(modified));
        }
    }

    return perturbations;
}

std::vector<std::string> CounterfactualGenerator::TokenizeText(
    const std::string& text) {
    std::vector<std::string> tokens;
    std::string current;

    for (char c : text) {
        if (std::isspace(static_cast<unsigned char>(c))) {
            if (!current.empty()) {
                tokens.push_back(current);
                current.clear();
            }
        } else {
            current += c;
        }
    }

    if (!current.empty()) {
        tokens.push_back(current);
    }

    return tokens;
}

std::string CounterfactualGenerator::JoinTokens(
    const std::vector<std::string>& tokens) {
    std::string result;
    for (size_t i = 0; i < tokens.size(); ++i) {
        if (i > 0) result += " ";
        result += tokens[i];
    }
    return result;
}

double CounterfactualGenerator::ComputeSimilarity(
    const std::string& a,
    const std::string& b) {
    if (a.empty() && b.empty()) return 1.0;
    if (a.empty() || b.empty()) return 0.0;

    size_t edit_dist = LevenshteinDistance(a, b);
    size_t max_len = std::max(a.length(), b.length());

    return 1.0 - static_cast<double>(edit_dist) / max_len;
}

size_t CounterfactualGenerator::ComputeEditDistance(
    const std::string& a,
    const std::string& b) {
    return LevenshteinDistance(a, b);
}

std::vector<CounterfactualChange> CounterfactualGenerator::ExtractChanges(
    const std::string& original,
    const std::string& modified) {
    std::vector<CounterfactualChange> changes;

    auto orig_tokens = TokenizeText(original);
    auto mod_tokens = TokenizeText(modified);

    // Simple diff: find added/removed tokens
    std::unordered_map<std::string, int> orig_counts, mod_counts;

    for (const auto& t : orig_tokens) orig_counts[t]++;
    for (const auto& t : mod_tokens) mod_counts[t]++;

    // Find deletions
    for (const auto& [token, count] : orig_counts) {
        int mod_count = mod_counts[token];
        if (mod_count < count) {
            CounterfactualChange change;
            change.type = CounterfactualChange::ChangeType::kDeletion;
            change.original_content = token;
            change.importance = 1.0 / orig_tokens.size();
            changes.push_back(change);
        }
    }

    // Find additions
    for (const auto& [token, count] : mod_counts) {
        int orig_count = orig_counts[token];
        if (orig_count < count) {
            CounterfactualChange change;
            change.type = CounterfactualChange::ChangeType::kAddition;
            change.new_content = token;
            change.importance = 1.0 / mod_tokens.size();
            changes.push_back(change);
        }
    }

    return changes;
}

std::string CounterfactualGenerator::ComputeCacheKey(
    const eval::InferenceRecord& record,
    const std::string& target_outcome) {
    std::string combined = record.input + "|" + record.output + "|" + target_outcome;
    uint64_t hash = HashString(combined);
    std::stringstream ss;
    ss << std::hex << hash;
    return ss.str();
}

// ============================================================================
// Utility Functions
// ============================================================================

std::unique_ptr<CounterfactualGenerator> CreateCounterfactualGenerator(
    CounterfactualConfig config) {
    auto gen = std::make_unique<CounterfactualGenerator>(std::move(config));
    gen->Initialize();
    return gen;
}

std::string ChangeTypeToString(CounterfactualChange::ChangeType type) {
    switch (type) {
        case CounterfactualChange::ChangeType::kAddition: return "addition";
        case CounterfactualChange::ChangeType::kDeletion: return "deletion";
        case CounterfactualChange::ChangeType::kReplacement: return "replacement";
        case CounterfactualChange::ChangeType::kReordering: return "reordering";
    }
    return "unknown";
}

std::string CounterfactualMethodToString(CounterfactualConfig::Method method) {
    switch (method) {
        case CounterfactualConfig::Method::kTextPerturbation: return "text_perturbation";
        case CounterfactualConfig::Method::kFeatureAttribution: return "feature_attribution";
        case CounterfactualConfig::Method::kGradientBased: return "gradient_based";
        case CounterfactualConfig::Method::kLLMBased: return "llm_based";
    }
    return "unknown";
}

}  // namespace pyflare::rca
