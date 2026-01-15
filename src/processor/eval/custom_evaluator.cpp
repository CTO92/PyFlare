/// @file custom_evaluator.cpp
/// @brief Custom evaluator framework implementation

#include "processor/eval/custom_evaluator.h"

#include <algorithm>
#include <chrono>
#include <fstream>
#include <regex>
#include <sstream>

#include <nlohmann/json.hpp>

namespace pyflare::eval {

using json = nlohmann::json;

// ============================================================================
// CustomEvaluatorRegistry Implementation
// ============================================================================

CustomEvaluatorRegistry::CustomEvaluatorRegistry() = default;
CustomEvaluatorRegistry::~CustomEvaluatorRegistry() = default;

absl::Status CustomEvaluatorRegistry::Register(
    const CustomEvaluatorDefinition& definition) {
    auto validation = ValidateDefinition(definition);
    if (!validation.ok()) {
        return validation;
    }

    std::lock_guard<std::mutex> lock(mutex_);

    if (definitions_.find(definition.name) != definitions_.end()) {
        return absl::AlreadyExistsError(
            "Evaluator already registered: " + definition.name);
    }

    auto def = definition;
    def.created_at = std::chrono::system_clock::now();
    def.updated_at = def.created_at;

    definitions_[definition.name] = std::move(def);

    return absl::OkStatus();
}

absl::Status CustomEvaluatorRegistry::Update(
    const CustomEvaluatorDefinition& definition) {
    auto validation = ValidateDefinition(definition);
    if (!validation.ok()) {
        return validation;
    }

    std::lock_guard<std::mutex> lock(mutex_);

    auto it = definitions_.find(definition.name);
    if (it == definitions_.end()) {
        return absl::NotFoundError(
            "Evaluator not found: " + definition.name);
    }

    auto def = definition;
    def.created_at = it->second.created_at;
    def.updated_at = std::chrono::system_clock::now();

    definitions_[definition.name] = std::move(def);

    return absl::OkStatus();
}

absl::Status CustomEvaluatorRegistry::Unregister(const std::string& name) {
    std::lock_guard<std::mutex> lock(mutex_);

    auto it = definitions_.find(name);
    if (it == definitions_.end()) {
        return absl::NotFoundError("Evaluator not found: " + name);
    }

    definitions_.erase(it);
    return absl::OkStatus();
}

absl::StatusOr<std::unique_ptr<Evaluator>>
CustomEvaluatorRegistry::Create(const std::string& name) {
    std::lock_guard<std::mutex> lock(mutex_);

    auto it = definitions_.find(name);
    if (it == definitions_.end()) {
        return absl::NotFoundError("Evaluator not found: " + name);
    }

    if (!it->second.enabled) {
        return absl::FailedPreconditionError(
            "Evaluator is disabled: " + name);
    }

    auto evaluator = std::make_unique<CustomEvaluator>(it->second);
    auto status = evaluator->Initialize();
    if (!status.ok()) {
        return status;
    }

    return evaluator;
}

bool CustomEvaluatorRegistry::Exists(const std::string& name) const {
    std::lock_guard<std::mutex> lock(mutex_);
    return definitions_.find(name) != definitions_.end();
}

std::vector<std::string> CustomEvaluatorRegistry::ListEvaluators() const {
    std::lock_guard<std::mutex> lock(mutex_);
    std::vector<std::string> names;
    names.reserve(definitions_.size());
    for (const auto& [name, _] : definitions_) {
        names.push_back(name);
    }
    std::sort(names.begin(), names.end());
    return names;
}

std::vector<std::string> CustomEvaluatorRegistry::ListByTag(
    const std::string& tag) const {
    std::lock_guard<std::mutex> lock(mutex_);
    std::vector<std::string> names;
    for (const auto& [name, def] : definitions_) {
        if (std::find(def.tags.begin(), def.tags.end(), tag) != def.tags.end()) {
            names.push_back(name);
        }
    }
    std::sort(names.begin(), names.end());
    return names;
}

absl::StatusOr<CustomEvaluatorDefinition>
CustomEvaluatorRegistry::GetDefinition(const std::string& name) const {
    std::lock_guard<std::mutex> lock(mutex_);

    auto it = definitions_.find(name);
    if (it == definitions_.end()) {
        return absl::NotFoundError("Evaluator not found: " + name);
    }

    return it->second;
}

absl::Status CustomEvaluatorRegistry::LoadFromFile(const std::string& path) {
    std::ifstream file(path);
    if (!file.is_open()) {
        return absl::NotFoundError("Could not open file: " + path);
    }

    std::stringstream buffer;
    buffer << file.rdbuf();
    std::string content = buffer.str();

    // Try JSON first, then YAML
    return LoadFromJson(content);
}

absl::Status CustomEvaluatorRegistry::SaveToFile(const std::string& path) const {
    auto json_result = ExportToJson();
    if (!json_result.ok()) {
        return json_result.status();
    }

    std::ofstream file(path);
    if (!file.is_open()) {
        return absl::PermissionDeniedError("Could not write to file: " + path);
    }

    file << *json_result;
    return absl::OkStatus();
}

absl::Status CustomEvaluatorRegistry::LoadFromJson(const std::string& json_str) {
    try {
        auto j = json::parse(json_str);

        if (!j.is_array()) {
            return absl::InvalidArgumentError("Expected JSON array of definitions");
        }

        for (const auto& item : j) {
            CustomEvaluatorDefinition def;
            def.name = item.value("name", "");
            def.description = item.value("description", "");
            def.version = item.value("version", "1.0.0");

            std::string type_str = item.value("type", "rule");
            def.type = StringToCustomEvaluatorType(type_str);

            def.enabled = item.value("enabled", true);

            if (item.contains("tags") && item["tags"].is_array()) {
                for (const auto& tag : item["tags"]) {
                    def.tags.push_back(tag.get<std::string>());
                }
            }

            // Parse type-specific config
            if (item.contains("config")) {
                const auto& config = item["config"];
                switch (def.type) {
                    case CustomEvaluatorType::kRule: {
                        RuleConfig rule;
                        rule.condition_expression = config.value("condition", "");
                        if (config.contains("regex_patterns")) {
                            for (const auto& p : config["regex_patterns"]) {
                                rule.regex_patterns.push_back(p.get<std::string>());
                            }
                        }
                        if (config.contains("required_keywords")) {
                            for (const auto& k : config["required_keywords"]) {
                                rule.required_keywords.push_back(k.get<std::string>());
                            }
                        }
                        if (config.contains("forbidden_keywords")) {
                            for (const auto& k : config["forbidden_keywords"]) {
                                rule.forbidden_keywords.push_back(k.get<std::string>());
                            }
                        }
                        if (config.contains("min_length")) {
                            rule.min_length = config["min_length"].get<size_t>();
                        }
                        if (config.contains("max_length")) {
                            rule.max_length = config["max_length"].get<size_t>();
                        }
                        def.config = rule;
                        break;
                    }
                    case CustomEvaluatorType::kLLM: {
                        LLMEvaluatorConfig llm;
                        llm.api_endpoint = config.value("api_endpoint",
                            "https://api.openai.com/v1/chat/completions");
                        llm.model = config.value("model", "gpt-4o-mini");
                        llm.system_prompt = config.value("system_prompt", "");
                        llm.user_prompt_template = config.value("user_prompt_template", "");
                        llm.temperature = config.value("temperature", 0.0);
                        llm.max_tokens = config.value("max_tokens", 256);
                        def.config = llm;
                        break;
                    }
                    case CustomEvaluatorType::kExternal: {
                        ExternalConfig ext;
                        ext.url = config.value("url", "");
                        ext.method = config.value("method", "POST");
                        ext.score_json_path = config.value("score_json_path", "$.score");
                        def.config = ext;
                        break;
                    }
                    case CustomEvaluatorType::kScript: {
                        ScriptConfig script;
                        script.script = config.value("script", "");
                        script.script_path = config.value("script_path", "");
                        std::string lang = config.value("language", "python");
                        script.language = lang == "javascript"
                            ? ScriptConfig::Language::kJavaScript
                            : ScriptConfig::Language::kPython;
                        def.config = script;
                        break;
                    }
                }
            }

            auto status = Register(def);
            if (!status.ok() && !absl::IsAlreadyExists(status)) {
                return status;
            }
        }

        return absl::OkStatus();
    } catch (const json::exception& e) {
        return absl::InvalidArgumentError(
            "JSON parse error: " + std::string(e.what()));
    }
}

absl::StatusOr<std::string> CustomEvaluatorRegistry::ExportToJson() const {
    std::lock_guard<std::mutex> lock(mutex_);

    json j = json::array();

    for (const auto& [name, def] : definitions_) {
        json item;
        item["name"] = def.name;
        item["description"] = def.description;
        item["version"] = def.version;
        item["type"] = CustomEvaluatorTypeToString(def.type);
        item["enabled"] = def.enabled;
        item["tags"] = def.tags;

        // Serialize type-specific config
        json config;
        switch (def.type) {
            case CustomEvaluatorType::kRule: {
                const auto& rule = std::get<RuleConfig>(def.config);
                config["condition"] = rule.condition_expression;
                config["regex_patterns"] = rule.regex_patterns;
                config["required_keywords"] = rule.required_keywords;
                config["forbidden_keywords"] = rule.forbidden_keywords;
                if (rule.min_length.has_value()) {
                    config["min_length"] = *rule.min_length;
                }
                if (rule.max_length.has_value()) {
                    config["max_length"] = *rule.max_length;
                }
                break;
            }
            case CustomEvaluatorType::kLLM: {
                const auto& llm = std::get<LLMEvaluatorConfig>(def.config);
                config["api_endpoint"] = llm.api_endpoint;
                config["model"] = llm.model;
                config["system_prompt"] = llm.system_prompt;
                config["user_prompt_template"] = llm.user_prompt_template;
                config["temperature"] = llm.temperature;
                config["max_tokens"] = llm.max_tokens;
                break;
            }
            case CustomEvaluatorType::kExternal: {
                const auto& ext = std::get<ExternalConfig>(def.config);
                config["url"] = ext.url;
                config["method"] = ext.method;
                config["score_json_path"] = ext.score_json_path;
                break;
            }
            case CustomEvaluatorType::kScript: {
                const auto& script = std::get<ScriptConfig>(def.config);
                config["script"] = script.script;
                config["script_path"] = script.script_path;
                config["language"] = script.language == ScriptConfig::Language::kPython
                    ? "python" : "javascript";
                break;
            }
        }
        item["config"] = config;

        j.push_back(item);
    }

    return j.dump(2);
}

CustomEvaluatorRegistry::Stats CustomEvaluatorRegistry::GetStats() const {
    std::lock_guard<std::mutex> lock(mutex_);

    Stats stats;
    stats.total_evaluators = definitions_.size();

    for (const auto& [_, def] : definitions_) {
        if (def.enabled) {
            stats.enabled_evaluators++;
        }
        stats.by_type[def.type]++;
    }

    return stats;
}

// ============================================================================
// CustomEvaluator Implementation
// ============================================================================

CustomEvaluator::CustomEvaluator(CustomEvaluatorDefinition definition)
    : definition_(std::move(definition)) {}

CustomEvaluator::~CustomEvaluator() = default;

absl::Status CustomEvaluator::Initialize() {
    if (initialized_) {
        return absl::OkStatus();
    }

    // Pre-compile regexes for rule-based evaluators
    if (definition_.type == CustomEvaluatorType::kRule) {
        const auto& rule = std::get<RuleConfig>(definition_.config);
        for (const auto& pattern : rule.regex_patterns) {
            try {
                compiled_patterns_.emplace_back(
                    pattern, std::regex::icase | std::regex::optimize);
            } catch (const std::regex_error& e) {
                return absl::InvalidArgumentError(
                    "Invalid regex pattern: " + pattern);
            }
        }
    }

    initialized_ = true;
    return absl::OkStatus();
}

absl::StatusOr<EvalResult> CustomEvaluator::Evaluate(
    const InferenceRecord& record) {
    if (!initialized_) {
        return absl::FailedPreconditionError("Evaluator not initialized");
    }

    auto start_time = std::chrono::steady_clock::now();

    EvalResult result;

    switch (definition_.type) {
        case CustomEvaluatorType::kRule:
            result = EvaluateRule(record);
            break;
        case CustomEvaluatorType::kScript:
            result = EvaluateScript(record);
            break;
        case CustomEvaluatorType::kLLM:
            result = EvaluateLLM(record);
            break;
        case CustomEvaluatorType::kExternal:
            result = EvaluateExternal(record);
            break;
    }

    result.evaluator_type = Type();

    auto end_time = std::chrono::steady_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::microseconds>(
        end_time - start_time);

    // Update statistics
    {
        std::lock_guard<std::mutex> lock(stats_mutex_);
        stats_.evaluations++;
        if (result.verdict == "pass") {
            stats_.passes++;
        } else if (result.verdict == "fail") {
            stats_.fails++;
        } else if (result.verdict == "error") {
            stats_.errors++;
        }
        stats_.avg_score = (stats_.avg_score * (stats_.evaluations - 1) +
                           result.score) / stats_.evaluations;
        stats_.avg_latency_ms = (stats_.avg_latency_ms * (stats_.evaluations - 1) +
                                 duration.count() / 1000.0) / stats_.evaluations;
    }

    return result;
}

absl::StatusOr<std::vector<EvalResult>> CustomEvaluator::EvaluateBatch(
    const std::vector<InferenceRecord>& records) {
    std::vector<EvalResult> results;
    results.reserve(records.size());

    for (const auto& record : records) {
        auto result = Evaluate(record);
        if (result.ok()) {
            results.push_back(*result);
        } else {
            EvalResult error_result;
            error_result.evaluator_type = Type();
            error_result.score = 0.0;
            error_result.verdict = "error";
            error_result.explanation = std::string(result.status().message());
            results.push_back(error_result);
        }
    }

    return results;
}

CustomEvaluator::Stats CustomEvaluator::GetStats() const {
    std::lock_guard<std::mutex> lock(stats_mutex_);
    return stats_;
}

EvalResult CustomEvaluator::EvaluateRule(const InferenceRecord& record) {
    const auto& rule = std::get<RuleConfig>(definition_.config);
    EvalResult result;
    result.score = rule.pass_score;
    result.verdict = "pass";

    std::vector<std::string> failures;

    // Check condition expression
    if (!rule.condition_expression.empty()) {
        if (!EvaluateCondition(rule.condition_expression, record)) {
            failures.push_back("Condition not met");
        }
    }

    // Check regex patterns
    for (size_t i = 0; i < compiled_patterns_.size(); ++i) {
        if (!std::regex_search(record.output, compiled_patterns_[i])) {
            failures.push_back("Pattern not matched: " + rule.regex_patterns[i]);
        }
    }

    // Check required keywords
    for (const auto& keyword : rule.required_keywords) {
        if (!ContainsKeyword(record.output, keyword)) {
            failures.push_back("Missing keyword: " + keyword);
        }
    }

    // Check forbidden keywords
    for (const auto& keyword : rule.forbidden_keywords) {
        if (ContainsKeyword(record.output, keyword)) {
            failures.push_back("Forbidden keyword found: " + keyword);
        }
    }

    // Check length constraints
    if (rule.min_length.has_value() && record.output.length() < *rule.min_length) {
        failures.push_back("Output too short (min: " +
            std::to_string(*rule.min_length) + ")");
    }

    if (rule.max_length.has_value() && record.output.length() > *rule.max_length) {
        failures.push_back("Output too long (max: " +
            std::to_string(*rule.max_length) + ")");
    }

    if (!failures.empty()) {
        result.score = rule.fail_score;
        result.verdict = "fail";

        std::stringstream ss;
        ss << "Failed checks: ";
        for (size_t i = 0; i < failures.size(); ++i) {
            if (i > 0) ss << "; ";
            ss << failures[i];
        }
        result.explanation = ss.str();
    } else {
        result.explanation = "All checks passed";
    }

    return result;
}

EvalResult CustomEvaluator::EvaluateScript(const InferenceRecord& record) {
    const auto& script = std::get<ScriptConfig>(definition_.config);

    // Placeholder implementation
    // In production, would use embedded Python/JS interpreter or subprocess
    EvalResult result;
    result.score = 0.0;
    result.verdict = "error";
    result.explanation = "Script execution not yet implemented";
    result.metadata["script_language"] = script.language == ScriptConfig::Language::kPython
        ? "python" : "javascript";

    return result;
}

EvalResult CustomEvaluator::EvaluateLLM(const InferenceRecord& record) {
    // Placeholder implementation
    // In production, would call the LLM API
    const auto& llm = std::get<LLMEvaluatorConfig>(definition_.config);

    EvalResult result;
    result.score = 0.5;
    result.verdict = "warn";
    result.explanation = "LLM evaluation placeholder";
    result.metadata["model"] = llm.model;

    return result;
}

EvalResult CustomEvaluator::EvaluateExternal(const InferenceRecord& record) {
    // Placeholder implementation
    // In production, would make HTTP request to external API
    const auto& ext = std::get<ExternalConfig>(definition_.config);

    EvalResult result;
    result.score = 0.0;
    result.verdict = "error";
    result.explanation = "External API call not yet implemented";
    result.metadata["url"] = ext.url;

    return result;
}

bool CustomEvaluator::EvaluateCondition(
    const std::string& expression,
    const InferenceRecord& record) {
    // Simple condition evaluator
    // Supports: input_len, output_len, >, <, >=, <=, ==, &&, ||

    std::string expr = expression;

    // Replace variables
    std::string input_len_str = std::to_string(record.input.length());
    std::string output_len_str = std::to_string(record.output.length());

    // Replace "len(input)" with actual length
    std::regex input_len_regex(R"(len\s*\(\s*input\s*\))");
    expr = std::regex_replace(expr, input_len_regex, input_len_str);

    // Replace "len(output)" with actual length
    std::regex output_len_regex(R"(len\s*\(\s*output\s*\))");
    expr = std::regex_replace(expr, output_len_regex, output_len_str);

    // Replace input_len and output_len directly
    std::regex input_len_var(R"(\binput_len\b)");
    expr = std::regex_replace(expr, input_len_var, input_len_str);

    std::regex output_len_var(R"(\boutput_len\b)");
    expr = std::regex_replace(expr, output_len_var, output_len_str);

    // Simple expression evaluator for numeric comparisons
    // This is a basic implementation - production would use a proper parser

    // Handle simple "X op Y" comparisons
    std::regex compare_regex(R"((\d+)\s*(>=|<=|>|<|==|!=)\s*(\d+))");
    std::smatch match;

    // Replace all comparisons with "1" (true) or "0" (false)
    while (std::regex_search(expr, match, compare_regex)) {
        int left = std::stoi(match[1]);
        std::string op = match[2];
        int right = std::stoi(match[3]);

        bool result = false;
        if (op == ">") result = left > right;
        else if (op == "<") result = left < right;
        else if (op == ">=") result = left >= right;
        else if (op == "<=") result = left <= right;
        else if (op == "==") result = left == right;
        else if (op == "!=") result = left != right;

        expr = match.prefix().str() + (result ? "1" : "0") + match.suffix().str();
    }

    // Handle && and ||
    std::regex and_regex(R"(1\s*&&\s*1)");
    std::regex and_fail_regex(R"(\d\s*&&\s*\d)");
    std::regex or_regex(R"(1\s*\|\|\s*\d|\d\s*\|\|\s*1)");
    std::regex or_fail_regex(R"(0\s*\|\|\s*0)");

    while (std::regex_search(expr, and_regex) ||
           std::regex_search(expr, and_fail_regex) ||
           std::regex_search(expr, or_regex) ||
           std::regex_search(expr, or_fail_regex)) {
        expr = std::regex_replace(expr, and_regex, "1");
        expr = std::regex_replace(expr, and_fail_regex, "0");
        expr = std::regex_replace(expr, or_regex, "1");
        expr = std::regex_replace(expr, or_fail_regex, "0");
    }

    // Final result
    expr.erase(std::remove_if(expr.begin(), expr.end(), ::isspace), expr.end());
    return expr == "1";
}

bool CustomEvaluator::MatchesRegex(
    const std::string& text,
    const std::string& pattern) {
    try {
        std::regex regex(pattern, std::regex::icase);
        return std::regex_search(text, regex);
    } catch (const std::regex_error&) {
        return false;
    }
}

bool CustomEvaluator::ContainsKeyword(
    const std::string& text,
    const std::string& keyword) {
    // Case-insensitive search
    std::string lower_text = text;
    std::string lower_keyword = keyword;
    std::transform(lower_text.begin(), lower_text.end(), lower_text.begin(), ::tolower);
    std::transform(lower_keyword.begin(), lower_keyword.end(), lower_keyword.begin(), ::tolower);
    return lower_text.find(lower_keyword) != std::string::npos;
}

// ============================================================================
// Utility Functions
// ============================================================================

std::unique_ptr<CustomEvaluatorRegistry> CreateCustomEvaluatorRegistry() {
    return std::make_unique<CustomEvaluatorRegistry>();
}

std::string CustomEvaluatorTypeToString(CustomEvaluatorType type) {
    switch (type) {
        case CustomEvaluatorType::kRule: return "rule";
        case CustomEvaluatorType::kScript: return "script";
        case CustomEvaluatorType::kLLM: return "llm";
        case CustomEvaluatorType::kExternal: return "external";
    }
    return "unknown";
}

CustomEvaluatorType StringToCustomEvaluatorType(const std::string& str) {
    if (str == "rule") return CustomEvaluatorType::kRule;
    if (str == "script") return CustomEvaluatorType::kScript;
    if (str == "llm") return CustomEvaluatorType::kLLM;
    if (str == "external") return CustomEvaluatorType::kExternal;
    return CustomEvaluatorType::kRule;
}

absl::Status ValidateDefinition(const CustomEvaluatorDefinition& definition) {
    if (definition.name.empty()) {
        return absl::InvalidArgumentError("Evaluator name cannot be empty");
    }

    if (definition.name.length() > 128) {
        return absl::InvalidArgumentError("Evaluator name too long (max 128)");
    }

    // Validate name format (alphanumeric, underscore, hyphen)
    std::regex name_regex(R"(^[a-zA-Z][a-zA-Z0-9_-]*$)");
    if (!std::regex_match(definition.name, name_regex)) {
        return absl::InvalidArgumentError(
            "Invalid evaluator name format (must start with letter, "
            "contain only alphanumeric, underscore, hyphen)");
    }

    return absl::OkStatus();
}

CustomEvaluatorDefinition CreateLengthCheckEvaluator(
    const std::string& name,
    size_t min_length,
    size_t max_length) {
    CustomEvaluatorDefinition def;
    def.name = name;
    def.description = "Check output length constraints";
    def.type = CustomEvaluatorType::kRule;
    def.tags = {"length", "quality"};

    RuleConfig rule;
    rule.min_length = min_length;
    rule.max_length = max_length;
    def.config = rule;

    return def;
}

CustomEvaluatorDefinition CreateKeywordEvaluator(
    const std::string& name,
    const std::vector<std::string>& required_keywords,
    const std::vector<std::string>& forbidden_keywords) {
    CustomEvaluatorDefinition def;
    def.name = name;
    def.description = "Check for required/forbidden keywords";
    def.type = CustomEvaluatorType::kRule;
    def.tags = {"keyword", "content"};

    RuleConfig rule;
    rule.required_keywords = required_keywords;
    rule.forbidden_keywords = forbidden_keywords;
    def.config = rule;

    return def;
}

CustomEvaluatorDefinition CreateLLMEvaluator(
    const std::string& name,
    const std::string& system_prompt,
    const std::string& user_prompt_template) {
    CustomEvaluatorDefinition def;
    def.name = name;
    def.description = "LLM-based evaluation";
    def.type = CustomEvaluatorType::kLLM;
    def.tags = {"llm", "quality"};

    LLMEvaluatorConfig llm;
    llm.system_prompt = system_prompt;
    llm.user_prompt_template = user_prompt_template;
    def.config = llm;

    return def;
}

}  // namespace pyflare::eval
