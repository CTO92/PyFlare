/// @file llm_judge.cpp
/// @brief LLM-as-Judge evaluator implementation

#include "processor/eval/llm_judge.h"

#include <algorithm>
#include <chrono>
#include <regex>

#include <nlohmann/json.hpp>
#include <spdlog/spdlog.h>

#ifdef PYFLARE_HAS_HTTPLIB
#include <httplib.h>
#endif

namespace pyflare::eval {

using json = nlohmann::json;
using namespace std::chrono;

// =============================================================================
// Default Prompt Templates
// =============================================================================

JudgePromptTemplate CreateHallucinationPromptTemplate() {
    JudgePromptTemplate tmpl;

    tmpl.system_prompt = R"(You are an expert evaluator assessing AI outputs for hallucinations.

A hallucination is when the AI makes claims that:
1. Are not supported by the provided context/sources
2. Contradict known facts
3. Include made-up details, dates, numbers, or names
4. State opinions as facts without attribution

Evaluate the output carefully and provide:
1. A verdict: PASS (no hallucination), FAIL (contains hallucination), or UNSURE
2. A confidence score from 0.0 to 1.0
3. A brief explanation of your reasoning
4. Specific issues found (if any)

Respond in JSON format.)";

    tmpl.user_prompt_template = R"(Evaluate the following AI output for hallucinations:

**User Input/Question:**
{input}

**AI Output:**
{output}

**Provided Context/Sources (if any):**
{context}

Analyze whether the output contains any hallucinations, unsupported claims, or factual errors.
Respond with a JSON object containing: verdict, score, explanation, has_hallucination, has_factual_error, has_contradiction, has_unsupported_claim)";

    tmpl.response_format = R"({
  "verdict": "PASS" | "FAIL" | "UNSURE",
  "score": 0.0-1.0,
  "explanation": "...",
  "has_hallucination": true/false,
  "has_factual_error": true/false,
  "has_contradiction": true/false,
  "has_unsupported_claim": true/false
})";

    return tmpl;
}

JudgePromptTemplate CreateQualityPromptTemplate() {
    JudgePromptTemplate tmpl;

    tmpl.system_prompt = R"(You are an expert evaluator assessing AI output quality.

Evaluate the output on these dimensions:
1. Relevance: Does it address the user's question/request?
2. Accuracy: Is the information correct?
3. Completeness: Does it fully answer the question?
4. Clarity: Is it well-written and easy to understand?
5. Helpfulness: Is it useful to the user?

Provide an overall score and detailed feedback.
Respond in JSON format.)";

    tmpl.user_prompt_template = R"(Evaluate the quality of this AI output:

**User Input:**
{input}

**AI Output:**
{output}

**Expected Output (if available):**
{expected}

Rate the output quality and provide feedback.
Respond with JSON: verdict, score, explanation, relevance_score, accuracy_score, completeness_score, clarity_score)";

    return tmpl;
}

JudgePromptTemplate CreateRAGGroundingPromptTemplate() {
    JudgePromptTemplate tmpl;

    tmpl.system_prompt = R"(You are an expert evaluator for RAG (Retrieval-Augmented Generation) systems.

Evaluate whether the AI output is properly grounded in the retrieved context:
1. All claims should be supported by the context
2. The output should not add information not in the context
3. The output should accurately represent the context
4. Key information from relevant context should be included

Respond in JSON format with your evaluation.)";

    tmpl.user_prompt_template = R"(Evaluate the grounding of this RAG response:

**User Query:**
{input}

**Retrieved Context:**
{context}

**AI Response:**
{output}

Assess whether the response is properly grounded in the retrieved context.
Respond with JSON: verdict, score, explanation, grounding_issues)";

    return tmpl;
}

// =============================================================================
// Implementation Class
// =============================================================================

class LLMJudgeEvaluator::Impl {
public:
    Impl(const LLMJudgeConfig& config) : config_(config) {}

#ifdef PYFLARE_HAS_HTTPLIB
    httplib::Client* GetClient() {
        if (!client_) {
            // Parse endpoint URL
            std::regex url_regex(R"(https?://([^/]+)(/.*)?)", std::regex::icase);
            std::smatch match;
            if (std::regex_match(config_.api_endpoint, match, url_regex)) {
                std::string host = match[1].str();
                bool is_https = config_.api_endpoint.find("https") == 0;

                if (is_https) {
                    client_ = std::make_unique<httplib::Client>(
                        "https://" + host);
                } else {
                    client_ = std::make_unique<httplib::Client>(
                        "http://" + host);
                }

                client_->set_connection_timeout(config_.timeout);
                client_->set_read_timeout(config_.timeout);
            }
        }
        return client_.get();
    }

    std::unique_ptr<httplib::Client> client_;
#endif

    const LLMJudgeConfig& config_;
    std::unordered_map<std::string, std::pair<JudgeVerdict, system_clock::time_point>> cache_;
    std::mutex cache_mutex_;
};

// =============================================================================
// LLMJudgeEvaluator Implementation
// =============================================================================

LLMJudgeEvaluator::LLMJudgeEvaluator(LLMJudgeConfig config)
    : config_(std::move(config)),
      hallucination_prompt_(CreateHallucinationPromptTemplate()),
      quality_prompt_(CreateQualityPromptTemplate()),
      impl_(std::make_unique<Impl>(config_)) {}

LLMJudgeEvaluator::~LLMJudgeEvaluator() = default;

absl::Status LLMJudgeEvaluator::Initialize() {
    // Check API key
    if (config_.api_key.empty()) {
        // Try environment variable
        const char* env_key = std::getenv("OPENAI_API_KEY");
        if (env_key) {
            config_.api_key = env_key;
        } else {
            return absl::FailedPreconditionError(
                "API key not set. Set api_key in config or OPENAI_API_KEY env var");
        }
    }

    spdlog::info("LLMJudgeEvaluator initialized with model: {}",
                 config_.judge_model);
    return absl::OkStatus();
}

absl::StatusOr<EvalResult> LLMJudgeEvaluator::Evaluate(
    const InferenceRecord& record) {

    auto verdict = EvaluateHallucination(record);
    if (!verdict.ok()) {
        return verdict.status();
    }

    EvalResult result;
    result.evaluator_type = "LLMJudge";
    result.score = verdict->score;

    switch (verdict->result) {
        case JudgeVerdict::Result::kPass:
            result.verdict = "pass";
            break;
        case JudgeVerdict::Result::kFail:
            result.verdict = "fail";
            break;
        case JudgeVerdict::Result::kUnsure:
            result.verdict = "warn";
            break;
        case JudgeVerdict::Result::kError:
            result.verdict = "error";
            break;
    }

    result.explanation = verdict->explanation;

    if (verdict->has_hallucination) {
        result.metadata["has_hallucination"] = "true";
    }
    if (verdict->has_factual_error) {
        result.metadata["has_factual_error"] = "true";
    }
    if (verdict->has_contradiction) {
        result.metadata["has_contradiction"] = "true";
    }
    if (verdict->has_unsupported_claim) {
        result.metadata["has_unsupported_claim"] = "true";
    }

    // Update stats
    stats_.total_evaluations++;
    if (result.verdict == "pass") stats_.pass_count++;
    else if (result.verdict == "fail") stats_.fail_count++;

    return result;
}

absl::StatusOr<std::vector<EvalResult>> LLMJudgeEvaluator::EvaluateBatch(
    const std::vector<InferenceRecord>& records) {

    std::vector<EvalResult> results;
    results.reserve(records.size());

    // For now, evaluate sequentially
    // TODO: Add parallel processing with batch_size
    for (const auto& record : records) {
        auto result = Evaluate(record);
        if (result.ok()) {
            results.push_back(std::move(*result));
        } else {
            // Add error result
            EvalResult err;
            err.evaluator_type = "LLMJudge";
            err.score = 0.0;
            err.verdict = "error";
            err.explanation = std::string(result.status().message());
            results.push_back(std::move(err));
        }
    }

    return results;
}

absl::StatusOr<JudgeVerdict> LLMJudgeEvaluator::EvaluateHallucination(
    const InferenceRecord& record) {

    // Check cache
    if (config_.enable_cache) {
        std::string cache_key = GetCacheKey(record, "hallucination");
        std::lock_guard<std::mutex> lock(impl_->cache_mutex_);
        auto it = impl_->cache_.find(cache_key);
        if (it != impl_->cache_.end()) {
            auto age = system_clock::now() - it->second.second;
            if (age < config_.cache_ttl) {
                stats_.cache_hits++;
                return it->second.first;
            }
        }
    }

    auto start = steady_clock::now();

    // Build prompt
    std::string user_prompt = BuildHallucinationPrompt(record);

    // Call LLM
    auto response = CallLLM(hallucination_prompt_.system_prompt, user_prompt);
    if (!response.ok()) {
        stats_.error_count++;
        return response.status();
    }

    // Parse verdict
    JudgeVerdict verdict = ParseVerdict(*response);
    verdict.raw_response = *response;

    // Update latency stats
    auto elapsed = duration_cast<milliseconds>(steady_clock::now() - start);
    stats_.avg_latency_ms = (stats_.avg_latency_ms * stats_.total_evaluations +
                            elapsed.count()) / (stats_.total_evaluations + 1);

    // Cache result
    if (config_.enable_cache) {
        std::string cache_key = GetCacheKey(record, "hallucination");
        std::lock_guard<std::mutex> lock(impl_->cache_mutex_);
        impl_->cache_[cache_key] = {verdict, system_clock::now()};
    }

    return verdict;
}

absl::StatusOr<JudgeVerdict> LLMJudgeEvaluator::EvaluateAgainstReference(
    const InferenceRecord& record) {

    if (!record.expected_output) {
        return absl::InvalidArgumentError(
            "Expected output required for reference comparison");
    }

    std::string user_prompt = BuildReferencePrompt(record);
    auto response = CallLLM(quality_prompt_.system_prompt, user_prompt);
    if (!response.ok()) {
        return response.status();
    }

    return ParseVerdict(*response);
}

absl::StatusOr<JudgeVerdict> LLMJudgeEvaluator::ComparePairwise(
    const std::string& input,
    const std::string& output_a,
    const std::string& output_b,
    const std::string& criteria) {

    std::string system_prompt = R"(You are comparing two AI outputs to determine which is better.
Evaluate both outputs and decide which one is better based on accuracy, relevance, and quality.
Respond in JSON format with: winner ("A", "B", or "tie"), score_a, score_b, explanation)";

    std::string user_prompt = "Compare these two outputs:\n\n"
        "**User Input:**\n" + input + "\n\n"
        "**Output A:**\n" + output_a + "\n\n"
        "**Output B:**\n" + output_b;

    if (!criteria.empty()) {
        user_prompt += "\n\n**Evaluation Criteria:**\n" + criteria;
    }

    auto response = CallLLM(system_prompt, user_prompt);
    if (!response.ok()) {
        return response.status();
    }

    // Parse comparison result
    JudgeVerdict verdict;
    try {
        json j = json::parse(*response);
        std::string winner = j.value("winner", "tie");
        verdict.score = j.value("score_a", 0.5);
        verdict.explanation = j.value("explanation", "");

        if (winner == "A") {
            verdict.result = JudgeVerdict::Result::kPass;
        } else if (winner == "B") {
            verdict.result = JudgeVerdict::Result::kFail;
        } else {
            verdict.result = JudgeVerdict::Result::kUnsure;
        }
    } catch (...) {
        verdict.result = JudgeVerdict::Result::kError;
        verdict.explanation = "Failed to parse comparison result";
    }

    return verdict;
}

absl::StatusOr<JudgeVerdict> LLMJudgeEvaluator::EvaluateWithTemplate(
    const InferenceRecord& record,
    const JudgePromptTemplate& prompt_template) {

    // Replace placeholders in template
    std::string user_prompt = prompt_template.user_prompt_template;

    auto replace = [&](const std::string& placeholder, const std::string& value) {
        size_t pos;
        while ((pos = user_prompt.find(placeholder)) != std::string::npos) {
            user_prompt.replace(pos, placeholder.length(), value);
        }
    };

    replace("{input}", record.input);
    replace("{output}", record.output);

    std::string context;
    if (record.retrieved_contexts && !record.retrieved_contexts->empty()) {
        for (const auto& ctx : *record.retrieved_contexts) {
            context += "- " + ctx + "\n";
        }
    } else {
        context = "(No context provided)";
    }
    replace("{context}", context);

    if (record.expected_output) {
        replace("{expected}", *record.expected_output);
    } else {
        replace("{expected}", "(Not provided)");
    }

    auto response = CallLLM(prompt_template.system_prompt, user_prompt);
    if (!response.ok()) {
        return response.status();
    }

    return ParseVerdict(*response);
}

void LLMJudgeEvaluator::SetHallucinationPrompt(const JudgePromptTemplate& prompt) {
    hallucination_prompt_ = prompt;
}

void LLMJudgeEvaluator::SetQualityPrompt(const JudgePromptTemplate& prompt) {
    quality_prompt_ = prompt;
}

// =============================================================================
// Private Methods
// =============================================================================

absl::StatusOr<std::string> LLMJudgeEvaluator::CallLLM(
    const std::string& system_prompt,
    const std::string& user_prompt) {

#ifdef PYFLARE_HAS_HTTPLIB
    auto* client = impl_->GetClient();
    if (!client) {
        return absl::InternalError("Failed to create HTTP client");
    }

    // Build request body
    json request_body;
    request_body["model"] = config_.judge_model;
    request_body["max_tokens"] = config_.max_tokens;
    request_body["temperature"] = config_.temperature;
    request_body["messages"] = json::array({
        {{"role", "system"}, {"content", system_prompt}},
        {{"role", "user"}, {"content", user_prompt}}
    });

    // Add response format for JSON
    request_body["response_format"] = {{"type", "json_object"}};

    httplib::Headers headers = {
        {"Authorization", "Bearer " + config_.api_key},
        {"Content-Type", "application/json"}
    };

    // Make request with retries
    size_t retries = 0;
    while (retries <= config_.max_retries) {
        auto result = client->Post(
            "/v1/chat/completions",
            headers,
            request_body.dump(),
            "application/json");

        if (result && result->status == 200) {
            try {
                json response = json::parse(result->body);
                if (response.contains("choices") &&
                    !response["choices"].empty() &&
                    response["choices"][0].contains("message")) {
                    return response["choices"][0]["message"]["content"]
                        .get<std::string>();
                }
            } catch (const json::exception& e) {
                return absl::InternalError(
                    std::string("Failed to parse LLM response: ") + e.what());
            }
        }

        retries++;
        if (retries <= config_.max_retries) {
            std::this_thread::sleep_for(milliseconds(500 * retries));
        }
    }

    return absl::UnavailableError("LLM API request failed after retries");

#else
    // Stub implementation without HTTP library
    spdlog::warn("LLM Judge called but PYFLARE_HAS_HTTPLIB not defined");

    // Return a mock response for testing
    json mock_response;
    mock_response["verdict"] = "UNSURE";
    mock_response["score"] = 0.5;
    mock_response["explanation"] = "Mock evaluation - HTTP client not available";
    mock_response["has_hallucination"] = false;
    mock_response["has_factual_error"] = false;
    mock_response["has_contradiction"] = false;
    mock_response["has_unsupported_claim"] = false;

    return mock_response.dump();
#endif
}

JudgeVerdict LLMJudgeEvaluator::ParseVerdict(const std::string& response) {
    JudgeVerdict verdict;

    try {
        json j = json::parse(response);

        // Parse verdict string
        std::string verdict_str = j.value("verdict", "UNSURE");
        std::transform(verdict_str.begin(), verdict_str.end(),
                       verdict_str.begin(), ::toupper);

        if (verdict_str == "PASS") {
            verdict.result = JudgeVerdict::Result::kPass;
        } else if (verdict_str == "FAIL") {
            verdict.result = JudgeVerdict::Result::kFail;
        } else {
            verdict.result = JudgeVerdict::Result::kUnsure;
        }

        verdict.score = j.value("score", 0.5);
        verdict.explanation = j.value("explanation", "");

        // Parse specific flags
        verdict.has_hallucination = j.value("has_hallucination", false);
        verdict.has_factual_error = j.value("has_factual_error", false);
        verdict.has_contradiction = j.value("has_contradiction", false);
        verdict.has_unsupported_claim = j.value("has_unsupported_claim", false);

    } catch (const json::exception& e) {
        spdlog::warn("Failed to parse judge response as JSON: {}", e.what());

        // Try to extract verdict from plain text
        std::string lower_response = response;
        std::transform(lower_response.begin(), lower_response.end(),
                       lower_response.begin(), ::tolower);

        if (lower_response.find("pass") != std::string::npos) {
            verdict.result = JudgeVerdict::Result::kPass;
            verdict.score = 0.8;
        } else if (lower_response.find("fail") != std::string::npos) {
            verdict.result = JudgeVerdict::Result::kFail;
            verdict.score = 0.2;
        }

        verdict.explanation = response;
    }

    return verdict;
}

std::string LLMJudgeEvaluator::BuildHallucinationPrompt(
    const InferenceRecord& record) {

    std::string prompt = hallucination_prompt_.user_prompt_template;

    auto replace = [&](const std::string& placeholder, const std::string& value) {
        size_t pos;
        while ((pos = prompt.find(placeholder)) != std::string::npos) {
            prompt.replace(pos, placeholder.length(), value);
        }
    };

    replace("{input}", record.input);
    replace("{output}", record.output);

    std::string context;
    if (record.retrieved_contexts && !record.retrieved_contexts->empty()) {
        for (size_t i = 0; i < record.retrieved_contexts->size(); ++i) {
            context += "[" + std::to_string(i + 1) + "] " +
                       (*record.retrieved_contexts)[i] + "\n\n";
        }
    } else {
        context = "(No context/sources provided - evaluate based on general knowledge)";
    }
    replace("{context}", context);

    return prompt;
}

std::string LLMJudgeEvaluator::BuildReferencePrompt(const InferenceRecord& record) {
    std::string prompt = quality_prompt_.user_prompt_template;

    auto replace = [&](const std::string& placeholder, const std::string& value) {
        size_t pos;
        while ((pos = prompt.find(placeholder)) != std::string::npos) {
            prompt.replace(pos, placeholder.length(), value);
        }
    };

    replace("{input}", record.input);
    replace("{output}", record.output);
    replace("{expected}", record.expected_output.value_or("(Not provided)"));

    return prompt;
}

std::string LLMJudgeEvaluator::GetCacheKey(const InferenceRecord& record,
                                            const std::string& eval_type) {
    // Simple hash-based key
    size_t hash = std::hash<std::string>{}(
        record.input + "|" + record.output + "|" + eval_type);
    return std::to_string(hash);
}

}  // namespace pyflare::eval
