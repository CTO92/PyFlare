#pragma once

/// @file evaluator.h
/// @brief Base interface for ML evaluators in PyFlare

#include <memory>
#include <optional>
#include <string>
#include <unordered_map>
#include <vector>

#include <absl/status/statusor.h>

namespace pyflare::eval {

/// @brief Result of an evaluation
struct EvalResult {
    std::string evaluator_type;
    double score;              ///< 0.0 - 1.0 (higher = better)
    std::string verdict;       ///< "pass", "fail", "warn"
    std::string explanation;
    std::unordered_map<std::string, std::string> metadata;
};

/// @brief An inference record to evaluate
struct InferenceRecord {
    std::string trace_id;
    std::string model_id;
    std::string input;
    std::string output;
    std::optional<std::string> expected_output;
    std::optional<std::vector<std::string>> retrieved_contexts;  ///< For RAG
    std::unordered_map<std::string, std::string> attributes;
};

/// @brief Abstract base class for evaluators
class Evaluator {
public:
    virtual ~Evaluator() = default;

    /// @brief Evaluate a single inference
    virtual absl::StatusOr<EvalResult> Evaluate(const InferenceRecord& record) = 0;

    /// @brief Batch evaluation
    virtual absl::StatusOr<std::vector<EvalResult>> EvaluateBatch(
        const std::vector<InferenceRecord>& records) = 0;

    /// @brief Get the evaluator type name
    virtual std::string Type() const = 0;
};

/// @brief Factory for creating evaluators
class EvaluatorFactory {
public:
    /// @brief Create a hallucination evaluator
    static std::unique_ptr<Evaluator> CreateHallucinationEvaluator();

    /// @brief Create a RAG quality evaluator
    static std::unique_ptr<Evaluator> CreateRAGEvaluator();

    /// @brief Create a toxicity evaluator
    static std::unique_ptr<Evaluator> CreateToxicityEvaluator();
};

}  // namespace pyflare::eval
