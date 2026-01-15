# PyFlare Extension Guide

This guide explains how to extend PyFlare with new features, components, and integrations. Whether you're adding a new sampling strategy, exporter, or SDK integration, this document provides patterns and examples.

## Table of Contents

- [Adding a New Sampler Strategy](#adding-a-new-sampler-strategy)
- [Adding a New Exporter](#adding-a-new-exporter)
- [Adding SDK Integrations](#adding-sdk-integrations)
- [Adding New Span Attributes](#adding-new-span-attributes)
- [Adding New Configuration Options](#adding-new-configuration-options)
- [Adding Cost Calculator Support](#adding-cost-calculator-support)
- [Testing Your Extensions](#testing-your-extensions)

---

## Adding a New Sampler Strategy

Let's add a new sampling strategy that samples based on span attributes (e.g., sample all error spans).

### Step 1: Define the Strategy Enum

Add to `src/collector/types.h`:

```cpp
enum class SamplerStrategy {
    ALWAYS_ON,
    ALWAYS_OFF,
    PROBABILISTIC,
    RATE_LIMITING,
    PARENT_BASED,
    COMPOSITE,
    ATTRIBUTE_BASED,  // NEW
};
```

### Step 2: Implement the Strategy

Add to `src/collector/sampler.cpp`:

```cpp
class AttributeBasedSampler {
public:
    struct Rule {
        std::string attribute_key;
        std::string attribute_value;  // Empty for "exists" check
        double sample_rate;
    };

    AttributeBasedSampler(std::vector<Rule> rules, double default_rate)
        : rules_(std::move(rules))
        , default_rate_(default_rate) {}

    bool ShouldSample(const Span& span) const {
        // Check rules in order
        for (const auto& rule : rules_) {
            auto it = span.attributes.find(rule.attribute_key);
            if (it != span.attributes.end()) {
                // Check value if specified
                if (rule.attribute_value.empty() ||
                    std::get<std::string>(it->second) == rule.attribute_value) {
                    return ProbabilisticSample(span.trace_id, rule.sample_rate);
                }
            }
        }

        // Default sampling
        return ProbabilisticSample(span.trace_id, default_rate_);
    }

private:
    std::vector<Rule> rules_;
    double default_rate_;

    bool ProbabilisticSample(const std::string& trace_id, double rate) const {
        uint64_t hash = FNV1aHash(trace_id);
        uint64_t threshold = static_cast<uint64_t>(rate * UINT64_MAX);
        return hash <= threshold;
    }
};
```

### Step 3: Update the Factory

In `CreateSampler()` function:

```cpp
std::unique_ptr<Sampler> CreateSampler(const SamplerConfig& config) {
    switch (config.strategy) {
        // ... existing cases ...

        case SamplerStrategy::ATTRIBUTE_BASED: {
            std::vector<AttributeBasedSampler::Rule> rules;
            for (const auto& rule_config : config.attribute_rules) {
                rules.push_back({
                    rule_config.attribute_key,
                    rule_config.attribute_value,
                    rule_config.sample_rate,
                });
            }
            return std::make_unique<AttributeBasedSamplerWrapper>(
                std::move(rules), config.probability);
        }
    }
}
```

### Step 4: Add Configuration Support

Update `config/collector.yaml`:

```yaml
sampler:
  strategy: attribute_based
  probability: 0.1  # Default rate

  # Attribute-based rules
  attribute_rules:
    # Sample 100% of error spans
    - attribute_key: "otel.status_code"
      attribute_value: "ERROR"
      sample_rate: 1.0

    # Sample 50% of LLM spans
    - attribute_key: "pyflare.inference.type"
      attribute_value: "llm"
      sample_rate: 0.5
```

### Step 5: Write Tests

Add to `tests/unit/collector/sampler_test.cpp`:

```cpp
TEST(AttributeBasedSamplerTest, SamplesErrorSpans) {
    SamplerConfig config;
    config.strategy = SamplerStrategy::ATTRIBUTE_BASED;
    config.probability = 0.0;  // Default: drop
    config.attribute_rules = {
        {"otel.status_code", "ERROR", 1.0},
    };

    auto sampler = CreateSampler(config);

    // Error span should be sampled
    Span error_span;
    error_span.trace_id = "test-trace-1";
    error_span.attributes["otel.status_code"] = std::string("ERROR");
    EXPECT_TRUE(sampler->ShouldSample(error_span));

    // Normal span should be dropped (default 0%)
    Span normal_span;
    normal_span.trace_id = "test-trace-2";
    EXPECT_FALSE(sampler->ShouldSample(normal_span));
}
```

---

## Adding a New Exporter

Let's add a file-based exporter for local development.

### Step 1: Create the Header

Create `src/collector/include/pyflare/file_exporter.h`:

```cpp
#pragma once

#include "pyflare/types.h"
#include <absl/status/status.h>
#include <memory>
#include <string>
#include <vector>

namespace pyflare {

struct FileExporterConfig {
    std::string output_path = "./spans.jsonl";
    bool append = true;
    size_t max_file_size_mb = 100;
    bool rotate = true;
};

class FileExporter {
public:
    explicit FileExporter(const FileExporterConfig& config);
    ~FileExporter();

    absl::Status Open();
    void Close();
    absl::Status Export(const std::vector<Span>& spans);

    uint64_t GetSpansWritten() const;

private:
    class Impl;
    std::unique_ptr<Impl> impl_;
};

}  // namespace pyflare
```

### Step 2: Implement the Exporter

Create `src/collector/file_exporter.cpp`:

```cpp
#include "pyflare/file_exporter.h"
#include <nlohmann/json.hpp>
#include <spdlog/spdlog.h>
#include <fstream>
#include <mutex>

namespace pyflare {

class FileExporter::Impl {
public:
    explicit Impl(const FileExporterConfig& config)
        : config_(config) {}

    ~Impl() {
        Close();
    }

    absl::Status Open() {
        std::lock_guard<std::mutex> lock(mutex_);

        auto mode = config_.append ? std::ios::app : std::ios::trunc;
        file_.open(config_.output_path, std::ios::out | mode);

        if (!file_.is_open()) {
            return absl::InternalError(
                absl::StrCat("Failed to open file: ", config_.output_path));
        }

        spdlog::info("FileExporter opened: {}", config_.output_path);
        return absl::OkStatus();
    }

    void Close() {
        std::lock_guard<std::mutex> lock(mutex_);
        if (file_.is_open()) {
            file_.close();
            spdlog::info("FileExporter closed");
        }
    }

    absl::Status Export(const std::vector<Span>& spans) {
        std::lock_guard<std::mutex> lock(mutex_);

        if (!file_.is_open()) {
            return absl::FailedPreconditionError("File not open");
        }

        // Check file rotation
        if (config_.rotate && NeedsRotation()) {
            RotateFile();
        }

        for (const auto& span : spans) {
            nlohmann::json json = SpanToJson(span);
            file_ << json.dump() << "\n";
            spans_written_++;
        }

        file_.flush();
        return absl::OkStatus();
    }

    uint64_t GetSpansWritten() const {
        return spans_written_.load();
    }

private:
    FileExporterConfig config_;
    std::ofstream file_;
    std::mutex mutex_;
    std::atomic<uint64_t> spans_written_{0};

    nlohmann::json SpanToJson(const Span& span) {
        nlohmann::json json = {
            {"trace_id", span.trace_id},
            {"span_id", span.span_id},
            {"parent_span_id", span.parent_span_id},
            {"name", span.name},
            {"kind", static_cast<int>(span.kind)},
            {"start_time_unix_nano", span.start_time_unix_nano},
            {"end_time_unix_nano", span.end_time_unix_nano},
            {"service_name", span.service_name},
        };

        // Add attributes
        nlohmann::json attrs;
        for (const auto& [k, v] : span.attributes) {
            std::visit([&attrs, &k](auto&& val) {
                attrs[k] = val;
            }, v);
        }
        json["attributes"] = attrs;

        return json;
    }

    bool NeedsRotation() {
        // Check file size
        auto pos = file_.tellp();
        size_t size_mb = static_cast<size_t>(pos) / (1024 * 1024);
        return size_mb >= config_.max_file_size_mb;
    }

    void RotateFile() {
        file_.close();

        // Rename with timestamp
        auto now = std::chrono::system_clock::now();
        auto timestamp = std::chrono::duration_cast<std::chrono::seconds>(
            now.time_since_epoch()).count();

        std::string rotated_path = config_.output_path + "." + std::to_string(timestamp);
        std::rename(config_.output_path.c_str(), rotated_path.c_str());

        file_.open(config_.output_path, std::ios::out | std::ios::trunc);
        spdlog::info("Rotated file to: {}", rotated_path);
    }
};

// Forwarding implementations
FileExporter::FileExporter(const FileExporterConfig& config)
    : impl_(std::make_unique<Impl>(config)) {}
FileExporter::~FileExporter() = default;
absl::Status FileExporter::Open() { return impl_->Open(); }
void FileExporter::Close() { impl_->Close(); }
absl::Status FileExporter::Export(const std::vector<Span>& spans) {
    return impl_->Export(spans);
}
uint64_t FileExporter::GetSpansWritten() const {
    return impl_->GetSpansWritten();
}

}  // namespace pyflare
```

### Step 3: Integrate with Collector

Update `src/collector/collector.cpp`:

```cpp
void Collector::OnBatchReady(std::vector<Span>&& batch) {
    // Export to Kafka
    if (kafka_exporter_) {
        auto status = kafka_exporter_->Export(batch);
        if (!status.ok()) {
            spdlog::warn("Kafka export failed: {}", status.message());
        }
    }

    // Export to file (for development/debugging)
    if (file_exporter_) {
        auto status = file_exporter_->Export(batch);
        if (!status.ok()) {
            spdlog::warn("File export failed: {}", status.message());
        }
    }
}
```

---

## Adding SDK Integrations

Let's add auto-instrumentation for a hypothetical ML framework.

### Step 1: Create the Integration Module

Create `sdk/python/pyflare/integrations/myframework.py`:

```python
"""Auto-instrumentation for MyFramework."""

from typing import Any, Callable, Optional
import functools

from pyflare import PyFlare, trace
from pyflare.types import InferenceType, TokenUsage
from pyflare.cost import calculate_cost


class MyFrameworkInstrumentation:
    """Auto-instrumentation for MyFramework library."""

    def __init__(self, capture_prompts: bool = False):
        self.capture_prompts = capture_prompts
        self._original_methods: dict[str, Callable] = {}

    def instrument(self) -> None:
        """Enable auto-instrumentation."""
        try:
            import myframework
        except ImportError:
            raise ImportError(
                "myframework is not installed. Install with: pip install myframework"
            )

        # Patch the main inference method
        self._patch_method(
            myframework.Model,
            "predict",
            self._wrap_predict,
        )

        # Patch batch inference
        self._patch_method(
            myframework.Model,
            "predict_batch",
            self._wrap_predict_batch,
        )

    def uninstrument(self) -> None:
        """Disable auto-instrumentation."""
        for key, original in self._original_methods.items():
            module_name, class_name, method_name = key.rsplit(".", 2)
            import importlib
            module = importlib.import_module(module_name)
            cls = getattr(module, class_name)
            setattr(cls, method_name, original)
        self._original_methods.clear()

    def _patch_method(
        self,
        cls: type,
        method_name: str,
        wrapper_factory: Callable,
    ) -> None:
        """Patch a method with instrumentation."""
        original = getattr(cls, method_name)
        key = f"{cls.__module__}.{cls.__name__}.{method_name}"
        self._original_methods[key] = original

        wrapped = wrapper_factory(original)
        setattr(cls, method_name, wrapped)

    def _wrap_predict(self, original: Callable) -> Callable:
        """Wrap the predict method."""
        capture_prompts = self.capture_prompts

        @functools.wraps(original)
        def wrapper(self_model: Any, input_data: Any, **kwargs: Any) -> Any:
            pyflare = PyFlare.get_instance()
            if not pyflare or not pyflare.enabled:
                return original(self_model, input_data, **kwargs)

            # Get model info
            model_id = getattr(self_model, "model_id", "unknown")
            model_version = getattr(self_model, "version", "")

            with pyflare.span(
                f"myframework.predict",
                attributes={
                    "pyflare.model.id": model_id,
                    "pyflare.model.version": model_version,
                    "pyflare.model.provider": "myframework",
                    "pyflare.inference.type": InferenceType.CUSTOM.value,
                },
            ) as span:
                # Capture input if enabled
                if capture_prompts:
                    span.set_attribute(
                        "pyflare.input.preview",
                        _truncate(str(input_data), 1000),
                    )

                try:
                    result = original(self_model, input_data, **kwargs)

                    # Capture output if enabled
                    if capture_prompts:
                        span.set_attribute(
                            "pyflare.output.preview",
                            _truncate(str(result), 1000),
                        )

                    # If result has usage info, capture it
                    if hasattr(result, "usage"):
                        span.set_attribute(
                            "pyflare.tokens.total",
                            result.usage.total_tokens,
                        )

                    return result

                except Exception as e:
                    span.set_status(Status(StatusCode.ERROR, str(e)))
                    span.record_exception(e)
                    raise

        return wrapper

    def _wrap_predict_batch(self, original: Callable) -> Callable:
        """Wrap the batch predict method."""
        @functools.wraps(original)
        def wrapper(self_model: Any, inputs: list, **kwargs: Any) -> list:
            pyflare = PyFlare.get_instance()
            if not pyflare or not pyflare.enabled:
                return original(self_model, inputs, **kwargs)

            model_id = getattr(self_model, "model_id", "unknown")

            with pyflare.span(
                "myframework.predict_batch",
                attributes={
                    "pyflare.model.id": model_id,
                    "pyflare.model.provider": "myframework",
                    "pyflare.inference.type": InferenceType.CUSTOM.value,
                    "pyflare.batch.size": len(inputs),
                },
            ) as span:
                result = original(self_model, inputs, **kwargs)
                return result

        return wrapper


def _truncate(text: str, max_length: int) -> str:
    """Truncate text to max length."""
    if len(text) <= max_length:
        return text
    return text[:max_length - 3] + "..."
```

### Step 2: Export from integrations package

Update `sdk/python/pyflare/integrations/__init__.py`:

```python
from pyflare.integrations.openai import OpenAIInstrumentation
from pyflare.integrations.anthropic import AnthropicInstrumentation
from pyflare.integrations.myframework import MyFrameworkInstrumentation  # NEW

__all__ = [
    "OpenAIInstrumentation",
    "AnthropicInstrumentation",
    "MyFrameworkInstrumentation",
]
```

### Step 3: Write Tests

Create `sdk/python/tests/test_myframework_integration.py`:

```python
"""Tests for MyFramework integration."""

import pytest
from unittest.mock import MagicMock, patch

from pyflare import PyFlare
from pyflare.integrations import MyFrameworkInstrumentation


class TestMyFrameworkInstrumentation:
    def test_instrumentation_patches_predict(self) -> None:
        """Test that predict method is patched."""
        pyflare = PyFlare(service_name="test", enabled=False)

        # Mock the myframework module
        mock_model = MagicMock()
        mock_model.model_id = "test-model"
        mock_model.predict = MagicMock(return_value="result")

        with patch.dict("sys.modules", {"myframework": MagicMock()}):
            import sys
            sys.modules["myframework"].Model = type(mock_model)

            instrumentation = MyFrameworkInstrumentation()
            instrumentation.instrument()

            # Verify patching occurred
            assert "myframework.Model.predict" in instrumentation._original_methods

    def test_uninstrument_restores_original(self) -> None:
        """Test that uninstrument restores original methods."""
        pyflare = PyFlare(service_name="test", enabled=False)

        with patch.dict("sys.modules", {"myframework": MagicMock()}):
            instrumentation = MyFrameworkInstrumentation()
            instrumentation.instrument()
            instrumentation.uninstrument()

            assert len(instrumentation._original_methods) == 0
```

---

## Adding New Span Attributes

To add new semantic conventions for AI/ML workloads:

### Step 1: Define Constants

Create or update `sdk/python/pyflare/semantic_conventions.py`:

```python
"""PyFlare semantic conventions for AI/ML observability."""

# Model attributes
MODEL_ID = "pyflare.model.id"
MODEL_VERSION = "pyflare.model.version"
MODEL_PROVIDER = "pyflare.model.provider"
MODEL_TEMPERATURE = "pyflare.model.temperature"
MODEL_MAX_TOKENS = "pyflare.model.max_tokens"

# Inference attributes
INFERENCE_TYPE = "pyflare.inference.type"
INFERENCE_LATENCY_MS = "pyflare.inference.latency_ms"

# Token attributes
TOKENS_INPUT = "pyflare.tokens.input"
TOKENS_OUTPUT = "pyflare.tokens.output"
TOKENS_TOTAL = "pyflare.tokens.total"

# Cost attributes
COST_MICROS = "pyflare.cost.micros"
COST_CURRENCY = "pyflare.cost.currency"

# Quality attributes (NEW)
QUALITY_SCORE = "pyflare.quality.score"
QUALITY_FEEDBACK = "pyflare.quality.feedback"
QUALITY_LATENCY_PERCENTILE = "pyflare.quality.latency_percentile"

# RAG attributes (NEW)
RAG_QUERY = "pyflare.rag.query"
RAG_RETRIEVED_DOCS = "pyflare.rag.retrieved_docs"
RAG_RELEVANCE_SCORES = "pyflare.rag.relevance_scores"
RAG_CONTEXT_TOKENS = "pyflare.rag.context_tokens"

# Embedding attributes (NEW)
EMBEDDING_DIMENSIONS = "pyflare.embedding.dimensions"
EMBEDDING_MODEL = "pyflare.embedding.model"
EMBEDDING_INPUT_TYPE = "pyflare.embedding.input_type"
```

### Step 2: Update Types

Update `sdk/python/pyflare/types.py`:

```python
@dataclass
class SpanAttributes:
    # ... existing fields ...

    # Quality attributes
    quality_score: Optional[float] = None
    quality_feedback: Optional[str] = None

    # RAG attributes
    rag_retrieved_docs: Optional[int] = None
    rag_relevance_scores: Optional[list[float]] = None

    def to_otel_attributes(self) -> dict[str, Any]:
        attrs = {}

        # ... existing conversions ...

        # Quality attributes
        if self.quality_score is not None:
            attrs["pyflare.quality.score"] = self.quality_score
        if self.quality_feedback is not None:
            attrs["pyflare.quality.feedback"] = self.quality_feedback

        # RAG attributes
        if self.rag_retrieved_docs is not None:
            attrs["pyflare.rag.retrieved_docs"] = self.rag_retrieved_docs

        return attrs
```

### Step 3: Update C++ Types

Update `src/collector/include/pyflare/types.h`:

```cpp
// Add to AttributeValue variant
using AttributeValue = std::variant<
    std::string,
    int64_t,
    double,
    bool,
    std::vector<std::string>,
    std::vector<int64_t>,
    std::vector<double>  // NEW: for relevance scores
>;
```

---

## Adding New Configuration Options

### Step 1: Update Config Struct

In `src/collector/include/pyflare/types.h`:

```cpp
struct NewFeatureConfig {
    bool enabled = false;
    std::string option1 = "default";
    int option2 = 100;
};

struct CollectorConfig {
    // ... existing fields ...
    NewFeatureConfig new_feature;
};
```

### Step 2: Add YAML Parsing

In `src/collector/collector.cpp`:

```cpp
void LoadNewFeatureConfig(const YAML::Node& node, NewFeatureConfig& config) {
    if (!node) return;

    if (node["enabled"]) {
        config.enabled = node["enabled"].as<bool>();
    }
    if (node["option1"]) {
        config.option1 = node["option1"].as<std::string>();
    }
    if (node["option2"]) {
        config.option2 = node["option2"].as<int>();
    }
}

// In LoadConfig()
LoadNewFeatureConfig(root["new_feature"], config.new_feature);
```

### Step 3: Add Environment Variable Override

```cpp
void ApplyEnvironmentOverrides(CollectorConfig& config) {
    // ... existing overrides ...

    if (const char* env = std::getenv("PYFLARE_NEW_FEATURE_ENABLED")) {
        config.new_feature.enabled = (std::string(env) == "true");
    }
}
```

### Step 4: Document Configuration

Update `config/collector.yaml`:

```yaml
# New Feature Configuration
new_feature:
  enabled: false
  option1: "default"
  option2: 100
```

---

## Adding Cost Calculator Support

To add pricing for a new model provider:

### Step 1: Add Provider Enum

Update `sdk/python/pyflare/cost.py`:

```python
class ModelProvider(Enum):
    OPENAI = "openai"
    ANTHROPIC = "anthropic"
    GOOGLE = "google"
    COHERE = "cohere"
    MISTRAL = "mistral"  # NEW
    CUSTOM = "custom"
```

### Step 2: Add Pricing Data

```python
MODEL_PRICING: dict[str, ModelPricing] = {
    # ... existing models ...

    # Mistral AI
    "mistral-large-latest": ModelPricing(
        "mistral-large-latest",
        ModelProvider.MISTRAL,
        input_price_per_million=2.00,
        output_price_per_million=6.00,
        context_window=128000,
    ),
    "mistral-medium-latest": ModelPricing(
        "mistral-medium-latest",
        ModelProvider.MISTRAL,
        input_price_per_million=0.70,
        output_price_per_million=2.10,
        context_window=32000,
    ),
    "mistral-small-latest": ModelPricing(
        "mistral-small-latest",
        ModelProvider.MISTRAL,
        input_price_per_million=0.20,
        output_price_per_million=0.60,
        context_window=32000,
    ),
}
```

### Step 3: Add Integration (Optional)

Create `sdk/python/pyflare/integrations/mistral.py` following the pattern in the SDK Integrations section.

---

## Testing Your Extensions

### C++ Unit Tests

Use GoogleTest for C++ components:

```cpp
// tests/unit/collector/new_feature_test.cpp
#include <gtest/gtest.h>
#include "pyflare/new_feature.h"

class NewFeatureTest : public ::testing::Test {
protected:
    void SetUp() override {
        // Setup code
    }
};

TEST_F(NewFeatureTest, BasicFunctionality) {
    NewFeatureConfig config;
    config.enabled = true;

    NewFeature feature(config);
    EXPECT_TRUE(feature.IsEnabled());
}

TEST_F(NewFeatureTest, DisabledByDefault) {
    NewFeatureConfig config;
    NewFeature feature(config);
    EXPECT_FALSE(feature.IsEnabled());
}
```

### Python Unit Tests

Use pytest for Python components:

```python
# sdk/python/tests/test_new_feature.py
import pytest
from pyflare import PyFlare


class TestNewFeature:
    def test_basic_functionality(self) -> None:
        pyflare = PyFlare(service_name="test", enabled=False)
        # Test assertions

    def test_with_custom_config(self) -> None:
        pyflare = PyFlare(
            service_name="test",
            enabled=False,
            new_option="value",
        )
        assert pyflare.new_option == "value"

    @pytest.mark.asyncio
    async def test_async_functionality(self) -> None:
        # Async test
        pass
```

### Integration Tests

Create integration tests for end-to-end validation:

```python
# tests/integration/test_collector_integration.py
import pytest
import requests
import json


@pytest.fixture
def collector_url():
    return "http://localhost:4318"


def test_http_endpoint_accepts_spans(collector_url: str) -> None:
    """Test that collector accepts spans via HTTP."""
    spans = {
        "resourceSpans": [{
            "resource": {"attributes": [{"key": "service.name", "value": {"stringValue": "test"}}]},
            "scopeSpans": [{
                "spans": [{
                    "traceId": "0123456789abcdef0123456789abcdef",
                    "spanId": "0123456789abcdef",
                    "name": "test-span",
                    "startTimeUnixNano": 1000000000,
                    "endTimeUnixNano": 2000000000,
                }]
            }]
        }]
    }

    response = requests.post(
        f"{collector_url}/v1/traces",
        json=spans,
        headers={"Content-Type": "application/json"},
    )

    assert response.status_code == 200
```

### Running Tests

```bash
# C++ tests
cd build
ctest --output-on-failure

# Python tests
cd sdk/python
pytest -v tests/

# Integration tests
cd tests/integration
pytest -v --collector-url=http://localhost:4318
```

---

## Best Practices

1. **Follow Existing Patterns**: Look at existing components for implementation patterns
2. **Use PIMPL**: Keep public headers clean with PIMPL pattern
3. **Thread Safety**: Use appropriate synchronization for concurrent access
4. **Error Handling**: Use `absl::Status` for error propagation
5. **Logging**: Use `spdlog` with appropriate log levels
6. **Testing**: Write unit tests for all new functionality
7. **Documentation**: Update configuration and API documentation
8. **Backward Compatibility**: Avoid breaking changes to public APIs

---

## Next Steps

- [Building](./building.md) - Build and test your changes
- [Architecture](./architecture.md) - Understand the overall system
- [Component Guide](./components.md) - Deep dive into existing components
