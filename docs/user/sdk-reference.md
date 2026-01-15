# PyFlare SDK Reference

Complete API reference for the PyFlare Python SDK.

## Table of Contents

- [Core Classes](#core-classes)
- [Decorators](#decorators)
- [Context Managers](#context-managers)
- [Cost Calculation](#cost-calculation)
- [Integrations](#integrations)
- [Types](#types)

---

## Core Classes

### PyFlare

The main SDK class for AI/ML observability.

```python
from pyflare import PyFlare

pyflare = PyFlare(
    service_name: str,                    # Required: Name of your service
    endpoint: str = "http://localhost:4317",
    environment: str = "development",
    version: str = "",
    headers: dict[str, str] | None = None,
    enabled: bool = True,
    sample_rate: float = 1.0,
    use_http: bool = False,
    batch_export: bool = True,
    debug: bool = False,
    resource_attributes: dict[str, str] | None = None,
)
```

#### Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `service_name` | `str` | Required | Name of your service (appears in traces) |
| `endpoint` | `str` | `"http://localhost:4317"` | Collector endpoint |
| `environment` | `str` | `"development"` | Deployment environment |
| `version` | `str` | `""` | Service version |
| `headers` | `dict` | `None` | Headers for OTLP exporter (auth tokens, etc.) |
| `enabled` | `bool` | `True` | Whether tracing is enabled |
| `sample_rate` | `float` | `1.0` | Sampling rate (0.0-1.0) |
| `use_http` | `bool` | `False` | Use HTTP instead of gRPC |
| `batch_export` | `bool` | `True` | Batch spans before export |
| `debug` | `bool` | `False` | Print spans to console |
| `resource_attributes` | `dict` | `None` | Custom resource attributes |

#### Methods

##### `span(name, kind=SpanKind.INTERNAL, attributes=None)`

Create a span using a context manager.

```python
with pyflare.span("process-data", attributes={"items": 100}) as span:
    result = process(data)
    span.set_attribute("result.count", len(result))
```

##### `llm_span(name, model_id, provider="")`

Create a span specifically for LLM calls.

```python
with pyflare.llm_span("chat", model_id="gpt-4o", provider="openai") as span:
    response = client.chat.completions.create(...)
    span.set_attribute("pyflare.tokens.total", response.usage.total_tokens)
```

##### `trace(name=None, model_id=None, inference_type=InferenceType.CUSTOM, ...)`

Decorator for tracing functions. See [Decorators](#decorators).

##### `set_user(user_id, attributes=None)`

Set the current user for cost attribution.

```python
pyflare.set_user("user-123", {"plan": "enterprise"})
```

##### `set_feature(feature_id)`

Set the current feature for cost attribution.

```python
pyflare.set_feature("chat-assistant")
```

##### `set_session(session_id)`

Set the current session ID.

```python
pyflare.set_session("session-abc-123")
```

##### `flush(timeout_millis=30000)`

Flush pending spans to the collector.

```python
success = pyflare.flush(timeout_millis=5000)
```

##### `shutdown()`

Shutdown the SDK and flush remaining spans.

```python
pyflare.shutdown()
```

##### `register_model(model_info)`

Register a model for tracking.

```python
from pyflare import ModelInfo, InferenceType

pyflare.register_model(ModelInfo(
    model_id="my-custom-model",
    model_version="1.0.0",
    provider="custom",
    inference_type=InferenceType.CLASSIFICATION,
))
```

### Convenience Functions

#### `init(service_name, endpoint="http://localhost:4317", **kwargs)`

Quick initialization function.

```python
import pyflare

pyflare.init("my-service", environment="production")
```

#### `get_pyflare()`

Get the global PyFlare instance.

```python
instance = pyflare.get_pyflare()
```

---

## Decorators

### @trace

General-purpose tracing decorator.

```python
from pyflare import trace, InferenceType

@trace(
    name: str | None = None,              # Span name (defaults to function name)
    model_id: str | None = None,          # Model identifier
    inference_type: InferenceType = InferenceType.CUSTOM,
    capture_input: bool = True,           # Capture function input
    capture_output: bool = True,          # Capture function output
    attributes: dict | None = None,       # Additional attributes
)
def my_function(x):
    return x * 2
```

**Example:**

```python
@trace(
    name="classify",
    model_id="bert-classifier",
    inference_type=InferenceType.CLASSIFICATION,
    attributes={"version": "1.0"}
)
def classify(text: str) -> str:
    return model.predict(text)
```

### @trace_llm

Specialized decorator for LLM inference. Automatically captures token usage.

```python
from pyflare import trace_llm

@trace_llm(
    name: str | None = None,
    model_id: str | None = None,
    provider: str | None = None,
    capture_input: bool = True,
    capture_output: bool = True,
)
def llm_function():
    ...
```

**Example:**

```python
@trace_llm(model_id="gpt-4o", provider="openai")
def generate_response(prompt: str) -> str:
    response = openai.chat.completions.create(
        model="gpt-4o",
        messages=[{"role": "user", "content": prompt}]
    )
    return response.choices[0].message.content
```

### @trace_embedding

Specialized decorator for embedding generation.

```python
from pyflare import trace_embedding

@trace_embedding(
    name: str | None = None,
    model_id: str | None = None,
    provider: str | None = None,
    capture_input: bool = True,
)
def embedding_function():
    ...
```

**Example:**

```python
@trace_embedding(model_id="text-embedding-3-small", provider="openai")
def get_embedding(text: str) -> list[float]:
    response = openai.embeddings.create(
        model="text-embedding-3-small",
        input=text
    )
    return response.data[0].embedding
```

---

## Context Managers

### Module-Level Functions

These work on the global PyFlare instance:

```python
import pyflare

# General span
with pyflare.span("operation-name") as span:
    span.set_attribute("key", "value")
    do_work()

# LLM-specific span
with pyflare.llm_span("chat", model_id="gpt-4o", provider="openai") as span:
    response = call_llm()
    span.set_attribute("pyflare.tokens.total", response.usage.total_tokens)
```

### Adding Attributes to Spans

```python
with pyflare.span("process") as span:
    # String attributes
    span.set_attribute("user.id", "user-123")

    # Numeric attributes
    span.set_attribute("items.count", 42)
    span.set_attribute("latency.ms", 123.45)

    # Boolean attributes
    span.set_attribute("cache.hit", True)
```

### Recording Events

```python
with pyflare.span("multi-step-process") as span:
    span.add_event("step-1-complete", {"items_processed": 100})

    do_more_work()

    span.add_event("step-2-complete", {"items_processed": 200})
```

### Handling Errors

Exceptions are automatically captured:

```python
try:
    with pyflare.span("risky-operation") as span:
        result = might_fail()
except Exception as e:
    # Span is automatically marked as error
    # Exception is recorded
    raise
```

---

## Cost Calculation

### CostCalculator

Calculate costs for model inference.

```python
from pyflare import CostCalculator, TokenUsage

calculator = CostCalculator()

result = calculator.calculate(
    model_id="gpt-4o",
    token_usage=TokenUsage(
        input_tokens=1000,
        output_tokens=500,
    )
)

print(f"Cost: ${result.total_dollars:.4f}")
print(f"Input cost: ${result.input_dollars:.6f}")
print(f"Output cost: ${result.output_dollars:.6f}")
```

### Convenience Function

```python
from pyflare import calculate_cost

result = calculate_cost(
    model_id="claude-3-opus",
    input_tokens=1000,
    output_tokens=500,
)
```

### Custom Pricing

```python
from pyflare import CostCalculator, ModelPricing, ModelProvider

calculator = CostCalculator()
calculator.add_pricing(ModelPricing(
    model_id="my-custom-model",
    provider=ModelProvider.CUSTOM,
    input_price_per_million=1.00,   # $1 per 1M input tokens
    output_price_per_million=2.00,  # $2 per 1M output tokens
))
```

### Supported Models

PyFlare includes pricing for:

**OpenAI:**
- gpt-4o, gpt-4o-mini
- gpt-4-turbo, gpt-4
- gpt-3.5-turbo
- text-embedding-3-small, text-embedding-3-large

**Anthropic:**
- claude-3-5-sonnet, claude-3-opus
- claude-3-sonnet, claude-3-haiku

**Google:**
- gemini-1.5-pro, gemini-1.5-flash

**Cohere:**
- command-r-plus, command-r

### CostResult

```python
@dataclass
class CostResult:
    model_id: str
    token_usage: TokenUsage
    input_cost_micros: int      # Cost in micro-dollars (1/1,000,000)
    output_cost_micros: int
    estimated: bool             # True if model pricing unknown

    @property
    def total_micros(self) -> int: ...
    @property
    def total_dollars(self) -> float: ...
    @property
    def input_dollars(self) -> float: ...
    @property
    def output_dollars(self) -> float: ...
```

---

## Integrations

### OpenAI Integration

Auto-instrument all OpenAI calls:

```python
from pyflare.integrations import OpenAIInstrumentation

# Enable instrumentation
OpenAIInstrumentation().instrument()

# All calls are now traced
response = openai.chat.completions.create(
    model="gpt-4o",
    messages=[{"role": "user", "content": "Hello!"}]
)
```

To disable:

```python
instrumentation = OpenAIInstrumentation()
instrumentation.instrument()

# Later...
instrumentation.uninstrument()
```

### Anthropic Integration

Auto-instrument all Anthropic calls:

```python
from pyflare.integrations import AnthropicInstrumentation

AnthropicInstrumentation().instrument()

# All calls are now traced
response = anthropic.messages.create(
    model="claude-3-opus",
    messages=[{"role": "user", "content": "Hello!"}]
)
```

Configuration options:

```python
AnthropicInstrumentation(
    capture_input=True,   # Capture input prompts
    capture_output=True,  # Capture output responses
).instrument()
```

### LangChain Integration

```python
from pyflare.integrations import LangChainInstrumentation

LangChainInstrumentation().instrument()
```

---

## Types

### InferenceType

```python
from pyflare import InferenceType

class InferenceType(str, Enum):
    LLM = "llm"
    EMBEDDING = "embedding"
    CLASSIFICATION = "classification"
    REGRESSION = "regression"
    OBJECT_DETECTION = "object_detection"
    CUSTOM = "custom"
```

### TokenUsage

```python
from pyflare import TokenUsage

usage = TokenUsage(
    input_tokens=1000,
    output_tokens=500,
    total_tokens=1500,  # Optional, calculated if not provided
)
```

### ModelInfo

```python
from pyflare import ModelInfo

model = ModelInfo(
    model_id="my-model",
    model_version="1.0.0",
    provider="custom",
    inference_type=InferenceType.CLASSIFICATION,
)
```

### SpanAttributes

Helper for building span attributes:

```python
from pyflare import SpanAttributes

attrs = SpanAttributes(
    model_id="gpt-4o",
    model_version="2024-08-06",
    model_provider="openai",
    inference_type=InferenceType.LLM,
    input_preview="User message...",
    output_preview="Assistant response...",
    input_tokens=100,
    output_tokens=50,
    total_tokens=150,
    cost_micros=1500,
    user_id="user-123",
    feature_id="chat",
    custom={"key": "value"},
)

# Convert to OpenTelemetry format
otel_attrs = attrs.to_otel_attributes()
```

### ModelProvider

```python
from pyflare import ModelProvider

class ModelProvider(str, Enum):
    OPENAI = "openai"
    ANTHROPIC = "anthropic"
    COHERE = "cohere"
    GOOGLE = "google"
    AZURE = "azure"
    AWS_BEDROCK = "aws_bedrock"
    CUSTOM = "custom"
```

---

## Async Support

All decorators work with async functions:

```python
@trace_llm(model_id="gpt-4o")
async def async_chat(prompt: str) -> str:
    response = await openai.chat.completions.create(
        model="gpt-4o",
        messages=[{"role": "user", "content": prompt}]
    )
    return response.choices[0].message.content
```

---

## Error Handling

PyFlare gracefully handles errors:

```python
# If collector is unavailable, traces are dropped silently
pyflare.init("my-service", enabled=True)

# Disable tracing entirely
pyflare.init("my-service", enabled=False)

# Check if initialized
if pyflare.get_pyflare() is not None:
    # PyFlare is available
    pass
```

Functions work normally even if tracing fails:

```python
@trace(name="my-func")
def my_func():
    return "result"

# Returns "result" even if collector is down
result = my_func()
```

---

## Drift Detection API

PyFlare provides APIs for monitoring distribution drift in production.

### DriftMonitor

Monitor feature and embedding drift:

```python
from pyflare.drift import DriftMonitor, DriftType

monitor = DriftMonitor(
    model_id="my-model",
    drift_types=[DriftType.FEATURE, DriftType.EMBEDDING],
    threshold=0.1,
)

# Set reference distribution from training data
monitor.set_reference(training_embeddings)

# Check for drift in production data
result = monitor.check(production_embeddings)

if result.is_drifted:
    print(f"Drift detected! Score: {result.score}")
    print(f"Affected features: {result.drifted_features}")
```

### DriftType

```python
from pyflare.drift import DriftType

class DriftType(str, Enum):
    FEATURE = "feature"      # Numerical feature drift (PSI, KS-test)
    EMBEDDING = "embedding"  # Embedding drift (MMD)
    CONCEPT = "concept"      # Label/prediction drift
    PREDICTION = "prediction"  # Output distribution drift
```

### DriftResult

```python
@dataclass
class DriftResult:
    is_drifted: bool
    score: float
    drift_type: DriftType
    drifted_features: list[str]
    p_value: float | None
    details: dict[str, Any]
```

### Drift Detection Algorithms

PyFlare supports multiple algorithms:

```python
from pyflare.drift import PSIDetector, MMDDetector, KSTestDetector

# Population Stability Index (for binned features)
psi = PSIDetector(threshold=0.2, num_bins=10)
psi.set_reference(reference_data)
result = psi.compute(current_data)

# Maximum Mean Discrepancy (for embeddings)
mmd = MMDDetector(threshold=0.1, rbf_sigma=1.0)
mmd.set_reference(reference_embeddings)
result = mmd.compute(current_embeddings)

# Kolmogorov-Smirnov Test (for continuous distributions)
ks = KSTestDetector(p_value_threshold=0.05)
ks.set_reference(reference_values)
result = ks.compute(current_values)
```

---

## Evaluation API

Evaluate LLM outputs for quality issues.

### Evaluators

```python
from pyflare.eval import (
    HallucinationEvaluator,
    RAGEvaluator,
    ToxicityEvaluator,
)

# Hallucination detection using LLM-as-judge
hallucination = HallucinationEvaluator(
    judge_model="gpt-4o-mini",
    api_key="sk-..."
)
result = hallucination.evaluate(
    input="What is the capital of France?",
    output="Paris was founded in 250 BC.",
    context=["Paris is the capital of France."]
)
# result.has_hallucination = True
# result.explanation = "The founding date is not supported by context"

# RAG quality evaluation
rag = RAGEvaluator()
metrics = rag.evaluate(
    query="What is Python?",
    answer="Python is a programming language.",
    contexts=["Python is a high-level programming language."]
)
# metrics.context_relevance = 0.95
# metrics.faithfulness = 1.0
# metrics.answer_relevance = 0.92
# metrics.groundedness = 1.0

# Toxicity detection
toxicity = ToxicityEvaluator()
result = toxicity.evaluate("Some text to check")
# result.is_toxic = False
# result.scores = {"hate": 0.01, "harassment": 0.02, ...}
```

### EvalResult

```python
@dataclass
class EvalResult:
    evaluator_type: str
    score: float
    verdict: str  # "pass", "fail", "warn"
    explanation: str
    metadata: dict[str, Any]
```

### RAGMetrics

```python
@dataclass
class RAGMetrics:
    context_relevance: float    # How relevant is retrieved context to query
    faithfulness: float         # Is answer faithful to context
    answer_relevance: float     # Does answer address the query
    groundedness: float         # Are claims grounded in context
    has_hallucination: bool
    overall_score: float
    issues: list[str]
```

### Using Evaluations with Traces

Enable automatic evaluation during tracing:

```python
@trace_llm(
    model_id="gpt-4o",
    enable_evaluations=True,
    evaluations=["hallucination", "toxicity"]
)
def chat(messages):
    return openai.chat.completions.create(model="gpt-4o", messages=messages)
```

---

## Budget Management API

Control and monitor LLM spending.

### BudgetManager

```python
from pyflare.cost import BudgetManager, BudgetConfig, BudgetDimension

manager = BudgetManager()

# Create a daily budget for a user
manager.create_budget(BudgetConfig(
    dimension=BudgetDimension.USER,
    dimension_value="user-123",
    daily_limit_micros=10_000_000,  # $10
    warning_threshold=0.8,
    block_on_exceeded=True,
))

# Check budget before making request
check = manager.check_budget(
    dimension=BudgetDimension.USER,
    dimension_value="user-123",
    proposed_spend_micros=500_000  # $0.50
)

if not check.allowed:
    raise Exception(f"Budget exceeded: {check.blocked_reason}")

# Record actual spend
manager.record_spend(
    dimension=BudgetDimension.USER,
    dimension_value="user-123",
    spend_micros=actual_cost
)
```

### BudgetDimension

```python
from pyflare.cost import BudgetDimension

class BudgetDimension(str, Enum):
    GLOBAL = "global"          # Total spend
    USER = "user"              # Per-user budget
    MODEL = "model"            # Per-model budget
    FEATURE = "feature"        # Per-feature/endpoint
    TEAM = "team"              # Per-team budget
    ENVIRONMENT = "environment"  # Per-environment
```

### BudgetConfig

```python
@dataclass
class BudgetConfig:
    dimension: BudgetDimension
    dimension_value: str
    period: BudgetPeriod = BudgetPeriod.DAILY
    soft_limit_micros: int = 0
    hard_limit_micros: int = 0
    warning_threshold: float = 0.8
    block_on_exceeded: bool = False
```

### Budget Alerts

Register callbacks for budget alerts:

```python
def on_budget_alert(alert):
    print(f"Budget alert: {alert.type} for {alert.dimension_value}")
    print(f"Current spend: ${alert.current_spend_micros / 1_000_000:.2f}")

manager.register_alert_callback(on_budget_alert)
```

---

## Query API Client

Access PyFlare data programmatically.

### TracesAPI

```python
from pyflare.api import TracesAPI

api = TracesAPI(endpoint="http://localhost:8080")

# List traces with filtering
traces = api.list(
    model_id="gpt-4o",
    status="error",
    start_time="2024-01-01T00:00:00Z",
    limit=100
)

# Get trace details
trace = api.get("trace-id-123")

# Get trace statistics
stats = api.stats(
    model_id="gpt-4o",
    start_time="2024-01-01T00:00:00Z",
    end_time="2024-01-02T00:00:00Z"
)
```

### DriftAPI

```python
from pyflare.api import DriftAPI

api = DriftAPI(endpoint="http://localhost:8080")

# Get drift alerts
alerts = api.get_alerts(
    model_id="my-model",
    severity="high",
    limit=50
)

# Get drift status
status = api.get_status(model_id="my-model")

# Get drift heatmap data
heatmap = api.get_heatmap(
    model_id="my-model",
    start_time="2024-01-01T00:00:00Z"
)
```

### CostsAPI

```python
from pyflare.api import CostsAPI

api = CostsAPI(endpoint="http://localhost:8080")

# Get cost summary
summary = api.get_summary(
    start_time="2024-01-01T00:00:00Z",
    group_by="model"
)

# Get cost breakdown
breakdown = api.get_breakdown(
    start_time="2024-01-01T00:00:00Z",
    dimensions=["model", "user"]
)

# Get budget status
budgets = api.get_budgets()
```

### RCAAPI

```python
from pyflare.api import RCAAPI

api = RCAAPI(endpoint="http://localhost:8080")

# Run RCA analysis
analysis = api.run_analysis(model_id="my-model")

# Get detected patterns
patterns = api.get_patterns(model_id="my-model")

# Get failure clusters
clusters = api.get_clusters(model_id="my-model")

# Get problematic slices
slices = api.get_slices(
    model_id="my-model",
    metric="error_rate",
    limit=10
)
```
