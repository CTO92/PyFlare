# Getting Started with PyFlare

PyFlare is an OpenTelemetry-native observability platform designed specifically for AI/ML workloads. This guide will help you get PyFlare up and running in minutes.

## Overview

PyFlare provides:
- **Automatic tracing** of AI/ML model inference calls
- **Cost tracking** for LLM API usage (OpenAI, Anthropic, etc.)
- **Budget management** with alerts and spend controls
- **Performance monitoring** with latency and throughput metrics
- **Advanced drift detection** (PSI, MMD, KS-test) for production model monitoring
- **LLM evaluations** including hallucination detection, RAG quality, and toxicity scoring
- **Root cause analysis** with slice analysis and failure clustering

## Quick Start

### 1. Install the Python SDK

```bash
# Clone the repository
git clone https://github.com/oaqlabs/pyflare.git
cd pyflare/sdk/python

# Install the SDK
pip install -e .

# With optional integrations
pip install -e ".[openai]"      # OpenAI support
pip install -e ".[anthropic]"   # Anthropic support
pip install -e ".[all]"         # All integrations
```

### 2. Initialize PyFlare

```python
import pyflare

# Initialize with your service name
pyflare.init(
    service_name="my-ml-service",
    endpoint="http://localhost:4317",  # PyFlare collector endpoint
    environment="development",
)
```

### 3. Trace Your First Function

```python
from pyflare import trace

@trace(name="predict", model_id="my-model")
def predict(input_data):
    # Your ML inference code
    return model.predict(input_data)

# Now your function is automatically traced!
result = predict(data)
```

### 4. View Your Traces

Open the PyFlare UI at `http://localhost:3000` to see your traces, costs, and performance metrics.

## Installation Options

### Using Docker (Recommended for Development)

The fastest way to get started is using Docker Compose:

```bash
git clone https://github.com/your-org/pyflare.git
cd pyflare/deploy/docker
docker-compose up -d
```

This starts:
- **Collector** on ports 4317 (gRPC) and 4318 (HTTP)
- **Kafka** for message streaming
- **ClickHouse** for trace storage
- **Query API** on port 8080
- **Web UI** on port 3000

### Installing the SDK Only

If you already have a collector running and just need the Python SDK:

```bash
git clone https://github.com/oaqlabs/pyflare.git
cd pyflare/sdk/python
pip install -e .
```

### Building from Source

See the [Developer Documentation](../developer/building.md) for instructions on building from source.

## Basic Usage

### Tracing LLM Calls

PyFlare provides specialized decorators for LLM tracing:

```python
from pyflare import trace_llm

@trace_llm(model_id="gpt-4o", provider="openai")
def chat_completion(messages):
    response = openai.chat.completions.create(
        model="gpt-4o",
        messages=messages
    )
    return response

# Token usage and cost are automatically captured
response = chat_completion([{"role": "user", "content": "Hello!"}])
```

### Tracing Embeddings

```python
from pyflare import trace_embedding

@trace_embedding(model_id="text-embedding-3-small", provider="openai")
def get_embedding(text):
    response = openai.embeddings.create(
        model="text-embedding-3-small",
        input=text
    )
    return response.data[0].embedding
```

### Using Context Managers

For more control over spans, use context managers:

```python
import pyflare

with pyflare.llm_span("chat-completion", model_id="gpt-4o", provider="openai") as span:
    response = client.chat.completions.create(
        model="gpt-4o",
        messages=messages
    )

    # Add custom attributes
    span.set_attribute("pyflare.tokens.input", response.usage.prompt_tokens)
    span.set_attribute("pyflare.tokens.output", response.usage.completion_tokens)
```

### Automatic Instrumentation

For zero-code instrumentation, enable auto-instrumentation:

```python
from pyflare.integrations import OpenAIInstrumentation, AnthropicInstrumentation

# Enable OpenAI auto-instrumentation
OpenAIInstrumentation().instrument()

# Enable Anthropic auto-instrumentation
AnthropicInstrumentation().instrument()

# Now all API calls are automatically traced
response = openai.chat.completions.create(...)  # Traced automatically!
```

## User and Feature Attribution

Track costs per user or feature:

```python
import pyflare

# Set the current user
pyflare.set_user("user-123", attributes={"plan": "premium"})

# Set the current feature
pyflare.set_feature("chat-assistant")

# All subsequent traces will include this attribution
response = chat_completion(messages)
```

## Configuration Options

### Environment Variables

PyFlare can be configured via environment variables:

```bash
export PYFLARE_ENDPOINT="http://collector.example.com:4317"
export PYFLARE_ENABLED="true"
export PYFLARE_SAMPLE_RATE="0.5"
export PYFLARE_ENVIRONMENT="production"
```

### Programmatic Configuration

```python
pyflare.init(
    service_name="my-service",
    endpoint="http://localhost:4317",
    environment="production",
    version="1.0.0",
    sample_rate=0.5,          # Sample 50% of traces
    debug=False,              # Disable debug output
    use_http=False,           # Use gRPC (default)
    batch_export=True,        # Batch spans for efficiency
    resource_attributes={     # Custom resource attributes
        "deployment.region": "us-west-2",
        "team": "ml-platform",
    },
)
```

## Advanced Features

### Drift Detection

PyFlare automatically monitors for distribution shifts in your model inputs and outputs:

```python
from pyflare import trace_llm

@trace_llm(model_id="gpt-4o", enable_drift_detection=True)
def chat(messages):
    return openai.chat.completions.create(model="gpt-4o", messages=messages)
```

PyFlare supports multiple drift detection algorithms:
- **PSI (Population Stability Index)** - For categorical and binned numerical features
- **MMD (Maximum Mean Discrepancy)** - For embedding drift with RBF kernel
- **KS Test (Kolmogorov-Smirnov)** - For continuous numerical distributions

View drift alerts in the UI or configure webhooks for notifications.

### LLM Evaluations

Automatically evaluate LLM outputs for quality issues:

```python
from pyflare import trace_llm

@trace_llm(
    model_id="gpt-4o",
    enable_evaluations=True,
    evaluations=["hallucination", "toxicity", "rag_quality"]
)
def rag_chat(query, context):
    return openai.chat.completions.create(
        model="gpt-4o",
        messages=[
            {"role": "system", "content": f"Context: {context}"},
            {"role": "user", "content": query}
        ]
    )
```

Available evaluators:
- **Hallucination Detection** - LLM-as-judge for detecting unsupported claims
- **RAG Quality** - Context relevance, faithfulness, answer relevance, groundedness
- **Toxicity Detection** - Detect harmful content with category breakdown
- **PII Detection** - Identify and optionally scrub sensitive data

### Budget Management

Control costs with configurable budgets and alerts:

```python
import pyflare

# Set budget alerts via configuration
pyflare.init(
    service_name="my-service",
    budget_config={
        "daily_limit_usd": 100.0,
        "warning_threshold": 0.8,  # Alert at 80%
        "block_on_exceeded": False,
    }
)

# Or set budgets per user/feature programmatically
pyflare.set_budget(
    dimension="user",
    dimension_value="user-123",
    daily_limit_usd=10.0,
    block_on_exceeded=True
)
```

### Root Cause Analysis

When issues occur, PyFlare helps identify problematic data slices:

```python
# Access RCA insights via the Query API
import requests

response = requests.get(
    "http://localhost:8080/api/v1/rca/slices",
    params={"model_id": "gpt-4o", "metric": "error_rate"}
)

# Returns problematic slices like:
# - "Users with input_length > 4000 have 3x higher error rate"
# - "Requests between 2-4am have 50% higher latency"
```

The RCA system provides:
- **Slice Analysis** - Identify segments with degraded performance
- **Pattern Detection** - Detect error spikes, latency degradation, quality drops
- **Failure Clustering** - Group similar failures for efficient debugging

## Next Steps

- [SDK Reference](./sdk-reference.md) - Complete API documentation
- [Configuration Guide](./configuration.md) - Detailed configuration options
- [Drift Detection Guide](./drift-detection.md) - Advanced drift monitoring
- [Evaluation Guide](./evaluations.md) - LLM quality evaluations
- [Integrations](./integrations.md) - Framework-specific integrations
- [Best Practices](./best-practices.md) - Production deployment tips

## Troubleshooting

### Traces Not Appearing

1. **Check collector connectivity:**
   ```python
   pyflare.init(service_name="test", debug=True)
   ```
   Debug mode will print spans to console.

2. **Verify endpoint:**
   ```bash
   curl http://localhost:4318/health
   ```

3. **Check sampling rate:**
   Ensure `sample_rate` is not set to 0.

### High Latency

1. Use batch export (default):
   ```python
   pyflare.init(..., batch_export=True)
   ```

2. Increase batch size in collector config.

### Missing Token Counts

Ensure you're using the correct decorator:
- `@trace_llm` for LLM calls (captures token usage)
- `@trace_embedding` for embedding calls
- `@trace` for general functions

## Getting Help

- GitHub Issues: [github.com/your-org/pyflare/issues](https://github.com/your-org/pyflare/issues)
- Documentation: [docs.pyflare.io](https://docs.pyflare.io)
