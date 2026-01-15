# PyFlare Python SDK

OpenTelemetry-native observability SDK for AI/ML workloads.

## Installation

Install from the GitHub repository:

```bash
# Clone the repository
git clone https://github.com/oaqlabs/pyflare.git
cd pyflare/sdk/python

# Install the SDK
pip install -e .

# With optional integrations
pip install -e ".[openai]"      # OpenAI integration
pip install -e ".[langchain]"   # LangChain integration
pip install -e ".[pytorch]"     # PyTorch integration
pip install -e ".[all]"         # All integrations
```

## Quick Start

```python
from pyflare import PyFlare, trace

# Initialize PyFlare
pyflare = PyFlare(
    service_name="my-ml-service",
    endpoint="http://localhost:4317",
)

# Trace any function
@trace(name="my_inference", model_id="my-model")
def predict(input_data):
    return model.predict(input_data)
```

## LLM Tracing

```python
from pyflare import trace_llm

@trace_llm(model_id="gpt-4", provider="openai")
def chat(messages):
    return openai.chat.completions.create(
        model="gpt-4",
        messages=messages
    )
```

## Automatic Instrumentation

```python
from pyflare import PyFlare
from pyflare.integrations import OpenAIInstrumentation

# Initialize
pyflare = PyFlare(service_name="my-service")

# Enable automatic OpenAI tracing
OpenAIInstrumentation().instrument()

# All OpenAI calls are now traced automatically
response = openai.chat.completions.create(...)
```

## Features

- **Automatic tracing** for OpenAI, LangChain, PyTorch
- **Token usage tracking** for cost analysis
- **Async support** for modern Python applications
- **Context propagation** across distributed systems
- **Custom attributes** for flexible metadata

## Configuration

```python
pyflare = PyFlare(
    service_name="my-service",      # Required
    endpoint="http://localhost:4317", # Collector endpoint
    environment="production",        # Environment tag
    version="1.0.0",                # Service version
    sample_rate=1.0,                # Sampling rate (0.0-1.0)
    enabled=True,                   # Enable/disable tracing
)
```

## License

Apache 2.0
