"""
PyFlare SDK - OpenTelemetry-native observability for AI/ML workloads.

PyFlare provides deep observability into AI/ML applications, including:
- Automatic tracing of model inference
- Drift detection for production models
- Cost tracking for LLM usage
- Hallucination and quality evaluation

Basic usage:
    from pyflare import PyFlare, trace

    # Initialize PyFlare
    pyflare = PyFlare(
        service_name="my-ml-service",
        endpoint="http://localhost:4317",
    )

    # Trace a function
    @trace(name="my_inference")
    def predict(input_data):
        return model.predict(input_data)

Quick initialization:
    import pyflare

    # One-liner initialization
    pyflare.init("my-service")

    # Use decorators
    @pyflare.trace(model_id="gpt-4")
    def chat(prompt):
        ...

Context manager:
    with pyflare.span("process-data") as span:
        span.set_attribute("items", len(data))
        result = process(data)
"""

from pyflare.sdk import PyFlare, get_pyflare, init
from pyflare.decorators import trace, trace_llm, trace_embedding
from pyflare.types import (
    InferenceType,
    SpanKind,
    TraceContext,
    SpanAttributes,
    ModelInfo,
    TokenUsage,
)
from pyflare.cost import (
    CostCalculator,
    CostResult,
    ModelPricing,
    ModelProvider,
    calculate_cost,
    get_cost_calculator,
)
from pyflare.exporters import (
    PyFlareExporter,
    ConsoleExporter,
)

__version__ = "1.0.0"
__all__ = [
    # Core
    "PyFlare",
    "get_pyflare",
    "init",
    # Decorators
    "trace",
    "trace_llm",
    "trace_embedding",
    # Types
    "InferenceType",
    "SpanKind",
    "TraceContext",
    "SpanAttributes",
    "ModelInfo",
    "TokenUsage",
    # Cost
    "CostCalculator",
    "CostResult",
    "ModelPricing",
    "ModelProvider",
    "calculate_cost",
    "get_cost_calculator",
    # Exporters
    "PyFlareExporter",
    "ConsoleExporter",
]


# Module-level convenience functions
def span(name: str, **kwargs):
    """Create a span context manager using the global instance."""
    instance = get_pyflare()
    if instance:
        return instance.span(name, **kwargs)
    # Return a no-op context manager if not initialized
    from contextlib import nullcontext
    return nullcontext()


def llm_span(name: str, model_id: str, provider: str = ""):
    """Create an LLM span context manager using the global instance."""
    instance = get_pyflare()
    if instance:
        return instance.llm_span(name, model_id, provider)
    from contextlib import nullcontext
    return nullcontext()


def set_user(user_id: str, attributes=None):
    """Set the current user on the global instance."""
    instance = get_pyflare()
    if instance:
        instance.set_user(user_id, attributes)


def set_feature(feature_id: str):
    """Set the current feature on the global instance."""
    instance = get_pyflare()
    if instance:
        instance.set_feature(feature_id)


def set_session(session_id: str):
    """Set the current session on the global instance."""
    instance = get_pyflare()
    if instance:
        instance.set_session(session_id)


def flush(timeout_millis: int = 30000) -> bool:
    """Flush pending spans on the global instance."""
    instance = get_pyflare()
    if instance:
        return instance.flush(timeout_millis)
    return True


def shutdown():
    """Shutdown the global instance."""
    instance = get_pyflare()
    if instance:
        instance.shutdown()
