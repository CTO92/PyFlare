"""Decorators for tracing ML functions."""

import functools
import inspect
from typing import Any, Callable, Optional, TypeVar, Union

from opentelemetry import trace
from opentelemetry.trace import SpanKind, Status, StatusCode

from pyflare.sdk import PyFlare
from pyflare.types import InferenceType, SpanAttributes, TokenUsage

F = TypeVar("F", bound=Callable[..., Any])


def trace(
    name: Optional[str] = None,
    *,
    model_id: Optional[str] = None,
    inference_type: InferenceType = InferenceType.CUSTOM,
    capture_input: bool = True,
    capture_output: bool = True,
    attributes: Optional[dict[str, Any]] = None,
) -> Callable[[F], F]:
    """
    Decorator to trace a function.

    Args:
        name: Span name (defaults to function name)
        model_id: Model identifier
        inference_type: Type of inference
        capture_input: Whether to capture function input
        capture_output: Whether to capture function output
        attributes: Additional span attributes

    Example:
        @trace(name="predict", model_id="my-model")
        def predict(input_data):
            return model(input_data)
    """

    def decorator(func: F) -> F:
        span_name = name or func.__name__

        @functools.wraps(func)
        def wrapper(*args: Any, **kwargs: Any) -> Any:
            pyflare = PyFlare.get_instance()
            if pyflare is None or not pyflare.enabled:
                return func(*args, **kwargs)

            tracer = pyflare.tracer

            with tracer.start_as_current_span(
                span_name, kind=SpanKind.INTERNAL
            ) as span:
                # Set basic attributes
                if model_id:
                    span.set_attribute("pyflare.model.id", model_id)
                span.set_attribute("pyflare.inference.type", inference_type.value)

                # Capture input
                if capture_input and args:
                    input_str = _safe_repr(args[0])
                    span.set_attribute("pyflare.input.preview", input_str[:1000])

                # Add custom attributes
                if attributes:
                    for key, value in attributes.items():
                        span.set_attribute(f"pyflare.custom.{key}", value)

                try:
                    result = func(*args, **kwargs)

                    # Capture output
                    if capture_output and result is not None:
                        output_str = _safe_repr(result)
                        span.set_attribute("pyflare.output.preview", output_str[:1000])

                    span.set_status(Status(StatusCode.OK))
                    return result

                except Exception as e:
                    span.set_status(Status(StatusCode.ERROR, str(e)))
                    span.record_exception(e)
                    raise

        # Handle async functions
        if inspect.iscoroutinefunction(func):

            @functools.wraps(func)
            async def async_wrapper(*args: Any, **kwargs: Any) -> Any:
                pyflare = PyFlare.get_instance()
                if pyflare is None or not pyflare.enabled:
                    return await func(*args, **kwargs)

                tracer = pyflare.tracer

                with tracer.start_as_current_span(
                    span_name, kind=SpanKind.INTERNAL
                ) as span:
                    if model_id:
                        span.set_attribute("pyflare.model.id", model_id)
                    span.set_attribute("pyflare.inference.type", inference_type.value)

                    if capture_input and args:
                        input_str = _safe_repr(args[0])
                        span.set_attribute("pyflare.input.preview", input_str[:1000])

                    if attributes:
                        for key, value in attributes.items():
                            span.set_attribute(f"pyflare.custom.{key}", value)

                    try:
                        result = await func(*args, **kwargs)

                        if capture_output and result is not None:
                            output_str = _safe_repr(result)
                            span.set_attribute("pyflare.output.preview", output_str[:1000])

                        span.set_status(Status(StatusCode.OK))
                        return result

                    except Exception as e:
                        span.set_status(Status(StatusCode.ERROR, str(e)))
                        span.record_exception(e)
                        raise

            return async_wrapper  # type: ignore

        return wrapper  # type: ignore

    return decorator


def trace_llm(
    name: Optional[str] = None,
    *,
    model_id: Optional[str] = None,
    provider: Optional[str] = None,
    capture_input: bool = True,
    capture_output: bool = True,
) -> Callable[[F], F]:
    """
    Decorator specifically for LLM inference.

    Automatically captures token usage if the return value has
    usage information (like OpenAI responses).

    Example:
        @trace_llm(model_id="gpt-4", provider="openai")
        def chat(messages):
            return openai.chat.completions.create(
                model="gpt-4",
                messages=messages
            )
    """

    def decorator(func: F) -> F:
        span_name = name or func.__name__

        @functools.wraps(func)
        def wrapper(*args: Any, **kwargs: Any) -> Any:
            pyflare = PyFlare.get_instance()
            if pyflare is None or not pyflare.enabled:
                return func(*args, **kwargs)

            tracer = pyflare.tracer

            with tracer.start_as_current_span(
                span_name, kind=SpanKind.CLIENT
            ) as span:
                span.set_attribute("pyflare.inference.type", InferenceType.LLM.value)

                if model_id:
                    span.set_attribute("pyflare.model.id", model_id)
                if provider:
                    span.set_attribute("pyflare.model.provider", provider)

                if capture_input and args:
                    input_str = _safe_repr(args[0])
                    span.set_attribute("pyflare.input.preview", input_str[:1000])

                try:
                    result = func(*args, **kwargs)

                    # Try to extract token usage
                    usage = _extract_token_usage(result)
                    if usage:
                        span.set_attribute("pyflare.tokens.input", usage.input_tokens)
                        span.set_attribute("pyflare.tokens.output", usage.output_tokens)
                        span.set_attribute("pyflare.tokens.total", usage.total_tokens)

                    if capture_output and result is not None:
                        output_str = _extract_llm_output(result)
                        span.set_attribute("pyflare.output.preview", output_str[:1000])

                    span.set_status(Status(StatusCode.OK))
                    return result

                except Exception as e:
                    span.set_status(Status(StatusCode.ERROR, str(e)))
                    span.record_exception(e)
                    raise

        return wrapper  # type: ignore

    return decorator


def trace_embedding(
    name: Optional[str] = None,
    *,
    model_id: Optional[str] = None,
    provider: Optional[str] = None,
    capture_input: bool = True,
) -> Callable[[F], F]:
    """
    Decorator for embedding generation.

    Example:
        @trace_embedding(model_id="text-embedding-3-small", provider="openai")
        def get_embedding(text):
            return openai.embeddings.create(
                model="text-embedding-3-small",
                input=text
            )
    """

    def decorator(func: F) -> F:
        span_name = name or func.__name__

        @functools.wraps(func)
        def wrapper(*args: Any, **kwargs: Any) -> Any:
            pyflare = PyFlare.get_instance()
            if pyflare is None or not pyflare.enabled:
                return func(*args, **kwargs)

            tracer = pyflare.tracer

            with tracer.start_as_current_span(
                span_name, kind=SpanKind.CLIENT
            ) as span:
                span.set_attribute(
                    "pyflare.inference.type", InferenceType.EMBEDDING.value
                )

                if model_id:
                    span.set_attribute("pyflare.model.id", model_id)
                if provider:
                    span.set_attribute("pyflare.model.provider", provider)

                if capture_input and args:
                    input_str = _safe_repr(args[0])
                    span.set_attribute("pyflare.input.preview", input_str[:1000])

                try:
                    result = func(*args, **kwargs)

                    # Record embedding dimension if available
                    embedding = _extract_embedding(result)
                    if embedding:
                        span.set_attribute("pyflare.embedding.dimensions", len(embedding))

                    span.set_status(Status(StatusCode.OK))
                    return result

                except Exception as e:
                    span.set_status(Status(StatusCode.ERROR, str(e)))
                    span.record_exception(e)
                    raise

        return wrapper  # type: ignore

    return decorator


def _safe_repr(obj: Any) -> str:
    """Safely convert object to string representation."""
    try:
        if isinstance(obj, str):
            return obj
        if isinstance(obj, (list, dict)):
            import json

            return json.dumps(obj, default=str)[:1000]
        return str(obj)[:1000]
    except Exception:
        return "<unserializable>"


def _extract_token_usage(result: Any) -> Optional[TokenUsage]:
    """Extract token usage from LLM response."""
    try:
        # OpenAI format
        if hasattr(result, "usage"):
            usage = result.usage
            return TokenUsage(
                input_tokens=getattr(usage, "prompt_tokens", 0),
                output_tokens=getattr(usage, "completion_tokens", 0),
                total_tokens=getattr(usage, "total_tokens", 0),
            )
        # Dict format
        if isinstance(result, dict) and "usage" in result:
            usage = result["usage"]
            return TokenUsage(
                input_tokens=usage.get("prompt_tokens", 0),
                output_tokens=usage.get("completion_tokens", 0),
                total_tokens=usage.get("total_tokens", 0),
            )
    except Exception:
        pass
    return None


def _extract_llm_output(result: Any) -> str:
    """Extract output text from LLM response."""
    try:
        # OpenAI ChatCompletion format
        if hasattr(result, "choices") and result.choices:
            choice = result.choices[0]
            if hasattr(choice, "message") and hasattr(choice.message, "content"):
                return choice.message.content or ""
        # Dict format
        if isinstance(result, dict):
            if "choices" in result and result["choices"]:
                return result["choices"][0].get("message", {}).get("content", "")
    except Exception:
        pass
    return _safe_repr(result)


def _extract_embedding(result: Any) -> Optional[list[float]]:
    """Extract embedding vector from response."""
    try:
        # OpenAI format
        if hasattr(result, "data") and result.data:
            return result.data[0].embedding
        # List format
        if isinstance(result, list) and result and isinstance(result[0], (int, float)):
            return result
    except Exception:
        pass
    return None
