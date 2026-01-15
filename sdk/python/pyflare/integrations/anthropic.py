"""Anthropic (Claude) integration for PyFlare."""

import functools
from typing import Any, Optional

from opentelemetry import trace
from opentelemetry.trace import SpanKind, Status, StatusCode

from pyflare.sdk import PyFlare
from pyflare.types import InferenceType, TokenUsage
from pyflare.cost import calculate_cost


class AnthropicInstrumentation:
    """
    Automatic instrumentation for Anthropic API calls.

    Example:
        from pyflare.integrations import AnthropicInstrumentation

        # Initialize PyFlare
        pyflare = PyFlare(service_name="my-service")

        # Enable Anthropic instrumentation
        AnthropicInstrumentation().instrument()

        # Now all Anthropic calls are automatically traced
        response = anthropic.messages.create(...)
    """

    def __init__(self, capture_input: bool = True, capture_output: bool = True) -> None:
        """
        Initialize Anthropic instrumentation.

        Args:
            capture_input: Whether to capture input prompts
            capture_output: Whether to capture output responses
        """
        self._instrumented = False
        self._capture_input = capture_input
        self._capture_output = capture_output
        self._original_methods: dict[str, Any] = {}

    def instrument(self) -> None:
        """Enable Anthropic instrumentation."""
        if self._instrumented:
            return

        try:
            import anthropic
        except ImportError:
            raise ImportError(
                "anthropic package is required for Anthropic instrumentation. "
                "Install with: pip install pyflare[anthropic]"
            )

        self._patch_messages_create(anthropic)
        self._instrumented = True

    def uninstrument(self) -> None:
        """Disable Anthropic instrumentation."""
        if not self._instrumented:
            return

        try:
            import anthropic
            # Restore original methods
            if "Messages.create" in self._original_methods:
                anthropic.Anthropic.messages.create = self._original_methods["Messages.create"]
        except ImportError:
            pass

        self._original_methods.clear()
        self._instrumented = False

    def _patch_messages_create(self, anthropic: Any) -> None:
        """Patch messages.create method."""
        original_create = anthropic.Anthropic.messages.create

        @functools.wraps(original_create)
        def traced_create(self_client: Any, **kwargs: Any) -> Any:
            pyflare = PyFlare.get_instance()
            if pyflare is None or not pyflare.enabled:
                return original_create(self_client, **kwargs)

            tracer = pyflare.tracer
            model = kwargs.get("model", "unknown")

            with tracer.start_as_current_span(
                "anthropic.messages.create",
                kind=SpanKind.CLIENT,
            ) as span:
                span.set_attribute("pyflare.inference.type", InferenceType.LLM.value)
                span.set_attribute("pyflare.model.id", model)
                span.set_attribute("pyflare.model.provider", "anthropic")

                # Capture input
                if self._capture_input:
                    messages = kwargs.get("messages", [])
                    if messages:
                        last_message = messages[-1]
                        content = last_message.get("content", "")
                        if isinstance(content, list):
                            # Handle multi-part messages
                            text_parts = [
                                p.get("text", "") for p in content
                                if p.get("type") == "text"
                            ]
                            content = " ".join(text_parts)
                        span.set_attribute("pyflare.input.preview", str(content)[:1000])

                    # Capture system prompt if present
                    system = kwargs.get("system")
                    if system:
                        span.set_attribute("pyflare.system.preview", str(system)[:500])

                # Capture parameters
                max_tokens = kwargs.get("max_tokens", 4096)
                span.set_attribute("pyflare.params.max_tokens", max_tokens)

                temperature = kwargs.get("temperature")
                if temperature is not None:
                    span.set_attribute("pyflare.params.temperature", temperature)

                try:
                    result = original_create(self_client, **kwargs)

                    # Extract token usage
                    if hasattr(result, "usage") and result.usage:
                        input_tokens = result.usage.input_tokens
                        output_tokens = result.usage.output_tokens
                        total_tokens = input_tokens + output_tokens

                        span.set_attribute("pyflare.tokens.input", input_tokens)
                        span.set_attribute("pyflare.tokens.output", output_tokens)
                        span.set_attribute("pyflare.tokens.total", total_tokens)

                        # Calculate cost
                        cost_result = calculate_cost(model, input_tokens, output_tokens)
                        span.set_attribute("pyflare.cost.micros", cost_result.total_micros)

                    # Extract output
                    if self._capture_output and hasattr(result, "content") and result.content:
                        output_parts = []
                        for block in result.content:
                            if hasattr(block, "text"):
                                output_parts.append(block.text)
                        output = "".join(output_parts)
                        span.set_attribute("pyflare.output.preview", output[:1000])

                    # Capture stop reason
                    if hasattr(result, "stop_reason"):
                        span.set_attribute("pyflare.stop_reason", result.stop_reason)

                    span.set_status(Status(StatusCode.OK))
                    return result

                except Exception as e:
                    span.set_status(Status(StatusCode.ERROR, str(e)))
                    span.record_exception(e)
                    raise

        self._original_methods["Messages.create"] = original_create
        anthropic.Anthropic.messages.create = traced_create

    def _patch_async_messages_create(self, anthropic: Any) -> None:
        """Patch async messages.create method."""
        if not hasattr(anthropic, "AsyncAnthropic"):
            return

        original_create = anthropic.AsyncAnthropic.messages.create

        @functools.wraps(original_create)
        async def traced_create(self_client: Any, **kwargs: Any) -> Any:
            pyflare = PyFlare.get_instance()
            if pyflare is None or not pyflare.enabled:
                return await original_create(self_client, **kwargs)

            tracer = pyflare.tracer
            model = kwargs.get("model", "unknown")

            with tracer.start_as_current_span(
                "anthropic.messages.create",
                kind=SpanKind.CLIENT,
            ) as span:
                span.set_attribute("pyflare.inference.type", InferenceType.LLM.value)
                span.set_attribute("pyflare.model.id", model)
                span.set_attribute("pyflare.model.provider", "anthropic")

                try:
                    result = await original_create(self_client, **kwargs)

                    # Extract usage and output similar to sync version
                    if hasattr(result, "usage") and result.usage:
                        input_tokens = result.usage.input_tokens
                        output_tokens = result.usage.output_tokens

                        span.set_attribute("pyflare.tokens.input", input_tokens)
                        span.set_attribute("pyflare.tokens.output", output_tokens)
                        span.set_attribute("pyflare.tokens.total", input_tokens + output_tokens)

                        cost_result = calculate_cost(model, input_tokens, output_tokens)
                        span.set_attribute("pyflare.cost.micros", cost_result.total_micros)

                    span.set_status(Status(StatusCode.OK))
                    return result

                except Exception as e:
                    span.set_status(Status(StatusCode.ERROR, str(e)))
                    span.record_exception(e)
                    raise

        self._original_methods["AsyncMessages.create"] = original_create
        anthropic.AsyncAnthropic.messages.create = traced_create


def trace_anthropic_call(
    model: Optional[str] = None,
    capture_input: bool = True,
    capture_output: bool = True,
):
    """
    Decorator for tracing Anthropic calls.

    Example:
        @trace_anthropic_call(model="claude-3-opus")
        def ask_claude(prompt: str) -> str:
            response = client.messages.create(
                model="claude-3-opus-20240229",
                messages=[{"role": "user", "content": prompt}]
            )
            return response.content[0].text
    """
    from pyflare.decorators import trace

    return trace(
        model_id=model,
        inference_type=InferenceType.LLM,
        capture_input=capture_input,
        capture_output=capture_output,
        attributes={"provider": "anthropic"},
    )
