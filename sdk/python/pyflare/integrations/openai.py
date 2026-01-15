"""
OpenAI integration for PyFlare.

Security Features:
- Capture disabled by default (SEC-005 fix)
- PII scrubbing for captured data
- Proper exception logging (SEC-007 fix)
"""

from typing import Any, AsyncIterator, Iterator, Optional, Union
import functools
import logging
import time

from opentelemetry import trace
from opentelemetry.trace import SpanKind, Status, StatusCode

from pyflare.sdk import PyFlare
from pyflare.types import InferenceType, TokenUsage
from pyflare.cost import calculate_cost
from pyflare.security import scrub_text

logger = logging.getLogger(__name__)


class OpenAIInstrumentation:
    """
    Automatic instrumentation for OpenAI API calls.

    Supports:
    - Chat completions (sync and async)
    - Embeddings
    - Streaming responses
    - Cost calculation
    - PII scrubbing

    Security Note:
        By default, input/output capture is DISABLED to prevent accidental
        storage of sensitive data. Enable with caution and ensure PII
        scrubbing is configured appropriately.

    Example:
        from pyflare.integrations import OpenAIInstrumentation

        # Initialize PyFlare
        pyflare = PyFlare(service_name="my-service")

        # Enable OpenAI instrumentation (capture disabled by default)
        OpenAIInstrumentation().instrument()

        # To enable capture with PII scrubbing:
        OpenAIInstrumentation(
            capture_input=True,
            capture_output=True,
            scrub_pii=True,
        ).instrument()
    """

    def __init__(
        self,
        capture_input: bool = False,  # SEC-005: Default to False
        capture_output: bool = False,  # SEC-005: Default to False
        calculate_costs: bool = True,
        scrub_pii: bool = True,  # SEC-005: Enable PII scrubbing by default
        max_capture_length: int = 500,  # Limit captured content
    ) -> None:
        """
        Initialize OpenAI instrumentation.

        Args:
            capture_input: Whether to capture input prompts (default: False for security)
            capture_output: Whether to capture output responses (default: False for security)
            calculate_costs: Whether to calculate and track costs
            scrub_pii: Whether to scrub PII from captured data (default: True)
            max_capture_length: Maximum length of captured content (default: 500)

        Security Warning:
            Enabling capture_input or capture_output may result in sensitive
            data being stored in traces. Ensure scrub_pii is enabled and
            review your data retention policies.
        """
        self._instrumented = False
        self._capture_input = capture_input
        self._capture_output = capture_output
        self._calculate_costs = calculate_costs
        self._scrub_pii = scrub_pii
        self._max_capture_length = max_capture_length
        self._original_methods: dict[str, Any] = {}

        if capture_input or capture_output:
            logger.warning(
                "OpenAI instrumentation: Data capture is enabled. "
                "Ensure PII scrubbing is configured and review data retention policies. "
                f"capture_input={capture_input}, capture_output={capture_output}, scrub_pii={scrub_pii}"
            )

    def instrument(self) -> None:
        """Enable OpenAI instrumentation."""
        if self._instrumented:
            return

        try:
            import openai
            from openai import OpenAI, AsyncOpenAI
        except ImportError as e:
            logger.error(
                "Failed to import openai package for instrumentation: %s",
                str(e)
            )
            raise ImportError(
                "openai package is required for OpenAI instrumentation. "
                "Install with: pip install pyflare[openai]"
            )

        # Patch sync client methods
        self._patch_chat_completions(openai)
        self._patch_embeddings(openai)

        # Patch async client methods
        self._patch_async_chat_completions(openai)
        self._patch_async_embeddings(openai)

        self._instrumented = True
        logger.info("OpenAI instrumentation enabled")

    def uninstrument(self) -> None:
        """Disable OpenAI instrumentation."""
        if not self._instrumented:
            return

        # Restore original methods
        for path, method in self._original_methods.items():
            try:
                parts = path.split(".")
                obj = __import__("openai")
                for part in parts[:-1]:
                    obj = getattr(obj, part)
                setattr(obj, parts[-1], method)
            except Exception as e:
                logger.warning(
                    "Failed to restore original method %s: %s",
                    path, str(e)
                )

        self._original_methods.clear()
        self._instrumented = False
        logger.info("OpenAI instrumentation disabled")

    def _sanitize_content(self, content: str) -> str:
        """Sanitize and optionally scrub PII from content."""
        if not content:
            return ""

        # Truncate to max length
        result = content[:self._max_capture_length]

        # Scrub PII if enabled
        if self._scrub_pii:
            result = scrub_text(result)

        return result

    def _extract_messages_preview(self, messages: list[dict]) -> str:
        """Extract a preview of the messages."""
        if not messages:
            return ""
        last_message = messages[-1]
        content = last_message.get("content", "")
        if isinstance(content, list):
            # Handle multi-modal messages
            text_parts = []
            for part in content:
                if isinstance(part, dict) and part.get("type") == "text":
                    text_parts.append(part.get("text", ""))
            content = " ".join(text_parts)

        return self._sanitize_content(str(content))

    def _patch_chat_completions(self, openai: Any) -> None:
        """Patch chat.completions.create method."""
        instrumentation = self

        try:
            from openai.resources.chat.completions import Completions
            original_create = Completions.create

            @functools.wraps(original_create)
            def traced_create(
                self_client: Any,
                *args: Any,
                **kwargs: Any,
            ) -> Any:
                pyflare = PyFlare.get_instance()
                if pyflare is None or not pyflare.enabled:
                    return original_create(self_client, *args, **kwargs)

                tracer = pyflare.tracer
                model = kwargs.get("model", "unknown")
                is_streaming = kwargs.get("stream", False)

                with tracer.start_as_current_span(
                    "openai.chat.completions.create",
                    kind=SpanKind.CLIENT,
                ) as span:
                    span.set_attribute("pyflare.inference.type", InferenceType.LLM.value)
                    span.set_attribute("pyflare.model.id", model)
                    span.set_attribute("pyflare.model.provider", "openai")
                    span.set_attribute("pyflare.streaming", is_streaming)

                    # Capture input (only if explicitly enabled)
                    if instrumentation._capture_input:
                        messages = kwargs.get("messages", [])
                        span.set_attribute(
                            "pyflare.input.preview",
                            instrumentation._extract_messages_preview(messages),
                        )

                    # Capture parameters (safe to capture)
                    if kwargs.get("temperature") is not None:
                        span.set_attribute("pyflare.params.temperature", kwargs["temperature"])
                    if kwargs.get("max_tokens") is not None:
                        span.set_attribute("pyflare.params.max_tokens", kwargs["max_tokens"])

                    start_time = time.perf_counter()

                    try:
                        result = original_create(self_client, *args, **kwargs)

                        if is_streaming:
                            # Wrap streaming response
                            return StreamingResponseWrapper(
                                result, span, model, instrumentation
                            )

                        # Non-streaming response
                        latency_ms = (time.perf_counter() - start_time) * 1000
                        span.set_attribute("pyflare.latency_ms", latency_ms)

                        # Extract token usage
                        if hasattr(result, "usage") and result.usage:
                            input_tokens = result.usage.prompt_tokens
                            output_tokens = result.usage.completion_tokens
                            total_tokens = result.usage.total_tokens

                            span.set_attribute("pyflare.tokens.input", input_tokens)
                            span.set_attribute("pyflare.tokens.output", output_tokens)
                            span.set_attribute("pyflare.tokens.total", total_tokens)

                            # Calculate cost
                            if instrumentation._calculate_costs:
                                cost = calculate_cost(model, input_tokens, output_tokens)
                                span.set_attribute("pyflare.cost.micros", cost.total_micros)

                        # Extract output (only if explicitly enabled)
                        if instrumentation._capture_output and result.choices:
                            output = result.choices[0].message.content or ""
                            span.set_attribute(
                                "pyflare.output.preview",
                                instrumentation._sanitize_content(output)
                            )
                            span.set_attribute("pyflare.finish_reason", result.choices[0].finish_reason)

                        span.set_status(Status(StatusCode.OK))
                        return result

                    except Exception as e:
                        span.set_status(Status(StatusCode.ERROR, str(e)))
                        span.record_exception(e)
                        raise

            Completions.create = traced_create
            self._original_methods["Completions.create"] = original_create

        except Exception as e:
            # SEC-007: Log exceptions instead of silently swallowing
            logger.warning(
                "Failed to patch chat.completions.create: %s",
                str(e),
                exc_info=True
            )

    def _patch_async_chat_completions(self, openai: Any) -> None:
        """Patch async chat.completions.create method."""
        instrumentation = self

        try:
            from openai.resources.chat.completions import AsyncCompletions
            original_create = AsyncCompletions.create

            @functools.wraps(original_create)
            async def traced_create(
                self_client: Any,
                *args: Any,
                **kwargs: Any,
            ) -> Any:
                pyflare = PyFlare.get_instance()
                if pyflare is None or not pyflare.enabled:
                    return await original_create(self_client, *args, **kwargs)

                tracer = pyflare.tracer
                model = kwargs.get("model", "unknown")
                is_streaming = kwargs.get("stream", False)

                with tracer.start_as_current_span(
                    "openai.chat.completions.create",
                    kind=SpanKind.CLIENT,
                ) as span:
                    span.set_attribute("pyflare.inference.type", InferenceType.LLM.value)
                    span.set_attribute("pyflare.model.id", model)
                    span.set_attribute("pyflare.model.provider", "openai")
                    span.set_attribute("pyflare.async", True)
                    span.set_attribute("pyflare.streaming", is_streaming)

                    # Capture input
                    if instrumentation._capture_input:
                        messages = kwargs.get("messages", [])
                        span.set_attribute(
                            "pyflare.input.preview",
                            instrumentation._extract_messages_preview(messages),
                        )

                    start_time = time.perf_counter()

                    try:
                        result = await original_create(self_client, *args, **kwargs)

                        if is_streaming:
                            return AsyncStreamingResponseWrapper(
                                result, span, model, instrumentation
                            )

                        latency_ms = (time.perf_counter() - start_time) * 1000
                        span.set_attribute("pyflare.latency_ms", latency_ms)

                        # Extract token usage
                        if hasattr(result, "usage") and result.usage:
                            input_tokens = result.usage.prompt_tokens
                            output_tokens = result.usage.completion_tokens

                            span.set_attribute("pyflare.tokens.input", input_tokens)
                            span.set_attribute("pyflare.tokens.output", output_tokens)
                            span.set_attribute("pyflare.tokens.total", result.usage.total_tokens)

                            if instrumentation._calculate_costs:
                                cost = calculate_cost(model, input_tokens, output_tokens)
                                span.set_attribute("pyflare.cost.micros", cost.total_micros)

                        # Extract output
                        if instrumentation._capture_output and result.choices:
                            output = result.choices[0].message.content or ""
                            span.set_attribute(
                                "pyflare.output.preview",
                                instrumentation._sanitize_content(output)
                            )

                        span.set_status(Status(StatusCode.OK))
                        return result

                    except Exception as e:
                        span.set_status(Status(StatusCode.ERROR, str(e)))
                        span.record_exception(e)
                        raise

            AsyncCompletions.create = traced_create
            self._original_methods["AsyncCompletions.create"] = original_create

        except Exception as e:
            logger.warning(
                "Failed to patch async chat.completions.create: %s",
                str(e),
                exc_info=True
            )

    def _patch_embeddings(self, openai: Any) -> None:
        """Patch embeddings.create method."""
        instrumentation = self

        try:
            from openai.resources.embeddings import Embeddings
            original_create = Embeddings.create

            @functools.wraps(original_create)
            def traced_create(
                self_client: Any,
                *args: Any,
                **kwargs: Any,
            ) -> Any:
                pyflare = PyFlare.get_instance()
                if pyflare is None or not pyflare.enabled:
                    return original_create(self_client, *args, **kwargs)

                tracer = pyflare.tracer
                model = kwargs.get("model", "unknown")

                with tracer.start_as_current_span(
                    "openai.embeddings.create",
                    kind=SpanKind.CLIENT,
                ) as span:
                    span.set_attribute("pyflare.inference.type", InferenceType.EMBEDDING.value)
                    span.set_attribute("pyflare.model.id", model)
                    span.set_attribute("pyflare.model.provider", "openai")

                    # Capture input count (safe metadata, not content)
                    input_data = kwargs.get("input", [])
                    if isinstance(input_data, str):
                        input_count = 1
                    else:
                        input_count = len(input_data)
                    span.set_attribute("pyflare.embedding.input_count", input_count)

                    start_time = time.perf_counter()

                    try:
                        result = original_create(self_client, *args, **kwargs)

                        latency_ms = (time.perf_counter() - start_time) * 1000
                        span.set_attribute("pyflare.latency_ms", latency_ms)

                        # Extract usage
                        if hasattr(result, "usage") and result.usage:
                            span.set_attribute("pyflare.tokens.input", result.usage.prompt_tokens)
                            span.set_attribute("pyflare.tokens.total", result.usage.total_tokens)

                            if instrumentation._calculate_costs:
                                cost = calculate_cost(model, result.usage.prompt_tokens, 0)
                                span.set_attribute("pyflare.cost.micros", cost.total_micros)

                        # Capture output dimensions (safe metadata)
                        if result.data:
                            span.set_attribute("pyflare.embedding.dimensions", len(result.data[0].embedding))
                            span.set_attribute("pyflare.embedding.count", len(result.data))

                        span.set_status(Status(StatusCode.OK))
                        return result

                    except Exception as e:
                        span.set_status(Status(StatusCode.ERROR, str(e)))
                        span.record_exception(e)
                        raise

            Embeddings.create = traced_create
            self._original_methods["Embeddings.create"] = original_create

        except Exception as e:
            logger.warning(
                "Failed to patch embeddings.create: %s",
                str(e),
                exc_info=True
            )

    def _patch_async_embeddings(self, openai: Any) -> None:
        """Patch async embeddings.create method."""
        instrumentation = self

        try:
            from openai.resources.embeddings import AsyncEmbeddings
            original_create = AsyncEmbeddings.create

            @functools.wraps(original_create)
            async def traced_create(
                self_client: Any,
                *args: Any,
                **kwargs: Any,
            ) -> Any:
                pyflare = PyFlare.get_instance()
                if pyflare is None or not pyflare.enabled:
                    return await original_create(self_client, *args, **kwargs)

                tracer = pyflare.tracer
                model = kwargs.get("model", "unknown")

                with tracer.start_as_current_span(
                    "openai.embeddings.create",
                    kind=SpanKind.CLIENT,
                ) as span:
                    span.set_attribute("pyflare.inference.type", InferenceType.EMBEDDING.value)
                    span.set_attribute("pyflare.model.id", model)
                    span.set_attribute("pyflare.model.provider", "openai")
                    span.set_attribute("pyflare.async", True)

                    start_time = time.perf_counter()

                    try:
                        result = await original_create(self_client, *args, **kwargs)

                        latency_ms = (time.perf_counter() - start_time) * 1000
                        span.set_attribute("pyflare.latency_ms", latency_ms)

                        if hasattr(result, "usage") and result.usage:
                            span.set_attribute("pyflare.tokens.input", result.usage.prompt_tokens)
                            span.set_attribute("pyflare.tokens.total", result.usage.total_tokens)

                            if instrumentation._calculate_costs:
                                cost = calculate_cost(model, result.usage.prompt_tokens, 0)
                                span.set_attribute("pyflare.cost.micros", cost.total_micros)

                        span.set_status(Status(StatusCode.OK))
                        return result

                    except Exception as e:
                        span.set_status(Status(StatusCode.ERROR, str(e)))
                        span.record_exception(e)
                        raise

            AsyncEmbeddings.create = traced_create
            self._original_methods["AsyncEmbeddings.create"] = original_create

        except Exception as e:
            logger.warning(
                "Failed to patch async embeddings.create: %s",
                str(e),
                exc_info=True
            )


class StreamingResponseWrapper:
    """Wrapper for streaming responses that tracks tokens."""

    def __init__(
        self,
        response: Iterator[Any],
        span: Any,
        model: str,
        instrumentation: OpenAIInstrumentation,
    ) -> None:
        self._response = response
        self._span = span
        self._model = model
        self._instrumentation = instrumentation
        self._chunks: list[str] = []
        self._finished = False

    def __iter__(self) -> Iterator[Any]:
        return self

    def __next__(self) -> Any:
        try:
            chunk = next(self._response)

            # Collect output chunks (only if capture enabled)
            if self._instrumentation._capture_output:
                if chunk.choices and chunk.choices[0].delta.content:
                    self._chunks.append(chunk.choices[0].delta.content)

            # Check if stream is finished
            if chunk.choices and chunk.choices[0].finish_reason:
                self._finish_span(chunk.choices[0].finish_reason)

            return chunk

        except StopIteration:
            if not self._finished:
                self._finish_span("stop")
            raise

    def _finish_span(self, finish_reason: str) -> None:
        """Finish the span with collected data."""
        if self._finished:
            return
        self._finished = True

        # Set output if captured
        if self._instrumentation._capture_output:
            output = "".join(self._chunks)
            self._span.set_attribute(
                "pyflare.output.preview",
                self._instrumentation._sanitize_content(output)
            )

        self._span.set_attribute("pyflare.finish_reason", finish_reason)
        self._span.set_status(Status(StatusCode.OK))


class AsyncStreamingResponseWrapper:
    """Wrapper for async streaming responses that tracks tokens."""

    def __init__(
        self,
        response: AsyncIterator[Any],
        span: Any,
        model: str,
        instrumentation: OpenAIInstrumentation,
    ) -> None:
        self._response = response
        self._span = span
        self._model = model
        self._instrumentation = instrumentation
        self._chunks: list[str] = []
        self._finished = False

    def __aiter__(self) -> AsyncIterator[Any]:
        return self

    async def __anext__(self) -> Any:
        try:
            chunk = await self._response.__anext__()

            if self._instrumentation._capture_output:
                if chunk.choices and chunk.choices[0].delta.content:
                    self._chunks.append(chunk.choices[0].delta.content)

            if chunk.choices and chunk.choices[0].finish_reason:
                self._finish_span(chunk.choices[0].finish_reason)

            return chunk

        except StopAsyncIteration:
            if not self._finished:
                self._finish_span("stop")
            raise

    def _finish_span(self, finish_reason: str) -> None:
        """Finish the span with collected data."""
        if self._finished:
            return
        self._finished = True

        if self._instrumentation._capture_output:
            output = "".join(self._chunks)
            self._span.set_attribute(
                "pyflare.output.preview",
                self._instrumentation._sanitize_content(output)
            )

        self._span.set_attribute("pyflare.finish_reason", finish_reason)
        self._span.set_status(Status(StatusCode.OK))


def trace_openai_call(
    model: Optional[str] = None,
    capture_input: bool = False,  # SEC-005: Default to False
    capture_output: bool = False,  # SEC-005: Default to False
    scrub_pii: bool = True,
):
    """
    Decorator for tracing OpenAI calls.

    Security Note:
        By default, input/output capture is disabled. Enable with caution.

    Example:
        @trace_openai_call(model="gpt-4")
        def ask_gpt(prompt: str) -> str:
            response = client.chat.completions.create(
                model="gpt-4",
                messages=[{"role": "user", "content": prompt}]
            )
            return response.choices[0].message.content
    """
    from pyflare.decorators import trace

    return trace(
        model_id=model,
        inference_type=InferenceType.LLM,
        capture_input=capture_input,
        capture_output=capture_output,
        attributes={"provider": "openai"},
    )
