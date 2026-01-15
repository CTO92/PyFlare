"""LangChain integration for PyFlare."""

from typing import Any, Dict, List, Optional, Sequence, Union
import time
from uuid import UUID

from opentelemetry import trace
from opentelemetry.trace import SpanKind, Status, StatusCode

from pyflare.sdk import PyFlare
from pyflare.types import InferenceType
from pyflare.cost import calculate_cost


class LangChainInstrumentation:
    """
    Automatic instrumentation for LangChain.

    Traces:
    - Chain invocations (LCEL and legacy)
    - LLM calls with token tracking
    - Tool/Agent executions
    - Retriever operations
    - Embedding generation

    Example:
        from pyflare.integrations import LangChainInstrumentation

        # Initialize PyFlare
        pyflare = PyFlare(service_name="my-rag-app")

        # Enable LangChain instrumentation
        LangChainInstrumentation().instrument()

        # Now all LangChain operations are traced
        chain.invoke({"question": "What is PyFlare?"})
    """

    def __init__(
        self,
        capture_input: bool = True,
        capture_output: bool = True,
        calculate_costs: bool = True,
    ) -> None:
        """
        Initialize LangChain instrumentation.

        Args:
            capture_input: Whether to capture input data
            capture_output: Whether to capture output data
            calculate_costs: Whether to calculate and track LLM costs
        """
        self._instrumented = False
        self._capture_input = capture_input
        self._capture_output = capture_output
        self._calculate_costs = calculate_costs
        self._callbacks: list[Any] = []

    def instrument(self) -> None:
        """Enable LangChain instrumentation."""
        if self._instrumented:
            return

        try:
            from langchain_core.callbacks import BaseCallbackHandler
        except ImportError:
            raise ImportError(
                "langchain-core package is required for LangChain instrumentation. "
                "Install with: pip install -e '.[langchain]' from the sdk/python directory"
            )

        # Create and register callback handler
        handler = PyFlareCallbackHandler(
            capture_input=self._capture_input,
            capture_output=self._capture_output,
            calculate_costs=self._calculate_costs,
        )
        self._callbacks.append(handler)

        self._instrumented = True

    def uninstrument(self) -> None:
        """Disable LangChain instrumentation."""
        self._callbacks.clear()
        self._instrumented = False

    def get_callback_handler(self) -> Any:
        """
        Get a callback handler for manual registration.

        Example:
            handler = instrumentation.get_callback_handler()
            chain.invoke({"question": "..."}, config={"callbacks": [handler]})
        """
        return PyFlareCallbackHandler(
            capture_input=self._capture_input,
            capture_output=self._capture_output,
            calculate_costs=self._calculate_costs,
        )


class PyFlareCallbackHandler:
    """
    LangChain callback handler for PyFlare tracing.

    Can be used directly with LangChain:
        from pyflare.integrations.langchain import PyFlareCallbackHandler

        handler = PyFlareCallbackHandler()
        chain.invoke({"question": "..."}, config={"callbacks": [handler]})
    """

    def __init__(
        self,
        capture_input: bool = True,
        capture_output: bool = True,
        calculate_costs: bool = True,
    ) -> None:
        self._capture_input = capture_input
        self._capture_output = capture_output
        self._calculate_costs = calculate_costs
        self._spans: Dict[str, Any] = {}
        self._span_start_times: Dict[str, float] = {}

    def _get_tracer(self) -> Optional[Any]:
        """Get the PyFlare tracer if available."""
        pyflare = PyFlare.get_instance()
        if pyflare is None or not pyflare.enabled:
            return None
        return pyflare.tracer

    # Chain callbacks
    def on_chain_start(
        self,
        serialized: Dict[str, Any],
        inputs: Dict[str, Any],
        *,
        run_id: UUID,
        parent_run_id: Optional[UUID] = None,
        tags: Optional[List[str]] = None,
        metadata: Optional[Dict[str, Any]] = None,
        **kwargs: Any,
    ) -> None:
        """Called when chain starts."""
        tracer = self._get_tracer()
        if tracer is None:
            return

        chain_name = serialized.get("name", serialized.get("id", ["chain"])[-1])
        run_id_str = str(run_id)

        span = tracer.start_span(
            f"langchain.chain.{chain_name}",
            kind=SpanKind.INTERNAL,
        )
        span.set_attribute("pyflare.langchain.run_id", run_id_str)
        span.set_attribute("pyflare.langchain.chain_name", chain_name)
        span.set_attribute("pyflare.langchain.component", "chain")

        if parent_run_id:
            span.set_attribute("pyflare.langchain.parent_run_id", str(parent_run_id))

        if tags:
            span.set_attribute("pyflare.langchain.tags", ",".join(tags))

        if metadata:
            for key, value in metadata.items():
                if isinstance(value, (str, int, float, bool)):
                    span.set_attribute(f"pyflare.metadata.{key}", value)

        # Capture input
        if self._capture_input:
            input_str = str(inputs)[:1000]
            span.set_attribute("pyflare.input.preview", input_str)

        self._spans[run_id_str] = span
        self._span_start_times[run_id_str] = time.perf_counter()

    def on_chain_end(
        self,
        outputs: Dict[str, Any],
        *,
        run_id: UUID,
        **kwargs: Any,
    ) -> None:
        """Called when chain ends."""
        run_id_str = str(run_id)
        span = self._spans.pop(run_id_str, None)
        start_time = self._span_start_times.pop(run_id_str, None)

        if span is None:
            return

        # Calculate latency
        if start_time:
            latency_ms = (time.perf_counter() - start_time) * 1000
            span.set_attribute("pyflare.latency_ms", latency_ms)

        # Capture output
        if self._capture_output:
            output_str = str(outputs)[:1000]
            span.set_attribute("pyflare.output.preview", output_str)

        span.set_status(Status(StatusCode.OK))
        span.end()

    def on_chain_error(
        self,
        error: BaseException,
        *,
        run_id: UUID,
        **kwargs: Any,
    ) -> None:
        """Called when chain errors."""
        run_id_str = str(run_id)
        span = self._spans.pop(run_id_str, None)
        self._span_start_times.pop(run_id_str, None)

        if span is None:
            return

        span.set_status(Status(StatusCode.ERROR, str(error)))
        span.record_exception(error)
        span.end()

    # LLM callbacks
    def on_llm_start(
        self,
        serialized: Dict[str, Any],
        prompts: List[str],
        *,
        run_id: UUID,
        parent_run_id: Optional[UUID] = None,
        tags: Optional[List[str]] = None,
        metadata: Optional[Dict[str, Any]] = None,
        **kwargs: Any,
    ) -> None:
        """Called when LLM starts."""
        tracer = self._get_tracer()
        if tracer is None:
            return

        run_id_str = str(run_id)
        model_name = serialized.get("name", "llm")

        # Try to extract model ID from kwargs or metadata
        model_id = kwargs.get("invocation_params", {}).get("model_name", model_name)
        if metadata and "model" in metadata:
            model_id = metadata["model"]

        span = tracer.start_span(
            f"langchain.llm.{model_name}",
            kind=SpanKind.CLIENT,
        )
        span.set_attribute("pyflare.inference.type", InferenceType.LLM.value)
        span.set_attribute("pyflare.langchain.run_id", run_id_str)
        span.set_attribute("pyflare.langchain.component", "llm")
        span.set_attribute("pyflare.model.id", model_id)

        if parent_run_id:
            span.set_attribute("pyflare.langchain.parent_run_id", str(parent_run_id))

        # Capture input
        if self._capture_input and prompts:
            span.set_attribute("pyflare.input.preview", prompts[0][:1000])
            span.set_attribute("pyflare.input.prompt_count", len(prompts))

        self._spans[run_id_str] = span
        self._span_start_times[run_id_str] = time.perf_counter()

    def on_llm_new_token(
        self,
        token: str,
        *,
        run_id: UUID,
        **kwargs: Any,
    ) -> None:
        """Called for each new token during streaming."""
        # Can be used to track streaming progress
        pass

    def on_llm_end(
        self,
        response: Any,
        *,
        run_id: UUID,
        **kwargs: Any,
    ) -> None:
        """Called when LLM ends."""
        run_id_str = str(run_id)
        span = self._spans.pop(run_id_str, None)
        start_time = self._span_start_times.pop(run_id_str, None)

        if span is None:
            return

        # Calculate latency
        if start_time:
            latency_ms = (time.perf_counter() - start_time) * 1000
            span.set_attribute("pyflare.latency_ms", latency_ms)

        # Extract output
        if self._capture_output and hasattr(response, "generations") and response.generations:
            if response.generations[0]:
                output = response.generations[0][0].text
                span.set_attribute("pyflare.output.preview", output[:1000])

        # Extract token usage if available
        if hasattr(response, "llm_output") and response.llm_output:
            usage = response.llm_output.get("token_usage", {})
            if usage:
                input_tokens = usage.get("prompt_tokens", 0)
                output_tokens = usage.get("completion_tokens", 0)
                total_tokens = usage.get("total_tokens", 0)

                span.set_attribute("pyflare.tokens.input", input_tokens)
                span.set_attribute("pyflare.tokens.output", output_tokens)
                span.set_attribute("pyflare.tokens.total", total_tokens)

                # Calculate cost if model info available
                if self._calculate_costs:
                    model_name = response.llm_output.get("model_name", "")
                    if model_name and input_tokens > 0:
                        try:
                            cost = calculate_cost(model_name, input_tokens, output_tokens)
                            span.set_attribute("pyflare.cost.micros", cost.total_micros)
                        except Exception:
                            pass

        span.set_status(Status(StatusCode.OK))
        span.end()

    def on_llm_error(
        self,
        error: BaseException,
        *,
        run_id: UUID,
        **kwargs: Any,
    ) -> None:
        """Called when LLM errors."""
        run_id_str = str(run_id)
        span = self._spans.pop(run_id_str, None)
        self._span_start_times.pop(run_id_str, None)

        if span is None:
            return

        span.set_status(Status(StatusCode.ERROR, str(error)))
        span.record_exception(error)
        span.end()

    # Chat model callbacks
    def on_chat_model_start(
        self,
        serialized: Dict[str, Any],
        messages: List[List[Any]],
        *,
        run_id: UUID,
        parent_run_id: Optional[UUID] = None,
        tags: Optional[List[str]] = None,
        metadata: Optional[Dict[str, Any]] = None,
        **kwargs: Any,
    ) -> None:
        """Called when chat model starts."""
        tracer = self._get_tracer()
        if tracer is None:
            return

        run_id_str = str(run_id)
        model_name = serialized.get("name", "chat_model")

        # Extract model ID from invocation params
        invocation_params = kwargs.get("invocation_params", {})
        model_id = invocation_params.get("model_name", invocation_params.get("model", model_name))

        span = tracer.start_span(
            f"langchain.chat.{model_name}",
            kind=SpanKind.CLIENT,
        )
        span.set_attribute("pyflare.inference.type", InferenceType.LLM.value)
        span.set_attribute("pyflare.langchain.run_id", run_id_str)
        span.set_attribute("pyflare.langchain.component", "chat_model")
        span.set_attribute("pyflare.model.id", model_id)

        if parent_run_id:
            span.set_attribute("pyflare.langchain.parent_run_id", str(parent_run_id))

        # Capture input (last message)
        if self._capture_input and messages and messages[0]:
            last_msg = messages[0][-1] if messages[0] else None
            if last_msg and hasattr(last_msg, "content"):
                span.set_attribute("pyflare.input.preview", str(last_msg.content)[:1000])
            span.set_attribute("pyflare.input.message_count", len(messages[0]))

        self._spans[run_id_str] = span
        self._span_start_times[run_id_str] = time.perf_counter()

    # Retriever callbacks
    def on_retriever_start(
        self,
        serialized: Dict[str, Any],
        query: str,
        *,
        run_id: UUID,
        parent_run_id: Optional[UUID] = None,
        tags: Optional[List[str]] = None,
        metadata: Optional[Dict[str, Any]] = None,
        **kwargs: Any,
    ) -> None:
        """Called when retriever starts."""
        tracer = self._get_tracer()
        if tracer is None:
            return

        run_id_str = str(run_id)
        retriever_name = serialized.get("name", "retriever")

        span = tracer.start_span(
            f"langchain.retriever.{retriever_name}",
            kind=SpanKind.CLIENT,
        )
        span.set_attribute("pyflare.langchain.run_id", run_id_str)
        span.set_attribute("pyflare.langchain.component", "retriever")

        if parent_run_id:
            span.set_attribute("pyflare.langchain.parent_run_id", str(parent_run_id))

        if self._capture_input:
            span.set_attribute("pyflare.input.preview", query[:1000])

        self._spans[run_id_str] = span
        self._span_start_times[run_id_str] = time.perf_counter()

    def on_retriever_end(
        self,
        documents: Sequence[Any],
        *,
        run_id: UUID,
        **kwargs: Any,
    ) -> None:
        """Called when retriever ends."""
        run_id_str = str(run_id)
        span = self._spans.pop(run_id_str, None)
        start_time = self._span_start_times.pop(run_id_str, None)

        if span is None:
            return

        if start_time:
            latency_ms = (time.perf_counter() - start_time) * 1000
            span.set_attribute("pyflare.latency_ms", latency_ms)

        span.set_attribute("pyflare.retriever.document_count", len(documents))

        # Capture document previews
        if self._capture_output and documents:
            doc_previews = []
            for doc in documents[:3]:  # First 3 docs
                if hasattr(doc, "page_content"):
                    doc_previews.append(doc.page_content[:200])
            if doc_previews:
                span.set_attribute("pyflare.output.preview", " | ".join(doc_previews)[:1000])

        span.set_status(Status(StatusCode.OK))
        span.end()

    def on_retriever_error(
        self,
        error: BaseException,
        *,
        run_id: UUID,
        **kwargs: Any,
    ) -> None:
        """Called when retriever errors."""
        run_id_str = str(run_id)
        span = self._spans.pop(run_id_str, None)
        self._span_start_times.pop(run_id_str, None)

        if span is None:
            return

        span.set_status(Status(StatusCode.ERROR, str(error)))
        span.record_exception(error)
        span.end()

    # Tool callbacks
    def on_tool_start(
        self,
        serialized: Dict[str, Any],
        input_str: str,
        *,
        run_id: UUID,
        parent_run_id: Optional[UUID] = None,
        tags: Optional[List[str]] = None,
        metadata: Optional[Dict[str, Any]] = None,
        **kwargs: Any,
    ) -> None:
        """Called when tool starts."""
        tracer = self._get_tracer()
        if tracer is None:
            return

        run_id_str = str(run_id)
        tool_name = serialized.get("name", "tool")

        span = tracer.start_span(
            f"langchain.tool.{tool_name}",
            kind=SpanKind.INTERNAL,
        )
        span.set_attribute("pyflare.langchain.run_id", run_id_str)
        span.set_attribute("pyflare.langchain.component", "tool")
        span.set_attribute("pyflare.tool.name", tool_name)

        if parent_run_id:
            span.set_attribute("pyflare.langchain.parent_run_id", str(parent_run_id))

        if self._capture_input:
            span.set_attribute("pyflare.input.preview", input_str[:1000])

        self._spans[run_id_str] = span
        self._span_start_times[run_id_str] = time.perf_counter()

    def on_tool_end(
        self,
        output: str,
        *,
        run_id: UUID,
        **kwargs: Any,
    ) -> None:
        """Called when tool ends."""
        run_id_str = str(run_id)
        span = self._spans.pop(run_id_str, None)
        start_time = self._span_start_times.pop(run_id_str, None)

        if span is None:
            return

        if start_time:
            latency_ms = (time.perf_counter() - start_time) * 1000
            span.set_attribute("pyflare.latency_ms", latency_ms)

        if self._capture_output:
            span.set_attribute("pyflare.output.preview", str(output)[:1000])

        span.set_status(Status(StatusCode.OK))
        span.end()

    def on_tool_error(
        self,
        error: BaseException,
        *,
        run_id: UUID,
        **kwargs: Any,
    ) -> None:
        """Called when tool errors."""
        run_id_str = str(run_id)
        span = self._spans.pop(run_id_str, None)
        self._span_start_times.pop(run_id_str, None)

        if span is None:
            return

        span.set_status(Status(StatusCode.ERROR, str(error)))
        span.record_exception(error)
        span.end()

    # Agent callbacks
    def on_agent_action(
        self,
        action: Any,
        *,
        run_id: UUID,
        **kwargs: Any,
    ) -> None:
        """Called when agent takes an action."""
        tracer = self._get_tracer()
        if tracer is None:
            return

        run_id_str = str(run_id)

        # Create a span for the agent action
        if hasattr(action, "tool") and hasattr(action, "tool_input"):
            span = tracer.start_span(
                f"langchain.agent.action.{action.tool}",
                kind=SpanKind.INTERNAL,
            )
            span.set_attribute("pyflare.langchain.run_id", run_id_str)
            span.set_attribute("pyflare.langchain.component", "agent_action")
            span.set_attribute("pyflare.agent.tool", action.tool)

            if self._capture_input:
                span.set_attribute("pyflare.input.preview", str(action.tool_input)[:1000])

            if hasattr(action, "log"):
                span.set_attribute("pyflare.agent.log", action.log[:500])

            span.end()

    def on_agent_finish(
        self,
        finish: Any,
        *,
        run_id: UUID,
        **kwargs: Any,
    ) -> None:
        """Called when agent finishes."""
        tracer = self._get_tracer()
        if tracer is None:
            return

        run_id_str = str(run_id)

        if hasattr(finish, "return_values"):
            span = tracer.start_span(
                "langchain.agent.finish",
                kind=SpanKind.INTERNAL,
            )
            span.set_attribute("pyflare.langchain.run_id", run_id_str)
            span.set_attribute("pyflare.langchain.component", "agent_finish")

            if self._capture_output:
                span.set_attribute("pyflare.output.preview", str(finish.return_values)[:1000])

            if hasattr(finish, "log"):
                span.set_attribute("pyflare.agent.log", finish.log[:500])

            span.end()


def create_callback_handler(
    capture_input: bool = True,
    capture_output: bool = True,
    calculate_costs: bool = True,
) -> PyFlareCallbackHandler:
    """
    Create a PyFlare callback handler for LangChain.

    Example:
        from pyflare.integrations.langchain import create_callback_handler

        handler = create_callback_handler()
        chain.invoke({"question": "..."}, config={"callbacks": [handler]})
    """
    return PyFlareCallbackHandler(
        capture_input=capture_input,
        capture_output=capture_output,
        calculate_costs=calculate_costs,
    )
