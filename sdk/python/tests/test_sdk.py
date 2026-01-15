"""Tests for PyFlare SDK core functionality."""

import pytest

from pyflare import PyFlare, trace, init, get_pyflare
from pyflare.types import InferenceType, SpanAttributes, TokenUsage


class TestPyFlare:
    """Tests for PyFlare class."""

    def test_initialization(self) -> None:
        """Test basic initialization."""
        pyflare = PyFlare(
            service_name="test-service",
            endpoint="http://localhost:4317",
            enabled=False,  # Disable for testing
        )

        assert pyflare.service_name == "test-service"
        assert pyflare.endpoint == "http://localhost:4317"
        assert not pyflare.enabled

    def test_initialization_with_all_options(self) -> None:
        """Test initialization with all options."""
        pyflare = PyFlare(
            service_name="test-service",
            endpoint="http://localhost:4317",
            environment="production",
            version="1.0.0",
            enabled=False,
            sample_rate=0.5,
            debug=False,
            resource_attributes={"custom.attr": "value"},
        )

        assert pyflare.environment == "production"
        assert pyflare.version == "1.0.0"
        assert pyflare.sample_rate == 0.5

    def test_singleton_instance(self) -> None:
        """Test that get_instance returns the last created instance."""
        pyflare1 = PyFlare(service_name="service-1", enabled=False)
        pyflare2 = PyFlare(service_name="service-2", enabled=False)

        assert PyFlare.get_instance() == pyflare2

    def test_disabled_mode(self) -> None:
        """Test that disabled mode doesn't create traces."""
        pyflare = PyFlare(service_name="test", enabled=False)

        @trace(name="test_func")
        def my_func() -> str:
            return "result"

        result = my_func()
        assert result == "result"

    def test_init_convenience_function(self) -> None:
        """Test the init convenience function."""
        pyflare = init("my-service", enabled=False)

        assert pyflare.service_name == "my-service"
        assert get_pyflare() == pyflare

    def test_context_vars(self) -> None:
        """Test context variable setting."""
        pyflare = PyFlare(service_name="test", enabled=False)

        pyflare.set_user("user-123", {"plan": "premium"})
        pyflare.set_feature("feature-abc")
        pyflare.set_session("session-xyz")

        assert pyflare.get_context_var("user_id") == "user-123"
        assert pyflare.get_context_var("feature_id") == "feature-abc"
        assert pyflare.get_context_var("session_id") == "session-xyz"


class TestDecorators:
    """Tests for trace decorators."""

    def test_trace_decorator(self) -> None:
        """Test basic trace decorator."""
        PyFlare(service_name="test", enabled=False)

        @trace(name="my_operation", model_id="test-model")
        def my_operation(x: int) -> int:
            return x * 2

        result = my_operation(5)
        assert result == 10

    def test_trace_with_exception(self) -> None:
        """Test that exceptions are propagated."""
        PyFlare(service_name="test", enabled=False)

        @trace(name="failing_func")
        def failing_func() -> None:
            raise ValueError("Test error")

        with pytest.raises(ValueError, match="Test error"):
            failing_func()

    def test_trace_with_inference_type(self) -> None:
        """Test trace decorator with inference type."""
        PyFlare(service_name="test", enabled=False)

        @trace(name="llm_call", model_id="gpt-4", inference_type=InferenceType.LLM)
        def llm_call(prompt: str) -> str:
            return "response"

        result = llm_call("hello")
        assert result == "response"

    def test_trace_without_capture(self) -> None:
        """Test trace decorator without input/output capture."""
        PyFlare(service_name="test", enabled=False)

        @trace(name="sensitive_func", capture_input=False, capture_output=False)
        def sensitive_func(secret: str) -> str:
            return secret.upper()

        result = sensitive_func("password")
        assert result == "PASSWORD"

    @pytest.mark.asyncio
    async def test_async_trace(self) -> None:
        """Test tracing async functions."""
        PyFlare(service_name="test", enabled=False)

        @trace(name="async_operation")
        async def async_operation(x: int) -> int:
            return x * 2

        result = await async_operation(5)
        assert result == 10


class TestTypes:
    """Tests for type definitions."""

    def test_token_usage(self) -> None:
        """Test TokenUsage model."""
        usage = TokenUsage(input_tokens=100, output_tokens=50)
        assert usage.input_tokens == 100
        assert usage.output_tokens == 50

    def test_token_usage_total(self) -> None:
        """Test TokenUsage total tokens."""
        usage = TokenUsage(input_tokens=100, output_tokens=50, total_tokens=150)
        assert usage.total_tokens == 150

    def test_span_attributes_to_otel(self) -> None:
        """Test SpanAttributes conversion to OTel format."""
        attrs = SpanAttributes(
            model_id="gpt-4",
            model_provider="openai",
            input_tokens=100,
            custom={"key": "value"},
        )

        otel_attrs = attrs.to_otel_attributes()

        assert otel_attrs["pyflare.model.id"] == "gpt-4"
        assert otel_attrs["pyflare.model.provider"] == "openai"
        assert otel_attrs["pyflare.tokens.input"] == 100
        assert otel_attrs["pyflare.custom.key"] == "value"

    def test_span_attributes_with_all_fields(self) -> None:
        """Test SpanAttributes with all fields."""
        attrs = SpanAttributes(
            model_id="claude-3-opus",
            model_version="20240229",
            model_provider="anthropic",
            inference_type=InferenceType.LLM,
            input_preview="Hello, world",
            output_preview="Hi there!",
            input_tokens=10,
            output_tokens=5,
            total_tokens=15,
            cost_micros=100,
            user_id="user-123",
            feature_id="feature-abc",
        )

        otel_attrs = attrs.to_otel_attributes()

        assert "pyflare.model.version" in otel_attrs
        assert "pyflare.inference.type" in otel_attrs
        assert "pyflare.cost.micros" in otel_attrs

    def test_inference_type_enum(self) -> None:
        """Test InferenceType enum values."""
        assert InferenceType.LLM.value == "llm"
        assert InferenceType.EMBEDDING.value == "embedding"
        assert InferenceType.CLASSIFICATION.value == "classification"
        assert InferenceType.REGRESSION.value == "regression"
        assert InferenceType.CUSTOM.value == "custom"


class TestCostCalculation:
    """Tests for cost calculation."""

    def test_calculate_cost(self) -> None:
        """Test basic cost calculation."""
        from pyflare.cost import calculate_cost

        result = calculate_cost(
            model_id="gpt-4o",
            input_tokens=1000,
            output_tokens=500,
        )

        assert result.total_micros > 0
        assert result.total_dollars > 0
        assert not result.estimated

    def test_calculate_cost_unknown_model(self) -> None:
        """Test cost calculation for unknown model."""
        from pyflare.cost import calculate_cost

        result = calculate_cost(
            model_id="unknown-model",
            input_tokens=1000,
            output_tokens=500,
        )

        assert result.total_micros == 0
        assert result.estimated

    def test_cost_calculator_custom_pricing(self) -> None:
        """Test cost calculator with custom pricing."""
        from pyflare.cost import CostCalculator, ModelPricing, ModelProvider

        calculator = CostCalculator()
        calculator.add_pricing(ModelPricing(
            model_id="custom-model",
            provider=ModelProvider.CUSTOM,
            input_price_per_million=1.0,
            output_price_per_million=2.0,
        ))

        result = calculator.calculate(
            model_id="custom-model",
            token_usage=TokenUsage(
                input_tokens=1000000,
                output_tokens=1000000,
                total_tokens=2000000,
            ),
        )

        # 1M input tokens at $1 + 1M output tokens at $2 = $3
        assert result.total_dollars == pytest.approx(3.0, rel=0.01)

    def test_model_pricing_lookup(self) -> None:
        """Test model pricing lookup with prefix matching."""
        from pyflare.cost import get_cost_calculator

        calculator = get_cost_calculator()

        # Exact match
        pricing = calculator.get_pricing("gpt-4o")
        assert pricing is not None
        assert pricing.model_id == "gpt-4o"

        # Prefix match (e.g., gpt-4o-2024-08-06)
        pricing = calculator.get_pricing("gpt-4o-2024-08-06")
        assert pricing is not None


class TestContextManager:
    """Tests for context manager usage."""

    def test_span_context_manager(self) -> None:
        """Test span context manager."""
        pyflare = PyFlare(service_name="test", enabled=False)

        result = None
        with pyflare.span("test-span") as span:
            result = 42

        assert result == 42

    def test_llm_span_context_manager(self) -> None:
        """Test LLM span context manager."""
        pyflare = PyFlare(service_name="test", enabled=False)

        result = None
        with pyflare.llm_span("chat", model_id="gpt-4", provider="openai") as span:
            result = "response"

        assert result == "response"

    def test_span_with_exception(self) -> None:
        """Test that exceptions propagate through context manager."""
        pyflare = PyFlare(service_name="test", enabled=False)

        with pytest.raises(ValueError, match="test error"):
            with pyflare.span("test-span"):
                raise ValueError("test error")


class TestIntegrations:
    """Tests for framework integrations."""

    def test_openai_instrumentation_import(self) -> None:
        """Test that OpenAI instrumentation can be imported."""
        from pyflare.integrations import OpenAIInstrumentation

        instrumentation = OpenAIInstrumentation()
        assert instrumentation is not None

    def test_anthropic_instrumentation_import(self) -> None:
        """Test that Anthropic instrumentation can be imported."""
        from pyflare.integrations import AnthropicInstrumentation

        instrumentation = AnthropicInstrumentation()
        assert instrumentation is not None
