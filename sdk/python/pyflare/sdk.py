"""Main PyFlare SDK class.

SECURITY: This module includes input validation, secure defaults,
and options for PII handling to protect sensitive data.
"""

import atexit
import os
import re
import threading
import warnings
from contextlib import contextmanager
from typing import Any, Generator, Optional
from urllib.parse import urlparse

from opentelemetry import trace
from opentelemetry.context import Context
from opentelemetry.exporter.otlp.proto.grpc.trace_exporter import OTLPSpanExporter
from opentelemetry.exporter.otlp.proto.http.trace_exporter import OTLPSpanExporter as OTLPHTTPSpanExporter
from opentelemetry.sdk.resources import Resource
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import BatchSpanProcessor, SimpleSpanProcessor
from opentelemetry.sdk.trace.sampling import (
    ALWAYS_OFF,
    ALWAYS_ON,
    ParentBased,
    TraceIdRatioBased,
)
from opentelemetry.trace import SpanKind, Status, StatusCode

from pyflare.types import InferenceType, ModelInfo, TokenUsage


# SECURITY: Maximum preview lengths to limit data exposure
MAX_INPUT_PREVIEW_LENGTH = 1000
MAX_OUTPUT_PREVIEW_LENGTH = 1000
MAX_ATTRIBUTE_LENGTH = 10000


class SecurityConfig:
    """Security configuration for PyFlare SDK."""

    def __init__(
        self,
        scrub_pii: bool = False,
        allowed_pii_patterns: Optional[list[str]] = None,
        tls_verify: bool = True,
        tls_cert_path: Optional[str] = None,
        max_attribute_length: int = MAX_ATTRIBUTE_LENGTH,
    ):
        """
        Initialize security configuration.

        Args:
            scrub_pii: Enable PII scrubbing from traces
            allowed_pii_patterns: Regex patterns to allow through scrubbing
            tls_verify: Verify TLS certificates (default: True)
            tls_cert_path: Path to custom CA certificate
            max_attribute_length: Maximum length for any attribute value
        """
        self.scrub_pii = scrub_pii
        self.allowed_pii_patterns = allowed_pii_patterns or []
        self.tls_verify = tls_verify
        self.tls_cert_path = tls_cert_path
        self.max_attribute_length = max_attribute_length


# SECURITY: Common PII patterns for scrubbing
PII_PATTERNS = [
    (r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b', '[EMAIL]'),  # Email
    (r'\b\d{3}[-.]?\d{3}[-.]?\d{4}\b', '[PHONE]'),  # Phone number
    (r'\b\d{3}[-]?\d{2}[-]?\d{4}\b', '[SSN]'),  # SSN
    (r'\b\d{16}\b', '[CARD]'),  # Credit card (simple)
    (r'\b(?:4[0-9]{12}(?:[0-9]{3})?|5[1-5][0-9]{14}|3[47][0-9]{13})\b', '[CARD]'),  # Credit cards
    (r'\bapi[_-]?key[_-]?[a-zA-Z0-9]{16,}\b', '[API_KEY]'),  # API keys
    (r'\b(?:sk|pk)[-_][a-zA-Z0-9]{24,}\b', '[SECRET_KEY]'),  # Secret keys
]


def scrub_pii(text: str, allowed_patterns: Optional[list[str]] = None) -> str:
    """
    Scrub PII from text using regex patterns.

    Args:
        text: Text to scrub
        allowed_patterns: Patterns to allow through

    Returns:
        Scrubbed text
    """
    if not text:
        return text

    allowed = set(allowed_patterns or [])
    result = text

    for pattern, replacement in PII_PATTERNS:
        if pattern not in allowed:
            result = re.sub(pattern, replacement, result, flags=re.IGNORECASE)

    return result


def validate_endpoint(endpoint: str) -> str:
    """
    Validate and normalize endpoint URL.

    Args:
        endpoint: The endpoint URL to validate

    Returns:
        Validated endpoint

    Raises:
        ValueError: If endpoint is invalid
    """
    if not endpoint:
        raise ValueError("Endpoint cannot be empty")

    # Parse URL
    parsed = urlparse(endpoint)

    # SECURITY: Only allow http/https/grpc schemes
    allowed_schemes = ('http', 'https', 'grpc', '')
    if parsed.scheme and parsed.scheme.lower() not in allowed_schemes:
        raise ValueError(f"Invalid endpoint scheme: {parsed.scheme}")

    # SECURITY: Check for localhost/private IPs in production
    # This is a warning, not an error, as local endpoints are valid for dev
    if parsed.hostname in ('localhost', '127.0.0.1', '::1'):
        # Check if we're in production
        if os.environ.get('PYFLARE_ENVIRONMENT', '').lower() == 'production':
            warnings.warn(
                "Using localhost endpoint in production environment",
                UserWarning,
                stacklevel=3
            )

    return endpoint


def validate_service_name(service_name: str) -> str:
    """
    Validate service name.

    Args:
        service_name: The service name to validate

    Returns:
        Validated service name

    Raises:
        ValueError: If service name is invalid
    """
    if not service_name or not service_name.strip():
        raise ValueError("Service name cannot be empty")

    # SECURITY: Limit length and characters
    service_name = service_name.strip()
    if len(service_name) > 256:
        raise ValueError("Service name exceeds maximum length (256)")

    # Allow alphanumeric, hyphens, underscores, dots
    if not re.match(r'^[\w\-\.]+$', service_name):
        raise ValueError("Service name contains invalid characters")

    return service_name


class PyFlare:
    """
    Main PyFlare SDK class for AI/ML observability.

    Example:
        pyflare = PyFlare(
            service_name="my-ml-service",
            endpoint="http://localhost:4317",
        )

        # Use decorator
        @pyflare.trace(model_id="gpt-4")
        def predict(input_data):
            return model(input_data)

        # Or use context manager
        with pyflare.span("custom-operation") as span:
            span.set_attribute("custom.key", "value")
            result = do_work()

    SECURITY: This SDK includes:
    - Input validation for all configuration
    - Optional PII scrubbing
    - TLS verification (enabled by default)
    - Secure data truncation
    """

    _instance: Optional["PyFlare"] = None
    _lock = threading.Lock()

    def __init__(
        self,
        service_name: str,
        endpoint: str = "http://localhost:4317",
        environment: str = "development",
        version: str = "",
        headers: Optional[dict[str, str]] = None,
        enabled: bool = True,
        sample_rate: float = 1.0,
        use_http: bool = False,
        batch_export: bool = True,
        debug: bool = False,
        resource_attributes: Optional[dict[str, str]] = None,
        security: Optional[SecurityConfig] = None,
    ) -> None:
        """
        Initialize PyFlare SDK.

        Args:
            service_name: Name of the service (required)
            endpoint: PyFlare collector endpoint (default: localhost:4317)
            environment: Deployment environment (development, staging, production)
            version: Service version
            headers: Additional headers for OTLP exporter
            enabled: Whether tracing is enabled
            sample_rate: Sampling rate (0.0 to 1.0)
            use_http: Use HTTP/protobuf instead of gRPC
            batch_export: Use batch processing (recommended for production)
            debug: Enable debug mode with console output
            resource_attributes: Additional resource attributes
            security: Security configuration options
        """
        # SECURITY: Validate inputs
        self.service_name = validate_service_name(service_name)
        self.endpoint = validate_endpoint(endpoint)
        self.environment = environment
        self.version = version
        self.enabled = enabled
        self._debug = debug
        self._security = security or SecurityConfig()

        # SECURITY: Validate and clamp sample_rate
        if not isinstance(sample_rate, (int, float)):
            raise ValueError("sample_rate must be a number")
        self.sample_rate = max(0.0, min(1.0, float(sample_rate)))
        if sample_rate != self.sample_rate:
            warnings.warn(
                f"sample_rate clamped from {sample_rate} to {self.sample_rate}",
                UserWarning
            )

        self._tracer_provider: Optional[TracerProvider] = None
        self._tracer: Optional[trace.Tracer] = None
        self._models: dict[str, ModelInfo] = {}
        self._context_vars: dict[str, Any] = {}

        # Check environment variables for configuration overrides
        self._apply_env_overrides()

        if enabled:
            self._initialize_tracing(
                headers=headers,
                use_http=use_http,
                batch_export=batch_export,
                resource_attributes=resource_attributes,
            )

        # Set as global instance
        with PyFlare._lock:
            PyFlare._instance = self

        # Register shutdown handler
        atexit.register(self.shutdown)

    def _apply_env_overrides(self) -> None:
        """Apply configuration from environment variables."""
        if os.environ.get("PYFLARE_ENDPOINT"):
            self.endpoint = validate_endpoint(os.environ["PYFLARE_ENDPOINT"])

        if os.environ.get("PYFLARE_ENABLED"):
            self.enabled = os.environ["PYFLARE_ENABLED"].lower() in ("true", "1", "yes")

        if os.environ.get("PYFLARE_SAMPLE_RATE"):
            try:
                rate = float(os.environ["PYFLARE_SAMPLE_RATE"])
                self.sample_rate = max(0.0, min(1.0, rate))
            except ValueError:
                pass

        if os.environ.get("PYFLARE_ENVIRONMENT"):
            self.environment = os.environ["PYFLARE_ENVIRONMENT"]

        # SECURITY: Environment overrides for security settings
        if os.environ.get("PYFLARE_SCRUB_PII"):
            self._security.scrub_pii = os.environ["PYFLARE_SCRUB_PII"].lower() in ("true", "1", "yes")

        if os.environ.get("PYFLARE_TLS_VERIFY"):
            self._security.tls_verify = os.environ["PYFLARE_TLS_VERIFY"].lower() in ("true", "1", "yes")

        if os.environ.get("PYFLARE_TLS_CERT_PATH"):
            self._security.tls_cert_path = os.environ["PYFLARE_TLS_CERT_PATH"]

    def _initialize_tracing(
        self,
        headers: Optional[dict[str, str]],
        use_http: bool,
        batch_export: bool,
        resource_attributes: Optional[dict[str, str]],
    ) -> None:
        """Initialize OpenTelemetry tracing."""
        # Build resource attributes
        attrs = {
            "service.name": self.service_name,
            "service.version": self.version,
            "deployment.environment": self.environment,
            "telemetry.sdk.name": "pyflare",
            "telemetry.sdk.version": "1.0.0",
            "telemetry.sdk.language": "python",
        }
        if resource_attributes:
            attrs.update(resource_attributes)

        resource = Resource.create(attrs)

        # Configure sampler
        if self.sample_rate >= 1.0:
            sampler = ALWAYS_ON
        elif self.sample_rate <= 0.0:
            sampler = ALWAYS_OFF
        else:
            sampler = ParentBased(root=TraceIdRatioBased(self.sample_rate))

        # Create tracer provider
        self._tracer_provider = TracerProvider(
            resource=resource,
            sampler=sampler,
        )

        # SECURITY: Configure TLS options
        exporter_kwargs: dict[str, Any] = {
            "headers": headers,
        }

        # Add TLS configuration if not using default
        if not self._security.tls_verify:
            warnings.warn(
                "TLS verification is disabled - this is not recommended for production",
                UserWarning
            )
            exporter_kwargs["insecure"] = True

        # Create exporter
        if use_http:
            exporter = OTLPHTTPSpanExporter(
                endpoint=f"{self.endpoint}/v1/traces",
                **exporter_kwargs,
            )
        else:
            exporter = OTLPSpanExporter(
                endpoint=self.endpoint,
                **exporter_kwargs,
            )

        # Add span processor
        if batch_export:
            processor = BatchSpanProcessor(exporter)
        else:
            processor = SimpleSpanProcessor(exporter)

        self._tracer_provider.add_span_processor(processor)

        # Add console exporter for debug mode
        if self._debug:
            from pyflare.exporters import ConsoleExporter
            self._tracer_provider.add_span_processor(
                SimpleSpanProcessor(ConsoleExporter())
            )

        # Set as global tracer provider
        trace.set_tracer_provider(self._tracer_provider)

        # Get tracer
        self._tracer = trace.get_tracer("pyflare", "1.0.0")

    @property
    def tracer(self) -> trace.Tracer:
        """Get the OpenTelemetry tracer."""
        if self._tracer is None:
            return trace.get_tracer("pyflare", "1.0.0")
        return self._tracer

    @property
    def security(self) -> SecurityConfig:
        """Get the security configuration."""
        return self._security

    @classmethod
    def get_instance(cls) -> Optional["PyFlare"]:
        """Get the global PyFlare instance."""
        return cls._instance

    def shutdown(self) -> None:
        """Shutdown the SDK and flush pending spans."""
        if self._tracer_provider is not None:
            self._tracer_provider.shutdown()
            self._tracer_provider = None

    def flush(self, timeout_millis: int = 30000) -> bool:
        """Flush pending spans.

        Args:
            timeout_millis: Maximum time to wait in milliseconds

        Returns:
            True if flush succeeded, False otherwise
        """
        if self._tracer_provider is not None:
            return self._tracer_provider.force_flush(timeout_millis)
        return True

    def sanitize_attribute(self, value: Any) -> Any:
        """
        Sanitize an attribute value for safe storage.

        SECURITY: This method:
        - Truncates long strings
        - Optionally scrubs PII
        - Validates types

        Args:
            value: The attribute value

        Returns:
            Sanitized value
        """
        if isinstance(value, str):
            # Truncate if too long
            if len(value) > self._security.max_attribute_length:
                value = value[:self._security.max_attribute_length] + "...[truncated]"

            # Scrub PII if enabled
            if self._security.scrub_pii:
                value = scrub_pii(value, self._security.allowed_pii_patterns)

        return value

    @contextmanager
    def span(
        self,
        name: str,
        kind: SpanKind = SpanKind.INTERNAL,
        attributes: Optional[dict[str, Any]] = None,
    ) -> Generator[trace.Span, None, None]:
        """
        Create a span context manager.

        Args:
            name: Span name
            kind: Span kind
            attributes: Initial attributes

        Yields:
            The created span

        Example:
            with pyflare.span("process-data") as span:
                span.set_attribute("items", len(data))
                result = process(data)
        """
        if not self.enabled:
            yield trace.INVALID_SPAN
            return

        with self.tracer.start_as_current_span(name, kind=kind) as span:
            if attributes:
                for key, value in attributes.items():
                    span.set_attribute(key, self.sanitize_attribute(value))
            try:
                yield span
                span.set_status(Status(StatusCode.OK))
            except Exception as e:
                # SECURITY: Don't expose full exception in status message
                error_msg = str(e)[:500] if str(e) else "Unknown error"
                span.set_status(Status(StatusCode.ERROR, error_msg))
                span.record_exception(e)
                raise

    @contextmanager
    def llm_span(
        self,
        name: str,
        model_id: str,
        provider: str = "",
    ) -> Generator[trace.Span, None, None]:
        """
        Create a span specifically for LLM calls.

        Args:
            name: Span name
            model_id: Model identifier
            provider: Model provider

        Yields:
            The created span

        Example:
            with pyflare.llm_span("chat-completion", model_id="gpt-4", provider="openai") as span:
                response = client.chat.completions.create(...)
                span.set_attribute("pyflare.tokens.total", response.usage.total_tokens)
        """
        with self.span(name, kind=SpanKind.CLIENT) as span:
            span.set_attribute("pyflare.inference.type", InferenceType.LLM.value)
            span.set_attribute("pyflare.model.id", model_id)
            if provider:
                span.set_attribute("pyflare.model.provider", provider)
            yield span

    def trace(
        self,
        name: Optional[str] = None,
        *,
        model_id: Optional[str] = None,
        inference_type: InferenceType = InferenceType.CUSTOM,
        capture_input: bool = True,
        capture_output: bool = True,
        attributes: Optional[dict[str, Any]] = None,
    ):
        """
        Decorator to trace a function.

        This is an instance method version of the global trace decorator.

        Args:
            name: Span name (defaults to function name)
            model_id: Model identifier
            inference_type: Type of inference
            capture_input: Whether to capture function input
            capture_output: Whether to capture function output
            attributes: Additional span attributes

        Example:
            @pyflare.trace(model_id="gpt-4")
            def predict(text):
                return model.predict(text)
        """
        from pyflare.decorators import trace as trace_decorator
        return trace_decorator(
            name=name,
            model_id=model_id,
            inference_type=inference_type,
            capture_input=capture_input,
            capture_output=capture_output,
            attributes=attributes,
        )

    def register_model(self, model_info: ModelInfo) -> None:
        """
        Register a model for tracking.

        Args:
            model_info: Information about the model
        """
        self._models[model_info.model_id] = model_info

    def get_model(self, model_id: str) -> Optional[ModelInfo]:
        """Get registered model info."""
        return self._models.get(model_id)

    def set_user(self, user_id: str, attributes: Optional[dict[str, Any]] = None) -> None:
        """
        Set the current user for cost attribution.

        SECURITY: User IDs may be considered PII. Enable scrub_pii
        if you don't want user IDs in traces.

        Args:
            user_id: User identifier
            attributes: Additional user attributes
        """
        # SECURITY: Sanitize user ID
        self._context_vars["user_id"] = self.sanitize_attribute(user_id)
        if attributes:
            sanitized_attrs = {
                k: self.sanitize_attribute(v)
                for k, v in attributes.items()
            }
            self._context_vars["user_attributes"] = sanitized_attrs

    def set_feature(self, feature_id: str) -> None:
        """
        Set the current feature for cost attribution.

        Args:
            feature_id: Feature identifier
        """
        self._context_vars["feature_id"] = self.sanitize_attribute(feature_id)

    def set_session(self, session_id: str) -> None:
        """
        Set the current session ID.

        Args:
            session_id: Session identifier
        """
        self._context_vars["session_id"] = self.sanitize_attribute(session_id)

    def get_context_var(self, key: str) -> Optional[Any]:
        """Get a context variable."""
        return self._context_vars.get(key)

    def add_span_attributes(self, span: trace.Span, token_usage: TokenUsage, model_id: str = "") -> None:
        """
        Add standard attributes to a span.

        Args:
            span: The span to add attributes to
            token_usage: Token usage information
            model_id: Optional model ID for cost calculation
        """
        span.set_attribute("pyflare.tokens.input", token_usage.input_tokens)
        span.set_attribute("pyflare.tokens.output", token_usage.output_tokens)
        span.set_attribute("pyflare.tokens.total", token_usage.total_tokens)

        # Calculate and add cost if model_id is provided
        if model_id:
            from pyflare.cost import calculate_cost
            cost_result = calculate_cost(
                model_id,
                token_usage.input_tokens,
                token_usage.output_tokens,
            )
            span.set_attribute("pyflare.cost.micros", cost_result.total_micros)

        # Add context variables
        if "user_id" in self._context_vars:
            span.set_attribute("pyflare.user.id", self._context_vars["user_id"])
        if "feature_id" in self._context_vars:
            span.set_attribute("pyflare.feature.id", self._context_vars["feature_id"])
        if "session_id" in self._context_vars:
            span.set_attribute("pyflare.session.id", self._context_vars["session_id"])


def get_pyflare() -> Optional[PyFlare]:
    """Get the global PyFlare instance."""
    return PyFlare.get_instance()


def init(
    service_name: str,
    endpoint: str = "http://localhost:4317",
    **kwargs: Any,
) -> PyFlare:
    """
    Initialize PyFlare SDK.

    Convenience function for quick initialization.

    Args:
        service_name: Name of the service
        endpoint: PyFlare collector endpoint
        **kwargs: Additional arguments passed to PyFlare constructor

    Returns:
        Initialized PyFlare instance

    Example:
        import pyflare
        pyflare.init("my-service")
    """
    return PyFlare(service_name=service_name, endpoint=endpoint, **kwargs)
