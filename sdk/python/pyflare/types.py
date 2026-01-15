"""Type definitions for PyFlare SDK."""

from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any, Optional

from pydantic import BaseModel, Field


class InferenceType(str, Enum):
    """Type of ML inference."""

    LLM = "llm"
    EMBEDDING = "embedding"
    CLASSIFICATION = "classification"
    REGRESSION = "regression"
    OBJECT_DETECTION = "object_detection"
    CUSTOM = "custom"


class SpanKind(str, Enum):
    """OpenTelemetry span kinds."""

    INTERNAL = "internal"
    SERVER = "server"
    CLIENT = "client"
    PRODUCER = "producer"
    CONSUMER = "consumer"


class ModelInfo(BaseModel):
    """Information about the model being traced."""

    model_id: str = Field(..., description="Unique identifier for the model")
    model_version: str = Field(default="", description="Model version string")
    provider: str = Field(default="", description="Model provider (openai, anthropic, etc.)")
    inference_type: InferenceType = Field(
        default=InferenceType.CUSTOM, description="Type of inference"
    )


class TokenUsage(BaseModel):
    """Token usage for LLM calls."""

    input_tokens: int = Field(default=0, ge=0)
    output_tokens: int = Field(default=0, ge=0)
    total_tokens: int = Field(default=0, ge=0)

    def __post_init__(self) -> None:
        if self.total_tokens == 0:
            self.total_tokens = self.input_tokens + self.output_tokens


class SpanAttributes(BaseModel):
    """Attributes to attach to a span."""

    # Model information
    model_id: Optional[str] = None
    model_version: Optional[str] = None
    model_provider: Optional[str] = None
    inference_type: Optional[InferenceType] = None

    # Input/Output
    input_preview: Optional[str] = None
    output_preview: Optional[str] = None

    # Token usage
    input_tokens: Optional[int] = None
    output_tokens: Optional[int] = None
    total_tokens: Optional[int] = None

    # Cost (in micro-dollars)
    cost_micros: Optional[int] = None

    # User/feature attribution
    user_id: Optional[str] = None
    feature_id: Optional[str] = None

    # Custom attributes
    custom: dict[str, Any] = Field(default_factory=dict)

    def to_otel_attributes(self) -> dict[str, Any]:
        """Convert to OpenTelemetry attribute format."""
        attrs: dict[str, Any] = {}

        if self.model_id:
            attrs["pyflare.model.id"] = self.model_id
        if self.model_version:
            attrs["pyflare.model.version"] = self.model_version
        if self.model_provider:
            attrs["pyflare.model.provider"] = self.model_provider
        if self.inference_type:
            attrs["pyflare.inference.type"] = self.inference_type.value

        if self.input_preview:
            attrs["pyflare.input.preview"] = self.input_preview[:1000]
        if self.output_preview:
            attrs["pyflare.output.preview"] = self.output_preview[:1000]

        if self.input_tokens is not None:
            attrs["pyflare.tokens.input"] = self.input_tokens
        if self.output_tokens is not None:
            attrs["pyflare.tokens.output"] = self.output_tokens
        if self.total_tokens is not None:
            attrs["pyflare.tokens.total"] = self.total_tokens

        if self.cost_micros is not None:
            attrs["pyflare.cost.micros"] = self.cost_micros

        if self.user_id:
            attrs["pyflare.user.id"] = self.user_id
        if self.feature_id:
            attrs["pyflare.feature.id"] = self.feature_id

        # Add custom attributes with prefix
        for key, value in self.custom.items():
            attrs[f"pyflare.custom.{key}"] = value

        return attrs


@dataclass
class TraceContext:
    """Context for the current trace."""

    trace_id: str
    span_id: str
    parent_span_id: Optional[str] = None
    start_time: datetime = field(default_factory=datetime.utcnow)
    attributes: dict[str, Any] = field(default_factory=dict)

    def add_attribute(self, key: str, value: Any) -> None:
        """Add an attribute to the context."""
        self.attributes[key] = value

    def add_event(self, name: str, attributes: Optional[dict[str, Any]] = None) -> None:
        """Add an event to the current span."""
        # This is a placeholder - actual implementation would use OTel API
        pass
