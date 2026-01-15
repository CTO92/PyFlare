"""Cost calculation utilities for PyFlare."""

from dataclasses import dataclass
from datetime import datetime
from enum import Enum
from typing import Optional

from pyflare.types import TokenUsage


class ModelProvider(str, Enum):
    """Supported model providers."""
    OPENAI = "openai"
    ANTHROPIC = "anthropic"
    COHERE = "cohere"
    GOOGLE = "google"
    AZURE = "azure"
    AWS_BEDROCK = "aws_bedrock"
    CUSTOM = "custom"


@dataclass
class ModelPricing:
    """Pricing information for a model.

    All prices are in micro-dollars (1/1,000,000 of a dollar) per token.
    """
    model_id: str
    provider: ModelProvider
    input_price_per_million: float  # Price per 1M input tokens in dollars
    output_price_per_million: float  # Price per 1M output tokens in dollars

    @property
    def input_price_micros(self) -> float:
        """Price per token in micro-dollars."""
        return self.input_price_per_million / 1_000_000 * 1_000_000

    @property
    def output_price_micros(self) -> float:
        """Price per token in micro-dollars."""
        return self.output_price_per_million / 1_000_000 * 1_000_000


# Pricing data (as of early 2025 - should be updated periodically)
MODEL_PRICING: dict[str, ModelPricing] = {
    # OpenAI GPT-4 models
    "gpt-4o": ModelPricing("gpt-4o", ModelProvider.OPENAI, 2.50, 10.00),
    "gpt-4o-mini": ModelPricing("gpt-4o-mini", ModelProvider.OPENAI, 0.15, 0.60),
    "gpt-4-turbo": ModelPricing("gpt-4-turbo", ModelProvider.OPENAI, 10.00, 30.00),
    "gpt-4": ModelPricing("gpt-4", ModelProvider.OPENAI, 30.00, 60.00),
    "gpt-3.5-turbo": ModelPricing("gpt-3.5-turbo", ModelProvider.OPENAI, 0.50, 1.50),

    # OpenAI embedding models
    "text-embedding-3-small": ModelPricing("text-embedding-3-small", ModelProvider.OPENAI, 0.02, 0.0),
    "text-embedding-3-large": ModelPricing("text-embedding-3-large", ModelProvider.OPENAI, 0.13, 0.0),
    "text-embedding-ada-002": ModelPricing("text-embedding-ada-002", ModelProvider.OPENAI, 0.10, 0.0),

    # Anthropic Claude models
    "claude-3-5-sonnet-20241022": ModelPricing("claude-3-5-sonnet-20241022", ModelProvider.ANTHROPIC, 3.00, 15.00),
    "claude-3-opus-20240229": ModelPricing("claude-3-opus-20240229", ModelProvider.ANTHROPIC, 15.00, 75.00),
    "claude-3-sonnet-20240229": ModelPricing("claude-3-sonnet-20240229", ModelProvider.ANTHROPIC, 3.00, 15.00),
    "claude-3-haiku-20240307": ModelPricing("claude-3-haiku-20240307", ModelProvider.ANTHROPIC, 0.25, 1.25),
    "claude-opus-4-5-20251101": ModelPricing("claude-opus-4-5-20251101", ModelProvider.ANTHROPIC, 15.00, 75.00),

    # Google models
    "gemini-1.5-pro": ModelPricing("gemini-1.5-pro", ModelProvider.GOOGLE, 1.25, 5.00),
    "gemini-1.5-flash": ModelPricing("gemini-1.5-flash", ModelProvider.GOOGLE, 0.075, 0.30),

    # Cohere models
    "command-r-plus": ModelPricing("command-r-plus", ModelProvider.COHERE, 2.50, 10.00),
    "command-r": ModelPricing("command-r", ModelProvider.COHERE, 0.15, 0.60),
}


class CostCalculator:
    """Calculate costs for model inference.

    Example:
        calculator = CostCalculator()

        # Calculate cost for a specific call
        cost = calculator.calculate(
            model_id="gpt-4o",
            token_usage=TokenUsage(input_tokens=1000, output_tokens=500)
        )
        print(f"Cost: ${cost.total_dollars:.4f}")

        # Or in micro-dollars for attribute storage
        print(f"Cost (micros): {cost.total_micros}")
    """

    def __init__(self, custom_pricing: Optional[dict[str, ModelPricing]] = None):
        """
        Initialize cost calculator.

        Args:
            custom_pricing: Additional or override pricing data
        """
        self._pricing = dict(MODEL_PRICING)
        if custom_pricing:
            self._pricing.update(custom_pricing)

    def add_pricing(self, pricing: ModelPricing) -> None:
        """Add or update pricing for a model."""
        self._pricing[pricing.model_id] = pricing

    def get_pricing(self, model_id: str) -> Optional[ModelPricing]:
        """Get pricing for a model."""
        # Try exact match first
        if model_id in self._pricing:
            return self._pricing[model_id]

        # Try prefix matching (e.g., "gpt-4o-2024-08-06" -> "gpt-4o")
        for key in self._pricing:
            if model_id.startswith(key):
                return self._pricing[key]

        return None

    def calculate(
        self,
        model_id: str,
        token_usage: TokenUsage,
        provider: Optional[ModelProvider] = None,
    ) -> "CostResult":
        """
        Calculate cost for a model call.

        Args:
            model_id: Model identifier
            token_usage: Token usage information
            provider: Optional provider override

        Returns:
            CostResult with cost breakdown
        """
        pricing = self.get_pricing(model_id)

        if pricing is None:
            # Unknown model - return zero cost
            return CostResult(
                model_id=model_id,
                token_usage=token_usage,
                input_cost_micros=0,
                output_cost_micros=0,
                estimated=True,
            )

        input_cost = (token_usage.input_tokens * pricing.input_price_per_million) / 1_000_000
        output_cost = (token_usage.output_tokens * pricing.output_price_per_million) / 1_000_000

        return CostResult(
            model_id=model_id,
            token_usage=token_usage,
            input_cost_micros=int(input_cost * 1_000_000),
            output_cost_micros=int(output_cost * 1_000_000),
            estimated=False,
        )

    def estimate_cost(
        self,
        model_id: str,
        input_text: str,
        estimated_output_tokens: int = 500,
    ) -> "CostResult":
        """
        Estimate cost before making a call.

        Args:
            model_id: Model identifier
            input_text: Input text to send
            estimated_output_tokens: Expected output tokens

        Returns:
            CostResult with estimated cost
        """
        # Rough token estimation (4 chars per token average)
        estimated_input_tokens = len(input_text) // 4

        return self.calculate(
            model_id=model_id,
            token_usage=TokenUsage(
                input_tokens=estimated_input_tokens,
                output_tokens=estimated_output_tokens,
                total_tokens=estimated_input_tokens + estimated_output_tokens,
            ),
        )


@dataclass
class CostResult:
    """Result of cost calculation."""
    model_id: str
    token_usage: TokenUsage
    input_cost_micros: int
    output_cost_micros: int
    estimated: bool = False

    @property
    def total_micros(self) -> int:
        """Total cost in micro-dollars."""
        return self.input_cost_micros + self.output_cost_micros

    @property
    def total_dollars(self) -> float:
        """Total cost in dollars."""
        return self.total_micros / 1_000_000

    @property
    def input_dollars(self) -> float:
        """Input cost in dollars."""
        return self.input_cost_micros / 1_000_000

    @property
    def output_dollars(self) -> float:
        """Output cost in dollars."""
        return self.output_cost_micros / 1_000_000

    def __str__(self) -> str:
        """Human-readable cost string."""
        prefix = "~" if self.estimated else ""
        return f"{prefix}${self.total_dollars:.6f}"


# Global cost calculator instance
_default_calculator: Optional[CostCalculator] = None


def get_cost_calculator() -> CostCalculator:
    """Get the default cost calculator."""
    global _default_calculator
    if _default_calculator is None:
        _default_calculator = CostCalculator()
    return _default_calculator


def calculate_cost(
    model_id: str,
    input_tokens: int,
    output_tokens: int,
) -> CostResult:
    """
    Convenience function to calculate cost.

    Args:
        model_id: Model identifier
        input_tokens: Number of input tokens
        output_tokens: Number of output tokens

    Returns:
        CostResult with cost breakdown
    """
    calculator = get_cost_calculator()
    return calculator.calculate(
        model_id=model_id,
        token_usage=TokenUsage(
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            total_tokens=input_tokens + output_tokens,
        ),
    )
