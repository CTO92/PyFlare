"""
PyFlare integrations for popular ML frameworks.

Available integrations:
- OpenAI: Automatic instrumentation for OpenAI API calls
- Anthropic: Automatic instrumentation for Anthropic (Claude) API calls
- LangChain: Tracing for LangChain chains and agents
- PyTorch: Model inference tracing
- PyFlame: Native integration with PyFlame ecosystem

Usage:
    from pyflare.integrations import OpenAIInstrumentation, AnthropicInstrumentation

    # Enable OpenAI instrumentation
    OpenAIInstrumentation().instrument()

    # Enable Anthropic instrumentation
    AnthropicInstrumentation().instrument()

    # Now all API calls are automatically traced
"""

from pyflare.integrations.openai import OpenAIInstrumentation
from pyflare.integrations.anthropic import AnthropicInstrumentation, trace_anthropic_call
from pyflare.integrations.langchain import LangChainInstrumentation

__all__ = [
    "OpenAIInstrumentation",
    "AnthropicInstrumentation",
    "trace_anthropic_call",
    "LangChainInstrumentation",
]
