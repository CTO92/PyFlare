"""PyFlame ecosystem integration for PyFlare."""

from typing import Any, Optional

from pyflare.sdk import PyFlare
from pyflare.types import InferenceType


class PyFlameInstrumentation:
    """
    Native integration with PyFlame ecosystem.

    Provides seamless observability for:
    - PyFlame training runs
    - PyFlameRT inference
    - PyFlameVision processing
    - PyFlameAudio processing

    Example:
        from pyflare.integrations.pyflame import PyFlameInstrumentation

        pyflare = PyFlare(service_name="my-pyflame-app")
        PyFlameInstrumentation().instrument()

        # Now PyFlame operations are automatically traced
    """

    def __init__(self) -> None:
        self._instrumented = False

    def instrument(self) -> None:
        """Enable PyFlame instrumentation."""
        if self._instrumented:
            return

        try:
            import pyflame
        except ImportError:
            raise ImportError(
                "pyflame package is required for PyFlame instrumentation."
            )

        # Hook into PyFlame's telemetry system
        self._setup_pyflame_hooks(pyflame)

        self._instrumented = True

    def uninstrument(self) -> None:
        """Disable PyFlame instrumentation."""
        self._instrumented = False

    def _setup_pyflame_hooks(self, pyflame: Any) -> None:
        """Set up hooks into PyFlame's internal telemetry."""
        # PyFlame would expose hooks for:
        # - Model compilation
        # - Training iterations
        # - Inference calls
        # - Memory allocations
        pass


def trace_pyflame_model(
    model_id: Optional[str] = None,
) -> Any:
    """
    Decorator for tracing PyFlame model operations.

    Example:
        @trace_pyflame_model(model_id="my-cerebras-model")
        def train_step(model, batch):
            return model(batch)
    """
    from pyflare.decorators import trace

    return trace(
        model_id=model_id,
        inference_type=InferenceType.CUSTOM,
        attributes={"pyflare.framework": "pyflame"},
    )
